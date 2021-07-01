/*
 * Copyright 2020 OmniSci, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * File:   DBHandler.cpp
 * Author: michael
 *
 * Created on Jan 1, 2017, 12:40 PM
 */

#include "DBHandler.h"
#include "DistributedLoader.h"
#include "QueryEngine/UDFCompiler.h"
#include "TokenCompletionHints.h"

#ifdef HAVE_PROFILER
#include <gperftools/heap-profiler.h>
#endif  // HAVE_PROFILER

#include "MapDRelease.h"

#include "Calcite/Calcite.h"
#include "gen-cpp/CalciteServer.h"

#include "QueryEngine/RelAlgExecutor.h"

#include "Catalog/Catalog.h"
#include "Catalog/DdlCommandExecutor.h"
#include "DataMgr/ForeignStorage/ArrowForeignStorage.h"
#include "DataMgr/ForeignStorage/DummyForeignStorage.h"
#include "DistributedHandler.h"
#include "Fragmenter/InsertOrderFragmenter.h"
#include "Geospatial/GDAL.h"
#include "Geospatial/Transforms.h"
#include "Geospatial/Types.h"
#include "ImportExport/Importer.h"
#include "L0Mgr/L0Mgr.h"
#include "LockMgr/LockMgr.h"
#include "OSDependent/omnisci_hostname.h"
#include "Parser/ParserWrapper.h"
#include "Parser/ReservedKeywords.h"
#include "Parser/parser.h"
#include "QueryEngine/ArrowResultSet.h"
#include "QueryEngine/CalciteAdapter.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ExtensionFunctionsWhitelist.h"
#include "QueryEngine/GpuMemUtils.h"
#include "QueryEngine/JoinFilterPushDown.h"
#include "QueryEngine/JsonAccessors.h"
#include "QueryEngine/QueryDispatchQueue.h"
#include "QueryEngine/ResultSetBuilder.h"
#include "QueryEngine/TableFunctions/TableFunctionsFactory.h"
#include "QueryEngine/TableOptimizer.h"
#include "QueryEngine/ThriftSerializers.h"
#include "Shared/ArrowUtil.h"
#include "Shared/StringTransform.h"
#include "Shared/import_helpers.h"
#include "Shared/mapd_shared_mutex.h"
#include "Shared/measure.h"
#include "Shared/scope.h"

#ifdef HAVE_AWS_S3
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#endif
#include <fcntl.h>
#include <picosha2.h>
#include <sys/types.h>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>
#include <boost/process/search_path.hpp>
#include <boost/program_options.hpp>
#include <boost/regex.hpp>
#include <boost/tokenizer.hpp>
#include <cmath>
#include <csignal>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <random>
#include <regex>
#include <string>
#include <thread>
#include <typeinfo>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>

#include "Shared/ArrowUtil.h"

#define ENABLE_GEO_IMPORT_COLUMN_MATCHING 0

#ifdef HAVE_AWS_S3
extern bool g_allow_s3_server_privileges;
#endif

using Catalog_Namespace::Catalog;
using Catalog_Namespace::SysCatalog;

#define INVALID_SESSION_ID ""

#define THROW_MAPD_EXCEPTION(errstr) \
  {                                  \
    TOmniSciException ex;            \
    ex.error_msg = errstr;           \
    LOG(ERROR) << ex.error_msg;      \
    throw ex;                        \
  }

thread_local std::string TrackingProcessor::client_address;
thread_local ClientProtocol TrackingProcessor::client_protocol;

namespace {

SessionMap::iterator get_session_from_map(const TSessionId& session,
                                          SessionMap& session_map) {
  auto session_it = session_map.find(session);
  if (session_it == session_map.end()) {
    THROW_MAPD_EXCEPTION("Session not valid.");
  }
  return session_it;
}

struct ForceDisconnect : public std::runtime_error {
  ForceDisconnect(const std::string& cause) : std::runtime_error(cause) {}
};

}  // namespace

template <>
SessionMap::iterator DBHandler::get_session_it_unsafe(
    const TSessionId& session,
    mapd_shared_lock<mapd_shared_mutex>& read_lock) {
  auto session_it = get_session_from_map(session, sessions_);
  try {
    check_session_exp_unsafe(session_it);
  } catch (const ForceDisconnect& e) {
    read_lock.unlock();
    mapd_unique_lock<mapd_shared_mutex> write_lock(sessions_mutex_);
    auto session_it2 = get_session_from_map(session, sessions_);
    disconnect_impl(session_it2, write_lock);
    THROW_MAPD_EXCEPTION(e.what());
  }
  return session_it;
}

template <>
SessionMap::iterator DBHandler::get_session_it_unsafe(
    const TSessionId& session,
    mapd_unique_lock<mapd_shared_mutex>& write_lock) {
  auto session_it = get_session_from_map(session, sessions_);
  try {
    check_session_exp_unsafe(session_it);
  } catch (const ForceDisconnect& e) {
    disconnect_impl(session_it, write_lock);
    THROW_MAPD_EXCEPTION(e.what());
  }
  return session_it;
}

template <>
void DBHandler::expire_idle_sessions_unsafe(
    mapd_unique_lock<mapd_shared_mutex>& write_lock) {
  std::vector<std::string> expired_sessions;
  for (auto session_pair : sessions_) {
    auto session_it = get_session_from_map(session_pair.first, sessions_);
    try {
      check_session_exp_unsafe(session_it);
    } catch (const ForceDisconnect& e) {
      expired_sessions.emplace_back(session_it->second->get_session_id());
    }
  }

  for (auto session_id : expired_sessions) {
    if (leaf_aggregator_.leafCount() > 0) {
      try {
        leaf_aggregator_.disconnect(session_id);
      } catch (TOmniSciException& toe) {
        LOG(INFO) << " Problem disconnecting from leaves : " << toe.what();
      } catch (std::exception& e) {
        LOG(INFO)
            << " Problem disconnecting from leaves, check leaf logs for additonal info";
      }
    }
    sessions_.erase(session_id);
  }
  if (render_handler_) {
    write_lock.unlock();
    for (auto session_id : expired_sessions) {
      // NOTE: the render disconnect is done after the session lock is released to
      // avoid a deadlock. See: https://omnisci.atlassian.net/browse/BE-3324
      // This out-of-scope solution is a compromise for now until a better session
      // handling/locking mechanism is developed for the renderer. Note as well that the
      // session_id cannot be immediately reused. If a render request were to slip in
      // after the lock is released and before the render disconnect could cause a
      // problem.
      render_handler_->disconnect(session_id);
    }
  }
}

#ifdef ENABLE_GEOS
extern std::unique_ptr<std::string> g_libgeos_so_filename;
#endif

DBHandler::DBHandler(const std::vector<LeafHostInfo>& db_leaves,
                     const std::vector<LeafHostInfo>& string_leaves,
                     const std::string& base_data_path,
                     const bool allow_multifrag,
                     const bool jit_debug,
                     const bool intel_jit_profile,
                     const bool read_only,
                     const bool allow_loop_joins,
                     const bool enable_rendering,
                     const bool renderer_use_vulkan_driver,
                     const bool enable_auto_clear_render_mem,
                     const int render_oom_retry_threshold,
                     const size_t render_mem_bytes,
                     const size_t max_concurrent_render_sessions,
                     const size_t reserved_gpu_mem,
                     const bool render_compositor_use_last_gpu,
                     const size_t num_reader_threads,
                     const AuthMetadata& authMetadata,
                     SystemParameters& system_parameters,
                     const bool legacy_syntax,
                     const int idle_session_duration,
                     const int max_session_duration,
                     const bool enable_runtime_udf_registration,
                     const std::string& udf_filename,
                     const std::string& clang_path,
                     const std::vector<std::string>& clang_options,
#ifdef ENABLE_GEOS
                     const std::string& libgeos_so_filename,
#endif
                     const File_Namespace::DiskCacheConfig& disk_cache_config,
                     const bool is_new_db)
    : leaf_aggregator_(db_leaves)
    , db_leaves_(db_leaves)
    , string_leaves_(string_leaves)
    , base_data_path_(base_data_path)
    , random_gen_(std::random_device{}())
    , session_id_dist_(0, INT32_MAX)
    , jit_debug_(jit_debug)
    , intel_jit_profile_(intel_jit_profile)
    , allow_multifrag_(allow_multifrag)
    , read_only_(read_only)
    , allow_loop_joins_(allow_loop_joins)
    , authMetadata_(authMetadata)
    , system_parameters_(system_parameters)
    , legacy_syntax_(legacy_syntax)
    , dispatch_queue_(
          std::make_unique<QueryDispatchQueue>(system_parameters.num_executors))
    , super_user_rights_(false)
    , idle_session_duration_(idle_session_duration * 60)
    , max_session_duration_(max_session_duration * 60)
    , runtime_udf_registration_enabled_(enable_runtime_udf_registration)

    , enable_rendering_(enable_rendering)
    , renderer_use_vulkan_driver_(renderer_use_vulkan_driver)
    , enable_auto_clear_render_mem_(enable_auto_clear_render_mem)
    , render_oom_retry_threshold_(render_oom_retry_threshold)
    , max_concurrent_render_sessions_(max_concurrent_render_sessions)
    , reserved_gpu_mem_(reserved_gpu_mem)
    , render_compositor_use_last_gpu_(render_compositor_use_last_gpu)
    , render_mem_bytes_(render_mem_bytes)
    , num_reader_threads_(num_reader_threads)
#ifdef ENABLE_GEOS
    , libgeos_so_filename_(libgeos_so_filename)
#endif
    , disk_cache_config_(disk_cache_config)
    , udf_filename_(udf_filename)
    , clang_path_(clang_path)
    , clang_options_(clang_options)

{
  LOG(INFO) << "OmniSci Server " << MAPD_RELEASE;
  initialize(is_new_db);
}

void DBHandler::initialize(const bool is_new_db) {
  if (!initialized_) {
    initialized_ = true;
  } else {
    THROW_MAPD_EXCEPTION(
        "Server already initialized; service restart required to activate any new "
        "entitlements.");
    return;
  }

  if (system_parameters_.cpu_only || system_parameters_.num_gpus == 0) {
    executor_device_type_ = ExecutorDeviceType::CPU;
    cpu_mode_only_ = true;
  } else {
#ifdef HAVE_CUDA
    executor_device_type_ = ExecutorDeviceType::GPU;
    cpu_mode_only_ = false;
#else
    executor_device_type_ = ExecutorDeviceType::CPU;
    LOG(ERROR) << "This build isn't CUDA enabled, will run on CPU";
    cpu_mode_only_ = true;
#endif
  }

  bool is_rendering_enabled = enable_rendering_;
  if (system_parameters_.num_gpus == 0) {
    is_rendering_enabled = false;
  }

  const auto data_path = boost::filesystem::path(base_data_path_) / "mapd_data";
  // calculate the total amount of memory we need to reserve from each gpu that the Buffer
  // manage cannot ask for
  size_t total_reserved = reserved_gpu_mem_;
  if (is_rendering_enabled) {
    total_reserved += render_mem_bytes_;
  }

  std::unique_ptr<CudaMgr_Namespace::CudaMgr> cuda_mgr;
  std::unique_ptr<l0::L0Manager> l0_mgr;

#ifdef HAVE_CUDA
  if (!cpu_mode_only_ || is_rendering_enabled) {
    try {
      cuda_mgr = std::make_unique<CudaMgr_Namespace::CudaMgr>(
          system_parameters_.num_gpus, system_parameters_.start_gpu);
    } catch (const std::exception& e) {
      LOG(ERROR) << "Unable to instantiate CudaMgr, falling back to CPU-only mode. "
                 << e.what();
      cpu_mode_only_ = true;
      is_rendering_enabled = false;
    }
  }
#elif HAVE_L0
  if (!cpu_mode_only_ || is_rendering_enabled) {
    try {
      l0_mgr = std::make_unique<l0::L0Manager>();
    } catch (const std::exception& e) {
      LOG(ERROR) << "Unable to instantiate L0 Manager, falling back to CPU-only mode. "
                 << e.what();
      cpu_mode_only_ = true;
      is_rendering_enabled = false;
    }
  }
#endif  // HAVE_CUDA

  try {
    data_mgr_.reset(new Data_Namespace::DataMgr(data_path.string(),
                                                system_parameters_,
                                                std::move(cuda_mgr),
                                                std::move(l0_mgr),
                                                !cpu_mode_only_,
                                                total_reserved,
                                                num_reader_threads_,
                                                disk_cache_config_));
  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to initialize data manager: " << e.what();
  }

  std::string udf_ast_filename("");

  try {
    if (!udf_filename_.empty()) {
      const auto cuda_mgr = data_mgr_->getCudaMgr();
      const CudaMgr_Namespace::NvidiaDeviceArch device_arch =
          cuda_mgr ? cuda_mgr->getDeviceArch()
                   : CudaMgr_Namespace::NvidiaDeviceArch::Kepler;
      UdfCompiler compiler(udf_filename_, device_arch, clang_path_, clang_options_);
      int compile_result = compiler.compileUdf();

      if (compile_result == 0) {
        udf_ast_filename = compiler.getAstFileName();
      }
    }
  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to initialize UDF compiler: " << e.what();
  }

  try {
    calcite_ =
        std::make_shared<Calcite>(system_parameters_, base_data_path_, udf_ast_filename);
  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to initialize Calcite server: " << e.what();
  }

  try {
    ExtensionFunctionsWhitelist::add(calcite_->getExtensionFunctionWhitelist());
    if (!udf_filename_.empty()) {
      ExtensionFunctionsWhitelist::addUdfs(calcite_->getUserDefinedFunctionWhitelist());
    }
  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to initialize extension functions: " << e.what();
  }

  try {
    table_functions::TableFunctionsFactory::init();
  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to initialize table functions factory: " << e.what();
  }

  try {
    auto udtfs = ThriftSerializers::to_thrift(
        table_functions::TableFunctionsFactory::get_table_funcs(/*is_runtime=*/false));
    std::vector<TUserDefinedFunction> udfs = {};
    calcite_->setRuntimeExtensionFunctions(udfs, udtfs, /*is_runtime=*/false);
  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to register compile-time table functions: " << e.what();
  }

  if (!data_mgr_->gpusPresent() && !cpu_mode_only_) {
    executor_device_type_ = ExecutorDeviceType::CPU;
    LOG(ERROR) << "No GPUs detected, falling back to CPU mode";
    cpu_mode_only_ = true;
  }

  switch (executor_device_type_) {
    case ExecutorDeviceType::GPU:
      LOG(INFO) << "Started in GPU mode" << std::endl;
      break;
    case ExecutorDeviceType::CPU:
      LOG(INFO) << "Started in CPU mode" << std::endl;
      break;
  }

  try {
    g_base_path = base_data_path_;
    SysCatalog::instance().init(base_data_path_,
                                data_mgr_,
                                authMetadata_,
                                calcite_,
                                is_new_db,
                                !db_leaves_.empty(),
                                string_leaves_);
  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to initialize system catalog: " << e.what();
  }

  import_path_ = boost::filesystem::path(base_data_path_) / "mapd_import";
  start_time_ = std::time(nullptr);

  if (is_rendering_enabled) {
    try {
      render_handler_.reset(new RenderHandler(this,
                                              render_mem_bytes_,
                                              max_concurrent_render_sessions_,
                                              render_compositor_use_last_gpu_,
                                              false,
                                              0,
                                              false,
                                              system_parameters_));
    } catch (const std::exception& e) {
      LOG(ERROR) << "Backend rendering disabled: " << e.what();
    }
  }

  if (leaf_aggregator_.leafCount() > 0) {
    try {
      agg_handler_.reset(new MapDAggHandler(this));
    } catch (const std::exception& e) {
      LOG(ERROR) << "Distributed aggregator support disabled: " << e.what();
    }
  } else if (g_cluster) {
    try {
      leaf_handler_.reset(new MapDLeafHandler(this));
    } catch (const std::exception& e) {
      LOG(ERROR) << "Distributed leaf support disabled: " << e.what();
    }
  }

#ifdef ENABLE_GEOS
  if (!libgeos_so_filename_.empty()) {
    g_libgeos_so_filename.reset(new std::string(libgeos_so_filename_));
    LOG(INFO) << "Overriding default geos library with '" + *g_libgeos_so_filename + "'";
  }
#endif
}

DBHandler::~DBHandler() {}

void DBHandler::parser_with_error_handler(
    const std::string& query_str,
    std::list<std::unique_ptr<Parser::Stmt>>& parse_trees) {
  int num_parse_errors = 0;
  std::string last_parsed;
  SQLParser parser;
  num_parse_errors = parser.parse(query_str, parse_trees, last_parsed);
  if (num_parse_errors > 0) {
    throw std::runtime_error("Syntax error at: " + last_parsed);
  }
  if (parse_trees.size() > 1) {
    throw std::runtime_error("multiple SQL statements not allowed");
  } else if (parse_trees.size() != 1) {
    throw std::runtime_error("empty SQL statment not allowed");
  }
}
void DBHandler::check_read_only(const std::string& str) {
  if (DBHandler::read_only_) {
    THROW_MAPD_EXCEPTION(str + " disabled: server running in read-only mode.");
  }
}

std::string const DBHandler::createInMemoryCalciteSession(
    const std::shared_ptr<Catalog_Namespace::Catalog>& catalog_ptr) {
  // We would create an in memory session for calcite with super user privileges which
  // would be used for getting all tables metadata when a user runs the query. The
  // session would be under the name of a proxy user/password which would only persist
  // till server's lifetime or execution of calcite query(in memory) whichever is the
  // earliest.
  mapd_lock_guard<mapd_shared_mutex> write_lock(sessions_mutex_);
  std::string session_id;
  do {
    session_id = generate_random_string(64);
  } while (sessions_.find(session_id) != sessions_.end());
  Catalog_Namespace::UserMetadata user_meta(-1,
                                            calcite_->getInternalSessionProxyUserName(),
                                            calcite_->getInternalSessionProxyPassword(),
                                            true,
                                            -1,
                                            true);
  const auto emplace_ret =
      sessions_.emplace(session_id,
                        std::make_shared<Catalog_Namespace::SessionInfo>(
                            catalog_ptr, user_meta, executor_device_type_, session_id));
  CHECK(emplace_ret.second);
  return session_id;
}

bool DBHandler::isInMemoryCalciteSession(
    const Catalog_Namespace::UserMetadata user_meta) {
  return user_meta.userName == calcite_->getInternalSessionProxyUserName() &&
         user_meta.userId == -1 && user_meta.defaultDbId == -1 &&
         user_meta.isSuper.load();
}

void DBHandler::removeInMemoryCalciteSession(const std::string& session_id) {
  // Remove InMemory calcite Session.
  mapd_lock_guard<mapd_shared_mutex> write_lock(sessions_mutex_);
  const auto it = sessions_.find(session_id);
  CHECK(it != sessions_.end());
  sessions_.erase(it);
}

// internal connection for connections with no password
void DBHandler::internal_connect(TSessionId& session,
                                 const std::string& username,
                                 const std::string& dbname) {
  auto stdlog = STDLOG();            // session_info set by connect_impl()
  std::string username2 = username;  // login() may reset username given as argument
  std::string dbname2 = dbname;      // login() may reset dbname given as argument
  Catalog_Namespace::UserMetadata user_meta;
  std::shared_ptr<Catalog> cat = nullptr;
  try {
    cat =
        SysCatalog::instance().login(dbname2, username2, std::string(), user_meta, false);
  } catch (std::exception& e) {
    THROW_MAPD_EXCEPTION(e.what());
  }

  DBObject dbObject(dbname2, DatabaseDBObjectType);
  dbObject.loadKey(*cat);
  dbObject.setPrivileges(AccessPrivileges::ACCESS);
  std::vector<DBObject> dbObjects;
  dbObjects.push_back(dbObject);
  if (!SysCatalog::instance().checkPrivileges(user_meta, dbObjects)) {
    THROW_MAPD_EXCEPTION("Unauthorized Access: user " + user_meta.userLoggable() +
                         " is not allowed to access database " + dbname2 + ".");
  }
  connect_impl(session, std::string(), dbname2, user_meta, cat, stdlog);
}

bool DBHandler::isAggregator() const {
  return leaf_aggregator_.leafCount() > 0;
}

void DBHandler::krb5_connect(TKrb5Session& session,
                             const std::string& inputToken,
                             const std::string& dbname) {
  THROW_MAPD_EXCEPTION("Unauthrorized Access. Kerberos login not supported");
}

void DBHandler::connect(TSessionId& session,
                        const std::string& username,
                        const std::string& passwd,
                        const std::string& dbname) {
  auto stdlog = STDLOG();  // session_info set by connect_impl()
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  std::string username2 = username;  // login() may reset username given as argument
  std::string dbname2 = dbname;      // login() may reset dbname given as argument
  Catalog_Namespace::UserMetadata user_meta;
  std::shared_ptr<Catalog> cat = nullptr;
  try {
    cat = SysCatalog::instance().login(
        dbname2, username2, passwd, user_meta, !super_user_rights_);
  } catch (std::exception& e) {
    stdlog.appendNameValuePairs("user", username, "db", dbname, "exception", e.what());
    THROW_MAPD_EXCEPTION(e.what());
  }

  DBObject dbObject(dbname2, DatabaseDBObjectType);
  dbObject.loadKey(*cat);
  dbObject.setPrivileges(AccessPrivileges::ACCESS);
  std::vector<DBObject> dbObjects;
  dbObjects.push_back(dbObject);
  if (!SysCatalog::instance().checkPrivileges(user_meta, dbObjects)) {
    stdlog.appendNameValuePairs(
        "user", username, "db", dbname, "exception", "Missing Privileges");
    THROW_MAPD_EXCEPTION("Unauthorized Access: user " + user_meta.userLoggable() +
                         " is not allowed to access database " + dbname2 + ".");
  }
  connect_impl(session, passwd, dbname2, user_meta, cat, stdlog);

  // Restriction is returned as part of the users metadata on login but
  // is per session so transfering it over here
  // Currently only SAML can even set a Restriction
  auto restriction = std::make_shared<Restriction>(user_meta.restriction);
  auto login_session = get_session_ptr(session);
  login_session->set_restriction(restriction);

  // if pki auth session will come back encrypted with user pubkey
  SysCatalog::instance().check_for_session_encryption(passwd, session);
}

std::shared_ptr<Catalog_Namespace::SessionInfo> DBHandler::create_new_session(
    TSessionId& session,
    const std::string& dbname,
    const Catalog_Namespace::UserMetadata& user_meta,
    std::shared_ptr<Catalog> cat) {
  do {
    session = generate_random_string(32);
  } while (sessions_.find(session) != sessions_.end());
  std::pair<SessionMap::iterator, bool> emplace_retval =
      sessions_.emplace(session,
                        std::make_shared<Catalog_Namespace::SessionInfo>(
                            cat, user_meta, executor_device_type_, session));
  CHECK(emplace_retval.second);
  auto& session_ptr = emplace_retval.first->second;
  LOG(INFO) << "User " << user_meta.userLoggable() << " connected to database " << dbname;
  return session_ptr;
}

void DBHandler::connect_impl(TSessionId& session,
                             const std::string& passwd,
                             const std::string& dbname,
                             const Catalog_Namespace::UserMetadata& user_meta,
                             std::shared_ptr<Catalog> cat,
                             query_state::StdLog& stdlog) {
  // TODO(sy): Is there any reason to have dbname as a parameter
  // here when the cat parameter already provides cat->name()?
  // Should dbname and cat->name() ever differ?
  {
    mapd_unique_lock<mapd_shared_mutex> write_lock(sessions_mutex_);
    expire_idle_sessions_unsafe(write_lock);
    if (system_parameters_.num_sessions > 0 &&
        sessions_.size() + 1 > static_cast<size_t>(system_parameters_.num_sessions)) {
      THROW_MAPD_EXCEPTION("Too many active sessions");
    }
  }
  {
    mapd_lock_guard<mapd_shared_mutex> write_lock(sessions_mutex_);
    auto session_ptr = create_new_session(session, dbname, user_meta, cat);
    stdlog.setSessionInfo(session_ptr);
    session_ptr->set_connection_info(getConnectionInfo().toString());
    if (!super_user_rights_) {  // no need to connect to leaf_aggregator_ at this time
      // while doing warmup
      if (leaf_aggregator_.leafCount() > 0) {
        leaf_aggregator_.connect(*session_ptr, user_meta.userName, passwd, dbname);
        return;
      }
    }
  }
  auto const roles =
      stdlog.getConstSessionInfo()->get_currentUser().isSuper
          ? std::vector<std::string>{{"super"}}
          : SysCatalog::instance().getRoles(
                false, false, stdlog.getConstSessionInfo()->get_currentUser().userName);
  stdlog.appendNameValuePairs("roles", boost::algorithm::join(roles, ","));
}

void DBHandler::disconnect(const TSessionId& session) {
  auto stdlog = STDLOG();
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());

  mapd_unique_lock<mapd_shared_mutex> write_lock(sessions_mutex_);
  auto session_it = get_session_it_unsafe(session, write_lock);
  stdlog.setSessionInfo(session_it->second);
  const auto dbname = session_it->second->getCatalog().getCurrentDB().dbName;

  LOG(INFO) << "User " << session_it->second->get_currentUser().userLoggable()
            << " disconnected from database " << dbname
            << " with public_session_id: " << session_it->second->get_public_session_id();

  disconnect_impl(session_it, write_lock);
}

void DBHandler::disconnect_impl(const SessionMap::iterator& session_it,
                                mapd_unique_lock<mapd_shared_mutex>& write_lock) {
  // session_it existence should already have been checked (i.e. called via
  // get_session_it_unsafe(...))

  const auto session_id = session_it->second->get_session_id();
  std::exception_ptr leaf_exception = nullptr;
  try {
    if (leaf_aggregator_.leafCount() > 0) {
      leaf_aggregator_.disconnect(session_id);
    }
  } catch (...) {
    leaf_exception = std::current_exception();
  }

  {
    std::lock_guard<std::mutex> lock(render_group_assignment_mutex_);
    render_group_assignment_map_.erase(session_id);
  }

  sessions_.erase(session_it);
  write_lock.unlock();

  if (render_handler_) {
    render_handler_->disconnect(session_id);
  }
}

void DBHandler::switch_database(const TSessionId& session, const std::string& dbname) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  mapd_unique_lock<mapd_shared_mutex> write_lock(sessions_mutex_);
  auto session_it = get_session_it_unsafe(session, write_lock);

  std::string dbname2 = dbname;  // switchDatabase() may reset dbname given as argument

  try {
    std::shared_ptr<Catalog> cat = SysCatalog::instance().switchDatabase(
        dbname2, session_it->second->get_currentUser().userName);
    session_it->second->set_catalog_ptr(cat);
    if (leaf_aggregator_.leafCount() > 0) {
      leaf_aggregator_.switch_database(session, dbname);
      return;
    }
  } catch (std::exception& e) {
    THROW_MAPD_EXCEPTION(e.what());
  }
}

void DBHandler::clone_session(TSessionId& session2, const TSessionId& session1) {
  auto stdlog = STDLOG(get_session_ptr(session1));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  mapd_unique_lock<mapd_shared_mutex> write_lock(sessions_mutex_);
  auto session_it = get_session_it_unsafe(session1, write_lock);

  try {
    const Catalog_Namespace::UserMetadata& user_meta =
        session_it->second->get_currentUser();
    std::shared_ptr<Catalog> cat = session_it->second->get_catalog_ptr();
    auto session2_ptr = create_new_session(session2, cat->name(), user_meta, cat);
    if (leaf_aggregator_.leafCount() > 0) {
      leaf_aggregator_.clone_session(session1, session2);
      return;
    }
  } catch (std::exception& e) {
    THROW_MAPD_EXCEPTION(e.what());
  }
}

void DBHandler::interrupt(const TSessionId& query_session,
                          const TSessionId& interrupt_session) {
  // if this is for distributed setting, query_session becomes a parent session (agg)
  // and the interrupt session is one of existing session in the leaf node (leaf)
  // so we can think there exists a logical mapping
  // between query_session (agg) and interrupt_session (leaf)
  auto stdlog = STDLOG(get_session_ptr(interrupt_session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  const auto allow_query_interrupt =
      g_enable_runtime_query_interrupt || g_enable_non_kernel_time_query_interrupt;
  if (g_enable_dynamic_watchdog || allow_query_interrupt) {
    // Shared lock to allow simultaneous interrupts of multiple sessions
    mapd_shared_lock<mapd_shared_mutex> read_lock(sessions_mutex_);

    auto session_it = get_session_it_unsafe(interrupt_session, read_lock);
    auto& cat = session_it->second.get()->getCatalog();
    const auto dbname = cat.getCurrentDB().dbName;
    auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID,
                                          jit_debug_ ? "/tmp" : "",
                                          jit_debug_ ? "mapdquery" : "",
                                          system_parameters_);
    CHECK(executor);

    VLOG(1) << "Received interrupt: "
            << "Session " << *session_it->second << ", Executor " << executor
            << ", leafCount " << leaf_aggregator_.leafCount() << ", User "
            << session_it->second->get_currentUser().userLoggable() << ", Database "
            << dbname << std::endl;

    if (leaf_aggregator_.leafCount() > 0) {
      leaf_aggregator_.interrupt(query_session, interrupt_session);
    }
    executor->interrupt(query_session, interrupt_session);

    LOG(INFO) << "User " << session_it->second->get_currentUser().userName
              << " interrupted session with database " << dbname << std::endl;
  }
}

TRole::type DBHandler::getServerRole() const {
  if (g_cluster) {
    if (leaf_aggregator_.leafCount() > 0) {
      return TRole::type::AGGREGATOR;
    }
    return TRole::type::LEAF;
  }
  return TRole::type::SERVER;
}

void DBHandler::get_server_status(TServerStatus& _return, const TSessionId& session) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  const auto rendering_enabled = bool(render_handler_);
  _return.read_only = read_only_;
  _return.version = MAPD_RELEASE;
  _return.rendering_enabled = rendering_enabled;
  _return.start_time = start_time_;
  _return.edition = MAPD_EDITION;
  _return.host_name = omnisci::get_hostname();
  _return.poly_rendering_enabled = rendering_enabled;
  _return.role = getServerRole();
  _return.renderer_status_json =
      render_handler_ ? render_handler_->get_renderer_status_json() : "";
}

void DBHandler::get_status(std::vector<TServerStatus>& _return,
                           const TSessionId& session) {
  //
  // get_status() is now called locally at startup on the aggregator
  // in order to validate that all nodes of a cluster are running the
  // same software version and the same renderer status
  //
  // In that context, it is called with the InvalidSessionID, and
  // with the local super-user flag set.
  //
  // Hence, we allow this session-less mode only in distributed mode, and
  // then on a leaf (always), or on the aggregator (only in super-user mode)
  //
  auto const allow_invalid_session = g_cluster && (!isAggregator() || super_user_rights_);
  if (!allow_invalid_session || session != getInvalidSessionId()) {
    auto stdlog = STDLOG(get_session_ptr(session));
    stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  } else {
    LOG(INFO) << "get_status() called in session-less mode";
  }
  const auto rendering_enabled = bool(render_handler_);
  TServerStatus ret;
  ret.read_only = read_only_;
  ret.version = MAPD_RELEASE;
  ret.rendering_enabled = rendering_enabled;
  ret.start_time = start_time_;
  ret.edition = MAPD_EDITION;
  ret.host_name = omnisci::get_hostname();
  ret.poly_rendering_enabled = rendering_enabled;
  ret.role = getServerRole();
  ret.renderer_status_json =
      render_handler_ ? render_handler_->get_renderer_status_json() : "";

  _return.push_back(ret);
  if (leaf_aggregator_.leafCount() > 0) {
    std::vector<TServerStatus> leaf_status = leaf_aggregator_.getLeafStatus(session);
    _return.insert(_return.end(), leaf_status.begin(), leaf_status.end());
  }
}

void DBHandler::get_hardware_info(TClusterHardwareInfo& _return,
                                  const TSessionId& session) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  THardwareInfo ret;
  const auto cuda_mgr = data_mgr_->getCudaMgr();
  if (cuda_mgr) {
    ret.num_gpu_hw = cuda_mgr->getDeviceCount();
    ret.start_gpu = cuda_mgr->getStartGpu();
    if (ret.start_gpu >= 0) {
      ret.num_gpu_allocated = cuda_mgr->getDeviceCount() - cuda_mgr->getStartGpu();
      // ^ This will break as soon as we allow non contiguous GPU allocations to MapD
    }
    for (int16_t device_id = 0; device_id < ret.num_gpu_hw; device_id++) {
      TGpuSpecification gpu_spec;
      auto deviceProperties = cuda_mgr->getDeviceProperties(device_id);
      gpu_spec.num_sm = deviceProperties->numMPs;
      gpu_spec.clock_frequency_kHz = deviceProperties->clockKhz;
      gpu_spec.memory = deviceProperties->globalMem;
      gpu_spec.compute_capability_major = deviceProperties->computeMajor;
      gpu_spec.compute_capability_minor = deviceProperties->computeMinor;
      ret.gpu_info.push_back(gpu_spec);
    }
  }

  // start  hardware/OS dependent code
  ret.num_cpu_hw = std::thread::hardware_concurrency();
  // ^ This might return diffrent results in case of hyper threading
  // end hardware/OS dependent code

  _return.hardware_info.push_back(ret);
  if (leaf_aggregator_.leafCount() > 0) {
    ret.host_name = "aggregator";
    TClusterHardwareInfo leaf_hardware = leaf_aggregator_.getHardwareInfo(session);
    _return.hardware_info.insert(_return.hardware_info.end(),
                                 leaf_hardware.hardware_info.begin(),
                                 leaf_hardware.hardware_info.end());
  }
}

void DBHandler::get_session_info(TSessionInfo& _return, const TSessionId& session) {
  auto session_ptr = get_session_ptr(session);
  auto stdlog = STDLOG(session_ptr);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto user_metadata = session_ptr->get_currentUser();

  _return.user = user_metadata.userName;
  _return.database = session_ptr->getCatalog().getCurrentDB().dbName;
  _return.start_time = session_ptr->get_start_time();
  _return.is_super = user_metadata.isSuper;
}

void DBHandler::value_to_thrift_column(const TargetValue& tv,
                                       const SQLTypeInfo& ti,
                                       TColumn& column) {
  if (ti.is_array()) {
    TColumn tColumn;
    const auto array_tv = boost::get<ArrayTargetValue>(&tv);
    CHECK(array_tv);
    bool is_null = !array_tv->is_initialized();
    if (!is_null) {
      const auto& vec = array_tv->get();
      for (const auto& elem_tv : vec) {
        value_to_thrift_column(elem_tv, ti.get_elem_type(), tColumn);
      }
    }
    column.data.arr_col.push_back(tColumn);
    column.nulls.push_back(is_null && !ti.get_notnull());
  } else if (ti.is_geometry()) {
    const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
    if (scalar_tv) {
      auto s_n = boost::get<NullableString>(scalar_tv);
      auto s = boost::get<std::string>(s_n);
      if (s) {
        column.data.str_col.push_back(*s);
      } else {
        column.data.str_col.emplace_back("");  // null string
        auto null_p = boost::get<void*>(s_n);
        CHECK(null_p && !*null_p);
      }
      column.nulls.push_back(!s && !ti.get_notnull());
    } else {
      const auto array_tv = boost::get<ArrayTargetValue>(&tv);
      CHECK(array_tv);
      bool is_null = !array_tv->is_initialized();
      if (!is_null) {
        auto elem_type = SQLTypeInfo(kDOUBLE, false);
        TColumn tColumn;
        const auto& vec = array_tv->get();
        for (const auto& elem_tv : vec) {
          value_to_thrift_column(elem_tv, elem_type, tColumn);
        }
        column.data.arr_col.push_back(tColumn);
        column.nulls.push_back(false);
      } else {
        TColumn tColumn;
        column.data.arr_col.push_back(tColumn);
        column.nulls.push_back(is_null && !ti.get_notnull());
      }
    }
  } else {
    CHECK(!ti.is_column());
    const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
    CHECK(scalar_tv);
    if (boost::get<int64_t>(scalar_tv)) {
      int64_t data = *(boost::get<int64_t>(scalar_tv));

      if (ti.is_decimal()) {
        double val = static_cast<double>(data);
        if (ti.get_scale() > 0) {
          val /= pow(10.0, std::abs(ti.get_scale()));
        }
        column.data.real_col.push_back(val);
      } else {
        column.data.int_col.push_back(data);
      }

      switch (ti.get_type()) {
        case kBOOLEAN:
          column.nulls.push_back(data == NULL_BOOLEAN && !ti.get_notnull());
          break;
        case kTINYINT:
          column.nulls.push_back(data == NULL_TINYINT && !ti.get_notnull());
          break;
        case kSMALLINT:
          column.nulls.push_back(data == NULL_SMALLINT && !ti.get_notnull());
          break;
        case kINT:
          column.nulls.push_back(data == NULL_INT && !ti.get_notnull());
          break;
        case kNUMERIC:
        case kDECIMAL:
        case kBIGINT:
          column.nulls.push_back(data == NULL_BIGINT && !ti.get_notnull());
          break;
        case kTIME:
        case kTIMESTAMP:
        case kDATE:
        case kINTERVAL_DAY_TIME:
        case kINTERVAL_YEAR_MONTH:
          column.nulls.push_back(data == NULL_BIGINT && !ti.get_notnull());
          break;
        default:
          column.nulls.push_back(false);
      }
    } else if (boost::get<double>(scalar_tv)) {
      double data = *(boost::get<double>(scalar_tv));
      column.data.real_col.push_back(data);
      if (ti.get_type() == kFLOAT) {
        column.nulls.push_back(data == NULL_FLOAT && !ti.get_notnull());
      } else {
        column.nulls.push_back(data == NULL_DOUBLE && !ti.get_notnull());
      }
    } else if (boost::get<float>(scalar_tv)) {
      CHECK_EQ(kFLOAT, ti.get_type());
      float data = *(boost::get<float>(scalar_tv));
      column.data.real_col.push_back(data);
      column.nulls.push_back(data == NULL_FLOAT && !ti.get_notnull());
    } else if (boost::get<NullableString>(scalar_tv)) {
      auto s_n = boost::get<NullableString>(scalar_tv);
      auto s = boost::get<std::string>(s_n);
      if (s) {
        column.data.str_col.push_back(*s);
      } else {
        column.data.str_col.emplace_back("");  // null string
        auto null_p = boost::get<void*>(s_n);
        CHECK(null_p && !*null_p);
      }
      column.nulls.push_back(!s && !ti.get_notnull());
    } else {
      CHECK(false);
    }
  }
}

TDatum DBHandler::value_to_thrift(const TargetValue& tv, const SQLTypeInfo& ti) {
  TDatum datum;
  const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
  if (!scalar_tv) {
    CHECK(ti.is_array());
    const auto array_tv = boost::get<ArrayTargetValue>(&tv);
    CHECK(array_tv);
    if (array_tv->is_initialized()) {
      const auto& vec = array_tv->get();
      for (const auto& elem_tv : vec) {
        const auto scalar_col_val = value_to_thrift(elem_tv, ti.get_elem_type());
        datum.val.arr_val.push_back(scalar_col_val);
      }
      // Datum is not null, at worst it's an empty array Datum
      datum.is_null = false;
    } else {
      datum.is_null = true;
    }
    return datum;
  }
  if (boost::get<int64_t>(scalar_tv)) {
    int64_t data = *(boost::get<int64_t>(scalar_tv));

    if (ti.is_decimal()) {
      double val = static_cast<double>(data);
      if (ti.get_scale() > 0) {
        val /= pow(10.0, std::abs(ti.get_scale()));
      }
      datum.val.real_val = val;
    } else {
      datum.val.int_val = data;
    }

    switch (ti.get_type()) {
      case kBOOLEAN:
        datum.is_null = (datum.val.int_val == NULL_BOOLEAN);
        break;
      case kTINYINT:
        datum.is_null = (datum.val.int_val == NULL_TINYINT);
        break;
      case kSMALLINT:
        datum.is_null = (datum.val.int_val == NULL_SMALLINT);
        break;
      case kINT:
        datum.is_null = (datum.val.int_val == NULL_INT);
        break;
      case kDECIMAL:
      case kNUMERIC:
      case kBIGINT:
        datum.is_null = (datum.val.int_val == NULL_BIGINT);
        break;
      case kTIME:
      case kTIMESTAMP:
      case kDATE:
      case kINTERVAL_DAY_TIME:
      case kINTERVAL_YEAR_MONTH:
        datum.is_null = (datum.val.int_val == NULL_BIGINT);
        break;
      default:
        datum.is_null = false;
    }
  } else if (boost::get<double>(scalar_tv)) {
    datum.val.real_val = *(boost::get<double>(scalar_tv));
    if (ti.get_type() == kFLOAT) {
      datum.is_null = (datum.val.real_val == NULL_FLOAT);
    } else {
      datum.is_null = (datum.val.real_val == NULL_DOUBLE);
    }
  } else if (boost::get<float>(scalar_tv)) {
    CHECK_EQ(kFLOAT, ti.get_type());
    datum.val.real_val = *(boost::get<float>(scalar_tv));
    datum.is_null = (datum.val.real_val == NULL_FLOAT);
  } else if (boost::get<NullableString>(scalar_tv)) {
    auto s_n = boost::get<NullableString>(scalar_tv);
    auto s = boost::get<std::string>(s_n);
    if (s) {
      datum.val.str_val = *s;
    } else {
      auto null_p = boost::get<void*>(s_n);
      CHECK(null_p && !*null_p);
    }
    datum.is_null = !s;
  } else {
    CHECK(false);
  }
  return datum;
}

void DBHandler::sql_execute_local(
    TQueryResult& _return,
    const QueryStateProxy& query_state_proxy,
    const std::shared_ptr<Catalog_Namespace::SessionInfo> session_ptr,
    const std::string& query_str,
    const bool column_format,
    const std::string& nonce,
    const int32_t first_n,
    const int32_t at_most_n,
    const bool use_calcite) {
  _return.total_time_ms = 0;
  _return.nonce = nonce;
  ParserWrapper pw{query_str};
  switch (pw.getQueryType()) {
    case ParserWrapper::QueryType::Read: {
      _return.query_type = TQueryType::READ;
      VLOG(1) << "query type: READ";
      break;
    }
    case ParserWrapper::QueryType::Write: {
      _return.query_type = TQueryType::WRITE;
      VLOG(1) << "query type: WRITE";
      break;
    }
    case ParserWrapper::QueryType::SchemaRead: {
      _return.query_type = TQueryType::SCHEMA_READ;
      VLOG(1) << "query type: SCHEMA READ";
      break;
    }
    case ParserWrapper::QueryType::SchemaWrite: {
      _return.query_type = TQueryType::SCHEMA_WRITE;
      VLOG(1) << "query type: SCHEMA WRITE";
      break;
    }
    default: {
      _return.query_type = TQueryType::UNKNOWN;
      LOG(WARNING) << "query type: UNKNOWN";
      break;
    }
  }
  ExecutionResult result;
  _return.total_time_ms += measure<>::execution([&]() {
    DBHandler::sql_execute_impl(result,
                                query_state_proxy,
                                column_format,
                                session_ptr->get_executor_device_type(),
                                first_n,
                                at_most_n,
                                use_calcite);
    DBHandler::convertData(
        _return, result, query_state_proxy, query_str, column_format, first_n, at_most_n);
  });
}

void DBHandler::convertData(TQueryResult& _return,
                            ExecutionResult& result,
                            const QueryStateProxy& query_state_proxy,
                            const std::string& query_str,
                            const bool column_format,
                            const int32_t first_n,
                            const int32_t at_most_n) {
  _return.execution_time_ms += result.getExecutionTime();
  if (result.empty()) {
    return;
  }

  switch (result.getResultType()) {
    case ExecutionResult::QueryResult:
      convertRows(_return,
                  query_state_proxy,
                  result.getTargetsMeta(),
                  *result.getRows(),
                  column_format,
                  first_n,
                  at_most_n);
      break;
    case ExecutionResult::SimpleResult:
      convertResult(_return, *result.getRows(), true);
      break;
    case ExecutionResult::Explaination:
      convertExplain(_return, *result.getRows(), true);
      break;
    case ExecutionResult::CalciteDdl:
      convertRows(_return,
                  query_state_proxy,
                  result.getTargetsMeta(),
                  *result.getRows(),
                  column_format,
                  -1,
                  -1);
      break;
  }
}

void DBHandler::sql_execute(TQueryResult& _return,
                            const TSessionId& session,
                            const std::string& query_str,
                            const bool column_format,
                            const std::string& nonce,
                            const int32_t first_n,
                            const int32_t at_most_n) {
  const std::string exec_ra_prefix = "execute relalg";
  const bool use_calcite = !boost::starts_with(query_str, exec_ra_prefix);
  auto actual_query =
      use_calcite ? query_str : boost::trim_copy(query_str.substr(exec_ra_prefix.size()));
  auto session_ptr = get_session_ptr(session);
  auto query_state = create_query_state(session_ptr, actual_query);
  auto stdlog = STDLOG(session_ptr, query_state);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  stdlog.appendNameValuePairs("nonce", nonce);
  auto timer = DEBUG_TIMER(__func__);
  try {
    ScopeGuard reset_was_geo_copy_from = [this, &session_ptr] {
      geo_copy_from_sessions.remove(session_ptr->get_session_id());
    };

    if (first_n >= 0 && at_most_n >= 0) {
      THROW_MAPD_EXCEPTION(
          std::string("At most one of first_n and at_most_n can be set"));
    }

    if (leaf_aggregator_.leafCount() > 0) {
      if (!agg_handler_) {
        THROW_MAPD_EXCEPTION("Distributed support is disabled.");
      }
      _return.total_time_ms = measure<>::execution([&]() {
        agg_handler_->cluster_execute(_return,
                                      query_state->createQueryStateProxy(),
                                      query_state->getQueryStr(),
                                      column_format,
                                      nonce,
                                      first_n,
                                      at_most_n,
                                      system_parameters_);
      });
      _return.nonce = nonce;
    } else {
      sql_execute_local(_return,
                        query_state->createQueryStateProxy(),
                        session_ptr,
                        actual_query,
                        column_format,
                        nonce,
                        first_n,
                        at_most_n,
                        use_calcite);
    }
    _return.total_time_ms += process_geo_copy_from(session);
    std::string debug_json = timer.stopAndGetJson();
    if (!debug_json.empty()) {
      _return.__set_debug(std::move(debug_json));
    }
    stdlog.appendNameValuePairs(
        "execution_time_ms",
        _return.execution_time_ms,
        "total_time_ms",  // BE-3420 - Redundant with duration field
        stdlog.duration<std::chrono::milliseconds>());
    VLOG(1) << "Table Schema Locks:\n" << lockmgr::TableSchemaLockMgr::instance();
    VLOG(1) << "Table Data Locks:\n" << lockmgr::TableDataLockMgr::instance();
  } catch (const std::exception& e) {
    if (strstr(e.what(), "java.lang.NullPointerException")) {
      THROW_MAPD_EXCEPTION(std::string("Exception: ") +
                           "query failed from broken view or other schema related issue");
    } else if (strstr(e.what(), "SQL Error: Encountered \";\"")) {
      THROW_MAPD_EXCEPTION("multiple SQL statements not allowed");
    } else if (strstr(e.what(), "SQL Error: Encountered \"<EOF>\" at line 0, column 0")) {
      THROW_MAPD_EXCEPTION("empty SQL statment not allowed");
    } else {
      THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
    }
  }
}

void DBHandler::sql_execute(ExecutionResult& _return,
                            const TSessionId& session,
                            const std::string& query_str,
                            const bool column_format,
                            const int32_t first_n,
                            const int32_t at_most_n) {
  const std::string exec_ra_prefix = "execute relalg";
  const bool use_calcite = !boost::starts_with(query_str, exec_ra_prefix);
  auto actual_query =
      use_calcite ? query_str : boost::trim_copy(query_str.substr(exec_ra_prefix.size()));

  auto session_ptr = get_session_ptr(session);
  auto query_state = create_query_state(session_ptr, actual_query);
  auto stdlog = STDLOG(session_ptr, query_state);
  auto timer = DEBUG_TIMER(__func__);

  try {
    ScopeGuard reset_was_geo_copy_from = [this, &session_ptr] {
      geo_copy_from_sessions.remove(session_ptr->get_session_id());
    };

    if (first_n >= 0 && at_most_n >= 0) {
      THROW_MAPD_EXCEPTION(
          std::string("At most one of first_n and at_most_n can be set"));
    }
    auto total_time_ms = measure<>::execution([&]() {
      DBHandler::sql_execute_impl(_return,
                                  query_state->createQueryStateProxy(),
                                  column_format,
                                  session_ptr->get_executor_device_type(),
                                  first_n,
                                  at_most_n,
                                  use_calcite);
    });

    _return.setExecutionTime(total_time_ms + process_geo_copy_from(session));

    stdlog.appendNameValuePairs(
        "execution_time_ms",
        _return.getExecutionTime(),
        "total_time_ms",  // BE-3420 - Redundant with duration field
        stdlog.duration<std::chrono::milliseconds>());
    VLOG(1) << "Table Schema Locks:\n" << lockmgr::TableSchemaLockMgr::instance();
    VLOG(1) << "Table Data Locks:\n" << lockmgr::TableDataLockMgr::instance();
  } catch (const std::exception& e) {
    if (strstr(e.what(), "java.lang.NullPointerException")) {
      THROW_MAPD_EXCEPTION(std::string("Exception: ") +
                           "query failed from broken view or other schema related issue");
    } else if (strstr(e.what(), "SQL Error: Encountered \";\"")) {
      THROW_MAPD_EXCEPTION("multiple SQL statements not allowed");
    } else if (strstr(e.what(), "SQL Error: Encountered \"<EOF>\" at line 0, column 0")) {
      THROW_MAPD_EXCEPTION("empty SQL statment not allowed");
    } else {
      THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
    }
  }
}

int64_t DBHandler::process_geo_copy_from(const TSessionId& session_id) {
  int64_t total_time_ms(0);
  // if the SQL statement we just executed was a geo COPY FROM, the import
  // parameters were captured, and this flag set, so we do the actual import here
  if (auto geo_copy_from_state = geo_copy_from_sessions(session_id)) {
    // import_geo_table() calls create_table() which calls this function to
    // do the work, so reset the flag now to avoid executing this part a
    // second time at the end of that, which would fail as the table was
    // already created! Also reset the flag with a ScopeGuard on exiting
    // this function any other way, such as an exception from the code above!
    geo_copy_from_sessions.remove(session_id);

    // create table as replicated?
    TCreateParams create_params;
    if (geo_copy_from_state->geo_copy_from_partitions == "REPLICATED") {
      create_params.is_replicated = true;
    }

    // now do (and time) the import
    total_time_ms = measure<>::execution([&]() {
      import_geo_table(
          session_id,
          geo_copy_from_state->geo_copy_from_table,
          geo_copy_from_state->geo_copy_from_file_name,
          copyparams_to_thrift(geo_copy_from_state->geo_copy_from_copy_params),
          TRowDescriptor(),
          create_params);
    });
  }
  return total_time_ms;
}

void DBHandler::sql_execute_df(TDataFrame& _return,
                               const TSessionId& session,
                               const std::string& query_str,
                               const TDeviceType::type device_type,
                               const int32_t device_id,
                               const int32_t first_n,
                               const TArrowTransport::type transport_method) {
  auto session_ptr = get_session_ptr(session);
  auto query_state = create_query_state(session_ptr, query_str);
  auto stdlog = STDLOG(session_ptr, query_state);

  if (device_type == TDeviceType::GPU) {
    const auto executor_device_type = session_ptr->get_executor_device_type();
    if (executor_device_type != ExecutorDeviceType::GPU) {
      THROW_MAPD_EXCEPTION(
          std::string("Exception: GPU mode is not allowed in this session"));
    }
    if (!data_mgr_->gpusPresent()) {
      THROW_MAPD_EXCEPTION(std::string("Exception: no GPU is available in this server"));
    }
    if (device_id < 0 || device_id >= data_mgr_->getCudaMgr()->getDeviceCount()) {
      THROW_MAPD_EXCEPTION(
          std::string("Exception: invalid device_id or unavailable GPU with this ID"));
    }
  }
  _return.execution_time_ms = 0;

  mapd_shared_lock<mapd_shared_mutex> executeReadLock(
      *legacylockmgr::LockMgr<mapd_shared_mutex, bool>::getMutex(
          legacylockmgr::ExecutorOuterLock, true));
  auto query_state_proxy = query_state->createQueryStateProxy();
  try {
    ParserWrapper pw{query_str};
    if (!pw.is_ddl && !pw.is_update_dml &&
        !(pw.getExplainType() == ParserWrapper::ExplainType::Other)) {
      std::string query_ra;
      lockmgr::LockedTableDescriptors locks;
      _return.execution_time_ms += measure<>::execution([&]() {
        TPlanResult result;
        std::tie(result, locks) =
            parse_to_ra(query_state_proxy, query_str, {}, true, system_parameters_);
        query_ra = result.plan_result;
      });

      if (pw.isCalciteExplain()) {
        throw std::runtime_error("explain is not unsupported by current thrift API");
      }
      if (g_enable_runtime_query_interrupt) {
        auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
        executor->enrollQuerySession(session_ptr->get_session_id(),
                                     query_str,
                                     query_state->getQuerySubmittedTime(),
                                     Executor::UNITARY_EXECUTOR_ID,
                                     QuerySessionStatus::QueryStatus::PENDING_QUEUE);
      }
      execute_rel_alg_df(_return,
                         query_ra,
                         query_state_proxy,
                         *session_ptr,
                         device_type == TDeviceType::CPU ? ExecutorDeviceType::CPU
                                                         : ExecutorDeviceType::GPU,
                         static_cast<size_t>(device_id),
                         first_n,
                         transport_method);
      return;
    }
  } catch (std::exception& e) {
    THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
  }
  THROW_MAPD_EXCEPTION(
      "Exception: DDL or update DML are not unsupported by current thrift API");
}

void DBHandler::sql_execute_gdf(TDataFrame& _return,
                                const TSessionId& session,
                                const std::string& query_str,
                                const int32_t device_id,
                                const int32_t first_n) {
  auto stdlog = STDLOG(get_session_ptr(session));
  sql_execute_df(_return,
                 session,
                 query_str,
                 TDeviceType::GPU,
                 device_id,
                 first_n,
                 TArrowTransport::SHARED_MEMORY);
}

// For now we have only one user of a data frame in all cases.
void DBHandler::deallocate_df(const TSessionId& session,
                              const TDataFrame& df,
                              const TDeviceType::type device_type,
                              const int32_t device_id) {
  auto stdlog = STDLOG(get_session_ptr(session));
  std::string serialized_cuda_handle = "";
  if (device_type == TDeviceType::GPU) {
    std::lock_guard<std::mutex> map_lock(handle_to_dev_ptr_mutex_);
    if (ipc_handle_to_dev_ptr_.count(df.df_handle) != size_t(1)) {
      TOmniSciException ex;
      ex.error_msg = std::string(
          "Exception: current data frame handle is not bookkept or been inserted "
          "twice");
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    serialized_cuda_handle = ipc_handle_to_dev_ptr_[df.df_handle];
    ipc_handle_to_dev_ptr_.erase(df.df_handle);
  }
  std::vector<char> sm_handle(df.sm_handle.begin(), df.sm_handle.end());
  std::vector<char> df_handle(df.df_handle.begin(), df.df_handle.end());
  ArrowResult result{
      sm_handle, df.sm_size, df_handle, df.df_size, serialized_cuda_handle};
  ArrowResultSet::deallocateArrowResultBuffer(
      result,
      device_type == TDeviceType::CPU ? ExecutorDeviceType::CPU : ExecutorDeviceType::GPU,
      device_id,
      data_mgr_);
}

std::string DBHandler::apply_copy_to_shim(const std::string& query_str) {
  auto result = query_str;
  {
    // boost::regex copy_to{R"(COPY\s\((.*)\)\sTO\s(.*))", boost::regex::extended |
    // boost::regex::icase};
    boost::regex copy_to{R"(COPY\s*\(([^#])(.+)\)\s+TO\s)",
                         boost::regex::extended | boost::regex::icase};
    apply_shim(result, copy_to, [](std::string& result, const boost::smatch& what) {
      result.replace(
          what.position(), what.length(), "COPY (#~#" + what[1] + what[2] + "#~#) TO  ");
    });
  }
  return result;
}

void DBHandler::sql_validate(TRowDescriptor& _return,
                             const TSessionId& session,
                             const std::string& query_str) {
  try {
    auto stdlog = STDLOG(get_session_ptr(session));
    stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
    auto query_state = create_query_state(stdlog.getSessionInfo(), query_str);
    stdlog.setQueryState(query_state);

    ParserWrapper pw{query_str};
    if ((pw.getExplainType() != ParserWrapper::ExplainType::None) || pw.is_ddl ||
        pw.is_update_dml) {
      throw std::runtime_error("Can only validate SELECT statements.");
    }

    const auto execute_read_lock = mapd_shared_lock<mapd_shared_mutex>(
        *legacylockmgr::LockMgr<mapd_shared_mutex, bool>::getMutex(
            legacylockmgr::ExecutorOuterLock, true));

    TPlanResult parse_result;
    lockmgr::LockedTableDescriptors locks;
    std::tie(parse_result, locks) = parse_to_ra(query_state->createQueryStateProxy(),
                                                query_state->getQueryStr(),
                                                {},
                                                true,
                                                system_parameters_,
                                                /*check_privileges=*/true);
    const auto query_ra = parse_result.plan_result;

    const auto result = validate_rel_alg(query_ra, query_state->createQueryStateProxy());
    _return = fixup_row_descriptor(result.row_set.row_desc,
                                   query_state->getConstSessionInfo()->getCatalog());
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION("Exception: " + std::string(e.what()));
  }
}

namespace {

struct ProjectionTokensForCompletion {
  std::unordered_set<std::string> uc_column_names;
  std::unordered_set<std::string> uc_column_table_qualifiers;
};

// Extract what looks like a (qualified) identifier from the partial query.
// The results will be used to rank the auto-completion results: tables which
// contain at least one of the identifiers first.
ProjectionTokensForCompletion extract_projection_tokens_for_completion(
    const std::string& sql) {
  boost::regex id_regex{R"(([[:alnum:]]|_|\.)+)",
                        boost::regex::extended | boost::regex::icase};
  boost::sregex_token_iterator tok_it(sql.begin(), sql.end(), id_regex, 0);
  boost::sregex_token_iterator end;
  std::unordered_set<std::string> uc_column_names;
  std::unordered_set<std::string> uc_column_table_qualifiers;
  for (; tok_it != end; ++tok_it) {
    std::string column_name = *tok_it;
    std::vector<std::string> column_tokens;
    boost::split(column_tokens, column_name, boost::is_any_of("."));
    if (column_tokens.size() == 2) {
      // If the column name is qualified, take user's word.
      uc_column_table_qualifiers.insert(to_upper(column_tokens.front()));
    } else {
      uc_column_names.insert(to_upper(column_name));
    }
  }
  return {uc_column_names, uc_column_table_qualifiers};
}

}  // namespace

void DBHandler::get_completion_hints(std::vector<TCompletionHint>& hints,
                                     const TSessionId& session,
                                     const std::string& sql,
                                     const int cursor) {
  auto stdlog = STDLOG(get_session_ptr(session));
  std::vector<std::string> visible_tables;  // Tables allowed for the given session.
  get_completion_hints_unsorted(hints, visible_tables, stdlog, sql, cursor);
  const auto proj_tokens = extract_projection_tokens_for_completion(sql);
  auto compatible_table_names = get_uc_compatible_table_names_by_column(
      proj_tokens.uc_column_names, visible_tables, stdlog);
  // Add the table qualifiers explicitly specified by the user.
  compatible_table_names.insert(proj_tokens.uc_column_table_qualifiers.begin(),
                                proj_tokens.uc_column_table_qualifiers.end());
  // Sort the hints by category, from COLUMN (most specific) to KEYWORD.
  std::sort(
      hints.begin(),
      hints.end(),
      [&compatible_table_names](const TCompletionHint& lhs, const TCompletionHint& rhs) {
        if (lhs.type == TCompletionHintType::TABLE &&
            rhs.type == TCompletionHintType::TABLE) {
          // Between two tables, one which is compatible with the specified
          // projections and one which isn't, pick the one which is compatible.
          if (compatible_table_names.find(to_upper(lhs.hints.back())) !=
                  compatible_table_names.end() &&
              compatible_table_names.find(to_upper(rhs.hints.back())) ==
                  compatible_table_names.end()) {
            return true;
          }
        }
        return lhs.type < rhs.type;
      });
}

void DBHandler::get_completion_hints_unsorted(std::vector<TCompletionHint>& hints,
                                              std::vector<std::string>& visible_tables,
                                              query_state::StdLog& stdlog,
                                              const std::string& sql,
                                              const int cursor) {
  const auto& session_info = *stdlog.getConstSessionInfo();
  try {
    get_tables_impl(visible_tables, session_info, GET_PHYSICAL_TABLES_AND_VIEWS);

    // Filter out keywords suggested by Calcite which we don't support.
    hints = just_whitelisted_keyword_hints(
        calcite_->getCompletionHints(session_info, visible_tables, sql, cursor));
  } catch (const std::exception& e) {
    TOmniSciException ex;
    ex.error_msg = "Exception: " + std::string(e.what());
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  boost::regex from_expr{R"(\s+from\s+)", boost::regex::extended | boost::regex::icase};
  const size_t length_to_cursor =
      cursor < 0 ? sql.size() : std::min(sql.size(), static_cast<size_t>(cursor));
  // Trust hints from Calcite after the FROM keyword.
  if (boost::regex_search(sql.cbegin(), sql.cbegin() + length_to_cursor, from_expr)) {
    return;
  }
  // Before FROM, the query is too incomplete for context-sensitive completions.
  get_token_based_completions(hints, stdlog, visible_tables, sql, cursor);
}

void DBHandler::get_token_based_completions(std::vector<TCompletionHint>& hints,
                                            query_state::StdLog& stdlog,
                                            std::vector<std::string>& visible_tables,
                                            const std::string& sql,
                                            const int cursor) {
  const auto last_word =
      find_last_word_from_cursor(sql, cursor < 0 ? sql.size() : cursor);
  boost::regex select_expr{R"(\s*select\s+)",
                           boost::regex::extended | boost::regex::icase};
  const size_t length_to_cursor =
      cursor < 0 ? sql.size() : std::min(sql.size(), static_cast<size_t>(cursor));
  // After SELECT but before FROM, look for all columns in all tables which match the
  // prefix.
  if (boost::regex_search(sql.cbegin(), sql.cbegin() + length_to_cursor, select_expr)) {
    const auto column_names_by_table = fill_column_names_by_table(visible_tables, stdlog);
    // Trust the fully qualified columns the most.
    if (get_qualified_column_hints(hints, last_word, column_names_by_table)) {
      return;
    }
    // Not much information to use, just retrieve column names which match the prefix.
    if (should_suggest_column_hints(sql)) {
      get_column_hints(hints, last_word, column_names_by_table);
      return;
    }
    const std::string kFromKeyword{"FROM"};
    if (boost::istarts_with(kFromKeyword, last_word)) {
      TCompletionHint keyword_hint;
      keyword_hint.type = TCompletionHintType::KEYWORD;
      keyword_hint.replaced = last_word;
      keyword_hint.hints.emplace_back(kFromKeyword);
      hints.push_back(keyword_hint);
    }
  } else {
    const std::string kSelectKeyword{"SELECT"};
    if (boost::istarts_with(kSelectKeyword, last_word)) {
      TCompletionHint keyword_hint;
      keyword_hint.type = TCompletionHintType::KEYWORD;
      keyword_hint.replaced = last_word;
      keyword_hint.hints.emplace_back(kSelectKeyword);
      hints.push_back(keyword_hint);
    }
  }
}

std::unordered_map<std::string, std::unordered_set<std::string>>
DBHandler::fill_column_names_by_table(std::vector<std::string>& table_names,
                                      query_state::StdLog& stdlog) {
  std::unordered_map<std::string, std::unordered_set<std::string>> column_names_by_table;
  for (auto it = table_names.begin(); it != table_names.end();) {
    TTableDetails table_details;
    try {
      get_table_details_impl(table_details, stdlog, *it, false, false);
    } catch (const TOmniSciException& e) {
      // Remove the corrupted Table/View name from the list for further processing.
      it = table_names.erase(it);
      continue;
    }
    for (const auto& column_type : table_details.row_desc) {
      column_names_by_table[*it].emplace(column_type.col_name);
    }
    ++it;
  }
  return column_names_by_table;
}

ConnectionInfo DBHandler::getConnectionInfo() const {
  return ConnectionInfo{TrackingProcessor::client_address,
                        TrackingProcessor::client_protocol};
}

std::unordered_set<std::string> DBHandler::get_uc_compatible_table_names_by_column(
    const std::unordered_set<std::string>& uc_column_names,
    std::vector<std::string>& table_names,
    query_state::StdLog& stdlog) {
  std::unordered_set<std::string> compatible_table_names_by_column;
  for (auto it = table_names.begin(); it != table_names.end();) {
    TTableDetails table_details;
    try {
      get_table_details_impl(table_details, stdlog, *it, false, false);
    } catch (const TOmniSciException& e) {
      // Remove the corrupted Table/View name from the list for further processing.
      it = table_names.erase(it);
      continue;
    }
    for (const auto& column_type : table_details.row_desc) {
      if (uc_column_names.find(to_upper(column_type.col_name)) != uc_column_names.end()) {
        compatible_table_names_by_column.emplace(to_upper(*it));
        break;
      }
    }
    ++it;
  }
  return compatible_table_names_by_column;
}

TQueryResult DBHandler::validate_rel_alg(const std::string& query_ra,
                                         QueryStateProxy query_state_proxy) {
  TQueryResult _return;
  ExecutionResult result;
  auto execute_rel_alg_task = std::make_shared<QueryDispatchQueue::Task>(
      [this, &result, &query_state_proxy, &query_ra](const size_t executor_index) {
        execute_rel_alg(result,
                        query_state_proxy,
                        query_ra,
                        true,
                        ExecutorDeviceType::CPU,
                        -1,
                        -1,
                        /*just_validate=*/true,
                        /*find_filter_push_down_candidates=*/false,
                        ExplainInfo::defaults(),
                        executor_index);
      });
  CHECK(dispatch_queue_);
  dispatch_queue_->submit(execute_rel_alg_task, /*is_update_delete=*/false);
  auto result_future = execute_rel_alg_task->get_future();
  result_future.get();
  DBHandler::convertData(_return, result, query_state_proxy, query_ra, true, -1, -1);
  return _return;
}

void DBHandler::get_roles(std::vector<std::string>& roles, const TSessionId& session) {
  auto stdlog = STDLOG(get_session_ptr(session));
  auto session_ptr = stdlog.getConstSessionInfo();
  if (!session_ptr->get_currentUser().isSuper) {
    // WARNING: This appears to not include roles a user is a member of,
    // if the role has no permissions granted to it.
    roles =
        SysCatalog::instance().getRoles(session_ptr->get_currentUser().userName,
                                        session_ptr->getCatalog().getCurrentDB().dbId);
  } else {
    roles = SysCatalog::instance().getRoles(
        false, true, session_ptr->get_currentUser().userName);
  }
}

bool DBHandler::has_role(const TSessionId& sessionId,
                         const std::string& granteeName,
                         const std::string& roleName) {
  const auto stdlog = STDLOG(get_session_ptr(sessionId));
  const auto session_ptr = stdlog.getConstSessionInfo();
  const auto current_user = session_ptr->get_currentUser();
  if (!current_user.isSuper) {
    if (const auto* user = SysCatalog::instance().getUserGrantee(granteeName);
        user && current_user.userName != granteeName) {
      THROW_MAPD_EXCEPTION("Only super users can check other user's roles.");
    } else if (!SysCatalog::instance().isRoleGrantedToGrantee(
                   current_user.userName, granteeName, true)) {
      THROW_MAPD_EXCEPTION(
          "Only super users can check roles assignment that have not been directly "
          "granted to a user.");
    }
  }
  return SysCatalog::instance().isRoleGrantedToGrantee(granteeName, roleName, false);
}

static TDBObject serialize_db_object(const std::string& roleName,
                                     const DBObject& inObject) {
  TDBObject outObject;
  outObject.objectName = inObject.getName();
  outObject.grantee = roleName;
  outObject.objectId = inObject.getObjectKey().objectId;
  const auto ap = inObject.getPrivileges();
  switch (inObject.getObjectKey().permissionType) {
    case DatabaseDBObjectType:
      outObject.privilegeObjectType = TDBObjectType::DatabaseDBObjectType;
      outObject.privs.push_back(ap.hasPermission(DatabasePrivileges::CREATE_DATABASE));
      outObject.privs.push_back(ap.hasPermission(DatabasePrivileges::DROP_DATABASE));
      outObject.privs.push_back(ap.hasPermission(DatabasePrivileges::VIEW_SQL_EDITOR));
      outObject.privs.push_back(ap.hasPermission(DatabasePrivileges::ACCESS));

      break;
    case TableDBObjectType:
      outObject.privilegeObjectType = TDBObjectType::TableDBObjectType;
      outObject.privs.push_back(ap.hasPermission(TablePrivileges::CREATE_TABLE));
      outObject.privs.push_back(ap.hasPermission(TablePrivileges::DROP_TABLE));
      outObject.privs.push_back(ap.hasPermission(TablePrivileges::SELECT_FROM_TABLE));
      outObject.privs.push_back(ap.hasPermission(TablePrivileges::INSERT_INTO_TABLE));
      outObject.privs.push_back(ap.hasPermission(TablePrivileges::UPDATE_IN_TABLE));
      outObject.privs.push_back(ap.hasPermission(TablePrivileges::DELETE_FROM_TABLE));
      outObject.privs.push_back(ap.hasPermission(TablePrivileges::TRUNCATE_TABLE));
      outObject.privs.push_back(ap.hasPermission(TablePrivileges::ALTER_TABLE));

      break;
    case DashboardDBObjectType:
      outObject.privilegeObjectType = TDBObjectType::DashboardDBObjectType;
      outObject.privs.push_back(ap.hasPermission(DashboardPrivileges::CREATE_DASHBOARD));
      outObject.privs.push_back(ap.hasPermission(DashboardPrivileges::DELETE_DASHBOARD));
      outObject.privs.push_back(ap.hasPermission(DashboardPrivileges::VIEW_DASHBOARD));
      outObject.privs.push_back(ap.hasPermission(DashboardPrivileges::EDIT_DASHBOARD));

      break;
    case ViewDBObjectType:
      outObject.privilegeObjectType = TDBObjectType::ViewDBObjectType;
      outObject.privs.push_back(ap.hasPermission(ViewPrivileges::CREATE_VIEW));
      outObject.privs.push_back(ap.hasPermission(ViewPrivileges::DROP_VIEW));
      outObject.privs.push_back(ap.hasPermission(ViewPrivileges::SELECT_FROM_VIEW));
      outObject.privs.push_back(ap.hasPermission(ViewPrivileges::INSERT_INTO_VIEW));
      outObject.privs.push_back(ap.hasPermission(ViewPrivileges::UPDATE_IN_VIEW));
      outObject.privs.push_back(ap.hasPermission(ViewPrivileges::DELETE_FROM_VIEW));

      break;
    case ServerDBObjectType:
      outObject.privilegeObjectType = TDBObjectType::ServerDBObjectType;
      outObject.privs.push_back(ap.hasPermission(ServerPrivileges::CREATE_SERVER));
      outObject.privs.push_back(ap.hasPermission(ServerPrivileges::DROP_SERVER));
      outObject.privs.push_back(ap.hasPermission(ServerPrivileges::ALTER_SERVER));
      outObject.privs.push_back(ap.hasPermission(ServerPrivileges::SERVER_USAGE));

      break;
    default:
      CHECK(false);
  }
  const int type_val = static_cast<int>(inObject.getType());
  CHECK(type_val >= 0 && type_val < 6);
  outObject.objectType = static_cast<TDBObjectType::type>(type_val);
  return outObject;
}

bool DBHandler::has_database_permission(const AccessPrivileges& privs,
                                        const TDBObjectPermissions& permissions) {
  if (!permissions.__isset.database_permissions_) {
    THROW_MAPD_EXCEPTION("Database permissions not set for check.")
  }
  auto perms = permissions.database_permissions_;
  if ((perms.create_ && !privs.hasPermission(DatabasePrivileges::CREATE_DATABASE)) ||
      (perms.delete_ && !privs.hasPermission(DatabasePrivileges::DROP_DATABASE)) ||
      (perms.view_sql_editor_ &&
       !privs.hasPermission(DatabasePrivileges::VIEW_SQL_EDITOR)) ||
      (perms.access_ && !privs.hasPermission(DatabasePrivileges::ACCESS))) {
    return false;
  } else {
    return true;
  }
}

bool DBHandler::has_table_permission(const AccessPrivileges& privs,
                                     const TDBObjectPermissions& permissions) {
  if (!permissions.__isset.table_permissions_) {
    THROW_MAPD_EXCEPTION("Table permissions not set for check.")
  }
  auto perms = permissions.table_permissions_;
  if ((perms.create_ && !privs.hasPermission(TablePrivileges::CREATE_TABLE)) ||
      (perms.drop_ && !privs.hasPermission(TablePrivileges::DROP_TABLE)) ||
      (perms.select_ && !privs.hasPermission(TablePrivileges::SELECT_FROM_TABLE)) ||
      (perms.insert_ && !privs.hasPermission(TablePrivileges::INSERT_INTO_TABLE)) ||
      (perms.update_ && !privs.hasPermission(TablePrivileges::UPDATE_IN_TABLE)) ||
      (perms.delete_ && !privs.hasPermission(TablePrivileges::DELETE_FROM_TABLE)) ||
      (perms.truncate_ && !privs.hasPermission(TablePrivileges::TRUNCATE_TABLE)) ||
      (perms.alter_ && !privs.hasPermission(TablePrivileges::ALTER_TABLE))) {
    return false;
  } else {
    return true;
  }
}

bool DBHandler::has_dashboard_permission(const AccessPrivileges& privs,
                                         const TDBObjectPermissions& permissions) {
  if (!permissions.__isset.dashboard_permissions_) {
    THROW_MAPD_EXCEPTION("Dashboard permissions not set for check.")
  }
  auto perms = permissions.dashboard_permissions_;
  if ((perms.create_ && !privs.hasPermission(DashboardPrivileges::CREATE_DASHBOARD)) ||
      (perms.delete_ && !privs.hasPermission(DashboardPrivileges::DELETE_DASHBOARD)) ||
      (perms.view_ && !privs.hasPermission(DashboardPrivileges::VIEW_DASHBOARD)) ||
      (perms.edit_ && !privs.hasPermission(DashboardPrivileges::EDIT_DASHBOARD))) {
    return false;
  } else {
    return true;
  }
}

bool DBHandler::has_view_permission(const AccessPrivileges& privs,
                                    const TDBObjectPermissions& permissions) {
  if (!permissions.__isset.view_permissions_) {
    THROW_MAPD_EXCEPTION("View permissions not set for check.")
  }
  auto perms = permissions.view_permissions_;
  if ((perms.create_ && !privs.hasPermission(ViewPrivileges::CREATE_VIEW)) ||
      (perms.drop_ && !privs.hasPermission(ViewPrivileges::DROP_VIEW)) ||
      (perms.select_ && !privs.hasPermission(ViewPrivileges::SELECT_FROM_VIEW)) ||
      (perms.insert_ && !privs.hasPermission(ViewPrivileges::INSERT_INTO_VIEW)) ||
      (perms.update_ && !privs.hasPermission(ViewPrivileges::UPDATE_IN_VIEW)) ||
      (perms.delete_ && !privs.hasPermission(ViewPrivileges::DELETE_FROM_VIEW))) {
    return false;
  } else {
    return true;
  }
}

bool DBHandler::has_server_permission(const AccessPrivileges& privs,
                                      const TDBObjectPermissions& permissions) {
  CHECK(permissions.__isset.server_permissions_);
  auto perms = permissions.server_permissions_;
  if ((perms.create_ && !privs.hasPermission(ServerPrivileges::CREATE_SERVER)) ||
      (perms.drop_ && !privs.hasPermission(ServerPrivileges::DROP_SERVER)) ||
      (perms.alter_ && !privs.hasPermission(ServerPrivileges::ALTER_SERVER)) ||
      (perms.usage_ && !privs.hasPermission(ServerPrivileges::SERVER_USAGE))) {
    return false;
  } else {
    return true;
  }
}

bool DBHandler::has_object_privilege(const TSessionId& sessionId,
                                     const std::string& granteeName,
                                     const std::string& objectName,
                                     const TDBObjectType::type objectType,
                                     const TDBObjectPermissions& permissions) {
  auto stdlog = STDLOG(get_session_ptr(sessionId));
  auto session_ptr = stdlog.getConstSessionInfo();
  auto const& cat = session_ptr->getCatalog();
  auto const& current_user = session_ptr->get_currentUser();
  if (!current_user.isSuper && !SysCatalog::instance().isRoleGrantedToGrantee(
                                   current_user.userName, granteeName, false)) {
    THROW_MAPD_EXCEPTION(
        "Users except superusers can only check privileges for self or roles granted "
        "to "
        "them.")
  }
  Catalog_Namespace::UserMetadata user_meta;
  if (SysCatalog::instance().getMetadataForUser(granteeName, user_meta) &&
      user_meta.isSuper) {
    return true;
  }
  Grantee* grnt = SysCatalog::instance().getGrantee(granteeName);
  if (!grnt) {
    THROW_MAPD_EXCEPTION("User or Role " + granteeName + " does not exist.")
  }
  DBObjectType type;
  std::string func_name;
  switch (objectType) {
    case TDBObjectType::DatabaseDBObjectType:
      type = DBObjectType::DatabaseDBObjectType;
      func_name = "database";
      break;
    case TDBObjectType::TableDBObjectType:
      type = DBObjectType::TableDBObjectType;
      func_name = "table";
      break;
    case TDBObjectType::DashboardDBObjectType:
      type = DBObjectType::DashboardDBObjectType;
      func_name = "dashboard";
      break;
    case TDBObjectType::ViewDBObjectType:
      type = DBObjectType::ViewDBObjectType;
      func_name = "view";
      break;
    case TDBObjectType::ServerDBObjectType:
      type = DBObjectType::ServerDBObjectType;
      func_name = "server";
      break;
    default:
      THROW_MAPD_EXCEPTION("Invalid object type (" + std::to_string(objectType) + ").");
  }
  DBObject req_object(objectName, type);
  req_object.loadKey(cat);

  auto grantee_object = grnt->findDbObject(req_object.getObjectKey(), false);
  if (grantee_object) {
    // if grantee has privs on the object
    return permissionFuncMap_[func_name](grantee_object->getPrivileges(), permissions);
  } else {
    // no privileges on that object
    return false;
  }
}

void DBHandler::get_db_objects_for_grantee(std::vector<TDBObject>& TDBObjectsForRole,
                                           const TSessionId& sessionId,
                                           const std::string& roleName) {
  auto stdlog = STDLOG(get_session_ptr(sessionId));
  auto session_ptr = stdlog.getConstSessionInfo();
  auto const& user = session_ptr->get_currentUser();
  if (!user.isSuper &&
      !SysCatalog::instance().isRoleGrantedToGrantee(user.userName, roleName, false)) {
    return;
  }
  auto* rl = SysCatalog::instance().getGrantee(roleName);
  if (rl) {
    auto dbId = session_ptr->getCatalog().getCurrentDB().dbId;
    for (auto& dbObject : *rl->getDbObjects(true)) {
      if (dbObject.first.dbId != dbId) {
        // TODO (max): it doesn't scale well in case we have many DBs (not a typical
        // usecase for now, though)
        continue;
      }
      TDBObject tdbObject = serialize_db_object(roleName, *dbObject.second);
      TDBObjectsForRole.push_back(tdbObject);
    }
  } else {
    THROW_MAPD_EXCEPTION("User or role " + roleName + " does not exist.");
  }
}

void DBHandler::get_db_object_privs(std::vector<TDBObject>& TDBObjects,
                                    const TSessionId& sessionId,
                                    const std::string& objectName,
                                    const TDBObjectType::type type) {
  auto stdlog = STDLOG(get_session_ptr(sessionId));
  auto session_ptr = stdlog.getConstSessionInfo();
  DBObjectType object_type;
  switch (type) {
    case TDBObjectType::DatabaseDBObjectType:
      object_type = DBObjectType::DatabaseDBObjectType;
      break;
    case TDBObjectType::TableDBObjectType:
      object_type = DBObjectType::TableDBObjectType;
      break;
    case TDBObjectType::DashboardDBObjectType:
      object_type = DBObjectType::DashboardDBObjectType;
      break;
    case TDBObjectType::ViewDBObjectType:
      object_type = DBObjectType::ViewDBObjectType;
      break;
    case TDBObjectType::ServerDBObjectType:
      object_type = DBObjectType::ServerDBObjectType;
      break;
    default:
      THROW_MAPD_EXCEPTION("Failed to get object privileges for " + objectName +
                           ": unknown object type (" + std::to_string(type) + ").");
  }
  DBObject object_to_find(objectName, object_type);

  // TODO(adb): Use DatabaseLock to protect method
  try {
    if (object_type == DashboardDBObjectType) {
      if (objectName == "") {
        object_to_find = DBObject(-1, object_type);
      } else {
        object_to_find = DBObject(std::stoi(objectName), object_type);
      }
    } else if ((object_type == TableDBObjectType || object_type == ViewDBObjectType) &&
               !objectName.empty()) {
      // special handling for view / table
      auto const& cat = session_ptr->getCatalog();
      auto td = cat.getMetadataForTable(objectName, false);
      if (td) {
        object_type = td->isView ? ViewDBObjectType : TableDBObjectType;
        object_to_find = DBObject(objectName, object_type);
      }
    }
    object_to_find.loadKey(session_ptr->getCatalog());
  } catch (const std::exception&) {
    THROW_MAPD_EXCEPTION("Object with name " + objectName + " does not exist.");
  }

  // object type on database level
  DBObject object_to_find_dblevel("", object_type);
  object_to_find_dblevel.loadKey(session_ptr->getCatalog());
  // if user is superuser respond with a full priv
  if (session_ptr->get_currentUser().isSuper) {
    // using ALL_TABLE here to set max permissions
    DBObject dbObj{object_to_find.getObjectKey(),
                   AccessPrivileges::ALL_TABLE,
                   session_ptr->get_currentUser().userId};
    dbObj.setName("super");
    TDBObjects.push_back(
        serialize_db_object(session_ptr->get_currentUser().userName, dbObj));
  };

  std::vector<std::string> grantees =
      SysCatalog::instance().getRoles(true,
                                      session_ptr->get_currentUser().isSuper,
                                      session_ptr->get_currentUser().userName);
  for (const auto& grantee : grantees) {
    DBObject* object_found;
    auto* gr = SysCatalog::instance().getGrantee(grantee);
    if (gr && (object_found = gr->findDbObject(object_to_find.getObjectKey(), true))) {
      TDBObjects.push_back(serialize_db_object(grantee, *object_found));
    }
    // check object permissions on Database level
    if (gr &&
        (object_found = gr->findDbObject(object_to_find_dblevel.getObjectKey(), true))) {
      TDBObjects.push_back(serialize_db_object(grantee, *object_found));
    }
  }
}

void DBHandler::get_all_roles_for_user(std::vector<std::string>& roles,
                                       const TSessionId& sessionId,
                                       const std::string& granteeName) {
  auto stdlog = STDLOG(get_session_ptr(sessionId));
  auto session_ptr = stdlog.getConstSessionInfo();
  auto* grantee = SysCatalog::instance().getGrantee(granteeName);
  if (grantee) {
    if (session_ptr->get_currentUser().isSuper) {
      roles = grantee->getRoles();
    } else if (grantee->isUser()) {
      if (session_ptr->get_currentUser().userName == granteeName) {
        roles = grantee->getRoles();
      } else {
        THROW_MAPD_EXCEPTION(
            "Only a superuser is authorized to request list of roles granted to "
            "another "
            "user.");
      }
    } else {
      CHECK(!grantee->isUser());
      // granteeName is actually a roleName here and we can check a role
      // only if it is granted to us
      if (SysCatalog::instance().isRoleGrantedToGrantee(
              session_ptr->get_currentUser().userName, granteeName, false)) {
        roles = grantee->getRoles();
      } else {
        THROW_MAPD_EXCEPTION("A user can check only roles granted to him.");
      }
    }
  } else {
    THROW_MAPD_EXCEPTION("Grantee " + granteeName + " does not exist.");
  }
}

namespace {
std::string dump_table_col_names(
    const std::map<std::string, std::vector<std::string>>& table_col_names) {
  std::ostringstream oss;
  for (const auto& [table_name, col_names] : table_col_names) {
    oss << ":" << table_name;
    for (const auto& col_name : col_names) {
      oss << "," << col_name;
    }
  }
  return oss.str();
}
}  // namespace

void DBHandler::get_result_row_for_pixel(
    TPixelTableRowResult& _return,
    const TSessionId& session,
    const int64_t widget_id,
    const TPixel& pixel,
    const std::map<std::string, std::vector<std::string>>& table_col_names,
    const bool column_format,
    const int32_t pixel_radius,
    const std::string& nonce) {
  auto stdlog = STDLOG(get_session_ptr(session),
                       "widget_id",
                       widget_id,
                       "pixel.x",
                       pixel.x,
                       "pixel.y",
                       pixel.y,
                       "column_format",
                       column_format,
                       "pixel_radius",
                       pixel_radius,
                       "table_col_names",
                       dump_table_col_names(table_col_names),
                       "nonce",
                       nonce);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getSessionInfo();
  if (!render_handler_) {
    THROW_MAPD_EXCEPTION("Backend rendering is disabled.");
  }

  try {
    render_handler_->get_result_row_for_pixel(_return,
                                              session_ptr,
                                              widget_id,
                                              pixel,
                                              table_col_names,
                                              column_format,
                                              pixel_radius,
                                              nonce);
  } catch (std::exception& e) {
    THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
  }
}

TColumnType DBHandler::populateThriftColumnType(const Catalog* cat,
                                                const ColumnDescriptor* cd) {
  TColumnType col_type;
  col_type.col_name = cd->columnName;
  col_type.src_name = cd->sourceName;
  col_type.col_id = cd->columnId;
  col_type.col_type.type = type_to_thrift(cd->columnType);
  col_type.col_type.encoding = encoding_to_thrift(cd->columnType);
  col_type.col_type.nullable = !cd->columnType.get_notnull();
  col_type.col_type.is_array = cd->columnType.get_type() == kARRAY;
  if (col_type.col_type.is_array || cd->columnType.get_type() == kDATE) {
    col_type.col_type.size = cd->columnType.get_size();  // only for arrays and dates
  }
  if (IS_GEO(cd->columnType.get_type())) {
    ThriftSerializers::fixup_geo_column_descriptor(
        col_type, cd->columnType.get_subtype(), cd->columnType.get_output_srid());
  } else {
    col_type.col_type.precision = cd->columnType.get_precision();
    col_type.col_type.scale = cd->columnType.get_scale();
  }
  col_type.is_system = cd->isSystemCol;
  if (cd->columnType.get_compression() == EncodingType::kENCODING_DICT &&
      cat != nullptr) {
    // have to get the actual size of the encoding from the dictionary definition
    const int dict_id = cd->columnType.get_comp_param();
    if (!cat->getMetadataForDict(dict_id, false)) {
      col_type.col_type.comp_param = 0;
      return col_type;
    }
    auto dd = cat->getMetadataForDict(dict_id, false);
    if (!dd) {
      THROW_MAPD_EXCEPTION("Dictionary doesn't exist");
    }
    col_type.col_type.comp_param = dd->dictNBits;
  } else {
    col_type.col_type.comp_param =
        (cd->columnType.is_date_in_days() && cd->columnType.get_comp_param() == 0)
            ? 32
            : cd->columnType.get_comp_param();
  }
  col_type.is_reserved_keyword = ImportHelpers::is_reserved_name(col_type.col_name);
  return col_type;
}

void DBHandler::get_internal_table_details(TTableDetails& _return,
                                           const TSessionId& session,
                                           const std::string& table_name) {
  auto stdlog = STDLOG(get_session_ptr(session), "table_name", table_name);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  get_table_details_impl(_return, stdlog, table_name, true, false);
}

void DBHandler::get_internal_table_details_for_database(
    TTableDetails& _return,
    const TSessionId& session,
    const std::string& table_name,
    const std::string& database_name) {
  auto stdlog = STDLOG(get_session_ptr(session), "table_name", table_name);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  get_table_details_impl(_return, stdlog, table_name, true, false, database_name);
}

void DBHandler::get_table_details(TTableDetails& _return,
                                  const TSessionId& session,
                                  const std::string& table_name) {
  auto stdlog = STDLOG(get_session_ptr(session), "table_name", table_name);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  get_table_details_impl(_return, stdlog, table_name, false, false);
}

void DBHandler::get_table_details_for_database(TTableDetails& _return,
                                               const TSessionId& session,
                                               const std::string& table_name,
                                               const std::string& database_name) {
  auto stdlog = STDLOG(get_session_ptr(session), "table_name", table_name);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  get_table_details_impl(_return, stdlog, table_name, false, false, database_name);
}

void DBHandler::get_table_details_impl(TTableDetails& _return,
                                       query_state::StdLog& stdlog,
                                       const std::string& table_name,
                                       const bool get_system,
                                       const bool get_physical,
                                       const std::string& database_name) {
  try {
    auto session_info = stdlog.getSessionInfo();
    auto& cat = (database_name.empty())
                    ? session_info->getCatalog()
                    : *SysCatalog::instance().getCatalog(database_name);
    const auto td_with_lock =
        lockmgr::TableSchemaLockContainer<lockmgr::ReadLock>::acquireTableDescriptor(
            cat, table_name, false);
    const auto td = td_with_lock();
    CHECK(td);

    bool have_privileges_on_view_sources = true;
    if (td->isView) {
      auto query_state = create_query_state(session_info, td->viewSQL);
      stdlog.setQueryState(query_state);
      try {
        if (hasTableAccessPrivileges(td, *session_info)) {
          // TODO(adb): we should take schema read locks on all tables making up the
          // view, at minimum
          const auto query_ra = parse_to_ra(query_state->createQueryStateProxy(),
                                            query_state->getQueryStr(),
                                            {},
                                            false,
                                            system_parameters_,
                                            false);
          try {
            calcite_->checkAccessedObjectsPrivileges(query_state->createQueryStateProxy(),
                                                     query_ra.first);
          } catch (const std::runtime_error&) {
            have_privileges_on_view_sources = false;
          }

          const auto result = validate_rel_alg(query_ra.first.plan_result,
                                               query_state->createQueryStateProxy());

          _return.row_desc = fixup_row_descriptor(result.row_set.row_desc, cat);
        } else {
          throw std::runtime_error(
              "Unable to access view " + table_name +
              ". The view may not exist, or the logged in user may not "
              "have permission to access the view.");
        }
      } catch (const std::exception& e) {
        throw std::runtime_error("View '" + table_name +
                                 "' query has failed with an error: '" +
                                 std::string(e.what()) +
                                 "'.\nThe view must be dropped and re-created to "
                                 "resolve the error. \nQuery:\n" +
                                 query_state->getQueryStr());
      }
    } else {
      if (hasTableAccessPrivileges(td, *session_info)) {
        const auto col_descriptors =
            cat.getAllColumnMetadataForTable(td->tableId, get_system, true, get_physical);
        const auto deleted_cd = cat.getDeletedColumn(td);
        for (const auto cd : col_descriptors) {
          if (cd == deleted_cd) {
            continue;
          }
          _return.row_desc.push_back(populateThriftColumnType(&cat, cd));
        }
      } else {
        throw std::runtime_error(
            "Unable to access table " + table_name +
            ". The table may not exist, or the logged in user may not "
            "have permission to access the table.");
      }
    }
    _return.fragment_size = td->maxFragRows;
    _return.page_size = td->fragPageSize;
    _return.max_rows = td->maxRows;
    _return.view_sql =
        (have_privileges_on_view_sources ? td->viewSQL
                                         : "[Not enough privileges to see the view SQL]");
    _return.shard_count = td->nShards;
    _return.key_metainfo = td->keyMetainfo;
    _return.is_temporary = td->persistenceLevel == Data_Namespace::MemoryLevel::CPU_LEVEL;
    _return.partition_detail =
        td->partitions.empty()
            ? TPartitionDetail::DEFAULT
            : (table_is_replicated(td)
                   ? TPartitionDetail::REPLICATED
                   : (td->partitions == "SHARDED" ? TPartitionDetail::SHARDED
                                                  : TPartitionDetail::OTHER));
  } catch (const std::runtime_error& e) {
    THROW_MAPD_EXCEPTION("Exception: " + std::string(e.what()));
  }
}

void DBHandler::get_link_view(TFrontendView& _return,
                              const TSessionId& session,
                              const std::string& link) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  auto const& cat = session_ptr->getCatalog();
  auto ld = cat.getMetadataForLink(std::to_string(cat.getCurrentDB().dbId) + link);
  if (!ld) {
    THROW_MAPD_EXCEPTION("Link " + link + " is not valid.");
  }
  _return.view_state = ld->viewState;
  _return.view_name = ld->link;
  _return.update_time = ld->updateTime;
  _return.view_metadata = ld->viewMetadata;
}

bool DBHandler::hasTableAccessPrivileges(
    const TableDescriptor* td,
    const Catalog_Namespace::SessionInfo& session_info) {
  auto& cat = session_info.getCatalog();
  auto user_metadata = session_info.get_currentUser();

  if (user_metadata.isSuper) {
    return true;
  }

  DBObject dbObject(td->tableName, td->isView ? ViewDBObjectType : TableDBObjectType);
  dbObject.loadKey(cat);
  std::vector<DBObject> privObjects = {dbObject};

  return SysCatalog::instance().hasAnyPrivileges(user_metadata, privObjects);
}

void DBHandler::get_tables_impl(std::vector<std::string>& table_names,
                                const Catalog_Namespace::SessionInfo& session_info,
                                const GetTablesType get_tables_type,
                                const std::string& database_name) {
  if (database_name.empty()) {
    table_names = session_info.getCatalog().getTableNamesForUser(
        session_info.get_currentUser(), get_tables_type);
  } else {
    auto request_cat = SysCatalog::instance().getCatalog(database_name);
    table_names = request_cat->getTableNamesForUser(session_info.get_currentUser(),
                                                    get_tables_type);
  }
}

void DBHandler::get_tables(std::vector<std::string>& table_names,
                           const TSessionId& session) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  get_tables_impl(
      table_names, *stdlog.getConstSessionInfo(), GET_PHYSICAL_TABLES_AND_VIEWS);
}

void DBHandler::get_tables_for_database(std::vector<std::string>& table_names,
                                        const TSessionId& session,
                                        const std::string& database_name) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());

  get_tables_impl(table_names,
                  *stdlog.getConstSessionInfo(),
                  GET_PHYSICAL_TABLES_AND_VIEWS,
                  database_name);
}

void DBHandler::get_physical_tables(std::vector<std::string>& table_names,
                                    const TSessionId& session) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  get_tables_impl(table_names, *stdlog.getConstSessionInfo(), GET_PHYSICAL_TABLES);
}

void DBHandler::get_views(std::vector<std::string>& table_names,
                          const TSessionId& session) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  get_tables_impl(table_names, *stdlog.getConstSessionInfo(), GET_VIEWS);
}

void DBHandler::get_tables_meta_impl(std::vector<TTableMeta>& _return,
                                     QueryStateProxy query_state_proxy,
                                     const Catalog_Namespace::SessionInfo& session_info,
                                     const bool with_table_locks) {
  const auto& cat = session_info.getCatalog();
  const auto tables = cat.getAllTableMetadata();
  _return.reserve(tables.size());

  for (const auto td : tables) {
    if (td->shard >= 0) {
      // skip shards, they're not standalone tables
      continue;
    }
    if (!hasTableAccessPrivileges(td, session_info)) {
      // skip table, as there are no privileges to access it
      continue;
    }

    TTableMeta ret;
    ret.table_name = td->tableName;
    ret.is_view = td->isView;
    ret.is_replicated = table_is_replicated(td);
    ret.shard_count = td->nShards;
    ret.max_rows = td->maxRows;
    ret.table_id = td->tableId;

    std::vector<TTypeInfo> col_types;
    std::vector<std::string> col_names;
    size_t num_cols = 0;
    if (td->isView) {
      try {
        TPlanResult parse_result;
        lockmgr::LockedTableDescriptors locks;
        std::tie(parse_result, locks) = parse_to_ra(
            query_state_proxy, td->viewSQL, {}, with_table_locks, system_parameters_);
        const auto query_ra = parse_result.plan_result;

        ExecutionResult ex_result;
        execute_rel_alg(ex_result,
                        query_state_proxy,
                        query_ra,
                        true,
                        ExecutorDeviceType::CPU,
                        -1,
                        -1,
                        /*just_validate=*/true,
                        /*find_push_down_candidates=*/false,
                        ExplainInfo::defaults());
        TQueryResult result;
        DBHandler::convertData(
            result, ex_result, query_state_proxy, query_ra, true, -1, -1);
        num_cols = result.row_set.row_desc.size();
        for (const auto& col : result.row_set.row_desc) {
          if (col.is_physical) {
            num_cols--;
            continue;
          }
          col_types.push_back(col.col_type);
          col_names.push_back(col.col_name);
        }
      } catch (std::exception& e) {
        LOG(WARNING) << "get_tables_meta: Ignoring broken view: " << td->tableName;
      }
    } else {
      try {
        if (hasTableAccessPrivileges(td, session_info)) {
          const auto col_descriptors =
              cat.getAllColumnMetadataForTable(td->tableId, false, true, false);
          const auto deleted_cd = cat.getDeletedColumn(td);
          for (const auto cd : col_descriptors) {
            if (cd == deleted_cd) {
              continue;
            }
            col_types.push_back(ThriftSerializers::type_info_to_thrift(cd->columnType));
            col_names.push_back(cd->columnName);
          }
          num_cols = col_descriptors.size();
        } else {
          continue;
        }
      } catch (const std::runtime_error& e) {
        THROW_MAPD_EXCEPTION(e.what());
      }
    }

    ret.num_cols = num_cols;
    std::copy(col_types.begin(), col_types.end(), std::back_inserter(ret.col_types));
    std::copy(col_names.begin(), col_names.end(), std::back_inserter(ret.col_names));

    _return.push_back(ret);
  }
}

void DBHandler::get_tables_meta(std::vector<TTableMeta>& _return,
                                const TSessionId& session) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  auto query_state = create_query_state(session_ptr, "");
  stdlog.setQueryState(query_state);

  auto execute_read_lock = mapd_shared_lock<mapd_shared_mutex>(
      *legacylockmgr::LockMgr<mapd_shared_mutex, bool>::getMutex(
          legacylockmgr::ExecutorOuterLock, true));

  try {
    get_tables_meta_impl(_return, query_state->createQueryStateProxy(), *session_ptr);
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION(e.what());
  }
}

void DBHandler::get_users(std::vector<std::string>& user_names,
                          const TSessionId& session) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  std::list<Catalog_Namespace::UserMetadata> user_list;

  if (!session_ptr->get_currentUser().isSuper) {
    user_list = SysCatalog::instance().getAllUserMetadata(
        session_ptr->getCatalog().getCurrentDB().dbId);
  } else {
    user_list = SysCatalog::instance().getAllUserMetadata();
  }
  for (auto u : user_list) {
    user_names.push_back(u.userName);
  }
}

void DBHandler::get_version(std::string& version) {
  version = MAPD_RELEASE;
}

void DBHandler::clear_gpu_memory(const TSessionId& session) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  if (!session_ptr->get_currentUser().isSuper) {
    THROW_MAPD_EXCEPTION("Superuser privilege is required to run clear_gpu_memory");
  }
  try {
    Executor::clearMemory(Data_Namespace::MemoryLevel::GPU_LEVEL);
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION(e.what());
  }
  if (render_handler_) {
    render_handler_->clear_gpu_memory();
  }

  if (leaf_aggregator_.leafCount() > 0) {
    leaf_aggregator_.clear_leaf_gpu_memory(session);
  }
}

void DBHandler::clear_cpu_memory(const TSessionId& session) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  if (!session_ptr->get_currentUser().isSuper) {
    THROW_MAPD_EXCEPTION("Superuser privilege is required to run clear_cpu_memory");
  }
  try {
    Executor::clearMemory(Data_Namespace::MemoryLevel::CPU_LEVEL);
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION(e.what());
  }
  if (render_handler_) {
    render_handler_->clear_cpu_memory();
  }

  if (leaf_aggregator_.leafCount() > 0) {
    leaf_aggregator_.clear_leaf_cpu_memory(session);
  }
}

void DBHandler::set_cur_session(const TSessionId& parent_session,
                                const TSessionId& leaf_session,
                                const std::string& start_time_str,
                                const std::string& label) {
  // internal API to manage query interruption in distributed mode
  auto stdlog = STDLOG(get_session_ptr(leaf_session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();

  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID).get();
  executor->enrollQuerySession(parent_session,
                               label,
                               start_time_str,
                               Executor::UNITARY_EXECUTOR_ID,
                               QuerySessionStatus::QueryStatus::RUNNING);
  if (leaf_aggregator_.leafCount() > 0) {
    leaf_aggregator_.set_cur_session(parent_session, start_time_str, label);
  }
}

void DBHandler::invalidate_cur_session(const TSessionId& parent_session,
                                       const TSessionId& leaf_session,
                                       const std::string& start_time_str,
                                       const std::string& label) {
  // internal API to manage query interruption in distributed mode
  auto stdlog = STDLOG(get_session_ptr(leaf_session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID).get();
  executor->clearQuerySessionStatus(parent_session, start_time_str, false);
  if (leaf_aggregator_.leafCount() > 0) {
    leaf_aggregator_.invalidate_cur_session(parent_session, start_time_str, label);
  }
}

TSessionId DBHandler::getInvalidSessionId() const {
  return INVALID_SESSION_ID;
}

void DBHandler::get_memory(std::vector<TNodeMemoryInfo>& _return,
                           const TSessionId& session,
                           const std::string& memory_level) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  std::vector<Data_Namespace::MemoryInfo> internal_memory;
  Data_Namespace::MemoryLevel mem_level;
  if (!memory_level.compare("gpu")) {
    mem_level = Data_Namespace::MemoryLevel::GPU_LEVEL;
    internal_memory =
        SysCatalog::instance().getDataMgr().getMemoryInfo(MemoryLevel::GPU_LEVEL);
  } else {
    mem_level = Data_Namespace::MemoryLevel::CPU_LEVEL;
    internal_memory =
        SysCatalog::instance().getDataMgr().getMemoryInfo(MemoryLevel::CPU_LEVEL);
  }

  for (auto memInfo : internal_memory) {
    TNodeMemoryInfo nodeInfo;
    if (leaf_aggregator_.leafCount() > 0) {
      nodeInfo.host_name = omnisci::get_hostname();
    }
    nodeInfo.page_size = memInfo.pageSize;
    nodeInfo.max_num_pages = memInfo.maxNumPages;
    nodeInfo.num_pages_allocated = memInfo.numPageAllocated;
    nodeInfo.is_allocation_capped = memInfo.isAllocationCapped;
    for (auto gpu : memInfo.nodeMemoryData) {
      TMemoryData md;
      md.slab = gpu.slabNum;
      md.start_page = gpu.startPage;
      md.num_pages = gpu.numPages;
      md.touch = gpu.touch;
      md.chunk_key.insert(md.chunk_key.end(), gpu.chunk_key.begin(), gpu.chunk_key.end());
      md.is_free = gpu.memStatus == Buffer_Namespace::MemStatus::FREE;
      nodeInfo.node_memory_data.push_back(md);
    }
    _return.push_back(nodeInfo);
  }
  if (leaf_aggregator_.leafCount() > 0) {
    std::vector<TNodeMemoryInfo> leafSummary =
        leaf_aggregator_.getLeafMemoryInfo(session, mem_level);
    _return.insert(_return.begin(), leafSummary.begin(), leafSummary.end());
  }
}

void DBHandler::get_databases(std::vector<TDBInfo>& dbinfos, const TSessionId& session) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  const auto& user = session_ptr->get_currentUser();
  Catalog_Namespace::DBSummaryList dbs =
      SysCatalog::instance().getDatabaseListForUser(user);
  for (auto& db : dbs) {
    TDBInfo dbinfo;
    dbinfo.db_name = std::move(db.dbName);
    dbinfo.db_owner = std::move(db.dbOwnerName);
    dbinfos.push_back(std::move(dbinfo));
  }
}

void DBHandler::set_execution_mode(const TSessionId& session,
                                   const TExecuteMode::type mode) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  mapd_unique_lock<mapd_shared_mutex> write_lock(sessions_mutex_);
  auto session_it = get_session_it_unsafe(session, write_lock);
  if (leaf_aggregator_.leafCount() > 0) {
    leaf_aggregator_.set_execution_mode(session, mode);
    try {
      DBHandler::set_execution_mode_nolock(session_it->second.get(), mode);
    } catch (const TOmniSciException& e) {
      LOG(INFO) << "Aggregator failed to set execution mode: " << e.error_msg;
    }
    return;
  }
  DBHandler::set_execution_mode_nolock(session_it->second.get(), mode);
}

namespace {

void check_table_not_sharded(const TableDescriptor* td) {
  if (td && td->nShards) {
    throw std::runtime_error("Cannot import a sharded table directly to a leaf");
  }
}

void check_valid_column_names(const std::list<const ColumnDescriptor*>& descs,
                              const std::vector<std::string>& column_names) {
  std::unordered_set<std::string> unique_names;
  for (const auto& name : column_names) {
    auto lower_name = to_lower(name);
    if (unique_names.find(lower_name) != unique_names.end()) {
      THROW_MAPD_EXCEPTION("Column " + name + " is mentioned multiple times");
    } else {
      unique_names.insert(lower_name);
    }
  }
  for (const auto& cd : descs) {
    auto iter = unique_names.find(to_lower(cd->columnName));
    if (iter != unique_names.end()) {
      unique_names.erase(iter);
    }
  }
  if (!unique_names.empty()) {
    THROW_MAPD_EXCEPTION("Column " + *unique_names.begin() + " does not exist");
  }
}

// Return vector of IDs mapping column descriptors to the list of comumn names.
// The size of the vector is the number of actual columns (geophisical columns excluded).
// ID is either a position in column_names matching the descriptor, or -1 if the column
// is missing from the column_names
std::vector<int> column_ids_by_names(const std::list<const ColumnDescriptor*>& descs,
                                     const std::vector<std::string>& column_names) {
  std::vector<int> desc_to_column_ids;
  if (column_names.empty()) {
    int col_idx = 0;
    for (const auto& cd : descs) {
      if (!cd->isGeoPhyCol) {
        desc_to_column_ids.push_back(col_idx);
        ++col_idx;
      }
    }
  } else {
    for (const auto& cd : descs) {
      if (!cd->isGeoPhyCol) {
        bool found = false;
        for (size_t j = 0; j < column_names.size(); ++j) {
          if (to_lower(cd->columnName) == to_lower(column_names[j])) {
            found = true;
            desc_to_column_ids.push_back(j);
            break;
          }
        }
        if (!found) {
          if (!cd->columnType.get_notnull()) {
            desc_to_column_ids.push_back(-1);
          } else {
            THROW_MAPD_EXCEPTION("Column '" + cd->columnName +
                                 "' cannot be omitted due to NOT NULL constraint");
          }
        }
      }
    }
  }
  return desc_to_column_ids;
}

}  // namespace

void DBHandler::load_table_binary(const TSessionId& session,
                                  const std::string& table_name,
                                  const std::vector<TRow>& rows,
                                  const std::vector<std::string>& column_names) {
  try {
    auto stdlog = STDLOG(get_session_ptr(session), "table_name", table_name);
    stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
    auto session_ptr = stdlog.getConstSessionInfo();

    if (rows.empty()) {
      THROW_MAPD_EXCEPTION("No rows to insert");
    }

    std::unique_ptr<import_export::Loader> loader;
    std::vector<std::unique_ptr<import_export::TypedImportBuffer>> import_buffers;
    auto schema_read_lock = prepare_loader_generic(*session_ptr,
                                                   table_name,
                                                   rows.front().cols.size(),
                                                   &loader,
                                                   &import_buffers,
                                                   column_names,
                                                   "load_table_binary");

    auto col_descs = loader->get_column_descs();
    auto desc_id_to_column_id = column_ids_by_names(col_descs, column_names);

    TDatum empty_value;
    empty_value.is_null = true;
    size_t rows_completed = 0;
    for (auto const& row : rows) {
      size_t col_idx = 0;
      try {
        for (auto cd : col_descs) {
          int mapped_idx = desc_id_to_column_id[col_idx];
          if (mapped_idx != -1) {
            import_buffers[col_idx]->add_value(
                cd, row.cols[mapped_idx], row.cols[mapped_idx].is_null);
          } else {
            import_buffers[col_idx]->add_value(cd, empty_value, true);
          }
          col_idx++;
        }
        rows_completed++;
      } catch (const std::exception& e) {
        for (size_t col_idx_to_pop = 0; col_idx_to_pop < col_idx; ++col_idx_to_pop) {
          import_buffers[col_idx_to_pop]->pop_value();
        }
        LOG(ERROR) << "Input exception thrown: " << e.what()
                   << ". Row discarded, issue at column : " << (col_idx + 1)
                   << " data :" << row;
      }
    }
    auto insert_data_lock = lockmgr::InsertDataLockMgr::getWriteLockForTable(
        session_ptr->getCatalog(), table_name);
    if (!loader->load(import_buffers, rows.size(), session_ptr.get())) {
      THROW_MAPD_EXCEPTION(loader->getErrorMessage());
    }
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION("Exception: " + std::string(e.what()));
  }
}

std::unique_ptr<lockmgr::AbstractLockContainer<const TableDescriptor*>>
DBHandler::prepare_loader_generic(
    const Catalog_Namespace::SessionInfo& session_info,
    const std::string& table_name,
    size_t num_cols,
    std::unique_ptr<import_export::Loader>* loader,
    std::vector<std::unique_ptr<import_export::TypedImportBuffer>>* import_buffers,
    const std::vector<std::string>& column_names,
    std::string load_type) {
  if (num_cols == 0) {
    THROW_MAPD_EXCEPTION("No columns to insert");
  }
  check_read_only(load_type);
  auto& cat = session_info.getCatalog();
  auto td_with_lock =
      std::make_unique<lockmgr::TableSchemaLockContainer<lockmgr::ReadLock>>(
          lockmgr::TableSchemaLockContainer<lockmgr::ReadLock>::acquireTableDescriptor(
              cat, table_name, true));
  const auto td = (*td_with_lock)();
  CHECK(td);

  if (g_cluster && !leaf_aggregator_.leafCount()) {
    // Sharded table rows need to be routed to the leaf by an aggregator.
    check_table_not_sharded(td);
  }
  check_table_load_privileges(session_info, table_name);

  if (leaf_aggregator_.leafCount() > 0) {
    loader->reset(new DistributedLoader(session_info, td, &leaf_aggregator_));
  } else {
    loader->reset(new import_export::Loader(cat, td));
  }

  auto col_descs = (*loader)->get_column_descs();
  check_valid_column_names(col_descs, column_names);
  if (column_names.empty()) {
    // TODO(andrew): nColumns should be number of non-virtual/non-system columns.
    //               Subtracting 1 (rowid) until TableDescriptor is updated.
    auto geo_physical_cols = std::count_if(
        col_descs.begin(), col_descs.end(), [](auto cd) { return cd->isGeoPhyCol; });
    const auto num_table_cols = static_cast<size_t>(td->nColumns) - geo_physical_cols -
                                (td->hasDeletedCol ? 2 : 1);
    if (num_cols != num_table_cols) {
      throw std::runtime_error("Number of columns to load (" + std::to_string(num_cols) +
                               ") does not match number of columns in table " +
                               td->tableName + " (" + std::to_string(num_table_cols) +
                               ")");
    }
  } else if (num_cols != column_names.size()) {
    THROW_MAPD_EXCEPTION(
        "Number of columns specified does not match the "
        "number of columns given (" +
        std::to_string(num_cols) + " vs " + std::to_string(column_names.size()) + ")");
  }

  *import_buffers = import_export::setup_column_loaders(td, loader->get());
  return std::move(td_with_lock);
}
namespace {

size_t get_column_size(const TColumn& column) {
  if (!column.nulls.empty()) {
    return column.nulls.size();
  } else {
    // it is a very bold estimate but later we check it against REAL data
    // and if this function returns a wrong result (e.g. both int and string
    // vectors are filled with values), we get an error
    return column.data.int_col.size() + column.data.arr_col.size() +
           column.data.real_col.size() + column.data.str_col.size();
  }
}

}  // namespace

void DBHandler::load_table_binary_columnar(const TSessionId& session,
                                           const std::string& table_name,
                                           const std::vector<TColumn>& cols,
                                           const std::vector<std::string>& column_names) {
  load_table_binary_columnar_internal(
      session, table_name, cols, column_names, AssignRenderGroupsMode::kNone);
}

void DBHandler::load_table_binary_columnar_polys(
    const TSessionId& session,
    const std::string& table_name,
    const std::vector<TColumn>& cols,
    const std::vector<std::string>& column_names,
    const bool assign_render_groups) {
  load_table_binary_columnar_internal(session,
                                      table_name,
                                      cols,
                                      column_names,
                                      assign_render_groups
                                          ? AssignRenderGroupsMode::kAssign
                                          : AssignRenderGroupsMode::kCleanUp);
}

void DBHandler::load_table_binary_columnar_internal(
    const TSessionId& session,
    const std::string& table_name,
    const std::vector<TColumn>& cols,
    const std::vector<std::string>& column_names,
    const AssignRenderGroupsMode assign_render_groups_mode) {
  auto stdlog = STDLOG(get_session_ptr(session), "table_name", table_name);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();

  if (assign_render_groups_mode == AssignRenderGroupsMode::kCleanUp) {
    // throw if the user tries to pass column data on a clean-up
    if (cols.size()) {
      THROW_MAPD_EXCEPTION(
          "load_table_binary_columnar_polys: Column data must be empty when called with "
          "assign_render_groups = false");
    }

    // mutex the map access
    std::lock_guard<std::mutex> lock(render_group_assignment_mutex_);

    // drop persistent render group assignment data for this session and table
    // keep the per-session map in case other tables are active (ideally not)
    auto itr_session = render_group_assignment_map_.find(session);
    if (itr_session != render_group_assignment_map_.end()) {
      LOG(INFO) << "load_table_binary_columnar_polys: Cleaning up Render Group "
                   "Assignment Persistent Data for Session '"
                << session << "', Table '" << table_name << "'";
      itr_session->second.erase(table_name);
    }

    // just doing clean-up, so we're done
    return;
  }

  std::unique_ptr<import_export::Loader> loader;
  std::vector<std::unique_ptr<import_export::TypedImportBuffer>> import_buffers;
  auto schema_read_lock = prepare_loader_generic(*session_ptr,
                                                 table_name,
                                                 cols.size(),
                                                 &loader,
                                                 &import_buffers,
                                                 column_names,
                                                 "load_table_binary_columnar");

  auto desc_id_to_column_id =
      column_ids_by_names(loader->get_column_descs(), column_names);
  size_t num_rows = get_column_size(cols.front());
  size_t import_idx = 0;  // index into the TColumn vector being loaded
  size_t col_idx = 0;     // index into column description vector
  TColumn empty_column;
  empty_column.nulls.resize(num_rows, true);
  empty_column.data.arr_col.resize(num_rows);
  empty_column.data.int_col.resize(num_rows);
  empty_column.data.str_col.resize(num_rows);
  empty_column.data.real_col.resize(num_rows);
  try {
    size_t skip_physical_cols = 0;
    for (auto cd : loader->get_column_descs()) {
      if (skip_physical_cols > 0) {
        if (!cd->isGeoPhyCol) {
          throw std::runtime_error("Unexpected physical column");
        }
        skip_physical_cols--;
        continue;
      }
      int mapped_idx = desc_id_to_column_id[import_idx];
      const TColumn& value = mapped_idx == -1 ? empty_column : cols[mapped_idx];
      size_t col_rows = import_buffers[col_idx]->add_values(cd, value);
      if (col_rows != num_rows) {
        std::ostringstream oss;
        oss << "load_table_binary_columnar: Inconsistent number of rows in column "
            << cd->columnName << " ,  expecting " << num_rows << " rows, column "
            << col_idx << " has " << col_rows << " rows";
        THROW_MAPD_EXCEPTION(oss.str());
      }
      // Advance to the next column in the table
      col_idx++;

      // For geometry columns: process WKT strings and fill physical columns
      if (cd->columnType.is_geometry()) {
        auto geo_col_idx = col_idx - 1;
        const auto wkt_or_wkb_hex_column =
            import_buffers[geo_col_idx]->getGeoStringBuffer();
        std::vector<std::vector<double>> coords_column, bounds_column;
        std::vector<std::vector<int>> ring_sizes_column, poly_rings_column;
        std::vector<int> render_groups_column;
        SQLTypeInfo ti = cd->columnType;
        if (num_rows != wkt_or_wkb_hex_column->size() ||
            !Geospatial::GeoTypesFactory::getGeoColumns(wkt_or_wkb_hex_column,
                                                        ti,
                                                        coords_column,
                                                        bounds_column,
                                                        ring_sizes_column,
                                                        poly_rings_column,
                                                        false)) {
          std::ostringstream oss;
          oss << "load_table_binary_columnar: Invalid geometry in column "
              << cd->columnName;
          THROW_MAPD_EXCEPTION(oss.str());
        }

        // start or continue assigning render groups for poly columns?
        if (IS_GEO_POLY(cd->columnType.get_type()) &&
            assign_render_groups_mode == AssignRenderGroupsMode::kAssign) {
          // get RGA to use
          import_export::RenderGroupAnalyzer* render_group_analyzer{};
          {
            // mutex the map access
            std::lock_guard<std::mutex> lock(render_group_assignment_mutex_);

            // emplace new RGA or fetch existing RGA from map
            auto [itr_table, emplaced_table] = render_group_assignment_map_.try_emplace(
                session, RenderGroupAssignmentTableMap());
            LOG_IF(INFO, emplaced_table)
                << "load_table_binary_columnar_polys: Creating Render Group Assignment "
                   "Persistent Data for Session '"
                << session << "'";
            auto [itr_column, emplaced_column] = itr_table->second.try_emplace(
                table_name, RenderGroupAssignmentColumnMap());
            LOG_IF(INFO, emplaced_column)
                << "load_table_binary_columnar_polys: Creating Render Group Assignment "
                   "Persistent Data for Table '"
                << table_name << "'";
            auto [itr_analyzer, emplaced_analyzer] = itr_column->second.try_emplace(
                cd->columnName, std::make_unique<import_export::RenderGroupAnalyzer>());
            LOG_IF(INFO, emplaced_analyzer)
                << "load_table_binary_columnar_polys: Creating Render Group Assignment "
                   "Persistent Data for Column '"
                << cd->columnName << "'";
            render_group_analyzer = itr_analyzer->second.get();
            CHECK(render_group_analyzer);

            // seed new RGA from existing table/column, to handle appends
            if (emplaced_analyzer) {
              LOG(INFO) << "load_table_binary_columnar_polys: Seeding Render Groups from "
                           "existing table...";
              render_group_analyzer->seedFromExistingTableContents(
                  session_ptr->getCatalog(), table_name, cd->columnName);
              LOG(INFO) << "load_table_binary_columnar_polys: Done";
            }
          }

          // assign render groups for this set of bounds
          LOG(INFO) << "load_table_binary_columnar_polys: Assigning Render Groups...";
          render_groups_column.reserve(bounds_column.size());
          for (auto const& bounds : bounds_column) {
            CHECK_EQ(bounds.size(), 4u);
            int rg = render_group_analyzer->insertBoundsAndReturnRenderGroup(bounds);
            render_groups_column.push_back(rg);
          }
          LOG(INFO) << "load_table_binary_columnar_polys: Done";
        } else {
          // render groups all zero
          render_groups_column.resize(bounds_column.size(), 0);
        }

        // Populate physical columns, advance col_idx
        import_export::Importer::set_geo_physical_import_buffer_columnar(
            session_ptr->getCatalog(),
            cd,
            import_buffers,
            col_idx,
            coords_column,
            bounds_column,
            ring_sizes_column,
            poly_rings_column,
            render_groups_column);
        skip_physical_cols = cd->columnType.get_physical_cols();
      }
      // Advance to the next column of values being loaded
      import_idx++;
    }
  } catch (const std::exception& e) {
    std::ostringstream oss;
    oss << "load_table_binary_columnar: Input exception thrown: " << e.what()
        << ". Issue at column : " << (col_idx + 1) << ". Import aborted";
    THROW_MAPD_EXCEPTION(oss.str());
  }
  auto insert_data_lock = lockmgr::InsertDataLockMgr::getWriteLockForTable(
      session_ptr->getCatalog(), table_name);
  if (!loader->load(import_buffers, num_rows, session_ptr.get())) {
    THROW_MAPD_EXCEPTION(loader->getErrorMessage());
  }
}

using RecordBatchVector = std::vector<std::shared_ptr<arrow::RecordBatch>>;

#define ARROW_THRIFT_THROW_NOT_OK(s) \
  do {                               \
    ::arrow::Status _s = (s);        \
    if (UNLIKELY(!_s.ok())) {        \
      TOmniSciException ex;          \
      ex.error_msg = _s.ToString();  \
      LOG(ERROR) << s.ToString();    \
      throw ex;                      \
    }                                \
  } while (0)

namespace {

RecordBatchVector loadArrowStream(const std::string& stream) {
  RecordBatchVector batches;
  try {
    // TODO(wesm): Make this simpler in general, see ARROW-1600
    auto stream_buffer =
        std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(stream.c_str()),
                                        static_cast<int64_t>(stream.size()));

    arrow::io::BufferReader buf_reader(stream_buffer);
    std::shared_ptr<arrow::RecordBatchReader> batch_reader;
    ARROW_ASSIGN_OR_THROW(batch_reader,
                          arrow::ipc::RecordBatchStreamReader::Open(&buf_reader));

    while (true) {
      std::shared_ptr<arrow::RecordBatch> batch;
      // Read batch (zero-copy) from the stream
      ARROW_THRIFT_THROW_NOT_OK(batch_reader->ReadNext(&batch));
      if (batch == nullptr) {
        break;
      }
      batches.emplace_back(std::move(batch));
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "Error parsing Arrow stream: " << e.what() << ". Import aborted";
  }
  return batches;
}

}  // namespace

void DBHandler::load_table_binary_arrow(const TSessionId& session,
                                        const std::string& table_name,
                                        const std::string& arrow_stream,
                                        const bool use_column_names) {
  auto stdlog = STDLOG(get_session_ptr(session), "table_name", table_name);
  auto session_ptr = stdlog.getConstSessionInfo();

  RecordBatchVector batches = loadArrowStream(arrow_stream);
  // Assuming have one batch for now
  if (batches.size() != 1) {
    THROW_MAPD_EXCEPTION("Expected a single Arrow record batch. Import aborted");
  }
  std::shared_ptr<arrow::RecordBatch> batch = batches[0];
  std::unique_ptr<import_export::Loader> loader;
  std::vector<std::unique_ptr<import_export::TypedImportBuffer>> import_buffers;
  std::vector<std::string> column_names;
  if (use_column_names) {
    column_names = batch->schema()->field_names();
  }
  auto schema_read_lock =
      prepare_loader_generic(*session_ptr,
                             table_name,
                             static_cast<size_t>(batch->num_columns()),
                             &loader,
                             &import_buffers,
                             column_names,
                             "load_table_binary_arrow");

  auto desc_id_to_column_id =
      column_ids_by_names(loader->get_column_descs(), column_names);
  size_t num_rows = 0;
  size_t col_idx = 0;
  std::shared_ptr<arrow::Array> empty_array;
  {
    arrow::BooleanBuilder builder;
    ARROW_THROW_NOT_OK(builder.Resize(batch->num_rows()));
    ARROW_THROW_NOT_OK(builder.AppendNulls(batch->num_rows()));
    auto status = builder.Finish(&empty_array);
    if (!status.ok()) {
      THROW_MAPD_EXCEPTION("Failed to load data: " + status.message());
    }
  }

  try {
    for (auto cd : loader->get_column_descs()) {
      int mapped_idx = desc_id_to_column_id[col_idx];
      if (mapped_idx != -1) {
        auto& array = *batch->column(mapped_idx);
        import_export::ArraySliceRange row_slice(0, array.length());
        num_rows = import_buffers[col_idx]->add_arrow_values(
            cd, array, true, row_slice, nullptr);
      } else {
        import_export::ArraySliceRange row_slice(0, empty_array->length());
        num_rows = import_buffers[col_idx]->add_arrow_values(
            cd, *empty_array, false, row_slice, nullptr);
      }
      col_idx++;
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "Input exception thrown: " << e.what()
               << ". Issue at column : " << (col_idx + 1) << ". Import aborted";
    // TODO(tmostak): Go row-wise on binary columnar import to be consistent with our
    // other import paths
    THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
  }
  auto insert_data_lock = lockmgr::InsertDataLockMgr::getWriteLockForTable(
      session_ptr->getCatalog(), table_name);
  if (!loader->load(import_buffers, num_rows, session_ptr.get())) {
    THROW_MAPD_EXCEPTION(loader->getErrorMessage());
  }
}

void DBHandler::load_table(const TSessionId& session,
                           const std::string& table_name,
                           const std::vector<TStringRow>& rows,
                           const std::vector<std::string>& column_names) {
  try {
    auto stdlog = STDLOG(get_session_ptr(session), "table_name", table_name);
    stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
    auto session_ptr = stdlog.getConstSessionInfo();

    if (rows.empty()) {
      THROW_MAPD_EXCEPTION("No rows to insert");
    }

    std::unique_ptr<import_export::Loader> loader;
    std::vector<std::unique_ptr<import_export::TypedImportBuffer>> import_buffers;
    auto schema_read_lock =
        prepare_loader_generic(*session_ptr,
                               table_name,
                               static_cast<size_t>(rows.front().cols.size()),
                               &loader,
                               &import_buffers,
                               column_names,
                               "load_table");

    auto col_descs = loader->get_column_descs();
    auto desc_id_to_column_id = column_ids_by_names(col_descs, column_names);
    import_export::CopyParams copy_params;
    size_t rows_completed = 0;
    for (auto const& row : rows) {
      size_t import_idx = 0;  // index into the TStringRow being loaded
      size_t col_idx = 0;     // index into column description vector
      try {
        size_t skip_physical_cols = 0;
        for (auto cd : col_descs) {
          if (skip_physical_cols > 0) {
            if (!cd->isGeoPhyCol) {
              throw std::runtime_error("Unexpected physical column");
            }
            skip_physical_cols--;
            continue;
          }
          const std::string empty_val = "";
          int mapped_idx = desc_id_to_column_id[import_idx];
          const std::string& value =
              mapped_idx == -1 ? empty_val : row.cols[mapped_idx].str_val;
          bool is_null = mapped_idx == -1 || row.cols[mapped_idx].is_null;
          import_buffers[col_idx]->add_value(cd, value, is_null, copy_params);
          // Advance to the next column within the table
          col_idx++;

          if (cd->columnType.is_geometry()) {
            // Populate physical columns
            std::vector<double> coords, bounds;
            std::vector<int> ring_sizes, poly_rings;
            int render_group = 0;
            SQLTypeInfo ti{cd->columnType};
            if (!Geospatial::GeoTypesFactory::getGeoColumns(
                    !is_null ? value : std::string(),
                    ti,
                    coords,
                    bounds,
                    ring_sizes,
                    poly_rings,
                    false)) {
              throw std::runtime_error("Invalid geometry");
            }
            if (cd->columnType.get_type() != ti.get_type()) {
              throw std::runtime_error("Geometry type mismatch");
            }
            import_export::Importer::set_geo_physical_import_buffer(
                session_ptr->getCatalog(),
                cd,
                import_buffers,
                col_idx,
                coords,
                bounds,
                ring_sizes,
                poly_rings,
                render_group);
            skip_physical_cols = cd->columnType.get_physical_cols();
          }
          // Advance to the next field within the row
          import_idx++;
        }
        rows_completed++;
      } catch (const std::exception& e) {
        for (size_t col_idx_to_pop = 0; col_idx_to_pop < col_idx; ++col_idx_to_pop) {
          import_buffers[col_idx_to_pop]->pop_value();
        }
        LOG(ERROR) << "Input exception thrown: " << e.what()
                   << ". Row discarded, issue at column : " << (col_idx + 1)
                   << " data :" << row;
        THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
      }
    }
    auto insert_data_lock = lockmgr::InsertDataLockMgr::getWriteLockForTable(
        session_ptr->getCatalog(), table_name);
    if (!loader->load(import_buffers, rows_completed, session_ptr.get())) {
      THROW_MAPD_EXCEPTION(loader->getErrorMessage());
    }

  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION("Exception: " + std::string(e.what()));
  }
}

char DBHandler::unescape_char(std::string str) {
  char out = str[0];
  if (str.size() == 2 && str[0] == '\\') {
    if (str[1] == 't') {
      out = '\t';
    } else if (str[1] == 'n') {
      out = '\n';
    } else if (str[1] == '0') {
      out = '\0';
    } else if (str[1] == '\'') {
      out = '\'';
    } else if (str[1] == '\\') {
      out = '\\';
    }
  }
  return out;
}

import_export::CopyParams DBHandler::thrift_to_copyparams(const TCopyParams& cp) {
  import_export::CopyParams copy_params;
  switch (cp.has_header) {
    case TImportHeaderRow::AUTODETECT:
      copy_params.has_header = import_export::ImportHeaderRow::AUTODETECT;
      break;
    case TImportHeaderRow::NO_HEADER:
      copy_params.has_header = import_export::ImportHeaderRow::NO_HEADER;
      break;
    case TImportHeaderRow::HAS_HEADER:
      copy_params.has_header = import_export::ImportHeaderRow::HAS_HEADER;
      break;
    default:
      THROW_MAPD_EXCEPTION("Invalid has_header in TCopyParams: " +
                           std::to_string((int)cp.has_header));
      break;
  }
  copy_params.quoted = cp.quoted;
  if (cp.delimiter.length() > 0) {
    copy_params.delimiter = unescape_char(cp.delimiter);
  } else {
    copy_params.delimiter = '\0';
  }
  if (cp.null_str.length() > 0) {
    copy_params.null_str = cp.null_str;
  }
  if (cp.quote.length() > 0) {
    copy_params.quote = unescape_char(cp.quote);
  }
  if (cp.escape.length() > 0) {
    copy_params.escape = unescape_char(cp.escape);
  }
  if (cp.line_delim.length() > 0) {
    copy_params.line_delim = unescape_char(cp.line_delim);
  }
  if (cp.array_delim.length() > 0) {
    copy_params.array_delim = unescape_char(cp.array_delim);
  }
  if (cp.array_begin.length() > 0) {
    copy_params.array_begin = unescape_char(cp.array_begin);
  }
  if (cp.array_end.length() > 0) {
    copy_params.array_end = unescape_char(cp.array_end);
  }
  if (cp.threads != 0) {
    copy_params.threads = cp.threads;
  }
  if (cp.s3_access_key.length() > 0) {
    copy_params.s3_access_key = cp.s3_access_key;
  }
  if (cp.s3_secret_key.length() > 0) {
    copy_params.s3_secret_key = cp.s3_secret_key;
  }
  if (cp.s3_session_token.length() > 0) {
    copy_params.s3_session_token = cp.s3_session_token;
  }
  if (cp.s3_region.length() > 0) {
    copy_params.s3_region = cp.s3_region;
  }
  if (cp.s3_endpoint.length() > 0) {
    copy_params.s3_endpoint = cp.s3_endpoint;
  }
#ifdef HAVE_AWS_S3
  if (g_allow_s3_server_privileges && cp.s3_access_key.length() == 0 &&
      cp.s3_secret_key.length() == 0 && cp.s3_session_token.length() == 0) {
    const auto& server_credentials =
        Aws::Auth::DefaultAWSCredentialsProviderChain().GetAWSCredentials();
    copy_params.s3_access_key = server_credentials.GetAWSAccessKeyId();
    copy_params.s3_secret_key = server_credentials.GetAWSSecretKey();
    copy_params.s3_session_token = server_credentials.GetSessionToken();
  }
#endif
  switch (cp.file_type) {
    case TFileType::POLYGON:
      copy_params.file_type = import_export::FileType::POLYGON;
      break;
    case TFileType::DELIMITED:
      copy_params.file_type = import_export::FileType::DELIMITED;
      break;
#ifdef ENABLE_IMPORT_PARQUET
    case TFileType::PARQUET:
      copy_params.file_type = import_export::FileType::PARQUET;
      break;
#endif
    default:
      THROW_MAPD_EXCEPTION("Invalid file_type in TCopyParams: " +
                           std::to_string((int)cp.file_type));
      break;
  }
  switch (cp.geo_coords_encoding) {
    case TEncodingType::GEOINT:
      copy_params.geo_coords_encoding = kENCODING_GEOINT;
      break;
    case TEncodingType::NONE:
      copy_params.geo_coords_encoding = kENCODING_NONE;
      break;
    default:
      THROW_MAPD_EXCEPTION("Invalid geo_coords_encoding in TCopyParams: " +
                           std::to_string((int)cp.geo_coords_encoding));
      break;
  }
  copy_params.geo_coords_comp_param = cp.geo_coords_comp_param;
  switch (cp.geo_coords_type) {
    case TDatumType::GEOGRAPHY:
      copy_params.geo_coords_type = kGEOGRAPHY;
      break;
    case TDatumType::GEOMETRY:
      copy_params.geo_coords_type = kGEOMETRY;
      break;
    default:
      THROW_MAPD_EXCEPTION("Invalid geo_coords_type in TCopyParams: " +
                           std::to_string((int)cp.geo_coords_type));
      break;
  }
  switch (cp.geo_coords_srid) {
    case 4326:
    case 3857:
    case 900913:
      copy_params.geo_coords_srid = cp.geo_coords_srid;
      break;
    default:
      THROW_MAPD_EXCEPTION("Invalid geo_coords_srid in TCopyParams (" +
                           std::to_string((int)cp.geo_coords_srid));
      break;
  }
  copy_params.sanitize_column_names = cp.sanitize_column_names;
  copy_params.geo_layer_name = cp.geo_layer_name;
  copy_params.geo_assign_render_groups = cp.geo_assign_render_groups;
  copy_params.geo_explode_collections = cp.geo_explode_collections;
  copy_params.source_srid = cp.source_srid;
  return copy_params;
}

TCopyParams DBHandler::copyparams_to_thrift(const import_export::CopyParams& cp) {
  TCopyParams copy_params;
  copy_params.delimiter = cp.delimiter;
  copy_params.null_str = cp.null_str;
  switch (cp.has_header) {
    case import_export::ImportHeaderRow::AUTODETECT:
      copy_params.has_header = TImportHeaderRow::AUTODETECT;
      break;
    case import_export::ImportHeaderRow::NO_HEADER:
      copy_params.has_header = TImportHeaderRow::NO_HEADER;
      break;
    case import_export::ImportHeaderRow::HAS_HEADER:
      copy_params.has_header = TImportHeaderRow::HAS_HEADER;
      break;
    default:
      CHECK(false);
      break;
  }
  copy_params.quoted = cp.quoted;
  copy_params.quote = cp.quote;
  copy_params.escape = cp.escape;
  copy_params.line_delim = cp.line_delim;
  copy_params.array_delim = cp.array_delim;
  copy_params.array_begin = cp.array_begin;
  copy_params.array_end = cp.array_end;
  copy_params.threads = cp.threads;
  copy_params.s3_access_key = cp.s3_access_key;
  copy_params.s3_secret_key = cp.s3_secret_key;
  copy_params.s3_session_token = cp.s3_session_token;
  copy_params.s3_region = cp.s3_region;
  copy_params.s3_endpoint = cp.s3_endpoint;
  switch (cp.file_type) {
    case import_export::FileType::POLYGON:
      copy_params.file_type = TFileType::POLYGON;
      break;
    default:
      copy_params.file_type = TFileType::DELIMITED;
      break;
  }
  switch (cp.geo_coords_encoding) {
    case kENCODING_GEOINT:
      copy_params.geo_coords_encoding = TEncodingType::GEOINT;
      break;
    default:
      copy_params.geo_coords_encoding = TEncodingType::NONE;
      break;
  }
  copy_params.geo_coords_comp_param = cp.geo_coords_comp_param;
  switch (cp.geo_coords_type) {
    case kGEOGRAPHY:
      copy_params.geo_coords_type = TDatumType::GEOGRAPHY;
      break;
    case kGEOMETRY:
      copy_params.geo_coords_type = TDatumType::GEOMETRY;
      break;
    default:
      CHECK(false);
      break;
  }
  copy_params.geo_coords_srid = cp.geo_coords_srid;
  copy_params.sanitize_column_names = cp.sanitize_column_names;
  copy_params.geo_layer_name = cp.geo_layer_name;
  copy_params.geo_assign_render_groups = cp.geo_assign_render_groups;
  copy_params.geo_explode_collections = cp.geo_explode_collections;
  copy_params.source_srid = cp.source_srid;
  return copy_params;
}

namespace {
void add_vsi_network_prefix(std::string& path) {
  // do we support network file access?
  bool gdal_network = Geospatial::GDAL::supportsNetworkFileAccess();

  // modify head of filename based on source location
  if (boost::istarts_with(path, "http://") || boost::istarts_with(path, "https://")) {
    if (!gdal_network) {
      THROW_MAPD_EXCEPTION(
          "HTTP geo file import not supported! Update to GDAL 2.2 or later!");
    }
    // invoke GDAL CURL virtual file reader
    path = "/vsicurl/" + path;
  } else if (boost::istarts_with(path, "s3://")) {
    if (!gdal_network) {
      THROW_MAPD_EXCEPTION(
          "S3 geo file import not supported! Update to GDAL 2.2 or later!");
    }
    // invoke GDAL S3 virtual file reader
    boost::replace_first(path, "s3://", "/vsis3/");
  }
}

void add_vsi_geo_prefix(std::string& path) {
  // single gzip'd file (not an archive)?
  if (boost::iends_with(path, ".gz") && !boost::iends_with(path, ".tar.gz")) {
    path = "/vsigzip/" + path;
  }
}

void add_vsi_archive_prefix(std::string& path) {
  // check for compressed file or file bundle
  if (boost::iends_with(path, ".zip")) {
    // zip archive
    path = "/vsizip/" + path;
  } else if (boost::iends_with(path, ".tar") || boost::iends_with(path, ".tgz") ||
             boost::iends_with(path, ".tar.gz")) {
    // tar archive (compressed or uncompressed)
    path = "/vsitar/" + path;
  }
}

std::string remove_vsi_prefixes(const std::string& path_in) {
  std::string path(path_in);

  // these will be first
  if (boost::istarts_with(path, "/vsizip/")) {
    boost::replace_first(path, "/vsizip/", "");
  } else if (boost::istarts_with(path, "/vsitar/")) {
    boost::replace_first(path, "/vsitar/", "");
  } else if (boost::istarts_with(path, "/vsigzip/")) {
    boost::replace_first(path, "/vsigzip/", "");
  }

  // then these
  if (boost::istarts_with(path, "/vsicurl/")) {
    boost::replace_first(path, "/vsicurl/", "");
  } else if (boost::istarts_with(path, "/vsis3/")) {
    boost::replace_first(path, "/vsis3/", "s3://");
  }

  return path;
}

bool path_is_relative(const std::string& path) {
  if (boost::istarts_with(path, "s3://") || boost::istarts_with(path, "http://") ||
      boost::istarts_with(path, "https://")) {
    return false;
  }
  return !boost::filesystem::path(path).is_absolute();
}

bool path_has_valid_filename(const std::string& path) {
  auto filename = boost::filesystem::path(path).filename().string();
  if (filename.size() == 0 || filename[0] == '.' || filename[0] == '/') {
    return false;
  }
  return true;
}

bool is_a_supported_geo_file(const std::string& path, bool include_gz) {
  if (!path_has_valid_filename(path)) {
    return false;
  }
  if (include_gz) {
    if (boost::iends_with(path, ".geojson.gz") || boost::iends_with(path, ".json.gz")) {
      return true;
    }
  }
  if (boost::iends_with(path, ".shp") || boost::iends_with(path, ".geojson") ||
      boost::iends_with(path, ".json") || boost::iends_with(path, ".kml") ||
      boost::iends_with(path, ".kmz") || boost::iends_with(path, ".gdb") ||
      boost::iends_with(path, ".gdb.zip")) {
    return true;
  }
  return false;
}

bool is_a_supported_archive_file(const std::string& path) {
  if (!path_has_valid_filename(path)) {
    return false;
  }
  if (boost::iends_with(path, ".zip") && !boost::iends_with(path, ".gdb.zip")) {
    return true;
  } else if (boost::iends_with(path, ".tar") || boost::iends_with(path, ".tgz") ||
             boost::iends_with(path, ".tar.gz")) {
    return true;
  }
  return false;
}

std::string find_first_geo_file_in_archive(const std::string& archive_path,
                                           const import_export::CopyParams& copy_params) {
  // get the recursive list of all files in the archive
  std::vector<std::string> files =
      import_export::Importer::gdalGetAllFilesInArchive(archive_path, copy_params);

  // report the list
  LOG(INFO) << "Found " << files.size() << " files in Archive "
            << remove_vsi_prefixes(archive_path);
  for (const auto& file : files) {
    LOG(INFO) << "  " << file;
  }

  // scan the list for the first candidate file
  bool found_suitable_file = false;
  std::string file_name;
  for (const auto& file : files) {
    if (is_a_supported_geo_file(file, false)) {
      file_name = file;
      found_suitable_file = true;
      break;
    }
  }

  // if we didn't find anything
  if (!found_suitable_file) {
    LOG(INFO) << "Failed to find any supported geo files in Archive: " +
                     remove_vsi_prefixes(archive_path);
    file_name.clear();
  }

  // done
  return file_name;
}

bool is_local_file(const std::string& file_path) {
  return (!boost::istarts_with(file_path, "s3://") &&
          !boost::istarts_with(file_path, "http://") &&
          !boost::istarts_with(file_path, "https://"));
}

void validate_import_file_path_if_local(const std::string& file_path) {
  if (is_local_file(file_path)) {
    ddl_utils::validate_allowed_file_path(file_path, ddl_utils::DataTransferType::IMPORT);
  }
}
}  // namespace

void DBHandler::detect_column_types(TDetectResult& _return,
                                    const TSessionId& session,
                                    const std::string& file_name_in,
                                    const TCopyParams& cp) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  check_read_only("detect_column_types");

  import_export::CopyParams copy_params = thrift_to_copyparams(cp);

  std::string file_name{file_name_in};
  if (path_is_relative(file_name)) {
    // assume relative paths are relative to data_path / mapd_import / <session>
    auto file_path = import_path_ / picosha2::hash256_hex_string(session) /
                     boost::filesystem::path(file_name).filename();
    file_name = file_path.string();
  }
  validate_import_file_path_if_local(file_name);

  // if it's a geo table, handle alternative paths (S3, HTTP, archive etc.)
  if (copy_params.file_type == import_export::FileType::POLYGON) {
    if (is_a_supported_geo_file(file_name, true)) {
      // prepare to detect geo file directly
      add_vsi_network_prefix(file_name);
      add_vsi_geo_prefix(file_name);
    } else if (is_a_supported_archive_file(file_name)) {
      // find the archive file
      add_vsi_network_prefix(file_name);
      if (!import_export::Importer::gdalFileExists(file_name, copy_params)) {
        THROW_MAPD_EXCEPTION("Archive does not exist: " + file_name_in);
      }
      // find geo file in archive
      add_vsi_archive_prefix(file_name);
      std::string geo_file = find_first_geo_file_in_archive(file_name, copy_params);
      // prepare to detect that geo file
      if (geo_file.size()) {
        file_name = file_name + std::string("/") + geo_file;
      }
    } else {
      THROW_MAPD_EXCEPTION("File is not a supported geo or geo archive format: " +
                           file_name_in);
    }
  }

  auto file_path = boost::filesystem::path(file_name);
  // can be a s3 url
  if (!boost::istarts_with(file_name, "s3://")) {
    if (!boost::filesystem::path(file_name).is_absolute()) {
      file_path = import_path_ / picosha2::hash256_hex_string(session) /
                  boost::filesystem::path(file_name).filename();
      file_name = file_path.string();
    }

    if (copy_params.file_type == import_export::FileType::POLYGON) {
      // check for geo file
      if (!import_export::Importer::gdalFileOrDirectoryExists(file_name, copy_params)) {
        THROW_MAPD_EXCEPTION("File does not exist: " + file_path.string());
      }
    } else {
      // check for regular file
      if (!boost::filesystem::exists(file_path)) {
        THROW_MAPD_EXCEPTION("File does not exist: " + file_path.string());
      }
    }
  }

  try {
    if (copy_params.file_type == import_export::FileType::DELIMITED
#ifdef ENABLE_IMPORT_PARQUET
        || (copy_params.file_type == import_export::FileType::PARQUET)
#endif
    ) {
      import_export::Detector detector(file_path, copy_params);
      std::vector<SQLTypes> best_types = detector.best_sqltypes;
      std::vector<EncodingType> best_encodings = detector.best_encodings;
      std::vector<std::string> headers = detector.get_headers();
      copy_params = detector.get_copy_params();

      _return.copy_params = copyparams_to_thrift(copy_params);
      _return.row_set.row_desc.resize(best_types.size());
      for (size_t col_idx = 0; col_idx < best_types.size(); col_idx++) {
        TColumnType col;
        SQLTypes t = best_types[col_idx];
        EncodingType encodingType = best_encodings[col_idx];
        SQLTypeInfo ti(t, false, encodingType);
        if (IS_GEO(t)) {
          // set this so encoding_to_thrift does the right thing
          ti.set_compression(copy_params.geo_coords_encoding);
          // fill in these directly
          col.col_type.precision = static_cast<int>(copy_params.geo_coords_type);
          col.col_type.scale = copy_params.geo_coords_srid;
          col.col_type.comp_param = copy_params.geo_coords_comp_param;
        }
        col.col_type.type = type_to_thrift(ti);
        col.col_type.encoding = encoding_to_thrift(ti);
        if (copy_params.sanitize_column_names) {
          col.col_name = ImportHelpers::sanitize_name(headers[col_idx]);
        } else {
          col.col_name = headers[col_idx];
        }
        col.is_reserved_keyword = ImportHelpers::is_reserved_name(col.col_name);
        _return.row_set.row_desc[col_idx] = col;
      }
      size_t num_samples = 100;
      auto sample_data = detector.get_sample_rows(num_samples);

      TRow sample_row;
      for (auto row : sample_data) {
        sample_row.cols.clear();
        for (const auto& s : row) {
          TDatum td;
          td.val.str_val = s;
          td.is_null = s.empty();
          sample_row.cols.push_back(td);
        }
        _return.row_set.rows.push_back(sample_row);
      }
    } else if (copy_params.file_type == import_export::FileType::POLYGON) {
      // @TODO simon.eves get this from somewhere!
      const std::string geoColumnName(OMNISCI_GEO_PREFIX);

      check_geospatial_files(file_path, copy_params);
      std::list<ColumnDescriptor> cds = import_export::Importer::gdalToColumnDescriptors(
          file_path.string(), geoColumnName, copy_params);
      for (auto cd : cds) {
        if (copy_params.sanitize_column_names) {
          cd.columnName = ImportHelpers::sanitize_name(cd.columnName);
        }
        _return.row_set.row_desc.push_back(populateThriftColumnType(nullptr, &cd));
      }
      std::map<std::string, std::vector<std::string>> sample_data;
      import_export::Importer::readMetadataSampleGDAL(
          file_path.string(), geoColumnName, sample_data, 100, copy_params);
      if (sample_data.size() > 0) {
        for (size_t i = 0; i < sample_data.begin()->second.size(); i++) {
          TRow sample_row;
          for (auto cd : cds) {
            TDatum td;
            td.val.str_val = sample_data[cd.sourceName].at(i);
            td.is_null = td.val.str_val.empty();
            sample_row.cols.push_back(td);
          }
          _return.row_set.rows.push_back(sample_row);
        }
      }
      _return.copy_params = copyparams_to_thrift(copy_params);
    }
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION("detect_column_types error: " + std::string(e.what()));
  }
}

void DBHandler::render_vega(TRenderResult& _return,
                            const TSessionId& session,
                            const int64_t widget_id,
                            const std::string& vega_json,
                            const int compression_level,
                            const std::string& nonce) {
  auto stdlog = STDLOG(get_session_ptr(session),
                       "widget_id",
                       widget_id,
                       "compression_level",
                       compression_level,
                       "vega_json",
                       vega_json,
                       "nonce",
                       nonce);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  stdlog.appendNameValuePairs("nonce", nonce);
  auto session_ptr = stdlog.getConstSessionInfo();
  if (!render_handler_) {
    THROW_MAPD_EXCEPTION("Backend rendering is disabled.");
  }

  _return.total_time_ms = measure<>::execution([&]() {
    try {
      render_handler_->render_vega(_return,
                                   stdlog.getSessionInfo(),
                                   widget_id,
                                   vega_json,
                                   compression_level,
                                   nonce);
    } catch (std::exception& e) {
      THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
    }
  });
}

static bool is_allowed_on_dashboard(const Catalog_Namespace::SessionInfo& session_info,
                                    int32_t dashboard_id,
                                    AccessPrivileges requestedPermissions) {
  DBObject object(dashboard_id, DashboardDBObjectType);
  auto& catalog = session_info.getCatalog();
  auto& user = session_info.get_currentUser();
  object.loadKey(catalog);
  object.setPrivileges(requestedPermissions);
  std::vector<DBObject> privs = {object};
  return SysCatalog::instance().checkPrivileges(user, privs);
}

// custom expressions
namespace {
using Catalog_Namespace::CustomExpression;
using Catalog_Namespace::DataSourceType;

std::unique_ptr<Catalog_Namespace::CustomExpression> create_custom_expr_from_thrift_obj(
    const TCustomExpression& t_custom_expr) {
  CHECK(t_custom_expr.data_source_type == TDataSourceType::type::TABLE)
      << "Unexpected data source type: "
      << static_cast<int>(t_custom_expr.data_source_type);
  DataSourceType data_source_type = DataSourceType::TABLE;
  return std::make_unique<CustomExpression>(t_custom_expr.name,
                                            t_custom_expr.expression_json,
                                            data_source_type,
                                            t_custom_expr.data_source_id);
}

TCustomExpression create_thrift_obj_from_custom_expr(
    const CustomExpression& custom_expr) {
  TCustomExpression t_custom_expr;
  t_custom_expr.id = custom_expr.id;
  t_custom_expr.name = custom_expr.name;
  t_custom_expr.expression_json = custom_expr.expression_json;
  t_custom_expr.data_source_id = custom_expr.data_source_id;
  t_custom_expr.is_deleted = custom_expr.is_deleted;
  CHECK(custom_expr.data_source_type == DataSourceType::TABLE)
      << "Unexpected data source type: "
      << static_cast<int>(custom_expr.data_source_type);
  t_custom_expr.data_source_type = TDataSourceType::type::TABLE;
  return t_custom_expr;
}
}  // namespace

int32_t DBHandler::create_custom_expression(const TSessionId& session,
                                            const TCustomExpression& t_custom_expr) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  check_read_only("create_custom_expression");

  auto session_ptr = stdlog.getConstSessionInfo();
  if (!session_ptr->get_currentUser().isSuper) {
    THROW_MAPD_EXCEPTION("Custom expressions can only be created by super users.")
  }
  auto& catalog = session_ptr->getCatalog();
  CHECK(t_custom_expr.data_source_type == TDataSourceType::type::TABLE)
      << "Unexpected data source type: "
      << static_cast<int>(t_custom_expr.data_source_type);
  if (catalog.getMetadataForTable(t_custom_expr.data_source_id, false) == nullptr) {
    THROW_MAPD_EXCEPTION("Custom expression references a table that does not exist.")
  }
  mapd_unique_lock<mapd_shared_mutex> write_lock(custom_expressions_mutex_);
  return catalog.createCustomExpression(
      create_custom_expr_from_thrift_obj(t_custom_expr));
}

void DBHandler::get_custom_expressions(std::vector<TCustomExpression>& _return,
                                       const TSessionId& session) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());

  auto session_ptr = stdlog.getConstSessionInfo();
  auto& catalog = session_ptr->getCatalog();
  mapd_shared_lock<mapd_shared_mutex> read_lock(custom_expressions_mutex_);
  auto custom_expressions =
      catalog.getCustomExpressionsForUser(session_ptr->get_currentUser());
  for (const auto& custom_expression : custom_expressions) {
    _return.emplace_back(create_thrift_obj_from_custom_expr(*custom_expression));
  }
}

void DBHandler::update_custom_expression(const TSessionId& session,
                                         const int32_t id,
                                         const std::string& expression_json) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  check_read_only("update_custom_expression");

  auto session_ptr = stdlog.getConstSessionInfo();
  if (!session_ptr->get_currentUser().isSuper) {
    THROW_MAPD_EXCEPTION("Custom expressions can only be updated by super users.")
  }
  auto& catalog = session_ptr->getCatalog();
  mapd_unique_lock<mapd_shared_mutex> write_lock(custom_expressions_mutex_);
  catalog.updateCustomExpression(id, expression_json);
}

void DBHandler::delete_custom_expressions(
    const TSessionId& session,
    const std::vector<int32_t>& custom_expression_ids,
    const bool do_soft_delete) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  check_read_only("delete_custom_expressions");

  auto session_ptr = stdlog.getConstSessionInfo();
  if (!session_ptr->get_currentUser().isSuper) {
    THROW_MAPD_EXCEPTION("Custom expressions can only be deleted by super users.")
  }
  auto& catalog = session_ptr->getCatalog();
  mapd_unique_lock<mapd_shared_mutex> write_lock(custom_expressions_mutex_);
  catalog.deleteCustomExpressions(custom_expression_ids, do_soft_delete);
}

// dashboards
void DBHandler::get_dashboard(TDashboard& dashboard,
                              const TSessionId& session,
                              const int32_t dashboard_id) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  auto const& cat = session_ptr->getCatalog();
  Catalog_Namespace::UserMetadata user_meta;
  auto dash = cat.getMetadataForDashboard(dashboard_id);
  if (!dash) {
    THROW_MAPD_EXCEPTION("Dashboard with dashboard id " + std::to_string(dashboard_id) +
                         " doesn't exist");
  }
  if (!is_allowed_on_dashboard(
          *session_ptr, dash->dashboardId, AccessPrivileges::VIEW_DASHBOARD)) {
    THROW_MAPD_EXCEPTION("User has no view privileges for the dashboard with id " +
                         std::to_string(dashboard_id));
  }
  user_meta.userName = "";
  SysCatalog::instance().getMetadataForUserById(dash->userId, user_meta);
  dashboard = get_dashboard_impl(session_ptr, user_meta, dash);
}

void DBHandler::get_dashboards(std::vector<TDashboard>& dashboards,
                               const TSessionId& session) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  auto const& cat = session_ptr->getCatalog();
  Catalog_Namespace::UserMetadata user_meta;
  const auto dashes = cat.getAllDashboardsMetadata();
  user_meta.userName = "";
  for (const auto dash : dashes) {
    if (is_allowed_on_dashboard(
            *session_ptr, dash->dashboardId, AccessPrivileges::VIEW_DASHBOARD)) {
      // dashboardState is intentionally not populated here
      // for payload reasons
      // use get_dashboard call to get state
      dashboards.push_back(get_dashboard_impl(session_ptr, user_meta, dash, false));
    }
  }
}

TDashboard DBHandler::get_dashboard_impl(
    const std::shared_ptr<Catalog_Namespace::SessionInfo const>& session_ptr,
    Catalog_Namespace::UserMetadata& user_meta,
    const DashboardDescriptor* dash,
    const bool populate_state) {
  auto const& cat = session_ptr->getCatalog();
  SysCatalog::instance().getMetadataForUserById(dash->userId, user_meta);
  auto objects_list = SysCatalog::instance().getMetadataForObject(
      cat.getCurrentDB().dbId,
      static_cast<int>(DBObjectType::DashboardDBObjectType),
      dash->dashboardId);
  TDashboard dashboard;
  dashboard.dashboard_name = dash->dashboardName;
  if (populate_state)
    dashboard.dashboard_state = dash->dashboardState;
  dashboard.image_hash = dash->imageHash;
  dashboard.update_time = dash->updateTime;
  dashboard.dashboard_metadata = dash->dashboardMetadata;
  dashboard.dashboard_id = dash->dashboardId;
  dashboard.dashboard_owner = dash->user;
  TDashboardPermissions perms;
  // Super user has all permissions.
  if (session_ptr->get_currentUser().isSuper) {
    perms.create_ = true;
    perms.delete_ = true;
    perms.edit_ = true;
    perms.view_ = true;
  } else {
    // Collect all grants on current user
    // add them to the permissions.
    auto obj_to_find =
        DBObject(dashboard.dashboard_id, DBObjectType::DashboardDBObjectType);
    obj_to_find.loadKey(session_ptr->getCatalog());
    std::vector<std::string> grantees =
        SysCatalog::instance().getRoles(true,
                                        session_ptr->get_currentUser().isSuper,
                                        session_ptr->get_currentUser().userName);
    for (const auto& grantee : grantees) {
      DBObject* object_found;
      auto* gr = SysCatalog::instance().getGrantee(grantee);
      if (gr && (object_found = gr->findDbObject(obj_to_find.getObjectKey(), true))) {
        const auto obj_privs = object_found->getPrivileges();
        perms.create_ |= obj_privs.hasPermission(DashboardPrivileges::CREATE_DASHBOARD);
        perms.delete_ |= obj_privs.hasPermission(DashboardPrivileges::DELETE_DASHBOARD);
        perms.edit_ |= obj_privs.hasPermission(DashboardPrivileges::EDIT_DASHBOARD);
        perms.view_ |= obj_privs.hasPermission(DashboardPrivileges::VIEW_DASHBOARD);
      }
    }
  }
  dashboard.dashboard_permissions = perms;
  if (objects_list.empty() ||
      (objects_list.size() == 1 && objects_list[0]->roleName == user_meta.userName)) {
    dashboard.is_dash_shared = false;
  } else {
    dashboard.is_dash_shared = true;
  }
  return dashboard;
}

int32_t DBHandler::create_dashboard(const TSessionId& session,
                                    const std::string& dashboard_name,
                                    const std::string& dashboard_state,
                                    const std::string& image_hash,
                                    const std::string& dashboard_metadata) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  check_read_only("create_dashboard");
  auto& cat = session_ptr->getCatalog();

  if (!session_ptr->checkDBAccessPrivileges(DBObjectType::DashboardDBObjectType,
                                            AccessPrivileges::CREATE_DASHBOARD)) {
    THROW_MAPD_EXCEPTION("Not enough privileges to create a dashboard.");
  }

  auto dash = cat.getMetadataForDashboard(
      std::to_string(session_ptr->get_currentUser().userId), dashboard_name);
  if (dash) {
    THROW_MAPD_EXCEPTION("Dashboard with name: " + dashboard_name + " already exists.");
  }

  DashboardDescriptor dd;
  dd.dashboardName = dashboard_name;
  dd.dashboardState = dashboard_state;
  dd.imageHash = image_hash;
  dd.dashboardMetadata = dashboard_metadata;
  dd.userId = session_ptr->get_currentUser().userId;
  dd.user = session_ptr->get_currentUser().userName;

  try {
    auto id = cat.createDashboard(dd);
    // TODO: transactionally unsafe
    SysCatalog::instance().createDBObject(
        session_ptr->get_currentUser(), dashboard_name, DashboardDBObjectType, cat, id);
    return id;
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
  }
}

void DBHandler::replace_dashboard(const TSessionId& session,
                                  const int32_t dashboard_id,
                                  const std::string& dashboard_name,
                                  const std::string& dashboard_owner,
                                  const std::string& dashboard_state,
                                  const std::string& image_hash,
                                  const std::string& dashboard_metadata) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  CHECK(session_ptr);
  check_read_only("replace_dashboard");
  auto& cat = session_ptr->getCatalog();

  if (!is_allowed_on_dashboard(
          *session_ptr, dashboard_id, AccessPrivileges::EDIT_DASHBOARD)) {
    THROW_MAPD_EXCEPTION("Not enough privileges to replace a dashboard.");
  }

  DashboardDescriptor dd;
  dd.dashboardName = dashboard_name;
  dd.dashboardState = dashboard_state;
  dd.imageHash = image_hash;
  dd.dashboardMetadata = dashboard_metadata;
  Catalog_Namespace::UserMetadata user;
  if (!SysCatalog::instance().getMetadataForUser(dashboard_owner, user)) {
    THROW_MAPD_EXCEPTION(std::string("Dashboard owner ") + dashboard_owner +
                         " does not exist");
  }
  dd.userId = user.userId;
  dd.user = dashboard_owner;
  dd.dashboardId = dashboard_id;

  try {
    cat.replaceDashboard(dd);
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
  }
}

void DBHandler::delete_dashboard(const TSessionId& session, const int32_t dashboard_id) {
  delete_dashboards(session, {dashboard_id});
}

void DBHandler::delete_dashboards(const TSessionId& session,
                                  const std::vector<int32_t>& dashboard_ids) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  check_read_only("delete_dashboards");
  auto& cat = session_ptr->getCatalog();
  // Checks will be performed in catalog
  try {
    cat.deleteMetadataForDashboards(dashboard_ids, session_ptr->get_currentUser());
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
  }
}

std::vector<std::string> DBHandler::get_valid_groups(const TSessionId& session,
                                                     int32_t dashboard_id,
                                                     std::vector<std::string> groups) {
  const auto session_info = get_session_copy(session);
  auto& cat = session_info.getCatalog();
  auto dash = cat.getMetadataForDashboard(dashboard_id);
  if (!dash) {
    THROW_MAPD_EXCEPTION("Exception: Dashboard id " + std::to_string(dashboard_id) +
                         " does not exist");
  } else if (session_info.get_currentUser().userId != dash->userId &&
             !session_info.get_currentUser().isSuper) {
    throw std::runtime_error(
        "User should be either owner of dashboard or super user to share/unshare it");
  }
  std::vector<std::string> valid_groups;
  Catalog_Namespace::UserMetadata user_meta;
  for (auto& group : groups) {
    user_meta.isSuper = false;  // initialize default flag
    if (!SysCatalog::instance().getGrantee(group)) {
      THROW_MAPD_EXCEPTION("Exception: User/Role " + group + " does not exist");
    } else if (!user_meta.isSuper) {
      valid_groups.push_back(group);
    }
  }
  return valid_groups;
}

void DBHandler::validateGroups(const std::vector<std::string>& groups) {
  for (auto const& group : groups) {
    if (!SysCatalog::instance().getGrantee(group)) {
      THROW_MAPD_EXCEPTION("Exception: User/Role '" + group + "' does not exist");
    }
  }
}

void DBHandler::validateDashboardIdsForSharing(
    const Catalog_Namespace::SessionInfo& session_info,
    const std::vector<int32_t>& dashboard_ids) {
  auto& cat = session_info.getCatalog();
  std::map<std::string, std::list<int32_t>> errors;
  for (auto const& dashboard_id : dashboard_ids) {
    auto dashboard = cat.getMetadataForDashboard(dashboard_id);
    if (!dashboard) {
      errors["Dashboard id does not exist"].push_back(dashboard_id);
    } else if (session_info.get_currentUser().userId != dashboard->userId &&
               !session_info.get_currentUser().isSuper) {
      errors["User should be either owner of dashboard or super user to share/unshare it"]
          .push_back(dashboard_id);
    }
  }
  if (!errors.empty()) {
    std::stringstream error_stream;
    error_stream << "Share/Unshare dashboard(s) failed with error(s)\n";
    for (const auto& [error, id_list] : errors) {
      error_stream << "Dashboard ids " << join(id_list, ", ") << ": " << error << "\n";
    }
    THROW_MAPD_EXCEPTION(error_stream.str());
  }
}

void DBHandler::shareOrUnshareDashboards(const TSessionId& session,
                                         const std::vector<int32_t>& dashboard_ids,
                                         const std::vector<std::string>& groups,
                                         const TDashboardPermissions& permissions,
                                         const bool do_share) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  check_read_only(do_share ? "share_dashboards" : "unshare_dashboards");
  if (!permissions.create_ && !permissions.delete_ && !permissions.edit_ &&
      !permissions.view_) {
    THROW_MAPD_EXCEPTION("At least one privilege should be assigned for " +
                         std::string(do_share ? "grants" : "revokes"));
  }
  auto session_ptr = stdlog.getConstSessionInfo();
  auto const& catalog = session_ptr->getCatalog();
  auto& sys_catalog = SysCatalog::instance();
  validateGroups(groups);
  validateDashboardIdsForSharing(*session_ptr, dashboard_ids);
  std::vector<DBObject> batch_objects;
  for (auto const& dashboard_id : dashboard_ids) {
    DBObject object(dashboard_id, DBObjectType::DashboardDBObjectType);
    AccessPrivileges privs;
    if (permissions.delete_) {
      privs.add(AccessPrivileges::DELETE_DASHBOARD);
    }
    if (permissions.create_) {
      privs.add(AccessPrivileges::CREATE_DASHBOARD);
    }
    if (permissions.edit_) {
      privs.add(AccessPrivileges::EDIT_DASHBOARD);
    }
    if (permissions.view_) {
      privs.add(AccessPrivileges::VIEW_DASHBOARD);
    }
    object.setPrivileges(privs);
    batch_objects.push_back(object);
  }
  if (do_share) {
    sys_catalog.grantDBObjectPrivilegesBatch(groups, batch_objects, catalog);
  } else {
    sys_catalog.revokeDBObjectPrivilegesBatch(groups, batch_objects, catalog);
  }
}

void DBHandler::share_dashboards(const TSessionId& session,
                                 const std::vector<int32_t>& dashboard_ids,
                                 const std::vector<std::string>& groups,
                                 const TDashboardPermissions& permissions) {
  shareOrUnshareDashboards(session, dashboard_ids, groups, permissions, true);
}

// NOOP: Grants not available for objects as of now
void DBHandler::share_dashboard(const TSessionId& session,
                                const int32_t dashboard_id,
                                const std::vector<std::string>& groups,
                                const std::vector<std::string>& objects,
                                const TDashboardPermissions& permissions,
                                const bool grant_role = false) {
  share_dashboards(session, {dashboard_id}, groups, permissions);
}

void DBHandler::unshare_dashboards(const TSessionId& session,
                                   const std::vector<int32_t>& dashboard_ids,
                                   const std::vector<std::string>& groups,
                                   const TDashboardPermissions& permissions) {
  shareOrUnshareDashboards(session, dashboard_ids, groups, permissions, false);
}

void DBHandler::unshare_dashboard(const TSessionId& session,
                                  const int32_t dashboard_id,
                                  const std::vector<std::string>& groups,
                                  const std::vector<std::string>& objects,
                                  const TDashboardPermissions& permissions) {
  unshare_dashboards(session, {dashboard_id}, groups, permissions);
}

void DBHandler::get_dashboard_grantees(
    std::vector<TDashboardGrantees>& dashboard_grantees,
    const TSessionId& session,
    const int32_t dashboard_id) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  auto const& cat = session_ptr->getCatalog();
  Catalog_Namespace::UserMetadata user_meta;
  auto dash = cat.getMetadataForDashboard(dashboard_id);
  if (!dash) {
    THROW_MAPD_EXCEPTION("Exception: Dashboard id " + std::to_string(dashboard_id) +
                         " does not exist");
  } else if (session_ptr->get_currentUser().userId != dash->userId &&
             !session_ptr->get_currentUser().isSuper) {
    THROW_MAPD_EXCEPTION(
        "User should be either owner of dashboard or super user to access grantees");
  }
  std::vector<ObjectRoleDescriptor*> objectsList;
  objectsList = SysCatalog::instance().getMetadataForObject(
      cat.getCurrentDB().dbId,
      static_cast<int>(DBObjectType::DashboardDBObjectType),
      dashboard_id);  // By default objecttypecan be only dashabaords
  user_meta.userId = -1;
  user_meta.userName = "";
  SysCatalog::instance().getMetadataForUserById(dash->userId, user_meta);
  for (auto object : objectsList) {
    if (user_meta.userName == object->roleName) {
      // Mask owner
      continue;
    }
    TDashboardGrantees grantee;
    TDashboardPermissions perm;
    grantee.name = object->roleName;
    grantee.is_user = object->roleType;
    perm.create_ = object->privs.hasPermission(DashboardPrivileges::CREATE_DASHBOARD);
    perm.delete_ = object->privs.hasPermission(DashboardPrivileges::DELETE_DASHBOARD);
    perm.edit_ = object->privs.hasPermission(DashboardPrivileges::EDIT_DASHBOARD);
    perm.view_ = object->privs.hasPermission(DashboardPrivileges::VIEW_DASHBOARD);
    grantee.permissions = perm;
    dashboard_grantees.push_back(grantee);
  }
}

void DBHandler::create_link(std::string& _return,
                            const TSessionId& session,
                            const std::string& view_state,
                            const std::string& view_metadata) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  // check_read_only("create_link");
  auto& cat = session_ptr->getCatalog();

  LinkDescriptor ld;
  ld.userId = session_ptr->get_currentUser().userId;
  ld.viewState = view_state;
  ld.viewMetadata = view_metadata;

  try {
    _return = cat.createLink(ld, 6);
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
  }
}

TColumnType DBHandler::create_geo_column(const TDatumType::type type,
                                         const std::string& name,
                                         const bool is_array) {
  TColumnType ct;
  ct.col_name = name;
  ct.col_type.type = type;
  ct.col_type.is_array = is_array;
  return ct;
}

void DBHandler::check_geospatial_files(const boost::filesystem::path file_path,
                                       const import_export::CopyParams& copy_params) {
  const std::list<std::string> shp_ext{".shp", ".shx", ".dbf"};
  if (std::find(shp_ext.begin(),
                shp_ext.end(),
                boost::algorithm::to_lower_copy(file_path.extension().string())) !=
      shp_ext.end()) {
    for (auto ext : shp_ext) {
      auto aux_file = file_path;
      if (!import_export::Importer::gdalFileExists(
              aux_file.replace_extension(boost::algorithm::to_upper_copy(ext)).string(),
              copy_params) &&
          !import_export::Importer::gdalFileExists(
              aux_file.replace_extension(ext).string(), copy_params)) {
        throw std::runtime_error("required file for shapefile does not exist: " +
                                 aux_file.filename().string());
      }
    }
  }
}

void DBHandler::create_table(const TSessionId& session,
                             const std::string& table_name,
                             const TRowDescriptor& rd,
                             const TFileType::type file_type,
                             const TCreateParams& create_params) {
  auto stdlog = STDLOG("table_name", table_name);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  check_read_only("create_table");

  if (ImportHelpers::is_reserved_name(table_name)) {
    THROW_MAPD_EXCEPTION("Invalid table name (reserved keyword): " + table_name);
  } else if (table_name != ImportHelpers::sanitize_name(table_name)) {
    THROW_MAPD_EXCEPTION("Invalid characters in table name: " + table_name);
  }

  auto rds = rd;

  // no longer need to manually add the poly column for a TFileType::POLYGON table
  // a column of the correct geo type has already been added
  // @TODO simon.eves rename TFileType::POLYGON to TFileType::GEO or something!

  std::string stmt{"CREATE TABLE " + table_name};
  std::vector<std::string> col_stmts;

  for (auto col : rds) {
    if (ImportHelpers::is_reserved_name(col.col_name)) {
      THROW_MAPD_EXCEPTION("Invalid column name (reserved keyword): " + col.col_name);
    } else if (col.col_name != ImportHelpers::sanitize_name(col.col_name)) {
      THROW_MAPD_EXCEPTION("Invalid characters in column name: " + col.col_name);
    }
    if (col.col_type.type == TDatumType::INTERVAL_DAY_TIME ||
        col.col_type.type == TDatumType::INTERVAL_YEAR_MONTH) {
      THROW_MAPD_EXCEPTION("Unsupported type: " + thrift_to_name(col.col_type) +
                           " for column: " + col.col_name);
    }

    if (col.col_type.type == TDatumType::DECIMAL) {
      // if no precision or scale passed in set to default 14,7
      if (col.col_type.precision == 0 && col.col_type.scale == 0) {
        col.col_type.precision = 14;
        col.col_type.scale = 7;
      }
    }

    std::string col_stmt;
    col_stmt.append(col.col_name + " " + thrift_to_name(col.col_type));

    // As of 2016-06-27 the Immerse v1 frontend does not explicitly set the
    // `nullable` argument, leading this to default to false. Uncomment for v2.
    // if (!col.col_type.nullable) col_stmt.append(" NOT NULL");

    if (thrift_to_encoding(col.col_type.encoding) != kENCODING_NONE) {
      col_stmt.append(" ENCODING " + thrift_to_encoding_name(col.col_type));
      if (thrift_to_encoding(col.col_type.encoding) == kENCODING_DICT ||
          thrift_to_encoding(col.col_type.encoding) == kENCODING_FIXED ||
          thrift_to_encoding(col.col_type.encoding) == kENCODING_GEOINT) {
        col_stmt.append("(" + std::to_string(col.col_type.comp_param) + ")");
      }
    } else if (col.col_type.type == TDatumType::STR) {
      // non DICT encoded strings
      col_stmt.append(" ENCODING NONE");
    } else if (col.col_type.type == TDatumType::POINT ||
               col.col_type.type == TDatumType::LINESTRING ||
               col.col_type.type == TDatumType::POLYGON ||
               col.col_type.type == TDatumType::MULTIPOLYGON) {
      // non encoded compressable geo
      if (col.col_type.scale == 4326) {
        col_stmt.append(" ENCODING NONE");
      }
    }
    col_stmts.push_back(col_stmt);
  }

  stmt.append(" (" + boost::algorithm::join(col_stmts, ", ") + ")");

  if (create_params.is_replicated) {
    stmt.append(" WITH (PARTITIONS = 'REPLICATED')");
  }

  stmt.append(";");

  TQueryResult ret;
  sql_execute(ret, session, stmt, true, "", -1, -1);
}

void DBHandler::import_table(const TSessionId& session,
                             const std::string& table_name,
                             const std::string& file_name_in,
                             const TCopyParams& cp) {
  try {
    auto stdlog = STDLOG(get_session_ptr(session), "table_name", table_name);
    stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
    auto session_ptr = stdlog.getConstSessionInfo();
    check_read_only("import_table");
    LOG(INFO) << "import_table " << table_name << " from " << file_name_in;

    auto& cat = session_ptr->getCatalog();
    auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
    auto start_time = ::toString(std::chrono::system_clock::now());
    if (g_enable_non_kernel_time_query_interrupt) {
      executor->enrollQuerySession(session,
                                   "IMPORT_TABLE",
                                   start_time,
                                   Executor::UNITARY_EXECUTOR_ID,
                                   QuerySessionStatus::QueryStatus::RUNNING);
    }

    ScopeGuard clearInterruptStatus = [executor, &session, &start_time] {
      // reset the runtime query interrupt status
      if (g_enable_non_kernel_time_query_interrupt) {
        executor->clearQuerySessionStatus(session, start_time, false);
      }
    };
    const auto td_with_lock =
        lockmgr::TableSchemaLockContainer<lockmgr::ReadLock>::acquireTableDescriptor(
            cat, table_name);
    const auto td = td_with_lock();
    CHECK(td);
    check_table_load_privileges(*session_ptr, table_name);

    std::string file_name{file_name_in};
    auto file_path = boost::filesystem::path(file_name);
    import_export::CopyParams copy_params = thrift_to_copyparams(cp);
    if (!boost::istarts_with(file_name, "s3://")) {
      if (!boost::filesystem::path(file_name).is_absolute()) {
        file_path = import_path_ / picosha2::hash256_hex_string(session) /
                    boost::filesystem::path(file_name).filename();
        file_name = file_path.string();
      }
      if (!boost::filesystem::exists(file_path)) {
        THROW_MAPD_EXCEPTION("File does not exist: " + file_path.string());
      }
    }
    validate_import_file_path_if_local(file_name);

    // TODO(andrew): add delimiter detection to Importer
    if (copy_params.delimiter == '\0') {
      copy_params.delimiter = ',';
      if (boost::filesystem::extension(file_path) == ".tsv") {
        copy_params.delimiter = '\t';
      }
    }

    const auto insert_data_lock = lockmgr::InsertDataLockMgr::getWriteLockForTable(
        session_ptr->getCatalog(), table_name);
    std::unique_ptr<import_export::Importer> importer;
    if (leaf_aggregator_.leafCount() > 0) {
      importer.reset(new import_export::Importer(
          new DistributedLoader(*session_ptr, td, &leaf_aggregator_),
          file_path.string(),
          copy_params));
    } else {
      importer.reset(
          new import_export::Importer(cat, td, file_path.string(), copy_params));
    }
    auto ms = measure<>::execution([&]() { importer->import(session_ptr.get()); });
    std::cout << "Total Import Time: " << (double)ms / 1000.0 << " Seconds." << std::endl;
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION("Exception: " + std::string(e.what()));
  }
}

namespace {

// helper functions for error checking below
// these would usefully be added as methods of TDatumType
// but that's not possible as it's auto-generated by Thrift

bool TTypeInfo_IsGeo(const TDatumType::type& t) {
  return (t == TDatumType::POLYGON || t == TDatumType::MULTIPOLYGON ||
          t == TDatumType::LINESTRING || t == TDatumType::POINT);
}

#if ENABLE_GEO_IMPORT_COLUMN_MATCHING

std::string TTypeInfo_TypeToString(const TDatumType::type& t) {
  std::stringstream ss;
  ss << t;
  return ss.str();
}

std::string TTypeInfo_GeoSubTypeToString(const int32_t p) {
  std::string result;
  switch (p) {
    case SQLTypes::kGEOGRAPHY:
      result = "GEOGRAPHY";
      break;
    case SQLTypes::kGEOMETRY:
      result = "GEOMETRY";
      break;
    default:
      result = "INVALID";
      break;
  }
  return result;
}

std::string TTypeInfo_EncodingToString(const TEncodingType::type& t) {
  std::stringstream ss;
  ss << t;
  return ss.str();
}

#endif

}  // namespace

#define THROW_COLUMN_ATTR_MISMATCH_EXCEPTION(attr, got, expected)                      \
  THROW_MAPD_EXCEPTION("Could not append geo file '" + file_path.filename().string() + \
                       "' to table '" + table_name + "'. Column '" + cd->columnName +  \
                       "' " + attr + " mismatch (got '" + got + "', expected '" +      \
                       expected + "')");

void DBHandler::import_geo_table(const TSessionId& session,
                                 const std::string& table_name,
                                 const std::string& file_name_in,
                                 const TCopyParams& cp,
                                 const TRowDescriptor& row_desc,
                                 const TCreateParams& create_params) {
  auto stdlog = STDLOG(get_session_ptr(session), "table_name", table_name);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  check_read_only("import_table");

  auto& cat = session_ptr->getCatalog();
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
  auto start_time = ::toString(std::chrono::system_clock::now());
  if (g_enable_non_kernel_time_query_interrupt) {
    executor->enrollQuerySession(session,
                                 "IMPORT_GEO_TABLE",
                                 start_time,
                                 Executor::UNITARY_EXECUTOR_ID,
                                 QuerySessionStatus::QueryStatus::RUNNING);
  }

  ScopeGuard clearInterruptStatus = [executor, &session, &start_time] {
    // reset the runtime query interrupt status
    if (g_enable_non_kernel_time_query_interrupt) {
      executor->clearQuerySessionStatus(session, start_time, false);
    }
  };

  import_export::CopyParams copy_params = thrift_to_copyparams(cp);

  std::string file_name{file_name_in};

  if (path_is_relative(file_name)) {
    // assume relative paths are relative to data_path / mapd_import / <session>
    auto file_path = import_path_ / picosha2::hash256_hex_string(session) /
                     boost::filesystem::path(file_name).filename();
    file_name = file_path.string();
  }
  validate_import_file_path_if_local(file_name);

  if (is_a_supported_geo_file(file_name, true)) {
    // prepare to load geo file directly
    add_vsi_network_prefix(file_name);
    add_vsi_geo_prefix(file_name);
  } else if (is_a_supported_archive_file(file_name)) {
    // find the archive file
    add_vsi_network_prefix(file_name);
    if (!import_export::Importer::gdalFileExists(file_name, copy_params)) {
      THROW_MAPD_EXCEPTION("Archive does not exist: " + file_name_in);
    }
    // find geo file in archive
    add_vsi_archive_prefix(file_name);
    std::string geo_file = find_first_geo_file_in_archive(file_name, copy_params);
    // prepare to load that geo file
    if (geo_file.size()) {
      file_name = file_name + std::string("/") + geo_file;
    }
  } else {
    THROW_MAPD_EXCEPTION("File is not a supported geo or geo archive file: " +
                         file_name_in);
  }

  // log what we're about to try to do
  LOG(INFO) << "import_geo_table: Original filename: " << file_name_in;
  LOG(INFO) << "import_geo_table: Actual filename: " << file_name;

  // use GDAL to check the primary file exists (even if on S3 and/or in archive)
  auto file_path = boost::filesystem::path(file_name);
  if (!import_export::Importer::gdalFileOrDirectoryExists(file_name, copy_params)) {
    THROW_MAPD_EXCEPTION("File does not exist: " + file_path.filename().string());
  }

  // use GDAL to check any dependent files exist (ditto)
  try {
    check_geospatial_files(file_path, copy_params);
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION("import_geo_table error: " + std::string(e.what()));
  }

  // get layer info and deconstruct
  // in general, we will get a combination of layers of these four types:
  //   EMPTY: no rows, report and skip
  //   GEO: create a geo table from this
  //   NON_GEO: create a regular table from this
  //   UNSUPPORTED_GEO: report and skip
  std::vector<import_export::Importer::GeoFileLayerInfo> layer_info;
  try {
    layer_info = import_export::Importer::gdalGetLayersInGeoFile(file_name, copy_params);
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION("import_geo_table error: " + std::string(e.what()));
  }

  // categorize the results
  using LayerNameToContentsMap =
      std::map<std::string, import_export::Importer::GeoFileLayerContents>;
  LayerNameToContentsMap load_layers;
  LOG(INFO) << "import_geo_table: Found the following layers in the geo file:";
  for (const auto& layer : layer_info) {
    switch (layer.contents) {
      case import_export::Importer::GeoFileLayerContents::GEO:
        LOG(INFO) << "import_geo_table:   '" << layer.name
                  << "' (will import as geo table)";
        load_layers[layer.name] = layer.contents;
        break;
      case import_export::Importer::GeoFileLayerContents::NON_GEO:
        LOG(INFO) << "import_geo_table:   '" << layer.name
                  << "' (will import as regular table)";
        load_layers[layer.name] = layer.contents;
        break;
      case import_export::Importer::GeoFileLayerContents::UNSUPPORTED_GEO:
        LOG(WARNING) << "import_geo_table:   '" << layer.name
                     << "' (will not import, unsupported geo type)";
        break;
      case import_export::Importer::GeoFileLayerContents::EMPTY:
        LOG(INFO) << "import_geo_table:   '" << layer.name << "' (ignoring, empty)";
        break;
      default:
        break;
    }
  }

  // if nothing is loadable, stop now
  if (load_layers.size() == 0) {
    THROW_MAPD_EXCEPTION("import_geo_table: No loadable layers found, aborting!");
  }

  // if we've been given an explicit layer name, check that it exists and is loadable
  // scan the original list, as it may exist but not have been gathered as loadable
  if (copy_params.geo_layer_name.size()) {
    bool found = false;
    for (const auto& layer : layer_info) {
      if (copy_params.geo_layer_name == layer.name) {
        if (layer.contents == import_export::Importer::GeoFileLayerContents::GEO ||
            layer.contents == import_export::Importer::GeoFileLayerContents::NON_GEO) {
          // forget all the other layers and just load this one
          load_layers.clear();
          load_layers[layer.name] = layer.contents;
          found = true;
          break;
        } else if (layer.contents ==
                   import_export::Importer::GeoFileLayerContents::UNSUPPORTED_GEO) {
          THROW_MAPD_EXCEPTION("import_geo_table: Explicit geo layer '" +
                               copy_params.geo_layer_name +
                               "' has unsupported geo type!");
        } else if (layer.contents ==
                   import_export::Importer::GeoFileLayerContents::EMPTY) {
          THROW_MAPD_EXCEPTION("import_geo_table: Explicit geo layer '" +
                               copy_params.geo_layer_name + "' is empty!");
        }
      }
    }
    if (!found) {
      THROW_MAPD_EXCEPTION("import_geo_table: Explicit geo layer '" +
                           copy_params.geo_layer_name + "' not found!");
    }
  }

  // Immerse import of multiple layers is not yet supported
  // @TODO fix this!
  if (row_desc.size() > 0 && load_layers.size() > 1) {
    THROW_MAPD_EXCEPTION(
        "import_geo_table: Multi-layer geo import not yet supported from Immerse!");
  }

  // one definition of layer table name construction
  // we append the layer name if we're loading more than one table
  auto construct_layer_table_name = [&load_layers](const std::string& table_name,
                                                   const std::string& layer_name) {
    if (load_layers.size() > 1) {
      auto sanitized_layer_name = ImportHelpers::sanitize_name(layer_name);
      if (sanitized_layer_name != layer_name) {
        LOG(INFO) << "import_geo_table: Using sanitized layer name '"
                  << sanitized_layer_name << "' for table name";
      }
      return table_name + "_" + sanitized_layer_name;
    }
    return table_name;
  };

  // if we're importing multiple tables, then NONE of them must exist already
  if (load_layers.size() > 1) {
    for (const auto& layer : load_layers) {
      // construct table name
      auto this_table_name = construct_layer_table_name(table_name, layer.first);

      // table must not exist
      if (cat.getMetadataForTable(this_table_name)) {
        THROW_MAPD_EXCEPTION("import_geo_table: Table '" + this_table_name +
                             "' already exists, aborting!");
      }
    }
  }

  // prepare to gather errors that would otherwise be exceptions, as we can only throw
  // one
  std::vector<std::string> caught_exception_messages;

  // prepare to time multi-layer import
  double total_import_ms = 0.0;

  // now we're safe to start importing
  // we loop over the layers we're going to attempt to load
  for (const auto& layer : load_layers) {
    // unpack
    const auto& layer_name = layer.first;
    const auto& layer_contents = layer.second;
    bool is_geo_layer =
        (layer_contents == import_export::Importer::GeoFileLayerContents::GEO);

    // construct table name again
    auto this_table_name = construct_layer_table_name(table_name, layer_name);

    // report
    LOG(INFO) << "import_geo_table: Creating table: " << this_table_name;

    // we need a row descriptor
    TRowDescriptor rd;
    if (row_desc.size() > 0) {
      // we have a valid RowDescriptor
      // this is the case where Immerse has already detected and created
      // all we need to do is import and trust that the data will match
      // use the provided row descriptor
      // table must already exist (we check this below)
      rd = row_desc;
    } else {
      // we don't have a RowDescriptor
      // we have to detect the file ourselves
      TDetectResult cds;
      TCopyParams cp_copy = cp;  // retain S3 auth tokens
      cp_copy.geo_layer_name = layer_name;
      cp_copy.file_type = TFileType::POLYGON;
      try {
        detect_column_types(cds, session, file_name_in, cp_copy);
      } catch (const std::exception& e) {
        // capture the error and abort this layer
        caught_exception_messages.emplace_back(
            "Invalid/Unsupported Column Types in Layer '" + layer_name + "':" + e.what());
        continue;
      }
      rd = cds.row_set.row_desc;

      // then, if the table does NOT already exist, create it
      const TableDescriptor* td = cat.getMetadataForTable(this_table_name);
      if (!td) {
        try {
          create_table(session, this_table_name, rd, TFileType::POLYGON, create_params);
        } catch (const std::exception& e) {
          // capture the error and abort this layer
          caught_exception_messages.emplace_back("Failed to create table for Layer '" +
                                                 layer_name + "':" + e.what());
          continue;
        }
      }
    }

    // by this point, the table should exist, one way or another
    const TableDescriptor* td = cat.getMetadataForTable(this_table_name);
    if (!td) {
      // capture the error and abort this layer
      std::string exception_message =
          "Could not import geo file '" + file_path.filename().string() + "' to table '" +
          this_table_name + "'; table does not exist or failed to create.";
      caught_exception_messages.emplace_back(exception_message);
      continue;
    }

    // then, we have to verify that the structure matches
    // get column descriptors (non-system, non-deleted, logical columns only)
    const auto col_descriptors =
        cat.getAllColumnMetadataForTable(td->tableId, false, false, false);

    // first, compare the column count
    if (col_descriptors.size() != rd.size()) {
      // capture the error and abort this layer
      std::string exception_message =
          "Could not append geo file '" + file_path.filename().string() + "' to table '" +
          this_table_name + "'. Column count mismatch (got " + std::to_string(rd.size()) +
          ", expecting " + std::to_string(col_descriptors.size()) + ")";
      caught_exception_messages.emplace_back(exception_message);
      continue;
    }

    try {
      // then the names and types
      int rd_index = 0;
      for (auto cd : col_descriptors) {
        TColumnType cd_col_type = populateThriftColumnType(&cat, cd);
        std::string gname = rd[rd_index].col_name;  // got
        std::string ename = cd->columnName;         // expecting
#if ENABLE_GEO_IMPORT_COLUMN_MATCHING
        TTypeInfo gti = rd[rd_index].col_type;  // got
#endif
        TTypeInfo eti = cd_col_type.col_type;  // expecting
        // check for name match
        if (gname != ename) {
          if (TTypeInfo_IsGeo(eti.type) && ename == LEGACY_GEO_PREFIX &&
              gname == OMNISCI_GEO_PREFIX) {
            // rename incoming geo column to match existing legacy default geo column
            rd[rd_index].col_name = gname;
            LOG(INFO)
                << "import_geo_table: Renaming incoming geo column to match existing "
                   "legacy default geo column";
          } else {
            THROW_COLUMN_ATTR_MISMATCH_EXCEPTION("name", gname, ename);
          }
        }
#if ENABLE_GEO_IMPORT_COLUMN_MATCHING
        // check for type attributes match
        // these attrs must always match regardless of type
        if (gti.type != eti.type) {
          THROW_COLUMN_ATTR_MISMATCH_EXCEPTION(
              "type", TTypeInfo_TypeToString(gti.type), TTypeInfo_TypeToString(eti.type));
        }
        if (gti.is_array != eti.is_array) {
          THROW_COLUMN_ATTR_MISMATCH_EXCEPTION(
              "array-ness", std::to_string(gti.is_array), std::to_string(eti.is_array));
        }
        if (gti.nullable != eti.nullable) {
          THROW_COLUMN_ATTR_MISMATCH_EXCEPTION(
              "nullability", std::to_string(gti.nullable), std::to_string(eti.nullable));
        }
        if (TTypeInfo_IsGeo(eti.type)) {
          // for geo, only these other attrs must also match
          // encoding and comp_param are allowed to differ
          // this allows appending to existing geo table
          // without needing to know the existing encoding
          if (gti.precision != eti.precision) {
            THROW_COLUMN_ATTR_MISMATCH_EXCEPTION(
                "geo sub-type",
                TTypeInfo_GeoSubTypeToString(gti.precision),
                TTypeInfo_GeoSubTypeToString(eti.precision));
          }
          if (gti.scale != eti.scale) {
            THROW_COLUMN_ATTR_MISMATCH_EXCEPTION(
                "SRID", std::to_string(gti.scale), std::to_string(eti.scale));
          }
          if (gti.encoding != eti.encoding) {
            LOG(INFO) << "import_geo_table: Ignoring geo encoding mismatch";
          }
          if (gti.comp_param != eti.comp_param) {
            LOG(INFO) << "import_geo_table: Ignoring geo comp_param mismatch";
          }
        } else {
          // non-geo, all other attrs must also match
          // @TODO consider relaxing some of these dependent on type
          // e.g. DECIMAL precision/scale, TEXT dict or non-dict, INTEGER up-sizing
          if (gti.precision != eti.precision) {
            THROW_COLUMN_ATTR_MISMATCH_EXCEPTION("precision",
                                                 std::to_string(gti.precision),
                                                 std::to_string(eti.precision));
          }
          if (gti.scale != eti.scale) {
            THROW_COLUMN_ATTR_MISMATCH_EXCEPTION(
                "scale", std::to_string(gti.scale), std::to_string(eti.scale));
          }
          if (gti.encoding != eti.encoding) {
            THROW_COLUMN_ATTR_MISMATCH_EXCEPTION(
                "encoding",
                TTypeInfo_EncodingToString(gti.encoding),
                TTypeInfo_EncodingToString(eti.encoding));
          }
          if (gti.comp_param != eti.comp_param) {
            THROW_COLUMN_ATTR_MISMATCH_EXCEPTION("comp param",
                                                 std::to_string(gti.comp_param),
                                                 std::to_string(eti.comp_param));
          }
        }
#endif
        rd_index++;
      }
    } catch (const std::exception& e) {
      // capture the error and abort this layer
      caught_exception_messages.emplace_back(e.what());
      continue;
    }

    std::map<std::string, std::string> colname_to_src;
    for (auto r : rd) {
      colname_to_src[r.col_name] =
          r.src_name.length() > 0 ? r.src_name : ImportHelpers::sanitize_name(r.src_name);
    }

    try {
      check_table_load_privileges(*session_ptr, this_table_name);
    } catch (const std::exception& e) {
      // capture the error and abort this layer
      caught_exception_messages.emplace_back(e.what());
      continue;
    }

    if (is_geo_layer) {
      // Final check to ensure that we actually have a geo column
      // of the expected name and type before doing the actual import,
      // in case the user naively overrode the name or type in Immerse
      // Preview (which as of 6/8/18 it still allows you to do).
      // This avoids a fatal assert later when it fails to find the
      // column. We should make Immerse more robust and disallow this.
      bool have_geo_column_with_correct_name = false;
      for (const auto& r : rd) {
        if (TTypeInfo_IsGeo(r.col_type.type)) {
          // TODO(team): allow user to override the geo column name
          if (r.col_name == OMNISCI_GEO_PREFIX) {
            have_geo_column_with_correct_name = true;
          } else if (r.col_name == LEGACY_GEO_PREFIX) {
            CHECK(colname_to_src.find(r.col_name) != colname_to_src.end());
            // Normalize column names for geo append with legacy column naming scheme
            colname_to_src[r.col_name] = r.col_name;
            have_geo_column_with_correct_name = true;
          }
        }
      }
      if (!have_geo_column_with_correct_name) {
        std::string exception_message = "Table " + this_table_name +
                                        " does not have a geo column with name '" +
                                        OMNISCI_GEO_PREFIX + "'. Import aborted!";
        caught_exception_messages.emplace_back(exception_message);
        continue;
      }
    }

    try {
      // import this layer only?
      copy_params.geo_layer_name = layer_name;

      // create an importer
      std::unique_ptr<import_export::Importer> importer;
      if (leaf_aggregator_.leafCount() > 0) {
        importer.reset(new import_export::Importer(
            new DistributedLoader(*session_ptr, td, &leaf_aggregator_),
            file_path.string(),
            copy_params));
      } else {
        importer.reset(
            new import_export::Importer(cat, td, file_path.string(), copy_params));
      }

      // import
      auto ms = measure<>::execution(
          [&]() { importer->importGDAL(colname_to_src, session_ptr.get()); });
      LOG(INFO) << "Import of Layer '" << layer_name << "' took " << (double)ms / 1000.0
                << "s";
      total_import_ms += ms;
    } catch (const std::exception& e) {
      std::string exception_message =
          "Import of Layer '" + this_table_name + "' failed: " + e.what();
      caught_exception_messages.emplace_back(exception_message);
      continue;
    }
  }

  // did we catch any exceptions?
  if (caught_exception_messages.size()) {
    // combine all the strings into one and throw a single Thrift exception
    std::string combined_exception_message = "Failed to import geo file:\n";
    for (const auto& message : caught_exception_messages) {
      combined_exception_message += message + "\n";
    }
    THROW_MAPD_EXCEPTION(combined_exception_message);
  } else {
    // report success and total time
    LOG(INFO) << "Import Successful!";
    LOG(INFO) << "Total Import Time: " << total_import_ms / 1000.0 << "s";
  }
}

#undef THROW_COLUMN_ATTR_MISMATCH_EXCEPTION

void DBHandler::import_table_status(TImportStatus& _return,
                                    const TSessionId& session,
                                    const std::string& import_id) {
  auto stdlog = STDLOG(get_session_ptr(session), "import_table_status", import_id);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto is = import_export::Importer::get_import_status(import_id);
  _return.elapsed = is.elapsed.count();
  _return.rows_completed = is.rows_completed;
  _return.rows_estimated = is.rows_estimated;
  _return.rows_rejected = is.rows_rejected;
}

void DBHandler::get_first_geo_file_in_archive(std::string& _return,
                                              const TSessionId& session,
                                              const std::string& archive_path_in,
                                              const TCopyParams& copy_params) {
  auto stdlog =
      STDLOG(get_session_ptr(session), "get_first_geo_file_in_archive", archive_path_in);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());

  std::string archive_path(archive_path_in);

  if (path_is_relative(archive_path)) {
    // assume relative paths are relative to data_path / mapd_import / <session>
    auto file_path = import_path_ / picosha2::hash256_hex_string(session) /
                     boost::filesystem::path(archive_path).filename();
    archive_path = file_path.string();
  }
  validate_import_file_path_if_local(archive_path);

  if (is_a_supported_archive_file(archive_path)) {
    // find the archive file
    add_vsi_network_prefix(archive_path);
    if (!import_export::Importer::gdalFileExists(archive_path,
                                                 thrift_to_copyparams(copy_params))) {
      THROW_MAPD_EXCEPTION("Archive does not exist: " + archive_path_in);
    }
    // find geo file in archive
    add_vsi_archive_prefix(archive_path);
    std::string geo_file =
        find_first_geo_file_in_archive(archive_path, thrift_to_copyparams(copy_params));
    // what did we get?
    if (geo_file.size()) {
      // prepend it with the original path
      _return = archive_path_in + std::string("/") + geo_file;
    } else {
      // just return the original path
      _return = archive_path_in;
    }
  } else {
    // just return the original path
    _return = archive_path_in;
  }
}

void DBHandler::get_all_files_in_archive(std::vector<std::string>& _return,
                                         const TSessionId& session,
                                         const std::string& archive_path_in,
                                         const TCopyParams& copy_params) {
  auto stdlog =
      STDLOG(get_session_ptr(session), "get_all_files_in_archive", archive_path_in);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());

  std::string archive_path(archive_path_in);
  if (path_is_relative(archive_path)) {
    // assume relative paths are relative to data_path / mapd_import / <session>
    auto file_path = import_path_ / picosha2::hash256_hex_string(session) /
                     boost::filesystem::path(archive_path).filename();
    archive_path = file_path.string();
  }
  validate_import_file_path_if_local(archive_path);

  if (is_a_supported_archive_file(archive_path)) {
    // find the archive file
    add_vsi_network_prefix(archive_path);
    if (!import_export::Importer::gdalFileExists(archive_path,
                                                 thrift_to_copyparams(copy_params))) {
      THROW_MAPD_EXCEPTION("Archive does not exist: " + archive_path_in);
    }
    // find all files in archive
    add_vsi_archive_prefix(archive_path);
    _return = import_export::Importer::gdalGetAllFilesInArchive(
        archive_path, thrift_to_copyparams(copy_params));
    // prepend them all with original path
    for (auto& s : _return) {
      s = archive_path_in + '/' + s;
    }
  }
}

void DBHandler::get_layers_in_geo_file(std::vector<TGeoFileLayerInfo>& _return,
                                       const TSessionId& session,
                                       const std::string& file_name_in,
                                       const TCopyParams& cp) {
  auto stdlog = STDLOG(get_session_ptr(session), "get_layers_in_geo_file", file_name_in);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());

  std::string file_name(file_name_in);

  import_export::CopyParams copy_params = thrift_to_copyparams(cp);

  // handle relative paths
  if (path_is_relative(file_name)) {
    // assume relative paths are relative to data_path / mapd_import / <session>
    auto file_path = import_path_ / picosha2::hash256_hex_string(session) /
                     boost::filesystem::path(file_name).filename();
    file_name = file_path.string();
  }
  validate_import_file_path_if_local(file_name);

  // validate file_name
  if (is_a_supported_geo_file(file_name, true)) {
    // prepare to load geo file directly
    add_vsi_network_prefix(file_name);
    add_vsi_geo_prefix(file_name);
  } else if (is_a_supported_archive_file(file_name)) {
    // find the archive file
    add_vsi_network_prefix(file_name);
    if (!import_export::Importer::gdalFileExists(file_name, copy_params)) {
      THROW_MAPD_EXCEPTION("Archive does not exist: " + file_name_in);
    }
    // find geo file in archive
    add_vsi_archive_prefix(file_name);
    std::string geo_file = find_first_geo_file_in_archive(file_name, copy_params);
    // prepare to load that geo file
    if (geo_file.size()) {
      file_name = file_name + std::string("/") + geo_file;
    }
  } else {
    THROW_MAPD_EXCEPTION("File is not a supported geo or geo archive file: " +
                         file_name_in);
  }

  // check the file actually exists
  if (!import_export::Importer::gdalFileOrDirectoryExists(file_name, copy_params)) {
    THROW_MAPD_EXCEPTION("Geo file/archive does not exist: " + file_name_in);
  }

  // find all layers
  auto internal_layer_info =
      import_export::Importer::gdalGetLayersInGeoFile(file_name, copy_params);

  // convert to Thrift type
  for (const auto& internal_layer : internal_layer_info) {
    TGeoFileLayerInfo layer;
    layer.name = internal_layer.name;
    switch (internal_layer.contents) {
      case import_export::Importer::GeoFileLayerContents::EMPTY:
        layer.contents = TGeoFileLayerContents::EMPTY;
        break;
      case import_export::Importer::GeoFileLayerContents::GEO:
        layer.contents = TGeoFileLayerContents::GEO;
        break;
      case import_export::Importer::GeoFileLayerContents::NON_GEO:
        layer.contents = TGeoFileLayerContents::NON_GEO;
        break;
      case import_export::Importer::GeoFileLayerContents::UNSUPPORTED_GEO:
        layer.contents = TGeoFileLayerContents::UNSUPPORTED_GEO;
        break;
      default:
        CHECK(false);
    }
    _return.emplace_back(layer);  // no suitable constructor to just pass parameters
  }
}

void DBHandler::start_heap_profile(const TSessionId& session) {
  auto stdlog = STDLOG(get_session_ptr(session));
#ifdef HAVE_PROFILER
  if (IsHeapProfilerRunning()) {
    THROW_MAPD_EXCEPTION("Profiler already started");
  }
  HeapProfilerStart("omnisci");
#else
  THROW_MAPD_EXCEPTION("Profiler not enabled");
#endif  // HAVE_PROFILER
}

void DBHandler::stop_heap_profile(const TSessionId& session) {
  auto stdlog = STDLOG(get_session_ptr(session));
#ifdef HAVE_PROFILER
  if (!IsHeapProfilerRunning()) {
    THROW_MAPD_EXCEPTION("Profiler not running");
  }
  HeapProfilerStop();
#else
  THROW_MAPD_EXCEPTION("Profiler not enabled");
#endif  // HAVE_PROFILER
}

void DBHandler::get_heap_profile(std::string& profile, const TSessionId& session) {
  auto stdlog = STDLOG(get_session_ptr(session));
#ifdef HAVE_PROFILER
  if (!IsHeapProfilerRunning()) {
    THROW_MAPD_EXCEPTION("Profiler not running");
  }
  auto profile_buff = GetHeapProfile();
  profile = profile_buff;
  free(profile_buff);
#else
  THROW_MAPD_EXCEPTION("Profiler not enabled");
#endif  // HAVE_PROFILER
}

// NOTE: Only call check_session_exp_unsafe() when you hold a lock on sessions_mutex_.
void DBHandler::check_session_exp_unsafe(const SessionMap::iterator& session_it) {
  if (session_it->second.use_count() > 2 ||
      isInMemoryCalciteSession(session_it->second->get_currentUser())) {
    // SessionInfo is being used in more than one active operation. Original copy + one
    // stored in StdLog. Skip the checks.
    return;
  }
  time_t last_used_time = session_it->second->get_last_used_time();
  time_t start_time = session_it->second->get_start_time();
  const auto current_session_duration = time(0) - last_used_time;
  if (current_session_duration > idle_session_duration_) {
    LOG(INFO) << "Session " << session_it->second->get_public_session_id()
              << " idle duration " << current_session_duration
              << " seconds exceeds maximum idle duration " << idle_session_duration_
              << " seconds. Invalidating session.";
    throw ForceDisconnect("Idle Session Timeout. User should re-authenticate.");
  }
  const auto total_session_duration = time(0) - start_time;
  if (total_session_duration > max_session_duration_) {
    LOG(INFO) << "Session " << session_it->second->get_public_session_id()
              << " total duration " << total_session_duration
              << " seconds exceeds maximum total session duration "
              << max_session_duration_ << " seconds. Invalidating session.";
    throw ForceDisconnect("Maximum active Session Timeout. User should re-authenticate.");
  }
}

std::shared_ptr<const Catalog_Namespace::SessionInfo> DBHandler::get_const_session_ptr(
    const TSessionId& session) {
  return get_session_ptr(session);
}

Catalog_Namespace::SessionInfo DBHandler::get_session_copy(const TSessionId& session) {
  mapd_shared_lock<mapd_shared_mutex> read_lock(sessions_mutex_);
  return *get_session_it_unsafe(session, read_lock)->second;
}

std::shared_ptr<Catalog_Namespace::SessionInfo> DBHandler::get_session_copy_ptr(
    const TSessionId& session) {
  // Note(Wamsi): We have `get_const_session_ptr` which would return as const
  // SessionInfo stored in the map. You can use `get_const_session_ptr` instead of the
  // copy of SessionInfo but beware that it can be changed in teh map. So if you do not
  // care about the changes then use `get_const_session_ptr` if you do then use this
  // function to get a copy. We should eventually aim to merge both
  // `get_const_session_ptr` and `get_session_copy_ptr`.
  mapd_shared_lock<mapd_shared_mutex> read_lock(sessions_mutex_);
  auto& session_info_ref = *get_session_it_unsafe(session, read_lock)->second;
  return std::make_shared<Catalog_Namespace::SessionInfo>(session_info_ref);
}

std::shared_ptr<Catalog_Namespace::SessionInfo> DBHandler::get_session_ptr(
    const TSessionId& session_id) {
  // Note(Wamsi): This method will give you a shared_ptr to master SessionInfo itself.
  // Should be used only when you need to make updates to original SessionInfo object.
  // Currently used by `update_session_last_used_duration`

  // 1) `session_id` will be empty during intial connect. 2)`sessionmapd iterator` will
  // be invalid during disconnect. SessionInfo will be erased from map by the time it
  // reaches here. In both the above cases, we would return `nullptr` and can skip
  // SessionInfo updates.
  if (session_id.empty()) {
    return {};
  }
  mapd_shared_lock<mapd_shared_mutex> read_lock(sessions_mutex_);
  return get_session_it_unsafe(session_id, read_lock)->second;
}

void DBHandler::check_table_load_privileges(
    const Catalog_Namespace::SessionInfo& session_info,
    const std::string& table_name) {
  auto user_metadata = session_info.get_currentUser();
  auto& cat = session_info.getCatalog();
  DBObject dbObject(table_name, TableDBObjectType);
  dbObject.loadKey(cat);
  dbObject.setPrivileges(AccessPrivileges::INSERT_INTO_TABLE);
  std::vector<DBObject> privObjects;
  privObjects.push_back(dbObject);
  if (!SysCatalog::instance().checkPrivileges(user_metadata, privObjects)) {
    THROW_MAPD_EXCEPTION("Violation of access privileges: user " +
                         user_metadata.userLoggable() +
                         " has no insert privileges for table " + table_name + ".");
  }
}

void DBHandler::check_table_load_privileges(const TSessionId& session,
                                            const std::string& table_name) {
  const auto session_info = get_session_copy(session);
  check_table_load_privileges(session_info, table_name);
}

void DBHandler::set_execution_mode_nolock(Catalog_Namespace::SessionInfo* session_ptr,
                                          const TExecuteMode::type mode) {
  const std::string user_name = session_ptr->get_currentUser().userLoggable();
  switch (mode) {
    case TExecuteMode::GPU:
      if (cpu_mode_only_) {
        TOmniSciException e;
        e.error_msg = "Cannot switch to GPU mode in a server started in CPU-only mode.";
        throw e;
      }
      session_ptr->set_executor_device_type(ExecutorDeviceType::GPU);
      LOG(INFO) << "User " << user_name << " sets GPU mode.";
      break;
    case TExecuteMode::CPU:
      session_ptr->set_executor_device_type(ExecutorDeviceType::CPU);
      LOG(INFO) << "User " << user_name << " sets CPU mode.";
      break;
  }
}

std::vector<PushedDownFilterInfo> DBHandler::execute_rel_alg(
    ExecutionResult& _return,
    QueryStateProxy query_state_proxy,
    const std::string& query_ra,
    const bool column_format,
    const ExecutorDeviceType executor_device_type,
    const int32_t first_n,
    const int32_t at_most_n,
    const bool just_validate,
    const bool find_push_down_candidates,
    const ExplainInfo& explain_info,
    const std::optional<size_t> executor_index) const {
  query_state::Timer timer = query_state_proxy.createTimer(__func__);

  VLOG(1) << "Table Schema Locks:\n" << lockmgr::TableSchemaLockMgr::instance();
  VLOG(1) << "Table Data Locks:\n" << lockmgr::TableDataLockMgr::instance();

  const auto& cat = query_state_proxy.getQueryState().getConstSessionInfo()->getCatalog();
  auto executor = Executor::getExecutor(
      executor_index ? *executor_index : Executor::UNITARY_EXECUTOR_ID,
      jit_debug_ ? "/tmp" : "",
      jit_debug_ ? "mapdquery" : "",
      system_parameters_);
  RelAlgExecutor ra_executor(executor.get(),
                             cat,
                             query_ra,
                             query_state_proxy.getQueryState().shared_from_this());
  // handle hints
  const auto& query_hints = ra_executor.getParsedQueryHints();
  const bool cpu_mode_enabled = query_hints.isHintRegistered(QueryHint::kCpuMode);
  CompilationOptions co = {
      cpu_mode_enabled ? ExecutorDeviceType::CPU : executor_device_type,
      /*hoist_literals=*/true,
      ExecutorOptLevel::Default,
      g_enable_dynamic_watchdog,
      /*allow_lazy_fetch=*/true,
      /*filter_on_deleted_column=*/true,
      explain_info.explain_optimized ? ExecutorExplainType::Optimized
                                     : ExecutorExplainType::Default,
      intel_jit_profile_};
  auto validate_or_explain_query =
      explain_info.justExplain() || explain_info.justCalciteExplain() || just_validate;
  ExecutionOptions eo = {g_enable_columnar_output,
                         allow_multifrag_,
                         explain_info.justExplain(),
                         allow_loop_joins_ || just_validate,
                         g_enable_watchdog,
                         jit_debug_,
                         just_validate,
                         g_enable_dynamic_watchdog,
                         g_dynamic_watchdog_time_limit,
                         find_push_down_candidates,
                         explain_info.justCalciteExplain(),
                         system_parameters_.gpu_input_mem_limit,
                         g_enable_runtime_query_interrupt && !validate_or_explain_query &&
                             !query_state_proxy.getQueryState()
                                  .getConstSessionInfo()
                                  ->get_session_id()
                                  .empty(),
                         g_running_query_interrupt_freq,
                         g_pending_query_interrupt_freq};
  auto execution_time_ms = _return.getExecutionTime() + measure<>::execution([&]() {
                             _return = ra_executor.executeRelAlgQuery(
                                 co, eo, explain_info.explain_plan, nullptr);
                           });
  // reduce execution time by the time spent during queue waiting
  const auto rs = _return.getRows();
  if (rs) {
    execution_time_ms -= rs->getQueueTime();
  }
  _return.setExecutionTime(execution_time_ms);
  VLOG(1) << cat.getDataMgr().getSystemMemoryUsage();
  const auto& filter_push_down_info = _return.getPushedDownFilterInfo();
  if (!filter_push_down_info.empty()) {
    return filter_push_down_info;
  }
  if (explain_info.justExplain()) {
    _return.setResultType(ExecutionResult::Explaination);
  } else if (!explain_info.justCalciteExplain()) {
    _return.setResultType(ExecutionResult::QueryResult);
  }
  return {};
}

void DBHandler::execute_rel_alg_df(TDataFrame& _return,
                                   const std::string& query_ra,
                                   QueryStateProxy query_state_proxy,
                                   const Catalog_Namespace::SessionInfo& session_info,
                                   const ExecutorDeviceType device_type,
                                   const size_t device_id,
                                   const int32_t first_n,
                                   const TArrowTransport::type transport_method) const {
  const auto& cat = session_info.getCatalog();
  CHECK(device_type == ExecutorDeviceType::CPU ||
        session_info.get_executor_device_type() == ExecutorDeviceType::GPU);
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID,
                                        jit_debug_ ? "/tmp" : "",
                                        jit_debug_ ? "mapdquery" : "",
                                        system_parameters_);
  RelAlgExecutor ra_executor(executor.get(),
                             cat,
                             query_ra,
                             query_state_proxy.getQueryState().shared_from_this());
  const auto& query_hints = ra_executor.getParsedQueryHints();
  const bool cpu_mode_enabled = query_hints.isHintRegistered(QueryHint::kCpuMode);
  CompilationOptions co = {cpu_mode_enabled ? ExecutorDeviceType::CPU : device_type,
                           /*hoist_literals=*/true,
                           ExecutorOptLevel::Default,
                           g_enable_dynamic_watchdog,
                           /*allow_lazy_fetch=*/true,
                           /*filter_on_deleted_column=*/true,
                           ExecutorExplainType::Default,
                           intel_jit_profile_};
  ExecutionOptions eo = {
      g_enable_columnar_output,
      allow_multifrag_,
      false,
      allow_loop_joins_,
      g_enable_watchdog,
      jit_debug_,
      false,
      g_enable_dynamic_watchdog,
      g_dynamic_watchdog_time_limit,
      false,
      false,
      system_parameters_.gpu_input_mem_limit,
      g_enable_runtime_query_interrupt && !query_state_proxy.getQueryState()
                                               .getConstSessionInfo()
                                               ->get_session_id()
                                               .empty(),
      g_running_query_interrupt_freq,
      g_pending_query_interrupt_freq};
  ExecutionResult result{std::make_shared<ResultSet>(std::vector<TargetInfo>{},
                                                     ExecutorDeviceType::CPU,
                                                     QueryMemoryDescriptor(),
                                                     nullptr,
                                                     nullptr,
                                                     0,
                                                     0),
                         {}};
  _return.execution_time_ms += measure<>::execution(
      [&]() { result = ra_executor.executeRelAlgQuery(co, eo, false, nullptr); });
  _return.execution_time_ms -= result.getRows()->getQueueTime();
  const auto rs = result.getRows();
  const auto converter =
      std::make_unique<ArrowResultSetConverter>(rs,
                                                data_mgr_,
                                                device_type,
                                                device_id,
                                                getTargetNames(result.getTargetsMeta()),
                                                first_n,
                                                ArrowTransport(transport_method));
  ArrowResult arrow_result;
  _return.arrow_conversion_time_ms +=
      measure<>::execution([&] { arrow_result = converter->getArrowResult(); });
  _return.sm_handle =
      std::string(arrow_result.sm_handle.begin(), arrow_result.sm_handle.end());
  _return.sm_size = arrow_result.sm_size;
  _return.df_handle =
      std::string(arrow_result.df_handle.begin(), arrow_result.df_handle.end());
  _return.df_buffer =
      std::string(arrow_result.df_buffer.begin(), arrow_result.df_buffer.end());
  if (device_type == ExecutorDeviceType::GPU) {
    std::lock_guard<std::mutex> map_lock(handle_to_dev_ptr_mutex_);
    CHECK(!ipc_handle_to_dev_ptr_.count(_return.df_handle));
    ipc_handle_to_dev_ptr_.insert(
        std::make_pair(_return.df_handle, arrow_result.serialized_cuda_handle));
  }
  _return.df_size = arrow_result.df_size;
}

std::vector<TargetMetaInfo> DBHandler::getTargetMetaInfo(
    const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& targets) const {
  std::vector<TargetMetaInfo> result;
  for (const auto& target : targets) {
    CHECK(target);
    CHECK(target->get_expr());
    result.emplace_back(target->get_resname(), target->get_expr()->get_type_info());
  }
  return result;
}

std::vector<std::string> DBHandler::getTargetNames(
    const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& targets) const {
  std::vector<std::string> names;
  for (const auto& target : targets) {
    CHECK(target);
    CHECK(target->get_expr());
    names.push_back(target->get_resname());
  }
  return names;
}

std::vector<std::string> DBHandler::getTargetNames(
    const std::vector<TargetMetaInfo>& targets) const {
  std::vector<std::string> names;
  for (const auto& target : targets) {
    names.push_back(target.get_resname());
  }
  return names;
}

void DBHandler::convertRows(TQueryResult& _return,
                            QueryStateProxy query_state_proxy,
                            const std::vector<TargetMetaInfo>& targets,
                            const ResultSet& results,
                            const bool column_format,
                            const int32_t first_n,
                            const int32_t at_most_n) {
  query_state::Timer timer = query_state_proxy.createTimer(__func__);
  _return.row_set.row_desc = ThriftSerializers::target_meta_infos_to_thrift(targets);
  int32_t fetched{0};
  if (column_format) {
    _return.row_set.is_columnar = true;
    std::vector<TColumn> tcolumns(results.colCount());
    while (first_n == -1 || fetched < first_n) {
      const auto crt_row = results.getNextRow(true, true);
      if (crt_row.empty()) {
        break;
      }
      ++fetched;
      if (at_most_n >= 0 && fetched > at_most_n) {
        THROW_MAPD_EXCEPTION("The result contains more rows than the specified cap of " +
                             std::to_string(at_most_n));
      }
      for (size_t i = 0; i < results.colCount(); ++i) {
        const auto agg_result = crt_row[i];
        value_to_thrift_column(agg_result, targets[i].get_type_info(), tcolumns[i]);
      }
    }
    for (size_t i = 0; i < results.colCount(); ++i) {
      _return.row_set.columns.push_back(tcolumns[i]);
    }
  } else {
    _return.row_set.is_columnar = false;
    while (first_n == -1 || fetched < first_n) {
      const auto crt_row = results.getNextRow(true, true);
      if (crt_row.empty()) {
        break;
      }
      ++fetched;
      if (at_most_n >= 0 && fetched > at_most_n) {
        THROW_MAPD_EXCEPTION("The result contains more rows than the specified cap of " +
                             std::to_string(at_most_n));
      }
      TRow trow;
      trow.cols.reserve(results.colCount());
      for (size_t i = 0; i < results.colCount(); ++i) {
        const auto agg_result = crt_row[i];
        trow.cols.push_back(value_to_thrift(agg_result, targets[i].get_type_info()));
      }
      _return.row_set.rows.push_back(trow);
    }
  }
}

TRowDescriptor DBHandler::fixup_row_descriptor(const TRowDescriptor& row_desc,
                                               const Catalog& cat) {
  TRowDescriptor fixedup_row_desc;
  for (const TColumnType& col_desc : row_desc) {
    auto fixedup_col_desc = col_desc;
    if (col_desc.col_type.encoding == TEncodingType::DICT &&
        col_desc.col_type.comp_param > 0) {
      const auto dd = cat.getMetadataForDict(col_desc.col_type.comp_param, false);
      fixedup_col_desc.col_type.comp_param = dd->dictNBits;
    }
    fixedup_row_desc.push_back(fixedup_col_desc);
  }

  return fixedup_row_desc;
}

// create simple result set to return a single column result
void DBHandler::createSimpleResult(TQueryResult& _return,
                                   const ResultSet& results,
                                   const bool column_format,
                                   const std::string label) {
  CHECK_EQ(size_t(1), results.rowCount());
  TColumnType proj_info;
  proj_info.col_name = label;
  proj_info.col_type.type = TDatumType::STR;
  proj_info.col_type.nullable = false;
  proj_info.col_type.is_array = false;
  _return.row_set.row_desc.push_back(proj_info);
  const auto crt_row = results.getNextRow(true, true);
  const auto tv = crt_row[0];
  CHECK(results.getNextRow(true, true).empty());
  const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
  CHECK(scalar_tv);
  const auto s_n = boost::get<NullableString>(scalar_tv);
  CHECK(s_n);
  const auto s = boost::get<std::string>(s_n);
  CHECK(s);
  if (column_format) {
    TColumn tcol;
    tcol.data.str_col.push_back(*s);
    tcol.nulls.push_back(false);
    _return.row_set.is_columnar = true;
    _return.row_set.columns.push_back(tcol);
  } else {
    TDatum explanation;
    explanation.val.str_val = *s;
    explanation.is_null = false;
    TRow trow;
    trow.cols.push_back(explanation);
    _return.row_set.is_columnar = false;
    _return.row_set.rows.push_back(trow);
  }
}

void DBHandler::convertExplain(TQueryResult& _return,
                               const ResultSet& results,
                               const bool column_format) {
  createSimpleResult(_return, results, column_format, "Explanation");
}

void DBHandler::convertResult(TQueryResult& _return,
                              const ResultSet& results,
                              const bool column_format) {
  createSimpleResult(_return, results, column_format, "Result");
}

// this all should be moved out of here to catalog
bool DBHandler::user_can_access_table(const Catalog_Namespace::SessionInfo& session_info,
                                      const TableDescriptor* td,
                                      const AccessPrivileges access_priv) {
  CHECK(td);
  auto& cat = session_info.getCatalog();
  std::vector<DBObject> privObjects;
  DBObject dbObject(td->tableName, TableDBObjectType);
  dbObject.loadKey(cat);
  dbObject.setPrivileges(access_priv);
  privObjects.push_back(dbObject);
  return SysCatalog::instance().checkPrivileges(session_info.get_currentUser(),
                                                privObjects);
};

void DBHandler::check_and_invalidate_sessions(Parser::DDLStmt* ddl) {
  const auto drop_db_stmt = dynamic_cast<Parser::DropDBStmt*>(ddl);
  if (drop_db_stmt) {
    invalidate_sessions(*drop_db_stmt->getDatabaseName(), drop_db_stmt);
    return;
  }
  const auto rename_db_stmt = dynamic_cast<Parser::RenameDatabaseStmt*>(ddl);
  if (rename_db_stmt) {
    invalidate_sessions(*rename_db_stmt->getPreviousDatabaseName(), rename_db_stmt);
    return;
  }
  const auto drop_user_stmt = dynamic_cast<Parser::DropUserStmt*>(ddl);
  if (drop_user_stmt) {
    invalidate_sessions(*drop_user_stmt->getUserName(), drop_user_stmt);
    return;
  }
  const auto rename_user_stmt = dynamic_cast<Parser::RenameUserStmt*>(ddl);
  if (rename_user_stmt) {
    invalidate_sessions(*rename_user_stmt->getOldUserName(), rename_user_stmt);
    return;
  }
}

void DBHandler::sql_execute_impl(ExecutionResult& _return,
                                 QueryStateProxy query_state_proxy,
                                 const bool column_format,
                                 const ExecutorDeviceType executor_device_type,
                                 const int32_t first_n,
                                 const int32_t at_most_n,
                                 const bool use_calcite) {
  if (leaf_handler_) {
    leaf_handler_->flush_queue();
  }
  auto const query_str = strip(query_state_proxy.getQueryState().getQueryStr());
  auto session_ptr = query_state_proxy.getQueryState().getConstSessionInfo();
  // Call to DistributedValidate() below may change cat.
  auto& cat = session_ptr->getCatalog();
  std::list<std::unique_ptr<Parser::Stmt>> parse_trees;

  mapd_unique_lock<mapd_shared_mutex> executeWriteLock;
  mapd_shared_lock<mapd_shared_mutex> executeReadLock;

  lockmgr::LockedTableDescriptors locks;
  ParserWrapper pw{query_str};

  if (pw.is_itas) {
    // itas can attempt to execute here
    check_read_only("insert_into_table");

    std::string query_ra;
    _return.addExecutionTime(measure<>::execution([&]() {
      TPlanResult result;
      std::tie(result, locks) =
          parse_to_ra(query_state_proxy, query_str, {}, false, system_parameters_);
      query_ra = result.plan_result;
    }));
    rapidjson::Document ddl_query;
    ddl_query.Parse(query_ra);
    CHECK(ddl_query.HasMember("payload"));
    CHECK(ddl_query["payload"].IsObject());
    auto stmt = Parser::InsertIntoTableAsSelectStmt(ddl_query["payload"].GetObject());
    _return.addExecutionTime(measure<>::execution([&]() { stmt.execute(*session_ptr); }));
    return;

  } else if (pw.is_ctas) {
    // ctas can attempt to execute here
    check_read_only("create_table_as");

    std::string query_ra;
    _return.addExecutionTime(measure<>::execution([&]() {
      TPlanResult result;
      std::tie(result, locks) =
          parse_to_ra(query_state_proxy, query_str, {}, false, system_parameters_);
      query_ra = result.plan_result;
    }));
    if (query_ra.size()) {
      rapidjson::Document ddl_query;
      ddl_query.Parse(query_ra);
      CHECK(ddl_query.HasMember("payload"));
      CHECK(ddl_query["payload"].IsObject());
      auto stmt = Parser::CreateTableAsSelectStmt(ddl_query["payload"].GetObject());
      _return.addExecutionTime(
          measure<>::execution([&]() { stmt.execute(*session_ptr); }));
    }
    return;

  } else if (pw.isCalcitePathPermissable(read_only_)) {
    // run DDL before the locks as DDL statements should handle their own locking
    if (pw.isCalciteDdl()) {
      std::string query_ra;
      _return.addExecutionTime(measure<>::execution([&]() {
        TPlanResult result;
        std::tie(result, locks) =
            parse_to_ra(query_state_proxy, query_str, {}, false, system_parameters_);
        query_ra = result.plan_result;
      }));
      executeDdl(_return, query_ra, session_ptr);
      return;
    }

    executeReadLock = mapd_shared_lock<mapd_shared_mutex>(
        *legacylockmgr::LockMgr<mapd_shared_mutex, bool>::getMutex(
            legacylockmgr::ExecutorOuterLock, true));

    std::string query_ra = query_str;
    if (use_calcite) {
      _return.addExecutionTime(measure<>::execution([&]() {
        TPlanResult result;
        std::tie(result, locks) =
            parse_to_ra(query_state_proxy, query_str, {}, true, system_parameters_);
        query_ra = result.plan_result;
      }));
    }
    std::string query_ra_calcite_explain;
    if (pw.isCalciteExplain() && (!g_enable_filter_push_down || g_cluster)) {
      // return the ra as the result
      _return.updateResultSet(query_ra, ExecutionResult::Explaination);
      return;
    } else if (pw.isCalciteExplain()) {
      // removing the "explain calcite " from the beginning of the "query_str":
      std::string temp_query_str =
          query_str.substr(std::string("explain calcite ").length());
      CHECK(!locks.empty());
      query_ra_calcite_explain =
          parse_to_ra(query_state_proxy, temp_query_str, {}, false, system_parameters_)
              .first.plan_result;
    }
    const auto explain_info = pw.getExplainInfo();
    std::vector<PushedDownFilterInfo> filter_push_down_requests;
    auto execute_rel_alg_task = std::make_shared<QueryDispatchQueue::Task>(
        [this,
         &filter_push_down_requests,
         &_return,
         &query_state_proxy,
         &explain_info,
         &query_ra_calcite_explain,
         &query_ra,
         &query_str,
         &locks,
         column_format,
         executor_device_type,
         first_n,
         at_most_n](const size_t executor_index) {
          // if we find proper filters we need to "re-execute" the query
          // with a modified query plan (i.e., which has pushdowned filter)
          // otherwise this trial just executes the query and keeps corresponding query
          // resultset in _return object
          filter_push_down_requests = execute_rel_alg(
              _return,
              query_state_proxy,
              explain_info.justCalciteExplain() ? query_ra_calcite_explain : query_ra,
              column_format,
              executor_device_type,
              first_n,
              at_most_n,
              /*just_validate=*/false,
              g_enable_filter_push_down && !g_cluster,
              explain_info,
              executor_index);
          if (explain_info.justCalciteExplain()) {
            if (filter_push_down_requests.empty()) {
              // we only reach here if filter push down was enabled, but no filter
              // push down candidate was found
              _return.updateResultSet(query_ra, ExecutionResult::Explaination);
            } else {
              CHECK(!locks.empty());
              std::vector<TFilterPushDownInfo> filter_push_down_info;
              for (const auto& req : filter_push_down_requests) {
                TFilterPushDownInfo filter_push_down_info_for_request;
                filter_push_down_info_for_request.input_prev = req.input_prev;
                filter_push_down_info_for_request.input_start = req.input_start;
                filter_push_down_info_for_request.input_next = req.input_next;
                filter_push_down_info.push_back(filter_push_down_info_for_request);
              }
              query_ra = parse_to_ra(query_state_proxy,
                                     query_str,
                                     filter_push_down_info,
                                     false,
                                     system_parameters_)
                             .first.plan_result;
              _return.updateResultSet(query_ra, ExecutionResult::Explaination);
            }
          } else {
            if (!filter_push_down_requests.empty()) {
              CHECK(!locks.empty());
              execute_rel_alg_with_filter_push_down(_return,
                                                    query_state_proxy,
                                                    query_ra,
                                                    column_format,
                                                    executor_device_type,
                                                    first_n,
                                                    at_most_n,
                                                    explain_info.justExplain(),
                                                    explain_info.justCalciteExplain(),
                                                    filter_push_down_requests);
            }
          }
        });
    CHECK(dispatch_queue_);
    if (g_enable_runtime_query_interrupt && !session_ptr->get_session_id().empty()) {
      auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
      auto submitted_time_str = query_state_proxy.getQueryState().getQuerySubmittedTime();
      executor->enrollQuerySession(session_ptr->get_session_id(),
                                   query_str,
                                   submitted_time_str,
                                   Executor::UNITARY_EXECUTOR_ID,
                                   QuerySessionStatus::QueryStatus::PENDING_QUEUE);
    }
    dispatch_queue_->submit(execute_rel_alg_task,
                            pw.getDMLType() == ParserWrapper::DMLType::Update ||
                                pw.getDMLType() == ParserWrapper::DMLType::Delete);
    auto result_future = execute_rel_alg_task->get_future();
    result_future.get();
    return;
  } else if (pw.is_optimize || pw.is_validate) {
    // Get the Stmt object
    DBHandler::parser_with_error_handler(query_str, parse_trees);

    if (pw.is_optimize) {
      const auto optimize_stmt =
          dynamic_cast<Parser::OptimizeTableStmt*>(parse_trees.front().get());
      CHECK(optimize_stmt);

      _return.addExecutionTime(measure<>::execution([&]() {
        const auto td_with_lock =
            lockmgr::TableSchemaLockContainer<lockmgr::WriteLock>::acquireTableDescriptor(
                cat, optimize_stmt->getTableName());
        const auto td = td_with_lock();

        if (!td || !user_can_access_table(
                       *session_ptr, td, AccessPrivileges::DELETE_FROM_TABLE)) {
          throw std::runtime_error("Table " + optimize_stmt->getTableName() +
                                   " does not exist.");
        }
        if (td->isView) {
          throw std::runtime_error("OPTIMIZE TABLE command is not supported on views.");
        }

        auto executor = Executor::getExecutor(
            Executor::UNITARY_EXECUTOR_ID, "", "", system_parameters_);
        const TableOptimizer optimizer(td, executor.get(), cat);
        if (optimize_stmt->shouldVacuumDeletedRows()) {
          optimizer.vacuumDeletedRows();
        }
        optimizer.recomputeMetadata();
      }));
      return;
    }
    if (pw.is_validate) {
      // check user is superuser
      if (!session_ptr->get_currentUser().isSuper) {
        throw std::runtime_error("Superuser is required to run VALIDATE");
      }
      const auto validate_stmt =
          dynamic_cast<Parser::ValidateStmt*>(parse_trees.front().get());
      CHECK(validate_stmt);

      // Prevent any other query from running while doing validate
      executeWriteLock = mapd_unique_lock<mapd_shared_mutex>(
          *legacylockmgr::LockMgr<mapd_shared_mutex, bool>::getMutex(
              legacylockmgr::ExecutorOuterLock, true));

      std::string output{"Result for validate"};
      if (g_cluster) {
        if (leaf_aggregator_.leafCount()) {
          _return.addExecutionTime(measure<>::execution([&]() {
            const system_validator::DistributedValidate validator(
                validate_stmt->getType(),
                validate_stmt->isRepairTypeRemove(),
                cat,  // tables may be dropped here
                leaf_aggregator_,
                *session_ptr,
                *this);
            output = validator.validate(query_state_proxy);
          }));
        } else {
          THROW_MAPD_EXCEPTION("Validate command should be executed on the aggregator.");
        }
      } else {
        _return.addExecutionTime(measure<>::execution([&]() {
          const system_validator::SingleNodeValidator validator(validate_stmt->getType(),
                                                                cat);
          output = validator.validate();
        }));
      }
      _return.updateResultSet(output, ExecutionResult::SimpleResult);
      return;
    }
  }

  LOG(INFO) << "passing query to legacy processor";
  const auto result = apply_copy_to_shim(query_str);
  DBHandler::parser_with_error_handler(result, parse_trees);
  auto handle_ddl = [&query_state_proxy, &session_ptr, &_return, &locks, this](
                        Parser::DDLStmt* ddl) -> bool {
    if (!ddl) {
      return false;
    }
    const auto show_create_stmt = dynamic_cast<Parser::ShowCreateTableStmt*>(ddl);
    if (show_create_stmt) {
      _return.addExecutionTime(
          measure<>::execution([&]() { ddl->execute(*session_ptr); }));
      const auto create_string = show_create_stmt->getCreateStmt();
      _return.updateResultSet(create_string, ExecutionResult::SimpleResult);
      return true;
    }

    const auto import_stmt = dynamic_cast<Parser::CopyTableStmt*>(ddl);
    if (import_stmt) {
      if (g_cluster && !leaf_aggregator_.leafCount()) {
        // Don't allow copy from imports directly on a leaf node
        throw std::runtime_error(
            "Cannot import on an individual leaf. Please import from the Aggregator.");
      } else if (leaf_aggregator_.leafCount() > 0) {
        _return.addExecutionTime(measure<>::execution(
            [&]() { execute_distributed_copy_statement(import_stmt, *session_ptr); }));
      } else {
        _return.addExecutionTime(
            measure<>::execution([&]() { ddl->execute(*session_ptr); }));
      }

      // Read response message
      _return.updateResultSet(*import_stmt->return_message.get(),
                              ExecutionResult::SimpleResult,
                              import_stmt->get_success());

      // get geo_copy_from info
      if (import_stmt->was_geo_copy_from()) {
        GeoCopyFromState geo_copy_from_state;
        import_stmt->get_geo_copy_from_payload(
            geo_copy_from_state.geo_copy_from_table,
            geo_copy_from_state.geo_copy_from_file_name,
            geo_copy_from_state.geo_copy_from_copy_params,
            geo_copy_from_state.geo_copy_from_partitions);
        geo_copy_from_sessions.add(session_ptr->get_session_id(), geo_copy_from_state);
      }
      return true;
    }

    // Check for DDL statements requiring locking and get locks
    auto export_stmt = dynamic_cast<Parser::ExportQueryStmt*>(ddl);
    if (export_stmt) {
      const auto query_string = export_stmt->get_select_stmt();
      TPlanResult result;
      CHECK(locks.empty());
      std::tie(result, locks) =
          parse_to_ra(query_state_proxy, query_string, {}, true, system_parameters_);
    }
    _return.addExecutionTime(measure<>::execution([&]() {
      ddl->execute(*session_ptr);
      check_and_invalidate_sessions(ddl);
    }));
    _return.setResultType(ExecutionResult::CalciteDdl);
    return true;
  };

  for (const auto& stmt : parse_trees) {
    if (DBHandler::read_only_) {
      // a limited set of commands are available in read-only mode
      auto select_stmt = dynamic_cast<Parser::SelectStmt*>(stmt.get());
      auto show_create_stmt = dynamic_cast<Parser::ShowCreateTableStmt*>(stmt.get());
      if (!select_stmt && !show_create_stmt) {
        THROW_MAPD_EXCEPTION(
            "Only read-only SQL commands are supported in read only mode.");
      }
    }

    auto ddl = dynamic_cast<Parser::DDLStmt*>(stmt.get());
    if (!handle_ddl(ddl)) {
      auto stmtp = dynamic_cast<Parser::InsertValuesStmt*>(stmt.get());
      CHECK(stmtp);  // no other statements supported
      if (parse_trees.size() != 1) {
        throw std::runtime_error("Can only run one INSERT INTO query at a time.");
      }
      _return.addExecutionTime(
          measure<>::execution([&]() { stmtp->execute(*session_ptr); }));
    }
  }
}

void DBHandler::execute_rel_alg_with_filter_push_down(
    ExecutionResult& _return,
    QueryStateProxy query_state_proxy,
    std::string& query_ra,
    const bool column_format,
    const ExecutorDeviceType executor_device_type,
    const int32_t first_n,
    const int32_t at_most_n,
    const bool just_explain,
    const bool just_calcite_explain,
    const std::vector<PushedDownFilterInfo>& filter_push_down_requests) {
  // collecting the selected filters' info to be sent to Calcite:
  std::vector<TFilterPushDownInfo> filter_push_down_info;
  for (const auto& req : filter_push_down_requests) {
    TFilterPushDownInfo filter_push_down_info_for_request;
    filter_push_down_info_for_request.input_prev = req.input_prev;
    filter_push_down_info_for_request.input_start = req.input_start;
    filter_push_down_info_for_request.input_next = req.input_next;
    filter_push_down_info.push_back(filter_push_down_info_for_request);
  }
  // deriving the new relational algebra plan with respect to the pushed down filters
  _return.addExecutionTime(measure<>::execution([&]() {
    query_ra = parse_to_ra(query_state_proxy,
                           query_state_proxy.getQueryState().getQueryStr(),
                           filter_push_down_info,
                           false,
                           system_parameters_)
                   .first.plan_result;
  }));

  if (just_calcite_explain) {
    // return the new ra as the result
    _return.updateResultSet(query_ra, ExecutionResult::Explaination);
    return;
  }

  // execute the new relational algebra plan:
  auto explain_info = ExplainInfo::defaults();
  explain_info.explain = just_explain;
  execute_rel_alg(_return,
                  query_state_proxy,
                  query_ra,
                  column_format,
                  executor_device_type,
                  first_n,
                  at_most_n,
                  /*just_validate=*/false,
                  /*find_push_down_candidates=*/false,
                  explain_info);
}

void DBHandler::execute_distributed_copy_statement(
    Parser::CopyTableStmt* copy_stmt,
    const Catalog_Namespace::SessionInfo& session_info) {
  auto importer_factory = [&session_info, this](
                              const Catalog& catalog,
                              const TableDescriptor* td,
                              const std::string& file_path,
                              const import_export::CopyParams& copy_params) {
    return std::make_unique<import_export::Importer>(
        new DistributedLoader(session_info, td, &leaf_aggregator_),
        file_path,
        copy_params);
  };
  copy_stmt->execute(session_info, importer_factory);
}

std::pair<TPlanResult, lockmgr::LockedTableDescriptors> DBHandler::parse_to_ra(
    QueryStateProxy query_state_proxy,
    const std::string& query_str,
    const std::vector<TFilterPushDownInfo>& filter_push_down_info,
    const bool acquire_locks,
    const SystemParameters& system_parameters,
    bool check_privileges) {
  query_state::Timer timer = query_state_proxy.createTimer(__func__);
  ParserWrapper pw{query_str};
  const std::string actual_query{pw.isSelectExplain() ? pw.actual_query : query_str};
  TPlanResult result;

  if (pw.isCalcitePathPermissable(read_only_)) {
    auto cat = query_state_proxy.getQueryState().getConstSessionInfo()->get_catalog_ptr();
    auto session_cleanup_handler = [&](const auto& session_id) {
      removeInMemoryCalciteSession(session_id);
    };
    auto process_calcite_request = [&] {
      const auto& in_memory_session_id = createInMemoryCalciteSession(cat);
      try {
        result = calcite_->process(timer.createQueryStateProxy(),
                                   legacy_syntax_ ? pg_shim(actual_query) : actual_query,
                                   filter_push_down_info,
                                   legacy_syntax_,
                                   pw.isCalciteExplain(),
                                   system_parameters.enable_calcite_view_optimize,
                                   check_privileges,
                                   in_memory_session_id);
        session_cleanup_handler(in_memory_session_id);
      } catch (std::exception&) {
        session_cleanup_handler(in_memory_session_id);
        throw;
      }
    };
    process_calcite_request();
    lockmgr::LockedTableDescriptors locks;
    if (acquire_locks) {
      std::set<std::vector<std::string>> write_only_tables;
      std::vector<std::vector<std::string>> tables;

      tables.insert(tables.end(),
                    result.resolved_accessed_objects.tables_updated_in.begin(),
                    result.resolved_accessed_objects.tables_updated_in.end());
      tables.insert(tables.end(),
                    result.resolved_accessed_objects.tables_deleted_from.begin(),
                    result.resolved_accessed_objects.tables_deleted_from.end());

      // Collect the tables that need a write lock
      for (const auto& table : tables) {
        write_only_tables.insert(table);
      }

      tables.insert(tables.end(),
                    result.resolved_accessed_objects.tables_selected_from.begin(),
                    result.resolved_accessed_objects.tables_selected_from.end());
      tables.insert(tables.end(),
                    result.resolved_accessed_objects.tables_inserted_into.begin(),
                    result.resolved_accessed_objects.tables_inserted_into.end());

      // avoid deadlocks by enforcing a deterministic locking sequence
      // first, obtain table schema locks
      // then, obtain table data locks
      // force sort into tableid order in case of name change to guarantee fixed order of
      // mutex access
      std::sort(
          tables.begin(),
          tables.end(),
          [&cat](const std::vector<std::string>& a, const std::vector<std::string>& b) {
            return cat->getMetadataForTable(a[0], false)->tableId <
                   cat->getMetadataForTable(b[0], false)->tableId;
          });

      // In the case of self-join and possibly other cases, we will
      // have duplicate tables. Ensure we only take one for locking below.
      tables.erase(unique(tables.begin(), tables.end()), tables.end());
      for (const auto& table : tables) {
        locks.emplace_back(
            std::make_unique<lockmgr::TableSchemaLockContainer<lockmgr::ReadLock>>(
                lockmgr::TableSchemaLockContainer<
                    lockmgr::ReadLock>::acquireTableDescriptor(*cat.get(), table[0])));
        if (write_only_tables.count(table)) {
          // Aquire an insert data lock for updates/deletes, consistent w/ insert. The
          // table data lock will be aquired in the fragmenter during checkpoint.
          locks.emplace_back(
              std::make_unique<lockmgr::TableInsertLockContainer<lockmgr::WriteLock>>(
                  lockmgr::TableInsertLockContainer<lockmgr::WriteLock>::acquire(
                      cat->getDatabaseId(), (*locks.back())())));
        } else {
          locks.emplace_back(
              std::make_unique<lockmgr::TableDataLockContainer<lockmgr::ReadLock>>(
                  lockmgr::TableDataLockContainer<lockmgr::ReadLock>::acquire(
                      cat->getDatabaseId(), (*locks.back())())));
        }
      }
    }
    return std::make_pair(result, std::move(locks));
  }
  return std::make_pair(result, lockmgr::LockedTableDescriptors{});
}

int64_t DBHandler::query_get_outer_fragment_count(const TSessionId& session,
                                                  const std::string& select_query) {
  auto stdlog = STDLOG(get_session_ptr(session));
  if (!leaf_handler_) {
    THROW_MAPD_EXCEPTION("Distributed support is disabled.");
  }
  try {
    return leaf_handler_->query_get_outer_fragment_count(session, select_query);
  } catch (std::exception& e) {
    THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
  }
}

void DBHandler::check_table_consistency(TTableMeta& _return,
                                        const TSessionId& session,
                                        const int32_t table_id) {
  auto stdlog = STDLOG(get_session_ptr(session));
  if (!leaf_handler_) {
    THROW_MAPD_EXCEPTION("Distributed support is disabled.");
  }
  try {
    leaf_handler_->check_table_consistency(_return, session, table_id);
  } catch (std::exception& e) {
    THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
  }
}

void DBHandler::start_query(TPendingQuery& _return,
                            const TSessionId& leaf_session,
                            const TSessionId& parent_session,
                            const std::string& query_ra,
                            const std::string& start_time_str,
                            const bool just_explain,
                            const std::vector<int64_t>& outer_fragment_indices) {
  auto stdlog = STDLOG(get_session_ptr(leaf_session));
  auto session_ptr = stdlog.getConstSessionInfo();
  if (!leaf_handler_) {
    THROW_MAPD_EXCEPTION("Distributed support is disabled.");
  }
  LOG(INFO) << "start_query :" << *session_ptr << " :" << just_explain;
  auto time_ms = measure<>::execution([&]() {
    try {
      leaf_handler_->start_query(_return,
                                 leaf_session,
                                 parent_session,
                                 query_ra,
                                 start_time_str,
                                 just_explain,
                                 outer_fragment_indices);
    } catch (std::exception& e) {
      THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
    }
  });
  LOG(INFO) << "start_query-COMPLETED " << time_ms << "ms "
            << "id is " << _return.id;
}

void DBHandler::execute_query_step(TStepResult& _return,
                                   const TPendingQuery& pending_query,
                                   const TSubqueryId subquery_id,
                                   const std::string& start_time_str) {
  if (!leaf_handler_) {
    THROW_MAPD_EXCEPTION("Distributed support is disabled.");
  }
  LOG(INFO) << "execute_query_step :  id:" << pending_query.id;
  auto time_ms = measure<>::execution([&]() {
    try {
      leaf_handler_->execute_query_step(
          _return, pending_query, subquery_id, start_time_str);
    } catch (std::exception& e) {
      THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
    }
  });
  LOG(INFO) << "execute_query_step-COMPLETED " << time_ms << "ms";
}

void DBHandler::broadcast_serialized_rows(const TSerializedRows& serialized_rows,
                                          const TRowDescriptor& row_desc,
                                          const TQueryId query_id,
                                          const TSubqueryId subquery_id,
                                          const bool is_final_subquery_result) {
  if (!leaf_handler_) {
    THROW_MAPD_EXCEPTION("Distributed support is disabled.");
  }
  LOG(INFO) << "BROADCAST-SERIALIZED-ROWS  id:" << query_id;
  auto time_ms = measure<>::execution([&]() {
    try {
      leaf_handler_->broadcast_serialized_rows(
          serialized_rows, row_desc, query_id, subquery_id, is_final_subquery_result);
    } catch (std::exception& e) {
      THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
    }
  });
  LOG(INFO) << "BROADCAST-SERIALIZED-ROWS COMPLETED " << time_ms << "ms";
}

void DBHandler::insert_data(const TSessionId& session,
                            const TInsertData& thrift_insert_data) {
  try {
    auto stdlog = STDLOG(get_session_ptr(session));
    auto session_ptr = stdlog.getConstSessionInfo();
    CHECK_EQ(thrift_insert_data.column_ids.size(), thrift_insert_data.data.size());
    CHECK(thrift_insert_data.is_default.size() == 0 ||
          thrift_insert_data.is_default.size() == thrift_insert_data.column_ids.size());
    auto const& cat = session_ptr->getCatalog();
    Fragmenter_Namespace::InsertData insert_data;
    insert_data.databaseId = thrift_insert_data.db_id;
    insert_data.tableId = thrift_insert_data.table_id;
    insert_data.columnIds = thrift_insert_data.column_ids;
    insert_data.is_default = thrift_insert_data.is_default;
    insert_data.numRows = thrift_insert_data.num_rows;
    std::vector<std::unique_ptr<std::vector<std::string>>> none_encoded_string_columns;
    std::vector<std::unique_ptr<std::vector<ArrayDatum>>> array_columns;
    for (size_t col_idx = 0; col_idx < insert_data.columnIds.size(); ++col_idx) {
      const int column_id = insert_data.columnIds[col_idx];
      DataBlockPtr p;
      const auto cd = cat.getMetadataForColumn(insert_data.tableId, column_id);
      CHECK(cd);
      const auto& ti = cd->columnType;
      size_t rows_expected =
          !insert_data.is_default.empty() && insert_data.is_default[col_idx]
              ? 1ul
              : insert_data.numRows;
      if (ti.is_number() || ti.is_time() || ti.is_boolean()) {
        p.numbersPtr = (int8_t*)thrift_insert_data.data[col_idx].fixed_len_data.data();
      } else if (ti.is_string()) {
        if (ti.get_compression() == kENCODING_DICT) {
          p.numbersPtr = (int8_t*)thrift_insert_data.data[col_idx].fixed_len_data.data();
        } else {
          CHECK_EQ(kENCODING_NONE, ti.get_compression());
          none_encoded_string_columns.emplace_back(new std::vector<std::string>());
          auto& none_encoded_strings = none_encoded_string_columns.back();

          CHECK_EQ(rows_expected, thrift_insert_data.data[col_idx].var_len_data.size());
          for (const auto& varlen_str : thrift_insert_data.data[col_idx].var_len_data) {
            none_encoded_strings->push_back(varlen_str.payload);
          }
          p.stringsPtr = none_encoded_strings.get();
        }
      } else if (ti.is_geometry()) {
        none_encoded_string_columns.emplace_back(new std::vector<std::string>());
        auto& none_encoded_strings = none_encoded_string_columns.back();
        CHECK_EQ(rows_expected, thrift_insert_data.data[col_idx].var_len_data.size());
        for (const auto& varlen_str : thrift_insert_data.data[col_idx].var_len_data) {
          none_encoded_strings->push_back(varlen_str.payload);
        }
        p.stringsPtr = none_encoded_strings.get();
      } else {
        CHECK(ti.is_array());
        array_columns.emplace_back(new std::vector<ArrayDatum>());
        auto& array_column = array_columns.back();
        CHECK_EQ(rows_expected, thrift_insert_data.data[col_idx].var_len_data.size());
        for (const auto& t_arr_datum : thrift_insert_data.data[col_idx].var_len_data) {
          if (t_arr_datum.is_null) {
            if (ti.get_size() > 0 && !ti.get_elem_type().is_string()) {
              array_column->push_back(import_export::ImporterUtils::composeNullArray(ti));
            } else {
              array_column->emplace_back(0, nullptr, true);
            }
          } else {
            ArrayDatum arr_datum;
            arr_datum.length = t_arr_datum.payload.size();
            int8_t* ptr = (int8_t*)(t_arr_datum.payload.data());
            arr_datum.pointer = ptr;
            // In this special case, ArrayDatum does not handle freeing the underlying
            // memory
            arr_datum.data_ptr = std::shared_ptr<int8_t>(ptr, [](auto p) {});
            arr_datum.is_null = false;
            array_column->push_back(arr_datum);
          }
        }
        p.arraysPtr = array_column.get();
      }
      insert_data.data.push_back(p);
    }
    const ChunkKey lock_chunk_key{cat.getDatabaseId(),
                                  cat.getLogicalTableId(insert_data.tableId)};
    auto table_read_lock =
        lockmgr::TableSchemaLockMgr::getReadLockForTable(lock_chunk_key);
    const auto td = cat.getMetadataForTable(insert_data.tableId);
    CHECK(td);

    // this should have the same lock seq as COPY FROM
    ChunkKey chunkKey = {insert_data.databaseId, insert_data.tableId};
    auto insert_data_lock = lockmgr::InsertDataLockMgr::getWriteLockForTable(chunkKey);
    auto data_memory_holder = import_export::fill_missing_columns(&cat, insert_data);
    td->fragmenter->insertDataNoCheckpoint(insert_data);
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION("Exception: " + std::string(e.what()));
  }
}

void DBHandler::start_render_query(TPendingRenderQuery& _return,
                                   const TSessionId& session,
                                   const int64_t widget_id,
                                   const int16_t node_idx,
                                   const std::string& vega_json) {
  auto stdlog = STDLOG(get_session_ptr(session));
  auto session_ptr = stdlog.getConstSessionInfo();
  if (!render_handler_) {
    THROW_MAPD_EXCEPTION("Backend rendering is disabled.");
  }
  LOG(INFO) << "start_render_query :" << *session_ptr << " :widget_id:" << widget_id
            << ":vega_json:" << vega_json;
  auto time_ms = measure<>::execution([&]() {
    try {
      render_handler_->start_render_query(
          _return, session, widget_id, node_idx, vega_json);
    } catch (std::exception& e) {
      THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
    }
  });
  LOG(INFO) << "start_render_query-COMPLETED " << time_ms << "ms "
            << "id is " << _return.id;
}

void DBHandler::execute_next_render_step(TRenderStepResult& _return,
                                         const TPendingRenderQuery& pending_render,
                                         const TRenderAggDataMap& merged_data) {
  if (!render_handler_) {
    THROW_MAPD_EXCEPTION("Backend rendering is disabled.");
  }

  LOG(INFO) << "execute_next_render_step: id:" << pending_render.id;
  auto time_ms = measure<>::execution([&]() {
    try {
      render_handler_->execute_next_render_step(_return, pending_render, merged_data);
    } catch (std::exception& e) {
      THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
    }
  });
  LOG(INFO) << "execute_next_render_step-COMPLETED id: " << pending_render.id
            << ", time: " << time_ms << "ms ";
}

void DBHandler::checkpoint(const TSessionId& session, const int32_t table_id) {
  auto stdlog = STDLOG(get_session_ptr(session));
  auto session_ptr = stdlog.getConstSessionInfo();
  auto& cat = session_ptr->getCatalog();
  cat.checkpoint(table_id);
}

// check and reset epoch if a request has been made
void DBHandler::set_table_epoch(const TSessionId& session,
                                const int db_id,
                                const int table_id,
                                const int new_epoch) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  if (!session_ptr->get_currentUser().isSuper) {
    throw std::runtime_error("Only superuser can set_table_epoch");
  }
  auto& cat = session_ptr->getCatalog();

  if (leaf_aggregator_.leafCount() > 0) {
    return leaf_aggregator_.set_table_epochLeaf(*session_ptr, db_id, table_id, new_epoch);
  }
  try {
    cat.setTableEpoch(db_id, table_id, new_epoch);
  } catch (const std::runtime_error& e) {
    THROW_MAPD_EXCEPTION("Exception: " + std::string(e.what()));
  }
}

// check and reset epoch if a request has been made
void DBHandler::set_table_epoch_by_name(const TSessionId& session,
                                        const std::string& table_name,
                                        const int new_epoch) {
  auto stdlog = STDLOG(get_session_ptr(session), "table_name", table_name);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  if (!session_ptr->get_currentUser().isSuper) {
    throw std::runtime_error("Only superuser can set_table_epoch");
  }
  auto& cat = session_ptr->getCatalog();
  auto td = cat.getMetadataForTable(
      table_name,
      false);  // don't populate fragmenter on this call since we only want metadata
  int32_t db_id = cat.getCurrentDB().dbId;
  if (leaf_aggregator_.leafCount() > 0) {
    return leaf_aggregator_.set_table_epochLeaf(
        *session_ptr, db_id, td->tableId, new_epoch);
  }
  try {
    cat.setTableEpoch(db_id, td->tableId, new_epoch);
  } catch (const std::runtime_error& e) {
    THROW_MAPD_EXCEPTION("Exception: " + std::string(e.what()));
  }
}

int32_t DBHandler::get_table_epoch(const TSessionId& session,
                                   const int32_t db_id,
                                   const int32_t table_id) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  auto const& cat = session_ptr->getCatalog();

  if (leaf_aggregator_.leafCount() > 0) {
    return leaf_aggregator_.get_table_epochLeaf(*session_ptr, db_id, table_id);
  }
  try {
    return cat.getTableEpoch(db_id, table_id);
  } catch (const std::runtime_error& e) {
    THROW_MAPD_EXCEPTION("Exception: " + std::string(e.what()));
  }
}

int32_t DBHandler::get_table_epoch_by_name(const TSessionId& session,
                                           const std::string& table_name) {
  auto stdlog = STDLOG(get_session_ptr(session), "table_name", table_name);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  auto const& cat = session_ptr->getCatalog();
  auto td = cat.getMetadataForTable(
      table_name,
      false);  // don't populate fragmenter on this call since we only want metadata
  int32_t db_id = cat.getCurrentDB().dbId;
  if (leaf_aggregator_.leafCount() > 0) {
    return leaf_aggregator_.get_table_epochLeaf(*session_ptr, db_id, td->tableId);
  }
  try {
    return cat.getTableEpoch(db_id, td->tableId);
  } catch (const std::runtime_error& e) {
    THROW_MAPD_EXCEPTION("Exception: " + std::string(e.what()));
  }
}

void DBHandler::get_table_epochs(std::vector<TTableEpochInfo>& _return,
                                 const TSessionId& session,
                                 const int32_t db_id,
                                 const int32_t table_id) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  auto const& cat = session_ptr->getCatalog();

  std::vector<Catalog_Namespace::TableEpochInfo> table_epochs;
  if (leaf_aggregator_.leafCount() > 0) {
    table_epochs = leaf_aggregator_.getLeafTableEpochs(*session_ptr, db_id, table_id);
  } else {
    table_epochs = cat.getTableEpochs(db_id, table_id);
  }
  CHECK(!table_epochs.empty());

  for (const auto& table_epoch : table_epochs) {
    TTableEpochInfo table_epoch_info;
    table_epoch_info.table_id = table_epoch.table_id;
    table_epoch_info.table_epoch = table_epoch.table_epoch;
    table_epoch_info.leaf_index = table_epoch.leaf_index;
    _return.emplace_back(table_epoch_info);
  }
}

void DBHandler::set_table_epochs(const TSessionId& session,
                                 const int32_t db_id,
                                 const std::vector<TTableEpochInfo>& table_epochs) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();

  // Only super users are allowed to call this API on a single node instance
  // or aggregator (for distributed mode)
  if (!g_cluster || leaf_aggregator_.leafCount() > 0) {
    if (!session_ptr->get_currentUser().isSuper) {
      THROW_MAPD_EXCEPTION("Only super users can set table epochs");
    }
  }
  std::vector<Catalog_Namespace::TableEpochInfo> table_epochs_vector;
  for (const auto& table_epoch : table_epochs) {
    table_epochs_vector.emplace_back(
        table_epoch.table_id, table_epoch.table_epoch, table_epoch.leaf_index);
  }
  if (leaf_aggregator_.leafCount() > 0) {
    leaf_aggregator_.setLeafTableEpochs(*session_ptr, db_id, table_epochs_vector);
  } else {
    auto& cat = session_ptr->getCatalog();
    cat.setTableEpochs(db_id, table_epochs_vector);
  }
}

void DBHandler::set_license_key(TLicenseInfo& _return,
                                const TSessionId& session,
                                const std::string& key,
                                const std::string& nonce) {
  auto stdlog = STDLOG(get_session_ptr(session));
  auto session_ptr = stdlog.getConstSessionInfo();
  check_read_only("set_license_key");
  THROW_MAPD_EXCEPTION(std::string("Licensing not supported."));
}

void DBHandler::get_license_claims(TLicenseInfo& _return,
                                   const TSessionId& session,
                                   const std::string& nonce) {
  auto stdlog = STDLOG(get_session_ptr(session));
  const auto session_info = get_session_copy(session);
  _return.claims.emplace_back("");
}

void DBHandler::shutdown() {
  emergency_shutdown();

  if (render_handler_) {
    render_handler_->shutdown();
  }
}

void DBHandler::emergency_shutdown() {
  if (calcite_) {
    calcite_->close_calcite_server(false);
  }
}

extern std::map<std::string, std::string> get_device_parameters(bool cpu_only);

#define EXPOSE_THRIFT_MAP(TYPENAME)                                             \
  {                                                                             \
    std::map<int, const char*>::const_iterator it =                             \
        _##TYPENAME##_VALUES_TO_NAMES.begin();                                  \
    while (it != _##TYPENAME##_VALUES_TO_NAMES.end()) {                         \
      _return.insert(std::pair<std::string, std::string>(                       \
          #TYPENAME "." + std::string(it->second), std::to_string(it->first))); \
      it++;                                                                     \
    }                                                                           \
  }

void DBHandler::get_device_parameters(std::map<std::string, std::string>& _return,
                                      const TSessionId& session) {
  const auto session_info = get_session_copy(session);
  auto params = ::get_device_parameters(cpu_mode_only_);
  for (auto item : params) {
    _return.insert(item);
  }
  EXPOSE_THRIFT_MAP(TDeviceType);
  EXPOSE_THRIFT_MAP(TDatumType);
  EXPOSE_THRIFT_MAP(TEncodingType);
  EXPOSE_THRIFT_MAP(TExtArgumentType);
  EXPOSE_THRIFT_MAP(TOutputBufferSizeType);
}

void DBHandler::register_runtime_extension_functions(
    const TSessionId& session,
    const std::vector<TUserDefinedFunction>& udfs,
    const std::vector<TUserDefinedTableFunction>& udtfs,
    const std::map<std::string, std::string>& device_ir_map) {
  const auto session_info = get_session_copy(session);
  VLOG(1) << "register_runtime_extension_functions: " << udfs.size() << " "
          << udtfs.size() << std::endl;

  if (!runtime_udf_registration_enabled_) {
    THROW_MAPD_EXCEPTION("Runtime extension functions registration is disabled.");
  }

  // TODO: add UDF registration permission scheme. Currently, UDFs are
  // registered globally, that means that all users can use as well as
  // overwrite UDFs that was created possibly by anoher user.

  /* Changing a UDF implementation (but not the signature) requires
     cleaning code caches. Nuking executors does that but at the cost
     of loosing all of the caches. TODO: implement more refined code
     cache cleaning. */
  Executor::nukeCacheOfExecutors();

  /* Parse LLVM/NVVM IR strings and store it as LLVM module. */
  auto it = device_ir_map.find(std::string{"cpu"});
  if (it != device_ir_map.end()) {
    read_rt_udf_cpu_module(it->second);
  }
  it = device_ir_map.find(std::string{"gpu"});
  if (it != device_ir_map.end()) {
    read_rt_udf_gpu_module(it->second);
  }

  VLOG(1) << "Registering runtime UDTFs:\n";

  table_functions::TableFunctionsFactory::reset();

  for (auto it = udtfs.begin(); it != udtfs.end(); it++) {
    VLOG(1) << "UDTF name=" << it->name << std::endl;
    table_functions::TableFunctionsFactory::add(
        it->name,
        table_functions::TableFunctionOutputRowSizer{
            ThriftSerializers::from_thrift(it->sizerType),
            static_cast<size_t>(it->sizerArgPos)},
        ThriftSerializers::from_thrift(it->inputArgTypes),
        ThriftSerializers::from_thrift(it->outputArgTypes),
        ThriftSerializers::from_thrift(it->sqlArgTypes),
        /*is_runtime =*/true);
  }

  /* Register extension functions with Calcite server */
  CHECK(calcite_);
  auto udtfs_ = ThriftSerializers::to_thrift(
      table_functions::TableFunctionsFactory::get_table_funcs(/*is_runtime=*/true));
  calcite_->setRuntimeExtensionFunctions(udfs, udtfs_, /*is_runtime =*/true);

  /* Update the extension function whitelist */
  std::string whitelist = calcite_->getRuntimeExtensionFunctionWhitelist();
  VLOG(1) << "Registering runtime extension functions with CodeGen using whitelist:\n"
          << whitelist;
  ExtensionFunctionsWhitelist::clearRTUdfs();
  ExtensionFunctionsWhitelist::addRTUdfs(whitelist);
}

void DBHandler::convertResultSet(ExecutionResult& result,
                                 const Catalog_Namespace::SessionInfo& session_info,
                                 const std::string& query_state_str,
                                 TQueryResult& _return) {
  // Stuff ResultSet into _return (which is a TQueryResult)
  // calls convertRows, but after some setup using session_info

  auto session_ptr = get_session_ptr(session_info.get_session_id());
  auto qs = create_query_state(session_ptr, query_state_str);
  QueryStateProxy qsp = qs->createQueryStateProxy();

  // omnisql only accepts column format as being 'VALID",
  //   assume that omnisci_server should only return column format
  int32_t nRows = result.getDataPtr()->rowCount();

  convertRows(_return,
              qsp,
              result.getTargetsMeta(),
              *result.getDataPtr(),
              /*column_format=*/true,
              /*first_n=*/nRows,
              /*at_most_n=*/nRows);
}

static std::unique_ptr<RexLiteral> genLiteralStr(std::string val) {
  return std::unique_ptr<RexLiteral>(
      new RexLiteral(val, SQLTypes::kTEXT, SQLTypes::kTEXT, 0, 0, 0, 0));
}

ExecutionResult DBHandler::getUserSessions(
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr) {
  std::shared_ptr<ResultSet> rSet = nullptr;
  std::vector<TargetMetaInfo> label_infos;

  if (!session_ptr->get_currentUser().isSuper) {
    throw std::runtime_error(
        "SHOW USER SESSIONS failed, because it can only be executed by super user.");

  } else {
    // label_infos -> column labels
    std::vector<std::string> labels{
        "session_id", "login_name", "client_address", "db_name"};
    for (const auto& label : labels) {
      label_infos.emplace_back(label, SQLTypeInfo(kTEXT, true));
    }

    // logical_values -> table data
    std::vector<RelLogicalValues::RowValues> logical_values;

    if (!sessions_.empty()) {
      mapd_lock_guard<mapd_shared_mutex> read_lock(sessions_mutex_);

      for (auto sessions = sessions_.begin(); sessions_.end() != sessions; sessions++) {
        const auto show_session_ptr = sessions->second;
        logical_values.emplace_back(RelLogicalValues::RowValues{});
        logical_values.back().emplace_back(
            genLiteralStr(show_session_ptr->get_public_session_id()));
        logical_values.back().emplace_back(
            genLiteralStr(show_session_ptr->get_currentUser().userName));
        logical_values.back().emplace_back(
            genLiteralStr(show_session_ptr->get_connection_info()));
        logical_values.back().emplace_back(
            genLiteralStr(show_session_ptr->getCatalog().getCurrentDB().dbName));
      }
    }

    // Create ResultSet
    rSet = std::shared_ptr<ResultSet>(
        ResultSetLogicalValuesBuilder::create(label_infos, logical_values));
  }
  return ExecutionResult(rSet, label_infos);
}

ExecutionResult DBHandler::getQueries(
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr) {
  std::shared_ptr<ResultSet> rSet = nullptr;
  std::vector<TargetMetaInfo> label_infos;

  if (!session_ptr->get_currentUser().isSuper) {
    throw std::runtime_error(
        "SHOW QUERIES failed, because it can only be executed by super user.");
  } else {
    mapd_lock_guard<mapd_shared_mutex> read_lock(sessions_mutex_);
    std::vector<std::string> labels{"query_session_id",
                                    "current_status",
                                    "executor_id",
                                    "submitted",
                                    "query_str",
                                    "login_name",
                                    "client_address",
                                    "db_name",
                                    "exec_device_type"};
    for (const auto& label : labels) {
      label_infos.emplace_back(label, SQLTypeInfo(kTEXT, true));
    }

    std::vector<RelLogicalValues::RowValues> logical_values;
    if (!sessions_.empty()) {
      for (auto session = sessions_.begin(); sessions_.end() != session; session++) {
        const auto id = session->first;
        const auto query_session_ptr = session->second;
        auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID,
                                              jit_debug_ ? "/tmp" : "",
                                              jit_debug_ ? "mapdquery" : "",
                                              system_parameters_);
        CHECK(executor);
        std::vector<QuerySessionStatus> query_infos;
        {
          mapd_shared_lock<mapd_shared_mutex> session_read_lock(
              executor->getSessionLock());
          query_infos = executor->getQuerySessionInfo(query_session_ptr->get_session_id(),
                                                      session_read_lock);
        }
        // if there exists query info fired from this session we report it to user
        const std::string getQueryStatusStr[] = {
            "UNDEFINED", "PENDING_QUEUE", "PENDING_EXECUTOR", "RUNNING"};
        for (QuerySessionStatus& query_info : query_infos) {
          logical_values.emplace_back(RelLogicalValues::RowValues{});
          logical_values.back().emplace_back(
              genLiteralStr(query_session_ptr->get_public_session_id()));
          logical_values.back().emplace_back(
              genLiteralStr(getQueryStatusStr[query_info.getQueryStatus()]));
          logical_values.back().emplace_back(
              genLiteralStr(::toString(query_info.getExecutorId())));
          logical_values.back().emplace_back(
              genLiteralStr(query_info.getQuerySubmittedTime()));
          logical_values.back().emplace_back(genLiteralStr(query_info.getQueryStr()));
          logical_values.back().emplace_back(
              genLiteralStr(query_session_ptr->get_currentUser().userName));
          logical_values.back().emplace_back(
              genLiteralStr(query_session_ptr->get_connection_info()));
          logical_values.back().emplace_back(
              genLiteralStr(query_session_ptr->getCatalog().getCurrentDB().dbName));
          if (query_session_ptr->get_executor_device_type() == ExecutorDeviceType::GPU) {
            logical_values.back().emplace_back(genLiteralStr("GPU"));
          } else {
            logical_values.back().emplace_back(genLiteralStr("CPU"));
          }
        }
      }
    }

    rSet = std::shared_ptr<ResultSet>(
        ResultSetLogicalValuesBuilder::create(label_infos, logical_values));
  }

  return ExecutionResult(rSet, label_infos);
}

void DBHandler::interruptQuery(const Catalog_Namespace::SessionInfo& session_info,
                               const std::string& target_session) {
  // capture the interrupt request from user and then pass to the query runtime (executor)
  // Basic-flow that each query session gets through:
  // Enroll --> Update (query session info / executor) --> Running -> Cleanup

  // 1. We have to separate 1) "target" query session to interrupt and 2) request session
  // Here, we have to focus on "target" session: all interruption management is based on
  // the "target" session
  // 2. Session info and its required data structures are global to executor, so
  // we can manage the interrupt status via UNITARY_EXECUTOR (note that the actual query
  // is processed by specific executor but can also access the global data structure)
  // 3. Three target session's status: PENDING_QUEUE / PENDING_EXECUTOR / RUNNING
  // (for now we can interrupt a query at "PENDING_EXECUTOR" and "RUNNING")
  // 4. each session has 1) a list of queries that the session tries to initiate and
  // 2) a interrupt flag map that indicates whether the session is interrupted
  // If a session is interrupted, we turn the flag for the session on so as to Executor
  // can know about the user's interrupt request on the query (after all queries are
  // removed then the session's query list and its flag are also deleted). And those
  // info is managed by Executor's global data structure
  // 5. To interrupt queries at "PENDING_EXECUTOR", corresponding Executor regularly
  // checks the interrupt flag of the session, and throws an exception if got interrupted
  // For the case of running query, we also turn the flag in device memory on in async
  // manner so as to inform the query kernel about the latest interrupt flag status
  // (it also checks the flag regularly during the query kernel execution and
  // query threads return with the error code if necessary -->
  // for this we inject interrupt flag checking logic in the generated query kernel)
  // 6. Interruption are implemented by throwing runtime_error that contains a visible
  // error message like "Query has been interrupted"

  if (!g_enable_runtime_query_interrupt && !g_enable_non_kernel_time_query_interrupt) {
    // at least type of query interruption is enabled to allow kill query
    // if non-kernel query interrupt is enabled but tries to kill that type's query?
    // then the request is skipped
    // todo(yoonmin): improve kill query cmd under both types of query
    throw std::runtime_error(
        "KILL QUERY failed: neither non-kernel time nor kernel-time query interrupt is "
        "disabled.");
  }
  if (!session_info.get_currentUser().isSuper.load()) {
    throw std::runtime_error(
        "KILL QUERY failed: only super user can interrupt the query via KILL QUERY "
        "command.");
  }
  CHECK_EQ(target_session.length(), (unsigned long)8);
  bool found_valid_session = false;
  for (auto& kv : sessions_) {
    if (kv.second->get_public_session_id().compare(target_session) == 0) {
      auto target_query_session = kv.second->get_session_id();
      auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID,
                                            jit_debug_ ? "/tmp" : "",
                                            jit_debug_ ? "mapdquery" : "",
                                            system_parameters_);
      CHECK(executor);
      if (leaf_aggregator_.leafCount() > 0) {
        leaf_aggregator_.interrupt(target_query_session, session_info.get_session_id());
      }
      executor->interrupt(target_query_session, session_info.get_session_id());
      found_valid_session = true;
      break;
    }
  }
  if (!found_valid_session) {
    throw std::runtime_error("KILL QUERY failed: invalid query session is given.");
  }
}

void DBHandler::executeDdl(
    TQueryResult& _return,
    const std::string& query_ra,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr) {
  DdlCommandExecutor executor = DdlCommandExecutor(query_ra, session_ptr);
  std::string commandStr = executor.commandStr();

  if (executor.isKillQuery()) {
    interruptQuery(*session_ptr, executor.getTargetQuerySessionToKill());
  } else {
    ExecutionResult result;

    if (executor.isShowQueries()) {
      // getQueries still requires Thrift cannot be nested into DdlCommandExecutor
      _return.execution_time_ms +=
          measure<>::execution([&]() { result = getQueries(session_ptr); });
    } else if (executor.isShowUserSessions()) {
      // getUserSessions still requires Thrift cannot be nested into DdlCommandExecutor
      _return.execution_time_ms +=
          measure<>::execution([&]() { result = getUserSessions(session_ptr); });
    } else {
      _return.execution_time_ms +=
          measure<>::execution([&]() { result = executor.execute(); });
    }

    if (!result.empty()) {
      // reduce execution time by the time spent during queue waiting
      _return.execution_time_ms -= result.getRows()->getQueueTime();

      convertResultSet(result, *session_ptr, commandStr, _return);
    }
  }
}

void DBHandler::executeDdl(
    ExecutionResult& _return,
    const std::string& query_ra,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr) {
  DdlCommandExecutor executor = DdlCommandExecutor(query_ra, session_ptr);
  std::string commandStr = executor.commandStr();

  if (executor.isKillQuery()) {
    interruptQuery(*session_ptr, executor.getTargetQuerySessionToKill());
  } else {
    int64_t execution_time_ms;
    if (executor.isShowQueries()) {
      // getQueries still requires Thrift cannot be nested into DdlCommandExecutor
      execution_time_ms =
          measure<>::execution([&]() { _return = getQueries(session_ptr); });
    } else if (executor.isShowUserSessions()) {
      // getUserSessions still requires Thrift cannot be nested into DdlCommandExecutor
      execution_time_ms =
          measure<>::execution([&]() { _return = getUserSessions(session_ptr); });
    } else {
      execution_time_ms = measure<>::execution([&]() { _return = executor.execute(); });
    }
    _return.setExecutionTime(execution_time_ms);
  }
  _return.setResultType(ExecutionResult::CalciteDdl);
}
