/*
 * Copyright 2021 OmniSci, Inc.
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

#include "Execute.h"

#include "AggregateUtils.h"
#include "CodeGenerator.h"
#include "ColumnFetcher.h"
#include "Descriptors/QueryCompilationDescriptor.h"
#include "Descriptors/QueryFragmentDescriptor.h"
#include "DynamicWatchdog.h"
#include "EquiJoinCondition.h"
#include "ErrorHandling.h"
#include "ExpressionRewrite.h"
#include "ExternalCacheInvalidators.h"
#include "GpuMemUtils.h"
#include "InPlaceSort.h"
#include "JoinHashTable/BaselineJoinHashTable.h"
#include "JoinHashTable/OverlapsJoinHashTable.h"
#include "JsonAccessors.h"
#include "OutputBufferInitialization.h"
#include "QueryEngine/QueryDispatchQueue.h"
#include "QueryRewrite.h"
#include "QueryTemplateGenerator.h"
#include "ResultSetReductionJIT.h"
#include "RuntimeFunctions.h"
#include "SpeculativeTopN.h"

#include "TableFunctions/TableFunctionCompilationContext.h"
#include "TableFunctions/TableFunctionExecutionContext.h"

#include "CudaMgr/CudaMgr.h"
#include "DataMgr/BufferMgr/BufferMgr.h"
#include "Parser/ParserNode.h"
#include "Shared/SystemParameters.h"
#include "Shared/TypedDataAccessors.h"
#include "Shared/checked_alloc.h"
#include "Shared/measure.h"
#include "Shared/misc.h"
#include "Shared/scope.h"
#include "Shared/shard_key.h"
#include "Shared/threadpool.h"

#include "AggregatedColRange.h"
#include "StringDictionaryGenerations.h"

#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#ifdef HAVE_CUDA
#include <cuda.h>
#endif  // HAVE_CUDA
#include <chrono>
#include <ctime>
#include <future>
#include <iostream>
#include <memory>
#include <numeric>
#include <set>
#include <thread>

bool g_enable_watchdog{false};
bool g_enable_dynamic_watchdog{false};
bool g_use_tbb_pool{false};
bool g_enable_filter_function{true};
unsigned g_dynamic_watchdog_time_limit{10000};
bool g_allow_cpu_retry{true};
bool g_null_div_by_zero{false};
unsigned g_trivial_loop_join_threshold{1000};
bool g_from_table_reordering{true};
bool g_inner_join_fragment_skipping{true};
extern bool g_enable_smem_group_by;
extern std::unique_ptr<llvm::Module> udf_gpu_module;
extern std::unique_ptr<llvm::Module> udf_cpu_module;
bool g_enable_filter_push_down{false};
float g_filter_push_down_low_frac{-1.0f};
float g_filter_push_down_high_frac{-1.0f};
size_t g_filter_push_down_passing_row_ubound{0};
bool g_enable_columnar_output{false};
bool g_enable_left_join_filter_hoisting{true};
bool g_optimize_row_initialization{true};
bool g_enable_overlaps_hashjoin{true};
bool g_enable_hashjoin_many_to_many{false};
size_t g_overlaps_max_table_size_bytes{1024 * 1024 * 1024};
double g_overlaps_target_entries_per_bin{1.3};
bool g_strip_join_covered_quals{false};
size_t g_constrained_by_in_threshold{10};
size_t g_default_max_groups_buffer_entry_guess{16384};
size_t g_big_group_threshold{g_default_max_groups_buffer_entry_guess};
bool g_enable_window_functions{true};
bool g_enable_table_functions{false};
size_t g_max_memory_allocation_size{2000000000};  // set to max slab size
size_t g_min_memory_allocation_size{
    256};  // minimum memory allocation required for projection query output buffer
           // without pre-flight count
bool g_enable_bump_allocator{false};
double g_bump_allocator_step_reduction{0.75};
bool g_enable_direct_columnarization{true};
extern bool g_enable_experimental_string_functions;
bool g_enable_lazy_fetch{true};
bool g_enable_runtime_query_interrupt{false};
bool g_enable_non_kernel_time_query_interrupt{true};
bool g_use_estimator_result_cache{true};
unsigned g_pending_query_interrupt_freq{1000};
double g_running_query_interrupt_freq{0.5};
size_t g_gpu_smem_threshold{
    4096};  // GPU shared memory threshold (in bytes), if larger
            // buffer sizes are required we do not use GPU shared
            // memory optimizations Setting this to 0 means unlimited
            // (subject to other dynamically calculated caps)
bool g_enable_smem_grouped_non_count_agg{
    true};  // enable use of shared memory when performing group-by with select non-count
            // aggregates
bool g_enable_smem_non_grouped_agg{
    true};  // enable optimizations for using GPU shared memory in implementation of
            // non-grouped aggregates
bool g_is_test_env{false};  // operating under a unit test environment. Currently only
                            // limits the allocation for the output buffer arena
size_t g_enable_parallel_linearization{
    10000};  // # rows that we are trying to linearize varlen col in parallel

size_t g_approx_quantile_buffer{1000};
size_t g_approx_quantile_centroids{300};

bool g_enable_automatic_ir_metadata{true};

extern bool g_cache_string_hash;

int const Executor::max_gpu_count;

const int32_t Executor::ERR_SINGLE_VALUE_FOUND_MULTIPLE_VALUES;

Executor::Executor(const ExecutorId executor_id,
                   const size_t block_size_x,
                   const size_t grid_size_x,
                   const size_t max_gpu_slab_size,
                   const std::string& debug_dir,
                   const std::string& debug_file)
    : cgen_state_(new CgenState({}, false))
    , cpu_code_cache_(code_cache_size)
    , gpu_code_cache_(code_cache_size)
    , block_size_x_(block_size_x)
    , grid_size_x_(grid_size_x)
    , max_gpu_slab_size_(max_gpu_slab_size)
    , debug_dir_(debug_dir)
    , debug_file_(debug_file)
    , executor_id_(executor_id)
    , catalog_(nullptr)
    , temporary_tables_(nullptr)
    , input_table_info_cache_(this) {}

std::shared_ptr<Executor> Executor::getExecutor(
    const ExecutorId executor_id,
    const std::string& debug_dir,
    const std::string& debug_file,
    const SystemParameters& system_parameters) {
  INJECT_TIMER(getExecutor);

  mapd_unique_lock<mapd_shared_mutex> write_lock(executors_cache_mutex_);
  auto it = executors_.find(executor_id);
  if (it != executors_.end()) {
    return it->second;
  }
  auto executor = std::make_shared<Executor>(executor_id,
                                             system_parameters.cuda_block_size,
                                             system_parameters.cuda_grid_size,
                                             system_parameters.max_gpu_slab_size,
                                             debug_dir,
                                             debug_file);
  CHECK(executors_.insert(std::make_pair(executor_id, executor)).second);
  return executor;
}

void Executor::clearMemory(const Data_Namespace::MemoryLevel memory_level) {
  switch (memory_level) {
    case Data_Namespace::MemoryLevel::CPU_LEVEL:
    case Data_Namespace::MemoryLevel::GPU_LEVEL: {
      mapd_unique_lock<mapd_shared_mutex> flush_lock(
          execute_mutex_);  // Don't flush memory while queries are running

      Catalog_Namespace::SysCatalog::instance().getDataMgr().clearMemory(memory_level);
      if (memory_level == Data_Namespace::MemoryLevel::CPU_LEVEL) {
        // The hash table cache uses CPU memory not managed by the buffer manager. In the
        // future, we should manage these allocations with the buffer manager directly.
        // For now, assume the user wants to purge the hash table cache when they clear
        // CPU memory (currently used in ExecuteTest to lower memory pressure)
        JoinHashTableCacheInvalidator::invalidateCaches();
      }
      break;
    }
    default: {
      throw std::runtime_error(
          "Clearing memory levels other than the CPU level or GPU level is not "
          "supported.");
    }
  }
}

size_t Executor::getArenaBlockSize() {
  return g_is_test_env ? 100000000 : (1UL << 32) + kArenaBlockOverhead;
}

StringDictionaryProxy* Executor::getStringDictionaryProxy(
    const int dict_id_in,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const bool with_generation) const {
  CHECK(row_set_mem_owner);
  std::lock_guard<std::mutex> lock(
      str_dict_mutex_);  // TODO: can we use RowSetMemOwner state mutex here?
  return row_set_mem_owner->getOrAddStringDictProxy(
      dict_id_in, with_generation, catalog_);
}

StringDictionaryProxy* RowSetMemoryOwner::getOrAddStringDictProxy(
    const int dict_id_in,
    const bool with_generation,
    const Catalog_Namespace::Catalog* catalog) {
  const int dict_id{dict_id_in < 0 ? REGULAR_DICT(dict_id_in) : dict_id_in};
  CHECK(catalog);
  const auto dd = catalog->getMetadataForDict(dict_id);
  if (dd) {
    CHECK(dd->stringDict);
    CHECK_LE(dd->dictNBits, 32);
    const int64_t generation =
        with_generation ? string_dictionary_generations_.getGeneration(dict_id) : -1;
    return addStringDict(dd->stringDict, dict_id, generation);
  }
  CHECK_EQ(0, dict_id);
  if (!lit_str_dict_proxy_) {
    std::shared_ptr<StringDictionary> tsd =
        std::make_shared<StringDictionary>("", false, true, g_cache_string_hash);
    lit_str_dict_proxy_.reset(new StringDictionaryProxy(tsd, 0));
  }
  return lit_str_dict_proxy_.get();
}

quantile::TDigest* RowSetMemoryOwner::nullTDigest() {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return t_digests_
      .emplace_back(std::make_unique<quantile::TDigest>(
          this, g_approx_quantile_buffer, g_approx_quantile_centroids))
      .get();
}

bool Executor::isCPUOnly() const {
  CHECK(catalog_);
  return !catalog_->getDataMgr().getCudaMgr();
}

const ColumnDescriptor* Executor::getColumnDescriptor(
    const Analyzer::ColumnVar* col_var) const {
  return get_column_descriptor_maybe(
      col_var->get_column_id(), col_var->get_table_id(), *catalog_);
}

const ColumnDescriptor* Executor::getPhysicalColumnDescriptor(
    const Analyzer::ColumnVar* col_var,
    int n) const {
  const auto cd = getColumnDescriptor(col_var);
  if (!cd || n > cd->columnType.get_physical_cols()) {
    return nullptr;
  }
  return get_column_descriptor_maybe(
      col_var->get_column_id() + n, col_var->get_table_id(), *catalog_);
}

const Catalog_Namespace::Catalog* Executor::getCatalog() const {
  return catalog_;
}

void Executor::setCatalog(const Catalog_Namespace::Catalog* catalog) {
  catalog_ = catalog;
}

const std::shared_ptr<RowSetMemoryOwner> Executor::getRowSetMemoryOwner() const {
  return row_set_mem_owner_;
}

const TemporaryTables* Executor::getTemporaryTables() const {
  return temporary_tables_;
}

Fragmenter_Namespace::TableInfo Executor::getTableInfo(const int table_id) const {
  return input_table_info_cache_.getTableInfo(table_id);
}

const TableGeneration& Executor::getTableGeneration(const int table_id) const {
  return table_generations_.getGeneration(table_id);
}

ExpressionRange Executor::getColRange(const PhysicalInput& phys_input) const {
  return agg_col_range_cache_.getColRange(phys_input);
}

size_t Executor::getNumBytesForFetchedRow(const std::set<int>& table_ids_to_fetch) const {
  size_t num_bytes = 0;
  if (!plan_state_) {
    return 0;
  }
  for (const auto& fetched_col_pair : plan_state_->columns_to_fetch_) {
    if (table_ids_to_fetch.count(fetched_col_pair.first) == 0) {
      continue;
    }

    if (fetched_col_pair.first < 0) {
      num_bytes += 8;
    } else {
      const auto cd =
          catalog_->getMetadataForColumn(fetched_col_pair.first, fetched_col_pair.second);
      const auto& ti = cd->columnType;
      const auto sz = ti.get_type() == kTEXT && ti.get_compression() == kENCODING_DICT
                          ? 4
                          : ti.get_size();
      if (sz < 0) {
        // for varlen types, only account for the pointer/size for each row, for now
        num_bytes += 16;
      } else {
        num_bytes += sz;
      }
    }
  }
  return num_bytes;
}

std::vector<ColumnLazyFetchInfo> Executor::getColLazyFetchInfo(
    const std::vector<Analyzer::Expr*>& target_exprs) const {
  CHECK(plan_state_);
  CHECK(catalog_);
  std::vector<ColumnLazyFetchInfo> col_lazy_fetch_info;
  for (const auto target_expr : target_exprs) {
    if (!plan_state_->isLazyFetchColumn(target_expr)) {
      col_lazy_fetch_info.emplace_back(
          ColumnLazyFetchInfo{false, -1, SQLTypeInfo(kNULLT, false)});
    } else {
      const auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(target_expr);
      CHECK(col_var);
      auto col_id = col_var->get_column_id();
      auto rte_idx = (col_var->get_rte_idx() == -1) ? 0 : col_var->get_rte_idx();
      auto cd = (col_var->get_table_id() > 0)
                    ? get_column_descriptor(col_id, col_var->get_table_id(), *catalog_)
                    : nullptr;
      if (cd && IS_GEO(cd->columnType.get_type())) {
        // Geo coords cols will be processed in sequence. So we only need to track the
        // first coords col in lazy fetch info.
        {
          auto cd0 =
              get_column_descriptor(col_id + 1, col_var->get_table_id(), *catalog_);
          auto col0_ti = cd0->columnType;
          CHECK(!cd0->isVirtualCol);
          auto col0_var = makeExpr<Analyzer::ColumnVar>(
              col0_ti, col_var->get_table_id(), cd0->columnId, rte_idx);
          auto local_col0_id = plan_state_->getLocalColumnId(col0_var.get(), false);
          col_lazy_fetch_info.emplace_back(
              ColumnLazyFetchInfo{true, local_col0_id, col0_ti});
        }
      } else {
        auto local_col_id = plan_state_->getLocalColumnId(col_var, false);
        const auto& col_ti = col_var->get_type_info();
        col_lazy_fetch_info.emplace_back(ColumnLazyFetchInfo{true, local_col_id, col_ti});
      }
    }
  }
  return col_lazy_fetch_info;
}

void Executor::clearMetaInfoCache() {
  input_table_info_cache_.clear();
  agg_col_range_cache_.clear();
  table_generations_.clear();
}

std::vector<int8_t> Executor::serializeLiterals(
    const std::unordered_map<int, CgenState::LiteralValues>& literals,
    const int device_id) {
  if (literals.empty()) {
    return {};
  }
  const auto dev_literals_it = literals.find(device_id);
  CHECK(dev_literals_it != literals.end());
  const auto& dev_literals = dev_literals_it->second;
  size_t lit_buf_size{0};
  std::vector<std::string> real_strings;
  std::vector<std::vector<double>> double_array_literals;
  std::vector<std::vector<int8_t>> align64_int8_array_literals;
  std::vector<std::vector<int32_t>> int32_array_literals;
  std::vector<std::vector<int8_t>> align32_int8_array_literals;
  std::vector<std::vector<int8_t>> int8_array_literals;
  for (const auto& lit : dev_literals) {
    lit_buf_size = CgenState::addAligned(lit_buf_size, CgenState::literalBytes(lit));
    if (lit.which() == 7) {
      const auto p = boost::get<std::string>(&lit);
      CHECK(p);
      real_strings.push_back(*p);
    } else if (lit.which() == 8) {
      const auto p = boost::get<std::vector<double>>(&lit);
      CHECK(p);
      double_array_literals.push_back(*p);
    } else if (lit.which() == 9) {
      const auto p = boost::get<std::vector<int32_t>>(&lit);
      CHECK(p);
      int32_array_literals.push_back(*p);
    } else if (lit.which() == 10) {
      const auto p = boost::get<std::vector<int8_t>>(&lit);
      CHECK(p);
      int8_array_literals.push_back(*p);
    } else if (lit.which() == 11) {
      const auto p = boost::get<std::pair<std::vector<int8_t>, int>>(&lit);
      CHECK(p);
      if (p->second == 64) {
        align64_int8_array_literals.push_back(p->first);
      } else if (p->second == 32) {
        align32_int8_array_literals.push_back(p->first);
      } else {
        CHECK(false);
      }
    }
  }
  if (lit_buf_size > static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
    throw TooManyLiterals();
  }
  int16_t crt_real_str_off = lit_buf_size;
  for (const auto& real_str : real_strings) {
    CHECK_LE(real_str.size(), static_cast<size_t>(std::numeric_limits<int16_t>::max()));
    lit_buf_size += real_str.size();
  }
  if (double_array_literals.size() > 0) {
    lit_buf_size = align(lit_buf_size, sizeof(double));
  }
  int16_t crt_double_arr_lit_off = lit_buf_size;
  for (const auto& double_array_literal : double_array_literals) {
    CHECK_LE(double_array_literal.size(),
             static_cast<size_t>(std::numeric_limits<int16_t>::max()));
    lit_buf_size += double_array_literal.size() * sizeof(double);
  }
  if (align64_int8_array_literals.size() > 0) {
    lit_buf_size = align(lit_buf_size, sizeof(uint64_t));
  }
  int16_t crt_align64_int8_arr_lit_off = lit_buf_size;
  for (const auto& align64_int8_array_literal : align64_int8_array_literals) {
    CHECK_LE(align64_int8_array_literals.size(),
             static_cast<size_t>(std::numeric_limits<int16_t>::max()));
    lit_buf_size += align64_int8_array_literal.size();
  }
  if (int32_array_literals.size() > 0) {
    lit_buf_size = align(lit_buf_size, sizeof(int32_t));
  }
  int16_t crt_int32_arr_lit_off = lit_buf_size;
  for (const auto& int32_array_literal : int32_array_literals) {
    CHECK_LE(int32_array_literal.size(),
             static_cast<size_t>(std::numeric_limits<int16_t>::max()));
    lit_buf_size += int32_array_literal.size() * sizeof(int32_t);
  }
  if (align32_int8_array_literals.size() > 0) {
    lit_buf_size = align(lit_buf_size, sizeof(int32_t));
  }
  int16_t crt_align32_int8_arr_lit_off = lit_buf_size;
  for (const auto& align32_int8_array_literal : align32_int8_array_literals) {
    CHECK_LE(align32_int8_array_literals.size(),
             static_cast<size_t>(std::numeric_limits<int16_t>::max()));
    lit_buf_size += align32_int8_array_literal.size();
  }
  int16_t crt_int8_arr_lit_off = lit_buf_size;
  for (const auto& int8_array_literal : int8_array_literals) {
    CHECK_LE(int8_array_literal.size(),
             static_cast<size_t>(std::numeric_limits<int16_t>::max()));
    lit_buf_size += int8_array_literal.size();
  }
  unsigned crt_real_str_idx = 0;
  unsigned crt_double_arr_lit_idx = 0;
  unsigned crt_align64_int8_arr_lit_idx = 0;
  unsigned crt_int32_arr_lit_idx = 0;
  unsigned crt_align32_int8_arr_lit_idx = 0;
  unsigned crt_int8_arr_lit_idx = 0;
  std::vector<int8_t> serialized(lit_buf_size);
  size_t off{0};
  for (const auto& lit : dev_literals) {
    const auto lit_bytes = CgenState::literalBytes(lit);
    off = CgenState::addAligned(off, lit_bytes);
    switch (lit.which()) {
      case 0: {
        const auto p = boost::get<int8_t>(&lit);
        CHECK(p);
        serialized[off - lit_bytes] = *p;
        break;
      }
      case 1: {
        const auto p = boost::get<int16_t>(&lit);
        CHECK(p);
        memcpy(&serialized[off - lit_bytes], p, lit_bytes);
        break;
      }
      case 2: {
        const auto p = boost::get<int32_t>(&lit);
        CHECK(p);
        memcpy(&serialized[off - lit_bytes], p, lit_bytes);
        break;
      }
      case 3: {
        const auto p = boost::get<int64_t>(&lit);
        CHECK(p);
        memcpy(&serialized[off - lit_bytes], p, lit_bytes);
        break;
      }
      case 4: {
        const auto p = boost::get<float>(&lit);
        CHECK(p);
        memcpy(&serialized[off - lit_bytes], p, lit_bytes);
        break;
      }
      case 5: {
        const auto p = boost::get<double>(&lit);
        CHECK(p);
        memcpy(&serialized[off - lit_bytes], p, lit_bytes);
        break;
      }
      case 6: {
        const auto p = boost::get<std::pair<std::string, int>>(&lit);
        CHECK(p);
        const auto str_id =
            g_enable_experimental_string_functions
                ? getStringDictionaryProxy(p->second, row_set_mem_owner_, true)
                      ->getOrAddTransient(p->first)
                : getStringDictionaryProxy(p->second, row_set_mem_owner_, true)
                      ->getIdOfString(p->first);
        memcpy(&serialized[off - lit_bytes], &str_id, lit_bytes);
        break;
      }
      case 7: {
        const auto p = boost::get<std::string>(&lit);
        CHECK(p);
        int32_t off_and_len = crt_real_str_off << 16;
        const auto& crt_real_str = real_strings[crt_real_str_idx];
        off_and_len |= static_cast<int16_t>(crt_real_str.size());
        memcpy(&serialized[off - lit_bytes], &off_and_len, lit_bytes);
        memcpy(&serialized[crt_real_str_off], crt_real_str.data(), crt_real_str.size());
        ++crt_real_str_idx;
        crt_real_str_off += crt_real_str.size();
        break;
      }
      case 8: {
        const auto p = boost::get<std::vector<double>>(&lit);
        CHECK(p);
        int32_t off_and_len = crt_double_arr_lit_off << 16;
        const auto& crt_double_arr_lit = double_array_literals[crt_double_arr_lit_idx];
        int32_t len = crt_double_arr_lit.size();
        CHECK_EQ((len >> 16), 0);
        off_and_len |= static_cast<int16_t>(len);
        int32_t double_array_bytesize = len * sizeof(double);
        memcpy(&serialized[off - lit_bytes], &off_and_len, lit_bytes);
        memcpy(&serialized[crt_double_arr_lit_off],
               crt_double_arr_lit.data(),
               double_array_bytesize);
        ++crt_double_arr_lit_idx;
        crt_double_arr_lit_off += double_array_bytesize;
        break;
      }
      case 9: {
        const auto p = boost::get<std::vector<int32_t>>(&lit);
        CHECK(p);
        int32_t off_and_len = crt_int32_arr_lit_off << 16;
        const auto& crt_int32_arr_lit = int32_array_literals[crt_int32_arr_lit_idx];
        int32_t len = crt_int32_arr_lit.size();
        CHECK_EQ((len >> 16), 0);
        off_and_len |= static_cast<int16_t>(len);
        int32_t int32_array_bytesize = len * sizeof(int32_t);
        memcpy(&serialized[off - lit_bytes], &off_and_len, lit_bytes);
        memcpy(&serialized[crt_int32_arr_lit_off],
               crt_int32_arr_lit.data(),
               int32_array_bytesize);
        ++crt_int32_arr_lit_idx;
        crt_int32_arr_lit_off += int32_array_bytesize;
        break;
      }
      case 10: {
        const auto p = boost::get<std::vector<int8_t>>(&lit);
        CHECK(p);
        int32_t off_and_len = crt_int8_arr_lit_off << 16;
        const auto& crt_int8_arr_lit = int8_array_literals[crt_int8_arr_lit_idx];
        int32_t len = crt_int8_arr_lit.size();
        CHECK_EQ((len >> 16), 0);
        off_and_len |= static_cast<int16_t>(len);
        int32_t int8_array_bytesize = len;
        memcpy(&serialized[off - lit_bytes], &off_and_len, lit_bytes);
        memcpy(&serialized[crt_int8_arr_lit_off],
               crt_int8_arr_lit.data(),
               int8_array_bytesize);
        ++crt_int8_arr_lit_idx;
        crt_int8_arr_lit_off += int8_array_bytesize;
        break;
      }
      case 11: {
        const auto p = boost::get<std::pair<std::vector<int8_t>, int>>(&lit);
        CHECK(p);
        if (p->second == 64) {
          int32_t off_and_len = crt_align64_int8_arr_lit_off << 16;
          const auto& crt_align64_int8_arr_lit =
              align64_int8_array_literals[crt_align64_int8_arr_lit_idx];
          int32_t len = crt_align64_int8_arr_lit.size();
          CHECK_EQ((len >> 16), 0);
          off_and_len |= static_cast<int16_t>(len);
          int32_t align64_int8_array_bytesize = len;
          memcpy(&serialized[off - lit_bytes], &off_and_len, lit_bytes);
          memcpy(&serialized[crt_align64_int8_arr_lit_off],
                 crt_align64_int8_arr_lit.data(),
                 align64_int8_array_bytesize);
          ++crt_align64_int8_arr_lit_idx;
          crt_align64_int8_arr_lit_off += align64_int8_array_bytesize;
        } else if (p->second == 32) {
          int32_t off_and_len = crt_align32_int8_arr_lit_off << 16;
          const auto& crt_align32_int8_arr_lit =
              align32_int8_array_literals[crt_align32_int8_arr_lit_idx];
          int32_t len = crt_align32_int8_arr_lit.size();
          CHECK_EQ((len >> 16), 0);
          off_and_len |= static_cast<int16_t>(len);
          int32_t align32_int8_array_bytesize = len;
          memcpy(&serialized[off - lit_bytes], &off_and_len, lit_bytes);
          memcpy(&serialized[crt_align32_int8_arr_lit_off],
                 crt_align32_int8_arr_lit.data(),
                 align32_int8_array_bytesize);
          ++crt_align32_int8_arr_lit_idx;
          crt_align32_int8_arr_lit_off += align32_int8_array_bytesize;
        } else {
          CHECK(false);
        }
        break;
      }
      default:
        CHECK(false);
    }
  }
  return serialized;
}

int Executor::deviceCount(const ExecutorDeviceType device_type) const {
  if (device_type == ExecutorDeviceType::GPU) {
    const auto cuda_mgr = catalog_->getDataMgr().getCudaMgr();
    CHECK(cuda_mgr);
    return cuda_mgr->getDeviceCount();
  } else {
    return 1;
  }
}

int Executor::deviceCountForMemoryLevel(
    const Data_Namespace::MemoryLevel memory_level) const {
  return memory_level == GPU_LEVEL ? deviceCount(ExecutorDeviceType::GPU)
                                   : deviceCount(ExecutorDeviceType::CPU);
}

// TODO(alex): remove or split
std::pair<int64_t, int32_t> Executor::reduceResults(const SQLAgg agg,
                                                    const SQLTypeInfo& ti,
                                                    const int64_t agg_init_val,
                                                    const int8_t out_byte_width,
                                                    const int64_t* out_vec,
                                                    const size_t out_vec_sz,
                                                    const bool is_group_by,
                                                    const bool float_argument_input) {
  switch (agg) {
    case kAVG:
    case kSUM:
      if (0 != agg_init_val) {
        if (ti.is_integer() || ti.is_decimal() || ti.is_time() || ti.is_boolean()) {
          int64_t agg_result = agg_init_val;
          for (size_t i = 0; i < out_vec_sz; ++i) {
            agg_sum_skip_val(&agg_result, out_vec[i], agg_init_val);
          }
          return {agg_result, 0};
        } else {
          CHECK(ti.is_fp());
          switch (out_byte_width) {
            case 4: {
              int agg_result = static_cast<int32_t>(agg_init_val);
              for (size_t i = 0; i < out_vec_sz; ++i) {
                agg_sum_float_skip_val(
                    &agg_result,
                    *reinterpret_cast<const float*>(may_alias_ptr(&out_vec[i])),
                    *reinterpret_cast<const float*>(may_alias_ptr(&agg_init_val)));
              }
              const int64_t converted_bin =
                  float_argument_input
                      ? static_cast<int64_t>(agg_result)
                      : float_to_double_bin(static_cast<int32_t>(agg_result), true);
              return {converted_bin, 0};
              break;
            }
            case 8: {
              int64_t agg_result = agg_init_val;
              for (size_t i = 0; i < out_vec_sz; ++i) {
                agg_sum_double_skip_val(
                    &agg_result,
                    *reinterpret_cast<const double*>(may_alias_ptr(&out_vec[i])),
                    *reinterpret_cast<const double*>(may_alias_ptr(&agg_init_val)));
              }
              return {agg_result, 0};
              break;
            }
            default:
              CHECK(false);
          }
        }
      }
      if (ti.is_integer() || ti.is_decimal() || ti.is_time()) {
        int64_t agg_result = 0;
        for (size_t i = 0; i < out_vec_sz; ++i) {
          agg_result += out_vec[i];
        }
        return {agg_result, 0};
      } else {
        CHECK(ti.is_fp());
        switch (out_byte_width) {
          case 4: {
            float r = 0.;
            for (size_t i = 0; i < out_vec_sz; ++i) {
              r += *reinterpret_cast<const float*>(may_alias_ptr(&out_vec[i]));
            }
            const auto float_bin = *reinterpret_cast<const int32_t*>(may_alias_ptr(&r));
            const int64_t converted_bin =
                float_argument_input ? float_bin : float_to_double_bin(float_bin, true);
            return {converted_bin, 0};
          }
          case 8: {
            double r = 0.;
            for (size_t i = 0; i < out_vec_sz; ++i) {
              r += *reinterpret_cast<const double*>(may_alias_ptr(&out_vec[i]));
            }
            return {*reinterpret_cast<const int64_t*>(may_alias_ptr(&r)), 0};
          }
          default:
            CHECK(false);
        }
      }
      break;
    case kCOUNT: {
      uint64_t agg_result = 0;
      for (size_t i = 0; i < out_vec_sz; ++i) {
        const uint64_t out = static_cast<uint64_t>(out_vec[i]);
        agg_result += out;
      }
      return {static_cast<int64_t>(agg_result), 0};
    }
    case kMIN: {
      if (ti.is_integer() || ti.is_decimal() || ti.is_time() || ti.is_boolean()) {
        int64_t agg_result = agg_init_val;
        for (size_t i = 0; i < out_vec_sz; ++i) {
          agg_min_skip_val(&agg_result, out_vec[i], agg_init_val);
        }
        return {agg_result, 0};
      } else {
        switch (out_byte_width) {
          case 4: {
            int32_t agg_result = static_cast<int32_t>(agg_init_val);
            for (size_t i = 0; i < out_vec_sz; ++i) {
              agg_min_float_skip_val(
                  &agg_result,
                  *reinterpret_cast<const float*>(may_alias_ptr(&out_vec[i])),
                  *reinterpret_cast<const float*>(may_alias_ptr(&agg_init_val)));
            }
            const int64_t converted_bin =
                float_argument_input
                    ? static_cast<int64_t>(agg_result)
                    : float_to_double_bin(static_cast<int32_t>(agg_result), true);
            return {converted_bin, 0};
          }
          case 8: {
            int64_t agg_result = agg_init_val;
            for (size_t i = 0; i < out_vec_sz; ++i) {
              agg_min_double_skip_val(
                  &agg_result,
                  *reinterpret_cast<const double*>(may_alias_ptr(&out_vec[i])),
                  *reinterpret_cast<const double*>(may_alias_ptr(&agg_init_val)));
            }
            return {agg_result, 0};
          }
          default:
            CHECK(false);
        }
      }
    }
    case kMAX:
      if (ti.is_integer() || ti.is_decimal() || ti.is_time() || ti.is_boolean()) {
        int64_t agg_result = agg_init_val;
        for (size_t i = 0; i < out_vec_sz; ++i) {
          agg_max_skip_val(&agg_result, out_vec[i], agg_init_val);
        }
        return {agg_result, 0};
      } else {
        switch (out_byte_width) {
          case 4: {
            int32_t agg_result = static_cast<int32_t>(agg_init_val);
            for (size_t i = 0; i < out_vec_sz; ++i) {
              agg_max_float_skip_val(
                  &agg_result,
                  *reinterpret_cast<const float*>(may_alias_ptr(&out_vec[i])),
                  *reinterpret_cast<const float*>(may_alias_ptr(&agg_init_val)));
            }
            const int64_t converted_bin =
                float_argument_input ? static_cast<int64_t>(agg_result)
                                     : float_to_double_bin(agg_result, !ti.get_notnull());
            return {converted_bin, 0};
          }
          case 8: {
            int64_t agg_result = agg_init_val;
            for (size_t i = 0; i < out_vec_sz; ++i) {
              agg_max_double_skip_val(
                  &agg_result,
                  *reinterpret_cast<const double*>(may_alias_ptr(&out_vec[i])),
                  *reinterpret_cast<const double*>(may_alias_ptr(&agg_init_val)));
            }
            return {agg_result, 0};
          }
          default:
            CHECK(false);
        }
      }
    case kSINGLE_VALUE: {
      int64_t agg_result = agg_init_val;
      for (size_t i = 0; i < out_vec_sz; ++i) {
        if (out_vec[i] != agg_init_val) {
          if (agg_result == agg_init_val) {
            agg_result = out_vec[i];
          } else if (out_vec[i] != agg_result) {
            return {agg_result, Executor::ERR_SINGLE_VALUE_FOUND_MULTIPLE_VALUES};
          }
        }
      }
      return {agg_result, 0};
    }
    case kSAMPLE: {
      int64_t agg_result = agg_init_val;
      for (size_t i = 0; i < out_vec_sz; ++i) {
        if (out_vec[i] != agg_init_val) {
          agg_result = out_vec[i];
          break;
        }
      }
      return {agg_result, 0};
    }
    default:
      CHECK(false);
  }
  abort();
}

namespace {

ResultSetPtr get_merged_result(
    std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device) {
  auto& first = results_per_device.front().first;
  CHECK(first);
  for (size_t dev_idx = 1; dev_idx < results_per_device.size(); ++dev_idx) {
    const auto& next = results_per_device[dev_idx].first;
    CHECK(next);
    first->append(*next);
  }
  return std::move(first);
}

}  // namespace

ResultSetPtr Executor::resultsUnion(SharedKernelContext& shared_context,
                                    const RelAlgExecutionUnit& ra_exe_unit) {
  auto& results_per_device = shared_context.getFragmentResults();
  if (results_per_device.empty()) {
    std::vector<TargetInfo> targets;
    for (const auto target_expr : ra_exe_unit.target_exprs) {
      targets.push_back(get_target_info(target_expr, g_bigint_count));
    }
    return std::make_shared<ResultSet>(targets,
                                       ExecutorDeviceType::CPU,
                                       QueryMemoryDescriptor(),
                                       row_set_mem_owner_,
                                       catalog_,
                                       blockSize(),
                                       gridSize());
  }
  using IndexedResultSet = std::pair<ResultSetPtr, std::vector<size_t>>;
  std::sort(results_per_device.begin(),
            results_per_device.end(),
            [](const IndexedResultSet& lhs, const IndexedResultSet& rhs) {
              CHECK_GE(lhs.second.size(), size_t(1));
              CHECK_GE(rhs.second.size(), size_t(1));
              return lhs.second.front() < rhs.second.front();
            });

  return get_merged_result(results_per_device);
}

ResultSetPtr Executor::reduceMultiDeviceResults(
    const RelAlgExecutionUnit& ra_exe_unit,
    std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const QueryMemoryDescriptor& query_mem_desc) const {
  auto timer = DEBUG_TIMER(__func__);
  if (ra_exe_unit.estimator) {
    return reduce_estimator_results(ra_exe_unit, results_per_device);
  }

  if (results_per_device.empty()) {
    std::vector<TargetInfo> targets;
    for (const auto target_expr : ra_exe_unit.target_exprs) {
      targets.push_back(get_target_info(target_expr, g_bigint_count));
    }
    return std::make_shared<ResultSet>(targets,
                                       ExecutorDeviceType::CPU,
                                       QueryMemoryDescriptor(),
                                       nullptr,
                                       catalog_,
                                       blockSize(),
                                       gridSize());
  }

  return reduceMultiDeviceResultSets(
      results_per_device,
      row_set_mem_owner,
      ResultSet::fixupQueryMemoryDescriptor(query_mem_desc));
}

namespace {

ReductionCode get_reduction_code(
    std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device,
    int64_t* compilation_queue_time) {
  auto clock_begin = timer_start();
  std::lock_guard<std::mutex> compilation_lock(Executor::compilation_mutex_);
  *compilation_queue_time = timer_stop(clock_begin);
  const auto& this_result_set = results_per_device[0].first;
  ResultSetReductionJIT reduction_jit(this_result_set->getQueryMemDesc(),
                                      this_result_set->getTargetInfos(),
                                      this_result_set->getTargetInitVals());
  return reduction_jit.codegen();
};

}  // namespace

ResultSetPtr Executor::reduceMultiDeviceResultSets(
    std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const QueryMemoryDescriptor& query_mem_desc) const {
  auto timer = DEBUG_TIMER(__func__);
  std::shared_ptr<ResultSet> reduced_results;

  const auto& first = results_per_device.front().first;

  if (query_mem_desc.getQueryDescriptionType() ==
          QueryDescriptionType::GroupByBaselineHash &&
      results_per_device.size() > 1) {
    const auto total_entry_count = std::accumulate(
        results_per_device.begin(),
        results_per_device.end(),
        size_t(0),
        [](const size_t init, const std::pair<ResultSetPtr, std::vector<size_t>>& rs) {
          const auto& r = rs.first;
          return init + r->getQueryMemDesc().getEntryCount();
        });
    CHECK(total_entry_count);
    auto query_mem_desc = first->getQueryMemDesc();
    query_mem_desc.setEntryCount(total_entry_count);
    reduced_results = std::make_shared<ResultSet>(first->getTargetInfos(),
                                                  ExecutorDeviceType::CPU,
                                                  query_mem_desc,
                                                  row_set_mem_owner,
                                                  catalog_,
                                                  blockSize(),
                                                  gridSize());
    auto result_storage = reduced_results->allocateStorage(plan_state_->init_agg_vals_);
    reduced_results->initializeStorage();
    switch (query_mem_desc.getEffectiveKeyWidth()) {
      case 4:
        first->getStorage()->moveEntriesToBuffer<int32_t>(
            result_storage->getUnderlyingBuffer(), query_mem_desc.getEntryCount());
        break;
      case 8:
        first->getStorage()->moveEntriesToBuffer<int64_t>(
            result_storage->getUnderlyingBuffer(), query_mem_desc.getEntryCount());
        break;
      default:
        CHECK(false);
    }
  } else {
    reduced_results = first;
  }

  int64_t compilation_queue_time = 0;
  const auto reduction_code =
      get_reduction_code(results_per_device, &compilation_queue_time);

  for (size_t i = 1; i < results_per_device.size(); ++i) {
    reduced_results->getStorage()->reduce(
        *(results_per_device[i].first->getStorage()), {}, reduction_code);
  }
  reduced_results->addCompilationQueueTime(compilation_queue_time);
  return reduced_results;
}

ResultSetPtr Executor::reduceSpeculativeTopN(
    const RelAlgExecutionUnit& ra_exe_unit,
    std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const QueryMemoryDescriptor& query_mem_desc) const {
  if (results_per_device.size() == 1) {
    return std::move(results_per_device.front().first);
  }
  const auto top_n = ra_exe_unit.sort_info.limit + ra_exe_unit.sort_info.offset;
  SpeculativeTopNMap m;
  for (const auto& result : results_per_device) {
    auto rows = result.first;
    CHECK(rows);
    if (!rows) {
      continue;
    }
    SpeculativeTopNMap that(
        *rows,
        ra_exe_unit.target_exprs,
        std::max(size_t(10000 * std::max(1, static_cast<int>(log(top_n)))), top_n));
    m.reduce(that);
  }
  CHECK_EQ(size_t(1), ra_exe_unit.sort_info.order_entries.size());
  const auto desc = ra_exe_unit.sort_info.order_entries.front().is_desc;
  return m.asRows(ra_exe_unit, row_set_mem_owner, query_mem_desc, this, top_n, desc);
}

std::unordered_set<int> get_available_gpus(const Catalog_Namespace::Catalog& cat) {
  std::unordered_set<int> available_gpus;
  if (cat.getDataMgr().gpusPresent()) {
    int gpu_count = cat.getDataMgr().getCudaMgr()->getDeviceCount();
    CHECK_GT(gpu_count, 0);
    for (int gpu_id = 0; gpu_id < gpu_count; ++gpu_id) {
      available_gpus.insert(gpu_id);
    }
  }
  return available_gpus;
}

size_t get_context_count(const ExecutorDeviceType device_type,
                         const size_t cpu_count,
                         const size_t gpu_count) {
  return device_type == ExecutorDeviceType::GPU ? gpu_count
                                                : static_cast<size_t>(cpu_count);
}

namespace {

// Compute a very conservative entry count for the output buffer entry count using no
// other information than the number of tuples in each table and multiplying them
// together.
size_t compute_buffer_entry_guess(const std::vector<InputTableInfo>& query_infos) {
  using Fragmenter_Namespace::FragmentInfo;
  // Check for overflows since we're multiplying potentially big table sizes.
  using checked_size_t = boost::multiprecision::number<
      boost::multiprecision::cpp_int_backend<64,
                                             64,
                                             boost::multiprecision::unsigned_magnitude,
                                             boost::multiprecision::checked,
                                             void>>;
  checked_size_t max_groups_buffer_entry_guess = 1;
  for (const auto& query_info : query_infos) {
    CHECK(!query_info.info.fragments.empty());
    auto it = std::max_element(query_info.info.fragments.begin(),
                               query_info.info.fragments.end(),
                               [](const FragmentInfo& f1, const FragmentInfo& f2) {
                                 return f1.getNumTuples() < f2.getNumTuples();
                               });
    max_groups_buffer_entry_guess *= it->getNumTuples();
  }
  // Cap the rough approximation to 100M entries, it's unlikely we can do a great job for
  // baseline group layout with that many entries anyway.
  constexpr size_t max_groups_buffer_entry_guess_cap = 100000000;
  try {
    return std::min(static_cast<size_t>(max_groups_buffer_entry_guess),
                    max_groups_buffer_entry_guess_cap);
  } catch (...) {
    return max_groups_buffer_entry_guess_cap;
  }
}

std::string get_table_name(const InputDescriptor& input_desc,
                           const Catalog_Namespace::Catalog& cat) {
  const auto source_type = input_desc.getSourceType();
  if (source_type == InputSourceType::TABLE) {
    const auto td = cat.getMetadataForTable(input_desc.getTableId());
    CHECK(td);
    return td->tableName;
  } else {
    return "$TEMPORARY_TABLE" + std::to_string(-input_desc.getTableId());
  }
}

inline size_t getDeviceBasedScanLimit(const ExecutorDeviceType device_type,
                                      const int device_count) {
  if (device_type == ExecutorDeviceType::GPU) {
    return device_count * Executor::high_scan_limit;
  }
  return Executor::high_scan_limit;
}

void checkWorkUnitWatchdog(const RelAlgExecutionUnit& ra_exe_unit,
                           const std::vector<InputTableInfo>& table_infos,
                           const Catalog_Namespace::Catalog& cat,
                           const ExecutorDeviceType device_type,
                           const int device_count) {
  for (const auto target_expr : ra_exe_unit.target_exprs) {
    if (dynamic_cast<const Analyzer::AggExpr*>(target_expr)) {
      return;
    }
  }
  if (!ra_exe_unit.scan_limit && table_infos.size() == 1 &&
      table_infos.front().info.getPhysicalNumTuples() < Executor::high_scan_limit) {
    // Allow a query with no scan limit to run on small tables
    return;
  }
  if (ra_exe_unit.use_bump_allocator) {
    // Bump allocator removes the scan limit (and any knowledge of the size of the output
    // relative to the size of the input), so we bypass this check for now
    return;
  }
  if (ra_exe_unit.sort_info.algorithm != SortAlgorithm::StreamingTopN &&
      ra_exe_unit.groupby_exprs.size() == 1 && !ra_exe_unit.groupby_exprs.front() &&
      (!ra_exe_unit.scan_limit ||
       ra_exe_unit.scan_limit > getDeviceBasedScanLimit(device_type, device_count))) {
    std::vector<std::string> table_names;
    const auto& input_descs = ra_exe_unit.input_descs;
    for (const auto& input_desc : input_descs) {
      table_names.push_back(get_table_name(input_desc, cat));
    }
    if (!ra_exe_unit.scan_limit) {
      throw WatchdogException(
          "Projection query would require a scan without a limit on table(s): " +
          boost::algorithm::join(table_names, ", "));
    } else {
      throw WatchdogException(
          "Projection query output result set on table(s): " +
          boost::algorithm::join(table_names, ", ") + "  would contain " +
          std::to_string(ra_exe_unit.scan_limit) +
          " rows, which is more than the current system limit of " +
          std::to_string(getDeviceBasedScanLimit(device_type, device_count)));
    }
  }
}

}  // namespace

bool is_trivial_loop_join(const std::vector<InputTableInfo>& query_infos,
                          const RelAlgExecutionUnit& ra_exe_unit) {
  if (ra_exe_unit.input_descs.size() < 2) {
    return false;
  }

  // We only support loop join at the end of folded joins
  // where ra_exe_unit.input_descs.size() > 2 for now.
  const auto inner_table_id = ra_exe_unit.input_descs.back().getTableId();

  std::optional<size_t> inner_table_idx;
  for (size_t i = 0; i < query_infos.size(); ++i) {
    if (query_infos[i].table_id == inner_table_id) {
      inner_table_idx = i;
      break;
    }
  }
  CHECK(inner_table_idx);
  return query_infos[*inner_table_idx].info.getNumTuples() <=
         g_trivial_loop_join_threshold;
}

namespace {

template <typename T>
std::vector<std::string> expr_container_to_string(const T& expr_container) {
  std::vector<std::string> expr_strs;
  for (const auto& expr : expr_container) {
    if (!expr) {
      expr_strs.emplace_back("NULL");
    } else {
      expr_strs.emplace_back(expr->toString());
    }
  }
  return expr_strs;
}

template <>
std::vector<std::string> expr_container_to_string(
    const std::list<Analyzer::OrderEntry>& expr_container) {
  std::vector<std::string> expr_strs;
  for (const auto& expr : expr_container) {
    expr_strs.emplace_back(expr.toString());
  }
  return expr_strs;
}

std::string join_type_to_string(const JoinType type) {
  switch (type) {
    case JoinType::INNER:
      return "INNER";
    case JoinType::LEFT:
      return "LEFT";
    case JoinType::SEMI:
      return "SEMI";
    case JoinType::ANTI:
      return "ANTI";
    case JoinType::INVALID:
      return "INVALID";
  }
  UNREACHABLE();
  return "";
}

std::string sort_algorithm_to_string(const SortAlgorithm algorithm) {
  switch (algorithm) {
    case SortAlgorithm::Default:
      return "ResultSet";
    case SortAlgorithm::SpeculativeTopN:
      return "Speculative Top N";
    case SortAlgorithm::StreamingTopN:
      return "Streaming Top N";
  }
  UNREACHABLE();
  return "";
}

}  // namespace

std::string ra_exec_unit_desc_for_caching(const RelAlgExecutionUnit& ra_exe_unit) {
  // todo(yoonmin): replace a cache key as a DAG representation of a query plan
  // instead of ra_exec_unit description if possible
  std::ostringstream os;
  for (const auto& input_col_desc : ra_exe_unit.input_col_descs) {
    const auto& scan_desc = input_col_desc->getScanDesc();
    os << scan_desc.getTableId() << "," << input_col_desc->getColId() << ","
       << scan_desc.getNestLevel();
  }
  if (!ra_exe_unit.simple_quals.empty()) {
    for (const auto& qual : ra_exe_unit.simple_quals) {
      if (qual) {
        os << qual->toString() << ",";
      }
    }
  }
  if (!ra_exe_unit.quals.empty()) {
    for (const auto& qual : ra_exe_unit.quals) {
      if (qual) {
        os << qual->toString() << ",";
      }
    }
  }
  if (!ra_exe_unit.join_quals.empty()) {
    for (size_t i = 0; i < ra_exe_unit.join_quals.size(); i++) {
      const auto& join_condition = ra_exe_unit.join_quals[i];
      os << std::to_string(i) << join_type_to_string(join_condition.type);
      for (const auto& qual : join_condition.quals) {
        if (qual) {
          os << qual->toString() << ",";
        }
      }
    }
  }
  if (!ra_exe_unit.groupby_exprs.empty()) {
    for (const auto& qual : ra_exe_unit.groupby_exprs) {
      if (qual) {
        os << qual->toString() << ",";
      }
    }
  }
  for (const auto& expr : ra_exe_unit.target_exprs) {
    if (expr) {
      os << expr->toString() << ",";
    }
  }
  os << ::toString(ra_exe_unit.estimator == nullptr);
  os << std::to_string(ra_exe_unit.scan_limit);
  return os.str();
}

std::ostream& operator<<(std::ostream& os, const RelAlgExecutionUnit& ra_exe_unit) {
  os << "\n\tTable/Col/Levels: ";
  for (const auto& input_col_desc : ra_exe_unit.input_col_descs) {
    const auto& scan_desc = input_col_desc->getScanDesc();
    os << "(" << scan_desc.getTableId() << ", " << input_col_desc->getColId() << ", "
       << scan_desc.getNestLevel() << ") ";
  }
  if (!ra_exe_unit.simple_quals.empty()) {
    os << "\n\tSimple Quals: "
       << boost::algorithm::join(expr_container_to_string(ra_exe_unit.simple_quals),
                                 ", ");
  }
  if (!ra_exe_unit.quals.empty()) {
    os << "\n\tQuals: "
       << boost::algorithm::join(expr_container_to_string(ra_exe_unit.quals), ", ");
  }
  if (!ra_exe_unit.join_quals.empty()) {
    os << "\n\tJoin Quals: ";
    for (size_t i = 0; i < ra_exe_unit.join_quals.size(); i++) {
      const auto& join_condition = ra_exe_unit.join_quals[i];
      os << "\t\t" << std::to_string(i) << " "
         << join_type_to_string(join_condition.type);
      os << boost::algorithm::join(expr_container_to_string(join_condition.quals), ", ");
    }
  }
  if (!ra_exe_unit.groupby_exprs.empty()) {
    os << "\n\tGroup By: "
       << boost::algorithm::join(expr_container_to_string(ra_exe_unit.groupby_exprs),
                                 ", ");
  }
  os << "\n\tProjected targets: "
     << boost::algorithm::join(expr_container_to_string(ra_exe_unit.target_exprs), ", ");
  os << "\n\tHas Estimator: " << ::toString(ra_exe_unit.estimator == nullptr);
  os << "\n\tSort Info: ";
  const auto& sort_info = ra_exe_unit.sort_info;
  os << "\n\t  Order Entries: "
     << boost::algorithm::join(expr_container_to_string(sort_info.order_entries), ", ");
  os << "\n\t  Algorithm: " << sort_algorithm_to_string(sort_info.algorithm);
  os << "\n\t  Limit: " << std::to_string(sort_info.limit);
  os << "\n\t  Offset: " << std::to_string(sort_info.offset);
  os << "\n\tScan Limit: " << std::to_string(ra_exe_unit.scan_limit);
  os << "\n\tBump Allocator: " << ::toString(ra_exe_unit.use_bump_allocator);
  if (ra_exe_unit.union_all) {
    os << "\n\tUnion: " << std::string(*ra_exe_unit.union_all ? "UNION ALL" : "UNION");
  }
  return os;
}

namespace {

RelAlgExecutionUnit replace_scan_limit(const RelAlgExecutionUnit& ra_exe_unit_in,
                                       const size_t new_scan_limit) {
  return {ra_exe_unit_in.input_descs,
          ra_exe_unit_in.input_col_descs,
          ra_exe_unit_in.simple_quals,
          ra_exe_unit_in.quals,
          ra_exe_unit_in.join_quals,
          ra_exe_unit_in.groupby_exprs,
          ra_exe_unit_in.target_exprs,
          ra_exe_unit_in.estimator,
          ra_exe_unit_in.sort_info,
          new_scan_limit,
          ra_exe_unit_in.query_hint,
          ra_exe_unit_in.use_bump_allocator,
          ra_exe_unit_in.union_all,
          ra_exe_unit_in.query_state};
}

}  // namespace

ResultSetPtr Executor::executeWorkUnit(size_t& max_groups_buffer_entry_guess,
                                       const bool is_agg,
                                       const std::vector<InputTableInfo>& query_infos,
                                       const RelAlgExecutionUnit& ra_exe_unit_in,
                                       const CompilationOptions& co,
                                       const ExecutionOptions& eo,
                                       const Catalog_Namespace::Catalog& cat,
                                       RenderInfo* render_info,
                                       const bool has_cardinality_estimation,
                                       ColumnCacheMap& column_cache) {
  VLOG(1) << "Executor " << executor_id_ << " is executing work unit:" << ra_exe_unit_in;

  ScopeGuard cleanup_post_execution = [this] {
    // cleanup/unpin GPU buffer allocations
    // TODO: separate out this state into a single object
    plan_state_.reset(nullptr);
    if (cgen_state_) {
      cgen_state_->in_values_bitmaps_.clear();
    }
  };

  try {
    auto result = executeWorkUnitImpl(max_groups_buffer_entry_guess,
                                      is_agg,
                                      true,
                                      query_infos,
                                      ra_exe_unit_in,
                                      co,
                                      eo,
                                      cat,
                                      row_set_mem_owner_,
                                      render_info,
                                      has_cardinality_estimation,
                                      column_cache);
    if (result) {
      result->setKernelQueueTime(kernel_queue_time_ms_);
      result->addCompilationQueueTime(compilation_queue_time_ms_);
      if (eo.just_validate) {
        result->setValidationOnlyRes();
      }
    }
    return result;
  } catch (const CompilationRetryNewScanLimit& e) {
    auto result =
        executeWorkUnitImpl(max_groups_buffer_entry_guess,
                            is_agg,
                            false,
                            query_infos,
                            replace_scan_limit(ra_exe_unit_in, e.new_scan_limit_),
                            co,
                            eo,
                            cat,
                            row_set_mem_owner_,
                            render_info,
                            has_cardinality_estimation,
                            column_cache);
    if (result) {
      result->setKernelQueueTime(kernel_queue_time_ms_);
      result->addCompilationQueueTime(compilation_queue_time_ms_);
      if (eo.just_validate) {
        result->setValidationOnlyRes();
      }
    }
    return result;
  }
}

ResultSetPtr Executor::executeWorkUnitImpl(
    size_t& max_groups_buffer_entry_guess,
    const bool is_agg,
    const bool allow_single_frag_table_opt,
    const std::vector<InputTableInfo>& query_infos,
    const RelAlgExecutionUnit& ra_exe_unit_in,
    const CompilationOptions& co,
    const ExecutionOptions& eo,
    const Catalog_Namespace::Catalog& cat,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    RenderInfo* render_info,
    const bool has_cardinality_estimation,
    ColumnCacheMap& column_cache) {
  INJECT_TIMER(Exec_executeWorkUnit);
  const auto [ra_exe_unit, deleted_cols_map] = addDeletedColumn(ra_exe_unit_in, co);
  const auto device_type = getDeviceTypeForTargets(ra_exe_unit, co.device_type);
  CHECK(!query_infos.empty());
  if (!max_groups_buffer_entry_guess) {
    // The query has failed the first execution attempt because of running out
    // of group by slots. Make the conservative choice: allocate fragment size
    // slots and run on the CPU.
    CHECK(device_type == ExecutorDeviceType::CPU);
    max_groups_buffer_entry_guess = compute_buffer_entry_guess(query_infos);
  }

  int8_t crt_min_byte_width{MAX_BYTE_WIDTH_SUPPORTED};
  do {
    SharedKernelContext shared_context(query_infos);
    ColumnFetcher column_fetcher(this, column_cache);
    ScopeGuard scope_guard = [&column_fetcher, &device_type] {
      column_fetcher.freeLinearizedBuf();
      if (device_type == ExecutorDeviceType::GPU) {
        column_fetcher.freeTemporaryCpuLinearizedIdxBuf();
      }
    };
    auto query_comp_desc_owned = std::make_unique<QueryCompilationDescriptor>();
    std::unique_ptr<QueryMemoryDescriptor> query_mem_desc_owned;
    if (eo.executor_type == ExecutorType::Native) {
      try {
        INJECT_TIMER(query_step_compilation);
        auto clock_begin = timer_start();
        std::lock_guard<std::mutex> compilation_lock(compilation_mutex_);
        compilation_queue_time_ms_ += timer_stop(clock_begin);

        query_mem_desc_owned =
            query_comp_desc_owned->compile(max_groups_buffer_entry_guess,
                                           crt_min_byte_width,
                                           has_cardinality_estimation,
                                           ra_exe_unit,
                                           query_infos,
                                           deleted_cols_map,
                                           column_fetcher,
                                           {device_type,
                                            co.hoist_literals,
                                            co.opt_level,
                                            co.with_dynamic_watchdog,
                                            co.allow_lazy_fetch,
                                            co.filter_on_deleted_column,
                                            co.explain_type,
                                            co.register_intel_jit_listener},
                                           eo,
                                           render_info,
                                           this);
        CHECK(query_mem_desc_owned);
        crt_min_byte_width = query_comp_desc_owned->getMinByteWidth();
      } catch (CompilationRetryNoCompaction&) {
        crt_min_byte_width = MAX_BYTE_WIDTH_SUPPORTED;
        continue;
      }
    } else {
      plan_state_.reset(new PlanState(false, query_infos, deleted_cols_map, this));
      plan_state_->allocateLocalColumnIds(ra_exe_unit.input_col_descs);
      CHECK(!query_mem_desc_owned);
      query_mem_desc_owned.reset(
          new QueryMemoryDescriptor(this, 0, QueryDescriptionType::Projection, false));
    }
    if (eo.just_explain) {
      return executeExplain(*query_comp_desc_owned);
    }

    for (const auto target_expr : ra_exe_unit.target_exprs) {
      plan_state_->target_exprs_.push_back(target_expr);
    }

    if (!eo.just_validate) {
      int available_cpus = cpu_threads();
      auto available_gpus = get_available_gpus(cat);

      const auto context_count =
          get_context_count(device_type, available_cpus, available_gpus.size());
      try {
        auto kernels = createKernels(shared_context,
                                     ra_exe_unit,
                                     column_fetcher,
                                     query_infos,
                                     eo,
                                     is_agg,
                                     allow_single_frag_table_opt,
                                     context_count,
                                     *query_comp_desc_owned,
                                     *query_mem_desc_owned,
                                     render_info,
                                     available_gpus,
                                     available_cpus);
        if (g_use_tbb_pool) {
#ifdef HAVE_TBB
          VLOG(1) << "Using TBB thread pool for kernel dispatch.";
          launchKernels<threadpool::TbbThreadPool<void>>(shared_context,
                                                         std::move(kernels));
#else
          throw std::runtime_error(
              "This build is not TBB enabled. Restart the server with "
              "\"enable-modern-thread-pool\" disabled.");
#endif
        } else {
          launchKernels<threadpool::FuturesThreadPool<void>>(shared_context,
                                                             std::move(kernels));
        }
      } catch (QueryExecutionError& e) {
        if (eo.with_dynamic_watchdog && interrupted_.load() &&
            e.getErrorCode() == ERR_OUT_OF_TIME) {
          throw QueryExecutionError(ERR_INTERRUPTED);
        }
        if (e.getErrorCode() == ERR_INTERRUPTED) {
          throw QueryExecutionError(ERR_INTERRUPTED);
        }
        if (e.getErrorCode() == ERR_OVERFLOW_OR_UNDERFLOW &&
            static_cast<size_t>(crt_min_byte_width << 1) <= sizeof(int64_t)) {
          crt_min_byte_width <<= 1;
          continue;
        }
        throw;
      }
    }
    if (is_agg) {
      if (eo.allow_runtime_query_interrupt && ra_exe_unit.query_state) {
        // update query status to let user know we are now in the reduction phase
        std::string curRunningSession{""};
        std::string curRunningQuerySubmittedTime{""};
        bool sessionEnrolled = false;
        auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
        {
          mapd_shared_lock<mapd_shared_mutex> session_read_lock(
              executor->getSessionLock());
          curRunningSession = executor->getCurrentQuerySession(session_read_lock);
          curRunningQuerySubmittedTime = ra_exe_unit.query_state->getQuerySubmittedTime();
          sessionEnrolled =
              executor->checkIsQuerySessionEnrolled(curRunningSession, session_read_lock);
        }
        if (!curRunningSession.empty() && !curRunningQuerySubmittedTime.empty() &&
            sessionEnrolled) {
          executor->updateQuerySessionStatus(curRunningSession,
                                             curRunningQuerySubmittedTime,
                                             QuerySessionStatus::RUNNING_REDUCTION);
        }
      }
      try {
        return collectAllDeviceResults(shared_context,
                                       ra_exe_unit,
                                       *query_mem_desc_owned,
                                       query_comp_desc_owned->getDeviceType(),
                                       row_set_mem_owner);
      } catch (ReductionRanOutOfSlots&) {
        throw QueryExecutionError(ERR_OUT_OF_SLOTS);
      } catch (OverflowOrUnderflow&) {
        crt_min_byte_width <<= 1;
        continue;
      } catch (QueryExecutionError& e) {
        VLOG(1) << "Error received! error_code: " << e.getErrorCode()
                << ", what(): " << e.what();
        throw QueryExecutionError(e.getErrorCode());
      }
    }
    return resultsUnion(shared_context, ra_exe_unit);

  } while (static_cast<size_t>(crt_min_byte_width) <= sizeof(int64_t));

  return std::make_shared<ResultSet>(std::vector<TargetInfo>{},
                                     ExecutorDeviceType::CPU,
                                     QueryMemoryDescriptor(),
                                     nullptr,
                                     catalog_,
                                     blockSize(),
                                     gridSize());
}

void Executor::executeWorkUnitPerFragment(
    const RelAlgExecutionUnit& ra_exe_unit_in,
    const InputTableInfo& table_info,
    const CompilationOptions& co,
    const ExecutionOptions& eo,
    const Catalog_Namespace::Catalog& cat,
    PerFragmentCallBack& cb,
    const std::set<size_t>& fragment_indexes_param) {
  const auto [ra_exe_unit, deleted_cols_map] = addDeletedColumn(ra_exe_unit_in, co);
  ColumnCacheMap column_cache;

  std::vector<InputTableInfo> table_infos{table_info};
  SharedKernelContext kernel_context(table_infos);

  ColumnFetcher column_fetcher(this, column_cache);
  auto query_comp_desc_owned = std::make_unique<QueryCompilationDescriptor>();
  std::unique_ptr<QueryMemoryDescriptor> query_mem_desc_owned;
  {
    auto clock_begin = timer_start();
    std::lock_guard<std::mutex> compilation_lock(compilation_mutex_);
    compilation_queue_time_ms_ += timer_stop(clock_begin);
    query_mem_desc_owned =
        query_comp_desc_owned->compile(0,
                                       8,
                                       /*has_cardinality_estimation=*/false,
                                       ra_exe_unit,
                                       table_infos,
                                       deleted_cols_map,
                                       column_fetcher,
                                       co,
                                       eo,
                                       nullptr,
                                       this);
  }
  CHECK(query_mem_desc_owned);
  CHECK_EQ(size_t(1), ra_exe_unit.input_descs.size());
  const auto table_id = ra_exe_unit.input_descs[0].getTableId();
  const auto& outer_fragments = table_info.info.fragments;

  std::set<size_t> fragment_indexes;
  if (fragment_indexes_param.empty()) {
    // An empty `fragment_indexes_param` set implies executing
    // the query for all fragments in the table. In this
    // case, populate `fragment_indexes` with all fragment indexes.
    for (size_t i = 0; i < outer_fragments.size(); i++) {
      fragment_indexes.emplace(i);
    }
  } else {
    fragment_indexes = fragment_indexes_param;
  }

  {
    auto clock_begin = timer_start();
    std::lock_guard<std::mutex> kernel_lock(kernel_mutex_);
    kernel_queue_time_ms_ += timer_stop(clock_begin);

    for (auto fragment_index : fragment_indexes) {
      // We may want to consider in the future allowing this to execute on devices other
      // than CPU
      FragmentsList fragments_list{{table_id, {fragment_index}}};
      ExecutionKernel kernel(ra_exe_unit,
                             co.device_type,
                             /*device_id=*/0,
                             eo,
                             column_fetcher,
                             *query_comp_desc_owned,
                             *query_mem_desc_owned,
                             fragments_list,
                             ExecutorDispatchMode::KernelPerFragment,
                             /*render_info=*/nullptr,
                             /*rowid_lookup_key=*/-1);
      kernel.run(this, 0, kernel_context);
    }
  }

  const auto& all_fragment_results = kernel_context.getFragmentResults();

  for (const auto& [result_set_ptr, result_fragment_indexes] : all_fragment_results) {
    CHECK_EQ(result_fragment_indexes.size(), 1);
    cb(result_set_ptr, outer_fragments[result_fragment_indexes[0]]);
  }
}

ResultSetPtr Executor::executeTableFunction(
    const TableFunctionExecutionUnit exe_unit,
    const std::vector<InputTableInfo>& table_infos,
    const CompilationOptions& co,
    const ExecutionOptions& eo,
    const Catalog_Namespace::Catalog& cat) {
  INJECT_TIMER(Exec_executeTableFunction);

  if (eo.just_validate) {
    QueryMemoryDescriptor query_mem_desc(this,
                                         /*entry_count=*/0,
                                         QueryDescriptionType::Projection,
                                         /*is_table_function=*/true);
    query_mem_desc.setOutputColumnar(true);
    return std::make_shared<ResultSet>(
        target_exprs_to_infos(exe_unit.target_exprs, query_mem_desc),
        co.device_type,
        ResultSet::fixupQueryMemoryDescriptor(query_mem_desc),
        this->getRowSetMemoryOwner(),
        this->getCatalog(),
        this->blockSize(),
        this->gridSize());
  }

  nukeOldState(false, table_infos, PlanState::DeletedColumnsMap{}, nullptr);

  ColumnCacheMap column_cache;  // Note: if we add retries to the table function
                                // framework, we may want to move this up a level

  ColumnFetcher column_fetcher(this, column_cache);
  TableFunctionCompilationContext compilation_context;
  compilation_context.compile(exe_unit, co, this);

  TableFunctionExecutionContext exe_context(getRowSetMemoryOwner());
  return exe_context.execute(
      exe_unit, table_infos, &compilation_context, column_fetcher, co.device_type, this);
}

ResultSetPtr Executor::executeExplain(const QueryCompilationDescriptor& query_comp_desc) {
  return std::make_shared<ResultSet>(query_comp_desc.getIR());
}

namespace {

void add_transient_string_literals_for_expression(
    const Analyzer::Expr* expr,
    Executor* executor,
    const std::shared_ptr<RowSetMemoryOwner>& row_set_mem_owner) {
  if (!expr) {
    return;
  }

  const auto array_expr = dynamic_cast<const Analyzer::ArrayExpr*>(expr);
  if (array_expr) {
    for (size_t i = 0; i < array_expr->getElementCount(); i++) {
      add_transient_string_literals_for_expression(
          array_expr->getElement(i), executor, row_set_mem_owner);
    }
    return;
  }

  const auto cast_expr = dynamic_cast<const Analyzer::UOper*>(expr);
  const auto& expr_ti = expr->get_type_info();
  if (cast_expr && cast_expr->get_optype() == kCAST && expr_ti.is_string()) {
    CHECK_EQ(kENCODING_DICT, expr_ti.get_compression());
    auto sdp = executor->getStringDictionaryProxy(
        expr_ti.get_comp_param(), row_set_mem_owner, true);
    CHECK(sdp);
    const auto str_lit_expr =
        dynamic_cast<const Analyzer::Constant*>(cast_expr->get_operand());
    if (str_lit_expr && str_lit_expr->get_constval().stringval) {
      sdp->getOrAddTransient(*str_lit_expr->get_constval().stringval);
    }
    return;
  }
  const auto case_expr = dynamic_cast<const Analyzer::CaseExpr*>(expr);
  if (!case_expr) {
    return;
  }
  Analyzer::DomainSet domain_set;
  case_expr->get_domain(domain_set);
  if (domain_set.empty()) {
    return;
  }
  if (expr_ti.is_string()) {
    CHECK_EQ(kENCODING_DICT, expr_ti.get_compression());
    auto sdp = executor->getStringDictionaryProxy(
        expr_ti.get_comp_param(), row_set_mem_owner, true);
    CHECK(sdp);
    for (const auto domain_expr : domain_set) {
      const auto cast_expr = dynamic_cast<const Analyzer::UOper*>(domain_expr);
      const auto str_lit_expr =
          cast_expr && cast_expr->get_optype() == kCAST
              ? dynamic_cast<const Analyzer::Constant*>(cast_expr->get_operand())
              : dynamic_cast<const Analyzer::Constant*>(domain_expr);
      if (str_lit_expr && str_lit_expr->get_constval().stringval) {
        sdp->getOrAddTransient(*str_lit_expr->get_constval().stringval);
      }
    }
  }
}

}  // namespace

void Executor::addTransientStringLiterals(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::shared_ptr<RowSetMemoryOwner>& row_set_mem_owner) {
  for (const auto& group_expr : ra_exe_unit.groupby_exprs) {
    add_transient_string_literals_for_expression(
        group_expr.get(), this, row_set_mem_owner);
  }
  for (const auto target_expr : ra_exe_unit.target_exprs) {
    const auto& target_type = target_expr->get_type_info();
    if (target_type.is_string() && target_type.get_compression() != kENCODING_DICT) {
      continue;
    }
    const auto agg_expr = dynamic_cast<const Analyzer::AggExpr*>(target_expr);
    if (agg_expr) {
      if (agg_expr->get_aggtype() == kSINGLE_VALUE ||
          agg_expr->get_aggtype() == kSAMPLE) {
        add_transient_string_literals_for_expression(
            agg_expr->get_arg(), this, row_set_mem_owner);
      }
    } else {
      add_transient_string_literals_for_expression(target_expr, this, row_set_mem_owner);
    }
  }
}

ExecutorDeviceType Executor::getDeviceTypeForTargets(
    const RelAlgExecutionUnit& ra_exe_unit,
    const ExecutorDeviceType requested_device_type) {
  for (const auto target_expr : ra_exe_unit.target_exprs) {
    const auto agg_info = get_target_info(target_expr, g_bigint_count);
    if (!ra_exe_unit.groupby_exprs.empty() &&
        !isArchPascalOrLater(requested_device_type)) {
      if ((agg_info.agg_kind == kAVG || agg_info.agg_kind == kSUM) &&
          agg_info.agg_arg_type.get_type() == kDOUBLE) {
        return ExecutorDeviceType::CPU;
      }
    }
    if (dynamic_cast<const Analyzer::RegexpExpr*>(target_expr)) {
      return ExecutorDeviceType::CPU;
    }
  }
  return requested_device_type;
}

namespace {

int64_t inline_null_val(const SQLTypeInfo& ti, const bool float_argument_input) {
  CHECK(ti.is_number() || ti.is_time() || ti.is_boolean() || ti.is_string());
  if (ti.is_fp()) {
    if (float_argument_input && ti.get_type() == kFLOAT) {
      int64_t float_null_val = 0;
      *reinterpret_cast<float*>(may_alias_ptr(&float_null_val)) =
          static_cast<float>(inline_fp_null_val(ti));
      return float_null_val;
    }
    const auto double_null_val = inline_fp_null_val(ti);
    return *reinterpret_cast<const int64_t*>(may_alias_ptr(&double_null_val));
  }
  return inline_int_null_val(ti);
}

void fill_entries_for_empty_input(std::vector<TargetInfo>& target_infos,
                                  std::vector<int64_t>& entry,
                                  const std::vector<Analyzer::Expr*>& target_exprs,
                                  const QueryMemoryDescriptor& query_mem_desc) {
  for (size_t target_idx = 0; target_idx < target_exprs.size(); ++target_idx) {
    const auto target_expr = target_exprs[target_idx];
    const auto agg_info = get_target_info(target_expr, g_bigint_count);
    CHECK(agg_info.is_agg);
    target_infos.push_back(agg_info);
    if (g_cluster) {
      const auto executor = query_mem_desc.getExecutor();
      CHECK(executor);
      auto row_set_mem_owner = executor->getRowSetMemoryOwner();
      CHECK(row_set_mem_owner);
      const auto& count_distinct_desc =
          query_mem_desc.getCountDistinctDescriptor(target_idx);
      if (count_distinct_desc.impl_type_ == CountDistinctImplType::Bitmap) {
        CHECK(row_set_mem_owner);
        auto count_distinct_buffer = row_set_mem_owner->allocateCountDistinctBuffer(
            count_distinct_desc.bitmapPaddedSizeBytes(),
            /*thread_idx=*/0);  // TODO: can we detect thread idx here?
        entry.push_back(reinterpret_cast<int64_t>(count_distinct_buffer));
        continue;
      }
      if (count_distinct_desc.impl_type_ == CountDistinctImplType::StdSet) {
        auto count_distinct_set = new std::set<int64_t>();
        CHECK(row_set_mem_owner);
        row_set_mem_owner->addCountDistinctSet(count_distinct_set);
        entry.push_back(reinterpret_cast<int64_t>(count_distinct_set));
        continue;
      }
    }
    const bool float_argument_input = takes_float_argument(agg_info);
    if (agg_info.agg_kind == kCOUNT || agg_info.agg_kind == kAPPROX_COUNT_DISTINCT) {
      entry.push_back(0);
    } else if (agg_info.agg_kind == kAVG) {
      entry.push_back(inline_null_val(agg_info.sql_type, float_argument_input));
      entry.push_back(0);
    } else if (agg_info.agg_kind == kSINGLE_VALUE || agg_info.agg_kind == kSAMPLE) {
      if (agg_info.sql_type.is_geometry()) {
        for (int i = 0; i < agg_info.sql_type.get_physical_coord_cols() * 2; i++) {
          entry.push_back(0);
        }
      } else if (agg_info.sql_type.is_varlen()) {
        entry.push_back(0);
        entry.push_back(0);
      } else {
        entry.push_back(inline_null_val(agg_info.sql_type, float_argument_input));
      }
    } else {
      entry.push_back(inline_null_val(agg_info.sql_type, float_argument_input));
    }
  }
}

ResultSetPtr build_row_for_empty_input(
    const std::vector<Analyzer::Expr*>& target_exprs_in,
    const QueryMemoryDescriptor& query_mem_desc,
    const ExecutorDeviceType device_type) {
  std::vector<std::shared_ptr<Analyzer::Expr>> target_exprs_owned_copies;
  std::vector<Analyzer::Expr*> target_exprs;
  for (const auto target_expr : target_exprs_in) {
    const auto target_expr_copy =
        std::dynamic_pointer_cast<Analyzer::AggExpr>(target_expr->deep_copy());
    CHECK(target_expr_copy);
    auto ti = target_expr->get_type_info();
    ti.set_notnull(false);
    target_expr_copy->set_type_info(ti);
    if (target_expr_copy->get_arg()) {
      auto arg_ti = target_expr_copy->get_arg()->get_type_info();
      arg_ti.set_notnull(false);
      target_expr_copy->get_arg()->set_type_info(arg_ti);
    }
    target_exprs_owned_copies.push_back(target_expr_copy);
    target_exprs.push_back(target_expr_copy.get());
  }
  std::vector<TargetInfo> target_infos;
  std::vector<int64_t> entry;
  fill_entries_for_empty_input(target_infos, entry, target_exprs, query_mem_desc);
  const auto executor = query_mem_desc.getExecutor();
  CHECK(executor);
  auto row_set_mem_owner = executor->getRowSetMemoryOwner();
  CHECK(row_set_mem_owner);
  auto rs = std::make_shared<ResultSet>(target_infos,
                                        device_type,
                                        query_mem_desc,
                                        row_set_mem_owner,
                                        executor->getCatalog(),
                                        executor->blockSize(),
                                        executor->gridSize());
  rs->allocateStorage();
  rs->fillOneEntry(entry);
  return rs;
}

}  // namespace

ResultSetPtr Executor::collectAllDeviceResults(
    SharedKernelContext& shared_context,
    const RelAlgExecutionUnit& ra_exe_unit,
    const QueryMemoryDescriptor& query_mem_desc,
    const ExecutorDeviceType device_type,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner) {
  auto timer = DEBUG_TIMER(__func__);
  auto& result_per_device = shared_context.getFragmentResults();
  if (result_per_device.empty() && query_mem_desc.getQueryDescriptionType() ==
                                       QueryDescriptionType::NonGroupedAggregate) {
    return build_row_for_empty_input(
        ra_exe_unit.target_exprs, query_mem_desc, device_type);
  }
  if (use_speculative_top_n(ra_exe_unit, query_mem_desc)) {
    try {
      return reduceSpeculativeTopN(
          ra_exe_unit, result_per_device, row_set_mem_owner, query_mem_desc);
    } catch (const std::bad_alloc&) {
      throw SpeculativeTopNFailed("Failed during multi-device reduction.");
    }
  }
  const auto shard_count =
      device_type == ExecutorDeviceType::GPU
          ? GroupByAndAggregate::shard_count_for_top_groups(ra_exe_unit, *catalog_)
          : 0;

  if (shard_count && !result_per_device.empty()) {
    return collectAllDeviceShardedTopResults(shared_context, ra_exe_unit);
  }
  return reduceMultiDeviceResults(
      ra_exe_unit, result_per_device, row_set_mem_owner, query_mem_desc);
}

namespace {
/**
 * This functions uses the permutation indices in "top_permutation", and permutes
 * all group columns (if any) and aggregate columns into the output storage. In columnar
 * layout, since different columns are not consecutive in the memory, different columns
 * are copied back into the output storage separetely and through different memcpy
 * operations.
 *
 * output_row_index contains the current index of the output storage (input storage will
 * be appended to it), and the final output row index is returned.
 */
size_t permute_storage_columnar(const ResultSetStorage* input_storage,
                                const QueryMemoryDescriptor& input_query_mem_desc,
                                const ResultSetStorage* output_storage,
                                size_t output_row_index,
                                const QueryMemoryDescriptor& output_query_mem_desc,
                                const std::vector<uint32_t>& top_permutation) {
  const auto output_buffer = output_storage->getUnderlyingBuffer();
  const auto input_buffer = input_storage->getUnderlyingBuffer();
  for (const auto sorted_idx : top_permutation) {
    // permuting all group-columns in this result set into the final buffer:
    for (size_t group_idx = 0; group_idx < input_query_mem_desc.getKeyCount();
         group_idx++) {
      const auto input_column_ptr =
          input_buffer + input_query_mem_desc.getPrependedGroupColOffInBytes(group_idx) +
          sorted_idx * input_query_mem_desc.groupColWidth(group_idx);
      const auto output_column_ptr =
          output_buffer +
          output_query_mem_desc.getPrependedGroupColOffInBytes(group_idx) +
          output_row_index * output_query_mem_desc.groupColWidth(group_idx);
      memcpy(output_column_ptr,
             input_column_ptr,
             output_query_mem_desc.groupColWidth(group_idx));
    }
    // permuting all agg-columns in this result set into the final buffer:
    for (size_t slot_idx = 0; slot_idx < input_query_mem_desc.getSlotCount();
         slot_idx++) {
      const auto input_column_ptr =
          input_buffer + input_query_mem_desc.getColOffInBytes(slot_idx) +
          sorted_idx * input_query_mem_desc.getPaddedSlotWidthBytes(slot_idx);
      const auto output_column_ptr =
          output_buffer + output_query_mem_desc.getColOffInBytes(slot_idx) +
          output_row_index * output_query_mem_desc.getPaddedSlotWidthBytes(slot_idx);
      memcpy(output_column_ptr,
             input_column_ptr,
             output_query_mem_desc.getPaddedSlotWidthBytes(slot_idx));
    }
    ++output_row_index;
  }
  return output_row_index;
}

/**
 * This functions uses the permutation indices in "top_permutation", and permutes
 * all group columns (if any) and aggregate columns into the output storage. In row-wise,
 * since different columns are consecutive within the memory, it suffices to perform a
 * single memcpy operation and copy the whole row.
 *
 * output_row_index contains the current index of the output storage (input storage will
 * be appended to it), and the final output row index is returned.
 */
size_t permute_storage_row_wise(const ResultSetStorage* input_storage,
                                const ResultSetStorage* output_storage,
                                size_t output_row_index,
                                const QueryMemoryDescriptor& output_query_mem_desc,
                                const std::vector<uint32_t>& top_permutation) {
  const auto output_buffer = output_storage->getUnderlyingBuffer();
  const auto input_buffer = input_storage->getUnderlyingBuffer();
  for (const auto sorted_idx : top_permutation) {
    const auto row_ptr = input_buffer + sorted_idx * output_query_mem_desc.getRowSize();
    memcpy(output_buffer + output_row_index * output_query_mem_desc.getRowSize(),
           row_ptr,
           output_query_mem_desc.getRowSize());
    ++output_row_index;
  }
  return output_row_index;
}
}  // namespace

// Collect top results from each device, stitch them together and sort. Partial
// results from each device are guaranteed to be disjunct because we only go on
// this path when one of the columns involved is a shard key.
ResultSetPtr Executor::collectAllDeviceShardedTopResults(
    SharedKernelContext& shared_context,
    const RelAlgExecutionUnit& ra_exe_unit) const {
  auto& result_per_device = shared_context.getFragmentResults();
  const auto first_result_set = result_per_device.front().first;
  CHECK(first_result_set);
  auto top_query_mem_desc = first_result_set->getQueryMemDesc();
  CHECK(!top_query_mem_desc.hasInterleavedBinsOnGpu());
  const auto top_n = ra_exe_unit.sort_info.limit + ra_exe_unit.sort_info.offset;
  top_query_mem_desc.setEntryCount(0);
  for (auto& result : result_per_device) {
    const auto result_set = result.first;
    CHECK(result_set);
    result_set->sort(ra_exe_unit.sort_info.order_entries, top_n, this);
    size_t new_entry_cnt = top_query_mem_desc.getEntryCount() + result_set->rowCount();
    top_query_mem_desc.setEntryCount(new_entry_cnt);
  }
  auto top_result_set = std::make_shared<ResultSet>(first_result_set->getTargetInfos(),
                                                    first_result_set->getDeviceType(),
                                                    top_query_mem_desc,
                                                    first_result_set->getRowSetMemOwner(),
                                                    catalog_,
                                                    blockSize(),
                                                    gridSize());
  auto top_storage = top_result_set->allocateStorage();
  size_t top_output_row_idx{0};
  for (auto& result : result_per_device) {
    const auto result_set = result.first;
    CHECK(result_set);
    const auto& top_permutation = result_set->getPermutationBuffer();
    CHECK_LE(top_permutation.size(), top_n);
    if (top_query_mem_desc.didOutputColumnar()) {
      top_output_row_idx = permute_storage_columnar(result_set->getStorage(),
                                                    result_set->getQueryMemDesc(),
                                                    top_storage,
                                                    top_output_row_idx,
                                                    top_query_mem_desc,
                                                    top_permutation);
    } else {
      top_output_row_idx = permute_storage_row_wise(result_set->getStorage(),
                                                    top_storage,
                                                    top_output_row_idx,
                                                    top_query_mem_desc,
                                                    top_permutation);
    }
  }
  CHECK_EQ(top_output_row_idx, top_query_mem_desc.getEntryCount());
  return top_result_set;
}

std::unordered_map<int, const Analyzer::BinOper*> Executor::getInnerTabIdToJoinCond()
    const {
  std::unordered_map<int, const Analyzer::BinOper*> id_to_cond;
  const auto& join_info = plan_state_->join_info_;
  CHECK_EQ(join_info.equi_join_tautologies_.size(), join_info.join_hash_tables_.size());
  for (size_t i = 0; i < join_info.join_hash_tables_.size(); ++i) {
    int inner_table_id = join_info.join_hash_tables_[i]->getInnerTableId();
    id_to_cond.insert(
        std::make_pair(inner_table_id, join_info.equi_join_tautologies_[i].get()));
  }
  return id_to_cond;
}

namespace {

bool has_lazy_fetched_columns(const std::vector<ColumnLazyFetchInfo>& fetched_cols) {
  for (const auto& col : fetched_cols) {
    if (col.is_lazily_fetched) {
      return true;
    }
  }
  return false;
}

}  // namespace

std::vector<std::unique_ptr<ExecutionKernel>> Executor::createKernels(
    SharedKernelContext& shared_context,
    const RelAlgExecutionUnit& ra_exe_unit,
    ColumnFetcher& column_fetcher,
    const std::vector<InputTableInfo>& table_infos,
    const ExecutionOptions& eo,
    const bool is_agg,
    const bool allow_single_frag_table_opt,
    const size_t context_count,
    const QueryCompilationDescriptor& query_comp_desc,
    const QueryMemoryDescriptor& query_mem_desc,
    RenderInfo* render_info,
    std::unordered_set<int>& available_gpus,
    int& available_cpus) {
  std::vector<std::unique_ptr<ExecutionKernel>> execution_kernels;

  QueryFragmentDescriptor fragment_descriptor(
      ra_exe_unit,
      table_infos,
      query_comp_desc.getDeviceType() == ExecutorDeviceType::GPU
          ? catalog_->getDataMgr().getMemoryInfo(Data_Namespace::MemoryLevel::GPU_LEVEL)
          : std::vector<Data_Namespace::MemoryInfo>{},
      eo.gpu_input_mem_limit_percent,
      eo.outer_fragment_indices);
  CHECK(!ra_exe_unit.input_descs.empty());

  const auto device_type = query_comp_desc.getDeviceType();
  const bool uses_lazy_fetch =
      plan_state_->allow_lazy_fetch_ &&
      has_lazy_fetched_columns(getColLazyFetchInfo(ra_exe_unit.target_exprs));
  const bool use_multifrag_kernel = (device_type == ExecutorDeviceType::GPU) &&
                                    eo.allow_multifrag && (!uses_lazy_fetch || is_agg);
  const auto device_count = deviceCount(device_type);
  CHECK_GT(device_count, 0);

  fragment_descriptor.buildFragmentKernelMap(ra_exe_unit,
                                             shared_context.getFragOffsets(),
                                             device_count,
                                             device_type,
                                             use_multifrag_kernel,
                                             g_inner_join_fragment_skipping,
                                             this);
  if (eo.with_watchdog && fragment_descriptor.shouldCheckWorkUnitWatchdog()) {
    checkWorkUnitWatchdog(ra_exe_unit, table_infos, *catalog_, device_type, device_count);
  }

  if (use_multifrag_kernel) {
    VLOG(1) << "Creating multifrag execution kernels";
    VLOG(1) << query_mem_desc.toString();

    // NB: We should never be on this path when the query is retried because of running
    // out of group by slots; also, for scan only queries on CPU we want the
    // high-granularity, fragment by fragment execution instead. For scan only queries on
    // GPU, we want the multifrag kernel path to save the overhead of allocating an output
    // buffer per fragment.
    auto multifrag_kernel_dispatch = [&ra_exe_unit,
                                      &execution_kernels,
                                      &column_fetcher,
                                      &eo,
                                      &query_comp_desc,
                                      &query_mem_desc,
                                      render_info](const int device_id,
                                                   const FragmentsList& frag_list,
                                                   const int64_t rowid_lookup_key) {
      execution_kernels.emplace_back(
          std::make_unique<ExecutionKernel>(ra_exe_unit,
                                            ExecutorDeviceType::GPU,
                                            device_id,
                                            eo,
                                            column_fetcher,
                                            query_comp_desc,
                                            query_mem_desc,
                                            frag_list,
                                            ExecutorDispatchMode::MultifragmentKernel,
                                            render_info,
                                            rowid_lookup_key));
    };
    fragment_descriptor.assignFragsToMultiDispatch(multifrag_kernel_dispatch);
  } else {
    VLOG(1) << "Creating one execution kernel per fragment";
    VLOG(1) << query_mem_desc.toString();

    if (!ra_exe_unit.use_bump_allocator && allow_single_frag_table_opt &&
        (query_mem_desc.getQueryDescriptionType() == QueryDescriptionType::Projection) &&
        table_infos.size() == 1 && table_infos.front().table_id > 0) {
      const auto max_frag_size =
          table_infos.front().info.getFragmentNumTuplesUpperBound();
      if (max_frag_size < query_mem_desc.getEntryCount()) {
        LOG(INFO) << "Lowering scan limit from " << query_mem_desc.getEntryCount()
                  << " to match max fragment size " << max_frag_size
                  << " for kernel per fragment execution path.";
        throw CompilationRetryNewScanLimit(max_frag_size);
      }
    }

    size_t frag_list_idx{0};
    auto fragment_per_kernel_dispatch = [&ra_exe_unit,
                                         &execution_kernels,
                                         &column_fetcher,
                                         &eo,
                                         &frag_list_idx,
                                         &device_type,
                                         &query_comp_desc,
                                         &query_mem_desc,
                                         render_info](const int device_id,
                                                      const FragmentsList& frag_list,
                                                      const int64_t rowid_lookup_key) {
      if (!frag_list.size()) {
        return;
      }
      CHECK_GE(device_id, 0);

      execution_kernels.emplace_back(
          std::make_unique<ExecutionKernel>(ra_exe_unit,
                                            device_type,
                                            device_id,
                                            eo,
                                            column_fetcher,
                                            query_comp_desc,
                                            query_mem_desc,
                                            frag_list,
                                            ExecutorDispatchMode::KernelPerFragment,
                                            render_info,
                                            rowid_lookup_key));
      ++frag_list_idx;
    };

    fragment_descriptor.assignFragsToKernelDispatch(fragment_per_kernel_dispatch,
                                                    ra_exe_unit);
  }

  return execution_kernels;
}

template <typename THREAD_POOL>
void Executor::launchKernels(SharedKernelContext& shared_context,
                             std::vector<std::unique_ptr<ExecutionKernel>>&& kernels) {
  auto clock_begin = timer_start();
  std::lock_guard<std::mutex> kernel_lock(kernel_mutex_);
  kernel_queue_time_ms_ += timer_stop(clock_begin);

  THREAD_POOL thread_pool;
  VLOG(1) << "Launching " << kernels.size() << " kernels for query.";
  size_t kernel_idx = 1;
  for (auto& kernel : kernels) {
    thread_pool.spawn(
        [this, &shared_context, parent_thread_id = logger::thread_id()](
            ExecutionKernel* kernel, const size_t crt_kernel_idx) {
          CHECK(kernel);
          DEBUG_TIMER_NEW_THREAD(parent_thread_id);
          const size_t thread_idx = crt_kernel_idx % cpu_threads();
          kernel->run(this, thread_idx, shared_context);
        },
        kernel.get(),
        kernel_idx++);
  }
  thread_pool.join();
}

std::vector<size_t> Executor::getTableFragmentIndices(
    const RelAlgExecutionUnit& ra_exe_unit,
    const ExecutorDeviceType device_type,
    const size_t table_idx,
    const size_t outer_frag_idx,
    std::map<int, const TableFragments*>& selected_tables_fragments,
    const std::unordered_map<int, const Analyzer::BinOper*>&
        inner_table_id_to_join_condition) {
  const int table_id = ra_exe_unit.input_descs[table_idx].getTableId();
  auto table_frags_it = selected_tables_fragments.find(table_id);
  CHECK(table_frags_it != selected_tables_fragments.end());
  const auto& outer_input_desc = ra_exe_unit.input_descs[0];
  const auto outer_table_fragments_it =
      selected_tables_fragments.find(outer_input_desc.getTableId());
  const auto outer_table_fragments = outer_table_fragments_it->second;
  CHECK(outer_table_fragments_it != selected_tables_fragments.end());
  CHECK_LT(outer_frag_idx, outer_table_fragments->size());
  if (!table_idx) {
    return {outer_frag_idx};
  }
  const auto& outer_fragment_info = (*outer_table_fragments)[outer_frag_idx];
  auto& inner_frags = table_frags_it->second;
  CHECK_LT(size_t(1), ra_exe_unit.input_descs.size());
  std::vector<size_t> all_frag_ids;
  for (size_t inner_frag_idx = 0; inner_frag_idx < inner_frags->size();
       ++inner_frag_idx) {
    const auto& inner_frag_info = (*inner_frags)[inner_frag_idx];
    if (skipFragmentPair(outer_fragment_info,
                         inner_frag_info,
                         table_idx,
                         inner_table_id_to_join_condition,
                         ra_exe_unit,
                         device_type)) {
      continue;
    }
    all_frag_ids.push_back(inner_frag_idx);
  }
  return all_frag_ids;
}

// Returns true iff the join between two fragments cannot yield any results, per
// shard information. The pair can be skipped to avoid full broadcast.
bool Executor::skipFragmentPair(
    const Fragmenter_Namespace::FragmentInfo& outer_fragment_info,
    const Fragmenter_Namespace::FragmentInfo& inner_fragment_info,
    const int table_idx,
    const std::unordered_map<int, const Analyzer::BinOper*>&
        inner_table_id_to_join_condition,
    const RelAlgExecutionUnit& ra_exe_unit,
    const ExecutorDeviceType device_type) {
  if (device_type != ExecutorDeviceType::GPU) {
    return false;
  }
  CHECK(table_idx >= 0 &&
        static_cast<size_t>(table_idx) < ra_exe_unit.input_descs.size());
  const int inner_table_id = ra_exe_unit.input_descs[table_idx].getTableId();
  // Both tables need to be sharded the same way.
  if (outer_fragment_info.shard == -1 || inner_fragment_info.shard == -1 ||
      outer_fragment_info.shard == inner_fragment_info.shard) {
    return false;
  }
  const Analyzer::BinOper* join_condition{nullptr};
  if (ra_exe_unit.join_quals.empty()) {
    CHECK(!inner_table_id_to_join_condition.empty());
    auto condition_it = inner_table_id_to_join_condition.find(inner_table_id);
    CHECK(condition_it != inner_table_id_to_join_condition.end());
    join_condition = condition_it->second;
    CHECK(join_condition);
  } else {
    CHECK_EQ(plan_state_->join_info_.equi_join_tautologies_.size(),
             plan_state_->join_info_.join_hash_tables_.size());
    for (size_t i = 0; i < plan_state_->join_info_.join_hash_tables_.size(); ++i) {
      if (plan_state_->join_info_.join_hash_tables_[i]->getInnerTableRteIdx() ==
          table_idx) {
        CHECK(!join_condition);
        join_condition = plan_state_->join_info_.equi_join_tautologies_[i].get();
      }
    }
  }
  if (!join_condition) {
    return false;
  }
  // TODO(adb): support fragment skipping based on the overlaps operator
  if (join_condition->is_overlaps_oper()) {
    return false;
  }
  size_t shard_count{0};
  if (dynamic_cast<const Analyzer::ExpressionTuple*>(
          join_condition->get_left_operand())) {
    auto inner_outer_pairs =
        normalize_column_pairs(join_condition, *getCatalog(), getTemporaryTables());
    shard_count = BaselineJoinHashTable::getShardCountForCondition(
        join_condition, this, inner_outer_pairs);
  } else {
    shard_count = get_shard_count(join_condition, this);
  }
  if (shard_count && !ra_exe_unit.join_quals.empty()) {
    plan_state_->join_info_.sharded_range_table_indices_.emplace(table_idx);
  }
  return shard_count;
}

namespace {

const ColumnDescriptor* try_get_column_descriptor(const InputColDescriptor* col_desc,
                                                  const Catalog_Namespace::Catalog& cat) {
  const int table_id = col_desc->getScanDesc().getTableId();
  const int col_id = col_desc->getColId();
  return get_column_descriptor_maybe(col_id, table_id, cat);
}

}  // namespace

std::map<size_t, std::vector<uint64_t>> get_table_id_to_frag_offsets(
    const std::vector<InputDescriptor>& input_descs,
    const std::map<int, const TableFragments*>& all_tables_fragments) {
  std::map<size_t, std::vector<uint64_t>> tab_id_to_frag_offsets;
  for (auto& desc : input_descs) {
    const auto fragments_it = all_tables_fragments.find(desc.getTableId());
    CHECK(fragments_it != all_tables_fragments.end());
    const auto& fragments = *fragments_it->second;
    std::vector<uint64_t> frag_offsets(fragments.size(), 0);
    for (size_t i = 0, off = 0; i < fragments.size(); ++i) {
      frag_offsets[i] = off;
      off += fragments[i].getNumTuples();
    }
    tab_id_to_frag_offsets.insert(std::make_pair(desc.getTableId(), frag_offsets));
  }
  return tab_id_to_frag_offsets;
}

std::pair<std::vector<std::vector<int64_t>>, std::vector<std::vector<uint64_t>>>
Executor::getRowCountAndOffsetForAllFrags(
    const RelAlgExecutionUnit& ra_exe_unit,
    const CartesianProduct<std::vector<std::vector<size_t>>>& frag_ids_crossjoin,
    const std::vector<InputDescriptor>& input_descs,
    const std::map<int, const TableFragments*>& all_tables_fragments) {
  std::vector<std::vector<int64_t>> all_num_rows;
  std::vector<std::vector<uint64_t>> all_frag_offsets;
  const auto tab_id_to_frag_offsets =
      get_table_id_to_frag_offsets(input_descs, all_tables_fragments);
  std::unordered_map<size_t, size_t> outer_id_to_num_row_idx;
  for (const auto& selected_frag_ids : frag_ids_crossjoin) {
    std::vector<int64_t> num_rows;
    std::vector<uint64_t> frag_offsets;
    if (!ra_exe_unit.union_all) {
      CHECK_EQ(selected_frag_ids.size(), input_descs.size());
    }
    for (size_t tab_idx = 0; tab_idx < input_descs.size(); ++tab_idx) {
      const auto frag_id = ra_exe_unit.union_all ? 0 : selected_frag_ids[tab_idx];
      const auto fragments_it =
          all_tables_fragments.find(input_descs[tab_idx].getTableId());
      CHECK(fragments_it != all_tables_fragments.end());
      const auto& fragments = *fragments_it->second;
      if (ra_exe_unit.join_quals.empty() || tab_idx == 0 ||
          plan_state_->join_info_.sharded_range_table_indices_.count(tab_idx)) {
        const auto& fragment = fragments[frag_id];
        num_rows.push_back(fragment.getNumTuples());
      } else {
        size_t total_row_count{0};
        for (const auto& fragment : fragments) {
          total_row_count += fragment.getNumTuples();
        }
        num_rows.push_back(total_row_count);
      }
      const auto frag_offsets_it =
          tab_id_to_frag_offsets.find(input_descs[tab_idx].getTableId());
      CHECK(frag_offsets_it != tab_id_to_frag_offsets.end());
      const auto& offsets = frag_offsets_it->second;
      CHECK_LT(frag_id, offsets.size());
      frag_offsets.push_back(offsets[frag_id]);
    }
    all_num_rows.push_back(num_rows);
    // Fragment offsets of outer table should be ONLY used by rowid for now.
    all_frag_offsets.push_back(frag_offsets);
  }
  return {all_num_rows, all_frag_offsets};
}

// Only fetch columns of hash-joined inner fact table whose fetch are not deferred from
// all the table fragments.
bool Executor::needFetchAllFragments(const InputColDescriptor& inner_col_desc,
                                     const RelAlgExecutionUnit& ra_exe_unit,
                                     const FragmentsList& selected_fragments) const {
  const auto& input_descs = ra_exe_unit.input_descs;
  const int nest_level = inner_col_desc.getScanDesc().getNestLevel();
  if (nest_level < 1 ||
      inner_col_desc.getScanDesc().getSourceType() != InputSourceType::TABLE ||
      ra_exe_unit.join_quals.empty() || input_descs.size() < 2 ||
      (ra_exe_unit.join_quals.empty() &&
       plan_state_->isLazyFetchColumn(inner_col_desc))) {
    return false;
  }
  const int table_id = inner_col_desc.getScanDesc().getTableId();
  CHECK_LT(static_cast<size_t>(nest_level), selected_fragments.size());
  CHECK_EQ(table_id, selected_fragments[nest_level].table_id);
  const auto& fragments = selected_fragments[nest_level].fragment_ids;
  return fragments.size() > 1;
}

bool Executor::needLinearizeAllFragments(
    const ColumnDescriptor* cd,
    const InputColDescriptor& inner_col_desc,
    const RelAlgExecutionUnit& ra_exe_unit,
    const FragmentsList& selected_fragments,
    const Data_Namespace::MemoryLevel memory_level) const {
  const int nest_level = inner_col_desc.getScanDesc().getNestLevel();
  const int table_id = inner_col_desc.getScanDesc().getTableId();
  CHECK_LT(static_cast<size_t>(nest_level), selected_fragments.size());
  CHECK_EQ(table_id, selected_fragments[nest_level].table_id);
  const auto& fragments = selected_fragments[nest_level].fragment_ids;
  auto need_linearize =
      cd->columnType.is_array() ||
      (cd->columnType.is_string() && !cd->columnType.is_dict_encoded_type());
  return table_id > 0 && need_linearize && fragments.size() > 1;
}

std::ostream& operator<<(std::ostream& os, FetchResult const& fetch_result) {
  return os << "col_buffers" << shared::printContainer(fetch_result.col_buffers)
            << " num_rows" << shared::printContainer(fetch_result.num_rows)
            << " frag_offsets" << shared::printContainer(fetch_result.frag_offsets);
}

FetchResult Executor::fetchChunks(
    const ColumnFetcher& column_fetcher,
    const RelAlgExecutionUnit& ra_exe_unit,
    const int device_id,
    const Data_Namespace::MemoryLevel memory_level,
    const std::map<int, const TableFragments*>& all_tables_fragments,
    const FragmentsList& selected_fragments,
    const Catalog_Namespace::Catalog& cat,
    std::list<ChunkIter>& chunk_iterators,
    std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunks,
    DeviceAllocator* device_allocator,
    const size_t thread_idx,
    const bool allow_runtime_interrupt) {
  auto timer = DEBUG_TIMER(__func__);
  INJECT_TIMER(fetchChunks);
  const auto& col_global_ids = ra_exe_unit.input_col_descs;
  std::vector<std::vector<size_t>> selected_fragments_crossjoin;
  std::vector<size_t> local_col_to_frag_pos;
  buildSelectedFragsMapping(selected_fragments_crossjoin,
                            local_col_to_frag_pos,
                            col_global_ids,
                            selected_fragments,
                            ra_exe_unit);

  CartesianProduct<std::vector<std::vector<size_t>>> frag_ids_crossjoin(
      selected_fragments_crossjoin);
  std::vector<std::vector<const int8_t*>> all_frag_col_buffers;
  std::vector<std::vector<int64_t>> all_num_rows;
  std::vector<std::vector<uint64_t>> all_frag_offsets;
  for (const auto& selected_frag_ids : frag_ids_crossjoin) {
    std::vector<const int8_t*> frag_col_buffers(
        plan_state_->global_to_local_col_ids_.size());
    for (const auto& col_id : col_global_ids) {
      if (allow_runtime_interrupt) {
        bool isInterrupted = false;
        {
          mapd_shared_lock<mapd_shared_mutex> session_read_lock(executor_session_mutex_);
          const auto query_session = getCurrentQuerySession(session_read_lock);
          isInterrupted =
              checkIsQuerySessionInterrupted(query_session, session_read_lock);
        }
        if (isInterrupted) {
          throw QueryExecutionError(ERR_INTERRUPTED);
        }
      }
      if (g_enable_dynamic_watchdog && interrupted_.load()) {
        throw QueryExecutionError(ERR_INTERRUPTED);
      }
      CHECK(col_id);
      const int table_id = col_id->getScanDesc().getTableId();
      const auto cd = try_get_column_descriptor(col_id.get(), cat);
      if (cd && cd->isVirtualCol) {
        CHECK_EQ("rowid", cd->columnName);
        continue;
      }
      const auto fragments_it = all_tables_fragments.find(table_id);
      CHECK(fragments_it != all_tables_fragments.end());
      const auto fragments = fragments_it->second;
      auto it = plan_state_->global_to_local_col_ids_.find(*col_id);
      CHECK(it != plan_state_->global_to_local_col_ids_.end());
      CHECK_LT(static_cast<size_t>(it->second),
               plan_state_->global_to_local_col_ids_.size());
      const size_t frag_id = selected_frag_ids[local_col_to_frag_pos[it->second]];
      if (!fragments->size()) {
        return {};
      }
      CHECK_LT(frag_id, fragments->size());
      auto memory_level_for_column = memory_level;
      if (plan_state_->columns_to_fetch_.find(
              std::make_pair(col_id->getScanDesc().getTableId(), col_id->getColId())) ==
          plan_state_->columns_to_fetch_.end()) {
        memory_level_for_column = Data_Namespace::CPU_LEVEL;
      }
      if (col_id->getScanDesc().getSourceType() == InputSourceType::RESULT) {
        frag_col_buffers[it->second] =
            column_fetcher.getResultSetColumn(col_id.get(),
                                              memory_level_for_column,
                                              device_id,
                                              device_allocator,
                                              thread_idx);
      } else {
        if (needFetchAllFragments(*col_id, ra_exe_unit, selected_fragments)) {
          // determine if we need special treatment to linearlize multi-frag table
          // i.e., a column that is classified as varlen type, i.e., array
          // for now, we only support fixed-length array that contains
          // geo point coordianates but we can support more types in this way
          if (needLinearizeAllFragments(cd,
                                        *col_id,
                                        ra_exe_unit,
                                        selected_fragments,
                                        memory_level_for_column)) {
            frag_col_buffers[it->second] =
                column_fetcher.linearizeColumnFragments(table_id,
                                                        col_id->getColId(),
                                                        all_tables_fragments,
                                                        chunks,
                                                        chunk_iterators,
                                                        memory_level_for_column,
                                                        device_id,
                                                        device_allocator,
                                                        thread_idx);
          } else {
            frag_col_buffers[it->second] =
                column_fetcher.getAllTableColumnFragments(table_id,
                                                          col_id->getColId(),
                                                          all_tables_fragments,
                                                          memory_level_for_column,
                                                          device_id,
                                                          device_allocator,
                                                          thread_idx);
          }
        } else {
          frag_col_buffers[it->second] =
              column_fetcher.getOneTableColumnFragment(table_id,
                                                       frag_id,
                                                       col_id->getColId(),
                                                       all_tables_fragments,
                                                       chunks,
                                                       chunk_iterators,
                                                       memory_level_for_column,
                                                       device_id,
                                                       device_allocator);
        }
      }
    }
    all_frag_col_buffers.push_back(frag_col_buffers);
  }
  std::tie(all_num_rows, all_frag_offsets) = getRowCountAndOffsetForAllFrags(
      ra_exe_unit, frag_ids_crossjoin, ra_exe_unit.input_descs, all_tables_fragments);
  return {all_frag_col_buffers, all_num_rows, all_frag_offsets};
}

// fetchChunks() is written under the assumption that multiple inputs implies a JOIN.
// This is written under the assumption that multiple inputs implies a UNION ALL.
FetchResult Executor::fetchUnionChunks(
    const ColumnFetcher& column_fetcher,
    const RelAlgExecutionUnit& ra_exe_unit,
    const int device_id,
    const Data_Namespace::MemoryLevel memory_level,
    const std::map<int, const TableFragments*>& all_tables_fragments,
    const FragmentsList& selected_fragments,
    const Catalog_Namespace::Catalog& cat,
    std::list<ChunkIter>& chunk_iterators,
    std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunks,
    DeviceAllocator* device_allocator,
    const size_t thread_idx,
    const bool allow_runtime_interrupt) {
  auto timer = DEBUG_TIMER(__func__);
  INJECT_TIMER(fetchUnionChunks);

  std::vector<std::vector<const int8_t*>> all_frag_col_buffers;
  std::vector<std::vector<int64_t>> all_num_rows;
  std::vector<std::vector<uint64_t>> all_frag_offsets;

  CHECK(!selected_fragments.empty());
  CHECK_LE(2u, ra_exe_unit.input_descs.size());
  CHECK_LE(2u, ra_exe_unit.input_col_descs.size());
  using TableId = int;
  TableId const selected_table_id = selected_fragments.front().table_id;
  bool const input_descs_index =
      selected_table_id == ra_exe_unit.input_descs[1].getTableId();
  if (!input_descs_index) {
    CHECK_EQ(selected_table_id, ra_exe_unit.input_descs[0].getTableId());
  }
  bool const input_col_descs_index =
      selected_table_id ==
      (*std::next(ra_exe_unit.input_col_descs.begin()))->getScanDesc().getTableId();
  if (!input_col_descs_index) {
    CHECK_EQ(selected_table_id,
             ra_exe_unit.input_col_descs.front()->getScanDesc().getTableId());
  }
  VLOG(2) << "selected_fragments.size()=" << selected_fragments.size()
          << " selected_table_id=" << selected_table_id
          << " input_descs_index=" << int(input_descs_index)
          << " input_col_descs_index=" << int(input_col_descs_index)
          << " ra_exe_unit.input_descs="
          << shared::printContainer(ra_exe_unit.input_descs)
          << " ra_exe_unit.input_col_descs="
          << shared::printContainer(ra_exe_unit.input_col_descs);

  // Partition col_global_ids by table_id
  std::unordered_map<TableId, std::list<std::shared_ptr<const InputColDescriptor>>>
      table_id_to_input_col_descs;
  for (auto const& input_col_desc : ra_exe_unit.input_col_descs) {
    TableId const table_id = input_col_desc->getScanDesc().getTableId();
    table_id_to_input_col_descs[table_id].push_back(input_col_desc);
  }
  for (auto const& pair : table_id_to_input_col_descs) {
    std::vector<std::vector<size_t>> selected_fragments_crossjoin;
    std::vector<size_t> local_col_to_frag_pos;

    buildSelectedFragsMappingForUnion(selected_fragments_crossjoin,
                                      local_col_to_frag_pos,
                                      pair.second,
                                      selected_fragments,
                                      ra_exe_unit);

    CartesianProduct<std::vector<std::vector<size_t>>> frag_ids_crossjoin(
        selected_fragments_crossjoin);

    for (const auto& selected_frag_ids : frag_ids_crossjoin) {
      if (allow_runtime_interrupt) {
        bool isInterrupted = false;
        {
          mapd_shared_lock<mapd_shared_mutex> session_read_lock(executor_session_mutex_);
          const auto query_session = getCurrentQuerySession(session_read_lock);
          isInterrupted =
              checkIsQuerySessionInterrupted(query_session, session_read_lock);
        }
        if (isInterrupted) {
          throw QueryExecutionError(ERR_INTERRUPTED);
        }
      }
      std::vector<const int8_t*> frag_col_buffers(
          plan_state_->global_to_local_col_ids_.size());
      for (const auto& col_id : pair.second) {
        CHECK(col_id);
        const int table_id = col_id->getScanDesc().getTableId();
        CHECK_EQ(table_id, pair.first);
        const auto cd = try_get_column_descriptor(col_id.get(), cat);
        if (cd && cd->isVirtualCol) {
          CHECK_EQ("rowid", cd->columnName);
          continue;
        }
        const auto fragments_it = all_tables_fragments.find(table_id);
        CHECK(fragments_it != all_tables_fragments.end());
        const auto fragments = fragments_it->second;
        auto it = plan_state_->global_to_local_col_ids_.find(*col_id);
        CHECK(it != plan_state_->global_to_local_col_ids_.end());
        CHECK_LT(static_cast<size_t>(it->second),
                 plan_state_->global_to_local_col_ids_.size());
        const size_t frag_id = ra_exe_unit.union_all
                                   ? 0
                                   : selected_frag_ids[local_col_to_frag_pos[it->second]];
        if (!fragments->size()) {
          return {};
        }
        CHECK_LT(frag_id, fragments->size());
        auto memory_level_for_column = memory_level;
        if (plan_state_->columns_to_fetch_.find(
                std::make_pair(col_id->getScanDesc().getTableId(), col_id->getColId())) ==
            plan_state_->columns_to_fetch_.end()) {
          memory_level_for_column = Data_Namespace::CPU_LEVEL;
        }
        if (col_id->getScanDesc().getSourceType() == InputSourceType::RESULT) {
          frag_col_buffers[it->second] =
              column_fetcher.getResultSetColumn(col_id.get(),
                                                memory_level_for_column,
                                                device_id,
                                                device_allocator,
                                                thread_idx);
        } else {
          if (needFetchAllFragments(*col_id, ra_exe_unit, selected_fragments)) {
            frag_col_buffers[it->second] =
                column_fetcher.getAllTableColumnFragments(table_id,
                                                          col_id->getColId(),
                                                          all_tables_fragments,
                                                          memory_level_for_column,
                                                          device_id,
                                                          device_allocator,
                                                          thread_idx);
          } else {
            frag_col_buffers[it->second] =
                column_fetcher.getOneTableColumnFragment(table_id,
                                                         frag_id,
                                                         col_id->getColId(),
                                                         all_tables_fragments,
                                                         chunks,
                                                         chunk_iterators,
                                                         memory_level_for_column,
                                                         device_id,
                                                         device_allocator);
          }
        }
      }
      all_frag_col_buffers.push_back(frag_col_buffers);
    }
    std::vector<std::vector<int64_t>> num_rows;
    std::vector<std::vector<uint64_t>> frag_offsets;
    std::tie(num_rows, frag_offsets) = getRowCountAndOffsetForAllFrags(
        ra_exe_unit, frag_ids_crossjoin, ra_exe_unit.input_descs, all_tables_fragments);
    all_num_rows.insert(all_num_rows.end(), num_rows.begin(), num_rows.end());
    all_frag_offsets.insert(
        all_frag_offsets.end(), frag_offsets.begin(), frag_offsets.end());
  }
  // The hack below assumes a particular table traversal order which is not
  // always achieved due to unordered map in the outermost loop. According
  // to the code below we expect NULLs in even positions of all_frag_col_buffers[0]
  // and odd positions of all_frag_col_buffers[1]. As an additional hack we
  // swap these vectors if NULLs are not on expected positions.
  if (all_frag_col_buffers[0].size() > 1 && all_frag_col_buffers[0][0] &&
      !all_frag_col_buffers[0][1]) {
    std::swap(all_frag_col_buffers[0], all_frag_col_buffers[1]);
  }
  // UNION ALL hacks.
  VLOG(2) << "all_frag_col_buffers=" << shared::printContainer(all_frag_col_buffers);
  for (size_t i = 0; i < all_frag_col_buffers.front().size(); ++i) {
    all_frag_col_buffers[i & 1][i] = all_frag_col_buffers[i & 1][i ^ 1];
  }
  if (input_descs_index == input_col_descs_index) {
    std::swap(all_frag_col_buffers[0], all_frag_col_buffers[1]);
  }

  VLOG(2) << "all_frag_col_buffers=" << shared::printContainer(all_frag_col_buffers)
          << " all_num_rows=" << shared::printContainer(all_num_rows)
          << " all_frag_offsets=" << shared::printContainer(all_frag_offsets)
          << " input_col_descs_index=" << input_col_descs_index;
  return {{all_frag_col_buffers[input_descs_index]},
          {{all_num_rows[0][input_descs_index]}},
          {{all_frag_offsets[0][input_descs_index]}}};
}

std::vector<size_t> Executor::getFragmentCount(const FragmentsList& selected_fragments,
                                               const size_t scan_idx,
                                               const RelAlgExecutionUnit& ra_exe_unit) {
  if ((ra_exe_unit.input_descs.size() > size_t(2) || !ra_exe_unit.join_quals.empty()) &&
      scan_idx > 0 &&
      !plan_state_->join_info_.sharded_range_table_indices_.count(scan_idx) &&
      !selected_fragments[scan_idx].fragment_ids.empty()) {
    // Fetch all fragments
    return {size_t(0)};
  }

  return selected_fragments[scan_idx].fragment_ids;
}

void Executor::buildSelectedFragsMapping(
    std::vector<std::vector<size_t>>& selected_fragments_crossjoin,
    std::vector<size_t>& local_col_to_frag_pos,
    const std::list<std::shared_ptr<const InputColDescriptor>>& col_global_ids,
    const FragmentsList& selected_fragments,
    const RelAlgExecutionUnit& ra_exe_unit) {
  local_col_to_frag_pos.resize(plan_state_->global_to_local_col_ids_.size());
  size_t frag_pos{0};
  const auto& input_descs = ra_exe_unit.input_descs;
  for (size_t scan_idx = 0; scan_idx < input_descs.size(); ++scan_idx) {
    const int table_id = input_descs[scan_idx].getTableId();
    CHECK_EQ(selected_fragments[scan_idx].table_id, table_id);
    selected_fragments_crossjoin.push_back(
        getFragmentCount(selected_fragments, scan_idx, ra_exe_unit));
    for (const auto& col_id : col_global_ids) {
      CHECK(col_id);
      const auto& input_desc = col_id->getScanDesc();
      if (input_desc.getTableId() != table_id ||
          input_desc.getNestLevel() != static_cast<int>(scan_idx)) {
        continue;
      }
      auto it = plan_state_->global_to_local_col_ids_.find(*col_id);
      CHECK(it != plan_state_->global_to_local_col_ids_.end());
      CHECK_LT(static_cast<size_t>(it->second),
               plan_state_->global_to_local_col_ids_.size());
      local_col_to_frag_pos[it->second] = frag_pos;
    }
    ++frag_pos;
  }
}

void Executor::buildSelectedFragsMappingForUnion(
    std::vector<std::vector<size_t>>& selected_fragments_crossjoin,
    std::vector<size_t>& local_col_to_frag_pos,
    const std::list<std::shared_ptr<const InputColDescriptor>>& col_global_ids,
    const FragmentsList& selected_fragments,
    const RelAlgExecutionUnit& ra_exe_unit) {
  local_col_to_frag_pos.resize(plan_state_->global_to_local_col_ids_.size());
  size_t frag_pos{0};
  const auto& input_descs = ra_exe_unit.input_descs;
  for (size_t scan_idx = 0; scan_idx < input_descs.size(); ++scan_idx) {
    const int table_id = input_descs[scan_idx].getTableId();
    // selected_fragments here is from assignFragsToKernelDispatch
    // execution_kernel.fragments
    if (selected_fragments[0].table_id != table_id) {  // TODO 0
      continue;
    }
    // CHECK_EQ(selected_fragments[scan_idx].table_id, table_id);
    selected_fragments_crossjoin.push_back(
        // getFragmentCount(selected_fragments, scan_idx, ra_exe_unit));
        {size_t(1)});  // TODO
    for (const auto& col_id : col_global_ids) {
      CHECK(col_id);
      const auto& input_desc = col_id->getScanDesc();
      if (input_desc.getTableId() != table_id ||
          input_desc.getNestLevel() != static_cast<int>(scan_idx)) {
        continue;
      }
      auto it = plan_state_->global_to_local_col_ids_.find(*col_id);
      CHECK(it != plan_state_->global_to_local_col_ids_.end());
      CHECK_LT(static_cast<size_t>(it->second),
               plan_state_->global_to_local_col_ids_.size());
      local_col_to_frag_pos[it->second] = frag_pos;
    }
    ++frag_pos;
  }
}

namespace {

class OutVecOwner {
 public:
  OutVecOwner(const std::vector<int64_t*>& out_vec) : out_vec_(out_vec) {}
  ~OutVecOwner() {
    for (auto out : out_vec_) {
      delete[] out;
    }
  }

 private:
  std::vector<int64_t*> out_vec_;
};
}  // namespace

int32_t Executor::executePlanWithoutGroupBy(
    const RelAlgExecutionUnit& ra_exe_unit,
    const CompilationResult& compilation_result,
    const bool hoist_literals,
    ResultSetPtr& results,
    const std::vector<Analyzer::Expr*>& target_exprs,
    const ExecutorDeviceType device_type,
    std::vector<std::vector<const int8_t*>>& col_buffers,
    QueryExecutionContext* query_exe_context,
    const std::vector<std::vector<int64_t>>& num_rows,
    const std::vector<std::vector<uint64_t>>& frag_offsets,
    Data_Namespace::DataMgr* data_mgr,
    const int device_id,
    const uint32_t start_rowid,
    const uint32_t num_tables,
    const bool allow_runtime_interrupt,
    RenderInfo* render_info) {
  INJECT_TIMER(executePlanWithoutGroupBy);
  auto timer = DEBUG_TIMER(__func__);
  CHECK(!results);
  if (col_buffers.empty()) {
    return 0;
  }

  RenderAllocatorMap* render_allocator_map_ptr = nullptr;
  if (render_info) {
    // TODO(adb): make sure that we either never get here in the CPU case, or if we do get
    // here, we are in non-insitu mode.
    CHECK(render_info->useCudaBuffers() || !render_info->isPotentialInSituRender())
        << "CUDA disabled rendering in the executePlanWithoutGroupBy query path is "
           "currently unsupported.";
    render_allocator_map_ptr = render_info->render_allocator_map_ptr.get();
  }

  int32_t error_code = device_type == ExecutorDeviceType::GPU ? 0 : start_rowid;
  std::vector<int64_t*> out_vec;
  const auto hoist_buf = serializeLiterals(compilation_result.literal_values, device_id);
  const auto join_hash_table_ptrs = getJoinHashTablePtrs(device_type, device_id);
  std::unique_ptr<OutVecOwner> output_memory_scope;
  if (allow_runtime_interrupt) {
    bool isInterrupted = false;
    {
      mapd_shared_lock<mapd_shared_mutex> session_read_lock(executor_session_mutex_);
      const auto query_session = getCurrentQuerySession(session_read_lock);
      isInterrupted = checkIsQuerySessionInterrupted(query_session, session_read_lock);
    }
    if (isInterrupted) {
      throw QueryExecutionError(ERR_INTERRUPTED);
    }
  }
  if (g_enable_dynamic_watchdog && interrupted_.load()) {
    throw QueryExecutionError(ERR_INTERRUPTED);
  }
  if (device_type == ExecutorDeviceType::CPU) {
    auto cpu_generated_code = std::dynamic_pointer_cast<CpuCompilationContext>(
        compilation_result.generated_code);
    CHECK(cpu_generated_code);
    out_vec = query_exe_context->launchCpuCode(ra_exe_unit,
                                               cpu_generated_code.get(),
                                               hoist_literals,
                                               hoist_buf,
                                               col_buffers,
                                               num_rows,
                                               frag_offsets,
                                               0,
                                               &error_code,
                                               num_tables,
                                               join_hash_table_ptrs);
    output_memory_scope.reset(new OutVecOwner(out_vec));
  } else {
    try {
      out_vec = query_exe_context->launchGpuCode(
          ra_exe_unit,
          compilation_result.generated_code.get(),
          hoist_literals,
          hoist_buf,
          col_buffers,
          num_rows,
          frag_offsets,
          0,
          data_mgr,
          blockSize(),
          gridSize(),
          device_id,
          compilation_result.gpu_smem_context.getSharedMemorySize(),
          &error_code,
          num_tables,
          allow_runtime_interrupt,
          join_hash_table_ptrs,
          render_allocator_map_ptr);
      output_memory_scope.reset(new OutVecOwner(out_vec));
    } catch (const OutOfMemory&) {
      return ERR_OUT_OF_GPU_MEM;
    } catch (const std::exception& e) {
      LOG(FATAL) << "Error launching the GPU kernel: " << e.what();
    }
  }
  if (error_code == Executor::ERR_OVERFLOW_OR_UNDERFLOW ||
      error_code == Executor::ERR_DIV_BY_ZERO ||
      error_code == Executor::ERR_OUT_OF_TIME ||
      error_code == Executor::ERR_INTERRUPTED ||
      error_code == Executor::ERR_SINGLE_VALUE_FOUND_MULTIPLE_VALUES ||
      error_code == Executor::ERR_GEOS) {
    return error_code;
  }
  if (ra_exe_unit.estimator) {
    CHECK(!error_code);
    results =
        std::shared_ptr<ResultSet>(query_exe_context->estimator_result_set_.release());
    return 0;
  }
  std::vector<int64_t> reduced_outs;
  const auto num_frags = col_buffers.size();
  const size_t entry_count =
      device_type == ExecutorDeviceType::GPU
          ? (compilation_result.gpu_smem_context.isSharedMemoryUsed()
                 ? 1
                 : blockSize() * gridSize() * num_frags)
          : num_frags;
  if (size_t(1) == entry_count) {
    for (auto out : out_vec) {
      CHECK(out);
      reduced_outs.push_back(*out);
    }
  } else {
    size_t out_vec_idx = 0;

    for (const auto target_expr : target_exprs) {
      const auto agg_info = get_target_info(target_expr, g_bigint_count);
      CHECK(agg_info.is_agg || dynamic_cast<Analyzer::Constant*>(target_expr))
          << target_expr->toString();

      const int num_iterations = agg_info.sql_type.is_geometry()
                                     ? agg_info.sql_type.get_physical_coord_cols()
                                     : 1;

      for (int i = 0; i < num_iterations; i++) {
        int64_t val1;
        const bool float_argument_input = takes_float_argument(agg_info);
        if (is_distinct_target(agg_info) || agg_info.agg_kind == kAPPROX_MEDIAN) {
          CHECK(agg_info.agg_kind == kCOUNT ||
                agg_info.agg_kind == kAPPROX_COUNT_DISTINCT ||
                agg_info.agg_kind == kAPPROX_MEDIAN);
          val1 = out_vec[out_vec_idx][0];
          error_code = 0;
        } else {
          const auto chosen_bytes = static_cast<size_t>(
              query_exe_context->query_mem_desc_.getPaddedSlotWidthBytes(out_vec_idx));
          std::tie(val1, error_code) = Executor::reduceResults(
              agg_info.agg_kind,
              agg_info.sql_type,
              query_exe_context->getAggInitValForIndex(out_vec_idx),
              float_argument_input ? sizeof(int32_t) : chosen_bytes,
              out_vec[out_vec_idx],
              entry_count,
              false,
              float_argument_input);
        }
        if (error_code) {
          break;
        }
        reduced_outs.push_back(val1);
        if (agg_info.agg_kind == kAVG ||
            (agg_info.agg_kind == kSAMPLE &&
             (agg_info.sql_type.is_varlen() || agg_info.sql_type.is_geometry()))) {
          const auto chosen_bytes = static_cast<size_t>(
              query_exe_context->query_mem_desc_.getPaddedSlotWidthBytes(out_vec_idx +
                                                                         1));
          int64_t val2;
          std::tie(val2, error_code) = Executor::reduceResults(
              agg_info.agg_kind == kAVG ? kCOUNT : agg_info.agg_kind,
              agg_info.sql_type,
              query_exe_context->getAggInitValForIndex(out_vec_idx + 1),
              float_argument_input ? sizeof(int32_t) : chosen_bytes,
              out_vec[out_vec_idx + 1],
              entry_count,
              false,
              false);
          if (error_code) {
            break;
          }
          reduced_outs.push_back(val2);
          ++out_vec_idx;
        }
        ++out_vec_idx;
      }
    }
  }

  if (error_code) {
    return error_code;
  }

  CHECK_EQ(size_t(1), query_exe_context->query_buffers_->result_sets_.size());
  auto rows_ptr = std::shared_ptr<ResultSet>(
      query_exe_context->query_buffers_->result_sets_[0].release());
  rows_ptr->fillOneEntry(reduced_outs);
  results = std::move(rows_ptr);
  return error_code;
}

namespace {

bool check_rows_less_than_needed(const ResultSetPtr& results, const size_t scan_limit) {
  CHECK(scan_limit);
  return results && results->rowCount() < scan_limit;
}

}  // namespace

int32_t Executor::executePlanWithGroupBy(
    const RelAlgExecutionUnit& ra_exe_unit,
    const CompilationResult& compilation_result,
    const bool hoist_literals,
    ResultSetPtr& results,
    const ExecutorDeviceType device_type,
    std::vector<std::vector<const int8_t*>>& col_buffers,
    const std::vector<size_t> outer_tab_frag_ids,
    QueryExecutionContext* query_exe_context,
    const std::vector<std::vector<int64_t>>& num_rows,
    const std::vector<std::vector<uint64_t>>& frag_offsets,
    Data_Namespace::DataMgr* data_mgr,
    const int device_id,
    const int outer_table_id,
    const int64_t scan_limit,
    const uint32_t start_rowid,
    const uint32_t num_tables,
    const bool allow_runtime_interrupt,
    RenderInfo* render_info) {
  auto timer = DEBUG_TIMER(__func__);
  INJECT_TIMER(executePlanWithGroupBy);
  CHECK(!results);
  if (col_buffers.empty()) {
    return 0;
  }
  CHECK_NE(ra_exe_unit.groupby_exprs.size(), size_t(0));
  // TODO(alex):
  // 1. Optimize size (make keys more compact).
  // 2. Resize on overflow.
  // 3. Optimize runtime.
  auto hoist_buf = serializeLiterals(compilation_result.literal_values, device_id);
  int32_t error_code = device_type == ExecutorDeviceType::GPU ? 0 : start_rowid;
  const auto join_hash_table_ptrs = getJoinHashTablePtrs(device_type, device_id);
  if (allow_runtime_interrupt) {
    bool isInterrupted = false;
    {
      mapd_shared_lock<mapd_shared_mutex> session_read_lock(executor_session_mutex_);
      const auto query_session = getCurrentQuerySession(session_read_lock);
      isInterrupted = checkIsQuerySessionInterrupted(query_session, session_read_lock);
    }
    if (isInterrupted) {
      throw QueryExecutionError(ERR_INTERRUPTED);
    }
  }
  if (g_enable_dynamic_watchdog && interrupted_.load()) {
    return ERR_INTERRUPTED;
  }

  RenderAllocatorMap* render_allocator_map_ptr = nullptr;
  if (render_info && render_info->useCudaBuffers()) {
    render_allocator_map_ptr = render_info->render_allocator_map_ptr.get();
  }

  VLOG(2) << "bool(ra_exe_unit.union_all)=" << bool(ra_exe_unit.union_all)
          << " ra_exe_unit.input_descs="
          << shared::printContainer(ra_exe_unit.input_descs)
          << " ra_exe_unit.input_col_descs="
          << shared::printContainer(ra_exe_unit.input_col_descs)
          << " ra_exe_unit.scan_limit=" << ra_exe_unit.scan_limit
          << " num_rows=" << shared::printContainer(num_rows)
          << " frag_offsets=" << shared::printContainer(frag_offsets)
          << " query_exe_context->query_buffers_->num_rows_="
          << query_exe_context->query_buffers_->num_rows_
          << " query_exe_context->query_mem_desc_.getEntryCount()="
          << query_exe_context->query_mem_desc_.getEntryCount()
          << " device_id=" << device_id << " outer_table_id=" << outer_table_id
          << " scan_limit=" << scan_limit << " start_rowid=" << start_rowid
          << " num_tables=" << num_tables;

  RelAlgExecutionUnit ra_exe_unit_copy = ra_exe_unit;
  // For UNION ALL, filter out input_descs and input_col_descs that are not associated
  // with outer_table_id.
  if (ra_exe_unit_copy.union_all) {
    // Sort outer_table_id first, then pop the rest off of ra_exe_unit_copy.input_descs.
    std::stable_sort(ra_exe_unit_copy.input_descs.begin(),
                     ra_exe_unit_copy.input_descs.end(),
                     [outer_table_id](auto const& a, auto const& b) {
                       return a.getTableId() == outer_table_id &&
                              b.getTableId() != outer_table_id;
                     });
    while (!ra_exe_unit_copy.input_descs.empty() &&
           ra_exe_unit_copy.input_descs.back().getTableId() != outer_table_id) {
      ra_exe_unit_copy.input_descs.pop_back();
    }
    // Filter ra_exe_unit_copy.input_col_descs.
    ra_exe_unit_copy.input_col_descs.remove_if(
        [outer_table_id](auto const& input_col_desc) {
          return input_col_desc->getScanDesc().getTableId() != outer_table_id;
        });
    query_exe_context->query_mem_desc_.setEntryCount(ra_exe_unit_copy.scan_limit);
  }

  if (device_type == ExecutorDeviceType::CPU) {
    auto cpu_generated_code = std::dynamic_pointer_cast<CpuCompilationContext>(
        compilation_result.generated_code);
    CHECK(cpu_generated_code);
    query_exe_context->launchCpuCode(
        ra_exe_unit_copy,
        cpu_generated_code.get(),
        hoist_literals,
        hoist_buf,
        col_buffers,
        num_rows,
        frag_offsets,
        ra_exe_unit_copy.union_all ? ra_exe_unit_copy.scan_limit : scan_limit,
        &error_code,
        num_tables,
        join_hash_table_ptrs);
  } else {
    try {
      query_exe_context->launchGpuCode(
          ra_exe_unit_copy,
          compilation_result.generated_code.get(),
          hoist_literals,
          hoist_buf,
          col_buffers,
          num_rows,
          frag_offsets,
          ra_exe_unit_copy.union_all ? ra_exe_unit_copy.scan_limit : scan_limit,
          data_mgr,
          blockSize(),
          gridSize(),
          device_id,
          compilation_result.gpu_smem_context.getSharedMemorySize(),
          &error_code,
          num_tables,
          allow_runtime_interrupt,
          join_hash_table_ptrs,
          render_allocator_map_ptr);
    } catch (const OutOfMemory&) {
      return ERR_OUT_OF_GPU_MEM;
    } catch (const OutOfRenderMemory&) {
      return ERR_OUT_OF_RENDER_MEM;
    } catch (const StreamingTopNNotSupportedInRenderQuery&) {
      return ERR_STREAMING_TOP_N_NOT_SUPPORTED_IN_RENDER_QUERY;
    } catch (const std::exception& e) {
      LOG(FATAL) << "Error launching the GPU kernel: " << e.what();
    }
  }

  if (error_code == Executor::ERR_OVERFLOW_OR_UNDERFLOW ||
      error_code == Executor::ERR_DIV_BY_ZERO ||
      error_code == Executor::ERR_OUT_OF_TIME ||
      error_code == Executor::ERR_INTERRUPTED ||
      error_code == Executor::ERR_SINGLE_VALUE_FOUND_MULTIPLE_VALUES ||
      error_code == Executor::ERR_GEOS) {
    return error_code;
  }

  if (error_code != Executor::ERR_OVERFLOW_OR_UNDERFLOW &&
      error_code != Executor::ERR_DIV_BY_ZERO && !render_allocator_map_ptr) {
    results = query_exe_context->getRowSet(ra_exe_unit_copy,
                                           query_exe_context->query_mem_desc_);
    CHECK(results);
    VLOG(2) << "results->rowCount()=" << results->rowCount();
    results->holdLiterals(hoist_buf);
  }
  if (error_code < 0 && render_allocator_map_ptr) {
    auto const adjusted_scan_limit =
        ra_exe_unit_copy.union_all ? ra_exe_unit_copy.scan_limit : scan_limit;
    // More rows passed the filter than available slots. We don't have a count to check,
    // so assume we met the limit if a scan limit is set
    if (adjusted_scan_limit != 0) {
      return 0;
    } else {
      return error_code;
    }
  }
  if (error_code && (!scan_limit || check_rows_less_than_needed(results, scan_limit))) {
    return error_code;  // unlucky, not enough results and we ran out of slots
  }

  return 0;
}

std::vector<int8_t*> Executor::getJoinHashTablePtrs(const ExecutorDeviceType device_type,
                                                    const int device_id) {
  std::vector<int8_t*> table_ptrs;
  const auto& join_hash_tables = plan_state_->join_info_.join_hash_tables_;
  for (auto hash_table : join_hash_tables) {
    if (!hash_table) {
      CHECK(table_ptrs.empty());
      return {};
    }
    table_ptrs.push_back(hash_table->getJoinHashBuffer(
        device_type, device_type == ExecutorDeviceType::GPU ? device_id : 0));
  }
  return table_ptrs;
}

void Executor::nukeOldState(const bool allow_lazy_fetch,
                            const std::vector<InputTableInfo>& query_infos,
                            const PlanState::DeletedColumnsMap& deleted_cols_map,
                            const RelAlgExecutionUnit* ra_exe_unit) {
  kernel_queue_time_ms_ = 0;
  compilation_queue_time_ms_ = 0;
  const bool contains_left_deep_outer_join =
      ra_exe_unit && std::find_if(ra_exe_unit->join_quals.begin(),
                                  ra_exe_unit->join_quals.end(),
                                  [](const JoinCondition& join_condition) {
                                    return join_condition.type == JoinType::LEFT;
                                  }) != ra_exe_unit->join_quals.end();
  cgen_state_.reset(new CgenState(query_infos.size(), contains_left_deep_outer_join));
  plan_state_.reset(new PlanState(allow_lazy_fetch && !contains_left_deep_outer_join,
                                  query_infos,
                                  deleted_cols_map,
                                  this));
}

void Executor::preloadFragOffsets(const std::vector<InputDescriptor>& input_descs,
                                  const std::vector<InputTableInfo>& query_infos) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  const auto ld_count = input_descs.size();
  auto frag_off_ptr = get_arg_by_name(cgen_state_->row_func_, "frag_row_off");
  for (size_t i = 0; i < ld_count; ++i) {
    CHECK_LT(i, query_infos.size());
    const auto frag_count = query_infos[i].info.fragments.size();
    if (i > 0) {
      cgen_state_->frag_offsets_.push_back(nullptr);
    } else {
      if (frag_count > 1) {
        cgen_state_->frag_offsets_.push_back(
            cgen_state_->ir_builder_.CreateLoad(frag_off_ptr));
      } else {
        cgen_state_->frag_offsets_.push_back(nullptr);
      }
    }
  }
}

Executor::JoinHashTableOrError Executor::buildHashTableForQualifier(
    const std::shared_ptr<Analyzer::BinOper>& qual_bin_oper,
    const std::vector<InputTableInfo>& query_infos,
    const MemoryLevel memory_level,
    const JoinType join_type,
    const HashType preferred_hash_type,
    ColumnCacheMap& column_cache,
    const RegisteredQueryHint& query_hint) {
  if (!g_enable_overlaps_hashjoin && qual_bin_oper->is_overlaps_oper()) {
    return {nullptr, "Overlaps hash join disabled, attempting to fall back to loop join"};
  }
  if (g_enable_dynamic_watchdog && interrupted_.load()) {
    throw QueryExecutionError(ERR_INTERRUPTED);
  }
  try {
    auto tbl = HashJoin::getInstance(qual_bin_oper,
                                     query_infos,
                                     memory_level,
                                     join_type,
                                     preferred_hash_type,
                                     deviceCountForMemoryLevel(memory_level),
                                     column_cache,
                                     this,
                                     query_hint);
    return {tbl, ""};
  } catch (const HashJoinFail& e) {
    return {nullptr, e.what()};
  }
}

int8_t Executor::warpSize() const {
  CHECK(catalog_);
  const auto cuda_mgr = catalog_->getDataMgr().getCudaMgr();
  CHECK(cuda_mgr);
  const auto& dev_props = cuda_mgr->getAllDeviceProperties();
  CHECK(!dev_props.empty());
  return dev_props.front().warpSize;
}

unsigned Executor::gridSize() const {
  CHECK(catalog_);
  const auto cuda_mgr = catalog_->getDataMgr().getCudaMgr();
  if (!cuda_mgr) {
    return 0;
  }
  return grid_size_x_ ? grid_size_x_ : 2 * cuda_mgr->getMinNumMPsForAllDevices();
}

unsigned Executor::numBlocksPerMP() const {
  CHECK(catalog_);
  const auto cuda_mgr = catalog_->getDataMgr().getCudaMgr();
  CHECK(cuda_mgr);
  return grid_size_x_ ? std::ceil(grid_size_x_ / cuda_mgr->getMinNumMPsForAllDevices())
                      : 2;
}

unsigned Executor::blockSize() const {
  CHECK(catalog_);
  const auto cuda_mgr = catalog_->getDataMgr().getCudaMgr();
  if (!cuda_mgr) {
    return 0;
  }
  const auto& dev_props = cuda_mgr->getAllDeviceProperties();
  return block_size_x_ ? block_size_x_ : dev_props.front().maxThreadsPerBlock;
}

size_t Executor::maxGpuSlabSize() const {
  return max_gpu_slab_size_;
}

int64_t Executor::deviceCycles(int milliseconds) const {
  CHECK(catalog_);
  const auto cuda_mgr = catalog_->getDataMgr().getCudaMgr();
  CHECK(cuda_mgr);
  const auto& dev_props = cuda_mgr->getAllDeviceProperties();
  return static_cast<int64_t>(dev_props.front().clockKhz) * milliseconds;
}

llvm::Value* Executor::castToFP(llvm::Value* value,
                                SQLTypeInfo const& from_ti,
                                SQLTypeInfo const& to_ti) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  if (value->getType()->isIntegerTy() && from_ti.is_number() && to_ti.is_fp() &&
      (!from_ti.is_fp() || from_ti.get_size() != to_ti.get_size())) {
    llvm::Type* fp_type{nullptr};
    switch (to_ti.get_size()) {
      case 4:
        fp_type = llvm::Type::getFloatTy(cgen_state_->context_);
        break;
      case 8:
        fp_type = llvm::Type::getDoubleTy(cgen_state_->context_);
        break;
      default:
        LOG(FATAL) << "Unsupported FP size: " << to_ti.get_size();
    }
    value = cgen_state_->ir_builder_.CreateSIToFP(value, fp_type);
    if (from_ti.get_scale()) {
      value = cgen_state_->ir_builder_.CreateFDiv(
          value,
          llvm::ConstantFP::get(value->getType(), exp_to_scale(from_ti.get_scale())));
    }
  }
  return value;
}

llvm::Value* Executor::castToIntPtrTyIn(llvm::Value* val, const size_t bitWidth) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  CHECK(val->getType()->isPointerTy());

  const auto val_ptr_type = static_cast<llvm::PointerType*>(val->getType());
  const auto val_type = val_ptr_type->getElementType();
  size_t val_width = 0;
  if (val_type->isIntegerTy()) {
    val_width = val_type->getIntegerBitWidth();
  } else {
    if (val_type->isFloatTy()) {
      val_width = 32;
    } else {
      CHECK(val_type->isDoubleTy());
      val_width = 64;
    }
  }
  CHECK_LT(size_t(0), val_width);
  if (bitWidth == val_width) {
    return val;
  }
  return cgen_state_->ir_builder_.CreateBitCast(
      val, llvm::PointerType::get(get_int_type(bitWidth, cgen_state_->context_), 0));
}

#define EXECUTE_INCLUDE
#include "ArrayOps.cpp"
#include "DateAdd.cpp"
#include "GeoOps.cpp"
#include "StringFunctions.cpp"
#include "TableFunctions/TableFunctionOps.cpp"
#undef EXECUTE_INCLUDE

namespace {
void add_deleted_col_to_map(PlanState::DeletedColumnsMap& deleted_cols_map,
                            const ColumnDescriptor* deleted_cd) {
  auto deleted_cols_it = deleted_cols_map.find(deleted_cd->tableId);
  if (deleted_cols_it == deleted_cols_map.end()) {
    CHECK(
        deleted_cols_map.insert(std::make_pair(deleted_cd->tableId, deleted_cd)).second);
  } else {
    CHECK_EQ(deleted_cd, deleted_cols_it->second);
  }
}
}  // namespace

std::tuple<RelAlgExecutionUnit, PlanState::DeletedColumnsMap> Executor::addDeletedColumn(
    const RelAlgExecutionUnit& ra_exe_unit,
    const CompilationOptions& co) {
  if (!co.filter_on_deleted_column) {
    return std::make_tuple(ra_exe_unit, PlanState::DeletedColumnsMap{});
  }
  auto ra_exe_unit_with_deleted = ra_exe_unit;
  PlanState::DeletedColumnsMap deleted_cols_map;
  for (const auto& input_table : ra_exe_unit_with_deleted.input_descs) {
    if (input_table.getSourceType() != InputSourceType::TABLE) {
      continue;
    }
    const auto td = catalog_->getMetadataForTable(input_table.getTableId());
    CHECK(td);
    const auto deleted_cd = catalog_->getDeletedColumnIfRowsDeleted(td);
    if (!deleted_cd) {
      continue;
    }
    CHECK(deleted_cd->columnType.is_boolean());
    // check deleted column is not already present
    bool found = false;
    for (const auto& input_col : ra_exe_unit_with_deleted.input_col_descs) {
      if (input_col.get()->getColId() == deleted_cd->columnId &&
          input_col.get()->getScanDesc().getTableId() == deleted_cd->tableId &&
          input_col.get()->getScanDesc().getNestLevel() == input_table.getNestLevel()) {
        found = true;
        add_deleted_col_to_map(deleted_cols_map, deleted_cd);
        break;
      }
    }
    if (!found) {
      // add deleted column
      ra_exe_unit_with_deleted.input_col_descs.emplace_back(new InputColDescriptor(
          deleted_cd->columnId, deleted_cd->tableId, input_table.getNestLevel()));
      add_deleted_col_to_map(deleted_cols_map, deleted_cd);
    }
  }
  return std::make_tuple(ra_exe_unit_with_deleted, deleted_cols_map);
}

namespace {
// Note(Wamsi): `get_hpt_overflow_underflow_safe_scaled_value` will return `true` for safe
// scaled epoch value and `false` for overflow/underflow values as the first argument of
// return type.
std::tuple<bool, int64_t, int64_t> get_hpt_overflow_underflow_safe_scaled_values(
    const int64_t chunk_min,
    const int64_t chunk_max,
    const SQLTypeInfo& lhs_type,
    const SQLTypeInfo& rhs_type) {
  const int32_t ldim = lhs_type.get_dimension();
  const int32_t rdim = rhs_type.get_dimension();
  CHECK(ldim != rdim);
  const auto scale = DateTimeUtils::get_timestamp_precision_scale(abs(rdim - ldim));
  if (ldim > rdim) {
    // LHS type precision is more than RHS col type. No chance of overflow/underflow.
    return {true, chunk_min / scale, chunk_max / scale};
  }

  using checked_int64_t = boost::multiprecision::number<
      boost::multiprecision::cpp_int_backend<64,
                                             64,
                                             boost::multiprecision::signed_magnitude,
                                             boost::multiprecision::checked,
                                             void>>;

  try {
    auto ret =
        std::make_tuple(true,
                        int64_t(checked_int64_t(chunk_min) * checked_int64_t(scale)),
                        int64_t(checked_int64_t(chunk_max) * checked_int64_t(scale)));
    return ret;
  } catch (const std::overflow_error& e) {
    // noop
  }
  return std::make_tuple(false, chunk_min, chunk_max);
}

}  // namespace

bool Executor::isFragmentFullyDeleted(
    const int table_id,
    const Fragmenter_Namespace::FragmentInfo& fragment) {
  // Skip temporary tables
  if (table_id < 0) {
    return false;
  }

  const auto td = catalog_->getMetadataForTable(fragment.physicalTableId);
  CHECK(td);
  const auto deleted_cd = catalog_->getDeletedColumnIfRowsDeleted(td);
  if (!deleted_cd) {
    return false;
  }

  const auto& chunk_type = deleted_cd->columnType;
  CHECK(chunk_type.is_boolean());

  const auto deleted_col_id = deleted_cd->columnId;
  auto chunk_meta_it = fragment.getChunkMetadataMap().find(deleted_col_id);
  if (chunk_meta_it != fragment.getChunkMetadataMap().end()) {
    const int64_t chunk_min =
        extract_min_stat(chunk_meta_it->second->chunkStats, chunk_type);
    const int64_t chunk_max =
        extract_max_stat(chunk_meta_it->second->chunkStats, chunk_type);
    if (chunk_min == 1 && chunk_max == 1) {  // Delete chunk if metadata says full bytemap
      // is true (signifying all rows deleted)
      return true;
    }
  }
  return false;
}

std::pair<bool, int64_t> Executor::skipFragment(
    const InputDescriptor& table_desc,
    const Fragmenter_Namespace::FragmentInfo& fragment,
    const std::list<std::shared_ptr<Analyzer::Expr>>& simple_quals,
    const std::vector<uint64_t>& frag_offsets,
    const size_t frag_idx) {
  const int table_id = table_desc.getTableId();

  // First check to see if all of fragment is deleted, in which case we know we can skip
  if (isFragmentFullyDeleted(table_id, fragment)) {
    VLOG(2) << "Skipping deleted fragment with table id: " << fragment.physicalTableId
            << ", fragment id: " << frag_idx;
    return {true, -1};
  }

  for (const auto& simple_qual : simple_quals) {
    const auto comp_expr =
        std::dynamic_pointer_cast<const Analyzer::BinOper>(simple_qual);
    if (!comp_expr) {
      // is this possible?
      return {false, -1};
    }
    const auto lhs = comp_expr->get_left_operand();
    auto lhs_col = dynamic_cast<const Analyzer::ColumnVar*>(lhs);
    if (!lhs_col || !lhs_col->get_table_id() || lhs_col->get_rte_idx()) {
      // See if lhs is a simple cast that was allowed through normalize_simple_predicate
      auto lhs_uexpr = dynamic_cast<const Analyzer::UOper*>(lhs);
      if (lhs_uexpr) {
        CHECK(lhs_uexpr->get_optype() ==
              kCAST);  // We should have only been passed a cast expression
        lhs_col = dynamic_cast<const Analyzer::ColumnVar*>(lhs_uexpr->get_operand());
        if (!lhs_col || !lhs_col->get_table_id() || lhs_col->get_rte_idx()) {
          continue;
        }
      } else {
        continue;
      }
    }
    const auto rhs = comp_expr->get_right_operand();
    const auto rhs_const = dynamic_cast<const Analyzer::Constant*>(rhs);
    if (!rhs_const) {
      // is this possible?
      return {false, -1};
    }
    if (!lhs->get_type_info().is_integer() && !lhs->get_type_info().is_time()) {
      continue;
    }
    const int col_id = lhs_col->get_column_id();
    auto chunk_meta_it = fragment.getChunkMetadataMap().find(col_id);
    int64_t chunk_min{0};
    int64_t chunk_max{0};
    bool is_rowid{false};
    size_t start_rowid{0};
    if (chunk_meta_it == fragment.getChunkMetadataMap().end()) {
      auto cd = get_column_descriptor(col_id, table_id, *catalog_);
      if (cd->isVirtualCol) {
        CHECK(cd->columnName == "rowid");
        const auto& table_generation = getTableGeneration(table_id);
        start_rowid = table_generation.start_rowid;
        chunk_min = frag_offsets[frag_idx] + start_rowid;
        chunk_max = frag_offsets[frag_idx + 1] - 1 + start_rowid;
        is_rowid = true;
      }
    } else {
      const auto& chunk_type = lhs_col->get_type_info();
      chunk_min = extract_min_stat(chunk_meta_it->second->chunkStats, chunk_type);
      chunk_max = extract_max_stat(chunk_meta_it->second->chunkStats, chunk_type);
    }
    if (chunk_min > chunk_max) {
      // invalid metadata range, do not skip fragment
      return {false, -1};
    }
    if (lhs->get_type_info().is_timestamp() &&
        (lhs_col->get_type_info().get_dimension() !=
         rhs_const->get_type_info().get_dimension()) &&
        (lhs_col->get_type_info().is_high_precision_timestamp() ||
         rhs_const->get_type_info().is_high_precision_timestamp())) {
      // If original timestamp lhs col has different precision,
      // column metadata holds value in original precision
      // therefore adjust rhs value to match lhs precision

      // Note(Wamsi): We adjust rhs const value instead of lhs value to not
      // artificially limit the lhs column range. RHS overflow/underflow is already
      // been validated in `TimeGM::get_overflow_underflow_safe_epoch`.
      bool is_valid;
      std::tie(is_valid, chunk_min, chunk_max) =
          get_hpt_overflow_underflow_safe_scaled_values(
              chunk_min, chunk_max, lhs_col->get_type_info(), rhs_const->get_type_info());
      if (!is_valid) {
        VLOG(4) << "Overflow/Underflow detecting in fragments skipping logic.\nChunk min "
                   "value: "
                << std::to_string(chunk_min)
                << "\nChunk max value: " << std::to_string(chunk_max)
                << "\nLHS col precision is: "
                << std::to_string(lhs_col->get_type_info().get_dimension())
                << "\nRHS precision is: "
                << std::to_string(rhs_const->get_type_info().get_dimension()) << ".";
        return {false, -1};
      }
    }
    llvm::LLVMContext local_context;
    CgenState local_cgen_state(local_context);
    CodeGenerator code_generator(&local_cgen_state, nullptr);

    const auto rhs_val =
        CodeGenerator::codegenIntConst(rhs_const, &local_cgen_state)->getSExtValue();

    switch (comp_expr->get_optype()) {
      case kGE:
        if (chunk_max < rhs_val) {
          return {true, -1};
        }
        break;
      case kGT:
        if (chunk_max <= rhs_val) {
          return {true, -1};
        }
        break;
      case kLE:
        if (chunk_min > rhs_val) {
          return {true, -1};
        }
        break;
      case kLT:
        if (chunk_min >= rhs_val) {
          return {true, -1};
        }
        break;
      case kEQ:
        if (chunk_min > rhs_val || chunk_max < rhs_val) {
          return {true, -1};
        } else if (is_rowid) {
          return {false, rhs_val - start_rowid};
        }
        break;
      default:
        break;
    }
  }
  return {false, -1};
}

/*
 *   The skipFragmentInnerJoins process all quals stored in the execution unit's
 * join_quals and gather all the ones that meet the "simple_qual" characteristics
 * (logical expressions with AND operations, etc.). It then uses the skipFragment function
 * to decide whether the fragment should be skipped or not. The fragment will be skipped
 * if at least one of these skipFragment calls return a true statment in its first value.
 *   - The code depends on skipFragment's output to have a meaningful (anything but -1)
 * second value only if its first value is "false".
 *   - It is assumed that {false, n  > -1} has higher priority than {true, -1},
 *     i.e., we only skip if none of the quals trigger the code to update the
 * rowid_lookup_key
 *   - Only AND operations are valid and considered:
 *     - `select * from t1,t2 where A and B and C`: A, B, and C are considered for causing
 * the skip
 *     - `select * from t1,t2 where (A or B) and C`: only C is considered
 *     - `select * from t1,t2 where A or B`: none are considered (no skipping).
 *   - NOTE: (re: intermediate projections) the following two queries are fundamentally
 * implemented differently, which cause the first one to skip correctly, but the second
 * one will not skip.
 *     -  e.g. #1, select * from t1 join t2 on (t1.i=t2.i) where (A and B); -- skips if
 * possible
 *     -  e.g. #2, select * from t1 join t2 on (t1.i=t2.i and A and B); -- intermediate
 * projection, no skipping
 */
std::pair<bool, int64_t> Executor::skipFragmentInnerJoins(
    const InputDescriptor& table_desc,
    const RelAlgExecutionUnit& ra_exe_unit,
    const Fragmenter_Namespace::FragmentInfo& fragment,
    const std::vector<uint64_t>& frag_offsets,
    const size_t frag_idx) {
  std::pair<bool, int64_t> skip_frag{false, -1};
  for (auto& inner_join : ra_exe_unit.join_quals) {
    if (inner_join.type != JoinType::INNER) {
      continue;
    }

    // extracting all the conjunctive simple_quals from the quals stored for the inner
    // join
    std::list<std::shared_ptr<Analyzer::Expr>> inner_join_simple_quals;
    for (auto& qual : inner_join.quals) {
      auto temp_qual = qual_to_conjunctive_form(qual);
      inner_join_simple_quals.insert(inner_join_simple_quals.begin(),
                                     temp_qual.simple_quals.begin(),
                                     temp_qual.simple_quals.end());
    }
    auto temp_skip_frag = skipFragment(
        table_desc, fragment, inner_join_simple_quals, frag_offsets, frag_idx);
    if (temp_skip_frag.second != -1) {
      skip_frag.second = temp_skip_frag.second;
      return skip_frag;
    } else {
      skip_frag.first = skip_frag.first || temp_skip_frag.first;
    }
  }
  return skip_frag;
}

AggregatedColRange Executor::computeColRangesCache(
    const std::unordered_set<PhysicalInput>& phys_inputs) {
  AggregatedColRange agg_col_range_cache;
  CHECK(catalog_);
  std::unordered_set<int> phys_table_ids;
  for (const auto& phys_input : phys_inputs) {
    phys_table_ids.insert(phys_input.table_id);
  }
  std::vector<InputTableInfo> query_infos;
  for (const int table_id : phys_table_ids) {
    query_infos.emplace_back(InputTableInfo{table_id, getTableInfo(table_id)});
  }
  for (const auto& phys_input : phys_inputs) {
    const auto cd =
        catalog_->getMetadataForColumn(phys_input.table_id, phys_input.col_id);
    CHECK(cd);
    if (ExpressionRange::typeSupportsRange(cd->columnType)) {
      const auto col_var = boost::make_unique<Analyzer::ColumnVar>(
          cd->columnType, phys_input.table_id, phys_input.col_id, 0);
      const auto col_range = getLeafColumnRange(col_var.get(), query_infos, this, false);
      agg_col_range_cache.setColRange(phys_input, col_range);
    }
  }
  return agg_col_range_cache;
}

StringDictionaryGenerations Executor::computeStringDictionaryGenerations(
    const std::unordered_set<PhysicalInput>& phys_inputs) {
  StringDictionaryGenerations string_dictionary_generations;
  CHECK(catalog_);
  for (const auto& phys_input : phys_inputs) {
    const auto cd =
        catalog_->getMetadataForColumn(phys_input.table_id, phys_input.col_id);
    CHECK(cd);
    const auto& col_ti =
        cd->columnType.is_array() ? cd->columnType.get_elem_type() : cd->columnType;
    if (col_ti.is_string() && col_ti.get_compression() == kENCODING_DICT) {
      const int dict_id = col_ti.get_comp_param();
      const auto dd = catalog_->getMetadataForDict(dict_id);
      CHECK(dd && dd->stringDict);
      string_dictionary_generations.setGeneration(dict_id,
                                                  dd->stringDict->storageEntryCount());
    }
  }
  return string_dictionary_generations;
}

TableGenerations Executor::computeTableGenerations(
    std::unordered_set<int> phys_table_ids) {
  TableGenerations table_generations;
  for (const int table_id : phys_table_ids) {
    const auto table_info = getTableInfo(table_id);
    table_generations.setGeneration(
        table_id,
        TableGeneration{static_cast<int64_t>(table_info.getPhysicalNumTuples()), 0});
  }
  return table_generations;
}

void Executor::setupCaching(const std::unordered_set<PhysicalInput>& phys_inputs,
                            const std::unordered_set<int>& phys_table_ids) {
  CHECK(catalog_);
  row_set_mem_owner_ =
      std::make_shared<RowSetMemoryOwner>(Executor::getArenaBlockSize(), cpu_threads());
  row_set_mem_owner_->setDictionaryGenerations(
      computeStringDictionaryGenerations(phys_inputs));
  agg_col_range_cache_ = computeColRangesCache(phys_inputs);
  table_generations_ = computeTableGenerations(phys_table_ids);
}

mapd_shared_mutex& Executor::getSessionLock() {
  return executor_session_mutex_;
}

QuerySessionId& Executor::getCurrentQuerySession(
    mapd_shared_lock<mapd_shared_mutex>& read_lock) {
  return current_query_session_;
}

void Executor::setCurrentQuerySession(const QuerySessionId& query_session,
                                      mapd_unique_lock<mapd_shared_mutex>& write_lock) {
  if (!query_session.empty()) {
    current_query_session_ = query_session;
  }
}

void Executor::setRunningExecutorId(const size_t id,
                                    mapd_unique_lock<mapd_shared_mutex>& write_lock) {
  running_query_executor_id_ = id;
}

size_t Executor::getRunningExecutorId(mapd_shared_lock<mapd_shared_mutex>& read_lock) {
  return running_query_executor_id_;
}

bool Executor::checkCurrentQuerySession(const QuerySessionId& candidate_query_session,
                                        mapd_shared_lock<mapd_shared_mutex>& read_lock) {
  // if current_query_session is equal to the candidate_query_session,
  // or it is empty session we consider
  return !candidate_query_session.empty() &&
         (current_query_session_ == candidate_query_session);
}

void Executor::invalidateRunningQuerySession(
    mapd_unique_lock<mapd_shared_mutex>& write_lock) {
  current_query_session_ = "";
  running_query_executor_id_ = Executor::UNITARY_EXECUTOR_ID;
}

CurrentQueryStatus Executor::attachExecutorToQuerySession(
    const QuerySessionId& query_session_id,
    const std::string& query_str,
    const std::string& query_submitted_time) {
  if (!query_session_id.empty()) {
    // if session is valid, do update 1) the exact executor id and 2) query status
    mapd_unique_lock<mapd_shared_mutex> write_lock(executor_session_mutex_);
    updateQuerySessionExecutorAssignment(
        query_session_id, query_submitted_time, executor_id_, write_lock);
    updateQuerySessionStatusWithLock(query_session_id,
                                     query_submitted_time,
                                     QuerySessionStatus::QueryStatus::PENDING_EXECUTOR,
                                     write_lock);
  }
  return {query_session_id, query_str};
}

void Executor::checkPendingQueryStatus(const QuerySessionId& query_session) {
  // check whether we are okay to execute the "pending" query
  // i.e., before running the query check if this query session is "ALREADY" interrupted
  mapd_unique_lock<mapd_shared_mutex> session_write_lock(executor_session_mutex_);
  if (query_session.empty()) {
    return;
  }
  if (queries_interrupt_flag_.find(query_session) == queries_interrupt_flag_.end()) {
    // something goes wrong since we assume this is caller's responsibility
    // (call this function only for enrolled query session)
    if (!queries_session_map_.count(query_session)) {
      VLOG(1) << "Interrupting pending query is not available since the query session is "
                 "not enrolled";
    } else {
      // here the query session is enrolled but the interrupt flag is not registered
      VLOG(1)
          << "Interrupting pending query is not available since its interrupt flag is "
             "not registered";
    }
    return;
  }
  if (queries_interrupt_flag_[query_session]) {
    throw QueryExecutionError(Executor::ERR_INTERRUPTED);
  }
}

void Executor::clearQuerySessionStatus(const QuerySessionId& query_session,
                                       const std::string& submitted_time_str,
                                       const bool acquire_spin_lock) {
  mapd_unique_lock<mapd_shared_mutex> session_write_lock(executor_session_mutex_);
  // clear the interrupt-related info for a finished query
  if (query_session.empty()) {
    return;
  }
  removeFromQuerySessionList(query_session, submitted_time_str, session_write_lock);
  if (query_session.compare(current_query_session_) == 0 &&
      running_query_executor_id_ == executor_id_) {
    invalidateRunningQuerySession(session_write_lock);
    if (acquire_spin_lock) {
      // try to unlock executor's internal spin lock (let say "L") iff it is acquired
      // otherwise we do not need to care about the "L" lock
      // i.e., import table does not have a code path towards Executor
      // so we just exploit executor's session management code and also global interrupt
      // flag excepting this "L" lock
      execute_spin_lock_.clear(std::memory_order_release);
    }
    resetInterrupt();
  }
}

void Executor::updateQuerySessionStatus(
    std::shared_ptr<const query_state::QueryState>& query_state,
    const QuerySessionStatus::QueryStatus new_query_status) {
  // update the running query session's the current status
  if (query_state) {
    mapd_unique_lock<mapd_shared_mutex> session_write_lock(executor_session_mutex_);
    auto query_session = query_state->getConstSessionInfo()->get_session_id();
    if (query_session.empty()) {
      return;
    }
    if (new_query_status == QuerySessionStatus::QueryStatus::RUNNING) {
      current_query_session_ = query_session;
      running_query_executor_id_ = executor_id_;
    }
    updateQuerySessionStatusWithLock(query_session,
                                     query_state->getQuerySubmittedTime(),
                                     new_query_status,
                                     session_write_lock);
  }
}

void Executor::updateQuerySessionStatus(
    const QuerySessionId& query_session,
    const std::string& submitted_time_str,
    const QuerySessionStatus::QueryStatus new_query_status) {
  // update the running query session's the current status
  mapd_unique_lock<mapd_shared_mutex> session_write_lock(executor_session_mutex_);
  if (query_session.empty()) {
    return;
  }
  if (new_query_status == QuerySessionStatus::QueryStatus::RUNNING) {
    current_query_session_ = query_session;
    running_query_executor_id_ = executor_id_;
  }
  updateQuerySessionStatusWithLock(
      query_session, submitted_time_str, new_query_status, session_write_lock);
}

void Executor::enrollQuerySession(
    const QuerySessionId& query_session,
    const std::string& query_str,
    const std::string& submitted_time_str,
    const size_t executor_id,
    const QuerySessionStatus::QueryStatus query_session_status) {
  // enroll the query session into the Executor's session map
  mapd_unique_lock<mapd_shared_mutex> session_write_lock(executor_session_mutex_);
  if (query_session.empty()) {
    return;
  }

  addToQuerySessionList(query_session,
                        query_str,
                        submitted_time_str,
                        executor_id,
                        query_session_status,
                        session_write_lock);

  if (query_session_status == QuerySessionStatus::QueryStatus::RUNNING) {
    current_query_session_ = query_session;
    running_query_executor_id_ = executor_id_;
  }
}

bool Executor::addToQuerySessionList(const QuerySessionId& query_session,
                                     const std::string& query_str,
                                     const std::string& submitted_time_str,
                                     const size_t executor_id,
                                     const QuerySessionStatus::QueryStatus query_status,
                                     mapd_unique_lock<mapd_shared_mutex>& write_lock) {
  // an internal API that enrolls the query session into the Executor's session map
  if (queries_session_map_.count(query_session)) {
    if (queries_session_map_.at(query_session).count(submitted_time_str)) {
      queries_session_map_.at(query_session).erase(submitted_time_str);
      queries_session_map_.at(query_session)
          .emplace(submitted_time_str,
                   QuerySessionStatus(query_session,
                                      executor_id,
                                      query_str,
                                      submitted_time_str,
                                      query_status));
    } else {
      queries_session_map_.at(query_session)
          .emplace(submitted_time_str,
                   QuerySessionStatus(query_session,
                                      executor_id,
                                      query_str,
                                      submitted_time_str,
                                      query_status));
    }
  } else {
    std::map<std::string, QuerySessionStatus> executor_per_query_map;
    executor_per_query_map.emplace(
        submitted_time_str,
        QuerySessionStatus(
            query_session, executor_id, query_str, submitted_time_str, query_status));
    queries_session_map_.emplace(query_session, executor_per_query_map);
  }
  return queries_interrupt_flag_.emplace(query_session, false).second;
}

bool Executor::updateQuerySessionStatusWithLock(
    const QuerySessionId& query_session,
    const std::string& submitted_time_str,
    const QuerySessionStatus::QueryStatus updated_query_status,
    mapd_unique_lock<mapd_shared_mutex>& write_lock) {
  // an internal API that updates query session status
  if (query_session.empty()) {
    return false;
  }
  if (queries_session_map_.count(query_session)) {
    for (auto& query_status : queries_session_map_.at(query_session)) {
      auto target_submitted_t_str = query_status.second.getQuerySubmittedTime();
      // no time difference --> found the target query status
      if (submitted_time_str.compare(target_submitted_t_str) == 0) {
        auto prev_status = query_status.second.getQueryStatus();
        if (prev_status == updated_query_status) {
          return false;
        }
        query_status.second.setQueryStatus(updated_query_status);
        return true;
      }
    }
  }
  return false;
}

bool Executor::updateQuerySessionExecutorAssignment(
    const QuerySessionId& query_session,
    const std::string& submitted_time_str,
    const size_t executor_id,
    mapd_unique_lock<mapd_shared_mutex>& write_lock) {
  // update the executor id of the query session
  if (query_session.empty()) {
    return false;
  }
  if (queries_session_map_.count(query_session)) {
    auto storage = queries_session_map_.at(query_session);
    for (auto it = storage.begin(); it != storage.end(); it++) {
      auto target_submitted_t_str = it->second.getQuerySubmittedTime();
      // no time difference --> found the target query status
      if (submitted_time_str.compare(target_submitted_t_str) == 0) {
        queries_session_map_.at(query_session)
            .at(submitted_time_str)
            .setExecutorId(executor_id);
        return true;
      }
    }
  }
  return false;
}

bool Executor::removeFromQuerySessionList(
    const QuerySessionId& query_session,
    const std::string& submitted_time_str,
    mapd_unique_lock<mapd_shared_mutex>& write_lock) {
  if (query_session.empty()) {
    return false;
  }
  if (queries_session_map_.count(query_session)) {
    auto& storage = queries_session_map_.at(query_session);
    if (storage.size() > 1) {
      // in this case we only remove query executor info
      for (auto it = storage.begin(); it != storage.end(); it++) {
        auto target_submitted_t_str = it->second.getQuerySubmittedTime();
        // no time difference && have the same executor id--> found the target query
        if (it->second.getExecutorId() == executor_id_ &&
            submitted_time_str.compare(target_submitted_t_str) == 0) {
          storage.erase(it);
          return true;
        }
      }
    } else if (storage.size() == 1) {
      // here this session only has a single query executor
      // so we clear both executor info and its interrupt flag
      queries_session_map_.erase(query_session);
      queries_interrupt_flag_.erase(query_session);
      if (interrupted_.load()) {
        interrupted_.store(false);
        VLOG(1) << "RESET Executor " << this << " that had previously been interrupted";
      }
      return true;
    }
  }
  return false;
}

void Executor::setQuerySessionAsInterrupted(
    const QuerySessionId& query_session,
    mapd_unique_lock<mapd_shared_mutex>& write_lock) {
  if (query_session.empty()) {
    return;
  }
  if (queries_interrupt_flag_.find(query_session) != queries_interrupt_flag_.end()) {
    queries_interrupt_flag_[query_session] = true;
  }
}

void Executor::resetQuerySessionInterruptFlag(
    const std::string& query_session,
    mapd_unique_lock<mapd_shared_mutex>& write_lock) {
  if (query_session.empty()) {
    return;
  }
  queries_interrupt_flag_[query_session] = false;
}

bool Executor::checkIsQuerySessionInterrupted(
    const QuerySessionId& query_session,
    mapd_shared_lock<mapd_shared_mutex>& read_lock) {
  if (query_session.empty()) {
    return false;
  }
  auto flag_it = queries_interrupt_flag_.find(query_session);
  return !query_session.empty() && flag_it != queries_interrupt_flag_.end() &&
         flag_it->second;
}

bool Executor::checkIsRunningQuerySessionInterrupted() {
  mapd_shared_lock<mapd_shared_mutex> session_read_lock(executor_session_mutex_);
  return checkIsQuerySessionInterrupted(current_query_session_, session_read_lock);
}

bool Executor::checkIsQuerySessionEnrolled(
    const QuerySessionId& query_session,
    mapd_shared_lock<mapd_shared_mutex>& read_lock) {
  if (query_session.empty()) {
    return false;
  }
  return !query_session.empty() && queries_session_map_.count(query_session);
}

void Executor::enableRuntimeQueryInterrupt(
    const double runtime_query_check_freq,
    const unsigned pending_query_check_freq) const {
  // The only one scenario that we intentionally call this function is
  // to allow runtime query interrupt in QueryRunner for test cases.
  // Because test machine's default setting does not allow runtime query interrupt,
  // so we have to turn it on within test code if necessary.
  g_enable_runtime_query_interrupt = true;
  g_pending_query_interrupt_freq = pending_query_check_freq;
  g_running_query_interrupt_freq = runtime_query_check_freq;
  if (g_running_query_interrupt_freq) {
    g_running_query_interrupt_freq = 0.5;
  }
}

void Executor::addToCardinalityCache(const std::string& cache_key,
                                     const size_t cache_value) {
  if (g_use_estimator_result_cache) {
    mapd_unique_lock<mapd_shared_mutex> lock(recycler_mutex_);
    cardinality_cache_[cache_key] = cache_value;
    VLOG(1) << "Put estimated cardinality to the cache";
  }
}

Executor::CachedCardinality Executor::getCachedCardinality(const std::string& cache_key) {
  mapd_shared_lock<mapd_shared_mutex> lock(recycler_mutex_);
  if (g_use_estimator_result_cache &&
      cardinality_cache_.find(cache_key) != cardinality_cache_.end()) {
    VLOG(1) << "Reuse cached cardinality";
    return {true, cardinality_cache_[cache_key]};
  }
  return {false, -1};
}

std::vector<QuerySessionStatus> Executor::getQuerySessionInfo(
    const QuerySessionId& query_session,
    mapd_shared_lock<mapd_shared_mutex>& read_lock) {
  if (!queries_session_map_.empty() && queries_session_map_.count(query_session)) {
    auto& query_infos = queries_session_map_.at(query_session);
    std::vector<QuerySessionStatus> ret;
    for (auto& info : query_infos) {
      ret.push_back(QuerySessionStatus(query_session,
                                       info.second.getExecutorId(),
                                       info.second.getQueryStr(),
                                       info.second.getQuerySubmittedTime(),
                                       info.second.getQueryStatus()));
    }
    return ret;
  }
  return {};
}

std::map<int, std::shared_ptr<Executor>> Executor::executors_;

std::atomic_flag Executor::execute_spin_lock_ = ATOMIC_FLAG_INIT;
// current running query's session ID
std::string Executor::current_query_session_{""};
// running executor's id
size_t Executor::running_query_executor_id_{0};
// contain the interrupt flag's status per query session
InterruptFlagMap Executor::queries_interrupt_flag_;
// contain a list of queries per query session
QuerySessionMap Executor::queries_session_map_;
// session lock
mapd_shared_mutex Executor::executor_session_mutex_;

mapd_shared_mutex Executor::execute_mutex_;
mapd_shared_mutex Executor::executors_cache_mutex_;

std::mutex Executor::gpu_active_modules_mutex_;
uint32_t Executor::gpu_active_modules_device_mask_{0x0};
void* Executor::gpu_active_modules_[max_gpu_count];
std::atomic<bool> Executor::interrupted_{false};

std::mutex Executor::compilation_mutex_;
std::mutex Executor::kernel_mutex_;

mapd_shared_mutex Executor::recycler_mutex_;
std::unordered_map<std::string, size_t> Executor::cardinality_cache_;
