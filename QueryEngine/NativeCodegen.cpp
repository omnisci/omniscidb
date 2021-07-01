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

#include "CodeGenerator.h"
#include "Execute.h"
#include "ExtensionFunctionsWhitelist.h"
#include "GpuSharedMemoryUtils.h"
#include "LLVMFunctionAttributesUtil.h"
#include "OutputBufferInitialization.h"
#include "QueryTemplateGenerator.h"

#include "CudaMgr/CudaMgr.h"
#include "OSDependent/omnisci_path.h"
#include "Shared/InlineNullValues.h"
#include "Shared/MathUtils.h"
#include "StreamingTopN.h"

#ifdef HAVE_L0
#include "LLVMSPIRVLib/LLVMSPIRVLib.h"
#endif

#if LLVM_VERSION_MAJOR < 9
static_assert(false, "LLVM Version >= 9 is required.");
#endif

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/FormattedStream.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Instrumentation.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/InstSimplifyPass.h>
#include <llvm/Transforms/Utils.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>

#if LLVM_VERSION_MAJOR >= 11
#include <llvm/Support/Host.h>
#endif

float g_fraction_code_cache_to_evict = 0.2;

std::unique_ptr<llvm::Module> udf_gpu_module;
std::unique_ptr<llvm::Module> udf_cpu_module;
std::unique_ptr<llvm::Module> rt_udf_gpu_module;
std::unique_ptr<llvm::Module> rt_udf_cpu_module;

extern std::unique_ptr<llvm::Module> g_rt_module;

#ifdef HAVE_CUDA
extern std::unique_ptr<llvm::Module> g_rt_libdevice_module;
#endif

#ifdef ENABLE_GEOS
extern std::unique_ptr<llvm::Module> g_rt_geos_module;

#include <llvm/Support/DynamicLibrary.h>

#ifndef GEOS_LIBRARY_FILENAME
#error Configuration should include GEOS library file name
#endif
std::unique_ptr<std::string> g_libgeos_so_filename(
    new std::string(GEOS_LIBRARY_FILENAME));
static llvm::sys::DynamicLibrary geos_dynamic_library;
static std::mutex geos_init_mutex;

namespace {

void load_geos_dynamic_library() {
  std::lock_guard<std::mutex> guard(geos_init_mutex);

  if (!geos_dynamic_library.isValid()) {
    if (!g_libgeos_so_filename || g_libgeos_so_filename->empty()) {
      LOG(WARNING) << "Misconfigured GEOS library file name, trying 'libgeos_c.so'";
      g_libgeos_so_filename.reset(new std::string("libgeos_c.so"));
    }
    auto filename = *g_libgeos_so_filename;
    std::string error_message;
    geos_dynamic_library =
        llvm::sys::DynamicLibrary::getPermanentLibrary(filename.c_str(), &error_message);
    if (!geos_dynamic_library.isValid()) {
      LOG(ERROR) << "Failed to load GEOS library '" + filename + "'";
      std::string exception_message = "Failed to load GEOS library: " + error_message;
      throw std::runtime_error(exception_message.c_str());
    } else {
      LOG(INFO) << "Loaded GEOS library '" + filename + "'";
    }
  }
}

}  // namespace
#endif

namespace {

void throw_parseIR_error(const llvm::SMDiagnostic& parse_error,
                         std::string src = "",
                         const bool is_gpu = false) {
  std::string excname = (is_gpu ? "NVVM IR ParseError: " : "LLVM IR ParseError: ");
  llvm::raw_string_ostream ss(excname);
  parse_error.print(src.c_str(), ss, false, false);
  throw ParseIRError(ss.str());
}

/* SHOW_DEFINED(<llvm::Module instance>) prints the function names
   that are defined in the given LLVM Module instance.

   SHOW_FUNCTIONS(<llvm::Module instance>) prints the function names
   of all used functions in the given LLVM Module
   instance. Declarations are marked with `[decl]` as a name suffix.

   Useful for debugging.
*/

#define SHOW_DEFINED(MODULE)                                         \
  {                                                                  \
    std::cout << __func__ << "#" << __LINE__ << ": " #MODULE << " "; \
    ::show_defined(MODULE);                                          \
  }

#define SHOW_FUNCTIONS(MODULE)                                       \
  {                                                                  \
    std::cout << __func__ << "#" << __LINE__ << ": " #MODULE << " "; \
    ::show_functions(MODULE);                                        \
  }

template <typename T = void>
void show_defined(llvm::Module& module) {
  std::cout << "defines: ";
  for (auto& f : module.getFunctionList()) {
    if (!f.isDeclaration()) {
      std::cout << f.getName().str() << ", ";
    }
  }
  std::cout << std::endl;
}

template <typename T = void>
void show_defined(llvm::Module* module) {
  if (module == nullptr) {
    std::cout << "is null" << std::endl;
  } else {
    show_defined(*module);
  }
}

template <typename T = void>
void show_defined(std::unique_ptr<llvm::Module>& module) {
  show_defined(module.get());
}

/*
  scan_function_calls(module, defined, undefined, ignored) computes
  defined and undefined sets of function names:

  - defined functions are those that are defined in the given module

  - undefined functions are those that are called by defined functions
    but that are not defined in the given module

  - ignored functions are functions that may be undefined but will not
    be listed in the set of undefined functions.

   Useful for debugging.
*/
template <typename T = void>
void scan_function_calls(llvm::Function& F,
                         std::unordered_set<std::string>& defined,
                         std::unordered_set<std::string>& undefined,
                         const std::unordered_set<std::string>& ignored) {
  for (llvm::inst_iterator I = llvm::inst_begin(F), E = llvm::inst_end(F); I != E; ++I) {
    if (auto* CI = llvm::dyn_cast<llvm::CallInst>(&*I)) {
      auto* F2 = CI->getCalledFunction();
      if (F2 != nullptr) {
        auto F2name = F2->getName().str();
        if (F2->isDeclaration()) {
          if (F2name.rfind("__", 0) !=
                  0  // assume symbols with double underscore are defined
              && F2name.rfind("llvm.", 0) !=
                     0  // TODO: this may give false positive for NVVM intrinsics
              && ignored.find(F2name) == ignored.end()  // not in ignored list
          ) {
            undefined.emplace(F2name);
          }
        } else {
          if (defined.find(F2name) == defined.end()) {
            defined.emplace(F2name);
            scan_function_calls<T>(*F2, defined, undefined, ignored);
          }
        }
      }
    }
  }
}

template <typename T = void>
void scan_function_calls(llvm::Module& module,
                         std::unordered_set<std::string>& defined,
                         std::unordered_set<std::string>& undefined,
                         const std::unordered_set<std::string>& ignored) {
  for (auto& F : module) {
    if (!F.isDeclaration()) {
      scan_function_calls(F, defined, undefined, ignored);
    }
  }
}

template <typename T = void>
std::tuple<std::unordered_set<std::string>, std::unordered_set<std::string>>
scan_function_calls(llvm::Module& module,
                    const std::unordered_set<std::string>& ignored = {}) {
  std::unordered_set<std::string> defined, undefined;
  scan_function_calls(module, defined, undefined, ignored);
  return std::make_tuple(defined, undefined);
}

#if defined(HAVE_CUDA) || !defined(WITH_JIT_DEBUG)
void eliminate_dead_self_recursive_funcs(
    llvm::Module& M,
    const std::unordered_set<llvm::Function*>& live_funcs) {
  std::vector<llvm::Function*> dead_funcs;
  for (auto& F : M) {
    bool bAlive = false;
    if (live_funcs.count(&F)) {
      continue;
    }
    for (auto U : F.users()) {
      auto* C = llvm::dyn_cast<const llvm::CallInst>(U);
      if (!C || C->getParent()->getParent() != &F) {
        bAlive = true;
        break;
      }
    }
    if (!bAlive) {
      dead_funcs.push_back(&F);
    }
  }
  for (auto pFn : dead_funcs) {
    pFn->eraseFromParent();
  }
}

#ifdef HAVE_CUDA

// check if linking with libdevice is required
// libdevice functions have a __nv_* prefix
bool check_module_requires_libdevice(llvm::Module* module) {
  for (llvm::Function& F : *module) {
    if (F.hasName() && F.getName().startswith("__nv_")) {
      LOG(INFO) << "Module requires linking with libdevice: " << std::string(F.getName());
      return true;
    }
  }
  LOG(DEBUG1) << "module does not require linking against libdevice";
  return false;
}

// Adds the missing intrinsics declarations to the given module
void add_intrinsics_to_module(llvm::Module* module) {
  for (llvm::Function& F : *module) {
    for (llvm::Instruction& I : instructions(F)) {
      if (llvm::IntrinsicInst* ii = llvm::dyn_cast<llvm::IntrinsicInst>(&I)) {
        if (llvm::Intrinsic::isOverloaded(ii->getIntrinsicID())) {
          llvm::Type* Tys[] = {ii->getFunctionType()->getReturnType()};
          llvm::Function& decl_fn =
              *llvm::Intrinsic::getDeclaration(module, ii->getIntrinsicID(), Tys);
          ii->setCalledFunction(&decl_fn);
        } else {
          // inserts the declaration into the module if not present
          llvm::Intrinsic::getDeclaration(module, ii->getIntrinsicID());
        }
      }
    }
  }
}

#endif

void optimize_ir(llvm::Function* query_func,
                 llvm::Module* module,
                 llvm::legacy::PassManager& pass_manager,
                 const std::unordered_set<llvm::Function*>& live_funcs,
                 const CompilationOptions& co) {
  pass_manager.add(llvm::createAlwaysInlinerLegacyPass());
  pass_manager.add(llvm::createPromoteMemoryToRegisterPass());
  pass_manager.add(llvm::createInstSimplifyLegacyPass());
  pass_manager.add(llvm::createInstructionCombiningPass());
  pass_manager.add(llvm::createGlobalOptimizerPass());

  pass_manager.add(llvm::createLICMPass());
  if (co.opt_level == ExecutorOptLevel::LoopStrengthReduction) {
    pass_manager.add(llvm::createLoopStrengthReducePass());
  }
  pass_manager.run(*module);

  eliminate_dead_self_recursive_funcs(*module, live_funcs);
}
#endif

}  // namespace

ExecutionEngineWrapper::ExecutionEngineWrapper() {}

ExecutionEngineWrapper::ExecutionEngineWrapper(llvm::ExecutionEngine* execution_engine)
    : execution_engine_(execution_engine) {}

ExecutionEngineWrapper::ExecutionEngineWrapper(llvm::ExecutionEngine* execution_engine,
                                               const CompilationOptions& co)
    : execution_engine_(execution_engine) {
  if (execution_engine_) {
    if (co.register_intel_jit_listener) {
#ifdef ENABLE_INTEL_JIT_LISTENER
      intel_jit_listener_.reset(llvm::JITEventListener::createIntelJITEventListener());
      CHECK(intel_jit_listener_);
      execution_engine_->RegisterJITEventListener(intel_jit_listener_.get());
      LOG(INFO) << "Registered IntelJITEventListener";
#else
      LOG(WARNING) << "This build is not Intel JIT Listener enabled. Ignoring Intel JIT "
                      "listener configuration parameter.";
#endif  // ENABLE_INTEL_JIT_LISTENER
    }
  }
}

ExecutionEngineWrapper& ExecutionEngineWrapper::operator=(
    llvm::ExecutionEngine* execution_engine) {
  execution_engine_.reset(execution_engine);
  intel_jit_listener_ = nullptr;
  return *this;
}

void verify_function_ir(const llvm::Function* func) {
  std::stringstream err_ss;
  llvm::raw_os_ostream err_os(err_ss);
  err_os << "\n-----\n";
  if (llvm::verifyFunction(*func, &err_os)) {
    err_os << "\n-----\n";
    func->print(err_os, nullptr);
    err_os << "\n-----\n";
    LOG(FATAL) << err_ss.str();
  }
}

std::shared_ptr<CompilationContext> Executor::getCodeFromCache(const CodeCacheKey& key,
                                                               const CodeCache& cache) {
  auto it = cache.find(key);
  if (it != cache.cend()) {
    delete cgen_state_->module_;
    cgen_state_->module_ = it->second.second;
    return it->second.first;
  }
  return {};
}

void Executor::addCodeToCache(const CodeCacheKey& key,
                              std::shared_ptr<CompilationContext> compilation_context,
                              llvm::Module* module,
                              CodeCache& cache) {
  cache.put(key,
            std::make_pair<std::shared_ptr<CompilationContext>, decltype(module)>(
                std::move(compilation_context), std::move(module)));
}

namespace {

std::string assemblyForCPU(ExecutionEngineWrapper& execution_engine,
                           llvm::Module* module) {
  llvm::legacy::PassManager pass_manager;
  auto cpu_target_machine = execution_engine->getTargetMachine();
  CHECK(cpu_target_machine);
  llvm::SmallString<256> code_str;
  llvm::raw_svector_ostream os(code_str);
#if LLVM_VERSION_MAJOR >= 10
  cpu_target_machine->addPassesToEmitFile(
      pass_manager, os, nullptr, llvm::CGFT_AssemblyFile);
#else
  cpu_target_machine->addPassesToEmitFile(
      pass_manager, os, nullptr, llvm::TargetMachine::CGFT_AssemblyFile);
#endif
  pass_manager.run(*module);
  return "Assembly for the CPU:\n" + std::string(code_str.str()) + "\nEnd of assembly";
}

}  // namespace

ExecutionEngineWrapper CodeGenerator::generateNativeCPUCode(
    llvm::Function* func,
    const std::unordered_set<llvm::Function*>& live_funcs,
    const CompilationOptions& co) {
  auto module = func->getParent();
  // run optimizations
#ifndef WITH_JIT_DEBUG
  llvm::legacy::PassManager pass_manager;
  optimize_ir(func, module, pass_manager, live_funcs, co);
#endif  // WITH_JIT_DEBUG

  auto init_err = llvm::InitializeNativeTarget();
  CHECK(!init_err);

  llvm::InitializeAllTargetMCs();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  std::string err_str;
  std::unique_ptr<llvm::Module> owner(module);
  llvm::EngineBuilder eb(std::move(owner));
  eb.setErrorStr(&err_str);
  eb.setEngineKind(llvm::EngineKind::JIT);
  llvm::TargetOptions to;
  to.EnableFastISel = true;
  eb.setTargetOptions(to);
  if (co.opt_level == ExecutorOptLevel::ReductionJIT) {
    eb.setOptLevel(llvm::CodeGenOpt::None);
  }

#ifdef _WIN32
  // TODO: workaround for data layout mismatch crash for now
  auto target_machine = eb.selectTarget();
  CHECK(target_machine);
  module->setDataLayout(target_machine->createDataLayout());
#endif

  ExecutionEngineWrapper execution_engine(eb.create(), co);
  CHECK(execution_engine.get());
  LOG(ASM) << assemblyForCPU(execution_engine, module);

  execution_engine->finalizeObject();
  return execution_engine;
}

std::shared_ptr<CompilationContext> Executor::optimizeAndCodegenCPU(
    llvm::Function* query_func,
    llvm::Function* multifrag_query_func,
    const std::unordered_set<llvm::Function*>& live_funcs,
    const CompilationOptions& co) {
  auto module = multifrag_query_func->getParent();
  CodeCacheKey key{serialize_llvm_object(query_func),
                   serialize_llvm_object(cgen_state_->row_func_)};
  if (cgen_state_->filter_func_) {
    key.push_back(serialize_llvm_object(cgen_state_->filter_func_));
  }
  for (const auto helper : cgen_state_->helper_functions_) {
    key.push_back(serialize_llvm_object(helper));
  }
  auto cached_code = getCodeFromCache(key, cpu_code_cache_);
  if (cached_code) {
    return cached_code;
  }

  if (cgen_state_->needs_geos_) {
#ifdef ENABLE_GEOS
    load_geos_dynamic_library();

    // Read geos runtime module and bind GEOS API function references to GEOS library
    auto rt_geos_module_copy = llvm::CloneModule(
        *g_rt_geos_module.get(), cgen_state_->vmap_, [](const llvm::GlobalValue* gv) {
          auto func = llvm::dyn_cast<llvm::Function>(gv);
          if (!func) {
            return true;
          }
          return (func->getLinkage() == llvm::GlobalValue::LinkageTypes::PrivateLinkage ||
                  func->getLinkage() ==
                      llvm::GlobalValue::LinkageTypes::InternalLinkage ||
                  func->getLinkage() == llvm::GlobalValue::LinkageTypes::ExternalLinkage);
        });
    CodeGenerator::link_udf_module(rt_geos_module_copy,
                                   *module,
                                   cgen_state_.get(),
                                   llvm::Linker::Flags::LinkOnlyNeeded);
#else
    throw std::runtime_error("GEOS is disabled in this build");
#endif
  }

  auto execution_engine =
      CodeGenerator::generateNativeCPUCode(query_func, live_funcs, co);
  auto cpu_compilation_context =
      std::make_shared<CpuCompilationContext>(std::move(execution_engine));
  cpu_compilation_context->setFunctionPointer(multifrag_query_func);
  addCodeToCache(key, cpu_compilation_context, module, cpu_code_cache_);
  return cpu_compilation_context;
}

void CodeGenerator::link_udf_module(const std::unique_ptr<llvm::Module>& udf_module,
                                    llvm::Module& module,
                                    CgenState* cgen_state,
                                    llvm::Linker::Flags flags) {
  // throw a runtime error if the target module contains functions
  // with the same name as in module of UDF functions.
  for (auto& f : *udf_module.get()) {
    auto func = module.getFunction(f.getName());
    if (!(func == nullptr) && !f.isDeclaration() && flags == llvm::Linker::Flags::None) {
      LOG(ERROR) << "  Attempt to overwrite " << f.getName().str() << " in "
                 << module.getModuleIdentifier() << " from `"
                 << udf_module->getModuleIdentifier() << "`" << std::endl;
      throw std::runtime_error(
          "link_udf_module: *** attempt to overwrite a runtime function with a UDF "
          "function ***");
    } else {
      VLOG(1) << "  Adding " << f.getName().str() << " to "
              << module.getModuleIdentifier() << " from `"
              << udf_module->getModuleIdentifier() << "`" << std::endl;
    }
  }

  std::unique_ptr<llvm::Module> udf_module_copy;

  udf_module_copy = llvm::CloneModule(*udf_module.get(), cgen_state->vmap_);

  udf_module_copy->setDataLayout(module.getDataLayout());
  udf_module_copy->setTargetTriple(module.getTargetTriple());

  // Initialize linker with module for RuntimeFunctions.bc
  llvm::Linker ld(module);
  bool link_error = false;

  link_error = ld.linkInModule(std::move(udf_module_copy), flags);

  if (link_error) {
    throw std::runtime_error("link_udf_module: *** error linking module ***");
  }
}

namespace {

std::string cpp_to_llvm_name(const std::string& s) {
  if (s == "int8_t") {
    return "i8";
  }
  if (s == "int16_t") {
    return "i16";
  }
  if (s == "int32_t") {
    return "i32";
  }
  if (s == "int64_t") {
    return "i64";
  }
  CHECK(s == "float" || s == "double");
  return s;
}

std::string gen_array_any_all_sigs() {
  std::string result;
  for (const std::string any_or_all : {"any", "all"}) {
    for (const std::string elem_type :
         {"int8_t", "int16_t", "int32_t", "int64_t", "float", "double"}) {
      for (const std::string needle_type :
           {"int8_t", "int16_t", "int32_t", "int64_t", "float", "double"}) {
        for (const std::string op_name : {"eq", "ne", "lt", "le", "gt", "ge"}) {
          result += ("declare i1 @array_" + any_or_all + "_" + op_name + "_" + elem_type +
                     "_" + needle_type + "(i8*, i64, " + cpp_to_llvm_name(needle_type) +
                     ", " + cpp_to_llvm_name(elem_type) + ");\n");
        }
      }
    }
  }
  return result;
}

std::string gen_translate_null_key_sigs() {
  std::string result;
  for (const std::string key_type : {"int8_t", "int16_t", "int32_t", "int64_t"}) {
    const auto key_llvm_type = cpp_to_llvm_name(key_type);
    result += "declare i64 @translate_null_key_" + key_type + "(" + key_llvm_type + ", " +
              key_llvm_type + ", i64);\n";
  }
  return result;
}

const std::string cuda_rt_decls =
    R"(
declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @llvm.dbg.value(metadata, metadata, metadata)
declare void @llvm.lifetime.start(i64, i8* nocapture) nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) nounwind
declare i64 @get_thread_index();
declare i64 @get_block_index();
declare i32 @pos_start_impl(i32*);
declare i32 @group_buff_idx_impl();
declare i32 @pos_step_impl();
declare i8 @thread_warp_idx(i8);
declare i64* @init_shared_mem(i64*, i32);
declare i64* @init_shared_mem_nop(i64*, i32);
declare i64* @declare_dynamic_shared_memory();
declare void @write_back_nop(i64*, i64*, i32);
declare void @write_back_non_grouped_agg(i64*, i64*, i32);
declare void @init_group_by_buffer_gpu(i64*, i64*, i32, i32, i32, i1, i8);
declare i64* @get_group_value(i64*, i32, i64*, i32, i32, i32);
declare i64* @get_group_value_with_watchdog(i64*, i32, i64*, i32, i32, i32);
declare i32 @get_group_value_columnar_slot(i64*, i32, i64*, i32, i32);
declare i32 @get_group_value_columnar_slot_with_watchdog(i64*, i32, i64*, i32, i32);
declare i64* @get_group_value_fast(i64*, i64, i64, i64, i32);
declare i64* @get_group_value_fast_with_original_key(i64*, i64, i64, i64, i64, i32);
declare i32 @get_columnar_group_bin_offset(i64*, i64, i64, i64);
declare i64 @baseline_hash_join_idx_32(i8*, i8*, i64, i64);
declare i64 @baseline_hash_join_idx_64(i8*, i8*, i64, i64);
declare i64 @get_composite_key_index_32(i32*, i64, i32*, i64);
declare i64 @get_composite_key_index_64(i64*, i64, i64*, i64);
declare i64 @get_bucket_key_for_range_compressed(i8*, i64, double);
declare i64 @get_bucket_key_for_range_double(i8*, i64, double);
declare i32 @get_num_buckets_for_bounds(i8*, i32, double, double);
declare i64 @get_candidate_rows(i32*, i32, i8*, i32, double, double, i32, i64, i64*, i64, i64, i64);
declare i64 @agg_count_shared(i64*, i64);
declare i64 @agg_count_skip_val_shared(i64*, i64, i64);
declare i32 @agg_count_int32_shared(i32*, i32);
declare i32 @agg_count_int32_skip_val_shared(i32*, i32, i32);
declare i64 @agg_count_double_shared(i64*, double);
declare i64 @agg_count_double_skip_val_shared(i64*, double, double);
declare i32 @agg_count_float_shared(i32*, float);
declare i32 @agg_count_float_skip_val_shared(i32*, float, float);
declare i64 @agg_sum_shared(i64*, i64);
declare i64 @agg_sum_skip_val_shared(i64*, i64, i64);
declare i32 @agg_sum_int32_shared(i32*, i32);
declare i32 @agg_sum_int32_skip_val_shared(i32*, i32, i32);
declare void @agg_sum_double_shared(i64*, double);
declare void @agg_sum_double_skip_val_shared(i64*, double, double);
declare void @agg_sum_float_shared(i32*, float);
declare void @agg_sum_float_skip_val_shared(i32*, float, float);
declare void @agg_max_shared(i64*, i64);
declare void @agg_max_skip_val_shared(i64*, i64, i64);
declare void @agg_max_int32_shared(i32*, i32);
declare void @agg_max_int32_skip_val_shared(i32*, i32, i32);
declare void @agg_max_int16_shared(i16*, i16);
declare void @agg_max_int16_skip_val_shared(i16*, i16, i16);
declare void @agg_max_int8_shared(i8*, i8);
declare void @agg_max_int8_skip_val_shared(i8*, i8, i8);
declare void @agg_max_double_shared(i64*, double);
declare void @agg_max_double_skip_val_shared(i64*, double, double);
declare void @agg_max_float_shared(i32*, float);
declare void @agg_max_float_skip_val_shared(i32*, float, float);
declare void @agg_min_shared(i64*, i64);
declare void @agg_min_skip_val_shared(i64*, i64, i64);
declare void @agg_min_int32_shared(i32*, i32);
declare void @agg_min_int32_skip_val_shared(i32*, i32, i32);
declare void @agg_min_int16_shared(i16*, i16);
declare void @agg_min_int16_skip_val_shared(i16*, i16, i16);
declare void @agg_min_int8_shared(i8*, i8);
declare void @agg_min_int8_skip_val_shared(i8*, i8, i8);
declare void @agg_min_double_shared(i64*, double);
declare void @agg_min_double_skip_val_shared(i64*, double, double);
declare void @agg_min_float_shared(i32*, float);
declare void @agg_min_float_skip_val_shared(i32*, float, float);
declare void @agg_id_shared(i64*, i64);
declare void @agg_id_int32_shared(i32*, i32);
declare void @agg_id_int16_shared(i16*, i16);
declare void @agg_id_int8_shared(i8*, i8);
declare void @agg_id_double_shared(i64*, double);
declare void @agg_id_double_shared_slow(i64*, double*);
declare void @agg_id_float_shared(i32*, float);
declare i32 @checked_single_agg_id_shared(i64*, i64, i64);
declare i32 @checked_single_agg_id_double_shared(i64*, double, double);
declare i32 @checked_single_agg_id_double_shared_slow(i64*, double*, double);
declare i32 @checked_single_agg_id_float_shared(i32*, float, float);
declare i1 @slotEmptyKeyCAS(i64*, i64, i64);
declare i1 @slotEmptyKeyCAS_int32(i32*, i32, i32);
declare i1 @slotEmptyKeyCAS_int16(i16*, i16, i16);
declare i1 @slotEmptyKeyCAS_int8(i8*, i8, i8);
declare i64 @datetrunc_century(i64);
declare i64 @datetrunc_day(i64);
declare i64 @datetrunc_decade(i64);
declare i64 @datetrunc_hour(i64);
declare i64 @datetrunc_millennium(i64);
declare i64 @datetrunc_minute(i64);
declare i64 @datetrunc_month(i64);
declare i64 @datetrunc_quarter(i64);
declare i64 @datetrunc_quarterday(i64);
declare i64 @datetrunc_week_monday(i64);
declare i64 @datetrunc_week_sunday(i64);
declare i64 @datetrunc_week_saturday(i64);
declare i64 @datetrunc_year(i64);
declare i64 @extract_epoch(i64);
declare i64 @extract_dateepoch(i64);
declare i64 @extract_quarterday(i64);
declare i64 @extract_hour(i64);
declare i64 @extract_minute(i64);
declare i64 @extract_second(i64);
declare i64 @extract_millisecond(i64);
declare i64 @extract_microsecond(i64);
declare i64 @extract_nanosecond(i64);
declare i64 @extract_dow(i64);
declare i64 @extract_isodow(i64);
declare i64 @extract_day(i64);
declare i64 @extract_week_monday(i64);
declare i64 @extract_week_sunday(i64);
declare i64 @extract_week_saturday(i64);
declare i64 @extract_day_of_year(i64);
declare i64 @extract_month(i64);
declare i64 @extract_quarter(i64);
declare i64 @extract_year(i64);
declare i64 @DateTruncateHighPrecisionToDate(i64, i64);
declare i64 @DateTruncateHighPrecisionToDateNullable(i64, i64, i64);
declare i64 @DateDiff(i32, i64, i64);
declare i64 @DateDiffNullable(i32, i64, i64, i64);
declare i64 @DateDiffHighPrecision(i32, i64, i64, i32, i32);
declare i64 @DateDiffHighPrecisionNullable(i32, i64, i64, i32, i32, i64);
declare i64 @DateAdd(i32, i64, i64);
declare i64 @DateAddNullable(i32, i64, i64, i64);
declare i64 @DateAddHighPrecision(i32, i64, i64, i32);
declare i64 @DateAddHighPrecisionNullable(i32, i64, i64, i32, i64);
declare i64 @string_decode(i8*, i64);
declare i32 @array_size(i8*, i64, i32);
declare i32 @array_size_nullable(i8*, i64, i32, i32);
declare i32 @fast_fixlen_array_size(i8*, i32);
declare i1 @array_is_null(i8*, i64);
declare i1 @point_coord_array_is_null(i8*, i64);
declare i8* @array_buff(i8*, i64);
declare i8* @fast_fixlen_array_buff(i8*, i64);
declare i8 @array_at_int8_t(i8*, i64, i32);
declare i16 @array_at_int16_t(i8*, i64, i32);
declare i32 @array_at_int32_t(i8*, i64, i32);
declare i64 @array_at_int64_t(i8*, i64, i32);
declare float @array_at_float(i8*, i64, i32);
declare double @array_at_double(i8*, i64, i32);
declare i8 @varlen_array_at_int8_t(i8*, i64, i32);
declare i16 @varlen_array_at_int16_t(i8*, i64, i32);
declare i32 @varlen_array_at_int32_t(i8*, i64, i32);
declare i64 @varlen_array_at_int64_t(i8*, i64, i32);
declare float @varlen_array_at_float(i8*, i64, i32);
declare double @varlen_array_at_double(i8*, i64, i32);
declare i8 @varlen_notnull_array_at_int8_t(i8*, i64, i32);
declare i16 @varlen_notnull_array_at_int16_t(i8*, i64, i32);
declare i32 @varlen_notnull_array_at_int32_t(i8*, i64, i32);
declare i64 @varlen_notnull_array_at_int64_t(i8*, i64, i32);
declare float @varlen_notnull_array_at_float(i8*, i64, i32);
declare double @varlen_notnull_array_at_double(i8*, i64, i32);
declare i8 @array_at_int8_t_checked(i8*, i64, i64, i8);
declare i16 @array_at_int16_t_checked(i8*, i64, i64, i16);
declare i32 @array_at_int32_t_checked(i8*, i64, i64, i32);
declare i64 @array_at_int64_t_checked(i8*, i64, i64, i64);
declare float @array_at_float_checked(i8*, i64, i64, float);
declare double @array_at_double_checked(i8*, i64, i64, double);
declare i32 @char_length(i8*, i32);
declare i32 @char_length_nullable(i8*, i32, i32);
declare i32 @char_length_encoded(i8*, i32);
declare i32 @char_length_encoded_nullable(i8*, i32, i32);
declare i32 @key_for_string_encoded(i32);
declare i1 @sample_ratio(double, i64);
declare i1 @string_like(i8*, i32, i8*, i32, i8);
declare i1 @string_ilike(i8*, i32, i8*, i32, i8);
declare i8 @string_like_nullable(i8*, i32, i8*, i32, i8, i8);
declare i8 @string_ilike_nullable(i8*, i32, i8*, i32, i8, i8);
declare i1 @string_like_simple(i8*, i32, i8*, i32);
declare i1 @string_ilike_simple(i8*, i32, i8*, i32);
declare i8 @string_like_simple_nullable(i8*, i32, i8*, i32, i8);
declare i8 @string_ilike_simple_nullable(i8*, i32, i8*, i32, i8);
declare i1 @string_lt(i8*, i32, i8*, i32);
declare i1 @string_le(i8*, i32, i8*, i32);
declare i1 @string_gt(i8*, i32, i8*, i32);
declare i1 @string_ge(i8*, i32, i8*, i32);
declare i1 @string_eq(i8*, i32, i8*, i32);
declare i1 @string_ne(i8*, i32, i8*, i32);
declare i8 @string_lt_nullable(i8*, i32, i8*, i32, i8);
declare i8 @string_le_nullable(i8*, i32, i8*, i32, i8);
declare i8 @string_gt_nullable(i8*, i32, i8*, i32, i8);
declare i8 @string_ge_nullable(i8*, i32, i8*, i32, i8);
declare i8 @string_eq_nullable(i8*, i32, i8*, i32, i8);
declare i8 @string_ne_nullable(i8*, i32, i8*, i32, i8);
declare i1 @regexp_like(i8*, i32, i8*, i32, i8);
declare i8 @regexp_like_nullable(i8*, i32, i8*, i32, i8, i8);
declare void @linear_probabilistic_count(i8*, i32, i8*, i32);
declare void @agg_count_distinct_bitmap_gpu(i64*, i64, i64, i64, i64, i64, i64);
declare void @agg_count_distinct_bitmap_skip_val_gpu(i64*, i64, i64, i64, i64, i64, i64, i64);
declare void @agg_approximate_count_distinct_gpu(i64*, i64, i32, i64, i64);
declare void @record_error_code(i32, i32*);
declare i32 @get_error_code(i32*);
declare i1 @dynamic_watchdog();
declare i1 @check_interrupt();
declare void @force_sync();
declare void @sync_warp();
declare void @sync_warp_protected(i64, i64);
declare void @sync_threadblock();
declare i64* @get_bin_from_k_heap_int32_t(i64*, i32, i32, i32, i1, i1, i1, i32, i32);
declare i64* @get_bin_from_k_heap_int64_t(i64*, i32, i32, i32, i1, i1, i1, i64, i64);
declare i64* @get_bin_from_k_heap_float(i64*, i32, i32, i32, i1, i1, i1, float, float);
declare i64* @get_bin_from_k_heap_double(i64*, i32, i32, i32, i1, i1, i1, double, double);
declare double @decompress_x_coord_geoint(i32);
declare double @decompress_y_coord_geoint(i32);
)" + gen_array_any_all_sigs() +
    gen_translate_null_key_sigs();

#ifdef HAVE_CUDA
std::string extension_function_decls(const std::unordered_set<std::string>& udf_decls) {
  const auto decls =
      ExtensionFunctionsWhitelist::getLLVMDeclarations(udf_decls, /*is_gpu=*/true);
  return boost::algorithm::join(decls, "\n");
}

void legalize_nvvm_ir(llvm::Function* query_func) {
  // optimizations might add attributes to the function
  // and NVPTX doesn't understand all of them; play it
  // safe and clear all attributes
  clear_function_attributes(query_func);
  verify_function_ir(query_func);

  std::vector<llvm::Instruction*> stackrestore_intrinsics;
  std::vector<llvm::Instruction*> stacksave_intrinsics;
  std::vector<llvm::Instruction*> lifetime;
  for (auto& BB : *query_func) {
    for (llvm::Instruction& I : BB) {
      if (const llvm::IntrinsicInst* II = llvm::dyn_cast<llvm::IntrinsicInst>(&I)) {
        if (II->getIntrinsicID() == llvm::Intrinsic::stacksave) {
          stacksave_intrinsics.push_back(&I);
        } else if (II->getIntrinsicID() == llvm::Intrinsic::stackrestore) {
          stackrestore_intrinsics.push_back(&I);
        } else if (II->getIntrinsicID() == llvm::Intrinsic::lifetime_start ||
                   II->getIntrinsicID() == llvm::Intrinsic::lifetime_end) {
          lifetime.push_back(&I);
        }
      }
    }
  }

  // stacksave and stackrestore intrinsics appear together, and
  // stackrestore uses stacksaved result as its argument
  // so it should be removed first.
  for (auto& II : stackrestore_intrinsics) {
    II->eraseFromParent();
  }
  for (auto& II : stacksave_intrinsics) {
    II->eraseFromParent();
  }
  // Remove lifetime intrinsics as well. NVPTX don't like them
  for (auto& II : lifetime) {
    II->eraseFromParent();
  }
}
#endif  // HAVE_CUDA

}  // namespace

llvm::StringRef get_gpu_target_triple_string() {
  return llvm::StringRef("nvptx64-nvidia-cuda");
}

llvm::StringRef get_l0_target_triple_string() {
  return llvm::StringRef("spir-unknown-unknown");
}

llvm::StringRef get_gpu_data_layout() {
  return llvm::StringRef(
      "e-p:64:64:64-i1:8:8-i8:8:8-"
      "i16:16:16-i32:32:32-i64:64:64-"
      "f32:32:32-f64:64:64-v16:16:16-"
      "v32:32:32-v64:64:64-v128:128:128-n16:32:64");
}

std::map<std::string, std::string> get_device_parameters(bool cpu_only) {
  std::map<std::string, std::string> result;

  result.insert(std::make_pair("cpu_name", llvm::sys::getHostCPUName()));
  result.insert(std::make_pair("cpu_triple", llvm::sys::getProcessTriple()));
  result.insert(
      std::make_pair("cpu_cores", std::to_string(llvm::sys::getHostNumPhysicalCores())));
  result.insert(std::make_pair("cpu_threads", std::to_string(cpu_threads())));

  // https://en.cppreference.com/w/cpp/language/types
  std::string sizeof_types;
  sizeof_types += "bool:" + std::to_string(sizeof(bool)) + ";";
  sizeof_types += "size_t:" + std::to_string(sizeof(size_t)) + ";";
  sizeof_types += "ssize_t:" + std::to_string(sizeof(ssize_t)) + ";";
  sizeof_types += "char:" + std::to_string(sizeof(char)) + ";";
  sizeof_types += "uchar:" + std::to_string(sizeof(unsigned char)) + ";";
  sizeof_types += "short:" + std::to_string(sizeof(short)) + ";";
  sizeof_types += "ushort:" + std::to_string(sizeof(unsigned short int)) + ";";
  sizeof_types += "int:" + std::to_string(sizeof(int)) + ";";
  sizeof_types += "uint:" + std::to_string(sizeof(unsigned int)) + ";";
  sizeof_types += "long:" + std::to_string(sizeof(long int)) + ";";
  sizeof_types += "ulong:" + std::to_string(sizeof(unsigned long int)) + ";";
  sizeof_types += "longlong:" + std::to_string(sizeof(long long int)) + ";";
  sizeof_types += "ulonglong:" + std::to_string(sizeof(unsigned long long int)) + ";";
  sizeof_types += "float:" + std::to_string(sizeof(float)) + ";";
  sizeof_types += "double:" + std::to_string(sizeof(double)) + ";";
  sizeof_types += "longdouble:" + std::to_string(sizeof(long double)) + ";";
  sizeof_types += "voidptr:" + std::to_string(sizeof(void*)) + ";";

  result.insert(std::make_pair("type_sizeof", sizeof_types));

  std::string null_values;
  null_values += "boolean1:" + std::to_string(serialized_null_value<bool>()) + ";";
  null_values += "boolean8:" + std::to_string(serialized_null_value<int8_t>()) + ";";
  null_values += "int8:" + std::to_string(serialized_null_value<int8_t>()) + ";";
  null_values += "int16:" + std::to_string(serialized_null_value<int16_t>()) + ";";
  null_values += "int32:" + std::to_string(serialized_null_value<int32_t>()) + ";";
  null_values += "int64:" + std::to_string(serialized_null_value<int64_t>()) + ";";
  null_values += "uint8:" + std::to_string(serialized_null_value<uint8_t>()) + ";";
  null_values += "uint16:" + std::to_string(serialized_null_value<uint16_t>()) + ";";
  null_values += "uint32:" + std::to_string(serialized_null_value<uint32_t>()) + ";";
  null_values += "uint64:" + std::to_string(serialized_null_value<uint64_t>()) + ";";
  null_values += "float32:" + std::to_string(serialized_null_value<float>()) + ";";
  null_values += "float64:" + std::to_string(serialized_null_value<double>()) + ";";
  null_values +=
      "Array<boolean8>:" + std::to_string(serialized_null_value<int8_t, true>()) + ";";
  null_values +=
      "Array<int8>:" + std::to_string(serialized_null_value<int8_t, true>()) + ";";
  null_values +=
      "Array<int16>:" + std::to_string(serialized_null_value<int16_t, true>()) + ";";
  null_values +=
      "Array<int32>:" + std::to_string(serialized_null_value<int32_t, true>()) + ";";
  null_values +=
      "Array<int64>:" + std::to_string(serialized_null_value<int64_t, true>()) + ";";
  null_values +=
      "Array<float32>:" + std::to_string(serialized_null_value<float, true>()) + ";";
  null_values +=
      "Array<float64>:" + std::to_string(serialized_null_value<double, true>()) + ";";

  result.insert(std::make_pair("null_values", null_values));

  llvm::StringMap<bool> cpu_features;
  if (llvm::sys::getHostCPUFeatures(cpu_features)) {
    std::string features_str = "";
    for (auto it = cpu_features.begin(); it != cpu_features.end(); ++it) {
      features_str += (it->getValue() ? " +" : " -");
      features_str += it->getKey().str();
    }
    result.insert(std::make_pair("cpu_features", features_str));
  }

  result.insert(std::make_pair("llvm_version",
                               std::to_string(LLVM_VERSION_MAJOR) + "." +
                                   std::to_string(LLVM_VERSION_MINOR) + "." +
                                   std::to_string(LLVM_VERSION_PATCH)));

#ifdef HAVE_CUDA
  if (!cpu_only) {
    int device_count = 0;
    checkCudaErrors(cuDeviceGetCount(&device_count));
    if (device_count) {
      CUdevice device{};
      char device_name[256];
      int major = 0, minor = 0;
      int driver_version;
      checkCudaErrors(cuDeviceGet(&device, 0));  // assuming homogeneous multi-GPU system
      checkCudaErrors(cuDeviceGetName(device_name, 256, device));
      checkCudaErrors(cuDeviceGetAttribute(
          &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
      checkCudaErrors(cuDeviceGetAttribute(
          &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
      checkCudaErrors(cuDriverGetVersion(&driver_version));

      result.insert(std::make_pair("gpu_name", device_name));
      result.insert(std::make_pair("gpu_count", std::to_string(device_count)));
      result.insert(std::make_pair("gpu_compute_capability",
                                   std::to_string(major) + "." + std::to_string(minor)));
      result.insert(std::make_pair("gpu_triple", get_gpu_target_triple_string()));
      result.insert(std::make_pair("gpu_datalayout", get_gpu_data_layout()));
      result.insert(std::make_pair("gpu_driver",
                                   "CUDA " + std::to_string(driver_version / 1000) + "." +
                                       std::to_string((driver_version % 1000) / 10)));
    }
  }
#endif

  return result;
}

std::shared_ptr<GpuCompilationContext> CodeGenerator::generateNativeGPUCode(
    llvm::Function* func,
    llvm::Function* wrapper_func,
    const std::unordered_set<llvm::Function*>& live_funcs,
    const CompilationOptions& co,
    const GPUTarget& gpu_target) {
#ifdef HAVE_CUDA
  auto module = func->getParent();
  /*
    `func` is one of the following generated functions:
    - `call_table_function(i8** %input_col_buffers, i64*
      %input_row_count, i64** %output_buffers, i64* %output_row_count)`
      that wraps the user-defined table function.
    - `multifrag_query`
    - `multifrag_query_hoisted_literals`
    - ...

    `wrapper_func` is table_func_kernel(i32*, i8**, i64*, i64**,
    i64*) that wraps `call_table_function`.

    `module` is from `build/QueryEngine/RuntimeFunctions.bc` and it
    contains `func` and `wrapper_func`.  `module` should also contain
    the definitions of user-defined table functions.

    `live_funcs` contains table_func_kernel and call_table_function

    `gpu_target.cgen_state->module_` appears to be the same as `module`
   */
  CHECK(gpu_target.cgen_state->module_ == module);
  module->setDataLayout(
      "e-p:64:64:64-i1:8:8-i8:8:8-"
      "i16:16:16-i32:32:32-i64:64:64-"
      "f32:32:32-f64:64:64-v16:16:16-"
      "v32:32:32-v64:64:64-v128:128:128-n16:32:64");
  module->setTargetTriple("nvptx64-nvidia-cuda");
  CHECK(gpu_target.nvptx_target_machine);
  auto pass_manager_builder = llvm::PassManagerBuilder();

  pass_manager_builder.OptLevel = 0;
  llvm::legacy::PassManager module_pass_manager;
  pass_manager_builder.populateModulePassManager(module_pass_manager);

  bool requires_libdevice = check_module_requires_libdevice(module);

  if (requires_libdevice) {
    // add nvvm reflect pass replacing any NVVM conditionals with constants
    gpu_target.nvptx_target_machine->adjustPassManager(pass_manager_builder);
    llvm::legacy::FunctionPassManager FPM(module);
    pass_manager_builder.populateFunctionPassManager(FPM);

    // Run the NVVMReflectPass here rather than inside optimize_ir
    FPM.doInitialization();
    for (auto& F : *module) {
      FPM.run(F);
    }
    FPM.doFinalization();
  }

  // run optimizations
  optimize_ir(func, module, module_pass_manager, live_funcs, co);
  legalize_nvvm_ir(func);

  std::stringstream ss;
  llvm::raw_os_ostream os(ss);

  llvm::LLVMContext& ctx = module->getContext();
  // Get "nvvm.annotations" metadata node
  llvm::NamedMDNode* md = module->getOrInsertNamedMetadata("nvvm.annotations");

  llvm::Metadata* md_vals[] = {llvm::ConstantAsMetadata::get(wrapper_func),
                               llvm::MDString::get(ctx, "kernel"),
                               llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                                   llvm::Type::getInt32Ty(ctx), 1))};

  // Append metadata to nvvm.annotations
  md->addOperand(llvm::MDNode::get(ctx, md_vals));

  std::unordered_set<llvm::Function*> roots{wrapper_func, func};
  if (gpu_target.row_func_not_inlined) {
    clear_function_attributes(gpu_target.cgen_state->row_func_);
    roots.insert(gpu_target.cgen_state->row_func_);
    if (gpu_target.cgen_state->filter_func_) {
      roots.insert(gpu_target.cgen_state->filter_func_);
    }
  }

  // prevent helper functions from being removed
  for (auto f : gpu_target.cgen_state->helper_functions_) {
    roots.insert(f);
  }

  if (requires_libdevice) {
    for (llvm::Function& F : *module) {
      // Some libdevice functions calls another functions that starts with "__internal_"
      // prefix.
      // __internal_trig_reduction_slowpathd
      // __internal_accurate_pow
      // __internal_lgamma_pos
      // Those functions have a "noinline" attribute which prevents the optimizer from
      // inlining them into the body of @query_func
      if (F.hasName() && F.getName().startswith("__internal") && !F.isDeclaration()) {
        roots.insert(&F);
      }
      legalize_nvvm_ir(&F);
    }
  }

  // Prevent the udf function(s) from being removed the way the runtime functions are
  std::unordered_set<std::string> udf_declarations;
  if (is_udf_module_present()) {
    for (auto& f : udf_gpu_module->getFunctionList()) {
      llvm::Function* udf_function = module->getFunction(f.getName());

      if (udf_function) {
        legalize_nvvm_ir(udf_function);
        roots.insert(udf_function);

        // If we have a udf that declares a external function
        // note it so we can avoid duplicate declarations
        if (f.isDeclaration()) {
          udf_declarations.insert(f.getName().str());
        }
      }
    }
  }

  if (is_rt_udf_module_present()) {
    for (auto& f : rt_udf_gpu_module->getFunctionList()) {
      llvm::Function* udf_function = module->getFunction(f.getName());
      if (udf_function) {
        legalize_nvvm_ir(udf_function);
        roots.insert(udf_function);

        // If we have a udf that declares a external function
        // note it so we can avoid duplicate declarations
        if (f.isDeclaration()) {
          udf_declarations.insert(f.getName().str());
        }
      }
    }
  }

  std::vector<llvm::Function*> rt_funcs;
  for (auto& Fn : *module) {
    if (roots.count(&Fn)) {
      continue;
    }
    rt_funcs.push_back(&Fn);
  }
  for (auto& pFn : rt_funcs) {
    pFn->removeFromParent();
  }

  if (requires_libdevice) {
    add_intrinsics_to_module(module);
  }

  module->print(os, nullptr);
  os.flush();

  for (auto& pFn : rt_funcs) {
    module->getFunctionList().push_back(pFn);
  }
  module->eraseNamedMetadata(md);

  auto cuda_llir = ss.str() + cuda_rt_decls + extension_function_decls(udf_declarations);
  std::string ptx;
  try {
    ptx = generatePTX(
        cuda_llir, gpu_target.nvptx_target_machine, gpu_target.cgen_state->context_);
  } catch (ParseIRError& e) {
    LOG(WARNING) << "Failed to generate PTX: " << e.what()
                 << ". Switching to CPU execution target.";
    throw QueryMustRunOnCpu();
  }
  LOG(PTX) << "PTX for the GPU:\n" << ptx << "\nEnd of PTX";

  auto cubin_result = ptx_to_cubin(ptx, gpu_target.block_size, gpu_target.cuda_mgr);
  auto& option_keys = cubin_result.option_keys;
  auto& option_values = cubin_result.option_values;
  auto cubin = cubin_result.cubin;
  auto link_state = cubin_result.link_state;
  const auto num_options = option_keys.size();

  auto func_name = wrapper_func->getName().str();
  auto gpu_compilation_context = std::make_shared<GpuCompilationContext>();
  for (int device_id = 0; device_id < gpu_target.cuda_mgr->getDeviceCount();
       ++device_id) {
    gpu_compilation_context->addDeviceCode(
        std::make_unique<GpuDeviceCompilationContext>(cubin,
                                                      func_name,
                                                      device_id,
                                                      gpu_target.cuda_mgr,
                                                      num_options,
                                                      &option_keys[0],
                                                      &option_values[0]));
  }

  checkCudaErrors(cuLinkDestroy(link_state));
  return gpu_compilation_context;
#else
  return {};
#endif
}

std::shared_ptr<L0CompilationContext> CodeGenerator::generateNativeL0Code(
    llvm::Function* func,
    llvm::Function* wrapper_func,
    const std::unordered_set<llvm::Function*>& live_funcs,
    const CompilationOptions& co,
    const l0::L0Manager* l0_mgr) {
#ifdef HAVE_L0
  auto module = func->getParent();

  auto pass_manager_builder = llvm::PassManagerBuilder();
  llvm::legacy::PassManager PM;
  pass_manager_builder.populateModulePassManager(PM);
  optimize_ir(func, module, PM, live_funcs, co);

  std::ostringstream ss;
  std::string err;

  module->setTargetTriple("spir64-unknown-unknown");

  llvm::LLVMContext& ctx = module->getContext();
  // set metadata -- pretend we're opencl (see
  // https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/master/docs/SPIRVRepresentationInLLVM.rst#spir-v-instructions-mapped-to-llvm-metadata)
  llvm::Metadata* spirv_src_ops[] = {
      llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 3 /*OpenCL_C*/)),
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx),
                                                           102000 /*OpenCL ver 1.2*/))};
  llvm::NamedMDNode* spirv_src = module->getOrInsertNamedMetadata("spirv.Source");
  spirv_src->addOperand(llvm::MDNode::get(ctx, spirv_src_ops));

  SPIRV::TranslatorOpts opts;
  opts.enableAllExtensions();
  opts.setDesiredBIsRepresentation(SPIRV::BIsRepresentation::OpenCL12);
  opts.setDebugInfoEIS(SPIRV::DebugInfoEIS::OpenCL_DebugInfo_100);

  std::unordered_set<llvm::Function*> roots{wrapper_func, func};

  // todo: add helper funcs
  // todo: add udf funcs

  std::vector<llvm::Function*> rt_funcs;
  for (auto& Fn : *module) {
    if (!roots.count(&Fn)) {
      rt_funcs.push_back(&Fn);
    }
  }

  for (auto& pFn : rt_funcs) {
    // pFn->removeFromParent();
    pFn->eraseFromParent();
  }

  // todo: enable when runtime functions are supported
  // for (auto& pFn : rt_funcs) {
  //   module->getFunctionList().push_back(pFn);
  // }

  for (auto& Fn : *module) {
    Fn.setCallingConv(llvm::CallingConv::SPIR_FUNC);
  }

  llvm::errs() << "func: " << (func ? func->getName() : "null") << "\n";
  llvm::errs() << "wrapper func: " << (wrapper_func ? wrapper_func->getName() : "null")
               << "\n";
  CHECK(wrapper_func);

  wrapper_func->setCallingConv(llvm::CallingConv::SPIR_KERNEL);

  std::error_code EC;
  llvm::raw_fd_ostream OS("ir.bc", EC, llvm::sys::fs::F_None);
  llvm::WriteBitcodeToFile(*module, OS);
  OS.flush();
  llvm::errs() << EC.category().name() << '\n';

  auto success = writeSpirv(module, opts, ss, err);
  if (!success) {
    llvm::errs() << "Spirv translation failed with error: " << err << "\n";
  } else {
    llvm::errs() << "Spirv tranlsation success.\n";
  }
  CHECK(success);

  const auto func_name = wrapper_func->getName().str();
  L0BinResult bin_result;
  try {
    bin_result = spv_to_bin(ss.str(), func_name, 1 /*todo block size*/, l0_mgr);
  } catch (l0::L0Exception& e) {
    llvm::errs() << e.what() << "\n";
    return {};
  }

  auto compilation_ctx = std::make_shared<L0CompilationContext>();
  auto device_compilation_ctx = std::make_unique<L0DeviceCompilationContext>(
      bin_result.kernel, bin_result.module, l0_mgr, 0, 1);
  compilation_ctx->addDeviceCode(move(device_compilation_ctx));
  return compilation_ctx;
#else
  return {};
#endif  // HAVE_L0
}

std::shared_ptr<CompilationContext> Executor::optimizeAndCodegenL0(
    llvm::Function* query_func,
    llvm::Function* multifrag_query_func,
    std::unordered_set<llvm::Function*>& live_funcs,
    const bool no_inline,
    const l0::L0Manager* l0_mgr,
    const CompilationOptions& co) {
#ifdef HAVE_L0
  auto module = multifrag_query_func->getParent();
  CHECK(l0_mgr);
  // todo: cache

  std::shared_ptr<L0CompilationContext> compilation_context;
  try {
    compilation_context = CodeGenerator::generateNativeL0Code(
        query_func, multifrag_query_func, live_funcs, co, l0_mgr);
  } catch (l0::L0Exception& e) {
    LOG(WARNING) << "Caught L0 exception: " << e.what() << "\n";
    throw;
  }
  return compilation_context;
#else
  return {};
#endif  // HAVE_L0
}

std::shared_ptr<CompilationContext> Executor::optimizeAndCodegenGPU(
    llvm::Function* query_func,
    llvm::Function* multifrag_query_func,
    std::unordered_set<llvm::Function*>& live_funcs,
    const bool no_inline,
    const CudaMgr_Namespace::CudaMgr* cuda_mgr,
    const CompilationOptions& co) {
#ifdef HAVE_CUDA
  auto module = multifrag_query_func->getParent();

  CHECK(cuda_mgr);
  CodeCacheKey key{serialize_llvm_object(query_func),
                   serialize_llvm_object(cgen_state_->row_func_)};
  if (cgen_state_->filter_func_) {
    key.push_back(serialize_llvm_object(cgen_state_->filter_func_));
  }
  for (const auto helper : cgen_state_->helper_functions_) {
    key.push_back(serialize_llvm_object(helper));
  }
  auto cached_code = getCodeFromCache(key, gpu_code_cache_);
  if (cached_code) {
    return cached_code;
  }

  bool row_func_not_inlined = false;
  if (no_inline) {
    for (auto it = llvm::inst_begin(cgen_state_->row_func_),
              e = llvm::inst_end(cgen_state_->row_func_);
         it != e;
         ++it) {
      if (llvm::isa<llvm::CallInst>(*it)) {
        auto& get_gv_call = llvm::cast<llvm::CallInst>(*it);
        if (get_gv_call.getCalledFunction()->getName() == "array_size" ||
            get_gv_call.getCalledFunction()->getName() == "linear_probabilistic_count") {
          mark_function_never_inline(cgen_state_->row_func_);
          row_func_not_inlined = true;
          break;
        }
      }
    }
  }

  initializeNVPTXBackend();
  CodeGenerator::GPUTarget gpu_target{nvptx_target_machine_.get(),
                                      cuda_mgr,
                                      blockSize(),
                                      cgen_state_.get(),
                                      row_func_not_inlined};
  std::shared_ptr<GpuCompilationContext> compilation_context;

  if (check_module_requires_libdevice(module)) {
    if (g_rt_libdevice_module == nullptr) {
      // raise error
      throw std::runtime_error(
          "libdevice library is not available but required by the UDF module");
    }

    // Bind libdevice it to the current module
    CodeGenerator::link_udf_module(g_rt_libdevice_module,
                                   *module,
                                   cgen_state_.get(),
                                   llvm::Linker::Flags::OverrideFromSrc);

    // activate nvvm-reflect-ftz flag on the module
    module->addModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz", (int)1);
    for (llvm::Function& fn : *module) {
      fn.addFnAttr("nvptx-f32ftz", "true");
    }
  }

  try {
    compilation_context = CodeGenerator::generateNativeGPUCode(
        query_func, multifrag_query_func, live_funcs, co, gpu_target);
    addCodeToCache(key, compilation_context, module, gpu_code_cache_);
  } catch (CudaMgr_Namespace::CudaErrorException& cuda_error) {
    if (cuda_error.getStatus() == CUDA_ERROR_OUT_OF_MEMORY) {
      // Thrown if memory not able to be allocated on gpu
      // Retry once after evicting portion of code cache
      LOG(WARNING) << "Failed to allocate GPU memory for generated code. Evicting "
                   << g_fraction_code_cache_to_evict * 100.
                   << "% of GPU code cache and re-trying.";
      gpu_code_cache_.evictFractionEntries(g_fraction_code_cache_to_evict);
      compilation_context = CodeGenerator::generateNativeGPUCode(
          query_func, multifrag_query_func, live_funcs, co, gpu_target);
      addCodeToCache(key, compilation_context, module, gpu_code_cache_);
    } else {
      throw;
    }
  }
  CHECK(compilation_context);
  return compilation_context;
#else
  return nullptr;
#endif
}

std::string CodeGenerator::generatePTX(const std::string& cuda_llir,
                                       llvm::TargetMachine* nvptx_target_machine,
                                       llvm::LLVMContext& context) {
  auto mem_buff = llvm::MemoryBuffer::getMemBuffer(cuda_llir, "", false);

  llvm::SMDiagnostic parse_error;

  auto module = llvm::parseIR(mem_buff->getMemBufferRef(), parse_error, context);
  if (!module) {
    LOG(IR) << "CodeGenerator::generatePTX:NVVM IR:\n" << cuda_llir << "\nEnd of NNVM IR";
    throw_parseIR_error(parse_error, "generatePTX", /* is_gpu= */ true);
  }

  llvm::SmallString<256> code_str;
  llvm::raw_svector_ostream formatted_os(code_str);
  CHECK(nvptx_target_machine);
  {
    llvm::legacy::PassManager ptxgen_pm;
    module->setDataLayout(nvptx_target_machine->createDataLayout());

#if LLVM_VERSION_MAJOR >= 10
    nvptx_target_machine->addPassesToEmitFile(
        ptxgen_pm, formatted_os, nullptr, llvm::CGFT_AssemblyFile);
#else
    nvptx_target_machine->addPassesToEmitFile(
        ptxgen_pm, formatted_os, nullptr, llvm::TargetMachine::CGFT_AssemblyFile);
#endif
    ptxgen_pm.run(*module);
  }

#if LLVM_VERSION_MAJOR >= 11
  return std::string(code_str);
#else
  return code_str.str();
#endif
}

std::unique_ptr<llvm::TargetMachine> CodeGenerator::initializeNVPTXBackend(
    const CudaMgr_Namespace::NvidiaDeviceArch arch) {
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  std::string err;
  auto target = llvm::TargetRegistry::lookupTarget("nvptx64", err);
  if (!target) {
    LOG(FATAL) << err;
  }
  return std::unique_ptr<llvm::TargetMachine>(
      target->createTargetMachine("nvptx64-nvidia-cuda",
                                  CudaMgr_Namespace::CudaMgr::deviceArchToSM(arch),
                                  "",
                                  llvm::TargetOptions(),
                                  llvm::Reloc::Static));
}

std::string Executor::generatePTX(const std::string& cuda_llir) const {
  return CodeGenerator::generatePTX(
      cuda_llir, nvptx_target_machine_.get(), cgen_state_->context_);
}

void Executor::initializeNVPTXBackend() const {
  if (nvptx_target_machine_) {
    return;
  }
  const auto cuda_mgr = catalog_->getDataMgr().getCudaMgr();
  LOG_IF(FATAL, cuda_mgr == nullptr) << "No CudaMgr instantiated, unable to check device "
                                        "architecture or generate code for nvidia GPUs.";
  const auto arch = cuda_mgr->getDeviceArch();
  nvptx_target_machine_ = CodeGenerator::initializeNVPTXBackend(arch);
}

// A small number of runtime functions don't get through CgenState::emitCall. List them
// explicitly here and always clone their implementation from the runtime module.
bool CodeGenerator::alwaysCloneRuntimeFunction(const llvm::Function* func) {
  return func->getName() == "query_stub_hoisted_literals" ||
         func->getName() == "multifrag_query_hoisted_literals" ||
         func->getName() == "query_stub" || func->getName() == "multifrag_query" ||
         func->getName() == "fixed_width_int_decode" ||
         func->getName() == "fixed_width_unsigned_decode" ||
         func->getName() == "diff_fixed_width_int_decode" ||
         func->getName() == "fixed_width_double_decode" ||
         func->getName() == "fixed_width_float_decode" ||
         func->getName() == "fixed_width_small_date_decode" ||
         func->getName() == "record_error_code" || func->getName() == "get_error_code" ||
         func->getName() == "pos_start_impl" || func->getName() == "pos_step_impl" ||
         func->getName() == "group_buff_idx_impl" ||
         func->getName() == "init_shared_mem" ||
         func->getName() == "init_shared_mem_nop" || func->getName() == "write_back_nop";
}

llvm::Module* read_template_module(llvm::LLVMContext& context) {
  llvm::SMDiagnostic err;

  auto buffer_or_error = llvm::MemoryBuffer::getFile(omnisci::get_root_abs_path() +
                                                     "/QueryEngine/RuntimeFunctions.bc");
  CHECK(!buffer_or_error.getError()) << "root path=" << omnisci::get_root_abs_path();
  llvm::MemoryBuffer* buffer = buffer_or_error.get().get();

  auto owner = llvm::parseBitcodeFile(buffer->getMemBufferRef(), context);
  CHECK(!owner.takeError());
  auto module = owner.get().release();
  CHECK(module);

  return module;
}

#ifdef HAVE_CUDA
llvm::Module* read_libdevice_module(llvm::LLVMContext& context) {
  llvm::SMDiagnostic err;
  const auto env = get_cuda_home();

  boost::filesystem::path cuda_path{env};
  cuda_path /= "nvvm";
  cuda_path /= "libdevice";
  cuda_path /= "libdevice.10.bc";

  if (!boost::filesystem::exists(cuda_path)) {
    LOG(WARNING) << "Could not find CUDA libdevice; support for some UDF "
                    "functions might not be available.";
    return nullptr;
  }

  auto buffer_or_error = llvm::MemoryBuffer::getFile(cuda_path.c_str());
  CHECK(!buffer_or_error.getError()) << "cuda_path=" << cuda_path.c_str();
  llvm::MemoryBuffer* buffer = buffer_or_error.get().get();

  auto owner = llvm::parseBitcodeFile(buffer->getMemBufferRef(), context);
  CHECK(!owner.takeError());
  auto module = owner.get().release();
  CHECK(module);

  return module;
}
#endif

#ifdef ENABLE_GEOS
llvm::Module* read_geos_module(llvm::LLVMContext& context) {
  llvm::SMDiagnostic err;

  auto buffer_or_error = llvm::MemoryBuffer::getFile(omnisci::get_root_abs_path() +
                                                     "/QueryEngine/GeosRuntime.bc");
  CHECK(!buffer_or_error.getError()) << "root path=" << omnisci::get_root_abs_path();
  llvm::MemoryBuffer* buffer = buffer_or_error.get().get();

  auto owner = llvm::parseBitcodeFile(buffer->getMemBufferRef(), context);
  CHECK(!owner.takeError());
  auto module = owner.get().release();
  CHECK(module);

  return module;
}
#endif

namespace {

void bind_pos_placeholders(const std::string& pos_fn_name,
                           const bool use_resume_param,
                           llvm::Function* query_func,
                           llvm::Module* module) {
  for (auto it = llvm::inst_begin(query_func), e = llvm::inst_end(query_func); it != e;
       ++it) {
    if (!llvm::isa<llvm::CallInst>(*it)) {
      continue;
    }
    auto& pos_call = llvm::cast<llvm::CallInst>(*it);
    if (std::string(pos_call.getCalledFunction()->getName()) == pos_fn_name) {
      if (use_resume_param) {
        const auto error_code_arg = get_arg_by_name(query_func, "error_code");
        llvm::ReplaceInstWithInst(
            &pos_call,
            llvm::CallInst::Create(module->getFunction(pos_fn_name + "_impl"),
                                   error_code_arg));
      } else {
        llvm::ReplaceInstWithInst(
            &pos_call,
            llvm::CallInst::Create(module->getFunction(pos_fn_name + "_impl")));
      }
      break;
    }
  }
}

void set_row_func_argnames(llvm::Function* row_func,
                           const size_t in_col_count,
                           const size_t agg_col_count,
                           const bool hoist_literals) {
  auto arg_it = row_func->arg_begin();

  if (agg_col_count) {
    for (size_t i = 0; i < agg_col_count; ++i) {
      arg_it->setName("out");
      ++arg_it;
    }
  } else {
    arg_it->setName("group_by_buff");
    ++arg_it;
    arg_it->setName("crt_matched");
    ++arg_it;
    arg_it->setName("total_matched");
    ++arg_it;
    arg_it->setName("old_total_matched");
    ++arg_it;
    arg_it->setName("max_matched");
    ++arg_it;
  }

  arg_it->setName("agg_init_val");
  ++arg_it;

  arg_it->setName("pos");
  ++arg_it;

  arg_it->setName("frag_row_off");
  ++arg_it;

  arg_it->setName("num_rows_per_scan");
  ++arg_it;

  if (hoist_literals) {
    arg_it->setName("literals");
    ++arg_it;
  }

  for (size_t i = 0; i < in_col_count; ++i) {
    arg_it->setName("col_buf" + std::to_string(i));
    ++arg_it;
  }

  arg_it->setName("join_hash_tables");
}

llvm::Function* create_row_function(const size_t in_col_count,
                                    const size_t agg_col_count,
                                    const CompilationOptions& co,
                                    llvm::Module* module,
                                    llvm::LLVMContext& context) {
  std::vector<llvm::Type*> row_process_arg_types;
  unsigned int AS = (co.device_type == ExecutorDeviceType::L0) ? 4 : 0;

  if (agg_col_count) {
    // output (aggregate) arguments
    for (size_t i = 0; i < agg_col_count; ++i) {
      row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context, AS));
    }
  } else {
    // group by buffer
    row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context, AS));
    // current match count
    row_process_arg_types.push_back(llvm::Type::getInt32PtrTy(context, AS));
    // total match count passed from the caller
    row_process_arg_types.push_back(llvm::Type::getInt32PtrTy(context, AS));
    // old total match count returned to the caller
    row_process_arg_types.push_back(llvm::Type::getInt32PtrTy(context, AS));
    // max matched (total number of slots in the output buffer)
    row_process_arg_types.push_back(llvm::Type::getInt32PtrTy(context, AS));
  }

  // aggregate init values
  row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context, AS));

  // position argument
  row_process_arg_types.push_back(llvm::Type::getInt64Ty(context));

  // fragment row offset argument
  row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context, AS));

  // number of rows for each scan
  row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context, AS));

  // literals buffer argument
  if (co.hoist_literals) {
    row_process_arg_types.push_back(llvm::Type::getInt8PtrTy(context, AS));
  }

  // column buffer arguments
  for (size_t i = 0; i < in_col_count; ++i) {
    row_process_arg_types.emplace_back(llvm::Type::getInt8PtrTy(context, AS));
  }

  // join hash table argument
  row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context, AS));

  // generate the function
  auto ft =
      llvm::FunctionType::get(get_int_type(32, context), row_process_arg_types, false);

  auto row_func =
      llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "row_func", module);

  // set the row function argument names; for debugging purposes only
  set_row_func_argnames(row_func, in_col_count, agg_col_count, co.hoist_literals);

  return row_func;
}

// Iterate through multifrag_query_func, replacing calls to query_fname with query_func.
void bind_query(llvm::Function* query_func,
                const std::string& query_fname,
                llvm::Function* multifrag_query_func,
                llvm::Module* module) {
  std::vector<llvm::CallInst*> query_stubs;
  for (auto it = llvm::inst_begin(multifrag_query_func),
            e = llvm::inst_end(multifrag_query_func);
       it != e;
       ++it) {
    if (!llvm::isa<llvm::CallInst>(*it)) {
      continue;
    }
    auto& query_call = llvm::cast<llvm::CallInst>(*it);
    if (std::string(query_call.getCalledFunction()->getName()) == query_fname) {
      query_stubs.push_back(&query_call);
    }
  }
  for (auto& S : query_stubs) {
    std::vector<llvm::Value*> args;
    for (size_t i = 0; i < S->getNumArgOperands(); ++i) {
      args.push_back(S->getArgOperand(i));
    }
    llvm::ReplaceInstWithInst(S, llvm::CallInst::Create(query_func, args, ""));
  }
}

std::vector<std::string> get_agg_fnames(const std::vector<Analyzer::Expr*>& target_exprs,
                                        const bool is_group_by) {
  std::vector<std::string> result;
  for (size_t target_idx = 0, agg_col_idx = 0; target_idx < target_exprs.size();
       ++target_idx, ++agg_col_idx) {
    const auto target_expr = target_exprs[target_idx];
    CHECK(target_expr);
    const auto target_type_info = target_expr->get_type_info();
    const auto agg_expr = dynamic_cast<Analyzer::AggExpr*>(target_expr);
    const bool is_varlen =
        (target_type_info.is_string() &&
         target_type_info.get_compression() == kENCODING_NONE) ||
        target_type_info.is_array();  // TODO: should it use is_varlen_array() ?
    if (!agg_expr || agg_expr->get_aggtype() == kSAMPLE) {
      result.emplace_back(target_type_info.is_fp() ? "agg_id_double" : "agg_id");
      if (is_varlen) {
        result.emplace_back("agg_id");
      }
      if (target_type_info.is_geometry()) {
        result.emplace_back("agg_id");
        for (auto i = 2; i < 2 * target_type_info.get_physical_coord_cols(); ++i) {
          result.emplace_back("agg_id");
        }
      }
      continue;
    }
    const auto agg_type = agg_expr->get_aggtype();
    const auto& agg_type_info =
        agg_type != kCOUNT ? agg_expr->get_arg()->get_type_info() : target_type_info;
    switch (agg_type) {
      case kAVG: {
        if (!agg_type_info.is_integer() && !agg_type_info.is_decimal() &&
            !agg_type_info.is_fp()) {
          throw std::runtime_error("AVG is only valid on integer and floating point");
        }
        result.emplace_back((agg_type_info.is_integer() || agg_type_info.is_time())
                                ? "agg_sum"
                                : "agg_sum_double");
        result.emplace_back((agg_type_info.is_integer() || agg_type_info.is_time())
                                ? "agg_count"
                                : "agg_count_double");
        break;
      }
      case kMIN: {
        if (agg_type_info.is_string() || agg_type_info.is_array() ||
            agg_type_info.is_geometry()) {
          throw std::runtime_error(
              "MIN on strings, arrays or geospatial types not supported yet");
        }
        result.emplace_back((agg_type_info.is_integer() || agg_type_info.is_time())
                                ? "agg_min"
                                : "agg_min_double");
        break;
      }
      case kMAX: {
        if (agg_type_info.is_string() || agg_type_info.is_array() ||
            agg_type_info.is_geometry()) {
          throw std::runtime_error(
              "MAX on strings, arrays or geospatial types not supported yet");
        }
        result.emplace_back((agg_type_info.is_integer() || agg_type_info.is_time())
                                ? "agg_max"
                                : "agg_max_double");
        break;
      }
      case kSUM: {
        if (!agg_type_info.is_integer() && !agg_type_info.is_decimal() &&
            !agg_type_info.is_fp()) {
          throw std::runtime_error("SUM is only valid on integer and floating point");
        }
        result.emplace_back((agg_type_info.is_integer() || agg_type_info.is_time())
                                ? "agg_sum"
                                : "agg_sum_double");
        break;
      }
      case kCOUNT:
        result.emplace_back(agg_expr->get_is_distinct() ? "agg_count_distinct"
                                                        : "agg_count");
        break;
      case kSINGLE_VALUE: {
        result.emplace_back(agg_type_info.is_fp() ? "agg_id_double" : "agg_id");
        break;
      }
      case kSAMPLE: {
        // Note that varlen SAMPLE arguments are handled separately above
        result.emplace_back(agg_type_info.is_fp() ? "agg_id_double" : "agg_id");
        break;
      }
      case kAPPROX_COUNT_DISTINCT:
        result.emplace_back("agg_approximate_count_distinct");
        break;
      case kAPPROX_MEDIAN:
        result.emplace_back("agg_approx_median");
        break;
      default:
        CHECK(false);
    }
  }
  return result;
}

}  // namespace

std::unique_ptr<llvm::Module> g_rt_module(read_template_module(getGlobalLLVMContext()));

#ifdef ENABLE_GEOS
std::unique_ptr<llvm::Module> g_rt_geos_module(read_geos_module(getGlobalLLVMContext()));
#endif

#ifdef HAVE_CUDA
std::unique_ptr<llvm::Module> g_rt_libdevice_module(
    read_libdevice_module(getGlobalLLVMContext()));
#endif

bool is_udf_module_present(bool cpu_only) {
  return (cpu_only || udf_gpu_module != nullptr) && (udf_cpu_module != nullptr);
}

bool is_rt_udf_module_present(bool cpu_only) {
  return (cpu_only || rt_udf_gpu_module != nullptr) && (rt_udf_cpu_module != nullptr);
}

void read_udf_gpu_module(const std::string& udf_ir_filename) {
  llvm::SMDiagnostic parse_error;

  llvm::StringRef file_name_arg(udf_ir_filename);
  udf_gpu_module = llvm::parseIRFile(file_name_arg, parse_error, getGlobalLLVMContext());

  if (!udf_gpu_module) {
    throw_parseIR_error(parse_error, udf_ir_filename, /* is_gpu= */ true);
  }

  llvm::Triple gpu_triple(udf_gpu_module->getTargetTriple());
  if (!gpu_triple.isNVPTX()) {
    LOG(WARNING)
        << "Expected triple nvptx64-nvidia-cuda for NVVM IR of loadtime UDFs but got "
        << gpu_triple.str() << ". Disabling the NVVM IR module.";
    udf_gpu_module = nullptr;
  }
}

void read_udf_cpu_module(const std::string& udf_ir_filename) {
  llvm::SMDiagnostic parse_error;

  llvm::StringRef file_name_arg(udf_ir_filename);

  udf_cpu_module = llvm::parseIRFile(file_name_arg, parse_error, getGlobalLLVMContext());
  if (!udf_cpu_module) {
    throw_parseIR_error(parse_error, udf_ir_filename);
  }
}

void read_rt_udf_gpu_module(const std::string& udf_ir_string) {
  llvm::SMDiagnostic parse_error;

  auto buf =
      std::make_unique<llvm::MemoryBufferRef>(udf_ir_string, "Runtime UDF for GPU");

  rt_udf_gpu_module = llvm::parseIR(*buf, parse_error, getGlobalLLVMContext());
  if (!rt_udf_gpu_module) {
    LOG(IR) << "read_rt_udf_gpu_module:NVVM IR:\n" << udf_ir_string << "\nEnd of NNVM IR";
    throw_parseIR_error(parse_error, "", /* is_gpu= */ true);
  }

  llvm::Triple gpu_triple(rt_udf_gpu_module->getTargetTriple());
  if (!gpu_triple.isNVPTX()) {
    LOG(IR) << "read_rt_udf_gpu_module:NVVM IR:\n" << udf_ir_string << "\nEnd of NNVM IR";
    LOG(WARNING) << "Expected triple nvptx64-nvidia-cuda for NVVM IR but got "
                 << gpu_triple.str()
                 << ". Executing runtime UDFs on GPU will be disabled.";
    rt_udf_gpu_module = nullptr;
    return;
  }
}

void read_rt_udf_cpu_module(const std::string& udf_ir_string) {
  llvm::SMDiagnostic parse_error;

  auto buf =
      std::make_unique<llvm::MemoryBufferRef>(udf_ir_string, "Runtime UDF for CPU");

  rt_udf_cpu_module = llvm::parseIR(*buf, parse_error, getGlobalLLVMContext());
  if (!rt_udf_cpu_module) {
    LOG(IR) << "read_rt_udf_cpu_module:LLVM IR:\n" << udf_ir_string << "\nEnd of LLVM IR";
    throw_parseIR_error(parse_error);
  }
}

std::unordered_set<llvm::Function*> CodeGenerator::markDeadRuntimeFuncs(
    llvm::Module& module,
    const std::vector<llvm::Function*>& roots,
    const std::vector<llvm::Function*>& leaves) {
  std::unordered_set<llvm::Function*> live_funcs;
  live_funcs.insert(roots.begin(), roots.end());
  live_funcs.insert(leaves.begin(), leaves.end());

  if (auto F = module.getFunction("init_shared_mem_nop")) {
    live_funcs.insert(F);
  }
  if (auto F = module.getFunction("write_back_nop")) {
    live_funcs.insert(F);
  }

  for (const llvm::Function* F : roots) {
    for (const llvm::BasicBlock& BB : *F) {
      for (const llvm::Instruction& I : BB) {
        if (const llvm::CallInst* CI = llvm::dyn_cast<const llvm::CallInst>(&I)) {
          live_funcs.insert(CI->getCalledFunction());
        }
      }
    }
  }

  for (llvm::Function& F : module) {
    if (!live_funcs.count(&F) && !F.isDeclaration()) {
      F.setLinkage(llvm::GlobalValue::InternalLinkage);
    }
  }

  return live_funcs;
}

namespace {
// searches for a particular variable within a specific basic block (or all if bb_name is
// empty)
template <typename InstType>
llvm::Value* find_variable_in_basic_block(llvm::Function* func,
                                          std::string bb_name,
                                          std::string variable_name) {
  llvm::Value* result = nullptr;
  if (func == nullptr || variable_name.empty()) {
    return result;
  }
  bool is_found = false;
  for (auto bb_it = func->begin(); bb_it != func->end() && !is_found; ++bb_it) {
    if (!bb_name.empty() && bb_it->getName() != bb_name) {
      continue;
    }
    for (auto inst_it = bb_it->begin(); inst_it != bb_it->end(); inst_it++) {
      if (llvm::isa<InstType>(*inst_it)) {
        if (inst_it->getName() == variable_name) {
          result = &*inst_it;
          is_found = true;
          break;
        }
      }
    }
  }
  return result;
}
};  // namespace

void Executor::createErrorCheckControlFlow(
    llvm::Function* query_func,
    bool run_with_dynamic_watchdog,
    bool run_with_allowing_runtime_interrupt,
    ExecutorDeviceType device_type,
    const std::vector<InputTableInfo>& input_table_infos) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());

  // check whether the row processing was successful; currently, it can
  // fail by running out of group by buffer slots

  if (run_with_dynamic_watchdog && run_with_allowing_runtime_interrupt) {
    // when both dynamic watchdog and runtime interrupt turns on
    // we use dynamic watchdog
    run_with_allowing_runtime_interrupt = false;
  }

  {
    // disable injecting query interrupt checker if the session info is invalid
    mapd_shared_lock<mapd_shared_mutex> session_read_lock(executor_session_mutex_);
    if (current_query_session_.empty()) {
      run_with_allowing_runtime_interrupt = false;
    }
  }

  llvm::Value* row_count = nullptr;
  if ((run_with_dynamic_watchdog || run_with_allowing_runtime_interrupt) &&
      device_type == ExecutorDeviceType::GPU) {
    row_count =
        find_variable_in_basic_block<llvm::LoadInst>(query_func, ".entry", "row_count");
  }

  bool done_splitting = false;
  for (auto bb_it = query_func->begin(); bb_it != query_func->end() && !done_splitting;
       ++bb_it) {
    llvm::Value* pos = nullptr;
    for (auto inst_it = bb_it->begin(); inst_it != bb_it->end(); ++inst_it) {
      if ((run_with_dynamic_watchdog || run_with_allowing_runtime_interrupt) &&
          llvm::isa<llvm::PHINode>(*inst_it)) {
        if (inst_it->getName() == "pos") {
          pos = &*inst_it;
        }
        continue;
      }
      if (!llvm::isa<llvm::CallInst>(*inst_it)) {
        continue;
      }
      auto& row_func_call = llvm::cast<llvm::CallInst>(*inst_it);
      if (std::string(row_func_call.getCalledFunction()->getName()) == "row_process") {
        auto next_inst_it = inst_it;
        ++next_inst_it;
        auto new_bb = bb_it->splitBasicBlock(next_inst_it);
        auto& br_instr = bb_it->back();
        llvm::IRBuilder<> ir_builder(&br_instr);
        llvm::Value* err_lv = &*inst_it;
        llvm::Value* err_lv_returned_from_row_func = nullptr;
        if (run_with_dynamic_watchdog) {
          CHECK(pos);
          llvm::Value* call_watchdog_lv = nullptr;
          if (device_type == ExecutorDeviceType::GPU) {
            // In order to make sure all threads within a block see the same barrier,
            // only those blocks whose none of their threads have experienced the critical
            // edge will go through the dynamic watchdog computation
            CHECK(row_count);
            auto crit_edge_rem =
                (blockSize() & (blockSize() - 1))
                    ? ir_builder.CreateSRem(
                          row_count,
                          cgen_state_->llInt(static_cast<int64_t>(blockSize())))
                    : ir_builder.CreateAnd(
                          row_count,
                          cgen_state_->llInt(static_cast<int64_t>(blockSize() - 1)));
            auto crit_edge_threshold = ir_builder.CreateSub(row_count, crit_edge_rem);
            crit_edge_threshold->setName("crit_edge_threshold");

            // only those threads where pos < crit_edge_threshold go through dynamic
            // watchdog call
            call_watchdog_lv =
                ir_builder.CreateICmp(llvm::ICmpInst::ICMP_SLT, pos, crit_edge_threshold);
          } else {
            // CPU path: run watchdog for every 64th row
            auto dw_predicate = ir_builder.CreateAnd(pos, uint64_t(0x3f));
            call_watchdog_lv = ir_builder.CreateICmp(
                llvm::ICmpInst::ICMP_EQ, dw_predicate, cgen_state_->llInt(int64_t(0LL)));
          }
          CHECK(call_watchdog_lv);
          auto error_check_bb = bb_it->splitBasicBlock(
              llvm::BasicBlock::iterator(br_instr), ".error_check");
          auto& watchdog_br_instr = bb_it->back();

          auto watchdog_check_bb = llvm::BasicBlock::Create(
              cgen_state_->context_, ".watchdog_check", query_func, error_check_bb);
          llvm::IRBuilder<> watchdog_ir_builder(watchdog_check_bb);
          auto detected_timeout = watchdog_ir_builder.CreateCall(
              cgen_state_->module_->getFunction("dynamic_watchdog"), {});
          auto timeout_err_lv = watchdog_ir_builder.CreateSelect(
              detected_timeout, cgen_state_->llInt(Executor::ERR_OUT_OF_TIME), err_lv);
          watchdog_ir_builder.CreateBr(error_check_bb);

          llvm::ReplaceInstWithInst(
              &watchdog_br_instr,
              llvm::BranchInst::Create(
                  watchdog_check_bb, error_check_bb, call_watchdog_lv));
          ir_builder.SetInsertPoint(&br_instr);
          auto unified_err_lv = ir_builder.CreatePHI(err_lv->getType(), 2);

          unified_err_lv->addIncoming(timeout_err_lv, watchdog_check_bb);
          unified_err_lv->addIncoming(err_lv, &*bb_it);
          err_lv = unified_err_lv;
        } else if (run_with_allowing_runtime_interrupt) {
          CHECK(pos);
          llvm::Value* call_check_interrupt_lv = nullptr;
          if (device_type == ExecutorDeviceType::GPU) {
            // approximate how many times the %pos variable
            // is increased --> the number of iteration
            // here we calculate the # bit shift by considering grid/block/fragment sizes
            // since if we use the fixed one (i.e., per 64-th increment)
            // some CUDA threads cannot enter the interrupt checking block depending on
            // the fragment size --> a thread may not take care of 64 threads if an outer
            // table is not sufficiently large, and so cannot be interrupted
            int32_t num_shift_by_gridDim = shared::getExpOfTwo(gridSize());
            int32_t num_shift_by_blockDim = shared::getExpOfTwo(blockSize());
            int total_num_shift = num_shift_by_gridDim + num_shift_by_blockDim;
            uint64_t interrupt_checking_freq = 32;
            auto freq_control_knob = g_running_query_interrupt_freq;
            CHECK_GT(freq_control_knob, 0);
            CHECK_LE(freq_control_knob, 1.0);
            if (!input_table_infos.empty()) {
              const auto& outer_table_info = *input_table_infos.begin();
              auto num_outer_table_tuples = outer_table_info.info.getNumTuples();
              if (outer_table_info.table_id < 0) {
                auto* rs = (*outer_table_info.info.fragments.begin()).resultSet;
                CHECK(rs);
                num_outer_table_tuples = rs->entryCount();
              } else {
                auto num_frags = outer_table_info.info.fragments.size();
                if (num_frags > 0) {
                  num_outer_table_tuples =
                      outer_table_info.info.fragments.begin()->getNumTuples();
                }
              }
              if (num_outer_table_tuples > 0) {
                // gridSize * blockSize --> pos_step (idx of the next row per thread)
                // we additionally multiply two to pos_step since the number of
                // dispatched blocks are double of the gridSize
                // # tuples (of fragment) / pos_step --> maximum # increment (K)
                // also we multiply 1 / freq_control_knob to K to control the frequency
                // So, needs to check the interrupt status more frequently? make K smaller
                auto max_inc = uint64_t(
                    floor(num_outer_table_tuples / (gridSize() * blockSize() * 2)));
                if (max_inc < 2) {
                  // too small `max_inc`, so this correction is necessary to make
                  // `interrupt_checking_freq` be valid (i.e., larger than zero)
                  max_inc = 2;
                }
                auto calibrated_inc = uint64_t(floor(max_inc * (1 - freq_control_knob)));
                interrupt_checking_freq =
                    uint64_t(pow(2, shared::getExpOfTwo(calibrated_inc)));
                // add the coverage when interrupt_checking_freq > K
                // if so, some threads still cannot be branched to the interrupt checker
                // so we manually use smaller but close to the max_inc as freq
                if (interrupt_checking_freq > max_inc) {
                  interrupt_checking_freq = max_inc / 2;
                }
                if (interrupt_checking_freq < 8) {
                  // such small freq incurs too frequent interrupt status checking,
                  // so we fixup to the minimum freq value at some reasonable degree
                  interrupt_checking_freq = 8;
                }
              }
            }
            VLOG(1) << "Set the running query interrupt checking frequency: "
                    << interrupt_checking_freq;
            // check the interrupt flag for every interrupt_checking_freq-th iteration
            llvm::Value* pos_shifted_per_iteration =
                ir_builder.CreateLShr(pos, cgen_state_->llInt(total_num_shift));
            auto interrupt_predicate =
                ir_builder.CreateAnd(pos_shifted_per_iteration, interrupt_checking_freq);
            call_check_interrupt_lv =
                ir_builder.CreateICmp(llvm::ICmpInst::ICMP_EQ,
                                      interrupt_predicate,
                                      cgen_state_->llInt(int64_t(0LL)));
          } else {
            // CPU path: run interrupt checker for every 64th row
            auto interrupt_predicate = ir_builder.CreateAnd(pos, uint64_t(0x3f));
            call_check_interrupt_lv =
                ir_builder.CreateICmp(llvm::ICmpInst::ICMP_EQ,
                                      interrupt_predicate,
                                      cgen_state_->llInt(int64_t(0LL)));
          }
          CHECK(call_check_interrupt_lv);
          auto error_check_bb = bb_it->splitBasicBlock(
              llvm::BasicBlock::iterator(br_instr), ".error_check");
          auto& check_interrupt_br_instr = bb_it->back();

          auto interrupt_check_bb = llvm::BasicBlock::Create(
              cgen_state_->context_, ".interrupt_check", query_func, error_check_bb);
          llvm::IRBuilder<> interrupt_checker_ir_builder(interrupt_check_bb);
          auto detected_interrupt = interrupt_checker_ir_builder.CreateCall(
              cgen_state_->module_->getFunction("check_interrupt"), {});
          auto interrupt_err_lv = interrupt_checker_ir_builder.CreateSelect(
              detected_interrupt, cgen_state_->llInt(Executor::ERR_INTERRUPTED), err_lv);
          interrupt_checker_ir_builder.CreateBr(error_check_bb);

          llvm::ReplaceInstWithInst(
              &check_interrupt_br_instr,
              llvm::BranchInst::Create(
                  interrupt_check_bb, error_check_bb, call_check_interrupt_lv));
          ir_builder.SetInsertPoint(&br_instr);
          auto unified_err_lv = ir_builder.CreatePHI(err_lv->getType(), 2);

          unified_err_lv->addIncoming(interrupt_err_lv, interrupt_check_bb);
          unified_err_lv->addIncoming(err_lv, &*bb_it);
          err_lv = unified_err_lv;
        }
        if (!err_lv_returned_from_row_func) {
          err_lv_returned_from_row_func = err_lv;
        }
        if (device_type == ExecutorDeviceType::GPU && g_enable_dynamic_watchdog) {
          // let kernel execution finish as expected, regardless of the observed error,
          // unless it is from the dynamic watchdog where all threads within that block
          // return together.
          err_lv = ir_builder.CreateICmp(llvm::ICmpInst::ICMP_EQ,
                                         err_lv,
                                         cgen_state_->llInt(Executor::ERR_OUT_OF_TIME));
        } else {
          err_lv = ir_builder.CreateICmp(llvm::ICmpInst::ICMP_NE,
                                         err_lv,
                                         cgen_state_->llInt(static_cast<int32_t>(0)));
        }
        auto error_bb = llvm::BasicBlock::Create(
            cgen_state_->context_, ".error_exit", query_func, new_bb);
        const auto error_code_arg = get_arg_by_name(query_func, "error_code");
        llvm::CallInst::Create(
            cgen_state_->module_->getFunction("record_error_code"),
            std::vector<llvm::Value*>{err_lv_returned_from_row_func, error_code_arg},
            "",
            error_bb);
        llvm::ReturnInst::Create(cgen_state_->context_, error_bb);
        llvm::ReplaceInstWithInst(&br_instr,
                                  llvm::BranchInst::Create(error_bb, new_bb, err_lv));
        done_splitting = true;
        break;
      }
    }
  }
  CHECK(done_splitting);
}

std::vector<llvm::Value*> Executor::inlineHoistedLiterals() {
  AUTOMATIC_IR_METADATA(cgen_state_.get());

  std::vector<llvm::Value*> hoisted_literals;

  // row_func_ is using literals whose defs have been hoisted up to the query_func_,
  // extend row_func_ signature to include extra args to pass these literal values.
  std::vector<llvm::Type*> row_process_arg_types;

  for (llvm::Function::arg_iterator I = cgen_state_->row_func_->arg_begin(),
                                    E = cgen_state_->row_func_->arg_end();
       I != E;
       ++I) {
    row_process_arg_types.push_back(I->getType());
  }

  for (auto& element : cgen_state_->query_func_literal_loads_) {
    for (auto value : element.second) {
      row_process_arg_types.push_back(value->getType());
    }
  }

  auto ft = llvm::FunctionType::get(
      get_int_type(32, cgen_state_->context_), row_process_arg_types, false);
  auto row_func_with_hoisted_literals =
      llvm::Function::Create(ft,
                             llvm::Function::ExternalLinkage,
                             "row_func_hoisted_literals",
                             cgen_state_->row_func_->getParent());

  auto row_func_arg_it = row_func_with_hoisted_literals->arg_begin();
  for (llvm::Function::arg_iterator I = cgen_state_->row_func_->arg_begin(),
                                    E = cgen_state_->row_func_->arg_end();
       I != E;
       ++I) {
    if (I->hasName()) {
      row_func_arg_it->setName(I->getName());
    }
    ++row_func_arg_it;
  }

  decltype(row_func_with_hoisted_literals) filter_func_with_hoisted_literals{nullptr};
  decltype(row_func_arg_it) filter_func_arg_it{nullptr};
  if (cgen_state_->filter_func_) {
    // filter_func_ is using literals whose defs have been hoisted up to the row_func_,
    // extend filter_func_ signature to include extra args to pass these literal values.
    std::vector<llvm::Type*> filter_func_arg_types;

    for (llvm::Function::arg_iterator I = cgen_state_->filter_func_->arg_begin(),
                                      E = cgen_state_->filter_func_->arg_end();
         I != E;
         ++I) {
      filter_func_arg_types.push_back(I->getType());
    }

    for (auto& element : cgen_state_->query_func_literal_loads_) {
      for (auto value : element.second) {
        filter_func_arg_types.push_back(value->getType());
      }
    }

    auto ft2 = llvm::FunctionType::get(
        get_int_type(32, cgen_state_->context_), filter_func_arg_types, false);
    filter_func_with_hoisted_literals =
        llvm::Function::Create(ft2,
                               llvm::Function::ExternalLinkage,
                               "filter_func_hoisted_literals",
                               cgen_state_->filter_func_->getParent());

    filter_func_arg_it = filter_func_with_hoisted_literals->arg_begin();
    for (llvm::Function::arg_iterator I = cgen_state_->filter_func_->arg_begin(),
                                      E = cgen_state_->filter_func_->arg_end();
         I != E;
         ++I) {
      if (I->hasName()) {
        filter_func_arg_it->setName(I->getName());
      }
      ++filter_func_arg_it;
    }
  }

  std::unordered_map<int, std::vector<llvm::Value*>>
      query_func_literal_loads_function_arguments,
      query_func_literal_loads_function_arguments2;

  for (auto& element : cgen_state_->query_func_literal_loads_) {
    std::vector<llvm::Value*> argument_values, argument_values2;

    for (auto value : element.second) {
      hoisted_literals.push_back(value);
      argument_values.push_back(&*row_func_arg_it);
      if (cgen_state_->filter_func_) {
        argument_values2.push_back(&*filter_func_arg_it);
        cgen_state_->filter_func_args_[&*row_func_arg_it] = &*filter_func_arg_it;
      }
      if (value->hasName()) {
        row_func_arg_it->setName("arg_" + value->getName());
        if (cgen_state_->filter_func_) {
          filter_func_arg_it->getContext();
          filter_func_arg_it->setName("arg_" + value->getName());
        }
      }
      ++row_func_arg_it;
      ++filter_func_arg_it;
    }

    query_func_literal_loads_function_arguments[element.first] = argument_values;
    query_func_literal_loads_function_arguments2[element.first] = argument_values2;
  }

  // copy the row_func function body over
  // see
  // https://stackoverflow.com/questions/12864106/move-function-body-avoiding-full-cloning/18751365
  row_func_with_hoisted_literals->getBasicBlockList().splice(
      row_func_with_hoisted_literals->begin(),
      cgen_state_->row_func_->getBasicBlockList());

  // also replace row_func arguments with the arguments from row_func_hoisted_literals
  for (llvm::Function::arg_iterator I = cgen_state_->row_func_->arg_begin(),
                                    E = cgen_state_->row_func_->arg_end(),
                                    I2 = row_func_with_hoisted_literals->arg_begin();
       I != E;
       ++I) {
    I->replaceAllUsesWith(&*I2);
    I2->takeName(&*I);
    cgen_state_->filter_func_args_.replace(&*I, &*I2);
    ++I2;
  }

  cgen_state_->row_func_ = row_func_with_hoisted_literals;

  // and finally replace  literal placeholders
  std::vector<llvm::Instruction*> placeholders;
  std::string prefix("__placeholder__literal_");
  for (auto it = llvm::inst_begin(row_func_with_hoisted_literals),
            e = llvm::inst_end(row_func_with_hoisted_literals);
       it != e;
       ++it) {
    if (it->hasName() && it->getName().startswith(prefix)) {
      auto offset_and_index_entry =
          cgen_state_->row_func_hoisted_literals_.find(llvm::dyn_cast<llvm::Value>(&*it));
      CHECK(offset_and_index_entry != cgen_state_->row_func_hoisted_literals_.end());

      int lit_off = offset_and_index_entry->second.offset_in_literal_buffer;
      int lit_idx = offset_and_index_entry->second.index_of_literal_load;

      it->replaceAllUsesWith(
          query_func_literal_loads_function_arguments[lit_off][lit_idx]);
      placeholders.push_back(&*it);
    }
  }
  for (auto placeholder : placeholders) {
    placeholder->removeFromParent();
  }

  if (cgen_state_->filter_func_) {
    // copy the filter_func function body over
    // see
    // https://stackoverflow.com/questions/12864106/move-function-body-avoiding-full-cloning/18751365
    filter_func_with_hoisted_literals->getBasicBlockList().splice(
        filter_func_with_hoisted_literals->begin(),
        cgen_state_->filter_func_->getBasicBlockList());

    // also replace filter_func arguments with the arguments from
    // filter_func_hoisted_literals
    for (llvm::Function::arg_iterator I = cgen_state_->filter_func_->arg_begin(),
                                      E = cgen_state_->filter_func_->arg_end(),
                                      I2 = filter_func_with_hoisted_literals->arg_begin();
         I != E;
         ++I) {
      I->replaceAllUsesWith(&*I2);
      I2->takeName(&*I);
      ++I2;
    }

    cgen_state_->filter_func_ = filter_func_with_hoisted_literals;

    // and finally replace  literal placeholders
    std::vector<llvm::Instruction*> placeholders;
    std::string prefix("__placeholder__literal_");
    for (auto it = llvm::inst_begin(filter_func_with_hoisted_literals),
              e = llvm::inst_end(filter_func_with_hoisted_literals);
         it != e;
         ++it) {
      if (it->hasName() && it->getName().startswith(prefix)) {
        auto offset_and_index_entry = cgen_state_->row_func_hoisted_literals_.find(
            llvm::dyn_cast<llvm::Value>(&*it));
        CHECK(offset_and_index_entry != cgen_state_->row_func_hoisted_literals_.end());

        int lit_off = offset_and_index_entry->second.offset_in_literal_buffer;
        int lit_idx = offset_and_index_entry->second.index_of_literal_load;

        it->replaceAllUsesWith(
            query_func_literal_loads_function_arguments2[lit_off][lit_idx]);
        placeholders.push_back(&*it);
      }
    }
    for (auto placeholder : placeholders) {
      placeholder->removeFromParent();
    }
  }

  return hoisted_literals;
}

namespace {

size_t get_shared_memory_size(const bool shared_mem_used,
                              const QueryMemoryDescriptor* query_mem_desc_ptr) {
  return shared_mem_used
             ? (query_mem_desc_ptr->getRowSize() * query_mem_desc_ptr->getEntryCount())
             : 0;
}

bool is_gpu_shared_mem_supported(const QueryMemoryDescriptor* query_mem_desc_ptr,
                                 const RelAlgExecutionUnit& ra_exe_unit,
                                 const CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                 const ExecutorDeviceType device_type,
                                 const unsigned gpu_blocksize,
                                 const unsigned num_blocks_per_mp) {
  if (device_type == ExecutorDeviceType::CPU) {
    return false;
  }
  if (query_mem_desc_ptr->didOutputColumnar()) {
    return false;
  }
  CHECK(query_mem_desc_ptr);
  CHECK(cuda_mgr);
  /*
   * We only use shared memory strategy if GPU hardware provides native shared
   * memory atomics support. From CUDA Toolkit documentation:
   * https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html#atomic-ops "Like
   * Maxwell, Pascal [and Volta] provides native shared memory atomic operations
   * for 32-bit integer arithmetic, along with native 32 or 64-bit compare-and-swap
   * (CAS)."
   *
   **/
  if (!cuda_mgr->isArchMaxwellOrLaterForAll()) {
    return false;
  }

  if (query_mem_desc_ptr->getQueryDescriptionType() ==
          QueryDescriptionType::NonGroupedAggregate &&
      g_enable_smem_non_grouped_agg &&
      query_mem_desc_ptr->countDistinctDescriptorsLogicallyEmpty()) {
    // TODO: relax this, if necessary
    if (gpu_blocksize < query_mem_desc_ptr->getEntryCount()) {
      return false;
    }
    // skip shared memory usage when dealing with 1) variable length targets, 2)
    // not a COUNT aggregate
    const auto target_infos =
        target_exprs_to_infos(ra_exe_unit.target_exprs, *query_mem_desc_ptr);
    std::unordered_set<SQLAgg> supported_aggs{kCOUNT};
    if (std::find_if(target_infos.begin(),
                     target_infos.end(),
                     [&supported_aggs](const TargetInfo& ti) {
                       if (ti.sql_type.is_varlen() ||
                           !supported_aggs.count(ti.agg_kind)) {
                         return true;
                       } else {
                         return false;
                       }
                     }) == target_infos.end()) {
      return true;
    }
  }
  if (query_mem_desc_ptr->getQueryDescriptionType() ==
          QueryDescriptionType::GroupByPerfectHash &&
      g_enable_smem_group_by) {
    /**
     * To simplify the implementation for practical purposes, we
     * initially provide shared memory support for cases where there are at most as many
     * entries in the output buffer as there are threads within each GPU device. In
     * order to relax this assumption later, we need to add a for loop in generated
     * codes such that each thread loops over multiple entries.
     * TODO: relax this if necessary
     */
    if (gpu_blocksize < query_mem_desc_ptr->getEntryCount()) {
      return false;
    }

    // Fundamentally, we should use shared memory whenever the output buffer
    // is small enough so that we can fit it in the shared memory and yet expect
    // good occupancy.
    // For now, we allow keyless, row-wise layout, and only for perfect hash
    // group by operations.
    if (query_mem_desc_ptr->hasKeylessHash() &&
        query_mem_desc_ptr->countDistinctDescriptorsLogicallyEmpty() &&
        !query_mem_desc_ptr->useStreamingTopN()) {
      const size_t shared_memory_threshold_bytes = std::min(
          g_gpu_smem_threshold == 0 ? SIZE_MAX : g_gpu_smem_threshold,
          cuda_mgr->getMinSharedMemoryPerBlockForAllDevices() / num_blocks_per_mp);
      const auto output_buffer_size =
          query_mem_desc_ptr->getRowSize() * query_mem_desc_ptr->getEntryCount();
      if (output_buffer_size > shared_memory_threshold_bytes) {
        return false;
      }

      // skip shared memory usage when dealing with 1) variable length targets, 2)
      // non-basic aggregates (COUNT, SUM, MIN, MAX, AVG)
      // TODO: relax this if necessary
      const auto target_infos =
          target_exprs_to_infos(ra_exe_unit.target_exprs, *query_mem_desc_ptr);
      std::unordered_set<SQLAgg> supported_aggs{kCOUNT};
      if (g_enable_smem_grouped_non_count_agg) {
        supported_aggs = {kCOUNT, kMIN, kMAX, kSUM, kAVG};
      }
      if (std::find_if(target_infos.begin(),
                       target_infos.end(),
                       [&supported_aggs](const TargetInfo& ti) {
                         if (ti.sql_type.is_varlen() ||
                             !supported_aggs.count(ti.agg_kind)) {
                           return true;
                         } else {
                           return false;
                         }
                       }) == target_infos.end()) {
        return true;
      }
    }
  }
  return false;
}

#ifndef NDEBUG
std::string serialize_llvm_metadata_footnotes(llvm::Function* query_func,
                                              CgenState* cgen_state) {
  std::string llvm_ir;
  std::unordered_set<llvm::MDNode*> md;

  // Loop over all instructions in the query function.
  for (auto bb_it = query_func->begin(); bb_it != query_func->end(); ++bb_it) {
    for (auto instr_it = bb_it->begin(); instr_it != bb_it->end(); ++instr_it) {
      llvm::SmallVector<std::pair<unsigned, llvm::MDNode*>, 100> imd;
      instr_it->getAllMetadata(imd);
      for (auto [kind, node] : imd) {
        md.insert(node);
      }
    }
  }

  // Loop over all instructions in the row function.
  for (auto bb_it = cgen_state->row_func_->begin(); bb_it != cgen_state->row_func_->end();
       ++bb_it) {
    for (auto instr_it = bb_it->begin(); instr_it != bb_it->end(); ++instr_it) {
      llvm::SmallVector<std::pair<unsigned, llvm::MDNode*>, 100> imd;
      instr_it->getAllMetadata(imd);
      for (auto [kind, node] : imd) {
        md.insert(node);
      }
    }
  }

  // Loop over all instructions in the filter function.
  if (cgen_state->filter_func_) {
    for (auto bb_it = cgen_state->filter_func_->begin();
         bb_it != cgen_state->filter_func_->end();
         ++bb_it) {
      for (auto instr_it = bb_it->begin(); instr_it != bb_it->end(); ++instr_it) {
        llvm::SmallVector<std::pair<unsigned, llvm::MDNode*>, 100> imd;
        instr_it->getAllMetadata(imd);
        for (auto [kind, node] : imd) {
          md.insert(node);
        }
      }
    }
  }

  // Sort the metadata by canonical number and convert to text.
  if (!md.empty()) {
    std::map<size_t, std::string> sorted_strings;
    for (auto p : md) {
      std::string str;
      llvm::raw_string_ostream os(str);
      p->print(os, cgen_state->module_, true);
      os.flush();
      auto fields = split(str, {}, 1);
      if (fields.empty() || fields[0].empty()) {
        continue;
      }
      sorted_strings.emplace(std::stoul(fields[0].substr(1)), str);
    }
    llvm_ir += "\n";
    for (auto [id, text] : sorted_strings) {
      llvm_ir += text;
      llvm_ir += "\n";
    }
  }

  return llvm_ir;
}
#endif  // NDEBUG

}  // namespace

std::tuple<CompilationResult, std::unique_ptr<QueryMemoryDescriptor>>
Executor::compileWorkUnit(const std::vector<InputTableInfo>& query_infos,
                          const PlanState::DeletedColumnsMap& deleted_cols_map,
                          const RelAlgExecutionUnit& ra_exe_unit,
                          const CompilationOptions& co,
                          const ExecutionOptions& eo,
                          const CudaMgr_Namespace::CudaMgr* cuda_mgr,
                          const l0::L0Manager* l0_mgr,
                          const bool allow_lazy_fetch,
                          std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                          const size_t max_groups_buffer_entry_guess,
                          const int8_t crt_min_byte_width,
                          const bool has_cardinality_estimation,
                          ColumnCacheMap& column_cache,
                          RenderInfo* render_info) {
  auto timer = DEBUG_TIMER(__func__);

  if (co.device_type == ExecutorDeviceType::GPU) {
    const auto cuda_mgr = catalog_->getDataMgr().getCudaMgr();
    if (!cuda_mgr) {
      throw QueryMustRunOnCpu();
    }
  }
  if (co.device_type == ExecutorDeviceType::L0) {
    const auto l0_mgr = catalog_->getDataMgr().getL0Mgr();
    if (!l0_mgr) {
      throw QueryMustRunOnCpu();
    }
  }

#ifndef NDEBUG
  static std::uint64_t counter = 0;
  ++counter;
  VLOG(1) << "CODEGEN #" << counter << ":";
  LOG(IR) << "CODEGEN #" << counter << ":";
  LOG(PTX) << "CODEGEN #" << counter << ":";
  LOG(ASM) << "CODEGEN #" << counter << ":";
#endif

  nukeOldState(allow_lazy_fetch, query_infos, deleted_cols_map, &ra_exe_unit);

  addTransientStringLiterals(ra_exe_unit, row_set_mem_owner);

  GroupByAndAggregate group_by_and_aggregate(
      this,
      co.device_type,
      ra_exe_unit,
      query_infos,
      row_set_mem_owner,
      has_cardinality_estimation ? std::optional<int64_t>(max_groups_buffer_entry_guess)
                                 : std::nullopt);
  auto query_mem_desc =
      group_by_and_aggregate.initQueryMemoryDescriptor(eo.allow_multifrag,
                                                       max_groups_buffer_entry_guess,
                                                       crt_min_byte_width,
                                                       render_info,
                                                       eo.output_columnar_hint);

  if (query_mem_desc->getQueryDescriptionType() ==
          QueryDescriptionType::GroupByBaselineHash &&
      !has_cardinality_estimation &&
      (!render_info || !render_info->isPotentialInSituRender()) && !eo.just_explain) {
    const auto col_range_info = group_by_and_aggregate.getColRangeInfo();
    throw CardinalityEstimationRequired(col_range_info.max - col_range_info.min);
  }

  const bool output_columnar = query_mem_desc->didOutputColumnar();
  const bool gpu_shared_mem_optimization =
      is_gpu_shared_mem_supported(query_mem_desc.get(),
                                  ra_exe_unit,
                                  cuda_mgr,
                                  co.device_type,
                                  cuda_mgr ? this->blockSize() : 1,
                                  cuda_mgr ? this->numBlocksPerMP() : 1);
  if (gpu_shared_mem_optimization) {
    // disable interleaved bins optimization on the GPU
    query_mem_desc->setHasInterleavedBinsOnGpu(false);
    LOG(DEBUG1) << "GPU shared memory is used for the " +
                       query_mem_desc->queryDescTypeToString() + " query(" +
                       std::to_string(get_shared_memory_size(gpu_shared_mem_optimization,
                                                             query_mem_desc.get())) +
                       " out of " + std::to_string(g_gpu_smem_threshold) + " bytes).";
  }

  const GpuSharedMemoryContext gpu_smem_context(
      get_shared_memory_size(gpu_shared_mem_optimization, query_mem_desc.get()));

  if (co.device_type == ExecutorDeviceType::GPU) {
    const size_t num_count_distinct_descs =
        query_mem_desc->getCountDistinctDescriptorsSize();
    for (size_t i = 0; i < num_count_distinct_descs; i++) {
      const auto& count_distinct_descriptor =
          query_mem_desc->getCountDistinctDescriptor(i);
      if (count_distinct_descriptor.impl_type_ == CountDistinctImplType::StdSet ||
          (count_distinct_descriptor.impl_type_ != CountDistinctImplType::Invalid &&
           !co.hoist_literals)) {
        throw QueryMustRunOnCpu();
      }
    }
  }

  // Read the module template and target either CPU or GPU
  // by binding the stream position functions to the right implementation:
  // stride access for GPU, contiguous for CPU
  auto rt_module_copy = llvm::CloneModule(
      *g_rt_module.get(), cgen_state_->vmap_, [](const llvm::GlobalValue* gv) {
        auto func = llvm::dyn_cast<llvm::Function>(gv);
        if (!func) {
          return true;
        }
        return (func->getLinkage() == llvm::GlobalValue::LinkageTypes::PrivateLinkage ||
                func->getLinkage() == llvm::GlobalValue::LinkageTypes::InternalLinkage ||
                CodeGenerator::alwaysCloneRuntimeFunction(func));
      });
  switch (co.device_type) {
    case ExecutorDeviceType::CPU:
      if (is_udf_module_present(true)) {
        CodeGenerator::link_udf_module(
            udf_cpu_module, *rt_module_copy, cgen_state_.get());
      }
      if (is_rt_udf_module_present(true)) {
        CodeGenerator::link_udf_module(
            rt_udf_cpu_module, *rt_module_copy, cgen_state_.get());
      }
      break;
    case ExecutorDeviceType::GPU:
      rt_module_copy->setDataLayout(get_gpu_data_layout());
      rt_module_copy->setTargetTriple(get_gpu_target_triple_string());
      if (is_udf_module_present()) {
        CodeGenerator::link_udf_module(
            udf_gpu_module, *rt_module_copy, cgen_state_.get());
      }
      if (is_rt_udf_module_present()) {
        CodeGenerator::link_udf_module(
            rt_udf_gpu_module, *rt_module_copy, cgen_state_.get());
      }
      break;
    case ExecutorDeviceType::L0:
      rt_module_copy->setTargetTriple(get_l0_target_triple_string());
      // todo: link udf & rt_udf
      break;

    default:
      CHECK(false) << "Invalid device type!\n";
  }

  cgen_state_->module_ = rt_module_copy.release();
  AUTOMATIC_IR_METADATA(cgen_state_.get());

  auto agg_fnames =
      get_agg_fnames(ra_exe_unit.target_exprs, !ra_exe_unit.groupby_exprs.empty());

  const auto agg_slot_count = ra_exe_unit.estimator ? size_t(1) : agg_fnames.size();

  const bool is_group_by{query_mem_desc->isGroupBy()};
  auto [query_func, row_func_call] = is_group_by
                                         ? query_group_by_template(cgen_state_->module_,
                                                                   co.hoist_literals,
                                                                   *query_mem_desc,
                                                                   co.device_type,
                                                                   ra_exe_unit.scan_limit,
                                                                   gpu_smem_context)
                                         : query_template(cgen_state_->module_,
                                                          agg_slot_count,
                                                          co,
                                                          !!ra_exe_unit.estimator,
                                                          gpu_smem_context);
  bind_pos_placeholders("pos_start", true, query_func, cgen_state_->module_);
  bind_pos_placeholders("group_buff_idx", false, query_func, cgen_state_->module_);
  bind_pos_placeholders("pos_step", false, query_func, cgen_state_->module_);

  cgen_state_->query_func_ = query_func;
  cgen_state_->row_func_call_ = row_func_call;
  cgen_state_->query_func_entry_ir_builder_.SetInsertPoint(
      &query_func->getEntryBlock().front());

  // Generate the function signature and column head fetches s.t.
  // double indirection isn't needed in the inner loop
  auto& fetch_bb = query_func->front();
  llvm::IRBuilder<> fetch_ir_builder(&fetch_bb);
  fetch_ir_builder.SetInsertPoint(&*fetch_bb.begin());
  auto col_heads = generate_column_heads_load(ra_exe_unit.input_col_descs.size(),
                                              query_func->args().begin(),
                                              fetch_ir_builder,
                                              cgen_state_->context_);
  CHECK_EQ(ra_exe_unit.input_col_descs.size(), col_heads.size());

  cgen_state_->row_func_ = create_row_function(ra_exe_unit.input_col_descs.size(),
                                               is_group_by ? 0 : agg_slot_count,
                                               co,
                                               cgen_state_->module_,
                                               cgen_state_->context_);
  CHECK(cgen_state_->row_func_);
  cgen_state_->row_func_bb_ =
      llvm::BasicBlock::Create(cgen_state_->context_, "entry", cgen_state_->row_func_);

  if (g_enable_filter_function) {
    auto filter_func_ft =
        llvm::FunctionType::get(get_int_type(32, cgen_state_->context_), {}, false);
    cgen_state_->filter_func_ = llvm::Function::Create(filter_func_ft,
                                                       llvm::Function::ExternalLinkage,
                                                       "filter_func",
                                                       cgen_state_->module_);
    CHECK(cgen_state_->filter_func_);
    cgen_state_->filter_func_bb_ = llvm::BasicBlock::Create(
        cgen_state_->context_, "entry", cgen_state_->filter_func_);
  }

  cgen_state_->current_func_ = cgen_state_->row_func_;
  cgen_state_->ir_builder_.SetInsertPoint(cgen_state_->row_func_bb_);

  preloadFragOffsets(ra_exe_unit.input_descs, query_infos);
  RelAlgExecutionUnit body_execution_unit = ra_exe_unit;
  const auto join_loops =
      buildJoinLoops(body_execution_unit, co, eo, query_infos, column_cache);

  plan_state_->allocateLocalColumnIds(ra_exe_unit.input_col_descs);
  const auto is_not_deleted_bb = codegenSkipDeletedOuterTableRow(ra_exe_unit, co);
  if (is_not_deleted_bb) {
    cgen_state_->row_func_bb_ = is_not_deleted_bb;
  }
  if (!join_loops.empty()) {
    codegenJoinLoops(join_loops,
                     body_execution_unit,
                     group_by_and_aggregate,
                     query_func,
                     cgen_state_->row_func_bb_,
                     *(query_mem_desc.get()),
                     co,
                     eo);
  } else {
    const bool can_return_error = compileBody(
        ra_exe_unit, group_by_and_aggregate, *query_mem_desc, co, gpu_smem_context);
    if (can_return_error || cgen_state_->needs_error_check_ || eo.with_dynamic_watchdog ||
        eo.allow_runtime_query_interrupt) {
      createErrorCheckControlFlow(query_func,
                                  eo.with_dynamic_watchdog,
                                  eo.allow_runtime_query_interrupt,
                                  co.device_type,
                                  group_by_and_aggregate.query_infos_);
    }
  }
  std::vector<llvm::Value*> hoisted_literals;

  if (co.hoist_literals) {
    VLOG(1) << "number of hoisted literals: "
            << cgen_state_->query_func_literal_loads_.size()
            << " / literal buffer usage: " << cgen_state_->getLiteralBufferUsage(0)
            << " bytes";
  }

  if (co.hoist_literals && !cgen_state_->query_func_literal_loads_.empty()) {
    // we have some hoisted literals...
    hoisted_literals = inlineHoistedLiterals();
  }

  // replace the row func placeholder call with the call to the actual row func
  std::vector<llvm::Value*> row_func_args;
  for (size_t i = 0; i < cgen_state_->row_func_call_->getNumArgOperands(); ++i) {
    row_func_args.push_back(cgen_state_->row_func_call_->getArgOperand(i));
  }
  row_func_args.insert(row_func_args.end(), col_heads.begin(), col_heads.end());
  row_func_args.push_back(get_arg_by_name(query_func, "join_hash_tables"));
  // push hoisted literals arguments, if any
  row_func_args.insert(
      row_func_args.end(), hoisted_literals.begin(), hoisted_literals.end());
  llvm::ReplaceInstWithInst(
      cgen_state_->row_func_call_,
      llvm::CallInst::Create(cgen_state_->row_func_, row_func_args, ""));

  // replace the filter func placeholder call with the call to the actual filter func
  if (cgen_state_->filter_func_) {
    std::vector<llvm::Value*> filter_func_args;
    for (auto arg_it = cgen_state_->filter_func_args_.begin();
         arg_it != cgen_state_->filter_func_args_.end();
         ++arg_it) {
      filter_func_args.push_back(arg_it->first);
    }
    llvm::ReplaceInstWithInst(
        cgen_state_->filter_func_call_,
        llvm::CallInst::Create(cgen_state_->filter_func_, filter_func_args, ""));
  }

  // Aggregate
  plan_state_->init_agg_vals_ =
      init_agg_val_vec(ra_exe_unit.target_exprs, ra_exe_unit.quals, *query_mem_desc);

  /*
   * If we have decided to use GPU shared memory (decision is not made here), then
   * we generate proper code for extra components that it needs (buffer initialization and
   * gpu reduction from shared memory to global memory). We then replace these functions
   * into the already compiled query_func (replacing two placeholders, write_back_nop and
   * init_smem_nop). The rest of the code should be as before (row_func, etc.).
   */
  if (gpu_smem_context.isSharedMemoryUsed()) {
    if (query_mem_desc->getQueryDescriptionType() ==
        QueryDescriptionType::GroupByPerfectHash) {
      GpuSharedMemCodeBuilder gpu_smem_code(
          cgen_state_->module_,
          cgen_state_->context_,
          *query_mem_desc,
          target_exprs_to_infos(ra_exe_unit.target_exprs, *query_mem_desc),
          plan_state_->init_agg_vals_);
      gpu_smem_code.codegen();
      gpu_smem_code.injectFunctionsInto(query_func);

      // helper functions are used for caching purposes later
      cgen_state_->helper_functions_.push_back(gpu_smem_code.getReductionFunction());
      cgen_state_->helper_functions_.push_back(gpu_smem_code.getInitFunction());
      LOG(IR) << gpu_smem_code.toString();
    }
  }

  auto multifrag_query_func = cgen_state_->module_->getFunction(
      "multifrag_query" + std::string(co.hoist_literals ? "_hoisted_literals" : ""));
  CHECK(multifrag_query_func);

  if (co.device_type == ExecutorDeviceType::GPU && eo.allow_multifrag) {
    insertErrorCodeChecker(
        multifrag_query_func, co.hoist_literals, eo.allow_runtime_query_interrupt);
  }

  bind_query(query_func,
             "query_stub" + std::string(co.hoist_literals ? "_hoisted_literals" : ""),
             multifrag_query_func,
             cgen_state_->module_);

  std::vector<llvm::Function*> root_funcs{query_func, cgen_state_->row_func_};
  if (cgen_state_->filter_func_) {
    root_funcs.push_back(cgen_state_->filter_func_);
  }
  auto live_funcs = CodeGenerator::markDeadRuntimeFuncs(
      *cgen_state_->module_, root_funcs, {multifrag_query_func});

  // Always inline the row function and the filter function.
  // We don't want register spills in the inner loops.
  // LLVM seems to correctly free up alloca instructions
  // in these functions even when they are inlined.
  mark_function_always_inline(cgen_state_->row_func_);
  if (cgen_state_->filter_func_) {
    mark_function_always_inline(cgen_state_->filter_func_);
  }

#ifndef NDEBUG
  // Add helpful metadata to the LLVM IR for debugging.
  AUTOMATIC_IR_METADATA_DONE();
#endif

  // Serialize the important LLVM IR functions to text for SQL EXPLAIN.
  std::string llvm_ir;
  if (eo.just_explain) {
    if (co.explain_type == ExecutorExplainType::Optimized) {
#ifdef WITH_JIT_DEBUG
      throw std::runtime_error(
          "Explain optimized not available when JIT runtime debug symbols are enabled");
#else
      // Note that we don't run the NVVM reflect pass here. Use LOG(IR) to get the
      // optimized IR after NVVM reflect
      llvm::legacy::PassManager pass_manager;
      optimize_ir(query_func, cgen_state_->module_, pass_manager, live_funcs, co);
#endif  // WITH_JIT_DEBUG
    }
    llvm_ir =
        serialize_llvm_object(multifrag_query_func) + serialize_llvm_object(query_func) +
        serialize_llvm_object(cgen_state_->row_func_) +
        (cgen_state_->filter_func_ ? serialize_llvm_object(cgen_state_->filter_func_)
                                   : "");

#ifndef NDEBUG
    llvm_ir += serialize_llvm_metadata_footnotes(query_func, cgen_state_.get());
#endif
  }

  LOG(IR) << "\n\n" << query_mem_desc->toString() << "\n";
  LOG(IR) << "IR for the "
          << (co.device_type == ExecutorDeviceType::CPU ? "CPU:\n" : "GPU:\n");
#ifdef NDEBUG
  LOG(IR) << serialize_llvm_object(query_func)
          << serialize_llvm_object(cgen_state_->row_func_)
          << (cgen_state_->filter_func_ ? serialize_llvm_object(cgen_state_->filter_func_)
                                        : "")
          << "\nEnd of IR";
#else
  LOG(IR) << serialize_llvm_object(cgen_state_->module_) << "\nEnd of IR";
#endif

  // Run some basic validation checks on the LLVM IR before code is generated below.
  verify_function_ir(cgen_state_->row_func_);
  if (cgen_state_->filter_func_) {
    verify_function_ir(cgen_state_->filter_func_);
  }

  // Generate final native code from the LLVM IR.
  std::shared_ptr<CompilationContext> compilation_context;
  switch (co.device_type) {
    case ExecutorDeviceType::CPU:
      compilation_context =
          optimizeAndCodegenCPU(query_func, multifrag_query_func, live_funcs, co);
      break;
    case ExecutorDeviceType::GPU:
      compilation_context = optimizeAndCodegenGPU(query_func,
                                                  multifrag_query_func,
                                                  live_funcs,
                                                  is_group_by || ra_exe_unit.estimator,
                                                  cuda_mgr,
                                                  co);
      break;
    case ExecutorDeviceType::L0:
      compilation_context = optimizeAndCodegenL0(
          query_func, multifrag_query_func, live_funcs, false, l0_mgr, co);
      break;

    default:
      LOG(FATAL) << "Invalid device type";
      return {};
  }
  return std::make_tuple(
      CompilationResult{compilation_context,
          cgen_state_->getLiterals(),
          output_columnar,
          llvm_ir,
          std::move(gpu_smem_context)},
      std::move(query_mem_desc));
}

void Executor::insertErrorCodeChecker(llvm::Function* query_func,
                                      bool hoist_literals,
                                      bool allow_runtime_query_interrupt) {
  auto query_stub_func_name =
      "query_stub" + std::string(hoist_literals ? "_hoisted_literals" : "");
  for (auto bb_it = query_func->begin(); bb_it != query_func->end(); ++bb_it) {
    for (auto inst_it = bb_it->begin(); inst_it != bb_it->end(); ++inst_it) {
      if (!llvm::isa<llvm::CallInst>(*inst_it)) {
        continue;
      }
      auto& row_func_call = llvm::cast<llvm::CallInst>(*inst_it);
      if (std::string(row_func_call.getCalledFunction()->getName()) ==
          query_stub_func_name) {
        auto next_inst_it = inst_it;
        ++next_inst_it;
        auto new_bb = bb_it->splitBasicBlock(next_inst_it);
        auto& br_instr = bb_it->back();
        llvm::IRBuilder<> ir_builder(&br_instr);
        llvm::Value* err_lv = &*inst_it;
        auto error_check_bb =
            bb_it->splitBasicBlock(llvm::BasicBlock::iterator(br_instr), ".error_check");
        llvm::Value* error_code_arg = nullptr;
        auto arg_cnt = 0;
        for (auto arg_it = query_func->arg_begin(); arg_it != query_func->arg_end();
             arg_it++, ++arg_cnt) {
          // since multi_frag_* func has anonymous arguments so we use arg_offset
          // explicitly to capture "error_code" argument in the func's argument list
          if (hoist_literals) {
            if (arg_cnt == 9) {
              error_code_arg = &*arg_it;
              break;
            }
          } else {
            if (arg_cnt == 8) {
              error_code_arg = &*arg_it;
              break;
            }
          }
        }
        CHECK(error_code_arg);
        llvm::Value* err_code = nullptr;
        if (allow_runtime_query_interrupt) {
          // decide the final error code with a consideration of interrupt status
          auto& check_interrupt_br_instr = bb_it->back();
          auto interrupt_check_bb = llvm::BasicBlock::Create(
              cgen_state_->context_, ".interrupt_check", query_func, error_check_bb);
          llvm::IRBuilder<> interrupt_checker_ir_builder(interrupt_check_bb);
          auto detected_interrupt = interrupt_checker_ir_builder.CreateCall(
              cgen_state_->module_->getFunction("check_interrupt"), {});
          auto detected_error = interrupt_checker_ir_builder.CreateCall(
              cgen_state_->module_->getFunction("get_error_code"),
              std::vector<llvm::Value*>{error_code_arg});
          err_code = interrupt_checker_ir_builder.CreateSelect(
              detected_interrupt,
              cgen_state_->llInt(Executor::ERR_INTERRUPTED),
              detected_error);
          interrupt_checker_ir_builder.CreateBr(error_check_bb);
          llvm::ReplaceInstWithInst(&check_interrupt_br_instr,
                                    llvm::BranchInst::Create(interrupt_check_bb));
          ir_builder.SetInsertPoint(&br_instr);
        } else {
          // uses error code returned from row_func and skip to check interrupt status
          ir_builder.SetInsertPoint(&br_instr);
          err_code =
              ir_builder.CreateCall(cgen_state_->module_->getFunction("get_error_code"),
                                    std::vector<llvm::Value*>{error_code_arg});
        }
        err_lv = ir_builder.CreateICmp(
            llvm::ICmpInst::ICMP_NE, err_code, cgen_state_->llInt(0));
        auto error_bb = llvm::BasicBlock::Create(
            cgen_state_->context_, ".error_exit", query_func, new_bb);
        llvm::CallInst::Create(cgen_state_->module_->getFunction("record_error_code"),
                               std::vector<llvm::Value*>{err_code, error_code_arg},
                               "",
                               error_bb);
        llvm::ReturnInst::Create(cgen_state_->context_, error_bb);
        llvm::ReplaceInstWithInst(&br_instr,
                                  llvm::BranchInst::Create(error_bb, new_bb, err_lv));
        break;
      }
    }
  }
}

llvm::BasicBlock* Executor::codegenSkipDeletedOuterTableRow(
    const RelAlgExecutionUnit& ra_exe_unit,
    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  if (!co.filter_on_deleted_column) {
    return nullptr;
  }
  CHECK(!ra_exe_unit.input_descs.empty());
  const auto& outer_input_desc = ra_exe_unit.input_descs[0];
  if (outer_input_desc.getSourceType() != InputSourceType::TABLE) {
    return nullptr;
  }
  const auto deleted_cd =
      plan_state_->getDeletedColForTable(outer_input_desc.getTableId());
  if (!deleted_cd) {
    return nullptr;
  }
  CHECK(deleted_cd->columnType.is_boolean());
  const auto deleted_expr =
      makeExpr<Analyzer::ColumnVar>(deleted_cd->columnType,
                                    outer_input_desc.getTableId(),
                                    deleted_cd->columnId,
                                    outer_input_desc.getNestLevel());
  CodeGenerator code_generator(this);
  const auto is_deleted =
      code_generator.toBool(code_generator.codegen(deleted_expr.get(), true, co).front());
  const auto is_deleted_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "is_deleted", cgen_state_->row_func_);
  llvm::BasicBlock* bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "is_not_deleted", cgen_state_->row_func_);
  cgen_state_->ir_builder_.CreateCondBr(is_deleted, is_deleted_bb, bb);
  cgen_state_->ir_builder_.SetInsertPoint(is_deleted_bb);
  cgen_state_->ir_builder_.CreateRet(cgen_state_->llInt<int32_t>(0));
  cgen_state_->ir_builder_.SetInsertPoint(bb);
  return bb;
}

bool Executor::compileBody(const RelAlgExecutionUnit& ra_exe_unit,
                           GroupByAndAggregate& group_by_and_aggregate,
                           const QueryMemoryDescriptor& query_mem_desc,
                           const CompilationOptions& co,
                           const GpuSharedMemoryContext& gpu_smem_context) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());

  // Switch the code generation into a separate filter function if enabled.
  // Note that accesses to function arguments are still codegenned from the
  // row function's arguments, then later automatically forwarded and
  // remapped into filter function arguments by redeclareFilterFunction().
  cgen_state_->row_func_bb_ = cgen_state_->ir_builder_.GetInsertBlock();
  llvm::Value* loop_done{nullptr};
  std::unique_ptr<Executor::FetchCacheAnchor> fetch_cache_anchor;
  if (cgen_state_->filter_func_) {
    if (cgen_state_->row_func_bb_->getName() == "loop_body") {
      auto row_func_entry_bb = &cgen_state_->row_func_->getEntryBlock();
      cgen_state_->ir_builder_.SetInsertPoint(row_func_entry_bb,
                                              row_func_entry_bb->begin());
      loop_done = cgen_state_->ir_builder_.CreateAlloca(
          get_int_type(1, cgen_state_->context_), nullptr, "loop_done");
      cgen_state_->ir_builder_.SetInsertPoint(cgen_state_->row_func_bb_);
      cgen_state_->ir_builder_.CreateStore(cgen_state_->llBool(true), loop_done);
    }
    cgen_state_->ir_builder_.SetInsertPoint(cgen_state_->filter_func_bb_);
    cgen_state_->current_func_ = cgen_state_->filter_func_;
    fetch_cache_anchor = std::make_unique<Executor::FetchCacheAnchor>(cgen_state_.get());
  }

  // generate the code for the filter
  std::vector<Analyzer::Expr*> primary_quals;
  std::vector<Analyzer::Expr*> deferred_quals;
  bool short_circuited = CodeGenerator::prioritizeQuals(
      ra_exe_unit, primary_quals, deferred_quals, plan_state_->hoisted_filters_);
  if (short_circuited) {
    VLOG(1) << "Prioritized " << std::to_string(primary_quals.size()) << " quals, "
            << "short-circuited and deferred " << std::to_string(deferred_quals.size())
            << " quals";
  }
  llvm::Value* filter_lv = cgen_state_->llBool(true);
  CodeGenerator code_generator(this);
  for (auto expr : primary_quals) {
    // Generate the filter for primary quals
    auto cond = code_generator.toBool(code_generator.codegen(expr, true, co).front());
    filter_lv = cgen_state_->ir_builder_.CreateAnd(filter_lv, cond);
  }
  CHECK(filter_lv->getType()->isIntegerTy(1));
  llvm::BasicBlock* sc_false{nullptr};
  if (!deferred_quals.empty()) {
    auto sc_true = llvm::BasicBlock::Create(
        cgen_state_->context_, "sc_true", cgen_state_->current_func_);
    sc_false = llvm::BasicBlock::Create(
        cgen_state_->context_, "sc_false", cgen_state_->current_func_);
    cgen_state_->ir_builder_.CreateCondBr(filter_lv, sc_true, sc_false);
    cgen_state_->ir_builder_.SetInsertPoint(sc_false);
    if (ra_exe_unit.join_quals.empty()) {
      cgen_state_->ir_builder_.CreateRet(cgen_state_->llInt(int32_t(0)));
    }
    cgen_state_->ir_builder_.SetInsertPoint(sc_true);
    filter_lv = cgen_state_->llBool(true);
  }
  for (auto expr : deferred_quals) {
    filter_lv = cgen_state_->ir_builder_.CreateAnd(
        filter_lv, code_generator.toBool(code_generator.codegen(expr, true, co).front()));
  }

  CHECK(filter_lv->getType()->isIntegerTy(1));
  auto ret = group_by_and_aggregate.codegen(
      filter_lv, sc_false, query_mem_desc, co, gpu_smem_context);

  // Switch the code generation back to the row function if a filter
  // function was enabled.
  if (cgen_state_->filter_func_) {
    if (cgen_state_->row_func_bb_->getName() == "loop_body") {
      cgen_state_->ir_builder_.CreateStore(cgen_state_->llBool(false), loop_done);
      cgen_state_->ir_builder_.CreateRet(cgen_state_->llInt<int32_t>(0));
    }

    cgen_state_->ir_builder_.SetInsertPoint(cgen_state_->row_func_bb_);
    cgen_state_->current_func_ = cgen_state_->row_func_;
    cgen_state_->filter_func_call_ =
        cgen_state_->ir_builder_.CreateCall(cgen_state_->filter_func_, {});

    // Create real filter function declaration after placeholder call
    // is emitted.
    redeclareFilterFunction();

    if (cgen_state_->row_func_bb_->getName() == "loop_body") {
      auto loop_done_true = llvm::BasicBlock::Create(
          cgen_state_->context_, "loop_done_true", cgen_state_->row_func_);
      auto loop_done_false = llvm::BasicBlock::Create(
          cgen_state_->context_, "loop_done_false", cgen_state_->row_func_);
      auto loop_done_flag = cgen_state_->ir_builder_.CreateLoad(loop_done);
      cgen_state_->ir_builder_.CreateCondBr(
          loop_done_flag, loop_done_true, loop_done_false);
      cgen_state_->ir_builder_.SetInsertPoint(loop_done_true);
      cgen_state_->ir_builder_.CreateRet(cgen_state_->filter_func_call_);
      cgen_state_->ir_builder_.SetInsertPoint(loop_done_false);
    } else {
      cgen_state_->ir_builder_.CreateRet(cgen_state_->filter_func_call_);
    }
  }
  return ret;
}

std::unique_ptr<llvm::Module> runtime_module_shallow_copy(CgenState* cgen_state) {
  return llvm::CloneModule(
      *g_rt_module.get(), cgen_state->vmap_, [](const llvm::GlobalValue* gv) {
        auto func = llvm::dyn_cast<llvm::Function>(gv);
        if (!func) {
          return true;
        }
        return (func->getLinkage() == llvm::GlobalValue::LinkageTypes::PrivateLinkage ||
                func->getLinkage() == llvm::GlobalValue::LinkageTypes::InternalLinkage);
      });
}

std::vector<llvm::Value*> generate_column_heads_load(const int num_columns,
                                                     llvm::Value* byte_stream_arg,
                                                     llvm::IRBuilder<>& ir_builder,
                                                     llvm::LLVMContext& ctx) {
  CHECK(byte_stream_arg);
  const auto max_col_local_id = num_columns - 1;

  std::vector<llvm::Value*> col_heads;
  for (int col_id = 0; col_id <= max_col_local_id; ++col_id) {
    col_heads.emplace_back(ir_builder.CreateLoad(ir_builder.CreateGEP(
        byte_stream_arg, llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), col_id))));
  }
  return col_heads;
}
