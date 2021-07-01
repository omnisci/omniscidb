/*
 * Copyright 2019 OmniSci, Inc.
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

#include <gtest/gtest.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_os_ostream.h>

#include "Analyzer/Analyzer.h"
#include "QueryEngine/CodeGenerator.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/IRCodegenUtils.h"
#include "QueryEngine/LLVMGlobalContext.h"
#include "TestHelpers.h"

TEST(CodeGeneratorTest, IntegerConstant) {
  auto& ctx = getGlobalLLVMContext();
  std::unique_ptr<llvm::Module> module(read_template_module(ctx));
  ScalarCodeGenerator code_generator(std::move(module));
  CompilationOptions co = CompilationOptions::defaults(ExecutorDeviceType::CPU);
  co.hoist_literals = false;

  Datum d;
  d.intval = 42;
  auto constant = makeExpr<Analyzer::Constant>(kINT, false, d);
  const auto compiled_expr = code_generator.compile(constant.get(), true, co);
  verify_function_ir(compiled_expr.func);
  ASSERT_TRUE(compiled_expr.inputs.empty());

  using FuncPtr = int (*)(int*);
  auto func_ptr = reinterpret_cast<FuncPtr>(
      code_generator.generateNativeCode(compiled_expr, co).front());
  CHECK(func_ptr);
  int out;
  int err = func_ptr(&out);
  ASSERT_EQ(err, 0);
  ASSERT_EQ(out, d.intval);
}

TEST(CodeGeneratorTest, IntegerAdd) {
  auto& ctx = getGlobalLLVMContext();
  std::unique_ptr<llvm::Module> module(read_template_module(ctx));
  ScalarCodeGenerator code_generator(std::move(module));
  CompilationOptions co = CompilationOptions::defaults(ExecutorDeviceType::CPU);
  co.hoist_literals = false;

  Datum d;
  d.intval = 42;
  auto lhs = makeExpr<Analyzer::Constant>(kINT, false, d);
  auto rhs = makeExpr<Analyzer::Constant>(kINT, false, d);
  auto plus = makeExpr<Analyzer::BinOper>(kINT, kPLUS, kONE, lhs, rhs);
  const auto compiled_expr = code_generator.compile(plus.get(), true, co);
  verify_function_ir(compiled_expr.func);
  ASSERT_TRUE(compiled_expr.inputs.empty());

  using FuncPtr = int (*)(int*);
  auto func_ptr = reinterpret_cast<FuncPtr>(
      code_generator.generateNativeCode(compiled_expr, co).front());
  CHECK(func_ptr);
  int out;
  int err = func_ptr(&out);
  ASSERT_EQ(err, 0);
  ASSERT_EQ(out, d.intval + d.intval);
}

TEST(CodeGeneratorTest, IntegerColumn) {
  auto& ctx = getGlobalLLVMContext();
  std::unique_ptr<llvm::Module> module(read_template_module(ctx));
  ScalarCodeGenerator code_generator(std::move(module));
  CompilationOptions co = CompilationOptions::defaults(ExecutorDeviceType::CPU);
  co.hoist_literals = false;

  SQLTypeInfo ti(kINT, false);
  int table_id = 1;
  int column_id = 5;
  int rte_idx = 0;
  auto col = makeExpr<Analyzer::ColumnVar>(ti, table_id, column_id, rte_idx);
  const auto compiled_expr = code_generator.compile(col.get(), true, co);
  verify_function_ir(compiled_expr.func);
  ASSERT_EQ(compiled_expr.inputs.size(), size_t(1));
  ASSERT_TRUE(*compiled_expr.inputs.front() == *col);

  using FuncPtr = int (*)(int*, int);
  auto func_ptr = reinterpret_cast<FuncPtr>(
      code_generator.generateNativeCode(compiled_expr, co).front());
  CHECK(func_ptr);
  int out;
  int err = func_ptr(&out, 17);
  ASSERT_EQ(err, 0);
  ASSERT_EQ(out, 17);
}

TEST(CodeGeneratorTest, IntegerExpr) {
  auto& ctx = getGlobalLLVMContext();
  std::unique_ptr<llvm::Module> module(read_template_module(ctx));
  ScalarCodeGenerator code_generator(std::move(module));
  CompilationOptions co = CompilationOptions::defaults(ExecutorDeviceType::CPU);
  co.hoist_literals = false;

  SQLTypeInfo ti(kINT, false);
  int table_id = 1;
  int column_id = 5;
  int rte_idx = 0;
  auto lhs = makeExpr<Analyzer::ColumnVar>(ti, table_id, column_id, rte_idx);
  Datum d;
  d.intval = 42;
  auto rhs = makeExpr<Analyzer::Constant>(kINT, false, d);
  auto plus = makeExpr<Analyzer::BinOper>(kINT, kPLUS, kONE, lhs, rhs);
  const auto compiled_expr = code_generator.compile(plus.get(), true, co);
  verify_function_ir(compiled_expr.func);
  ASSERT_EQ(compiled_expr.inputs.size(), size_t(1));
  ASSERT_TRUE(*compiled_expr.inputs.front() == *lhs);

  using FuncPtr = int (*)(int*, int);
  auto func_ptr = reinterpret_cast<FuncPtr>(
      code_generator.generateNativeCode(compiled_expr, co).front());
  CHECK(func_ptr);
  int out;
  int err = func_ptr(&out, 58);
  ASSERT_EQ(err, 0);
  ASSERT_EQ(out, 100);
}

#ifdef HAVE_CUDA
void free_param_pointers(const std::vector<void*>& param_ptrs,
                         CudaMgr_Namespace::CudaMgr* cuda_mgr) {
  for (const auto param_ptr : param_ptrs) {
    const auto device_ptr =
        reinterpret_cast<int8_t*>(*reinterpret_cast<CUdeviceptr*>(param_ptr));
    cuda_mgr->freeDeviceMem(device_ptr);
  }
}

TEST(CodeGeneratorTest, IntegerConstantGPU) {
  auto& ctx = getGlobalLLVMContext();
  std::unique_ptr<llvm::Module> module(read_template_module(ctx));
  ScalarCodeGenerator code_generator(std::move(module));
  CompilationOptions co = CompilationOptions::defaults(ExecutorDeviceType::GPU);
  co.hoist_literals = false;

  Datum d;
  d.intval = 42;
  auto constant = makeExpr<Analyzer::Constant>(kINT, false, d);
  const auto compiled_expr = code_generator.compile(constant.get(), true, co);
  verify_function_ir(compiled_expr.func);
  ASSERT_TRUE(compiled_expr.inputs.empty());

  const auto native_function_pointers =
      code_generator.generateNativeCode(compiled_expr, co);

  for (size_t gpu_idx = 0; gpu_idx < native_function_pointers.size(); ++gpu_idx) {
    const auto native_function_pointer = native_function_pointers[gpu_idx];
    auto func_ptr = reinterpret_cast<CUfunction>(native_function_pointer);

    std::vector<void*> param_ptrs;
    CUdeviceptr err = reinterpret_cast<CUdeviceptr>(
        code_generator.getCudaMgr()->allocateDeviceMem(4, gpu_idx));
    CUdeviceptr out = reinterpret_cast<CUdeviceptr>(
        code_generator.getCudaMgr()->allocateDeviceMem(4, gpu_idx));
    param_ptrs.push_back(&err);
    param_ptrs.push_back(&out);
    cuLaunchKernel(func_ptr, 1, 1, 1, 1, 1, 1, 0, nullptr, &param_ptrs[0], nullptr);
    int32_t host_err;
    int32_t host_out;
    code_generator.getCudaMgr()->copyDeviceToHost(reinterpret_cast<int8_t*>(&host_err),
                                                  reinterpret_cast<const int8_t*>(err),
                                                  4,
                                                  gpu_idx);
    code_generator.getCudaMgr()->copyDeviceToHost(reinterpret_cast<int8_t*>(&host_out),
                                                  reinterpret_cast<const int8_t*>(out),
                                                  4,
                                                  gpu_idx);

    ASSERT_EQ(host_err, 0);
    ASSERT_EQ(host_out, d.intval);
    free_param_pointers(param_ptrs, code_generator.getCudaMgr());
  }
}

TEST(CodeGeneratorTest, IntegerAddGPU) {
  auto& ctx = getGlobalLLVMContext();
  std::unique_ptr<llvm::Module> module(read_template_module(ctx));
  ScalarCodeGenerator code_generator(std::move(module));
  CompilationOptions co = CompilationOptions::defaults(ExecutorDeviceType::GPU);
  co.hoist_literals = false;

  Datum d;
  d.intval = 42;
  auto lhs = makeExpr<Analyzer::Constant>(kINT, false, d);
  auto rhs = makeExpr<Analyzer::Constant>(kINT, false, d);
  auto plus = makeExpr<Analyzer::BinOper>(kINT, kPLUS, kONE, lhs, rhs);
  const auto compiled_expr = code_generator.compile(plus.get(), true, co);
  verify_function_ir(compiled_expr.func);
  ASSERT_TRUE(compiled_expr.inputs.empty());

  const auto native_function_pointers =
      code_generator.generateNativeCode(compiled_expr, co);

  for (size_t gpu_idx = 0; gpu_idx < native_function_pointers.size(); ++gpu_idx) {
    const auto native_function_pointer = native_function_pointers[gpu_idx];
    auto func_ptr = reinterpret_cast<CUfunction>(native_function_pointer);

    std::vector<void*> param_ptrs;
    CUdeviceptr err = reinterpret_cast<CUdeviceptr>(
        code_generator.getCudaMgr()->allocateDeviceMem(4, gpu_idx));
    CUdeviceptr out = reinterpret_cast<CUdeviceptr>(
        code_generator.getCudaMgr()->allocateDeviceMem(4, gpu_idx));
    param_ptrs.push_back(&err);
    param_ptrs.push_back(&out);
    cuLaunchKernel(func_ptr, 1, 1, 1, 1, 1, 1, 0, nullptr, &param_ptrs[0], nullptr);
    int32_t host_err;
    int32_t host_out;
    code_generator.getCudaMgr()->copyDeviceToHost(reinterpret_cast<int8_t*>(&host_err),
                                                  reinterpret_cast<const int8_t*>(err),
                                                  4,
                                                  gpu_idx);
    code_generator.getCudaMgr()->copyDeviceToHost(reinterpret_cast<int8_t*>(&host_out),
                                                  reinterpret_cast<const int8_t*>(out),
                                                  4,
                                                  gpu_idx);

    ASSERT_EQ(host_err, 0);
    ASSERT_EQ(host_out, d.intval + d.intval);
    free_param_pointers(param_ptrs, code_generator.getCudaMgr());
  }
}

TEST(CodeGeneratorTest, IntegerColumnGPU) {
  auto& ctx = getGlobalLLVMContext();
  std::unique_ptr<llvm::Module> module(read_template_module(ctx));
  ScalarCodeGenerator code_generator(std::move(module));
  CompilationOptions co = CompilationOptions::defaults(ExecutorDeviceType::GPU);
  co.hoist_literals = false;

  SQLTypeInfo ti(kINT, false);
  int table_id = 1;
  int column_id = 5;
  int rte_idx = 0;
  auto col = makeExpr<Analyzer::ColumnVar>(ti, table_id, column_id, rte_idx);
  const auto compiled_expr = code_generator.compile(col.get(), true, co);
  verify_function_ir(compiled_expr.func);
  ASSERT_EQ(compiled_expr.inputs.size(), size_t(1));
  ASSERT_TRUE(*compiled_expr.inputs.front() == *col);

  const auto native_function_pointers =
      code_generator.generateNativeCode(compiled_expr, co);

  for (size_t gpu_idx = 0; gpu_idx < native_function_pointers.size(); ++gpu_idx) {
    const auto native_function_pointer = native_function_pointers[gpu_idx];
    auto func_ptr = reinterpret_cast<CUfunction>(native_function_pointer);

    std::vector<void*> param_ptrs;
    CUdeviceptr err = reinterpret_cast<CUdeviceptr>(
        code_generator.getCudaMgr()->allocateDeviceMem(4, gpu_idx));
    CUdeviceptr out = reinterpret_cast<CUdeviceptr>(
        code_generator.getCudaMgr()->allocateDeviceMem(4, gpu_idx));
    CUdeviceptr in = reinterpret_cast<CUdeviceptr>(
        code_generator.getCudaMgr()->allocateDeviceMem(4, gpu_idx));
    int host_in = 17;
    code_generator.getCudaMgr()->copyHostToDevice(
        reinterpret_cast<int8_t*>(in),
        reinterpret_cast<const int8_t*>(&host_in),
        4,
        gpu_idx);
    param_ptrs.push_back(&err);
    param_ptrs.push_back(&out);
    param_ptrs.push_back(&in);
    cuLaunchKernel(func_ptr, 1, 1, 1, 1, 1, 1, 0, nullptr, &param_ptrs[0], nullptr);
    int32_t host_err;
    int32_t host_out;
    code_generator.getCudaMgr()->copyDeviceToHost(reinterpret_cast<int8_t*>(&host_err),
                                                  reinterpret_cast<const int8_t*>(err),
                                                  4,
                                                  gpu_idx);
    code_generator.getCudaMgr()->copyDeviceToHost(reinterpret_cast<int8_t*>(&host_out),
                                                  reinterpret_cast<const int8_t*>(out),
                                                  4,
                                                  gpu_idx);

    ASSERT_EQ(host_err, 0);
    ASSERT_EQ(host_out, 17);
    free_param_pointers(param_ptrs, code_generator.getCudaMgr());
  }
}

TEST(CodeGeneratorTest, IntegerExprGPU) {
  auto& ctx = getGlobalLLVMContext();
  std::unique_ptr<llvm::Module> module(read_template_module(ctx));
  ScalarCodeGenerator code_generator(std::move(module));
  CompilationOptions co = CompilationOptions::defaults(ExecutorDeviceType::GPU);
  co.hoist_literals = false;

  SQLTypeInfo ti(kINT, false);
  int table_id = 1;
  int column_id = 5;
  int rte_idx = 0;
  auto lhs = makeExpr<Analyzer::ColumnVar>(ti, table_id, column_id, rte_idx);
  Datum d;
  d.intval = 42;
  auto rhs = makeExpr<Analyzer::Constant>(kINT, false, d);
  auto plus = makeExpr<Analyzer::BinOper>(kINT, kPLUS, kONE, lhs, rhs);
  const auto compiled_expr = code_generator.compile(plus.get(), true, co);
  verify_function_ir(compiled_expr.func);
  ASSERT_EQ(compiled_expr.inputs.size(), size_t(1));
  ASSERT_TRUE(*compiled_expr.inputs.front() == *lhs);

  const auto native_function_pointers =
      code_generator.generateNativeCode(compiled_expr, co);

  for (size_t gpu_idx = 0; gpu_idx < native_function_pointers.size(); ++gpu_idx) {
    const auto native_function_pointer = native_function_pointers[gpu_idx];
    auto func_ptr = reinterpret_cast<CUfunction>(native_function_pointer);

    std::vector<void*> param_ptrs;
    CUdeviceptr err = reinterpret_cast<CUdeviceptr>(
        code_generator.getCudaMgr()->allocateDeviceMem(4, gpu_idx));
    CUdeviceptr out = reinterpret_cast<CUdeviceptr>(
        code_generator.getCudaMgr()->allocateDeviceMem(4, gpu_idx));
    CUdeviceptr in = reinterpret_cast<CUdeviceptr>(
        code_generator.getCudaMgr()->allocateDeviceMem(4, gpu_idx));
    int host_in = 58;
    code_generator.getCudaMgr()->copyHostToDevice(
        reinterpret_cast<int8_t*>(in),
        reinterpret_cast<const int8_t*>(&host_in),
        4,
        gpu_idx);
    param_ptrs.push_back(&err);
    param_ptrs.push_back(&out);
    param_ptrs.push_back(&in);
    cuLaunchKernel(func_ptr, 1, 1, 1, 1, 1, 1, 0, nullptr, &param_ptrs[0], nullptr);
    int32_t host_err;
    int32_t host_out;
    code_generator.getCudaMgr()->copyDeviceToHost(reinterpret_cast<int8_t*>(&host_err),
                                                  reinterpret_cast<const int8_t*>(err),
                                                  4,
                                                  gpu_idx);
    code_generator.getCudaMgr()->copyDeviceToHost(reinterpret_cast<int8_t*>(&host_out),
                                                  reinterpret_cast<const int8_t*>(out),
                                                  4,
                                                  gpu_idx);

    ASSERT_EQ(host_err, 0);
    ASSERT_EQ(host_out, 100);
    free_param_pointers(param_ptrs, code_generator.getCudaMgr());
  }
}
#endif  // HAVE_CUDA
#ifdef HAVE_L0
namespace {
template <typename T, size_t N>
struct alignas(4096) AlignedArray {
  T data[N];
};
}  // namespace

TEST(CodeGeneratorTest, IntegerConstantL0) {
  auto& ctx = getGlobalLLVMContext();
  std::unique_ptr<llvm::Module> module(read_template_module(ctx));
  ScalarCodeGenerator code_generator(std::move(module));
  CompilationOptions co = CompilationOptions::defaults(ExecutorDeviceType::L0);
  co.hoist_literals = false;

  Datum d;
  d.intval = 42;
  auto constant = makeExpr<Analyzer::Constant>(kINT, false, d);
  const auto compiled_expr = code_generator.compile(constant.get(), true, co);
  verify_function_ir(compiled_expr.func);
  ASSERT_TRUE(compiled_expr.inputs.empty());

  const auto l0_kernels = code_generator.generateNativeL0Code(compiled_expr, co);

  ASSERT_EQ(l0_kernels.size(), 1);

  auto mgr = code_generator.getL0Mgr();
  auto driver = mgr->drivers()[0];

  for (size_t gpu_idx = 0; gpu_idx < l0_kernels.size(); ++gpu_idx) {
    const auto kernel = l0_kernels[gpu_idx];

    std::vector<void*> param_ptrs;
    auto device = driver->devices()[gpu_idx];
    auto command_queue = device->command_queue();
    auto command_list = device->create_command_list();

    const int elements = 1;
    AlignedArray<int, elements> in, out;
    in.data[0] = 51;
    out.data[0] = -1;
    const int copy_size = sizeof(int);

    void* in_void = in.data;
    void* out_void = out.data;

    void* dIn = l0::allocate_device_mem(copy_size, *device);
    void* dOut = l0::allocate_device_mem(copy_size, *device);

    command_list->copy(dIn, in_void, copy_size);
    command_list->launch(*kernel, &dIn, &dOut);
    command_list->copy(out_void, dOut, copy_size);
    command_list->submit(*command_queue);

    std::cout << "out" << out.data[0] << std::endl;

    L0_SAFE_CALL(zeMemFree(device->ctx(), dIn));
    L0_SAFE_CALL(zeMemFree(device->ctx(), dOut));

    ASSERT_EQ(out.data[0], d.intval);
  }
}

TEST(CodeGeneratorTest, IntegerAddL0) {
  auto& ctx = getGlobalLLVMContext();
  std::unique_ptr<llvm::Module> module(read_template_module(ctx));
  ScalarCodeGenerator code_generator(std::move(module));
  CompilationOptions co = CompilationOptions::defaults(ExecutorDeviceType::L0);
  co.hoist_literals = false;

  Datum d;
  d.intval = 42;
  auto lhs = makeExpr<Analyzer::Constant>(kINT, false, d);
  auto rhs = makeExpr<Analyzer::Constant>(kINT, false, d);
  auto plus = makeExpr<Analyzer::BinOper>(kINT, kPLUS, kONE, lhs, rhs);
  const auto compiled_expr = code_generator.compile(plus.get(), true, co);
  verify_function_ir(compiled_expr.func);
  ASSERT_TRUE(compiled_expr.inputs.empty());

  const auto l0_kernels = code_generator.generateNativeL0Code(compiled_expr, co);

  ASSERT_EQ(l0_kernels.size(), 1);

  auto mgr = code_generator.getL0Mgr();
  auto driver = mgr->drivers()[0];

  for (size_t gpu_idx = 0; gpu_idx < l0_kernels.size(); ++gpu_idx) {
    const auto kernel = l0_kernels[gpu_idx];

    auto device = driver->devices()[gpu_idx];
    auto command_queue = device->command_queue();
    auto command_list = device->create_command_list();

    const int elements = 1;
    AlignedArray<int, elements> in, out;
    in.data[0] = 51;
    out.data[0] = -1;
    const int copy_size = sizeof(int);

    void* in_void = in.data;
    void* out_void = out.data;

    void* dIn = l0::allocate_device_mem(copy_size, *device);
    void* dOut = l0::allocate_device_mem(copy_size, *device);

    // command_list->copy(dIn, in_void, copy_size);
    command_list->launch(*kernel, &dIn, &dOut);
    command_list->copy(out_void, dOut, copy_size);
    command_list->submit(*command_queue);

    L0_SAFE_CALL(zeMemFree(device->ctx(), dIn));
    L0_SAFE_CALL(zeMemFree(device->ctx(), dOut));

    ASSERT_EQ(out.data[0], d.intval + d.intval);
  }
}

TEST(CodeGenerationTest, IntegerColumnL0) {
  auto& ctx = getGlobalLLVMContext();
  std::unique_ptr<llvm::Module> module(read_template_module(ctx));
  ScalarCodeGenerator code_generator(std::move(module));
  CompilationOptions co = CompilationOptions::defaults(ExecutorDeviceType::L0);
  co.hoist_literals = false;

  SQLTypeInfo ti(kINT, false);
  int table_id = 1;
  int column_id = 5;
  int rte_idx = 0;
  auto col = makeExpr<Analyzer::ColumnVar>(ti, table_id, column_id, rte_idx);
  const auto compiled_expr = code_generator.compile(col.get(), true, co);
  verify_function_ir(compiled_expr.func);
  ASSERT_EQ(compiled_expr.inputs.size(), size_t(1));
  ASSERT_TRUE(*compiled_expr.inputs.front() == *col);

  const auto l0_kernels = code_generator.generateNativeL0Code(compiled_expr, co);

  auto mgr = code_generator.getL0Mgr();
  auto driver = mgr->drivers()[0];

  for (size_t gpu_idx = 0; gpu_idx < l0_kernels.size(); ++gpu_idx) {
    const auto kernel = l0_kernels[gpu_idx];

    auto device = driver->devices()[gpu_idx];
    auto command_queue = device->command_queue();
    auto command_list = device->create_command_list();

    const int elements = 1;
    AlignedArray<int, elements> in, out;
    in.data[0] = 17;
    out.data[0] = -1;
    const int copy_size = sizeof(int);

    void* in_void = in.data;
    void* out_void = out.data;

    void* dIn = l0::allocate_device_mem(copy_size, *device);
    void* dOut = l0::allocate_device_mem(copy_size, *device);

    command_list->copy(dIn, in_void, copy_size);
    command_list->launch(*kernel, nullptr, &dOut, &dIn);
    command_list->copy(out_void, dOut, copy_size);
    command_list->submit(*command_queue);

    L0_SAFE_CALL(zeMemFree(device->ctx(), dIn));
    L0_SAFE_CALL(zeMemFree(device->ctx(), dOut));

    ASSERT_EQ(out.data[0], 17);
  }
}

TEST(CodeGeneratorTest, IntegerExprL0) {
  auto& ctx = getGlobalLLVMContext();
  std::unique_ptr<llvm::Module> module(read_template_module(ctx));
  ScalarCodeGenerator code_generator(std::move(module));
  CompilationOptions co = CompilationOptions::defaults(ExecutorDeviceType::L0);
  co.hoist_literals = false;

  SQLTypeInfo ti(kINT, false);
  int table_id = 1;
  int column_id = 5;
  int rte_idx = 0;
  auto lhs = makeExpr<Analyzer::ColumnVar>(ti, table_id, column_id, rte_idx);
  Datum d;
  d.intval = 42;
  auto rhs = makeExpr<Analyzer::Constant>(kINT, false, d);
  auto plus = makeExpr<Analyzer::BinOper>(kINT, kPLUS, kONE, lhs, rhs);
  const auto compiled_expr = code_generator.compile(plus.get(), true, co);
  verify_function_ir(compiled_expr.func);
  ASSERT_EQ(compiled_expr.inputs.size(), size_t(1));
  ASSERT_TRUE(*compiled_expr.inputs.front() == *lhs);

  const auto l0_kernels = code_generator.generateNativeL0Code(compiled_expr, co);

  auto mgr = code_generator.getL0Mgr();
  auto driver = mgr->drivers()[0];

  for (size_t gpu_idx = 0; gpu_idx < l0_kernels.size(); ++gpu_idx) {
    const auto kernel = l0_kernels[gpu_idx];

    auto device = driver->devices()[gpu_idx];
    auto command_queue = device->command_queue();
    auto command_list = device->create_command_list();

    const int elements = 1;
    AlignedArray<int, elements> in, out;
    in.data[0] = 58;
    out.data[0] = -1;
    const int copy_size = sizeof(int);

    void* in_void = in.data;
    void* out_void = out.data;

    void* dIn = l0::allocate_device_mem(copy_size, *device);
    void* dOut = l0::allocate_device_mem(copy_size, *device);

    command_list->copy(dIn, in_void, copy_size);
    command_list->launch(*kernel, nullptr, &dOut, &dIn);
    command_list->copy(out_void, dOut, copy_size);
    command_list->submit(*command_queue);

    L0_SAFE_CALL(zeMemFree(device->ctx(), dIn));
    L0_SAFE_CALL(zeMemFree(device->ctx(), dOut));

    ASSERT_EQ(out.data[0], 100);
  }
}
#endif  // HAVE_L0

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  int err = RUN_ALL_TESTS();
  return err;
}
