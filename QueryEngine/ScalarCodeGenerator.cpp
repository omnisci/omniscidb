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

#include "CodeGenerator.h"
#include "ScalarExprVisitor.h"

namespace {

class UsedColumnExpressions : public ScalarExprVisitor<ScalarCodeGenerator::ColumnMap> {
 protected:
  ScalarCodeGenerator::ColumnMap visitColumnVar(
      const Analyzer::ColumnVar* column) const override {
    ScalarCodeGenerator::ColumnMap m;
    InputColDescriptor input_desc(
        column->get_column_id(), column->get_table_id(), column->get_rte_idx());
    m.emplace(input_desc,
              std::static_pointer_cast<Analyzer::ColumnVar>(column->deep_copy()));
    return m;
  }

  ScalarCodeGenerator::ColumnMap aggregateResult(
      const ScalarCodeGenerator::ColumnMap& aggregate,
      const ScalarCodeGenerator::ColumnMap& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};

std::vector<InputTableInfo> g_table_infos;

llvm::Type* llvm_type_from_sql(const SQLTypeInfo& ti, llvm::LLVMContext& ctx) {
  switch (ti.get_type()) {
    case kINT: {
      return get_int_type(32, ctx);
    }
    default: {
      LOG(FATAL) << "Unsupported type";
      return nullptr;  // satisfy -Wreturn-type
    }
  }
}

}  // namespace

ScalarCodeGenerator::ColumnMap ScalarCodeGenerator::prepare(const Analyzer::Expr* expr) {
  UsedColumnExpressions visitor;
  const auto used_columns = visitor.visit(expr);
  std::list<std::shared_ptr<const InputColDescriptor>> global_col_ids;
  for (const auto& used_column : used_columns) {
    global_col_ids.push_back(std::make_shared<InputColDescriptor>(
        used_column.first.getColId(),
        used_column.first.getScanDesc().getTableId(),
        used_column.first.getScanDesc().getNestLevel()));
  }
  plan_state_->allocateLocalColumnIds(global_col_ids);
  return used_columns;
}

ScalarCodeGenerator::CompiledExpression ScalarCodeGenerator::compile(
    const Analyzer::Expr* expr,
    const bool fetch_columns,
    const CompilationOptions& co) {
  unsigned int addr_space = (co.device_type == ExecutorDeviceType::L0) ? 1 : 0;
  own_plan_state_ = std::make_unique<PlanState>(
      false, std::vector<InputTableInfo>{}, PlanState::DeletedColumnsMap{}, nullptr);
  plan_state_ = own_plan_state_.get();
  const auto used_columns = prepare(expr);
  std::vector<llvm::Type*> arg_types(plan_state_->global_to_local_col_ids_.size() + 1);
  std::vector<std::shared_ptr<Analyzer::ColumnVar>> inputs(arg_types.size() - 1);
  auto& ctx = module_->getContext();
  for (const auto& kv : plan_state_->global_to_local_col_ids_) {
    size_t arg_idx = kv.second;
    CHECK_LT(arg_idx, arg_types.size());
    const auto it = used_columns.find(kv.first);
    const auto col_expr = it->second;
    inputs[arg_idx] = col_expr;
    const auto& ti = col_expr->get_type_info();
    arg_types[arg_idx + 1] = llvm_type_from_sql(ti, ctx);
  }
  arg_types[0] =
      llvm::PointerType::get(llvm_type_from_sql(expr->get_type_info(), ctx), addr_space);
  auto ft = llvm::FunctionType::get(get_int_type(32, ctx), arg_types, false);
  auto scalar_expr_func = llvm::Function::Create(
      ft, llvm::Function::ExternalLinkage, "scalar_expr", module_.get());
  auto bb_entry = llvm::BasicBlock::Create(ctx, ".entry", scalar_expr_func, 0);
  own_cgen_state_ = std::make_unique<CgenState>(g_table_infos.size(), false);
  own_cgen_state_->module_ = module_.get();
  own_cgen_state_->row_func_ = own_cgen_state_->current_func_ = scalar_expr_func;
  own_cgen_state_->ir_builder_.SetInsertPoint(bb_entry);
  cgen_state_ = own_cgen_state_.get();
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto expr_lvs = codegen(expr, fetch_columns, co);
  CHECK_EQ(expr_lvs.size(), size_t(1));
  cgen_state_->ir_builder_.CreateStore(expr_lvs.front(),
                                       cgen_state_->row_func_->arg_begin());
  cgen_state_->ir_builder_.CreateRet(ll_int<int32_t>(0, ctx));
  if (co.device_type == ExecutorDeviceType::GPU ||
      co.device_type == ExecutorDeviceType::L0) {
    std::vector<llvm::Type*> wrapper_arg_types(arg_types.size() + 1);
    wrapper_arg_types[0] = llvm::PointerType::get(get_int_type(32, ctx), addr_space);
    wrapper_arg_types[1] = arg_types[0];
    for (size_t i = 1; i < arg_types.size(); ++i) {
      wrapper_arg_types[i + 1] = llvm::PointerType::get(arg_types[i], addr_space);
    }
    auto wrapper_ft =
        llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), wrapper_arg_types, false);
    auto wrapper_scalar_expr_func =
        llvm::Function::Create(wrapper_ft,
                               llvm::Function::ExternalLinkage,
                               "wrapper_scalar_expr",
                               module_.get());
    auto wrapper_bb_entry =
        llvm::BasicBlock::Create(ctx, ".entry", wrapper_scalar_expr_func, 0);
    llvm::IRBuilder<> b(ctx);
    b.SetInsertPoint(wrapper_bb_entry);
    std::vector<llvm::Value*> loaded_args = {wrapper_scalar_expr_func->arg_begin() + 1};
    for (size_t i = 2; i < wrapper_arg_types.size(); ++i) {
      loaded_args.push_back(b.CreateLoad(wrapper_scalar_expr_func->arg_begin() + i));
    }
    auto error_lv = b.CreateCall(scalar_expr_func, loaded_args);
    if (co.device_type != ExecutorDeviceType::L0)
      b.CreateStore(error_lv, wrapper_scalar_expr_func->arg_begin());
    b.CreateRetVoid();
    return {scalar_expr_func, wrapper_scalar_expr_func, inputs};
  }
  return {scalar_expr_func, nullptr, inputs};
}

std::vector<void*> ScalarCodeGenerator::generateNativeCode(
    const CompiledExpression& compiled_expression,
    const CompilationOptions& co) {
  CHECK(module_ && !execution_engine_.get()) << "Invalid code generator state";
  module_.release();
  switch (co.device_type) {
    case ExecutorDeviceType::CPU: {
      execution_engine_ =
          generateNativeCPUCode(compiled_expression.func, {compiled_expression.func}, co);
      return {execution_engine_->getPointerToFunction(compiled_expression.func)};
    }
    case ExecutorDeviceType::GPU: {
      return generateNativeGPUCode(
          compiled_expression.func, compiled_expression.wrapper_func, co);
    }
    case ExecutorDeviceType::L0:  // todo wrapper?
    default: {
      LOG(FATAL) << "Invalid device type";
      return {};  // satisfy -Wreturn-type
    }
  }
}

std::vector<l0::L0Kernel*> ScalarCodeGenerator::generateNativeL0Code(
    const CompiledExpression& compiled_expression,
    const CompilationOptions& co) {
  CHECK(co.device_type == ExecutorDeviceType::L0);
  return generateNativeL0Code(
      compiled_expression.func, compiled_expression.wrapper_func, co);
}

std::vector<llvm::Value*> ScalarCodeGenerator::codegenColumn(
    const Analyzer::ColumnVar* column,
    const bool fetch_column,
    const CompilationOptions& co) {
  int arg_idx = plan_state_->getLocalColumnId(column, fetch_column);
  CHECK_LT(static_cast<size_t>(arg_idx), cgen_state_->row_func_->arg_size());
  llvm::Value* arg = cgen_state_->row_func_->arg_begin() + arg_idx + 1;
  return {arg};
}

std::vector<void*> ScalarCodeGenerator::generateNativeGPUCode(
    llvm::Function* func,
    llvm::Function* wrapper_func,
    const CompilationOptions& co) {
  if (!nvptx_target_machine_) {
    nvptx_target_machine_ =
        initializeNVPTXBackend(CudaMgr_Namespace::NvidiaDeviceArch::Kepler);
  }
  if (!cuda_mgr_) {
    cuda_mgr_ = std::make_unique<CudaMgr_Namespace::CudaMgr>(0);
  }
  const auto& dev_props = cuda_mgr_->getAllDeviceProperties();
  int block_size = dev_props.front().maxThreadsPerBlock;
  GPUTarget gpu_target;
  gpu_target.nvptx_target_machine = nvptx_target_machine_.get();
  gpu_target.cuda_mgr = cuda_mgr_.get();
  gpu_target.block_size = block_size;
  gpu_target.cgen_state = cgen_state_;
  gpu_target.row_func_not_inlined = false;
  gpu_compilation_context_ = CodeGenerator::generateNativeGPUCode(
      func, wrapper_func, {func, wrapper_func}, co, gpu_target);
  return gpu_compilation_context_->getNativeFunctionPointers();
}

std::vector<l0::L0Kernel*> ScalarCodeGenerator::generateNativeL0Code(
    llvm::Function* func,
    llvm::Function* wrapper_func,
    const CompilationOptions& co) {
  if (!l0_mgr_) {
    l0_mgr_ = std::make_unique<l0::L0Manager>();
  }
  l0_compilation_context_ = CodeGenerator::generateNativeL0Code(
      func, wrapper_func, {func, wrapper_func}, co, l0_mgr_.get());
  return l0_compilation_context_->getNativeFunctionPointers();
}
