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

/**
 * @file    TargetExprBuilder.cpp
 * @author  Alex Baden <alex.baden@omnisci.com>
 * @brief   Helpers for codegen of target expressions
 */

#include "TargetExprBuilder.h"

#include "CodeGenerator.h"
#include "Execute.h"
#include "GroupByAndAggregate.h"
#include "Logger/Logger.h"
#include "MaxwellCodegenPatch.h"
#include "OutputBufferInitialization.h"

#define LL_CONTEXT executor->cgen_state_->context_
#define LL_BUILDER executor->cgen_state_->ir_builder_
#define LL_BOOL(v) executor->ll_bool(v)
#define LL_INT(v) executor->cgen_state_->llInt(v)
#define LL_FP(v) executor->cgen_state_->llFp(v)
#define ROW_FUNC executor->cgen_state_->row_func_

namespace {

std::vector<std::string> agg_fn_base_names(const TargetInfo& target_info) {
  const auto& chosen_type = get_compact_type(target_info);
  if (!target_info.is_agg || target_info.agg_kind == kSAMPLE) {
    if (chosen_type.is_geometry()) {
      return std::vector<std::string>(2 * chosen_type.get_physical_coord_cols(),
                                      "agg_id");
    }
    if (chosen_type.is_varlen()) {
      return {"agg_id", "agg_id"};
    }
    return {"agg_id"};
  }
  switch (target_info.agg_kind) {
    case kAVG:
      return {"agg_sum", "agg_count"};
    case kCOUNT:
      return {target_info.is_distinct ? "agg_count_distinct" : "agg_count"};
    case kMAX:
      return {"agg_max"};
    case kMIN:
      return {"agg_min"};
    case kSUM:
      return {"agg_sum"};
    case kAPPROX_COUNT_DISTINCT:
      return {"agg_approximate_count_distinct"};
    case kAPPROX_MEDIAN:
      return {"agg_approx_median"};
    case kSINGLE_VALUE:
      return {"checked_single_agg_id"};
    case kSAMPLE:
      return {"agg_id"};
    default:
      UNREACHABLE() << "Unrecognized agg kind: " << std::to_string(target_info.agg_kind);
  }
  return {};
}

inline bool is_columnar_projection(const QueryMemoryDescriptor& query_mem_desc) {
  return query_mem_desc.getQueryDescriptionType() == QueryDescriptionType::Projection &&
         query_mem_desc.didOutputColumnar();
}

bool is_simple_count(const TargetInfo& target_info) {
  return target_info.is_agg && target_info.agg_kind == kCOUNT && !target_info.is_distinct;
}

bool target_has_geo(const TargetInfo& target_info) {
  return target_info.is_agg ? target_info.agg_arg_type.is_geometry()
                            : target_info.sql_type.is_geometry();
}

}  // namespace

void TargetExprCodegen::codegen(
    GroupByAndAggregate* group_by_and_agg,
    Executor* executor,
    const QueryMemoryDescriptor& query_mem_desc,
    const CompilationOptions& co,
    const GpuSharedMemoryContext& gpu_smem_context,
    const std::tuple<llvm::Value*, llvm::Value*>& agg_out_ptr_w_idx_in,
    const std::vector<llvm::Value*>& agg_out_vec,
    llvm::Value* output_buffer_byte_stream,
    llvm::Value* out_row_idx,
    GroupByAndAggregate::DiamondCodegen& diamond_codegen,
    GroupByAndAggregate::DiamondCodegen* sample_cfg) const {
  CHECK(group_by_and_agg);
  CHECK(executor);
  AUTOMATIC_IR_METADATA(executor->cgen_state_.get());

  auto agg_out_ptr_w_idx = agg_out_ptr_w_idx_in;
  const auto arg_expr = agg_arg(target_expr);

  const auto agg_fn_names = agg_fn_base_names(target_info);
  const auto window_func = dynamic_cast<const Analyzer::WindowFunction*>(target_expr);
  WindowProjectNodeContext::resetWindowFunctionContext(executor);
  auto target_lvs =
      window_func
          ? std::vector<llvm::Value*>{executor->codegenWindowFunction(target_idx, co)}
          : group_by_and_agg->codegenAggArg(target_expr, co);
  const auto window_row_ptr = window_func
                                  ? group_by_and_agg->codegenWindowRowPointer(
                                        window_func, query_mem_desc, co, diamond_codegen)
                                  : nullptr;
  if (window_row_ptr) {
    agg_out_ptr_w_idx =
        std::make_tuple(window_row_ptr, std::get<1>(agg_out_ptr_w_idx_in));
    if (window_function_is_aggregate(window_func->getKind())) {
      out_row_idx = window_row_ptr;
    }
  }

  llvm::Value* str_target_lv{nullptr};
  if (target_lvs.size() == 3 && !target_has_geo(target_info)) {
    // none encoding string, pop the packed pointer + length since
    // it's only useful for IS NULL checks and assumed to be only
    // two components (pointer and length) for the purpose of projection
    str_target_lv = target_lvs.front();
    target_lvs.erase(target_lvs.begin());
  }
  if (target_info.sql_type.is_geometry()) {
    // Geo cols are expanded to the physical coord cols. Each physical coord col is an
    // array. Ensure that the target values generated match the number of agg
    // functions before continuing
    if (target_lvs.size() < agg_fn_names.size()) {
      CHECK_EQ(target_lvs.size(), agg_fn_names.size() / 2);
      std::vector<llvm::Value*> new_target_lvs;
      new_target_lvs.reserve(agg_fn_names.size());
      for (const auto& target_lv : target_lvs) {
        new_target_lvs.push_back(target_lv);
        new_target_lvs.push_back(target_lv);
      }
      target_lvs = new_target_lvs;
    }
  }
  if (target_lvs.size() < agg_fn_names.size()) {
    CHECK_EQ(size_t(1), target_lvs.size());
    CHECK_EQ(size_t(2), agg_fn_names.size());
    for (size_t i = 1; i < agg_fn_names.size(); ++i) {
      target_lvs.push_back(target_lvs.front());
    }
  } else {
    if (target_has_geo(target_info)) {
      if (!target_info.is_agg) {
        CHECK_EQ(static_cast<size_t>(2 * target_info.sql_type.get_physical_coord_cols()),
                 target_lvs.size());
        CHECK_EQ(agg_fn_names.size(), target_lvs.size());
      }
    } else {
      CHECK(str_target_lv || (agg_fn_names.size() == target_lvs.size()));
      CHECK(target_lvs.size() == 1 || target_lvs.size() == 2);
    }
  }

  int32_t slot_index = base_slot_index;
  CHECK_GE(slot_index, 0);
  CHECK(is_group_by || static_cast<size_t>(slot_index) < agg_out_vec.size());

  uint32_t col_off{0};
  if (co.device_type == ExecutorDeviceType::GPU && query_mem_desc.threadsShareMemory() &&
      is_simple_count(target_info) &&
      (!arg_expr || arg_expr->get_type_info().get_notnull())) {
    CHECK_EQ(size_t(1), agg_fn_names.size());
    const auto chosen_bytes = query_mem_desc.getPaddedSlotWidthBytes(slot_index);
    llvm::Value* agg_col_ptr{nullptr};
    if (is_group_by) {
      if (query_mem_desc.didOutputColumnar()) {
        col_off = query_mem_desc.getColOffInBytes(slot_index);
        CHECK_EQ(size_t(0), col_off % chosen_bytes);
        col_off /= chosen_bytes;
        CHECK(std::get<1>(agg_out_ptr_w_idx));
        auto offset =
            LL_BUILDER.CreateAdd(std::get<1>(agg_out_ptr_w_idx), LL_INT(col_off));
        agg_col_ptr = LL_BUILDER.CreateGEP(
            LL_BUILDER.CreateBitCast(
                std::get<0>(agg_out_ptr_w_idx),
                llvm::PointerType::get(get_int_type((chosen_bytes << 3), LL_CONTEXT), 0)),
            offset);
      } else {
        col_off = query_mem_desc.getColOnlyOffInBytes(slot_index);
        CHECK_EQ(size_t(0), col_off % chosen_bytes);
        col_off /= chosen_bytes;
        agg_col_ptr = LL_BUILDER.CreateGEP(
            LL_BUILDER.CreateBitCast(
                std::get<0>(agg_out_ptr_w_idx),
                llvm::PointerType::get(get_int_type((chosen_bytes << 3), LL_CONTEXT), 0)),
            LL_INT(col_off));
      }
    }

    if (chosen_bytes != sizeof(int32_t)) {
      CHECK_EQ(8, chosen_bytes);
      if (g_bigint_count) {
        const auto acc_i64 = LL_BUILDER.CreateBitCast(
            is_group_by ? agg_col_ptr : agg_out_vec[slot_index],
            llvm::PointerType::get(get_int_type(64, LL_CONTEXT), 0));
        if (gpu_smem_context.isSharedMemoryUsed()) {
          group_by_and_agg->emitCall(
              "agg_count_shared", std::vector<llvm::Value*>{acc_i64, LL_INT(int64_t(1))});
        } else {
          LL_BUILDER.CreateAtomicRMW(llvm::AtomicRMWInst::Add,
                                     acc_i64,
                                     LL_INT(int64_t(1)),
                                     llvm::AtomicOrdering::Monotonic);
        }
      } else {
        auto acc_i32 = LL_BUILDER.CreateBitCast(
            is_group_by ? agg_col_ptr : agg_out_vec[slot_index],
            llvm::PointerType::get(get_int_type(32, LL_CONTEXT), 0));
        if (gpu_smem_context.isSharedMemoryUsed()) {
          acc_i32 = LL_BUILDER.CreatePointerCast(
              acc_i32, llvm::Type::getInt32PtrTy(LL_CONTEXT, 3));
        }
        LL_BUILDER.CreateAtomicRMW(llvm::AtomicRMWInst::Add,
                                   acc_i32,
                                   LL_INT(1),
                                   llvm::AtomicOrdering::Monotonic);
      }
    } else {
      const auto acc_i32 = (is_group_by ? agg_col_ptr : agg_out_vec[slot_index]);
      if (gpu_smem_context.isSharedMemoryUsed()) {
        // Atomic operation on address space level 3 (Shared):
        const auto shared_acc_i32 = LL_BUILDER.CreatePointerCast(
            acc_i32, llvm::Type::getInt32PtrTy(LL_CONTEXT, 3));
        LL_BUILDER.CreateAtomicRMW(llvm::AtomicRMWInst::Add,
                                   shared_acc_i32,
                                   LL_INT(1),
                                   llvm::AtomicOrdering::Monotonic);
      } else {
        LL_BUILDER.CreateAtomicRMW(llvm::AtomicRMWInst::Add,
                                   acc_i32,
                                   LL_INT(1),
                                   llvm::AtomicOrdering::Monotonic);
      }
    }
    return;
  }

  codegenAggregate(group_by_and_agg,
                   executor,
                   query_mem_desc,
                   co,
                   target_lvs,
                   agg_out_ptr_w_idx,
                   agg_out_vec,
                   output_buffer_byte_stream,
                   out_row_idx,
                   slot_index);
}

void TargetExprCodegen::codegenAggregate(
    GroupByAndAggregate* group_by_and_agg,
    Executor* executor,
    const QueryMemoryDescriptor& query_mem_desc,
    const CompilationOptions& co,
    const std::vector<llvm::Value*>& target_lvs,
    const std::tuple<llvm::Value*, llvm::Value*>& agg_out_ptr_w_idx,
    const std::vector<llvm::Value*>& agg_out_vec,
    llvm::Value* output_buffer_byte_stream,
    llvm::Value* out_row_idx,
    int32_t slot_index) const {
  AUTOMATIC_IR_METADATA(executor->cgen_state_.get());
  size_t target_lv_idx = 0;
  const bool lazy_fetched{executor->plan_state_->isLazyFetchColumn(target_expr)};

  CodeGenerator code_generator(executor);

  const auto agg_fn_names = agg_fn_base_names(target_info);
  auto arg_expr = agg_arg(target_expr);

  for (const auto& agg_base_name : agg_fn_names) {
    if (target_info.is_distinct && arg_expr->get_type_info().is_array()) {
      CHECK_EQ(static_cast<size_t>(query_mem_desc.getLogicalSlotWidthBytes(slot_index)),
               sizeof(int64_t));
      // TODO(miyu): check if buffer may be columnar here
      CHECK(!query_mem_desc.didOutputColumnar());
      const auto& elem_ti = arg_expr->get_type_info().get_elem_type();
      uint32_t col_off{0};
      if (is_group_by) {
        const auto col_off_in_bytes = query_mem_desc.getColOnlyOffInBytes(slot_index);
        CHECK_EQ(size_t(0), col_off_in_bytes % sizeof(int64_t));
        col_off /= sizeof(int64_t);
      }
      executor->cgen_state_->emitExternalCall(
          "agg_count_distinct_array_" + numeric_type_name(elem_ti),
          llvm::Type::getVoidTy(LL_CONTEXT),
          {is_group_by
               ? LL_BUILDER.CreateGEP(std::get<0>(agg_out_ptr_w_idx), LL_INT(col_off))
               : agg_out_vec[slot_index],
           target_lvs[target_lv_idx],
           code_generator.posArg(arg_expr),
           elem_ti.is_fp()
               ? static_cast<llvm::Value*>(executor->cgen_state_->inlineFpNull(elem_ti))
               : static_cast<llvm::Value*>(
                     executor->cgen_state_->inlineIntNull(elem_ti))});
      ++slot_index;
      ++target_lv_idx;
      continue;
    }

    llvm::Value* agg_col_ptr{nullptr};
    const auto chosen_bytes =
        static_cast<size_t>(query_mem_desc.getPaddedSlotWidthBytes(slot_index));
    const auto& chosen_type = get_compact_type(target_info);
    const auto& arg_type =
        ((arg_expr && arg_expr->get_type_info().get_type() != kNULLT) &&
         !target_info.is_distinct)
            ? target_info.agg_arg_type
            : target_info.sql_type;
    const bool is_fp_arg =
        !lazy_fetched && arg_type.get_type() != kNULLT && arg_type.is_fp();
    if (is_group_by) {
      agg_col_ptr = group_by_and_agg->codegenAggColumnPtr(output_buffer_byte_stream,
                                                          out_row_idx,
                                                          agg_out_ptr_w_idx,
                                                          query_mem_desc,
                                                          chosen_bytes,
                                                          slot_index,
                                                          target_idx);
      CHECK(agg_col_ptr);
      agg_col_ptr->setName("agg_col_ptr");
    }

    const bool float_argument_input = takes_float_argument(target_info);
    const bool is_count_in_avg = target_info.agg_kind == kAVG && target_lv_idx == 1;
    // The count component of an average should never be compacted.
    const auto agg_chosen_bytes =
        float_argument_input && !is_count_in_avg ? sizeof(float) : chosen_bytes;
    if (float_argument_input) {
      CHECK_GE(chosen_bytes, sizeof(float));
    }

    auto target_lv = target_lvs[target_lv_idx];
    const auto needs_unnest_double_patch = group_by_and_agg->needsUnnestDoublePatch(
        target_lv, agg_base_name, query_mem_desc.threadsShareMemory(), co);
    const auto need_skip_null = !needs_unnest_double_patch && target_info.skip_null_val;
    if (!needs_unnest_double_patch) {
      if (need_skip_null && !is_agg_domain_range_equivalent(target_info.agg_kind)) {
        target_lv = group_by_and_agg->convertNullIfAny(arg_type, target_info, target_lv);
      } else if (is_fp_arg) {
        target_lv = executor->castToFP(target_lv, arg_type, target_info.sql_type);
      }
      if (!dynamic_cast<const Analyzer::AggExpr*>(target_expr) || arg_expr) {
        target_lv =
            executor->cgen_state_->castToTypeIn(target_lv, (agg_chosen_bytes << 3));
      }
    }

    const bool is_simple_count_target = is_simple_count(target_info);
    llvm::Value* str_target_lv{nullptr};
    if (target_lvs.size() == 3 && !target_has_geo(target_info)) {
      // none encoding string
      str_target_lv = target_lvs.front();
    }
    std::vector<llvm::Value*> agg_args{
        executor->castToIntPtrTyIn((is_group_by ? agg_col_ptr : agg_out_vec[slot_index]),
                                   (agg_chosen_bytes << 3)),
        (is_simple_count_target && !arg_expr)
            ? (agg_chosen_bytes == sizeof(int32_t) ? LL_INT(int32_t(0))
                                                   : LL_INT(int64_t(0)))
            : (is_simple_count_target && arg_expr && str_target_lv ? str_target_lv
                                                                   : target_lv)};
    if (query_mem_desc.isLogicalSizedColumnsAllowed()) {
      if (is_simple_count_target && arg_expr && str_target_lv) {
        agg_args[1] =
            agg_chosen_bytes == sizeof(int32_t) ? LL_INT(int32_t(0)) : LL_INT(int64_t(0));
      }
    }
    std::string agg_fname{agg_base_name};
    if (is_fp_arg) {
      if (!lazy_fetched) {
        if (agg_chosen_bytes == sizeof(float)) {
          CHECK_EQ(arg_type.get_type(), kFLOAT);
          agg_fname += "_float";
        } else {
          CHECK_EQ(agg_chosen_bytes, sizeof(double));
          agg_fname += "_double";
        }
      }
    } else if (agg_chosen_bytes == sizeof(int32_t)) {
      agg_fname += "_int32";
    } else if (agg_chosen_bytes == sizeof(int16_t) &&
               query_mem_desc.didOutputColumnar()) {
      agg_fname += "_int16";
    } else if (agg_chosen_bytes == sizeof(int8_t) && query_mem_desc.didOutputColumnar()) {
      agg_fname += "_int8";
    }

    if (is_distinct_target(target_info)) {
      CHECK_EQ(agg_chosen_bytes, sizeof(int64_t));
      CHECK(!chosen_type.is_fp());
      group_by_and_agg->codegenCountDistinct(
          target_idx, target_expr, agg_args, query_mem_desc, co.device_type);
    } else if (target_info.agg_kind == kAPPROX_MEDIAN) {
      CHECK_EQ(agg_chosen_bytes, sizeof(int64_t));
      group_by_and_agg->codegenApproxMedian(
          target_idx, target_expr, agg_args, query_mem_desc, co.device_type);
    } else {
      const auto& arg_ti = target_info.agg_arg_type;
      if (need_skip_null && !arg_ti.is_geometry()) {
        agg_fname += "_skip_val";
      }

      if (target_info.agg_kind == kSINGLE_VALUE ||
          (need_skip_null && !arg_ti.is_geometry())) {
        llvm::Value* null_in_lv{nullptr};
        if (arg_ti.is_fp()) {
          null_in_lv =
              static_cast<llvm::Value*>(executor->cgen_state_->inlineFpNull(arg_ti));
        } else {
          null_in_lv = static_cast<llvm::Value*>(executor->cgen_state_->inlineIntNull(
              is_agg_domain_range_equivalent(target_info.agg_kind)
                  ? arg_ti
                  : target_info.sql_type));
        }
        CHECK(null_in_lv);
        auto null_lv =
            executor->cgen_state_->castToTypeIn(null_in_lv, (agg_chosen_bytes << 3));
        agg_args.push_back(null_lv);
      }
      if (!target_info.is_distinct) {
        if (co.device_type == ExecutorDeviceType::GPU &&
            query_mem_desc.threadsShareMemory()) {
          agg_fname += "_shared";
          if (needs_unnest_double_patch) {
            agg_fname = patch_agg_fname(agg_fname);
          }
        } else if (co.device_type == ExecutorDeviceType::CPU &&
                   query_mem_desc.cpuThreadsShareMemory()) {
          agg_fname += "_cpu_shared";
        }
        auto agg_fname_call_ret_lv = group_by_and_agg->emitCall(agg_fname, agg_args);

        if (agg_fname.find("checked") != std::string::npos) {
          group_by_and_agg->checkErrorCode(agg_fname_call_ret_lv);
        }
      }
    }
    const auto window_func = dynamic_cast<const Analyzer::WindowFunction*>(target_expr);
    if (window_func && window_function_requires_peer_handling(window_func)) {
      const auto window_func_context =
          WindowProjectNodeContext::getActiveWindowFunctionContext(executor);
      const auto pending_outputs =
          LL_INT(window_func_context->aggregateStatePendingOutputs());
      executor->cgen_state_->emitExternalCall("add_window_pending_output",
                                              llvm::Type::getVoidTy(LL_CONTEXT),
                                              {agg_args.front(), pending_outputs});
      const auto& window_func_ti = window_func->get_type_info();
      std::string apply_window_pending_outputs_name = "apply_window_pending_outputs";
      switch (window_func_ti.get_type()) {
        case kFLOAT: {
          apply_window_pending_outputs_name += "_float";
          if (query_mem_desc.didOutputColumnar()) {
            apply_window_pending_outputs_name += "_columnar";
          }
          break;
        }
        case kDOUBLE: {
          apply_window_pending_outputs_name += "_double";
          break;
        }
        default: {
          apply_window_pending_outputs_name += "_int";
          if (query_mem_desc.didOutputColumnar()) {
            apply_window_pending_outputs_name +=
                std::to_string(window_func_ti.get_size() * 8);
          } else {
            apply_window_pending_outputs_name += "64";
          }
          break;
        }
      }
      const auto partition_end =
          LL_INT(reinterpret_cast<int64_t>(window_func_context->partitionEnd()));
      executor->cgen_state_->emitExternalCall(apply_window_pending_outputs_name,
                                              llvm::Type::getVoidTy(LL_CONTEXT),
                                              {pending_outputs,
                                               target_lvs.front(),
                                               partition_end,
                                               code_generator.posArg(nullptr)});
    }

    ++slot_index;
    ++target_lv_idx;
  }
}

void TargetExprCodegenBuilder::operator()(const Analyzer::Expr* target_expr,
                                          const Executor* executor,
                                          const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(executor->cgen_state_.get());
  if (query_mem_desc.getPaddedSlotWidthBytes(slot_index_counter) == 0) {
    CHECK(!dynamic_cast<const Analyzer::AggExpr*>(target_expr));
    ++slot_index_counter;
    ++target_index_counter;
    return;
  }
  if (dynamic_cast<const Analyzer::UOper*>(target_expr) &&
      static_cast<const Analyzer::UOper*>(target_expr)->get_optype() == kUNNEST) {
    throw std::runtime_error("UNNEST not supported in the projection list yet.");
  }
  if ((executor->plan_state_->isLazyFetchColumn(target_expr) || !is_group_by) &&
      (static_cast<size_t>(query_mem_desc.getPaddedSlotWidthBytes(slot_index_counter)) <
       sizeof(int64_t)) &&
      !is_columnar_projection(query_mem_desc)) {
    // TODO(miyu): enable different byte width in the layout w/o padding
    throw CompilationRetryNoCompaction();
  }

  auto target_info = get_target_info(target_expr, g_bigint_count);
  auto arg_expr = agg_arg(target_expr);
  if (arg_expr) {
    if (target_info.agg_kind == kSINGLE_VALUE || target_info.agg_kind == kSAMPLE ||
        target_info.agg_kind == kAPPROX_MEDIAN) {
      target_info.skip_null_val = false;
    } else if (query_mem_desc.getQueryDescriptionType() ==
                   QueryDescriptionType::NonGroupedAggregate &&
               !arg_expr->get_type_info().is_varlen()) {
      // TODO: COUNT is currently not null-aware for varlen types. Need to add proper code
      // generation for handling varlen nulls.
      target_info.skip_null_val = true;
    } else if (constrained_not_null(arg_expr, ra_exe_unit.quals)) {
      target_info.skip_null_val = false;
    }
  }

  if (!(query_mem_desc.getQueryDescriptionType() ==
        QueryDescriptionType::NonGroupedAggregate) &&
      (co.device_type == ExecutorDeviceType::GPU) && target_info.is_agg &&
      (target_info.agg_kind == kSAMPLE)) {
    sample_exprs_to_codegen.emplace_back(target_expr,
                                         target_info,
                                         slot_index_counter,
                                         target_index_counter++,
                                         is_group_by);
  } else {
    target_exprs_to_codegen.emplace_back(target_expr,
                                         target_info,
                                         slot_index_counter,
                                         target_index_counter++,
                                         is_group_by);
  }

  const auto agg_fn_names = agg_fn_base_names(target_info);
  slot_index_counter += agg_fn_names.size();
}

namespace {

inline int64_t get_initial_agg_val(const TargetInfo& target_info,
                                   const QueryMemoryDescriptor& query_mem_desc) {
  const bool is_group_by{query_mem_desc.isGroupBy()};
  if (target_info.agg_kind == kSAMPLE && target_info.sql_type.is_string() &&
      target_info.sql_type.get_compression() != kENCODING_NONE) {
    return get_agg_initial_val(target_info.agg_kind,
                               target_info.sql_type,
                               is_group_by,
                               query_mem_desc.getCompactByteWidth());
  }
  return 0;
}

}  // namespace

void TargetExprCodegenBuilder::codegen(
    GroupByAndAggregate* group_by_and_agg,
    Executor* executor,
    const QueryMemoryDescriptor& query_mem_desc,
    const CompilationOptions& co,
    const GpuSharedMemoryContext& gpu_smem_context,
    const std::tuple<llvm::Value*, llvm::Value*>& agg_out_ptr_w_idx,
    const std::vector<llvm::Value*>& agg_out_vec,
    llvm::Value* output_buffer_byte_stream,
    llvm::Value* out_row_idx,
    GroupByAndAggregate::DiamondCodegen& diamond_codegen) const {
  CHECK(group_by_and_agg);
  CHECK(executor);
  AUTOMATIC_IR_METADATA(executor->cgen_state_.get());

  for (const auto& target_expr_codegen : target_exprs_to_codegen) {
    target_expr_codegen.codegen(group_by_and_agg,
                                executor,
                                query_mem_desc,
                                co,
                                gpu_smem_context,
                                agg_out_ptr_w_idx,
                                agg_out_vec,
                                output_buffer_byte_stream,
                                out_row_idx,
                                diamond_codegen);
  }
  if (!sample_exprs_to_codegen.empty()) {
    codegenSampleExpressions(group_by_and_agg,
                             executor,
                             query_mem_desc,
                             co,
                             agg_out_ptr_w_idx,
                             agg_out_vec,
                             output_buffer_byte_stream,
                             out_row_idx,
                             diamond_codegen);
  }
}

void TargetExprCodegenBuilder::codegenSampleExpressions(
    GroupByAndAggregate* group_by_and_agg,
    Executor* executor,
    const QueryMemoryDescriptor& query_mem_desc,
    const CompilationOptions& co,
    const std::tuple<llvm::Value*, llvm::Value*>& agg_out_ptr_w_idx,
    const std::vector<llvm::Value*>& agg_out_vec,
    llvm::Value* output_buffer_byte_stream,
    llvm::Value* out_row_idx,
    GroupByAndAggregate::DiamondCodegen& diamond_codegen) const {
  AUTOMATIC_IR_METADATA(executor->cgen_state_.get());
  CHECK(!sample_exprs_to_codegen.empty());
  CHECK(co.device_type == ExecutorDeviceType::GPU);
  if (sample_exprs_to_codegen.size() == 1 &&
      !sample_exprs_to_codegen.front().target_info.sql_type.is_varlen()) {
    codegenSingleSlotSampleExpression(group_by_and_agg,
                                      executor,
                                      query_mem_desc,
                                      co,
                                      agg_out_ptr_w_idx,
                                      agg_out_vec,
                                      output_buffer_byte_stream,
                                      out_row_idx,
                                      diamond_codegen);
  } else {
    codegenMultiSlotSampleExpressions(group_by_and_agg,
                                      executor,
                                      query_mem_desc,
                                      co,
                                      agg_out_ptr_w_idx,
                                      agg_out_vec,
                                      output_buffer_byte_stream,
                                      out_row_idx,
                                      diamond_codegen);
  }
}

void TargetExprCodegenBuilder::codegenSingleSlotSampleExpression(
    GroupByAndAggregate* group_by_and_agg,
    Executor* executor,
    const QueryMemoryDescriptor& query_mem_desc,
    const CompilationOptions& co,
    const std::tuple<llvm::Value*, llvm::Value*>& agg_out_ptr_w_idx,
    const std::vector<llvm::Value*>& agg_out_vec,
    llvm::Value* output_buffer_byte_stream,
    llvm::Value* out_row_idx,
    GroupByAndAggregate::DiamondCodegen& diamond_codegen) const {
  AUTOMATIC_IR_METADATA(executor->cgen_state_.get());
  CHECK_EQ(size_t(1), sample_exprs_to_codegen.size());
  CHECK(!sample_exprs_to_codegen.front().target_info.sql_type.is_varlen());
  CHECK(co.device_type == ExecutorDeviceType::GPU);
  // no need for the atomic if we only have one SAMPLE target
  sample_exprs_to_codegen.front().codegen(group_by_and_agg,
                                          executor,
                                          query_mem_desc,
                                          co,
                                          {},
                                          agg_out_ptr_w_idx,
                                          agg_out_vec,
                                          output_buffer_byte_stream,
                                          out_row_idx,
                                          diamond_codegen);
}

void TargetExprCodegenBuilder::codegenMultiSlotSampleExpressions(
    GroupByAndAggregate* group_by_and_agg,
    Executor* executor,
    const QueryMemoryDescriptor& query_mem_desc,
    const CompilationOptions& co,
    const std::tuple<llvm::Value*, llvm::Value*>& agg_out_ptr_w_idx,
    const std::vector<llvm::Value*>& agg_out_vec,
    llvm::Value* output_buffer_byte_stream,
    llvm::Value* out_row_idx,
    GroupByAndAggregate::DiamondCodegen& diamond_codegen) const {
  AUTOMATIC_IR_METADATA(executor->cgen_state_.get());
  CHECK(sample_exprs_to_codegen.size() > 1 ||
        sample_exprs_to_codegen.front().target_info.sql_type.is_varlen());
  CHECK(co.device_type == ExecutorDeviceType::GPU);
  const auto& first_sample_expr = sample_exprs_to_codegen.front();
  auto target_lvs = group_by_and_agg->codegenAggArg(first_sample_expr.target_expr, co);
  CHECK_GE(target_lvs.size(), size_t(1));

  const auto init_val =
      get_initial_agg_val(first_sample_expr.target_info, query_mem_desc);

  llvm::Value* agg_col_ptr{nullptr};
  if (is_group_by) {
    const auto agg_column_size_bytes =
        query_mem_desc.isLogicalSizedColumnsAllowed() &&
                !first_sample_expr.target_info.sql_type.is_varlen()
            ? first_sample_expr.target_info.sql_type.get_size()
            : sizeof(int64_t);
    agg_col_ptr = group_by_and_agg->codegenAggColumnPtr(output_buffer_byte_stream,
                                                        out_row_idx,
                                                        agg_out_ptr_w_idx,
                                                        query_mem_desc,
                                                        agg_column_size_bytes,
                                                        first_sample_expr.base_slot_index,
                                                        first_sample_expr.target_idx);
  } else {
    CHECK_LT(static_cast<size_t>(first_sample_expr.base_slot_index), agg_out_vec.size());
    agg_col_ptr =
        executor->castToIntPtrTyIn(agg_out_vec[first_sample_expr.base_slot_index], 64);
  }

  auto sample_cas_lv = codegenSlotEmptyKey(agg_col_ptr, target_lvs, executor, init_val);

  GroupByAndAggregate::DiamondCodegen sample_cfg(
      sample_cas_lv, executor, false, "sample_valcheck", &diamond_codegen, false);

  for (const auto& target_expr_codegen : sample_exprs_to_codegen) {
    target_expr_codegen.codegen(group_by_and_agg,
                                executor,
                                query_mem_desc,
                                co,
                                {},
                                agg_out_ptr_w_idx,
                                agg_out_vec,
                                output_buffer_byte_stream,
                                out_row_idx,
                                diamond_codegen,
                                &sample_cfg);
  }
}

llvm::Value* TargetExprCodegenBuilder::codegenSlotEmptyKey(
    llvm::Value* agg_col_ptr,
    std::vector<llvm::Value*>& target_lvs,
    Executor* executor,
    const int64_t init_val) const {
  AUTOMATIC_IR_METADATA(executor->cgen_state_.get());
  const auto& first_sample_expr = sample_exprs_to_codegen.front();
  const auto first_sample_slot_bytes =
      first_sample_expr.target_info.sql_type.is_varlen()
          ? sizeof(int64_t)
          : first_sample_expr.target_info.sql_type.get_size();
  llvm::Value* target_lv_casted{nullptr};
  // deciding whether proper casting is required for the first sample's slot:
  if (first_sample_expr.target_info.sql_type.is_varlen()) {
    target_lv_casted =
        LL_BUILDER.CreatePtrToInt(target_lvs.front(), llvm::Type::getInt64Ty(LL_CONTEXT));
  } else if (first_sample_expr.target_info.sql_type.is_fp()) {
    // Initialization value for SAMPLE on a float column should be 0
    CHECK_EQ(init_val, 0);
    if (query_mem_desc.isLogicalSizedColumnsAllowed()) {
      target_lv_casted = executor->cgen_state_->ir_builder_.CreateFPToSI(
          target_lvs.front(),
          first_sample_slot_bytes == sizeof(float) ? llvm::Type::getInt32Ty(LL_CONTEXT)
                                                   : llvm::Type::getInt64Ty(LL_CONTEXT));
    } else {
      target_lv_casted = executor->cgen_state_->ir_builder_.CreateFPToSI(
          target_lvs.front(), llvm::Type::getInt64Ty(LL_CONTEXT));
    }
  } else if (first_sample_slot_bytes != sizeof(int64_t) &&
             !query_mem_desc.isLogicalSizedColumnsAllowed()) {
    target_lv_casted =
        executor->cgen_state_->ir_builder_.CreateCast(llvm::Instruction::CastOps::SExt,
                                                      target_lvs.front(),
                                                      llvm::Type::getInt64Ty(LL_CONTEXT));
  } else {
    target_lv_casted = target_lvs.front();
  }

  std::string slot_empty_cas_func_name("slotEmptyKeyCAS");
  llvm::Value* init_val_lv{LL_INT(init_val)};
  if (query_mem_desc.isLogicalSizedColumnsAllowed() &&
      !first_sample_expr.target_info.sql_type.is_varlen()) {
    // add proper suffix to the function name:
    switch (first_sample_slot_bytes) {
      case 1:
        slot_empty_cas_func_name += "_int8";
        break;
      case 2:
        slot_empty_cas_func_name += "_int16";
        break;
      case 4:
        slot_empty_cas_func_name += "_int32";
        break;
      case 8:
        break;
      default:
        UNREACHABLE() << "Invalid slot size for slotEmptyKeyCAS function.";
        break;
    }
    if (first_sample_slot_bytes != sizeof(int64_t)) {
      init_val_lv = llvm::ConstantInt::get(
          get_int_type(first_sample_slot_bytes * 8, LL_CONTEXT), init_val);
    }
  }

  auto sample_cas_lv = executor->cgen_state_->emitExternalCall(
      slot_empty_cas_func_name,
      llvm::Type::getInt1Ty(executor->cgen_state_->context_),
      {agg_col_ptr, target_lv_casted, init_val_lv});
  return sample_cas_lv;
}
