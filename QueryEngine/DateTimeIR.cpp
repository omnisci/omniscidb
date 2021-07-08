/*
 * Copyright 2017 MapD Technologies, Inc.
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

#include "DateTimeUtils.h"
#include "Execute.h"

using namespace DateTimeUtils;

namespace {

const char* get_extract_function_name(ExtractField field) {
  switch (field) {
    case kEPOCH:
      return "extract_epoch";
    case kDATEEPOCH:
      return "extract_dateepoch";
    case kQUARTERDAY:
      return "extract_quarterday";
    case kHOUR:
      return "extract_hour";
    case kMINUTE:
      return "extract_minute";
    case kSECOND:
      return "extract_second";
    case kMILLISECOND:
      return "extract_millisecond";
    case kMICROSECOND:
      return "extract_microsecond";
    case kNANOSECOND:
      return "extract_nanosecond";
    case kDOW:
      return "extract_dow";
    case kISODOW:
      return "extract_isodow";
    case kDAY:
      return "extract_day";
    case kWEEK:
      return "extract_week_monday";
    case kWEEK_SUNDAY:
      return "extract_week_sunday";
    case kWEEK_SATURDAY:
      return "extract_week_saturday";
    case kDOY:
      return "extract_day_of_year";
    case kMONTH:
      return "extract_month";
    case kQUARTER:
      return "extract_quarter";
    case kYEAR:
      return "extract_year";
    case kLDOM:
      return "extract_last_day_of_month";
    case kISLEAP:
      return "extract_is_leap_year";
    case kISEOM:
      return "extract_is_month_end";
    case kISSOM:
      return "extract_is_month_start";
    case kISEOQ:
      return "extract_is_quarter_end";
    case kISSOQ:
      return "extract_is_quarter_start";
    case kISEOY:
      return "extract_is_year_end";
    case kISSOY:
      return "extract_is_year_start";
  }
  UNREACHABLE();
  return "";
}

}  // namespace

llvm::Value* CodeGenerator::codegen(const Analyzer::ExtractExpr* extract_expr,
                                    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  auto from_expr = codegen(extract_expr->get_from_expr(), true, co).front();
  const int32_t extract_field{extract_expr->get_field()};
  const auto& extract_expr_ti = extract_expr->get_from_expr()->get_type_info();
  if (extract_field == kEPOCH) {
    CHECK(extract_expr_ti.get_type() == kTIMESTAMP ||
          extract_expr_ti.get_type() == kDATE);
    if (from_expr->getType()->isIntegerTy(32)) {
      from_expr =
          cgen_state_->ir_builder_.CreateCast(llvm::Instruction::CastOps::SExt,
                                              from_expr,
                                              get_int_type(64, cgen_state_->context_));
      return from_expr;
    }
  }
  CHECK(from_expr->getType()->isIntegerTy(64));
  if (extract_expr_ti.is_high_precision_timestamp()) {
    from_expr = codegenExtractHighPrecisionTimestamps(
        from_expr, extract_expr_ti, extract_expr->get_field());
  }
  if (!extract_expr_ti.is_high_precision_timestamp() &&
      is_subsecond_extract_field(extract_expr->get_field())) {
    from_expr =
        extract_expr_ti.get_notnull()
            ? cgen_state_->ir_builder_.CreateMul(
                  from_expr,
                  cgen_state_->llInt(
                      get_extract_timestamp_precision_scale(extract_expr->get_field())))
            : cgen_state_->emitCall(
                  "mul_int64_t_nullable_lhs",
                  {from_expr,
                   cgen_state_->llInt(
                       get_extract_timestamp_precision_scale(extract_expr->get_field())),
                   cgen_state_->inlineIntNull(extract_expr_ti)});
  }
  const auto extract_fname = get_extract_function_name(extract_expr->get_field());
  if (!extract_expr_ti.get_notnull()) {
    llvm::BasicBlock* extract_nullcheck_bb{nullptr};
    llvm::PHINode* extract_nullcheck_value{nullptr};
    {
      DiamondCodegen null_check(cgen_state_->ir_builder_.CreateICmp(
                                    llvm::ICmpInst::ICMP_EQ,
                                    from_expr,
                                    cgen_state_->inlineIntNull(extract_expr_ti)),
                                executor(),
                                false,
                                "extract_nullcheck",
                                nullptr,
                                false);
      // generate a phi node depending on whether we got a null or not
      extract_nullcheck_bb = llvm::BasicBlock::Create(
          cgen_state_->context_, "extract_nullcheck_bb", cgen_state_->current_func_);

      // update the blocks created by diamond codegen to point to the newly created phi
      // block
      cgen_state_->ir_builder_.SetInsertPoint(null_check.cond_true_);
      cgen_state_->ir_builder_.CreateBr(extract_nullcheck_bb);
      cgen_state_->ir_builder_.SetInsertPoint(null_check.cond_false_);
      auto extract_call =
          cgen_state_->emitExternalCall(extract_fname,
                                        get_int_type(64, cgen_state_->context_),
                                        std::vector<llvm::Value*>{from_expr});
      cgen_state_->ir_builder_.CreateBr(extract_nullcheck_bb);

      cgen_state_->ir_builder_.SetInsertPoint(extract_nullcheck_bb);
      extract_nullcheck_value = cgen_state_->ir_builder_.CreatePHI(
          get_int_type(64, cgen_state_->context_), 2, "extract_value");
      extract_nullcheck_value->addIncoming(extract_call, null_check.cond_false_);
      extract_nullcheck_value->addIncoming(cgen_state_->inlineIntNull(extract_expr_ti),
                                           null_check.cond_true_);
    }

    // diamond codegen will set the insert point in its destructor. override it to
    // continue using the extract nullcheck bb
    CHECK(extract_nullcheck_bb);
    cgen_state_->ir_builder_.SetInsertPoint(extract_nullcheck_bb);
    CHECK(extract_nullcheck_value);
    return extract_nullcheck_value;
  } else {
    return cgen_state_->emitExternalCall(extract_fname,
                                         get_int_type(64, cgen_state_->context_),
                                         std::vector<llvm::Value*>{from_expr});
  }
}

llvm::Value* CodeGenerator::codegen(const Analyzer::DateaddExpr* dateadd_expr,
                                    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto& dateadd_expr_ti = dateadd_expr->get_type_info();
  CHECK(dateadd_expr_ti.get_type() == kTIMESTAMP || dateadd_expr_ti.get_type() == kDATE);
  auto datetime = codegen(dateadd_expr->get_datetime_expr(), true, co).front();
  CHECK(datetime->getType()->isIntegerTy(64));
  auto number = codegen(dateadd_expr->get_number_expr(), true, co).front();

  const auto& datetime_ti = dateadd_expr->get_datetime_expr()->get_type_info();
  std::vector<llvm::Value*> dateadd_args{
      cgen_state_->llInt(static_cast<int32_t>(dateadd_expr->get_field())),
      number,
      datetime};
  std::string dateadd_fname{"DateAdd"};
  if (is_subsecond_dateadd_field(dateadd_expr->get_field()) ||
      dateadd_expr_ti.is_high_precision_timestamp()) {
    dateadd_fname += "HighPrecision";
    dateadd_args.push_back(
        cgen_state_->llInt(static_cast<int32_t>(datetime_ti.get_dimension())));
  }
  if (!datetime_ti.get_notnull()) {
    dateadd_args.push_back(cgen_state_->inlineIntNull(datetime_ti));
    dateadd_fname += "Nullable";
  }
  return cgen_state_->emitExternalCall(dateadd_fname,
                                       get_int_type(64, cgen_state_->context_),
                                       dateadd_args,
                                       {llvm::Attribute::NoUnwind,
                                        llvm::Attribute::ReadNone,
                                        llvm::Attribute::Speculatable});
}

llvm::Value* CodeGenerator::codegen(const Analyzer::DatediffExpr* datediff_expr,
                                    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  auto start = codegen(datediff_expr->get_start_expr(), true, co).front();
  CHECK(start->getType()->isIntegerTy(64));
  auto end = codegen(datediff_expr->get_end_expr(), true, co).front();
  CHECK(end->getType()->isIntegerTy(32) || end->getType()->isIntegerTy(64));
  const auto& start_ti = datediff_expr->get_start_expr()->get_type_info();
  const auto& end_ti = datediff_expr->get_end_expr()->get_type_info();
  std::vector<llvm::Value*> datediff_args{
      cgen_state_->llInt(static_cast<int32_t>(datediff_expr->get_field())), start, end};
  std::string datediff_fname{"DateDiff"};
  if (start_ti.is_high_precision_timestamp() || end_ti.is_high_precision_timestamp()) {
    datediff_fname += "HighPrecision";
    datediff_args.push_back(
        cgen_state_->llInt(static_cast<int32_t>(start_ti.get_dimension())));
    datediff_args.push_back(
        cgen_state_->llInt(static_cast<int32_t>(end_ti.get_dimension())));
  }
  const auto& ret_ti = datediff_expr->get_type_info();
  if (!start_ti.get_notnull() || !end_ti.get_notnull()) {
    datediff_args.push_back(cgen_state_->inlineIntNull(ret_ti));
    datediff_fname += "Nullable";
  }
  return cgen_state_->emitExternalCall(
      datediff_fname, get_int_type(64, cgen_state_->context_), datediff_args);
}

llvm::Value* CodeGenerator::codegen(const Analyzer::DatetruncExpr* datetrunc_expr,
                                    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  auto from_expr = codegen(datetrunc_expr->get_from_expr(), true, co).front();
  const auto& datetrunc_expr_ti = datetrunc_expr->get_from_expr()->get_type_info();
  CHECK(from_expr->getType()->isIntegerTy(64));
  DatetruncField const field = datetrunc_expr->get_field();
  if (datetrunc_expr_ti.is_high_precision_timestamp()) {
    return codegenDateTruncHighPrecisionTimestamps(from_expr, datetrunc_expr_ti, field);
  }
  static_assert(dtSECOND + 1 == dtMILLISECOND, "Please keep these consecutive.");
  static_assert(dtMILLISECOND + 1 == dtMICROSECOND, "Please keep these consecutive.");
  static_assert(dtMICROSECOND + 1 == dtNANOSECOND, "Please keep these consecutive.");
  if (dtSECOND <= field && field <= dtNANOSECOND) {
    return cgen_state_->ir_builder_.CreateCast(llvm::Instruction::CastOps::SExt,
                                               from_expr,
                                               get_int_type(64, cgen_state_->context_));
  }
  std::unique_ptr<CodeGenerator::NullCheckCodegen> nullcheck_codegen;
  const bool is_nullable = !datetrunc_expr_ti.get_notnull();
  if (is_nullable) {
    nullcheck_codegen = std::make_unique<NullCheckCodegen>(
        cgen_state_, executor(), from_expr, datetrunc_expr_ti, "date_trunc_nullcheck");
  }
  char const* const fname = datetrunc_fname_lookup.at(field);
  auto ret = cgen_state_->emitExternalCall(
      fname, get_int_type(64, cgen_state_->context_), {from_expr});
  if (is_nullable) {
    ret = nullcheck_codegen->finalize(ll_int(NULL_BIGINT, cgen_state_->context_), ret);
  }
  return ret;
}

llvm::Value* CodeGenerator::codegenExtractHighPrecisionTimestamps(
    llvm::Value* ts_lv,
    const SQLTypeInfo& ti,
    const ExtractField& field) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  CHECK(ti.is_high_precision_timestamp());
  CHECK(ts_lv->getType()->isIntegerTy(64));
  if (is_subsecond_extract_field(field)) {
    const auto result =
        get_extract_high_precision_adjusted_scale(field, ti.get_dimension());
    if (result.first == kMULTIPLY) {
      return ti.get_notnull()
                 ? cgen_state_->ir_builder_.CreateMul(
                       ts_lv, cgen_state_->llInt(static_cast<int64_t>(result.second)))
                 : cgen_state_->emitCall(
                       "mul_int64_t_nullable_lhs",
                       {ts_lv,
                        cgen_state_->llInt(static_cast<int64_t>(result.second)),
                        cgen_state_->inlineIntNull(ti)});
    } else if (result.first == kDIVIDE) {
      return ti.get_notnull()
                 ? cgen_state_->ir_builder_.CreateSDiv(
                       ts_lv, cgen_state_->llInt(static_cast<int64_t>(result.second)))
                 : cgen_state_->emitCall(
                       "floor_div_nullable_lhs",
                       {ts_lv,
                        cgen_state_->llInt(static_cast<int64_t>(result.second)),
                        cgen_state_->inlineIntNull(ti)});
    } else {
      return ts_lv;
    }
  }
  return ti.get_notnull()
             ? cgen_state_->ir_builder_.CreateSDiv(
                   ts_lv,
                   cgen_state_->llInt(static_cast<int64_t>(
                       get_timestamp_precision_scale(ti.get_dimension()))))
             : cgen_state_->emitCall(
                   "floor_div_nullable_lhs",
                   {ts_lv,
                    cgen_state_->llInt(get_timestamp_precision_scale(ti.get_dimension())),
                    cgen_state_->inlineIntNull(ti)});
}

llvm::Value* CodeGenerator::codegenDateTruncHighPrecisionTimestamps(
    llvm::Value* ts_lv,
    const SQLTypeInfo& ti,
    const DatetruncField& field) {
  // Only needed for i in { 0, 3, 6, 9 }.
  constexpr int64_t pow10[10]{1, 0, 0, 1000, 0, 0, 1000000, 0, 0, 1000000000};
  AUTOMATIC_IR_METADATA(cgen_state_);
  CHECK(ti.is_high_precision_timestamp());
  CHECK(ts_lv->getType()->isIntegerTy(64));
  bool const is_nullable = !ti.get_notnull();
  static_assert(dtSECOND + 1 == dtMILLISECOND, "Please keep these consecutive.");
  static_assert(dtMILLISECOND + 1 == dtMICROSECOND, "Please keep these consecutive.");
  static_assert(dtMICROSECOND + 1 == dtNANOSECOND, "Please keep these consecutive.");
  if (dtSECOND <= field && field <= dtNANOSECOND) {
    unsigned const start_dim = ti.get_dimension();      // 0, 3, 6, 9
    unsigned const trunc_dim = (field - dtSECOND) * 3;  // 0, 3, 6, 9
    if (start_dim <= trunc_dim) {
      return ts_lv;  // Truncating to an equal or higher precision has no effect.
    }
    int64_t const dscale = pow10[start_dim - trunc_dim];  // 1e3, 1e6, 1e9
    if (is_nullable) {
      ts_lv = cgen_state_->emitCall(
          "floor_div_nullable_lhs",
          {ts_lv, cgen_state_->llInt(dscale), cgen_state_->inlineIntNull(ti)});
      return cgen_state_->emitCall(
          "mul_int64_t_nullable_lhs",
          {ts_lv, cgen_state_->llInt(dscale), cgen_state_->inlineIntNull(ti)});
    } else {
      ts_lv = cgen_state_->ir_builder_.CreateSDiv(ts_lv, cgen_state_->llInt(dscale));
      return cgen_state_->ir_builder_.CreateMul(ts_lv, cgen_state_->llInt(dscale));
    }
  }
  int64_t const scale = pow10[ti.get_dimension()];
  ts_lv = is_nullable
              ? cgen_state_->emitCall(
                    "floor_div_nullable_lhs",
                    {ts_lv, cgen_state_->llInt(scale), cgen_state_->inlineIntNull(ti)})
              : cgen_state_->ir_builder_.CreateSDiv(ts_lv, cgen_state_->llInt(scale));

  std::unique_ptr<CodeGenerator::NullCheckCodegen> nullcheck_codegen;
  if (is_nullable) {
    nullcheck_codegen = std::make_unique<NullCheckCodegen>(
        cgen_state_, executor(), ts_lv, ti, "date_trunc_hp_nullcheck");
  }
  char const* const fname = datetrunc_fname_lookup.at(field);
  ts_lv = cgen_state_->emitExternalCall(
      fname, get_int_type(64, cgen_state_->context_), {ts_lv});
  if (is_nullable) {
    ts_lv =
        nullcheck_codegen->finalize(ll_int(NULL_BIGINT, cgen_state_->context_), ts_lv);
  }

  return is_nullable
             ? cgen_state_->emitCall(
                   "mul_int64_t_nullable_lhs",
                   {ts_lv, cgen_state_->llInt(scale), cgen_state_->inlineIntNull(ti)})
             : cgen_state_->ir_builder_.CreateMul(ts_lv, cgen_state_->llInt(scale));
}
