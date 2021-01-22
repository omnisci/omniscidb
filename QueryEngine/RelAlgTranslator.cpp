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

#include "RelAlgTranslator.h"
#include "Shared/SqlTypesLayout.h"

#include "CalciteDeserializerUtils.h"
#include "DateTimePlusRewrite.h"
#include "DateTimeTranslator.h"
#include "Descriptors/RelAlgExecutionDescriptor.h"
#include "ExpressionRewrite.h"
#include "ExtensionFunctionsBinding.h"
#include "ExtensionFunctionsWhitelist.h"
#include "RelAlgDagBuilder.h"
#include "WindowContext.h"

#include <future>

#include "Analyzer/Analyzer.h"
#include "Parser/ParserNode.h"
#include "Shared/likely.h"
#include "Shared/thread_count.h"

#include "Utils/Threading.h"

extern bool g_enable_watchdog;

bool g_enable_experimental_string_functions = false;

namespace {

SQLTypeInfo build_type_info(const SQLTypes sql_type,
                            const int scale,
                            const int precision) {
  SQLTypeInfo ti(sql_type, 0, 0, true);
  if (ti.is_decimal()) {
    ti.set_scale(scale);
    ti.set_precision(precision);
  }
  return ti;
}

std::pair<std::shared_ptr<Analyzer::Expr>, SQLQualifier> get_quantified_rhs(
    const RexScalar* rex_scalar,
    const RelAlgTranslator& translator) {
  std::shared_ptr<Analyzer::Expr> rhs;
  SQLQualifier sql_qual{kONE};
  const auto rex_operator = dynamic_cast<const RexOperator*>(rex_scalar);
  if (!rex_operator) {
    return std::make_pair(rhs, sql_qual);
  }
  const auto rex_function = dynamic_cast<const RexFunctionOperator*>(rex_operator);
  const auto qual_str = rex_function ? rex_function->getName() : "";
  if (qual_str == "PG_ANY"sv || qual_str == "PG_ALL"sv) {
    CHECK_EQ(size_t(1), rex_function->size());
    rhs = translator.translateScalarRex(rex_function->getOperand(0));
    sql_qual = (qual_str == "PG_ANY"sv) ? kANY : kALL;
  }
  if (!rhs && rex_operator->getOperator() == kCAST) {
    CHECK_EQ(size_t(1), rex_operator->size());
    std::tie(rhs, sql_qual) = get_quantified_rhs(rex_operator->getOperand(0), translator);
  }
  return std::make_pair(rhs, sql_qual);
}

std::pair<Datum, bool> datum_from_scalar_tv(const ScalarTargetValue* scalar_tv,
                                            const SQLTypeInfo& ti) noexcept {
  Datum d{0};
  bool is_null_const{false};
  switch (ti.get_type()) {
    case kBOOLEAN: {
      const auto ival = boost::get<int64_t>(scalar_tv);
      CHECK(ival);
      if (*ival == inline_int_null_val(ti)) {
        is_null_const = true;
      } else {
        d.boolval = *ival;
      }
      break;
    }
    case kTINYINT: {
      const auto ival = boost::get<int64_t>(scalar_tv);
      CHECK(ival);
      if (*ival == inline_int_null_val(ti)) {
        is_null_const = true;
      } else {
        d.tinyintval = *ival;
      }
      break;
    }
    case kSMALLINT: {
      const auto ival = boost::get<int64_t>(scalar_tv);
      CHECK(ival);
      if (*ival == inline_int_null_val(ti)) {
        is_null_const = true;
      } else {
        d.smallintval = *ival;
      }
      break;
    }
    case kINT: {
      const auto ival = boost::get<int64_t>(scalar_tv);
      CHECK(ival);
      if (*ival == inline_int_null_val(ti)) {
        is_null_const = true;
      } else {
        d.intval = *ival;
      }
      break;
    }
    case kDECIMAL:
    case kNUMERIC:
    case kBIGINT:
    case kDATE:
    case kTIME:
    case kTIMESTAMP: {
      const auto ival = boost::get<int64_t>(scalar_tv);
      CHECK(ival);
      if (*ival == inline_int_null_val(ti)) {
        is_null_const = true;
      } else {
        d.bigintval = *ival;
      }
      break;
    }
    case kDOUBLE: {
      const auto dval = boost::get<double>(scalar_tv);
      CHECK(dval);
      if (*dval == inline_fp_null_val(ti)) {
        is_null_const = true;
      } else {
        d.doubleval = *dval;
      }
      break;
    }
    case kFLOAT: {
      const auto fval = boost::get<float>(scalar_tv);
      CHECK(fval);
      if (*fval == inline_fp_null_val(ti)) {
        is_null_const = true;
      } else {
        d.floatval = *fval;
      }
      break;
    }
    case kTEXT:
    case kVARCHAR:
    case kCHAR: {
      auto nullable_sptr = boost::get<NullableString>(scalar_tv);
      CHECK(nullable_sptr);
      if (boost::get<void*>(nullable_sptr)) {
        is_null_const = true;
      } else {
        auto sptr = boost::get<std::string>(nullable_sptr);
        d.stringval = new std::string(*sptr);
      }
      break;
    }
    default:
      CHECK(false) << "Unhandled type: " << ti.get_type_name();
  }
  return {d, is_null_const};
}

}  // namespace

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateScalarRex(
    const RexScalar* rex) const {
  const auto rex_input = dynamic_cast<const RexInput*>(rex);
  if (rex_input) {
    return translateInput(rex_input);
  }
  const auto rex_literal = dynamic_cast<const RexLiteral*>(rex);
  if (rex_literal) {
    return translateLiteral(rex_literal);
  }
  const auto rex_window_function = dynamic_cast<const RexWindowFunctionOperator*>(rex);
  if (rex_window_function) {
    return translateWindowFunction(rex_window_function);
  }
  const auto rex_function = dynamic_cast<const RexFunctionOperator*>(rex);
  if (rex_function) {
    return translateFunction(rex_function);
  }
  const auto rex_operator = dynamic_cast<const RexOperator*>(rex);
  if (rex_operator) {
    return translateOper(rex_operator);
  }
  const auto rex_case = dynamic_cast<const RexCase*>(rex);
  if (rex_case) {
    return translateCase(rex_case);
  }
  const auto rex_subquery = dynamic_cast<const RexSubQuery*>(rex);
  if (rex_subquery) {
    return translateScalarSubquery(rex_subquery);
  }
  CHECK(false);
  return nullptr;
}

namespace {

bool is_agg_supported_for_type(const SQLAgg& agg_kind, const SQLTypeInfo& arg_ti) {
  if ((agg_kind == kMIN || agg_kind == kMAX || agg_kind == kSUM || agg_kind == kAVG) &&
      !(arg_ti.is_number() || arg_ti.is_boolean() || arg_ti.is_time())) {
    return false;
  }

  return true;
}

}  // namespace

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateAggregateRex(
    const RexAgg* rex,
    const std::vector<std::shared_ptr<Analyzer::Expr>>& scalar_sources) {
  const auto agg_kind = rex->getKind();
  const bool is_distinct = rex->isDistinct();
  const bool takes_arg{rex->size() > 0};
  std::shared_ptr<Analyzer::Expr> arg_expr;
  std::shared_ptr<Analyzer::Constant> err_rate;
  if (takes_arg) {
    const auto operand = rex->getOperand(0);
    CHECK_LT(operand, scalar_sources.size());
    CHECK_LE(rex->size(), 2u);
    arg_expr = scalar_sources[operand];
    if (agg_kind == kAPPROX_COUNT_DISTINCT && rex->size() == 2) {
      err_rate = std::dynamic_pointer_cast<Analyzer::Constant>(
          scalar_sources[rex->getOperand(1)]);
      if (!err_rate || err_rate->get_type_info().get_type() != kINT ||
          err_rate->get_constval().intval < 1 || err_rate->get_constval().intval > 100) {
        throw std::runtime_error(
            "APPROX_COUNT_DISTINCT's second parameter should be SMALLINT literal between "
            "1 and 100");
      }
    }
    const auto& arg_ti = arg_expr->get_type_info();
    if (!is_agg_supported_for_type(agg_kind, arg_ti)) {
      throw std::runtime_error("Aggregate on " + arg_ti.get_type_name() +
                               " is not supported yet.");
    }
  }
  const auto agg_ti = get_agg_type(agg_kind, arg_expr.get());
  return makeExpr<Analyzer::AggExpr>(agg_ti, agg_kind, arg_expr, is_distinct, err_rate);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateLiteral(
    const RexLiteral* rex_literal) {
  auto lit_ti = build_type_info(
      rex_literal->getType(), rex_literal->getScale(), rex_literal->getPrecision());
  auto target_ti = build_type_info(rex_literal->getTargetType(),
                                   rex_literal->getTypeScale(),
                                   rex_literal->getTypePrecision());
  switch (rex_literal->getType()) {
    case kINT:
    case kBIGINT: {
      Datum d;
      d.bigintval = rex_literal->getVal<int64_t>();
      return makeExpr<Analyzer::Constant>(rex_literal->getType(), false, d);
    }
    case kDECIMAL: {
      const auto val = rex_literal->getVal<int64_t>();
      const int precision = rex_literal->getPrecision();
      const int scale = rex_literal->getScale();
      if (target_ti.is_fp() && !scale) {
        return make_fp_constant(val, target_ti);
      }
      auto lit_expr = scale ? Parser::FixedPtLiteral::analyzeValue(val, scale, precision)
                            : Parser::IntLiteral::analyzeValue(val);
      return lit_ti != target_ti ? lit_expr->add_cast(target_ti) : lit_expr;
    }
    case kTEXT: {
      return Parser::StringLiteral::analyzeValue(rex_literal->getVal<std::string>());
    }
    case kBOOLEAN: {
      Datum d;
      d.boolval = rex_literal->getVal<bool>();
      return makeExpr<Analyzer::Constant>(kBOOLEAN, false, d);
    }
    case kDOUBLE: {
      Datum d;
      d.doubleval = rex_literal->getVal<double>();
      auto lit_expr = makeExpr<Analyzer::Constant>(kDOUBLE, false, d);
      return lit_ti != target_ti ? lit_expr->add_cast(target_ti) : lit_expr;
    }
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH: {
      Datum d;
      d.bigintval = rex_literal->getVal<int64_t>();
      return makeExpr<Analyzer::Constant>(rex_literal->getType(), false, d);
    }
    case kTIME:
    case kTIMESTAMP: {
      Datum d;
      d.bigintval =
          rex_literal->getType() == kTIMESTAMP && rex_literal->getPrecision() > 0
              ? rex_literal->getVal<int64_t>()
              : rex_literal->getVal<int64_t>() / 1000;
      return makeExpr<Analyzer::Constant>(
          SQLTypeInfo(rex_literal->getType(), rex_literal->getPrecision(), 0, false),
          false,
          d);
    }
    case kDATE: {
      Datum d;
      d.bigintval = rex_literal->getVal<int64_t>() * 24 * 3600;
      return makeExpr<Analyzer::Constant>(rex_literal->getType(), false, d);
    }
    case kNULLT: {
      if (target_ti.is_array()) {
        Analyzer::ExpressionPtrVector args;
        // defaulting to valid sub-type for convenience
        target_ti.set_subtype(kBOOLEAN);
        return makeExpr<Analyzer::ArrayExpr>(target_ti, args, true);
      }
      return makeExpr<Analyzer::Constant>(rex_literal->getTargetType(), true, Datum{0});
    }
    default: {
      LOG(FATAL) << "Unexpected literal type " << lit_ti.get_type_name();
    }
  }
  return nullptr;
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateScalarSubquery(
    const RexSubQuery* rex_subquery) const {
  if (just_explain_) {
    throw std::runtime_error("EXPLAIN is not supported with sub-queries");
  }
  CHECK(rex_subquery);
  auto result = rex_subquery->getExecutionResult();
  auto row_set = result->getRows();
  const size_t row_count = row_set->rowCount();
  if (row_count > size_t(1)) {
    throw std::runtime_error("Scalar sub-query returned multiple rows");
  }
  if (row_count == size_t(0)) {
    if (row_set->isValidationOnlyRes()) {
      Datum d{0};
      return makeExpr<Analyzer::Constant>(rex_subquery->getType(), false, d);
    }
    throw std::runtime_error("Scalar sub-query returned no results");
  }
  CHECK_EQ(row_count, size_t(1));
  row_set->moveToBegin();
  auto first_row = row_set->getNextRow(false, false);
  CHECK_EQ(first_row.size(), size_t(1));
  auto scalar_tv = boost::get<ScalarTargetValue>(&first_row[0]);
  auto ti = rex_subquery->getType();
  if (ti.is_string()) {
    throw std::runtime_error("Scalar sub-queries which return strings not supported");
  }
  Datum d{0};
  bool is_null_const{false};
  std::tie(d, is_null_const) = datum_from_scalar_tv(scalar_tv, ti);
  return makeExpr<Analyzer::Constant>(ti, is_null_const, d);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateInput(
    const RexInput* rex_input) const {
  const auto source = rex_input->getSourceNode();
  const auto it_rte_idx = input_to_nest_level_.find(source);
  CHECK(it_rte_idx != input_to_nest_level_.end())
      << "Not found in input_to_nest_level_, source=" << source->toString();
  const int rte_idx = it_rte_idx->second;
  const auto scan_source = dynamic_cast<const RelScan*>(source);
  const auto& in_metainfo = source->getOutputMetainfo();
  if (scan_source) {
    // We're at leaf (scan) level and not supposed to have input metadata,
    // the name and type information come directly from the catalog.
    CHECK(in_metainfo.empty());
    const auto table_desc = scan_source->getTableDescriptor();
    const auto cd =
        cat_.getMetadataForColumnBySpi(table_desc->tableId, rex_input->getIndex() + 1);
    CHECK(cd);
    auto col_ti = cd->columnType;
    if (col_ti.is_string()) {
      col_ti.set_type(kTEXT);
    }
    if (cd->isVirtualCol) {
      // TODO(alex): remove at some point, we only need this fixup for backwards
      // compatibility with old imported data
      CHECK_EQ("rowid", cd->columnName);
      col_ti.set_size(8);
    }
    CHECK_LE(static_cast<size_t>(rte_idx), join_types_.size());
    if (rte_idx > 0 && join_types_[rte_idx - 1] == JoinType::LEFT) {
      col_ti.set_notnull(false);
    }
    return std::make_shared<Analyzer::ColumnVar>(
        col_ti, table_desc->tableId, cd->columnId, rte_idx);
  }
  CHECK(!in_metainfo.empty()) << "for " << source->toString();
  CHECK_GE(rte_idx, 0);
  const size_t col_id = rex_input->getIndex();
  CHECK_LT(col_id, in_metainfo.size());
  auto col_ti = in_metainfo[col_id].get_type_info();

  if (join_types_.size() > 0) {
    CHECK_LE(static_cast<size_t>(rte_idx), join_types_.size());
    if (rte_idx > 0 && join_types_[rte_idx - 1] == JoinType::LEFT) {
      col_ti.set_notnull(false);
    }
  }

  return std::make_shared<Analyzer::ColumnVar>(col_ti, -source->getId(), col_id, rte_idx);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateUoper(
    const RexOperator* rex_operator) const {
  CHECK_EQ(size_t(1), rex_operator->size());
  const auto operand_expr = translateScalarRex(rex_operator->getOperand(0));
  const auto sql_op = rex_operator->getOperator();
  switch (sql_op) {
    case kCAST: {
      const auto& target_ti = rex_operator->getType();
      CHECK_NE(kNULLT, target_ti.get_type());
      const auto& operand_ti = operand_expr->get_type_info();
      if (operand_ti.is_string() && target_ti.is_string()) {
        return operand_expr;
      }
      if (target_ti.is_time() ||
          operand_ti
              .is_string()) {  // TODO(alex): check and unify with the rest of the cases
        // Do not propogate encoding on small dates
        return target_ti.is_date_in_days()
                   ? operand_expr->add_cast(SQLTypeInfo(kDATE, false))
                   : operand_expr->add_cast(target_ti);
      }
      if (!operand_ti.is_string() && target_ti.is_string()) {
        return operand_expr->add_cast(target_ti);
      }

      return std::make_shared<Analyzer::UOper>(target_ti, false, sql_op, operand_expr);
    }
    case kNOT:
    case kISNULL: {
      return std::make_shared<Analyzer::UOper>(kBOOLEAN, sql_op, operand_expr);
    }
    case kISNOTNULL: {
      auto is_null = std::make_shared<Analyzer::UOper>(kBOOLEAN, kISNULL, operand_expr);
      return std::make_shared<Analyzer::UOper>(kBOOLEAN, kNOT, is_null);
    }
    case kMINUS: {
      const auto& ti = operand_expr->get_type_info();
      return std::make_shared<Analyzer::UOper>(ti, false, kUMINUS, operand_expr);
    }
    case kUNNEST: {
      const auto& ti = operand_expr->get_type_info();
      CHECK(ti.is_array());
      return makeExpr<Analyzer::UOper>(ti.get_elem_type(), false, kUNNEST, operand_expr);
    }
    default:
      CHECK(false);
  }
  return nullptr;
}

namespace {

std::shared_ptr<Analyzer::Expr> get_in_values_expr(std::shared_ptr<Analyzer::Expr> arg,
                                                   const ResultSet& val_set) {
  if (!result_set::can_use_parallel_algorithms(val_set)) {
    return nullptr;
  }
  if (val_set.rowCount() > 5000000 && g_enable_watchdog) {
    throw std::runtime_error(
        "Unable to handle 'expr IN (subquery)', subquery returned 5M+ rows.");
  }
  std::list<std::shared_ptr<Analyzer::Expr>> value_exprs;
  const size_t fetcher_count = cpu_threads();
  std::vector<std::list<std::shared_ptr<Analyzer::Expr>>> expr_set(
      fetcher_count, std::list<std::shared_ptr<Analyzer::Expr>>());
  std::vector<utils::future<void>> fetcher_threads;
  const auto& ti = arg->get_type_info();
  const auto entry_count = val_set.entryCount();
  for (size_t i = 0,
              start_entry = 0,
              stride = (entry_count + fetcher_count - 1) / fetcher_count;
       i < fetcher_count && start_entry < entry_count;
       ++i, start_entry += stride) {
    const auto end_entry = std::min(start_entry + stride, entry_count);
    fetcher_threads.push_back(utils::async(
        [&](std::list<std::shared_ptr<Analyzer::Expr>>& in_vals,
            const size_t start,
            const size_t end) {
          for (auto index = start; index < end; ++index) {
            auto row = val_set.getRowAt(index);
            if (row.empty()) {
              continue;
            }
            auto scalar_tv = boost::get<ScalarTargetValue>(&row[0]);
            Datum d{0};
            bool is_null_const{false};
            std::tie(d, is_null_const) = datum_from_scalar_tv(scalar_tv, ti);
            if (ti.is_string() && ti.get_compression() != kENCODING_NONE) {
              auto ti_none_encoded = ti;
              ti_none_encoded.set_compression(kENCODING_NONE);
              auto none_encoded_string =
                  makeExpr<Analyzer::Constant>(ti, is_null_const, d);
              auto dict_encoded_string = std::make_shared<Analyzer::UOper>(
                  ti, false, kCAST, none_encoded_string);
              in_vals.push_back(dict_encoded_string);
            } else {
              in_vals.push_back(makeExpr<Analyzer::Constant>(ti, is_null_const, d));
            }
          }
        },
        std::ref(expr_set[i]),
        start_entry,
        end_entry));
  }
  for (auto& child : fetcher_threads) {
    child.get();
  }

  val_set.moveToBegin();
  for (auto& exprs : expr_set) {
    value_exprs.splice(value_exprs.end(), exprs);
  }
  return makeExpr<Analyzer::InValues>(arg, value_exprs);
}

}  // namespace

// Creates an Analyzer expression for an IN subquery which subsequently goes through the
// regular Executor::codegen() mechanism. The creation of the expression out of
// subquery's result set is parallelized whenever possible. In addition, take advantage
// of additional information that elements in the right hand side are constants; see
// getInIntegerSetExpr().
std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateInOper(
    const RexOperator* rex_operator) const {
  if (just_explain_) {
    throw std::runtime_error("EXPLAIN is not supported with sub-queries");
  }
  CHECK(rex_operator->size() == 2);
  const auto lhs = translateScalarRex(rex_operator->getOperand(0));
  const auto rhs = rex_operator->getOperand(1);
  const auto rex_subquery = dynamic_cast<const RexSubQuery*>(rhs);
  CHECK(rex_subquery);
  auto ti = lhs->get_type_info();
  auto result = rex_subquery->getExecutionResult();
  CHECK(result);
  auto& row_set = result->getRows();
  CHECK_EQ(size_t(1), row_set->colCount());
  const auto& rhs_ti = row_set->getColType(0);
  if (rhs_ti.get_type() != ti.get_type()) {
    throw std::runtime_error(
        "The two sides of the IN operator must have the same type; found " +
        ti.get_type_name() + " and " + rhs_ti.get_type_name());
  }
  row_set->moveToBegin();
  if (row_set->entryCount() > 10000) {
    std::shared_ptr<Analyzer::Expr> expr;
    if ((ti.is_integer() || (ti.is_string() && ti.get_compression() == kENCODING_DICT)) &&
        !row_set->getQueryMemDesc().didOutputColumnar()) {
      expr = getInIntegerSetExpr(lhs, *row_set);
      // Handle the highly unlikely case when the InIntegerSet ended up being tiny.
      // Just let it fall through the usual InValues path at the end of this method,
      // its codegen knows to use inline comparisons for few values.
      if (expr && std::static_pointer_cast<Analyzer::InIntegerSet>(expr)
                          ->get_value_list()
                          .size() <= 100) {
        expr = nullptr;
      }
    } else {
      expr = get_in_values_expr(lhs, *row_set);
    }
    if (expr) {
      return expr;
    }
  }
  std::list<std::shared_ptr<Analyzer::Expr>> value_exprs;
  while (true) {
    auto row = row_set->getNextRow(true, false);
    if (row.empty()) {
      break;
    }
    if (g_enable_watchdog && value_exprs.size() >= 10000) {
      throw std::runtime_error(
          "Unable to handle 'expr IN (subquery)', subquery returned 10000+ rows.");
    }
    auto scalar_tv = boost::get<ScalarTargetValue>(&row[0]);
    Datum d{0};
    bool is_null_const{false};
    std::tie(d, is_null_const) = datum_from_scalar_tv(scalar_tv, ti);
    if (ti.is_string() && ti.get_compression() != kENCODING_NONE) {
      auto ti_none_encoded = ti;
      ti_none_encoded.set_compression(kENCODING_NONE);
      auto none_encoded_string = makeExpr<Analyzer::Constant>(ti, is_null_const, d);
      auto dict_encoded_string =
          std::make_shared<Analyzer::UOper>(ti, false, kCAST, none_encoded_string);
      value_exprs.push_back(dict_encoded_string);
    } else {
      value_exprs.push_back(makeExpr<Analyzer::Constant>(ti, is_null_const, d));
    }
  }
  return makeExpr<Analyzer::InValues>(lhs, value_exprs);
}

namespace {

const size_t g_max_integer_set_size{1 << 25};

void fill_dictionary_encoded_in_vals(
    std::vector<int64_t>& in_vals,
    std::atomic<size_t>& total_in_vals_count,
    const ResultSet* values_rowset,
    const std::pair<int64_t, int64_t> values_rowset_slice,
    const StringDictionaryProxy* source_dict,
    const StringDictionaryProxy* dest_dict,
    const int64_t needle_null_val) {
  CHECK(in_vals.empty());
  bool dicts_are_equal = source_dict == dest_dict;
  for (auto index = values_rowset_slice.first; index < values_rowset_slice.second;
       ++index) {
    const auto row = values_rowset->getOneColRow(index);
    if (UNLIKELY(!row.valid)) {
      continue;
    }
    if (dicts_are_equal) {
      in_vals.push_back(row.value);
    } else {
      const int string_id =
          row.value == needle_null_val
              ? needle_null_val
              : dest_dict->getIdOfString(source_dict->getString(row.value));
      if (string_id != StringDictionary::INVALID_STR_ID) {
        in_vals.push_back(string_id);
      }
    }
    if (UNLIKELY(g_enable_watchdog && (in_vals.size() & 1023) == 0 &&
                 total_in_vals_count.fetch_add(1024) >= g_max_integer_set_size)) {
      throw std::runtime_error(
          "Unable to handle 'expr IN (subquery)', subquery returned 30M+ rows.");
    }
  }
}

void fill_integer_in_vals(std::vector<int64_t>& in_vals,
                          std::atomic<size_t>& total_in_vals_count,
                          const ResultSet* values_rowset,
                          const std::pair<int64_t, int64_t> values_rowset_slice) {
  CHECK(in_vals.empty());
  for (auto index = values_rowset_slice.first; index < values_rowset_slice.second;
       ++index) {
    const auto row = values_rowset->getOneColRow(index);
    if (row.valid) {
      in_vals.push_back(row.value);
      if (UNLIKELY(g_enable_watchdog && (in_vals.size() & 1023) == 0 &&
                   total_in_vals_count.fetch_add(1024) >= g_max_integer_set_size)) {
        throw std::runtime_error(
            "Unable to handle 'expr IN (subquery)', subquery returned 30M+ rows.");
      }
    }
  }
}

// Multi-node counterpart of the other version. Saves round-trips, which is crucial
// for a big right-hand side result. It only handles physical string dictionary ids,
// therefore it won't be able to handle a right-hand side sub-query with a CASE
// returning literals on some branches. That case isn't hard too handle either, but
// it's not clear it's actually important in practice.
// RelAlgTranslator::getInIntegerSetExpr makes sure, by checking the encodings, that
// this function isn't called in such cases.
void fill_dictionary_encoded_in_vals(
    std::vector<int64_t>& in_vals,
    std::atomic<size_t>& total_in_vals_count,
    const ResultSet* values_rowset,
    const std::pair<int64_t, int64_t> values_rowset_slice,
    const std::vector<LeafHostInfo>& leaf_hosts,
    const DictRef source_dict_ref,
    const DictRef dest_dict_ref,
    const int32_t dest_generation,
    const int64_t needle_null_val) {
  CHECK(in_vals.empty());
  std::vector<int32_t> source_ids;
  source_ids.reserve(values_rowset->entryCount());
  bool has_nulls = false;
  if (source_dict_ref == dest_dict_ref) {
    in_vals.reserve(values_rowset_slice.second - values_rowset_slice.first +
                    1);  // Add 1 to cover interval
    for (auto index = values_rowset_slice.first; index < values_rowset_slice.second;
         ++index) {
      const auto row = values_rowset->getOneColRow(index);
      if (!row.valid) {
        continue;
      }
      if (row.value != needle_null_val) {
        in_vals.push_back(row.value);
        if (UNLIKELY(g_enable_watchdog && (in_vals.size() & 1023) == 0 &&
                     total_in_vals_count.fetch_add(1024) >= g_max_integer_set_size)) {
          throw std::runtime_error(
              "Unable to handle 'expr IN (subquery)', subquery returned 30M+ rows.");
        }
      } else {
        has_nulls = true;
      }
    }
    if (has_nulls) {
      in_vals.push_back(
          needle_null_val);  // we've deduped null values as an optimization, although
                             // this is not required by consumer
    }
    return;
  }
  // Code path below is for when dictionaries are not shared
  for (auto index = values_rowset_slice.first; index < values_rowset_slice.second;
       ++index) {
    const auto row = values_rowset->getOneColRow(index);
    if (row.valid) {
      if (row.value != needle_null_val) {
        source_ids.push_back(row.value);
      } else {
        has_nulls = true;
      }
    }
  }
  std::vector<int32_t> dest_ids;
  translate_string_ids(dest_ids,
                       leaf_hosts.front(),
                       dest_dict_ref,
                       source_ids,
                       source_dict_ref,
                       dest_generation);
  CHECK_EQ(dest_ids.size(), source_ids.size());
  in_vals.reserve(dest_ids.size() + (has_nulls ? 1 : 0));
  if (has_nulls) {
    in_vals.push_back(needle_null_val);
  }
  for (const int32_t dest_id : dest_ids) {
    if (dest_id != StringDictionary::INVALID_STR_ID) {
      in_vals.push_back(dest_id);
      if (UNLIKELY(g_enable_watchdog && (in_vals.size() & 1023) == 0 &&
                   total_in_vals_count.fetch_add(1024) >= g_max_integer_set_size)) {
        throw std::runtime_error(
            "Unable to handle 'expr IN (subquery)', subquery returned 30M+ rows.");
      }
    }
  }
}

}  // namespace

// The typical IN subquery involves either dictionary-encoded strings or integers.
// Analyzer::InValues is a very heavy representation of the right hand side of such
// a query since we already know the right hand would be a list of Analyzer::Constant
// shared pointers. We can avoid the big overhead of each Analyzer::Constant and the
// refcounting associated with shared pointers by creating an abbreviated InIntegerSet
// representation of the IN expression which takes advantage of the this information.
std::shared_ptr<Analyzer::Expr> RelAlgTranslator::getInIntegerSetExpr(
    std::shared_ptr<Analyzer::Expr> arg,
    const ResultSet& val_set) const {
  if (!result_set::can_use_parallel_algorithms(val_set)) {
    return nullptr;
  }
  std::vector<int64_t> value_exprs;
  const size_t fetcher_count = cpu_threads();
  std::vector<std::vector<int64_t>> expr_set(fetcher_count);
  std::vector<utils::future<void>> fetcher_threads;
  const auto& arg_type = arg->get_type_info();
  const auto entry_count = val_set.entryCount();
  CHECK_EQ(size_t(1), val_set.colCount());
  const auto& col_type = val_set.getColType(0);
  if (g_cluster && arg_type.is_string() &&
      (col_type.get_comp_param() <= 0 || arg_type.get_comp_param() <= 0)) {
    // Skip this case for now, see comment for fill_dictionary_encoded_in_vals.
    return nullptr;
  }
  std::atomic<size_t> total_in_vals_count{0};
  for (size_t i = 0,
              start_entry = 0,
              stride = (entry_count + fetcher_count - 1) / fetcher_count;
       i < fetcher_count && start_entry < entry_count;
       ++i, start_entry += stride) {
    expr_set[i].reserve(entry_count / fetcher_count);
    const auto end_entry = std::min(start_entry + stride, entry_count);
    if (arg_type.is_string()) {
      CHECK_EQ(kENCODING_DICT, arg_type.get_compression());
      // const int32_t dest_dict_id = arg_type.get_comp_param();
      // const int32_t source_dict_id = col_type.get_comp_param();
      const DictRef dest_dict_ref(arg_type.get_comp_param(), cat_.getDatabaseId());
      const DictRef source_dict_ref(col_type.get_comp_param(), cat_.getDatabaseId());
      const auto dd = executor_->getStringDictionaryProxy(
          arg_type.get_comp_param(), val_set.getRowSetMemOwner(), true);
      const auto sd = executor_->getStringDictionaryProxy(
          col_type.get_comp_param(), val_set.getRowSetMemOwner(), true);
      CHECK(sd);
      const auto needle_null_val = inline_int_null_val(arg_type);
      fetcher_threads.push_back(utils::async(
          [this,
           &val_set,
           &total_in_vals_count,
           sd,
           dd,
           source_dict_ref,
           dest_dict_ref,
           needle_null_val](
              std::vector<int64_t>& in_vals, const size_t start, const size_t end) {
            if (g_cluster) {
              CHECK_GE(dd->getGeneration(), 0);
              fill_dictionary_encoded_in_vals(in_vals,
                                              total_in_vals_count,
                                              &val_set,
                                              {start, end},
                                              cat_.getStringDictionaryHosts(),
                                              source_dict_ref,
                                              dest_dict_ref,
                                              dd->getGeneration(),
                                              needle_null_val);
            } else {
              fill_dictionary_encoded_in_vals(in_vals,
                                              total_in_vals_count,
                                              &val_set,
                                              {start, end},
                                              sd,
                                              dd,
                                              needle_null_val);
            }
          },
          std::ref(expr_set[i]),
          start_entry,
          end_entry));
    } else {
      CHECK(arg_type.is_integer());
      fetcher_threads.push_back(utils::async(
          [&val_set, &total_in_vals_count](
              std::vector<int64_t>& in_vals, const size_t start, const size_t end) {
            fill_integer_in_vals(in_vals, total_in_vals_count, &val_set, {start, end});
          },
          std::ref(expr_set[i]),
          start_entry,
          end_entry));
    }
  }
  for (auto& child : fetcher_threads) {
    child.get();
  }

  val_set.moveToBegin();
  value_exprs.reserve(entry_count);
  for (auto& exprs : expr_set) {
    value_exprs.insert(value_exprs.end(), exprs.begin(), exprs.end());
  }
  return makeExpr<Analyzer::InIntegerSet>(
      arg, value_exprs, arg_type.get_notnull() && col_type.get_notnull());
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateOper(
    const RexOperator* rex_operator) const {
  CHECK_GT(rex_operator->size(), size_t(0));
  if (rex_operator->size() == 1) {
    return translateUoper(rex_operator);
  }
  const auto sql_op = rex_operator->getOperator();
  if (sql_op == kIN) {
    return translateInOper(rex_operator);
  }
  if (sql_op == kMINUS || sql_op == kPLUS) {
    auto date_plus_minus = translateDatePlusMinus(rex_operator);
    if (date_plus_minus) {
      return date_plus_minus;
    }
  }
  if (sql_op == kOVERLAPS) {
    return translateOverlapsOper(rex_operator);
  } else if (IS_COMPARISON(sql_op)) {
    auto geo_comp = translateGeoComparison(rex_operator);
    if (geo_comp) {
      return geo_comp;
    }
  }
  auto lhs = translateScalarRex(rex_operator->getOperand(0));
  for (size_t i = 1; i < rex_operator->size(); ++i) {
    std::shared_ptr<Analyzer::Expr> rhs;
    SQLQualifier sql_qual{kONE};
    const auto rhs_op = rex_operator->getOperand(i);
    std::tie(rhs, sql_qual) = get_quantified_rhs(rhs_op, *this);
    if (!rhs) {
      rhs = translateScalarRex(rhs_op);
    }
    CHECK(rhs);
    lhs = Parser::OperExpr::normalize(sql_op, sql_qual, lhs, rhs);
  }
  return lhs;
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateOverlapsOper(
    const RexOperator* rex_operator) const {
  const auto sql_op = rex_operator->getOperator();
  CHECK(sql_op == kOVERLAPS);

  const auto lhs = translateScalarRex(rex_operator->getOperand(0));
  const auto lhs_ti = lhs->get_type_info();
  if (lhs_ti.is_geometry()) {
    return translateGeoOverlapsOper(rex_operator);
  } else {
    throw std::runtime_error(
        "Overlaps equivalence is currently only supported for geospatial types");
  }
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateCase(
    const RexCase* rex_case) const {
  std::shared_ptr<Analyzer::Expr> else_expr;
  std::list<std::pair<std::shared_ptr<Analyzer::Expr>, std::shared_ptr<Analyzer::Expr>>>
      expr_list;
  for (size_t i = 0; i < rex_case->branchCount(); ++i) {
    const auto when_expr = translateScalarRex(rex_case->getWhen(i));
    const auto then_expr = translateScalarRex(rex_case->getThen(i));
    expr_list.emplace_back(when_expr, then_expr);
  }
  if (rex_case->getElse()) {
    else_expr = translateScalarRex(rex_case->getElse());
  }
  return Parser::CaseExpr::normalize(expr_list, else_expr);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateLike(
    const RexFunctionOperator* rex_function) const {
  CHECK(rex_function->size() == 2 || rex_function->size() == 3);
  const auto arg = translateScalarRex(rex_function->getOperand(0));
  const auto like = translateScalarRex(rex_function->getOperand(1));
  if (!std::dynamic_pointer_cast<const Analyzer::Constant>(like)) {
    throw std::runtime_error("The matching pattern must be a literal.");
  }
  const auto escape = (rex_function->size() == 3)
                          ? translateScalarRex(rex_function->getOperand(2))
                          : nullptr;
  const bool is_ilike = rex_function->getName() == "PG_ILIKE"sv;
  return Parser::LikeExpr::get(arg, like, escape, is_ilike, false);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateRegexp(
    const RexFunctionOperator* rex_function) const {
  CHECK(rex_function->size() == 2 || rex_function->size() == 3);
  const auto arg = translateScalarRex(rex_function->getOperand(0));
  const auto pattern = translateScalarRex(rex_function->getOperand(1));
  if (!std::dynamic_pointer_cast<const Analyzer::Constant>(pattern)) {
    throw std::runtime_error("The matching pattern must be a literal.");
  }
  const auto escape = (rex_function->size() == 3)
                          ? translateScalarRex(rex_function->getOperand(2))
                          : nullptr;
  return Parser::RegexpExpr::get(arg, pattern, escape, false);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateLikely(
    const RexFunctionOperator* rex_function) const {
  CHECK(rex_function->size() == 1);
  const auto arg = translateScalarRex(rex_function->getOperand(0));
  return makeExpr<Analyzer::LikelihoodExpr>(arg, 0.9375);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateUnlikely(
    const RexFunctionOperator* rex_function) const {
  CHECK(rex_function->size() == 1);
  const auto arg = translateScalarRex(rex_function->getOperand(0));
  return makeExpr<Analyzer::LikelihoodExpr>(arg, 0.0625);
}

namespace {

inline void validate_datetime_datepart_argument(
    const std::shared_ptr<Analyzer::Constant> literal_expr) {
  if (!literal_expr || literal_expr->get_is_null()) {
    throw std::runtime_error("The 'DatePart' argument must be a not 'null' literal.");
  }
}

}  // namespace

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateExtract(
    const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(2), rex_function->size());
  const auto timeunit = translateScalarRex(rex_function->getOperand(0));
  const auto timeunit_lit = std::dynamic_pointer_cast<Analyzer::Constant>(timeunit);
  validate_datetime_datepart_argument(timeunit_lit);
  const auto from_expr = translateScalarRex(rex_function->getOperand(1));
  const bool is_date_trunc = rex_function->getName() == "PG_DATE_TRUNC"sv;
  if (is_date_trunc) {
    return DateTruncExpr::generate(from_expr, *timeunit_lit->get_constval().stringval);
  } else {
    return ExtractExpr::generate(from_expr, *timeunit_lit->get_constval().stringval);
  }
}

namespace {

std::shared_ptr<Analyzer::Constant> makeNumericConstant(const SQLTypeInfo& ti,
                                                        const long val) {
  CHECK(ti.is_number());
  Datum datum{0};
  switch (ti.get_type()) {
    case kTINYINT: {
      datum.tinyintval = val;
      break;
    }
    case kSMALLINT: {
      datum.smallintval = val;
      break;
    }
    case kINT: {
      datum.intval = val;
      break;
    }
    case kBIGINT: {
      datum.bigintval = val;
      break;
    }
    case kDECIMAL:
    case kNUMERIC: {
      datum.bigintval = val * exp_to_scale(ti.get_scale());
      break;
    }
    case kFLOAT: {
      datum.floatval = val;
      break;
    }
    case kDOUBLE: {
      datum.doubleval = val;
      break;
    }
    default:
      CHECK(false);
  }
  return makeExpr<Analyzer::Constant>(ti, false, datum);
}

}  // namespace

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateDateadd(
    const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(3), rex_function->size());
  const auto timeunit = translateScalarRex(rex_function->getOperand(0));
  const auto timeunit_lit = std::dynamic_pointer_cast<Analyzer::Constant>(timeunit);
  validate_datetime_datepart_argument(timeunit_lit);
  const auto number_units = translateScalarRex(rex_function->getOperand(1));
  const auto number_units_const =
      std::dynamic_pointer_cast<Analyzer::Constant>(number_units);
  if (number_units_const && number_units_const->get_is_null()) {
    throw std::runtime_error("The 'Interval' argument literal must not be 'null'.");
  }
  const auto cast_number_units = number_units->add_cast(SQLTypeInfo(kBIGINT, false));
  const auto datetime = translateScalarRex(rex_function->getOperand(2));
  const auto& datetime_ti = datetime->get_type_info();
  if (datetime_ti.get_type() == kTIME) {
    throw std::runtime_error("DateAdd operation not supported for TIME.");
  }
  const auto& field = to_dateadd_field(*timeunit_lit->get_constval().stringval);
  const int dim = datetime_ti.get_dimension();
  return makeExpr<Analyzer::DateaddExpr>(
      SQLTypeInfo(kTIMESTAMP, dim, 0, false), field, cast_number_units, datetime);
}

namespace {

std::string get_datetimeplus_rewrite_funcname(const SQLOps& op) {
  CHECK(op == kPLUS);
  return "DATETIME_PLUS"s;
}

}  // namespace

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateDatePlusMinus(
    const RexOperator* rex_operator) const {
  if (rex_operator->size() != 2) {
    return nullptr;
  }
  const auto datetime = translateScalarRex(rex_operator->getOperand(0));
  const auto datetime_ti = datetime->get_type_info();
  if (!datetime_ti.is_timestamp() && !datetime_ti.is_date()) {
    if (datetime_ti.get_type() == kTIME) {
      throw std::runtime_error("DateTime addition/subtraction not supported for TIME.");
    }
    return nullptr;
  }
  const auto rhs = translateScalarRex(rex_operator->getOperand(1));
  const auto rhs_ti = rhs->get_type_info();
  if (rhs_ti.get_type() == kTIMESTAMP || rhs_ti.get_type() == kDATE) {
    if (datetime_ti.is_high_precision_timestamp() ||
        rhs_ti.is_high_precision_timestamp()) {
      throw std::runtime_error(
          "High Precision timestamps are not supported for TIMESTAMPDIFF operation. "
          "Use "
          "DATEDIFF.");
    }
    auto bigint_ti = SQLTypeInfo(kBIGINT, false);
    const auto& rex_operator_ti = rex_operator->getType();
    const auto datediff_field =
        (rex_operator_ti.get_type() == kINTERVAL_DAY_TIME) ? dtSECOND : dtMONTH;
    auto result =
        makeExpr<Analyzer::DatediffExpr>(bigint_ti, datediff_field, rhs, datetime);
    // multiply 1000 to result since expected result should be in millisecond precision.
    if (rex_operator_ti.get_type() == kINTERVAL_DAY_TIME) {
      return makeExpr<Analyzer::BinOper>(bigint_ti.get_type(),
                                         kMULTIPLY,
                                         kONE,
                                         result,
                                         makeNumericConstant(bigint_ti, 1000));
    } else {
      return result;
    }
  }
  const auto op = rex_operator->getOperator();
  if (op == kPLUS) {
    std::vector<std::shared_ptr<Analyzer::Expr>> args = {datetime, rhs};
    auto dt_plus = makeExpr<Analyzer::FunctionOper>(
        datetime_ti, get_datetimeplus_rewrite_funcname(op), args);
    const auto date_trunc = rewrite_to_date_trunc(dt_plus.get());
    if (date_trunc) {
      return date_trunc;
    }
  }
  const auto interval = fold_expr(rhs.get());
  auto interval_ti = interval->get_type_info();
  auto bigint_ti = SQLTypeInfo(kBIGINT, false);
  const auto interval_lit = std::dynamic_pointer_cast<Analyzer::Constant>(interval);
  if (interval_ti.get_type() == kINTERVAL_DAY_TIME) {
    std::shared_ptr<Analyzer::Expr> interval_sec;
    if (interval_lit) {
      interval_sec =
          makeNumericConstant(bigint_ti,
                              (op == kMINUS ? -interval_lit->get_constval().bigintval
                                            : interval_lit->get_constval().bigintval) /
                                  1000);
    } else {
      interval_sec = makeExpr<Analyzer::BinOper>(bigint_ti.get_type(),
                                                 kDIVIDE,
                                                 kONE,
                                                 interval,
                                                 makeNumericConstant(bigint_ti, 1000));
      if (op == kMINUS) {
        interval_sec =
            std::make_shared<Analyzer::UOper>(bigint_ti, false, kUMINUS, interval_sec);
      }
    }
    return makeExpr<Analyzer::DateaddExpr>(datetime_ti, daSECOND, interval_sec, datetime);
  }
  CHECK(interval_ti.get_type() == kINTERVAL_YEAR_MONTH);
  const auto interval_months = op == kMINUS ? std::make_shared<Analyzer::UOper>(
                                                  bigint_ti, false, kUMINUS, interval)
                                            : interval;
  return makeExpr<Analyzer::DateaddExpr>(datetime_ti, daMONTH, interval_months, datetime);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateDatediff(
    const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(3), rex_function->size());
  const auto timeunit = translateScalarRex(rex_function->getOperand(0));
  const auto timeunit_lit = std::dynamic_pointer_cast<Analyzer::Constant>(timeunit);
  validate_datetime_datepart_argument(timeunit_lit);
  const auto start = translateScalarRex(rex_function->getOperand(1));
  const auto end = translateScalarRex(rex_function->getOperand(2));
  const auto field = to_datediff_field(*timeunit_lit->get_constval().stringval);
  return makeExpr<Analyzer::DatediffExpr>(SQLTypeInfo(kBIGINT, false), field, start, end);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateDatepart(
    const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(2), rex_function->size());
  const auto timeunit = translateScalarRex(rex_function->getOperand(0));
  const auto timeunit_lit = std::dynamic_pointer_cast<Analyzer::Constant>(timeunit);
  validate_datetime_datepart_argument(timeunit_lit);
  const auto from_expr = translateScalarRex(rex_function->getOperand(1));
  return ExtractExpr::generate(
      from_expr, to_datepart_field(*timeunit_lit->get_constval().stringval));
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateLength(
    const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(1), rex_function->size());
  const auto str_arg = translateScalarRex(rex_function->getOperand(0));
  return makeExpr<Analyzer::CharLengthExpr>(str_arg->decompress(),
                                            rex_function->getName() == "CHAR_LENGTH"sv);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateKeyForString(
    const RexFunctionOperator* rex_function) const {
  const auto& args = translateFunctionArgs(rex_function);
  CHECK_EQ(size_t(1), args.size());
  const auto expr = dynamic_cast<Analyzer::Expr*>(args[0].get());
  if (nullptr == expr || !expr->get_type_info().is_string() ||
      expr->get_type_info().is_varlen()) {
    throw std::runtime_error(rex_function->getName() +
                             " expects a dictionary encoded text column.");
  }
  return makeExpr<Analyzer::KeyForStringExpr>(args[0]);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateSampleRatio(
    const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(1), rex_function->size());
  auto arg = translateScalarRex(rex_function->getOperand(0));
  const auto& arg_ti = arg->get_type_info();
  if (arg_ti.get_type() != kDOUBLE) {
    const auto& double_ti = SQLTypeInfo(kDOUBLE, arg_ti.get_notnull());
    arg = arg->add_cast(double_ti);
  }
  return makeExpr<Analyzer::SampleRatioExpr>(arg);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateCurrentUser(
    const RexFunctionOperator* rex_function) const {
  std::string user{"SESSIONLESS_USER"};
  if (query_state_) {
    user = query_state_->getConstSessionInfo()->get_currentUser().userName;
  }
  return Parser::UserLiteral::get(user);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateLower(
    const RexFunctionOperator* rex_function) const {
  const auto& args = translateFunctionArgs(rex_function);
  CHECK_EQ(size_t(1), args.size());
  CHECK(args[0]);

  if (args[0]->get_type_info().is_dict_encoded_string() ||
      dynamic_cast<Analyzer::Constant*>(args[0].get())) {
    return makeExpr<Analyzer::LowerExpr>(args[0]);
  }

  throw std::runtime_error(rex_function->getName() +
                           " expects a dictionary encoded text column or a literal.");
}

Analyzer::ExpressionPtr RelAlgTranslator::translateCardinality(
    const RexFunctionOperator* rex_function) const {
  const auto ret_ti = rex_function->getType();
  const auto arg = translateScalarRex(rex_function->getOperand(0));
  const auto arg_ti = arg->get_type_info();
  if (!arg_ti.is_array()) {
    throw std::runtime_error(rex_function->getName() + " expects an array expression.");
  }
  if (arg_ti.get_subtype() == kARRAY) {
    throw std::runtime_error(rex_function->getName() +
                             " expects one-dimension array expression.");
  }
  const auto array_size = arg_ti.get_size();
  const auto array_elem_size = arg_ti.get_elem_type().get_array_context_logical_size();

  if (array_size > 0) {
    if (array_elem_size <= 0) {
      throw std::runtime_error(rex_function->getName() +
                               ": unexpected array element type.");
    }
    // Return cardinality of a fixed length array
    return makeNumericConstant(ret_ti, array_size / array_elem_size);
  }
  // Variable length array cardinality will be calculated at runtime
  return makeExpr<Analyzer::CardinalityExpr>(arg);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateItem(
    const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(2), rex_function->size());
  const auto base = translateScalarRex(rex_function->getOperand(0));
  const auto index = translateScalarRex(rex_function->getOperand(1));
  return makeExpr<Analyzer::BinOper>(
      base->get_type_info().get_elem_type(), false, kARRAY_AT, kONE, base, index);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateNow() const {
  return Parser::TimestampLiteral::get(now_);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateDatetime(
    const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(1), rex_function->size());
  const auto arg = translateScalarRex(rex_function->getOperand(0));
  const auto arg_lit = std::dynamic_pointer_cast<Analyzer::Constant>(arg);
  const std::string datetime_err{R"(Only DATETIME('NOW') supported for now.)"};
  if (!arg_lit || arg_lit->get_is_null()) {
    throw std::runtime_error(datetime_err);
  }
  CHECK(arg_lit->get_type_info().is_string());
  if (*arg_lit->get_constval().stringval != "NOW"sv) {
    throw std::runtime_error(datetime_err);
  }
  return translateNow();
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateAbs(
    const RexFunctionOperator* rex_function) const {
  std::list<std::pair<std::shared_ptr<Analyzer::Expr>, std::shared_ptr<Analyzer::Expr>>>
      expr_list;
  CHECK_EQ(size_t(1), rex_function->size());
  const auto operand = translateScalarRex(rex_function->getOperand(0));
  const auto& operand_ti = operand->get_type_info();
  CHECK(operand_ti.is_number());
  const auto zero = makeNumericConstant(operand_ti, 0);
  const auto lt_zero = makeExpr<Analyzer::BinOper>(kBOOLEAN, kLT, kONE, operand, zero);
  const auto uminus_operand =
      makeExpr<Analyzer::UOper>(operand_ti.get_type(), kUMINUS, operand);
  expr_list.emplace_back(lt_zero, uminus_operand);
  return makeExpr<Analyzer::CaseExpr>(operand_ti, false, expr_list, operand);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateSign(
    const RexFunctionOperator* rex_function) const {
  std::list<std::pair<std::shared_ptr<Analyzer::Expr>, std::shared_ptr<Analyzer::Expr>>>
      expr_list;
  CHECK_EQ(size_t(1), rex_function->size());
  const auto operand = translateScalarRex(rex_function->getOperand(0));
  const auto& operand_ti = operand->get_type_info();
  CHECK(operand_ti.is_number());
  const auto zero = makeNumericConstant(operand_ti, 0);
  const auto lt_zero = makeExpr<Analyzer::BinOper>(kBOOLEAN, kLT, kONE, operand, zero);
  expr_list.emplace_back(lt_zero, makeNumericConstant(operand_ti, -1));
  const auto eq_zero = makeExpr<Analyzer::BinOper>(kBOOLEAN, kEQ, kONE, operand, zero);
  expr_list.emplace_back(eq_zero, makeNumericConstant(operand_ti, 0));
  const auto gt_zero = makeExpr<Analyzer::BinOper>(kBOOLEAN, kGT, kONE, operand, zero);
  expr_list.emplace_back(gt_zero, makeNumericConstant(operand_ti, 1));
  return makeExpr<Analyzer::CaseExpr>(
      operand_ti,
      false,
      expr_list,
      makeExpr<Analyzer::Constant>(operand_ti, true, Datum{0}));
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateOffsetInFragment() const {
  return makeExpr<Analyzer::OffsetInFragment>();
}

Analyzer::ExpressionPtr RelAlgTranslator::translateArrayFunction(
    const RexFunctionOperator* rex_function) const {
  if (rex_function->getType().get_subtype() == kNULLT) {
    auto sql_type = rex_function->getType();
    CHECK(sql_type.get_type() == kARRAY);

    // FIX-ME:  Deal with NULL arrays
    auto translated_function_args(translateFunctionArgs(rex_function));
    if (translated_function_args.size() > 0) {
      const auto first_element_logical_type =
          get_nullable_logical_type_info(translated_function_args[0]->get_type_info());

      auto diff_elem_itr =
          std::find_if(translated_function_args.begin(),
                       translated_function_args.end(),
                       [first_element_logical_type](const auto expr) {
                         return first_element_logical_type !=
                                get_nullable_logical_type_info(expr->get_type_info());
                       });
      if (diff_elem_itr != translated_function_args.end()) {
        throw std::runtime_error(
            "Element " +
            std::to_string(diff_elem_itr - translated_function_args.begin()) +
            " is not of the same type as other elements of the array. Consider casting "
            "to force this condition.\nElement Type: " +
            get_nullable_logical_type_info((*diff_elem_itr)->get_type_info())
                .to_string() +
            "\nArray type: " + first_element_logical_type.to_string());
      }

      if (first_element_logical_type.is_string() &&
          !first_element_logical_type.is_dict_encoded_string()) {
        sql_type.set_subtype(first_element_logical_type.get_type());
        sql_type.set_compression(kENCODING_FIXED);
      } else if (first_element_logical_type.is_dict_encoded_string()) {
        sql_type.set_subtype(first_element_logical_type.get_type());
        sql_type.set_comp_param(TRANSIENT_DICT_ID);
      } else {
        sql_type.set_subtype(first_element_logical_type.get_type());
        sql_type.set_scale(first_element_logical_type.get_scale());
        sql_type.set_precision(first_element_logical_type.get_precision());
      }

      return makeExpr<Analyzer::ArrayExpr>(sql_type, translated_function_args);
    } else {
      // defaulting to valid sub-type for convenience
      sql_type.set_subtype(kBOOLEAN);
      return makeExpr<Analyzer::ArrayExpr>(sql_type, translated_function_args);
    }
  } else {
    return makeExpr<Analyzer::ArrayExpr>(rex_function->getType(),
                                         translateFunctionArgs(rex_function));
  }
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateFunction(
    const RexFunctionOperator* rex_function) const {
  if (func_resolve(rex_function->getName(), "LIKE"sv, "PG_ILIKE"sv)) {
    return translateLike(rex_function);
  }
  if (rex_function->getName() == "REGEXP_LIKE"sv) {
    return translateRegexp(rex_function);
  }
  if (rex_function->getName() == "LIKELY"sv) {
    return translateLikely(rex_function);
  }
  if (rex_function->getName() == "UNLIKELY"sv) {
    return translateUnlikely(rex_function);
  }
  if (func_resolve(rex_function->getName(), "PG_EXTRACT"sv, "PG_DATE_TRUNC"sv)) {
    return translateExtract(rex_function);
  }
  if (rex_function->getName() == "DATEADD"sv) {
    return translateDateadd(rex_function);
  }
  if (rex_function->getName() == "DATEDIFF"sv) {
    return translateDatediff(rex_function);
  }
  if (rex_function->getName() == "DATEPART"sv) {
    return translateDatepart(rex_function);
  }
  if (func_resolve(rex_function->getName(), "LENGTH"sv, "CHAR_LENGTH"sv)) {
    return translateLength(rex_function);
  }
  if (rex_function->getName() == "KEY_FOR_STRING"sv) {
    return translateKeyForString(rex_function);
  }
  if (rex_function->getName() == "SAMPLE_RATIO"sv) {
    return translateSampleRatio(rex_function);
  }
  if (rex_function->getName() == "CURRENT_USER"sv) {
    return translateCurrentUser(rex_function);
  }
  if (g_enable_experimental_string_functions && rex_function->getName() == "LOWER"sv) {
    return translateLower(rex_function);
  }
  if (func_resolve(rex_function->getName(), "CARDINALITY"sv, "ARRAY_LENGTH"sv)) {
    return translateCardinality(rex_function);
  }
  if (rex_function->getName() == "ITEM"sv) {
    return translateItem(rex_function);
  }
  if (rex_function->getName() == "NOW"sv) {
    return translateNow();
  }
  if (rex_function->getName() == "DATETIME"sv) {
    return translateDatetime(rex_function);
  }
  if (func_resolve(rex_function->getName(), "usTIMESTAMP"sv, "nsTIMESTAMP"sv)) {
    return translateHPTLiteral(rex_function);
  }
  if (rex_function->getName() == "ABS"sv) {
    return translateAbs(rex_function);
  }
  if (rex_function->getName() == "SIGN"sv) {
    return translateSign(rex_function);
  }
  if (func_resolve(rex_function->getName(), "CEIL"sv, "FLOOR"sv)) {
    return makeExpr<Analyzer::FunctionOperWithCustomTypeHandling>(
        rex_function->getType(),
        rex_function->getName(),
        translateFunctionArgs(rex_function));
  } else if (rex_function->getName() == "ROUND"sv) {
    std::vector<std::shared_ptr<Analyzer::Expr>> args =
        translateFunctionArgs(rex_function);

    if (rex_function->size() == 1) {
      // push a 0 constant if 2nd operand is missing.
      // this needs to be done as calcite returns
      // only the 1st operand without defaulting the 2nd one
      // when the user did not specify the 2nd operand.
      SQLTypes t = kSMALLINT;
      Datum d;
      d.smallintval = 0;
      args.push_back(makeExpr<Analyzer::Constant>(t, false, d));
    }

    // make sure we have only 2 operands
    CHECK(args.size() == 2);

    if (!args[0]->get_type_info().is_number()) {
      throw std::runtime_error("Only numeric 1st operands are supported");
    }

    // the 2nd operand does not need to be a constant
    // it can happily reference another integer column
    if (!args[1]->get_type_info().is_integer()) {
      throw std::runtime_error("Only integer 2nd operands are supported");
    }

    // Calcite may upcast decimals in a way that is
    // incompatible with the extension function input. Play it safe and stick with the
    // argument type instead.
    const SQLTypeInfo ret_ti = args[0]->get_type_info().is_decimal()
                                   ? args[0]->get_type_info()
                                   : rex_function->getType();

    return makeExpr<Analyzer::FunctionOperWithCustomTypeHandling>(
        ret_ti, rex_function->getName(), args);
  }
  if (rex_function->getName() == "DATETIME_PLUS"sv) {
    auto dt_plus = makeExpr<Analyzer::FunctionOper>(rex_function->getType(),
                                                    rex_function->getName(),
                                                    translateFunctionArgs(rex_function));
    const auto date_trunc = rewrite_to_date_trunc(dt_plus.get());
    if (date_trunc) {
      return date_trunc;
    }
    return translateDateadd(rex_function);
  }
  if (rex_function->getName() == "/INT"sv) {
    CHECK_EQ(size_t(2), rex_function->size());
    std::shared_ptr<Analyzer::Expr> lhs = translateScalarRex(rex_function->getOperand(0));
    std::shared_ptr<Analyzer::Expr> rhs = translateScalarRex(rex_function->getOperand(1));
    const auto rhs_lit = std::dynamic_pointer_cast<Analyzer::Constant>(rhs);
    return Parser::OperExpr::normalize(kDIVIDE, kONE, lhs, rhs);
  }
  if (rex_function->getName() == "Reinterpret"sv) {
    CHECK_EQ(size_t(1), rex_function->size());
    return translateScalarRex(rex_function->getOperand(0));
  }
  if (func_resolve(rex_function->getName(),
                   "ST_X"sv,
                   "ST_Y"sv,
                   "ST_XMin"sv,
                   "ST_YMin"sv,
                   "ST_XMax"sv,
                   "ST_YMax"sv,
                   "ST_NRings"sv,
                   "ST_NPoints"sv,
                   "ST_Length"sv,
                   "ST_Perimeter"sv,
                   "ST_Area"sv,
                   "ST_SRID"sv,
                   "MapD_GeoPolyBoundsPtr"sv /* deprecated */,
                   "MapD_GeoPolyBoundsPtr"sv /* deprecated */,
                   "OmniSci_Geo_PolyBoundsPtr"sv,
                   "OmniSci_Geo_PolyRenderGroup"sv)) {
    CHECK_EQ(rex_function->size(), size_t(1));
    return translateUnaryGeoFunction(rex_function);
  }
  if (func_resolve(rex_function->getName(),
                   "convert_meters_to_pixel_width"sv,
                   "convert_meters_to_pixel_height"sv,
                   "is_point_in_view"sv,
                   "is_point_size_in_view"sv)) {
    return translateFunctionWithGeoArg(rex_function);
  }
  if (func_resolve(rex_function->getName(),
                   "ST_Distance"sv,
                   "ST_MaxDistance"sv,
                   "ST_Intersects"sv,
                   "ST_Disjoint"sv,
                   "ST_Contains"sv,
                   "ST_Overlaps"sv,
                   "ST_Within"sv)) {
    CHECK_EQ(rex_function->size(), size_t(2));
    return translateBinaryGeoFunction(rex_function);
  }
  if (func_resolve(rex_function->getName(), "ST_DWithin"sv, "ST_DFullyWithin"sv)) {
    CHECK_EQ(rex_function->size(), size_t(3));
    return translateTernaryGeoFunction(rex_function);
  }
  if (rex_function->getName() == "OFFSET_IN_FRAGMENT"sv) {
    CHECK_EQ(size_t(0), rex_function->size());
    return translateOffsetInFragment();
  }
  if (rex_function->getName() == "ARRAY"sv) {
    // Var args; currently no check.  Possible fix-me -- can array have 0 elements?
    return translateArrayFunction(rex_function);
  }
  if (func_resolve(rex_function->getName(),
                   "ST_GeomFromText"sv,
                   "ST_GeogFromText"sv,
                   "ST_Point"sv,
                   "ST_Centroid"sv,
                   "ST_SetSRID"sv)) {
    SQLTypeInfo ti;
    return translateGeoProjection(rex_function, ti, false);
  }
  if (func_resolve(rex_function->getName(),
                   "ST_Intersection"sv,
                   "ST_Difference"sv,
                   "ST_Union"sv,
                   "ST_Buffer"sv)) {
    SQLTypeInfo ti;
    return translateGeoBinaryConstructor(rex_function, ti, false);
  }
  if (func_resolve(rex_function->getName(), "ST_IsEmpty"sv, "ST_IsValid"sv)) {
    SQLTypeInfo ti;
    return translateGeoPredicate(rex_function, ti, false);
  }

  auto arg_expr_list = translateFunctionArgs(rex_function);
  if (rex_function->getName() == std::string("||") ||
      rex_function->getName() == std::string("SUBSTRING")) {
    SQLTypeInfo ret_ti(kTEXT, false);
    return makeExpr<Analyzer::FunctionOper>(
        ret_ti, rex_function->getName(), arg_expr_list);
  }
  // Reset possibly wrong return type of rex_function to the return
  // type of the optimal valid implementation. The return type can be
  // wrong in the case of multiple implementations of UDF functions
  // that have different return types but Calcite specifies the return
  // type according to the first implementation.
  SQLTypeInfo ret_ti;
  try {
    auto ext_func_sig = bind_function(rex_function->getName(), arg_expr_list);

    auto ext_func_args = ext_func_sig.getArgs();
    CHECK_EQ(arg_expr_list.size(), ext_func_args.size());
    for (size_t i = 0; i < arg_expr_list.size(); i++) {
      // fold casts on constants
      if (auto constant =
              std::dynamic_pointer_cast<Analyzer::Constant>(arg_expr_list[i])) {
        auto ext_func_arg_ti = ext_arg_type_to_type_info(ext_func_args[i]);
        if (ext_func_arg_ti != arg_expr_list[i]->get_type_info()) {
          arg_expr_list[i] = constant->add_cast(ext_func_arg_ti);
        }
      }
    }

    ret_ti = ext_arg_type_to_type_info(ext_func_sig.getRet());
  } catch (ExtensionFunctionBindingError& e) {
    LOG(WARNING) << "RelAlgTranslator::translateFunction: " << e.what();
    throw;
  }

  // By default, the extension function type will not allow nulls. If one of the arguments
  // is nullable, the extension function must also explicitly allow nulls.
  bool arguments_not_null = true;
  for (const auto& arg_expr : arg_expr_list) {
    if (!arg_expr->get_type_info().get_notnull()) {
      arguments_not_null = false;
      break;
    }
  }
  ret_ti.set_notnull(arguments_not_null);

  return makeExpr<Analyzer::FunctionOper>(ret_ti, rex_function->getName(), arg_expr_list);
}

namespace {

std::vector<Analyzer::OrderEntry> translate_collation(
    const std::vector<SortField>& sort_fields) {
  std::vector<Analyzer::OrderEntry> collation;
  for (size_t i = 0; i < sort_fields.size(); ++i) {
    const auto& sort_field = sort_fields[i];
    collation.emplace_back(i,
                           sort_field.getSortDir() == SortDirection::Descending,
                           sort_field.getNullsPosition() == NullSortedPosition::First);
  }
  return collation;
}

bool supported_lower_bound(
    const RexWindowFunctionOperator::RexWindowBound& window_bound) {
  return window_bound.unbounded && window_bound.preceding && !window_bound.following &&
         !window_bound.is_current_row && !window_bound.offset &&
         window_bound.order_key == 0;
}

bool supported_upper_bound(const RexWindowFunctionOperator* rex_window_function) {
  const auto& window_bound = rex_window_function->getUpperBound();
  const bool to_current_row = !window_bound.unbounded && !window_bound.preceding &&
                              !window_bound.following && window_bound.is_current_row &&
                              !window_bound.offset && window_bound.order_key == 1;
  switch (rex_window_function->getKind()) {
    case SqlWindowFunctionKind::ROW_NUMBER:
    case SqlWindowFunctionKind::RANK:
    case SqlWindowFunctionKind::DENSE_RANK:
    case SqlWindowFunctionKind::CUME_DIST: {
      return to_current_row;
    }
    default: {
      return rex_window_function->getOrderKeys().empty()
                 ? (window_bound.unbounded && !window_bound.preceding &&
                    window_bound.following && !window_bound.is_current_row &&
                    !window_bound.offset && window_bound.order_key == 2)
                 : to_current_row;
    }
  }
}

}  // namespace

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateWindowFunction(
    const RexWindowFunctionOperator* rex_window_function) const {
  if (!supported_lower_bound(rex_window_function->getLowerBound()) ||
      !supported_upper_bound(rex_window_function) ||
      ((rex_window_function->getKind() == SqlWindowFunctionKind::ROW_NUMBER) !=
       rex_window_function->isRows())) {
    throw std::runtime_error("Frame specification not supported");
  }
  std::vector<std::shared_ptr<Analyzer::Expr>> args;
  for (size_t i = 0; i < rex_window_function->size(); ++i) {
    args.push_back(translateScalarRex(rex_window_function->getOperand(i)));
  }
  std::vector<std::shared_ptr<Analyzer::Expr>> partition_keys;
  for (const auto& partition_key : rex_window_function->getPartitionKeys()) {
    partition_keys.push_back(translateScalarRex(partition_key.get()));
  }
  std::vector<std::shared_ptr<Analyzer::Expr>> order_keys;
  for (const auto& order_key : rex_window_function->getOrderKeys()) {
    order_keys.push_back(translateScalarRex(order_key.get()));
  }
  auto ti = rex_window_function->getType();
  if (window_function_is_value(rex_window_function->getKind())) {
    CHECK_GE(args.size(), 1u);
    ti = args.front()->get_type_info();
  }
  return makeExpr<Analyzer::WindowFunction>(
      ti,
      rex_window_function->getKind(),
      args,
      partition_keys,
      order_keys,
      translate_collation(rex_window_function->getCollation()));
}

Analyzer::ExpressionPtrVector RelAlgTranslator::translateFunctionArgs(
    const RexFunctionOperator* rex_function) const {
  std::vector<std::shared_ptr<Analyzer::Expr>> args;
  for (size_t i = 0; i < rex_function->size(); ++i) {
    args.push_back(translateScalarRex(rex_function->getOperand(i)));
  }
  return args;
}

QualsConjunctiveForm qual_to_conjunctive_form(
    const std::shared_ptr<Analyzer::Expr> qual_expr) {
  CHECK(qual_expr);
  auto bin_oper = std::dynamic_pointer_cast<const Analyzer::BinOper>(qual_expr);
  if (!bin_oper) {
    const auto rewritten_qual_expr = rewrite_expr(qual_expr.get());
    return {{}, {rewritten_qual_expr ? rewritten_qual_expr : qual_expr}};
  }

  if (bin_oper->get_optype() == kAND) {
    const auto lhs_cf = qual_to_conjunctive_form(bin_oper->get_own_left_operand());
    const auto rhs_cf = qual_to_conjunctive_form(bin_oper->get_own_right_operand());
    auto simple_quals = lhs_cf.simple_quals;
    simple_quals.insert(
        simple_quals.end(), rhs_cf.simple_quals.begin(), rhs_cf.simple_quals.end());
    auto quals = lhs_cf.quals;
    quals.insert(quals.end(), rhs_cf.quals.begin(), rhs_cf.quals.end());
    return {simple_quals, quals};
  }
  int rte_idx{0};
  const auto simple_qual = bin_oper->normalize_simple_predicate(rte_idx);
  return simple_qual ? QualsConjunctiveForm{{simple_qual}, {}}
                     : QualsConjunctiveForm{{}, {qual_expr}};
}

std::vector<std::shared_ptr<Analyzer::Expr>> qual_to_disjunctive_form(
    const std::shared_ptr<Analyzer::Expr>& qual_expr) {
  CHECK(qual_expr);
  const auto bin_oper = std::dynamic_pointer_cast<const Analyzer::BinOper>(qual_expr);
  if (!bin_oper) {
    const auto rewritten_qual_expr = rewrite_expr(qual_expr.get());
    return {rewritten_qual_expr ? rewritten_qual_expr : qual_expr};
  }
  if (bin_oper->get_optype() == kOR) {
    const auto lhs_df = qual_to_disjunctive_form(bin_oper->get_own_left_operand());
    const auto rhs_df = qual_to_disjunctive_form(bin_oper->get_own_right_operand());
    auto quals = lhs_df;
    quals.insert(quals.end(), rhs_df.begin(), rhs_df.end());
    return quals;
  }
  return {qual_expr};
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateHPTLiteral(
    const RexFunctionOperator* rex_function) const {
  /* since calcite uses Avatica package called DateTimeUtils to parse timestamp strings.
     Therefore any string having fractional seconds more 3 places after the decimal
     (milliseconds) will get truncated to 3 decimal places, therefore we lose precision
     (us|ns). Issue: [BE-2461] Here we are hijacking literal cast to Timestamp(6|9) from
     calcite and translating them to generate our own casts.
  */
  CHECK_EQ(size_t(1), rex_function->size());
  const auto operand = translateScalarRex(rex_function->getOperand(0));
  const auto& operand_ti = operand->get_type_info();
  const auto& target_ti = rex_function->getType();
  if (!operand_ti.is_string()) {
    throw std::runtime_error(
        "High precision timestamp cast argument must be a string. Input type is: " +
        operand_ti.get_type_name());
  } else if (!target_ti.is_high_precision_timestamp()) {
    throw std::runtime_error(
        "Cast target type should be high precision timestamp. Input type is: " +
        target_ti.get_type_name());
  } else if (target_ti.get_dimension() != 6 && target_ti.get_dimension() != 9) {
    throw std::runtime_error(
        "Cast target type should be TIMESTAMP(6|9). Input type is: TIMESTAMP(" +
        std::to_string(target_ti.get_dimension()) + ")");
  } else {
    return operand->add_cast(target_ti);
  }
}
