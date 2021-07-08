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

#include "DateTimeTranslator.h"

#include <boost/algorithm/string.hpp>

namespace {

std::string from_extract_field(const ExtractField& fieldno) {
  switch (fieldno) {
    case kYEAR:
      return "year";
    case kQUARTER:
      return "quarter";
    case kMONTH:
      return "month";
    case kDAY:
      return "day";
    case kHOUR:
      return "hour";
    case kMINUTE:
      return "minute";
    case kSECOND:
      return "second";
    case kMILLISECOND:
      return "millisecond";
    case kMICROSECOND:
      return "microsecond";
    case kNANOSECOND:
      return "nanosecond";
    case kDOW:
      return "dow";
    case kISODOW:
      return "isodow";
    case kDOY:
      return "doy";
    case kEPOCH:
      return "epoch";
    case kQUARTERDAY:
      return "quarterday";
    case kWEEK:
      return "week";
    case kWEEK_SUNDAY:
      return "week_sunday";
    case kWEEK_SATURDAY:
      return "week_saturday";
    case kDATEEPOCH:
      return "dateepoch";
    case kLDOM:
      return "daysinmonth";
    case kISLEAP:
      return "isleapyear";
    case kISEOM:
      return "ismonthend";
    case kISSOM:
      return "ismonthstart";
    case kISEOQ:
      return "isquarterend";
    case kISSOQ:
      return "isquarterstart";
    case kISEOY:
      return "isyearend";
    case kISSOY:
      return "isyearstart";
    default:
      UNREACHABLE();
  }
  return "";
}

std::string from_datetrunc_field(const DatetruncField& fieldno) {
  switch (fieldno) {
    case dtYEAR:
      return "year";
    case dtQUARTER:
      return "quarter";
    case dtMONTH:
      return "month";
    case dtQUARTERDAY:
      return "quarterday";
    case dtDAY:
      return "day";
    case dtHOUR:
      return "hour";
    case dtMINUTE:
      return "minute";
    case dtSECOND:
      return "second";
    case dtMILLENNIUM:
      return "millennium";
    case dtCENTURY:
      return "century";
    case dtDECADE:
      return "decade";
    case dtMILLISECOND:
      return "millisecond";
    case dtMICROSECOND:
      return "microsecond";
    case dtNANOSECOND:
      return "nanosecond";
    case dtWEEK:
      return "week";
    case dtWEEK_SUNDAY:
      return "week_sunday";
    case dtWEEK_SATURDAY:
      return "week_saturday";
    case dtINVALID:
    default:
      UNREACHABLE();
  }
  return "";
}

}  // namespace

ExtractField ExtractExpr::to_extract_field(const std::string& field) {
  ExtractField fieldno;
  if (boost::iequals(field, "year")) {
    fieldno = kYEAR;
  } else if (boost::iequals(field, "quarter")) {
    fieldno = kQUARTER;
  } else if (boost::iequals(field, "month")) {
    fieldno = kMONTH;
  } else if (boost::iequals(field, "day")) {
    fieldno = kDAY;
  } else if (boost::iequals(field, "quarterday")) {
    fieldno = kQUARTERDAY;
  } else if (boost::iequals(field, "hour")) {
    fieldno = kHOUR;
  } else if (boost::iequals(field, "minute")) {
    fieldno = kMINUTE;
  } else if (boost::iequals(field, "second")) {
    fieldno = kSECOND;
  } else if (boost::iequals(field, "millisecond")) {
    fieldno = kMILLISECOND;
  } else if (boost::iequals(field, "microsecond")) {
    fieldno = kMICROSECOND;
  } else if (boost::iequals(field, "nanosecond")) {
    fieldno = kNANOSECOND;
  } else if (boost::iequals(field, "dow")) {
    fieldno = kDOW;
  } else if (boost::iequals(field, "isodow")) {
    fieldno = kISODOW;
  } else if (boost::iequals(field, "doy")) {
    fieldno = kDOY;
  } else if (boost::iequals(field, "epoch")) {
    fieldno = kEPOCH;
  } else if (boost::iequals(field, "week")) {
    fieldno = kWEEK;
  } else if (boost::iequals(field, "week_sunday")) {
    fieldno = kWEEK_SUNDAY;
  } else if (boost::iequals(field, "week_saturday")) {
    fieldno = kWEEK_SATURDAY;
  } else if (boost::iequals(field, "dateepoch")) {
    fieldno = kDATEEPOCH;
  } else if (boost::iequals(field, "daysinmonth")) {
    fieldno = kLDOM;
  } else if (boost::iequals(field, "isleapyear")) {
    fieldno = kISLEAP;
  } else if (boost::iequals(field, "ismonthend")) {
    fieldno = kISEOM;
  } else if (boost::iequals(field, "ismonthstart")) {
    fieldno = kISSOM;
  } else if (boost::iequals(field, "isquarterend")) {
    fieldno = kISEOQ;
  } else if (boost::iequals(field, "isquarterstart")) {
    fieldno = kISSOQ;
  } else if (boost::iequals(field, "isyearend")) {
    fieldno = kISEOY;
  } else if (boost::iequals(field, "isyearstart")) {
    fieldno = kISSOY;
  } else {
    throw std::runtime_error("Unsupported field in EXTRACT function " + field);
  }
  return fieldno;
}

std::shared_ptr<Analyzer::Expr> ExtractExpr::generate(
    const std::shared_ptr<Analyzer::Expr> from_expr,
    const std::string& field_name) {
  const auto field = to_extract_field(field_name);
  return ExtractExpr::generate(from_expr, field);
}

std::shared_ptr<Analyzer::Expr> ExtractExpr::generate(
    const std::shared_ptr<Analyzer::Expr> from_expr,
    const ExtractField& field) {
  const auto expr_ti = from_expr->get_type_info();
  if (!expr_ti.is_time()) {
    throw std::runtime_error(
        "Only TIME, TIMESTAMP and DATE types can be in EXTRACT function.");
  }
  if (expr_ti.get_type() == kTIME && field != kHOUR && field != kMINUTE &&
      field != kSECOND) {
    throw std::runtime_error("Cannot EXTRACT " + from_extract_field(field) +
                             " from TIME.");
  }
  const SQLTypeInfo ti(kBIGINT, 0, 0, expr_ti.get_notnull());
  auto constant = std::dynamic_pointer_cast<Analyzer::Constant>(from_expr);
  if (constant != nullptr) {
    Datum d;
    d.bigintval = field == kEPOCH
                      ? floor_div(constant->get_constval().bigintval,
                                  get_timestamp_precision_scale(expr_ti.get_dimension()))
                      : getExtractFromTimeConstantValue(
                            constant->get_constval().bigintval, field, expr_ti);
    constant->set_constval(d);
    constant->set_type_info(ti);
    return constant;
  }
  return makeExpr<Analyzer::ExtractExpr>(
      ti, from_expr->get_contains_agg(), field, from_expr->decompress());
}

DatetruncField DateTruncExpr::to_datetrunc_field(const std::string& field) {
  DatetruncField fieldno;
  if (boost::iequals(field, "year")) {
    fieldno = dtYEAR;
  } else if (boost::iequals(field, "quarter")) {
    fieldno = dtQUARTER;
  } else if (boost::iequals(field, "month")) {
    fieldno = dtMONTH;
  } else if (boost::iequals(field, "quarterday")) {
    fieldno = dtQUARTERDAY;
  } else if (boost::iequals(field, "day")) {
    fieldno = dtDAY;
  } else if (boost::iequals(field, "hour")) {
    fieldno = dtHOUR;
  } else if (boost::iequals(field, "minute")) {
    fieldno = dtMINUTE;
  } else if (boost::iequals(field, "second")) {
    fieldno = dtSECOND;
  } else if (boost::iequals(field, "millennium")) {
    fieldno = dtMILLENNIUM;
  } else if (boost::iequals(field, "century")) {
    fieldno = dtCENTURY;
  } else if (boost::iequals(field, "decade")) {
    fieldno = dtDECADE;
  } else if (boost::iequals(field, "millisecond")) {
    fieldno = dtMILLISECOND;
  } else if (boost::iequals(field, "microsecond")) {
    fieldno = dtMICROSECOND;
  } else if (boost::iequals(field, "nanosecond")) {
    fieldno = dtNANOSECOND;
  } else if (boost::iequals(field, "week")) {
    fieldno = dtWEEK;
  } else if (boost::iequals(field, "week_sunday")) {
    fieldno = dtWEEK_SUNDAY;
  } else if (boost::iequals(field, "week_saturday")) {
    fieldno = dtWEEK_SATURDAY;
  } else {
    throw std::runtime_error("Invalid field in DATE_TRUNC function " + field);
  }
  return fieldno;
}

std::shared_ptr<Analyzer::Expr> DateTruncExpr::generate(
    const std::shared_ptr<Analyzer::Expr> from_expr,
    const std::string& field_name) {
  const auto field = to_datetrunc_field(field_name);
  return DateTruncExpr::generate(from_expr, field);
}

std::shared_ptr<Analyzer::Expr> DateTruncExpr::generate(
    const std::shared_ptr<Analyzer::Expr> from_expr,
    const DatetruncField& field) {
  const auto& expr_ti = from_expr->get_type_info();
  if (!expr_ti.is_time()) {
    throw std::runtime_error(
        "Only TIME, TIMESTAMP and DATE types can be in DATE_TRUNC function.");
  }
  if (from_expr->get_type_info().get_type() == kTIME && field != dtHOUR &&
      field != dtMINUTE && field != dtSECOND) {
    throw std::runtime_error("Cannot DATE_TRUNC " + from_datetrunc_field(field) +
                             " from TIME.");
  }
  SQLTypeInfo ti(kTIMESTAMP, expr_ti.get_dimension(), 0, expr_ti.get_notnull());
  auto constant = std::dynamic_pointer_cast<Analyzer::Constant>(from_expr);
  if (constant) {
    Datum d{0};
    d.bigintval =
        getDateTruncConstantValue(constant->get_constval().bigintval, field, expr_ti);
    constant->set_constval(d);
    constant->set_type_info(ti);
    return constant;
  }
  return makeExpr<Analyzer::DatetruncExpr>(
      ti, from_expr->get_contains_agg(), field, from_expr->decompress());
}
