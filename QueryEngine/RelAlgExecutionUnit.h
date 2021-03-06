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

/**
 * @file    RelAlgExecutionUnit.h
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Execution unit for relational algebra. It's a low-level description
 *          of any relational algebra operation in a format understood by our VM.
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
 **/

#ifndef QUERYENGINE_RELALGEXECUTIONUNIT_H
#define QUERYENGINE_RELALGEXECUTIONUNIT_H

#include "Descriptors/InputDescriptors.h"
#include "QueryHint.h"
#include "Shared/sqldefs.h"
#include "Shared/toString.h"
#include "TableFunctions/TableFunctionOutputBufferSizeType.h"
#include "TableFunctions/TableFunctionsFactory.h"
#include "ThriftHandler/QueryState.h"

#include <list>
#include <memory>
#include <optional>
#include <vector>

enum class SortAlgorithm { Default, SpeculativeTopN, StreamingTopN };

namespace Analyzer {
class Expr;
class ColumnVar;
class Estimator;
struct OrderEntry;

}  // namespace Analyzer

struct SortInfo {
  const std::list<Analyzer::OrderEntry> order_entries;
  const SortAlgorithm algorithm;
  const size_t limit;
  const size_t offset;
};

struct JoinCondition {
  std::list<std::shared_ptr<Analyzer::Expr>> quals;
  JoinType type;
};

using JoinQualsPerNestingLevel = std::vector<JoinCondition>;

struct RelAlgExecutionUnit {
  std::vector<InputDescriptor> input_descs;
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  std::list<std::shared_ptr<Analyzer::Expr>> simple_quals;
  std::list<std::shared_ptr<Analyzer::Expr>> quals;
  const JoinQualsPerNestingLevel join_quals;
  const std::list<std::shared_ptr<Analyzer::Expr>> groupby_exprs;
  std::vector<Analyzer::Expr*> target_exprs;
  const std::shared_ptr<Analyzer::Estimator> estimator;
  const SortInfo sort_info;
  size_t scan_limit;
  RegisteredQueryHint query_hint;
  bool use_bump_allocator{false};
  // empty if not a UNION, true if UNION ALL, false if regular UNION
  const std::optional<bool> union_all;
  std::shared_ptr<const query_state::QueryState> query_state;
};

std::ostream& operator<<(std::ostream& os, const RelAlgExecutionUnit& ra_exe_unit);
std::string ra_exec_unit_desc_for_caching(const RelAlgExecutionUnit& ra_exe_unit);

struct TableFunctionExecutionUnit {
  const std::vector<InputDescriptor> input_descs;
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  std::vector<Analyzer::Expr*> input_exprs;
  std::vector<Analyzer::ColumnVar*> table_func_inputs;
  std::vector<Analyzer::Expr*> target_exprs;
  const size_t output_buffer_size_param;
  const table_functions::TableFunction table_func;

 public:
  std::string toString() const {
    return typeName(this) + "(" + "input_exprs=" + ::toString(input_exprs) +
           ", table_func_inputs=" + ::toString(table_func_inputs) +
           ", target_exprs=" + ::toString(target_exprs) +
           ", output_buffer_size_param=" + ::toString(output_buffer_size_param) +
           ", table_func=" + ::toString(table_func) + ")";
  }
};

class ResultSet;
using ResultSetPtr = std::shared_ptr<ResultSet>;

#endif  // QUERYENGINE_RELALGEXECUTIONUNIT_H
