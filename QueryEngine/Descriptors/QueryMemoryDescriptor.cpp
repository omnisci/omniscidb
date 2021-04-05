/*
 * Copyright 2018 MapD Technologies, Inc.
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

#include "QueryMemoryDescriptor.h"

#include "../Execute.h"
#include "../ExpressionRewrite.h"
#include "../GroupByAndAggregate.h"
#include "../StreamingTopN.h"
#include "../UsedColumnsVisitor.h"
#include "ColSlotContext.h"

#include <boost/algorithm/cxx11/any_of.hpp>

bool g_enable_smem_group_by{true};
extern bool g_enable_columnar_output;
extern bool g_enable_cpu_shmem;

namespace {

bool is_int_and_no_bigger_than(const SQLTypeInfo& ti, const size_t byte_width) {
  if (!ti.is_integer()) {
    return false;
  }
  return get_bit_width(ti) <= (byte_width * 8);
}

std::vector<int64_t> target_expr_group_by_indices(
    const std::list<std::shared_ptr<Analyzer::Expr>>& groupby_exprs,
    const std::vector<Analyzer::Expr*>& target_exprs) {
  std::vector<int64_t> indices(target_exprs.size(), -1);
  for (size_t target_idx = 0; target_idx < target_exprs.size(); ++target_idx) {
    const auto target_expr = target_exprs[target_idx];
    if (dynamic_cast<const Analyzer::AggExpr*>(target_expr)) {
      continue;
    }
    const auto var_expr = dynamic_cast<const Analyzer::Var*>(target_expr);
    if (var_expr && var_expr->get_which_row() == Analyzer::Var::kGROUPBY) {
      indices[target_idx] = var_expr->get_varno() - 1;
      continue;
    }
  }
  return indices;
}

std::vector<int64_t> target_expr_proj_indices(const RelAlgExecutionUnit& ra_exe_unit,
                                              const Catalog_Namespace::Catalog& cat) {
  if (ra_exe_unit.input_descs.size() > 1 ||
      !ra_exe_unit.sort_info.order_entries.empty()) {
    return {};
  }
  std::vector<int64_t> target_indices(ra_exe_unit.target_exprs.size(), -1);
  UsedColumnsVisitor columns_visitor;
  std::unordered_set<int> used_columns;
  for (const auto& simple_qual : ra_exe_unit.simple_quals) {
    const auto crt_used_columns = columns_visitor.visit(simple_qual.get());
    used_columns.insert(crt_used_columns.begin(), crt_used_columns.end());
  }
  for (const auto& qual : ra_exe_unit.quals) {
    const auto crt_used_columns = columns_visitor.visit(qual.get());
    used_columns.insert(crt_used_columns.begin(), crt_used_columns.end());
  }
  for (const auto& target : ra_exe_unit.target_exprs) {
    const auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(target);
    if (col_var) {
      const auto cd = get_column_descriptor_maybe(
          col_var->get_column_id(), col_var->get_table_id(), cat);
      if (!cd || !cd->isVirtualCol) {
        continue;
      }
    }
    const auto crt_used_columns = columns_visitor.visit(target);
    used_columns.insert(crt_used_columns.begin(), crt_used_columns.end());
  }
  for (size_t target_idx = 0; target_idx < ra_exe_unit.target_exprs.size();
       ++target_idx) {
    const auto target_expr = ra_exe_unit.target_exprs[target_idx];
    CHECK(target_expr);
    const auto& ti = target_expr->get_type_info();
    // TODO: add proper lazy fetch for varlen types in result set
    if (ti.is_varlen()) {
      continue;
    }
    const auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(target_expr);
    if (!col_var) {
      continue;
    }
    if (!ti.is_varlen() &&
        used_columns.find(col_var->get_column_id()) == used_columns.end()) {
      // setting target index to be zero so that later it can be decoded properly (in lazy
      // fetch, the zeroth target index indicates the corresponding rowid column for the
      // projected entry)
      target_indices[target_idx] = 0;
    }
  }
  return target_indices;
}

int8_t pick_baseline_key_component_width(const ExpressionRange& range,
                                         const size_t group_col_width) {
  if (range.getType() == ExpressionRangeType::Invalid) {
    return sizeof(int64_t);
  }
  switch (range.getType()) {
    case ExpressionRangeType::Integer:
      if (group_col_width == sizeof(int64_t) && range.hasNulls()) {
        return sizeof(int64_t);
      }
      return range.getIntMax() < EMPTY_KEY_32 - 1 ? sizeof(int32_t) : sizeof(int64_t);
    case ExpressionRangeType::Float:
    case ExpressionRangeType::Double:
      return sizeof(int64_t);  // No compaction for floating point yet.
    default:
      UNREACHABLE();
  }
  return sizeof(int64_t);
}

// TODO(miyu): make sure following setting of compact width is correct in all cases.
int8_t pick_baseline_key_width(const RelAlgExecutionUnit& ra_exe_unit,
                               const std::vector<InputTableInfo>& query_infos,
                               const Executor* executor) {
  int8_t compact_width{4};
  for (const auto& groupby_expr : ra_exe_unit.groupby_exprs) {
    const auto expr_range = getExpressionRange(groupby_expr.get(), query_infos, executor);
    compact_width = std::max(compact_width,
                             pick_baseline_key_component_width(
                                 expr_range, groupby_expr->get_type_info().get_size()));
  }
  return compact_width;
}

bool use_streaming_top_n(const RelAlgExecutionUnit& ra_exe_unit,
                         const bool output_columnar) {
  if (g_cluster) {
    return false;  // TODO(miyu)
  }

  for (const auto target_expr : ra_exe_unit.target_exprs) {
    if (dynamic_cast<const Analyzer::AggExpr*>(target_expr)) {
      return false;
    }
    if (dynamic_cast<const Analyzer::WindowFunction*>(target_expr)) {
      return false;
    }
  }

  // TODO: Allow streaming top n for columnar output
  if (!output_columnar && ra_exe_unit.sort_info.order_entries.size() == 1 &&
      ra_exe_unit.sort_info.limit &&
      ra_exe_unit.sort_info.algorithm == SortAlgorithm::StreamingTopN) {
    const auto only_order_entry = ra_exe_unit.sort_info.order_entries.front();
    CHECK_GT(only_order_entry.tle_no, int(0));
    CHECK_LE(static_cast<size_t>(only_order_entry.tle_no),
             ra_exe_unit.target_exprs.size());
    const auto order_entry_expr = ra_exe_unit.target_exprs[only_order_entry.tle_no - 1];
    const auto n = ra_exe_unit.sort_info.offset + ra_exe_unit.sort_info.limit;
    if ((order_entry_expr->get_type_info().is_number() ||
         order_entry_expr->get_type_info().is_time()) &&
        n <= 100000) {  // TODO(miyu): relax?
      return true;
    }
  }

  return false;
}

}  // namespace

std::unique_ptr<QueryMemoryDescriptor> QueryMemoryDescriptor::init(
    const Executor* executor,
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<InputTableInfo>& query_infos,
    const ColRangeInfo& col_range_info,
    const KeylessInfo& keyless_info,
    const bool allow_multifrag,
    const ExecutorDeviceType device_type,
    const int8_t crt_min_byte_width,
    const bool sort_on_gpu_hint,
    const size_t shard_count,
    const size_t max_groups_buffer_entry_count,
    RenderInfo* render_info,
    const CountDistinctDescriptors count_distinct_descriptors,
    const bool must_use_baseline_sort,
    const bool output_columnar_hint,
    const bool streaming_top_n_hint) {
  auto group_col_widths = get_col_byte_widths(ra_exe_unit.groupby_exprs);
  const bool is_group_by{!group_col_widths.empty()};

  auto col_slot_context = ColSlotContext(ra_exe_unit.target_exprs, {});

  const auto min_slot_size = QueryMemoryDescriptor::pick_target_compact_width(
      ra_exe_unit, query_infos, crt_min_byte_width);

  col_slot_context.setAllSlotsPaddedSize(min_slot_size);
  col_slot_context.validate();

  if (!is_group_by) {
    CHECK(!must_use_baseline_sort);

    return std::make_unique<QueryMemoryDescriptor>(
        executor,
        ra_exe_unit,
        query_infos,
        allow_multifrag,
        false,
        false,
        -1,
        ColRangeInfo{ra_exe_unit.estimator ? QueryDescriptionType::Estimator
                                           : QueryDescriptionType::NonGroupedAggregate,
                     0,
                     0,
                     0,
                     false},
        col_slot_context,
        std::vector<int8_t>{},
        /*group_col_compact_width=*/0,
        std::vector<int64_t>{},
        /*entry_count=*/1,
        count_distinct_descriptors,
        false,
        output_columnar_hint,
        render_info && render_info->isPotentialInSituRender(),
        must_use_baseline_sort,
        /*use_streaming_top_n=*/false);
  }

  size_t entry_count = 1;
  auto actual_col_range_info = col_range_info;
  bool interleaved_bins_on_gpu = false;
  bool keyless_hash = false;
  bool streaming_top_n = false;
  int8_t group_col_compact_width = 0;
  int32_t idx_target_as_key = -1;
  auto output_columnar = output_columnar_hint;
  std::vector<int64_t> target_groupby_indices;

  switch (col_range_info.hash_type_) {
    case QueryDescriptionType::GroupByPerfectHash: {
      if (render_info) {
        render_info->setInSituDataIfUnset(false);
      }
      // keyless hash: whether or not group columns are stored at the beginning of the
      // output buffer
      keyless_hash =
          (!sort_on_gpu_hint ||
           !QueryMemoryDescriptor::many_entries(
               col_range_info.max, col_range_info.min, col_range_info.bucket)) &&
          !col_range_info.bucket && !must_use_baseline_sort && keyless_info.keyless;

      // if keyless, then this target index indicates wheter an entry is empty or not
      // (acts as a key)
      idx_target_as_key = keyless_info.target_index;

      if (group_col_widths.size() > 1) {
        // col range info max contains the expected cardinality of the output
        entry_count = static_cast<size_t>(actual_col_range_info.max);
        actual_col_range_info.bucket = 0;
      } else {
        // single column perfect hash
        entry_count = std::max(
            GroupByAndAggregate::getBucketedCardinality(col_range_info), int64_t(1));
        const size_t interleaved_max_threshold{512};

        if (must_use_baseline_sort) {
          target_groupby_indices = target_expr_group_by_indices(ra_exe_unit.groupby_exprs,
                                                                ra_exe_unit.target_exprs);
          col_slot_context =
              ColSlotContext(ra_exe_unit.target_exprs, target_groupby_indices);
        }

        bool has_varlen_sample_agg = false;
        for (const auto& target_expr : ra_exe_unit.target_exprs) {
          if (target_expr->get_contains_agg()) {
            const auto agg_expr = dynamic_cast<Analyzer::AggExpr*>(target_expr);
            CHECK(agg_expr);
            if (agg_expr->get_aggtype() == kSAMPLE &&
                agg_expr->get_type_info().is_varlen()) {
              has_varlen_sample_agg = true;
              break;
            }
          }
        }

        interleaved_bins_on_gpu = keyless_hash && !has_varlen_sample_agg &&
                                  (entry_count <= interleaved_max_threshold) &&
                                  (device_type == ExecutorDeviceType::GPU) &&
                                  QueryMemoryDescriptor::countDescriptorsLogicallyEmpty(
                                      count_distinct_descriptors) &&
                                  !output_columnar;
      }
      break;
    }
    case QueryDescriptionType::GroupByBaselineHash: {
      if (render_info) {
        render_info->setInSituDataIfUnset(false);
      }
      entry_count = shard_count
                        ? (max_groups_buffer_entry_count + shard_count - 1) / shard_count
                        : max_groups_buffer_entry_count;
      target_groupby_indices = target_expr_group_by_indices(ra_exe_unit.groupby_exprs,
                                                            ra_exe_unit.target_exprs);
      col_slot_context = ColSlotContext(ra_exe_unit.target_exprs, target_groupby_indices);

      group_col_compact_width =
          output_columnar ? 8
                          : pick_baseline_key_width(ra_exe_unit, query_infos, executor);

      actual_col_range_info =
          ColRangeInfo{QueryDescriptionType::GroupByBaselineHash, 0, 0, 0, false};
      break;
    }
    case QueryDescriptionType::Projection: {
      CHECK(!must_use_baseline_sort);

      if (streaming_top_n_hint && use_streaming_top_n(ra_exe_unit, output_columnar)) {
        streaming_top_n = true;
        entry_count = ra_exe_unit.sort_info.offset + ra_exe_unit.sort_info.limit;
      } else {
        if (ra_exe_unit.use_bump_allocator) {
          output_columnar = false;
          entry_count = 0;
        } else {
          entry_count = ra_exe_unit.scan_limit
                            ? static_cast<size_t>(ra_exe_unit.scan_limit)
                            : max_groups_buffer_entry_count;
        }
      }

      const auto catalog = executor->getCatalog();
      CHECK(catalog);
      target_groupby_indices = executor->plan_state_->allow_lazy_fetch_
                                   ? target_expr_proj_indices(ra_exe_unit, *catalog)
                                   : std::vector<int64_t>{};

      col_slot_context = ColSlotContext(ra_exe_unit.target_exprs, target_groupby_indices);
      break;
    }
    default:
      UNREACHABLE() << "Unknown query type";
  }

  return std::make_unique<QueryMemoryDescriptor>(
      executor,
      ra_exe_unit,
      query_infos,
      allow_multifrag,
      keyless_hash,
      interleaved_bins_on_gpu,
      idx_target_as_key,
      actual_col_range_info,
      col_slot_context,
      group_col_widths,
      group_col_compact_width,
      target_groupby_indices,
      entry_count,
      count_distinct_descriptors,
      sort_on_gpu_hint,
      output_columnar,
      render_info && render_info->isPotentialInSituRender(),
      must_use_baseline_sort,
      streaming_top_n);
}

namespace {
bool anyOf(std::vector<Analyzer::Expr*> const& target_exprs, SQLAgg const agg_kind) {
  return boost::algorithm::any_of(target_exprs, [agg_kind](Analyzer::Expr const* expr) {
    auto const* const agg = dynamic_cast<Analyzer::AggExpr const*>(expr);
    return agg && agg->get_aggtype() == agg_kind;
  });
}
}  // namespace

QueryMemoryDescriptor::QueryMemoryDescriptor(
    const Executor* executor,
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<InputTableInfo>& query_infos,
    const bool allow_multifrag,
    const bool keyless_hash,
    const bool interleaved_bins_on_gpu,
    const int32_t idx_target_as_key,
    const ColRangeInfo& col_range_info,
    const ColSlotContext& col_slot_context,
    const std::vector<int8_t>& group_col_widths,
    const int8_t group_col_compact_width,
    const std::vector<int64_t>& target_groupby_indices,
    const size_t entry_count,
    const CountDistinctDescriptors count_distinct_descriptors,
    const bool sort_on_gpu_hint,
    const bool output_columnar_hint,
    const bool render_output,
    const bool must_use_baseline_sort,
    const bool use_streaming_top_n)
    : executor_(executor)
    , allow_multifrag_(allow_multifrag)
    , query_desc_type_(col_range_info.hash_type_)
    , keyless_hash_(keyless_hash)
    , interleaved_bins_on_gpu_(interleaved_bins_on_gpu)
    , idx_target_as_key_(idx_target_as_key)
    , group_col_widths_(group_col_widths)
    , group_col_compact_width_(group_col_compact_width)
    , target_groupby_indices_(target_groupby_indices)
    , entry_count_(entry_count)
    , min_val_(col_range_info.min)
    , max_val_(col_range_info.max)
    , bucket_(col_range_info.bucket)
    , has_nulls_(col_range_info.has_nulls)
    , count_distinct_descriptors_(count_distinct_descriptors)
    , output_columnar_(false)
    , render_output_(render_output)
    , must_use_baseline_sort_(must_use_baseline_sort)
    , is_table_function_(false)
    , use_streaming_top_n_(use_streaming_top_n)
    , force_4byte_float_(false)
    , col_slot_context_(col_slot_context) {
  col_slot_context_.setAllUnsetSlotsPaddedSize(8);
  col_slot_context_.validate();

  sort_on_gpu_ = sort_on_gpu_hint && canOutputColumnar() && !keyless_hash_;

  if (sort_on_gpu_) {
    CHECK(!ra_exe_unit.use_bump_allocator);
    output_columnar_ = true;
  } else {
    switch (query_desc_type_) {
      case QueryDescriptionType::Projection:
        output_columnar_ = output_columnar_hint;
        break;
      case QueryDescriptionType::GroupByPerfectHash:
        output_columnar_ = output_columnar_hint &&
                           QueryMemoryDescriptor::countDescriptorsLogicallyEmpty(
                               count_distinct_descriptors_) &&
                           !anyOf(ra_exe_unit.target_exprs, kAPPROX_MEDIAN);
        break;
      case QueryDescriptionType::GroupByBaselineHash:
        output_columnar_ = output_columnar_hint;
        break;
      case QueryDescriptionType::NonGroupedAggregate:
        output_columnar_ = output_columnar_hint &&
                           QueryMemoryDescriptor::countDescriptorsLogicallyEmpty(
                               count_distinct_descriptors_) &&
                           !anyOf(ra_exe_unit.target_exprs, kAPPROX_MEDIAN);
        break;
      default:
        output_columnar_ = false;
        break;
    }
  }

  if (isLogicalSizedColumnsAllowed()) {
    // TODO(adb): Ensure fixed size buffer allocations are correct with all logical column
    // sizes
    CHECK(!ra_exe_unit.use_bump_allocator);
    col_slot_context_.setAllSlotsPaddedSizeToLogicalSize();
    col_slot_context_.validate();
  }

#ifdef HAVE_CUDA
  // Check Streaming Top N heap usage, bail if > max slab size, CUDA ONLY
  if (use_streaming_top_n_ && executor->catalog_->getDataMgr().gpusPresent()) {
    const auto thread_count = executor->blockSize() * executor->gridSize();
    const auto total_buff_size =
        streaming_top_n::get_heap_size(getRowSize(), getEntryCount(), thread_count);
    if (total_buff_size > executor_->maxGpuSlabSize()) {
      throw StreamingTopNOOM(total_buff_size);
    }
  }
#endif
}

QueryMemoryDescriptor::QueryMemoryDescriptor()
    : executor_(nullptr)
    , allow_multifrag_(false)
    , query_desc_type_(QueryDescriptionType::Projection)
    , keyless_hash_(false)
    , interleaved_bins_on_gpu_(false)
    , idx_target_as_key_(0)
    , group_col_compact_width_(0)
    , entry_count_(0)
    , min_val_(0)
    , max_val_(0)
    , bucket_(0)
    , has_nulls_(false)
    , sort_on_gpu_(false)
    , output_columnar_(false)
    , render_output_(false)
    , must_use_baseline_sort_(false)
    , is_table_function_(false)
    , use_streaming_top_n_(false)
    , force_4byte_float_(false) {}

QueryMemoryDescriptor::QueryMemoryDescriptor(const Executor* executor,
                                             const size_t entry_count,
                                             const QueryDescriptionType query_desc_type,
                                             const bool is_table_function)
    : executor_(executor)
    , allow_multifrag_(false)
    , query_desc_type_(query_desc_type)
    , keyless_hash_(false)
    , interleaved_bins_on_gpu_(false)
    , idx_target_as_key_(0)
    , group_col_compact_width_(0)
    , entry_count_(entry_count)
    , min_val_(0)
    , max_val_(0)
    , bucket_(0)
    , has_nulls_(false)
    , sort_on_gpu_(false)
    , output_columnar_(false)
    , render_output_(false)
    , must_use_baseline_sort_(false)
    , is_table_function_(is_table_function)
    , use_streaming_top_n_(false)
    , force_4byte_float_(false) {}

QueryMemoryDescriptor::QueryMemoryDescriptor(const QueryDescriptionType query_desc_type,
                                             const int64_t min_val,
                                             const int64_t max_val,
                                             const bool has_nulls,
                                             const std::vector<int8_t>& group_col_widths)
    : executor_(nullptr)
    , allow_multifrag_(false)
    , query_desc_type_(query_desc_type)
    , keyless_hash_(false)
    , interleaved_bins_on_gpu_(false)
    , idx_target_as_key_(0)
    , group_col_widths_(group_col_widths)
    , group_col_compact_width_(0)
    , entry_count_(0)
    , min_val_(min_val)
    , max_val_(max_val)
    , bucket_(0)
    , has_nulls_(false)
    , sort_on_gpu_(false)
    , output_columnar_(false)
    , render_output_(false)
    , must_use_baseline_sort_(false)
    , is_table_function_(false)
    , use_streaming_top_n_(false)
    , force_4byte_float_(false) {}

bool QueryMemoryDescriptor::operator==(const QueryMemoryDescriptor& other) const {
  // Note that this method does not check ptr reference members (e.g. executor_) or
  // entry_count_
  if (query_desc_type_ != other.query_desc_type_) {
    return false;
  }
  if (keyless_hash_ != other.keyless_hash_) {
    return false;
  }
  if (interleaved_bins_on_gpu_ != other.interleaved_bins_on_gpu_) {
    return false;
  }
  if (idx_target_as_key_ != other.idx_target_as_key_) {
    return false;
  }
  if (force_4byte_float_ != other.force_4byte_float_) {
    return false;
  }
  if (group_col_widths_ != other.group_col_widths_) {
    return false;
  }
  if (group_col_compact_width_ != other.group_col_compact_width_) {
    return false;
  }
  if (target_groupby_indices_ != other.target_groupby_indices_) {
    return false;
  }
  if (min_val_ != other.min_val_) {
    return false;
  }
  if (max_val_ != other.max_val_) {
    return false;
  }
  if (bucket_ != other.bucket_) {
    return false;
  }
  if (has_nulls_ != other.has_nulls_) {
    return false;
  }
  if (count_distinct_descriptors_.size() != count_distinct_descriptors_.size()) {
    return false;
  } else {
    // Count distinct descriptors can legitimately differ in device only.
    for (size_t i = 0; i < count_distinct_descriptors_.size(); ++i) {
      auto ref_count_distinct_desc = other.count_distinct_descriptors_[i];
      auto count_distinct_desc = count_distinct_descriptors_[i];
      count_distinct_desc.device_type = ref_count_distinct_desc.device_type;
      if (ref_count_distinct_desc != count_distinct_desc) {
        return false;
      }
    }
  }
  if (sort_on_gpu_ != other.sort_on_gpu_) {
    return false;
  }
  if (output_columnar_ != other.output_columnar_) {
    return false;
  }
  if (col_slot_context_ != other.col_slot_context_) {
    return false;
  }
  return true;
}

std::unique_ptr<QueryExecutionContext> QueryMemoryDescriptor::getQueryExecutionContext(
    const RelAlgExecutionUnit& ra_exe_unit,
    const Executor* executor,
    const ExecutorDeviceType device_type,
    const ExecutorDispatchMode dispatch_mode,
    const int device_id,
    const int64_t num_rows,
    const std::vector<std::vector<const int8_t*>>& col_buffers,
    const std::vector<std::vector<uint64_t>>& frag_offsets,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const bool output_columnar,
    const bool sort_on_gpu,
    const size_t thread_idx,
    RenderInfo* render_info) const {
  auto timer = DEBUG_TIMER(__func__);
  if (frag_offsets.empty()) {
    return nullptr;
  }
  return std::unique_ptr<QueryExecutionContext>(
      new QueryExecutionContext(ra_exe_unit,
                                *this,
                                executor,
                                device_type,
                                dispatch_mode,
                                device_id,
                                num_rows,
                                col_buffers,
                                frag_offsets,
                                row_set_mem_owner,
                                output_columnar,
                                sort_on_gpu,
                                thread_idx,
                                render_info));
}

int8_t QueryMemoryDescriptor::pick_target_compact_width(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<InputTableInfo>& query_infos,
    const int8_t crt_min_byte_width) {
  if (g_bigint_count) {
    return sizeof(int64_t);
  }
  int8_t compact_width{0};
  auto col_it = ra_exe_unit.input_col_descs.begin();
  int unnest_array_col_id{std::numeric_limits<int>::min()};
  for (const auto& groupby_expr : ra_exe_unit.groupby_exprs) {
    const auto uoper = dynamic_cast<Analyzer::UOper*>(groupby_expr.get());
    if (uoper && uoper->get_optype() == kUNNEST) {
      const auto& arg_ti = uoper->get_operand()->get_type_info();
      CHECK(arg_ti.is_array());
      const auto& elem_ti = arg_ti.get_elem_type();
      if (elem_ti.is_string() && elem_ti.get_compression() == kENCODING_DICT) {
        unnest_array_col_id = (*col_it)->getColId();
      } else {
        compact_width = crt_min_byte_width;
        break;
      }
    }
    ++col_it;
  }
  if (!compact_width &&
      (ra_exe_unit.groupby_exprs.size() != 1 || !ra_exe_unit.groupby_exprs.front())) {
    compact_width = crt_min_byte_width;
  }
  if (!compact_width) {
    col_it = ra_exe_unit.input_col_descs.begin();
    std::advance(col_it, ra_exe_unit.groupby_exprs.size());
    for (const auto target : ra_exe_unit.target_exprs) {
      const auto& ti = target->get_type_info();
      const auto agg = dynamic_cast<const Analyzer::AggExpr*>(target);
      if (agg && agg->get_arg()) {
        compact_width = crt_min_byte_width;
        break;
      }

      if (agg) {
        CHECK_EQ(kCOUNT, agg->get_aggtype());
        CHECK(!agg->get_is_distinct());
        ++col_it;
        continue;
      }

      if (is_int_and_no_bigger_than(ti, 4) ||
          (ti.is_string() && ti.get_compression() == kENCODING_DICT)) {
        ++col_it;
        continue;
      }

      const auto uoper = dynamic_cast<Analyzer::UOper*>(target);
      if (uoper && uoper->get_optype() == kUNNEST &&
          (*col_it)->getColId() == unnest_array_col_id) {
        const auto arg_ti = uoper->get_operand()->get_type_info();
        CHECK(arg_ti.is_array());
        const auto& elem_ti = arg_ti.get_elem_type();
        if (elem_ti.is_string() && elem_ti.get_compression() == kENCODING_DICT) {
          ++col_it;
          continue;
        }
      }

      compact_width = crt_min_byte_width;
      break;
    }
  }
  if (!compact_width) {
    size_t total_tuples{0};
    for (const auto& qi : query_infos) {
      total_tuples += qi.info.getNumTuples();
    }
    return total_tuples <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
                   unnest_array_col_id != std::numeric_limits<int>::min()
               ? 4
               : crt_min_byte_width;
  } else {
    // TODO(miyu): relax this condition to allow more cases just w/o padding
    for (auto wid : get_col_byte_widths(ra_exe_unit.target_exprs)) {
      compact_width = std::max(compact_width, wid);
    }
    return compact_width;
  }
}

size_t QueryMemoryDescriptor::getColsSize() const {
  return col_slot_context_.getAllSlotsAlignedPaddedSize();
}

size_t QueryMemoryDescriptor::getRowSize() const {
  CHECK(!output_columnar_);
  size_t total_bytes{0};
  if (keyless_hash_) {
    // ignore, there's no group column in the output buffer
    CHECK(query_desc_type_ == QueryDescriptionType::GroupByPerfectHash);
  } else {
    total_bytes += group_col_widths_.size() * getEffectiveKeyWidth();
    total_bytes = align_to_int64(total_bytes);
  }
  total_bytes += getColsSize();
  return align_to_int64(total_bytes);
}

size_t QueryMemoryDescriptor::getWarpCount() const {
  return (interleaved_bins_on_gpu_ ? executor_->warpSize() : 1);
}

size_t QueryMemoryDescriptor::getCompactByteWidth() const {
  return col_slot_context_.getCompactByteWidth();
}

/**
 * Returns the maximum total number of bytes (including required paddings) to store
 * all non-lazy columns' results for columnar cases.
 *
 */
size_t QueryMemoryDescriptor::getTotalBytesOfColumnarBuffers() const {
  CHECK(output_columnar_);
  return col_slot_context_.getTotalBytesOfColumnarBuffers(entry_count_);
}

/**
 * This is a helper function that returns the total number of bytes (including
 * required paddings) to store all non-lazy columns' results for columnar cases.
 */
size_t QueryMemoryDescriptor::getTotalBytesOfColumnarBuffers(
    const size_t num_entries_per_column) const {
  return col_slot_context_.getTotalBytesOfColumnarBuffers(num_entries_per_column);
}

/**
 * Returns the effective total number of bytes from columnar projections, which
 * includes 1) total number of bytes used to store all non-lazy columns 2) total
 * number of bytes used to store row indices (for lazy fetches, etc.)
 *
 * NOTE: this function does not represent the buffer sizes dedicated for the results,
 * but the required memory to fill all valid results into a compact new buffer (with
 * no holes in it)
 */
size_t QueryMemoryDescriptor::getTotalBytesOfColumnarProjections(
    const size_t projection_count) const {
  constexpr size_t row_index_width = sizeof(int64_t);
  return getTotalBytesOfColumnarBuffers(projection_count) +
         row_index_width * projection_count;
}

size_t QueryMemoryDescriptor::getColOnlyOffInBytes(const size_t col_idx) const {
  return col_slot_context_.getColOnlyOffInBytes(col_idx);
}

/*
 * Returns the memory offset in bytes for a specific agg column in the output
 * memory buffer. Depending on the query type, there may be some extra portion
 * of memory prepended at the beginning of the buffer. A brief description of
 * the memory layout is as follows:
 * 1. projections: index column (64bit) + all target columns
 * 2. group by: all group columns (64-bit each) + all agg columns
 * 2a. if keyless, there is no prepending group column stored at the beginning
 */
size_t QueryMemoryDescriptor::getColOffInBytes(const size_t col_idx) const {
  const auto warp_count = getWarpCount();
  if (output_columnar_) {
    CHECK_EQ(size_t(1), warp_count);
    size_t offset{0};
    if (!keyless_hash_) {
      offset += getPrependedGroupBufferSizeInBytes();
    }
    for (size_t index = 0; index < col_idx; ++index) {
      offset += align_to_int64(getPaddedSlotWidthBytes(index) * entry_count_);
    }
    return offset;
  }

  size_t offset{0};
  if (keyless_hash_) {
    // ignore, there's no group column in the output buffer
    CHECK(query_desc_type_ == QueryDescriptionType::GroupByPerfectHash);
  } else {
    offset += group_col_widths_.size() * getEffectiveKeyWidth();
    offset = align_to_int64(offset);
  }
  offset += getColOnlyOffInBytes(col_idx);
  return offset;
}

/*
 * Returns the memory offset for a particular group column in the prepended group
 * columns portion of the memory.
 */
size_t QueryMemoryDescriptor::getPrependedGroupColOffInBytes(
    const size_t group_idx) const {
  CHECK(output_columnar_);
  CHECK(group_idx < getGroupbyColCount());
  size_t offset{0};
  for (size_t col_idx = 0; col_idx < group_idx; col_idx++) {
    // TODO(Saman): relax that int64_bit part immediately
    offset += align_to_int64(
        std::max(groupColWidth(col_idx), static_cast<int8_t>(sizeof(int64_t))) *
        getEntryCount());
  }
  return offset;
}

/*
 * Returns total amount of memory prepended at the beginning of the output memory
 * buffer.
 */
size_t QueryMemoryDescriptor::getPrependedGroupBufferSizeInBytes() const {
  CHECK(output_columnar_);
  size_t buffer_size{0};
  for (size_t group_idx = 0; group_idx < getGroupbyColCount(); group_idx++) {
    buffer_size += align_to_int64(
        std::max(groupColWidth(group_idx), static_cast<int8_t>(sizeof(int64_t))) *
        getEntryCount());
  }
  return buffer_size;
}

size_t QueryMemoryDescriptor::getColOffInBytesInNextBin(const size_t col_idx) const {
  auto warp_count = getWarpCount();
  if (output_columnar_) {
    CHECK_EQ(size_t(1), group_col_widths_.size());
    CHECK_EQ(size_t(1), warp_count);
    return getPaddedSlotWidthBytes(col_idx);
  }

  return warp_count * getRowSize();
}

size_t QueryMemoryDescriptor::getNextColOffInBytes(const int8_t* col_ptr,
                                                   const size_t bin,
                                                   const size_t col_idx) const {
  CHECK(!output_columnar_ || bin < entry_count_);
  size_t offset{0};
  auto warp_count = getWarpCount();
  const auto chosen_bytes = getPaddedSlotWidthBytes(col_idx);
  const auto total_slot_count = getSlotCount();
  if (col_idx + 1 == total_slot_count) {
    if (output_columnar_) {
      return (entry_count_ - bin) * chosen_bytes;
    } else {
      return static_cast<size_t>(align_to_int64(col_ptr + chosen_bytes) - col_ptr);
    }
  }

  const auto next_chosen_bytes = getPaddedSlotWidthBytes(col_idx + 1);
  if (output_columnar_) {
    CHECK_EQ(size_t(1), group_col_widths_.size());
    CHECK_EQ(size_t(1), warp_count);

    offset = align_to_int64(entry_count_ * chosen_bytes);

    offset += bin * (next_chosen_bytes - chosen_bytes);
    return offset;
  }

  if (next_chosen_bytes == sizeof(int64_t)) {
    return static_cast<size_t>(align_to_int64(col_ptr + chosen_bytes) - col_ptr);
  } else {
    return chosen_bytes;
  }
}

size_t QueryMemoryDescriptor::getNextColOffInBytesRowOnly(const int8_t* col_ptr,
                                                          const size_t col_idx) const {
  const auto chosen_bytes = getPaddedSlotWidthBytes(col_idx);
  const auto total_slot_count = getSlotCount();
  if (col_idx + 1 == total_slot_count) {
    return static_cast<size_t>(align_to_int64(col_ptr + chosen_bytes) - col_ptr);
  }

  const auto next_chosen_bytes = getPaddedSlotWidthBytes(col_idx + 1);

  if (next_chosen_bytes == sizeof(int64_t)) {
    return static_cast<size_t>(align_to_int64(col_ptr + chosen_bytes) - col_ptr);
  } else {
    return chosen_bytes;
  }
}

size_t QueryMemoryDescriptor::getBufferSizeBytes(
    const RelAlgExecutionUnit& ra_exe_unit,
    const unsigned thread_count,
    const ExecutorDeviceType device_type) const {
  if (use_streaming_top_n_) {
    const size_t n = ra_exe_unit.sort_info.offset + ra_exe_unit.sort_info.limit;
    return streaming_top_n::get_heap_size(getRowSize(), n, thread_count);
  }
  return getBufferSizeBytes(device_type, entry_count_);
}

/**
 * Returns total amount of output buffer memory for each device (CPU/GPU)
 *
 * Columnar:
 *  if projection: it returns index buffer + columnar buffer (all non-lazy columns)
 *  if group by: it returns the amount required for each group column (assumes 64-bit
 * per group) + columnar buffer (all involved agg columns)
 *
 * Row-wise:
 *  returns required memory per row multiplied by number of entries
 */
size_t QueryMemoryDescriptor::getBufferSizeBytes(const ExecutorDeviceType device_type,
                                                 const size_t entry_count) const {
  if (keyless_hash_ && !output_columnar_) {
    CHECK_GE(group_col_widths_.size(), size_t(1));
    auto row_bytes = align_to_int64(getColsSize());

    return (interleavedBins(device_type) ? executor_->warpSize() : 1) * entry_count *
           row_bytes;
  }

  constexpr size_t row_index_width = sizeof(int64_t);
  size_t total_bytes{0};
  if (output_columnar_) {
    total_bytes = (query_desc_type_ == QueryDescriptionType::Projection
                       ? row_index_width * entry_count
                       : sizeof(int64_t) * group_col_widths_.size() * entry_count) +
                  getTotalBytesOfColumnarBuffers();
  } else {
    total_bytes = getRowSize() * entry_count;
  }

  return total_bytes;
}

size_t QueryMemoryDescriptor::getBufferSizeBytes(
    const ExecutorDeviceType device_type) const {
  return getBufferSizeBytes(device_type, entry_count_);
}

void QueryMemoryDescriptor::setOutputColumnar(const bool val) {
  output_columnar_ = val;
  if (isLogicalSizedColumnsAllowed()) {
    col_slot_context_.setAllSlotsPaddedSizeToLogicalSize();
  }
}

/*
 * Indicates the query types that are currently allowed to use the logical
 * sized columns instead of padded sized ones.
 */
bool QueryMemoryDescriptor::isLogicalSizedColumnsAllowed() const {
  // In distributed mode, result sets are serialized using rowwise iterators, so we use
  // consistent slot widths for now
  return output_columnar_ && !g_cluster &&
         (query_desc_type_ == QueryDescriptionType::Projection);
}

size_t QueryMemoryDescriptor::getBufferColSlotCount() const {
  size_t total_slot_count = col_slot_context_.getSlotCount();

  if (target_groupby_indices_.empty()) {
    return total_slot_count;
  }
  return total_slot_count - std::count_if(target_groupby_indices_.begin(),
                                          target_groupby_indices_.end(),
                                          [](const int64_t i) { return i >= 0; });
}

bool QueryMemoryDescriptor::usesGetGroupValueFast() const {
  return (query_desc_type_ == QueryDescriptionType::GroupByPerfectHash &&
          getGroupbyColCount() == 1);
}

bool QueryMemoryDescriptor::threadsShareMemory() const {
  return query_desc_type_ != QueryDescriptionType::NonGroupedAggregate;
}

bool QueryMemoryDescriptor::cpuThreadsShareMemory() const {
  return g_enable_cpu_shmem &&
         query_desc_type_ == QueryDescriptionType::GroupByPerfectHash && keyless_hash_;
}

bool QueryMemoryDescriptor::blocksShareMemory() const {
  if (g_cluster || is_table_function_) {
    return true;
  }
  if (!countDescriptorsLogicallyEmpty(count_distinct_descriptors_)) {
    return true;
  }
  if (executor_->isCPUOnly() || render_output_ ||
      query_desc_type_ == QueryDescriptionType::GroupByBaselineHash ||
      query_desc_type_ == QueryDescriptionType::Projection ||
      (query_desc_type_ == QueryDescriptionType::GroupByPerfectHash &&
       getGroupbyColCount() > 1)) {
    return true;
  }
  return query_desc_type_ == QueryDescriptionType::GroupByPerfectHash &&
         many_entries(max_val_, min_val_, bucket_);
}

bool QueryMemoryDescriptor::lazyInitGroups(const ExecutorDeviceType device_type) const {
  return device_type == ExecutorDeviceType::GPU && !render_output_ &&
         countDescriptorsLogicallyEmpty(count_distinct_descriptors_);
}

bool QueryMemoryDescriptor::interleavedBins(const ExecutorDeviceType device_type) const {
  return interleaved_bins_on_gpu_ && device_type == ExecutorDeviceType::GPU;
}

// TODO(Saman): an implementation detail, so move this out of QMD
bool QueryMemoryDescriptor::isWarpSyncRequired(
    const ExecutorDeviceType device_type) const {
  if (device_type != ExecutorDeviceType::GPU) {
    return false;
  } else {
    auto cuda_mgr = executor_->getCatalog()->getDataMgr().getCudaMgr();
    CHECK(cuda_mgr);
    return cuda_mgr->isArchVoltaOrGreaterForAll();
  }
}

size_t QueryMemoryDescriptor::getColCount() const {
  return col_slot_context_.getColCount();
}

size_t QueryMemoryDescriptor::getSlotCount() const {
  return col_slot_context_.getSlotCount();
}

const int8_t QueryMemoryDescriptor::getPaddedSlotWidthBytes(const size_t slot_idx) const {
  return col_slot_context_.getSlotInfo(slot_idx).padded_size;
}

const int8_t QueryMemoryDescriptor::getLogicalSlotWidthBytes(
    const size_t slot_idx) const {
  return col_slot_context_.getSlotInfo(slot_idx).logical_size;
}

const int8_t QueryMemoryDescriptor::getSlotIndexForSingleSlotCol(
    const size_t col_idx) const {
  const auto& col_slots = col_slot_context_.getSlotsForCol(col_idx);
  CHECK_EQ(col_slots.size(), size_t(1));
  return col_slots.front();
}

void QueryMemoryDescriptor::useConsistentSlotWidthSize(const int8_t slot_width_size) {
  col_slot_context_.setAllSlotsSize(slot_width_size);
}

size_t QueryMemoryDescriptor::getRowWidth() const {
  // Note: Actual row size may include padding (see ResultSetBufferAccessors.h)
  return col_slot_context_.getAllSlotsPaddedSize();
}

int8_t QueryMemoryDescriptor::updateActualMinByteWidth(
    const int8_t actual_min_byte_width) const {
  return col_slot_context_.getMinPaddedByteSize(actual_min_byte_width);
}

void QueryMemoryDescriptor::addColSlotInfo(
    const std::vector<std::tuple<int8_t, int8_t>>& slots_for_col) {
  col_slot_context_.addColumn(slots_for_col);
}

void QueryMemoryDescriptor::clearSlotInfo() {
  col_slot_context_.clear();
}

void QueryMemoryDescriptor::alignPaddedSlots() {
  col_slot_context_.alignPaddedSlots(sortOnGpu());
}

bool QueryMemoryDescriptor::canOutputColumnar() const {
  return usesGetGroupValueFast() && threadsShareMemory() && blocksShareMemory() &&
         !interleavedBins(ExecutorDeviceType::GPU) &&
         countDescriptorsLogicallyEmpty(count_distinct_descriptors_);
}

std::string QueryMemoryDescriptor::queryDescTypeToString() const {
  switch (query_desc_type_) {
    case QueryDescriptionType::GroupByPerfectHash:
      return "Perfect Hash";
    case QueryDescriptionType::GroupByBaselineHash:
      return "Baseline Hash";
    case QueryDescriptionType::Projection:
      return "Projection";
    case QueryDescriptionType::NonGroupedAggregate:
      return "Non-grouped Aggregate";
    case QueryDescriptionType::Estimator:
      return "Estimator";
    default:
      UNREACHABLE();
  }
  return "";
}

std::string QueryMemoryDescriptor::toString() const {
  auto str = reductionKey();
  str += "\tAllow Multifrag: " + ::toString(allow_multifrag_) + "\n";
  str += "\tInterleaved Bins on GPU: " + ::toString(interleaved_bins_on_gpu_) + "\n";
  str += "\tBlocks Share Memory: " + ::toString(blocksShareMemory()) + "\n";
  str += "\tThreads Share Memory: " + ::toString(threadsShareMemory()) + "\n";
  str += "\tUses Fast Group Values: " + ::toString(usesGetGroupValueFast()) + "\n";
  str +=
      "\tLazy Init Groups (GPU): " + ::toString(lazyInitGroups(ExecutorDeviceType::GPU)) +
      "\n";
  str += "\tEntry Count: " + std::to_string(entry_count_) + "\n";
  str += "\tMin Val (perfect hash only): " + std::to_string(min_val_) + "\n";
  str += "\tMax Val (perfect hash only): " + std::to_string(max_val_) + "\n";
  str += "\tBucket Val (perfect hash only): " + std::to_string(bucket_) + "\n";
  str += "\tSort on GPU: " + ::toString(sort_on_gpu_) + "\n";
  str += "\tUse Streaming Top N: " + ::toString(use_streaming_top_n_) + "\n";
  str += "\tOutput Columnar: " + ::toString(output_columnar_) + "\n";
  str += "\tRender Output: " + ::toString(render_output_) + "\n";
  str += "\tUse Baseline Sort: " + ::toString(must_use_baseline_sort_) + "\n";
  return str;
}

std::string QueryMemoryDescriptor::reductionKey() const {
  std::string str;
  str += "Query Memory Descriptor State\n";
  str += "\tQuery Type: " + queryDescTypeToString() + "\n";
  str +=
      "\tKeyless Hash: " + ::toString(keyless_hash_) +
      (keyless_hash_ ? ", target index for key: " + std::to_string(getTargetIdxForKey())
                     : "") +
      "\n";
  str += "\tEffective key width: " + std::to_string(getEffectiveKeyWidth()) + "\n";
  str += "\tNumber of group columns: " + std::to_string(getGroupbyColCount()) + "\n";
  const auto group_indices_size = targetGroupbyIndicesSize();
  if (group_indices_size) {
    std::vector<std::string> group_indices_strings;
    for (size_t target_idx = 0; target_idx < group_indices_size; ++target_idx) {
      group_indices_strings.push_back(std::to_string(getTargetGroupbyIndex(target_idx)));
    }
    str += "\tTarget group by indices: " +
           boost::algorithm::join(group_indices_strings, ",") + "\n";
  }
  str += "\t" + col_slot_context_.toString();
  return str;
}

std::vector<TargetInfo> target_exprs_to_infos(
    const std::vector<Analyzer::Expr*>& targets,
    const QueryMemoryDescriptor& query_mem_desc) {
  std::vector<TargetInfo> target_infos;
  for (const auto target_expr : targets) {
    auto target = get_target_info(target_expr, g_bigint_count);
    if (query_mem_desc.getQueryDescriptionType() ==
        QueryDescriptionType::NonGroupedAggregate) {
      set_notnull(target, false);
      target.sql_type.set_notnull(false);
    }
    target_infos.push_back(target);
  }
  return target_infos;
}
