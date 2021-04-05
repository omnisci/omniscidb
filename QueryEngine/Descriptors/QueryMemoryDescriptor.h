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
 * @file    QueryMemoryDescriptor.h
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Descriptor for the result set buffer layout.
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef QUERYENGINE_QUERYMEMORYDESCRIPTOR_H
#define QUERYENGINE_QUERYMEMORYDESCRIPTOR_H

#include "../CompilationOptions.h"
#include "../CountDistinct.h"
#include "ColSlotContext.h"
#include "Types.h"

#include <boost/optional.hpp>
#include "Logger/Logger.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <vector>

#include <Shared/SqlTypesLayout.h>
#include <Shared/TargetInfo.h>

extern bool g_cluster;

class Executor;
class QueryExecutionContext;
class RenderInfo;
class RowSetMemoryOwner;
struct InputTableInfo;
struct RelAlgExecutionUnit;
class TResultSetBufferDescriptor;
class GroupByAndAggregate;
struct ColRangeInfo;
struct KeylessInfo;

class StreamingTopNOOM : public std::runtime_error {
 public:
  StreamingTopNOOM(const size_t heap_size_bytes)
      : std::runtime_error("Unable to use streaming top N due to required heap size of " +
                           std::to_string(heap_size_bytes) +
                           " bytes exceeding maximum slab size.") {}
};

class QueryMemoryDescriptor {
 public:
  QueryMemoryDescriptor();

  // constructor for init call
  QueryMemoryDescriptor(const Executor* executor,
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
                        const bool output_columnar,
                        const bool render_output,
                        const bool must_use_baseline_sort,
                        const bool use_streaming_top_n);

  QueryMemoryDescriptor(const Executor* executor,
                        const size_t entry_count,
                        const QueryDescriptionType query_desc_type,
                        const bool is_table_function);

  QueryMemoryDescriptor(const QueryDescriptionType query_desc_type,
                        const int64_t min_val,
                        const int64_t max_val,
                        const bool has_nulls,
                        const std::vector<int8_t>& group_col_widths);

  // Serialization
  QueryMemoryDescriptor(const TResultSetBufferDescriptor& thrift_query_memory_descriptor);
  static TResultSetBufferDescriptor toThrift(const QueryMemoryDescriptor&);

  bool operator==(const QueryMemoryDescriptor& other) const;

  static std::unique_ptr<QueryMemoryDescriptor> init(
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
      const bool streaming_top_n_hint);

  std::unique_ptr<QueryExecutionContext> getQueryExecutionContext(
      const RelAlgExecutionUnit&,
      const Executor* executor,
      const ExecutorDeviceType device_type,
      const ExecutorDispatchMode dispatch_mode,
      const int device_id,
      const int64_t num_rows,
      const std::vector<std::vector<const int8_t*>>& col_buffers,
      const std::vector<std::vector<uint64_t>>& frag_offsets,
      std::shared_ptr<RowSetMemoryOwner>,
      const bool output_columnar,
      const bool sort_on_gpu,
      const size_t thread_idx,
      RenderInfo*) const;

  static bool many_entries(const int64_t max_val,
                           const int64_t min_val,
                           const int64_t bucket) {
    return max_val - min_val > 10000 * std::max(bucket, int64_t(1));
  }

  static bool countDescriptorsLogicallyEmpty(
      const CountDistinctDescriptors& count_distinct_descriptors) {
    return std::all_of(count_distinct_descriptors.begin(),
                       count_distinct_descriptors.end(),
                       [](const CountDistinctDescriptor& desc) {
                         return desc.impl_type_ == CountDistinctImplType::Invalid;
                       });
  }

  bool countDistinctDescriptorsLogicallyEmpty() const {
    return countDescriptorsLogicallyEmpty(count_distinct_descriptors_);
  }

  static int8_t pick_target_compact_width(const RelAlgExecutionUnit& ra_exe_unit,
                                          const std::vector<InputTableInfo>& query_infos,
                                          const int8_t crt_min_byte_width);

  // Getters and Setters
  const Executor* getExecutor() const { return executor_; }

  QueryDescriptionType getQueryDescriptionType() const { return query_desc_type_; }
  void setQueryDescriptionType(const QueryDescriptionType val) { query_desc_type_ = val; }
  bool isSingleColumnGroupByWithPerfectHash() const {
    return getQueryDescriptionType() == QueryDescriptionType::GroupByPerfectHash &&
           getGroupbyColCount() == 1;
  }

  bool hasKeylessHash() const { return keyless_hash_; }
  void setHasKeylessHash(const bool val) { keyless_hash_ = val; }

  bool hasInterleavedBinsOnGpu() const { return interleaved_bins_on_gpu_; }
  void setHasInterleavedBinsOnGpu(const bool val) { interleaved_bins_on_gpu_ = val; }

  int32_t getTargetIdxForKey() const { return idx_target_as_key_; }
  void setTargetIdxForKey(const int32_t val) { idx_target_as_key_ = val; }

  int8_t groupColWidth(const size_t key_idx) const {
    CHECK_LT(key_idx, group_col_widths_.size());
    return group_col_widths_[key_idx];
  }
  size_t getPrependedGroupColOffInBytes(const size_t group_idx) const;
  size_t getPrependedGroupBufferSizeInBytes() const;

  const auto groupColWidthsBegin() const { return group_col_widths_.begin(); }
  const auto groupColWidthsEnd() const { return group_col_widths_.end(); }
  void clearGroupColWidths() { group_col_widths_.clear(); }

  bool isGroupBy() const { return !group_col_widths_.empty(); }

  void setGroupColCompactWidth(const int8_t val) { group_col_compact_width_ = val; }

  size_t getColCount() const;
  size_t getSlotCount() const;

  const int8_t getPaddedSlotWidthBytes(const size_t slot_idx) const;
  const int8_t getLogicalSlotWidthBytes(const size_t slot_idx) const;

  const int8_t getSlotIndexForSingleSlotCol(const size_t col_idx) const;

  size_t getPaddedColWidthForRange(const size_t offset, const size_t range) const {
    size_t ret = 0;
    for (size_t i = offset; i < offset + range; i++) {
      ret += static_cast<size_t>(getPaddedSlotWidthBytes(i));
    }
    return ret;
  }

  void useConsistentSlotWidthSize(const int8_t slot_width_size);
  size_t getRowWidth() const;

  int8_t updateActualMinByteWidth(const int8_t actual_min_byte_width) const;

  void addColSlotInfo(const std::vector<std::tuple<int8_t, int8_t>>& slots_for_col);

  void clearSlotInfo();

  void alignPaddedSlots();

  int64_t getTargetGroupbyIndex(const size_t target_idx) const {
    CHECK_LT(target_idx, target_groupby_indices_.size());
    return target_groupby_indices_[target_idx];
  }

  void setAllTargetGroupbyIndices(std::vector<int64_t> group_by_indices) {
    target_groupby_indices_ = group_by_indices;
  }

  size_t targetGroupbyIndicesSize() const { return target_groupby_indices_.size(); }
  size_t targetGroupbyNegativeIndicesSize() const {
    return std::count_if(
        target_groupby_indices_.begin(),
        target_groupby_indices_.end(),
        [](const int64_t& target_group_by_index) { return target_group_by_index < 0; });
  }
  void clearTargetGroupbyIndices() { target_groupby_indices_.clear(); }

  size_t getEntryCount() const { return entry_count_; }
  void setEntryCount(const size_t val) { entry_count_ = val; }

  int64_t getMinVal() const { return min_val_; }
  int64_t getMaxVal() const { return max_val_; }
  int64_t getBucket() const { return bucket_; }

  bool hasNulls() const { return has_nulls_; }

  const CountDistinctDescriptor& getCountDistinctDescriptor(const size_t idx) const {
    CHECK_LT(idx, count_distinct_descriptors_.size());
    return count_distinct_descriptors_[idx];
  }
  size_t getCountDistinctDescriptorsSize() const {
    return count_distinct_descriptors_.size();
  }

  bool sortOnGpu() const { return sort_on_gpu_; }

  bool canOutputColumnar() const;
  bool didOutputColumnar() const { return output_columnar_; }
  void setOutputColumnar(const bool val);

  bool useStreamingTopN() const { return use_streaming_top_n_; }

  bool isLogicalSizedColumnsAllowed() const;

  bool mustUseBaselineSort() const { return must_use_baseline_sort_; }

  // TODO(adb): remove and store this info more naturally in another
  // member
  bool forceFourByteFloat() const { return force_4byte_float_; }
  void setForceFourByteFloat(const bool val) { force_4byte_float_ = val; }

  // Getters derived from state
  size_t getGroupbyColCount() const { return group_col_widths_.size(); }
  size_t getKeyCount() const { return keyless_hash_ ? 0 : getGroupbyColCount(); }
  size_t getBufferColSlotCount() const;

  size_t getBufferSizeBytes(const RelAlgExecutionUnit& ra_exe_unit,
                            const unsigned thread_count,
                            const ExecutorDeviceType device_type) const;
  size_t getBufferSizeBytes(const ExecutorDeviceType device_type) const;
  size_t getBufferSizeBytes(const ExecutorDeviceType device_type,
                            const size_t override_entry_count) const;

  const ColSlotContext& getColSlotContext() const { return col_slot_context_; }

  // TODO(alex): remove
  bool usesGetGroupValueFast() const;

  bool blocksShareMemory() const;
  bool threadsShareMemory() const;

  bool cpuThreadsShareMemory() const;

  bool lazyInitGroups(const ExecutorDeviceType) const;

  bool interleavedBins(const ExecutorDeviceType) const;

  size_t getColOffInBytes(const size_t col_idx) const;
  size_t getColOffInBytesInNextBin(const size_t col_idx) const;
  size_t getNextColOffInBytes(const int8_t* col_ptr,
                              const size_t bin,
                              const size_t col_idx) const;

  // returns the ptr offset of the next column, 64-bit aligned
  size_t getNextColOffInBytesRowOnly(const int8_t* col_ptr, const size_t col_idx) const;
  // returns the ptr offset of the current column, 64-bit aligned
  size_t getColOnlyOffInBytes(const size_t col_idx) const;
  size_t getRowSize() const;
  size_t getColsSize() const;
  size_t getWarpCount() const;

  size_t getCompactByteWidth() const;

  inline size_t getEffectiveKeyWidth() const {
    return group_col_compact_width_ ? group_col_compact_width_ : sizeof(int64_t);
  }

  bool isWarpSyncRequired(const ExecutorDeviceType) const;

  std::string queryDescTypeToString() const;
  std::string toString() const;

  std::string reductionKey() const;

 protected:
  void resetGroupColWidths(const std::vector<int8_t>& new_group_col_widths) {
    group_col_widths_ = new_group_col_widths;
  }

 private:
  const Executor* executor_;
  bool allow_multifrag_;
  QueryDescriptionType query_desc_type_;
  bool keyless_hash_;
  bool interleaved_bins_on_gpu_;
  int32_t idx_target_as_key_;  // If keyless_hash_ enabled, then represents what target
                               // expression should be used to identify the key (e.g., in
                               // locating empty entries). Currently only valid with
                               // keyless_hash_ and single-column GroupByPerfectHash
  std::vector<int8_t> group_col_widths_;
  int8_t group_col_compact_width_;  // compact width for all group
                                    // cols if able to be consistent
                                    // otherwise 0
  std::vector<int64_t> target_groupby_indices_;
  size_t entry_count_;  // the number of entries in the main buffer
  int64_t min_val_;     // meaningful for OneColKnownRange,
                        // MultiColPerfectHash only
  int64_t max_val_;
  int64_t bucket_;
  bool has_nulls_;
  CountDistinctDescriptors count_distinct_descriptors_;
  bool sort_on_gpu_;
  bool output_columnar_;
  bool render_output_;
  bool must_use_baseline_sort_;
  bool is_table_function_;
  bool use_streaming_top_n_;

  bool force_4byte_float_;

  ColSlotContext col_slot_context_;

  size_t getTotalBytesOfColumnarBuffers() const;
  size_t getTotalBytesOfColumnarBuffers(const size_t num_entries_per_column) const;
  size_t getTotalBytesOfColumnarProjections(const size_t projection_count) const;

  friend class ResultSet;
  friend class QueryExecutionContext;
};

inline void set_notnull(TargetInfo& target, const bool not_null) {
  target.skip_null_val = !not_null;
  auto new_type = get_compact_type(target);
  new_type.set_notnull(not_null);
  set_compact_type(target, new_type);
}

std::vector<TargetInfo> target_exprs_to_infos(
    const std::vector<Analyzer::Expr*>& targets,
    const QueryMemoryDescriptor& query_mem_desc);

#endif  // QUERYENGINE_QUERYMEMORYDESCRIPTOR_H
