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

#pragma once

#include "DataMgr/Allocators/DeviceAllocator.h"

#include "Descriptors/QueryMemoryDescriptor.h"
#include "GpuMemUtils.h"
#include "ResultSet.h"

#include "Rendering/RenderAllocator.h"

#include <memory>

#ifdef HAVE_CUDA
#include <cuda.h>
#else
#include <Shared/nocuda.h>
#endif

class QueryMemoryInitializer {
 public:
  // Row-based execution constructor
  QueryMemoryInitializer(const RelAlgExecutionUnit& ra_exe_unit,
                         const QueryMemoryDescriptor& query_mem_desc,
                         const int device_id,
                         const ExecutorDeviceType device_type,
                         const ExecutorDispatchMode dispatch_mode,
                         const bool output_columnar,
                         const bool sort_on_gpu,
                         const int64_t num_rows,
                         const std::vector<std::vector<const int8_t*>>& col_buffers,
                         const std::vector<std::vector<uint64_t>>& frag_offsets,
                         RenderAllocatorMap* render_allocator_map,
                         RenderInfo* render_info,
                         std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                         DeviceAllocator* gpu_allocator,
                         const size_t thread_idx,
                         const Executor* executor);

  // Table functions execution constructor
  QueryMemoryInitializer(const TableFunctionExecutionUnit& exe_unit,
                         const QueryMemoryDescriptor& query_mem_desc,
                         const int device_id,
                         const ExecutorDeviceType device_type,
                         const int64_t num_rows,
                         const std::vector<std::vector<const int8_t*>>& col_buffers,
                         const std::vector<std::vector<uint64_t>>& frag_offsets,
                         std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                         DeviceAllocator* device_allocator,
                         const Executor* executor);

  const auto getCountDistinctBitmapPtr() const { return count_distinct_bitmap_mem_; }

  const auto getCountDistinctHostPtr() const { return count_distinct_bitmap_host_mem_; }

  const auto getCountDistinctBitmapBytes() const {
    return count_distinct_bitmap_mem_bytes_;
  }

  ResultSet* getResultSet(const size_t index) const {
    CHECK_LT(index, result_sets_.size());
    return result_sets_[index].get();
  }

  std::unique_ptr<ResultSet> getResultSetOwned(const size_t index) {
    CHECK_LT(index, result_sets_.size());
    return std::move(result_sets_[index]);
  }

  void resetResultSet(const size_t index) {
    CHECK_LT(index, result_sets_.size());
    result_sets_[index].reset();
  }

  int64_t getAggInitValForIndex(const size_t index) const {
    CHECK_LT(index, init_agg_vals_.size());
    return init_agg_vals_[index];
  }

  const auto getGroupByBuffersPtr() {
    return reinterpret_cast<int64_t**>(group_by_buffers_.data());
  }

  const auto getGroupByBuffersSize() const { return group_by_buffers_.size(); }

  const auto getNumBuffers() const {
    CHECK_EQ(num_buffers_, group_by_buffers_.size());
    return num_buffers_;
  }

  GpuGroupByBuffers setupTableFunctionGpuBuffers(
      const QueryMemoryDescriptor& query_mem_desc,
      const int device_id,
      const unsigned block_size_x,
      const unsigned grid_size_x);
  void copyFromTableFunctionGpuBuffers(Data_Namespace::DataMgr* data_mgr,
                                       const QueryMemoryDescriptor& query_mem_desc,
                                       const size_t entry_count,
                                       const GpuGroupByBuffers& gpu_group_by_buffers,
                                       const int device_id,
                                       const unsigned block_size_x,
                                       const unsigned grid_size_x);

  void copyGroupByBuffersFromGpu(DeviceAllocator& device_allocator,
                                 const QueryMemoryDescriptor& query_mem_desc,
                                 const size_t entry_count,
                                 const GpuGroupByBuffers& gpu_group_by_buffers,
                                 const RelAlgExecutionUnit* ra_exe_unit,
                                 const unsigned block_size_x,
                                 const unsigned grid_size_x,
                                 const int device_id,
                                 const bool prepend_index_buffer) const;

 private:
  void initGroupByBuffer(int64_t* buffer,
                         const RelAlgExecutionUnit& ra_exe_unit,
                         const QueryMemoryDescriptor& query_mem_desc,
                         const ExecutorDeviceType device_type,
                         const bool output_columnar,
                         const Executor* executor);

  void initRowGroups(const QueryMemoryDescriptor& query_mem_desc,
                     int64_t* groups_buffer,
                     const std::vector<int64_t>& init_vals,
                     const int32_t groups_buffer_entry_count,
                     const size_t warp_size,
                     const Executor* executor);

  void initColumnarGroups(const QueryMemoryDescriptor& query_mem_desc,
                          int64_t* groups_buffer,
                          const std::vector<int64_t>& init_vals,
                          const Executor* executor);

  void initColumnsPerRow(const QueryMemoryDescriptor& query_mem_desc,
                         int8_t* row_ptr,
                         const std::vector<int64_t>& init_vals,
                         const std::vector<int64_t>& bitmap_sizes,
                         const std::vector<bool>& tdigest_deferred);

  void allocateCountDistinctGpuMem(const QueryMemoryDescriptor& query_mem_desc);

  std::vector<int64_t> allocateCountDistinctBuffers(
      const QueryMemoryDescriptor& query_mem_desc,
      const bool deferred,
      const Executor* executor);

  int64_t allocateCountDistinctBitmap(const size_t bitmap_byte_sz);

  int64_t allocateCountDistinctSet();

  std::vector<bool> allocateTDigests(const QueryMemoryDescriptor& query_mem_desc,
                                     const bool deferred,
                                     const Executor* executor);

  GpuGroupByBuffers prepareTopNHeapsDevBuffer(const QueryMemoryDescriptor& query_mem_desc,
                                              const int8_t* init_agg_vals_dev_ptr,
                                              const size_t n,
                                              const int device_id,
                                              const unsigned block_size_x,
                                              const unsigned grid_size_x);

  GpuGroupByBuffers createAndInitializeGroupByBufferGpu(
      const RelAlgExecutionUnit& ra_exe_unit,
      const QueryMemoryDescriptor& query_mem_desc,
      const int8_t* init_agg_vals_dev_ptr,
      const int device_id,
      const ExecutorDispatchMode dispatch_mode,
      const unsigned block_size_x,
      const unsigned grid_size_x,
      const int8_t warp_size,
      const bool can_sort_on_gpu,
      const bool output_columnar,
      RenderAllocator* render_allocator);

  size_t computeNumberOfBuffers(const QueryMemoryDescriptor& query_mem_desc,
                                const ExecutorDeviceType device_type,
                                const Executor* executor) const;

  void compactProjectionBuffersCpu(const QueryMemoryDescriptor& query_mem_desc,
                                   const size_t projection_count);
  void compactProjectionBuffersGpu(const QueryMemoryDescriptor& query_mem_desc,
                                   Data_Namespace::DataMgr* data_mgr,
                                   const GpuGroupByBuffers& gpu_group_by_buffers,
                                   const size_t projection_count,
                                   const int device_id);

  void applyStreamingTopNOffsetCpu(const QueryMemoryDescriptor& query_mem_desc,
                                   const RelAlgExecutionUnit& ra_exe_unit);

  void applyStreamingTopNOffsetGpu(Data_Namespace::DataMgr* data_mgr,
                                   const QueryMemoryDescriptor& query_mem_desc,
                                   const GpuGroupByBuffers& gpu_group_by_buffers,
                                   const RelAlgExecutionUnit& ra_exe_unit,
                                   const unsigned total_thread_count,
                                   const int device_id);

  const int64_t num_rows_;
  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;
  std::vector<std::unique_ptr<ResultSet>> result_sets_;

  std::vector<int64_t> init_agg_vals_;

  const size_t num_buffers_;
  std::vector<int64_t*> group_by_buffers_;

  CUdeviceptr count_distinct_bitmap_mem_;
  size_t count_distinct_bitmap_mem_bytes_;
  int8_t* count_distinct_bitmap_crt_ptr_;
  int8_t* count_distinct_bitmap_host_mem_;

  DeviceAllocator* device_allocator_{nullptr};
  std::vector<Data_Namespace::AbstractBuffer*> temporary_buffers_;

  const size_t thread_idx_;

  friend class Executor;  // Accesses result_sets_
  friend class QueryExecutionContext;
};
