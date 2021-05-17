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

#ifndef QUERYENGINE_QUERYEXECUTIONCONTEXT_H
#define QUERYENGINE_QUERYEXECUTIONCONTEXT_H

#include "CompilationOptions.h"
#include "DataMgr/Allocators/CudaAllocator.h"
#include "GpuMemUtils.h"
#include "Rendering/RenderInfo.h"
#include "ResultSet.h"

#include "QueryMemoryInitializer.h"

#include <boost/core/noncopyable.hpp>
#include <vector>

class GpuCompilationContext;
class CpuCompilationContext;

struct RelAlgExecutionUnit;
class QueryMemoryDescriptor;
class Executor;

class QueryExecutionContext : boost::noncopyable {
 public:
  // TODO(alex): remove device_type
  QueryExecutionContext(const RelAlgExecutionUnit& ra_exe_unit,
                        const QueryMemoryDescriptor&,
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
                        RenderInfo*);

  ResultSetPtr getRowSet(const RelAlgExecutionUnit& ra_exe_unit,
                         const QueryMemoryDescriptor& query_mem_desc) const;

  ResultSetPtr groupBufferToResults(const size_t i) const;

  std::vector<int64_t*> launchGpuCode(
      const RelAlgExecutionUnit& ra_exe_unit,
      const GpuCompilationContext* cu_functions,
      const bool hoist_literals,
      const std::vector<int8_t>& literal_buff,
      std::vector<std::vector<const int8_t*>> col_buffers,
      const std::vector<std::vector<int64_t>>& num_rows,
      const std::vector<std::vector<uint64_t>>& frag_row_offsets,
      const int32_t scan_limit,
      Data_Namespace::DataMgr* data_mgr,
      const unsigned block_size_x,
      const unsigned grid_size_x,
      const int device_id,
      const size_t shared_memory_size,
      int32_t* error_code,
      const uint32_t num_tables,
      const bool allow_runtime_interrupt,
      const std::vector<int64_t>& join_hash_tables,
      RenderAllocatorMap* render_allocator_map);

  std::vector<int64_t*> launchCpuCode(
      const RelAlgExecutionUnit& ra_exe_unit,
      const CpuCompilationContext* fn_ptrs,
      const bool hoist_literals,
      const std::vector<int8_t>& literal_buff,
      std::vector<std::vector<const int8_t*>> col_buffers,
      const std::vector<std::vector<int64_t>>& num_rows,
      const std::vector<std::vector<uint64_t>>& frag_row_offsets,
      const int32_t scan_limit,
      int32_t* error_code,
      const uint32_t num_tables,
      const std::vector<int64_t>& join_hash_tables,
      const int64_t num_rows_to_process = -1);

  int64_t getAggInitValForIndex(const size_t index) const;

 private:
#ifdef HAVE_CUDA
  enum {
    COL_BUFFERS,
    NUM_FRAGMENTS,
    LITERALS,
    NUM_ROWS,
    FRAG_ROW_OFFSETS,
    MAX_MATCHED,
    TOTAL_MATCHED,
    INIT_AGG_VALS,
    GROUPBY_BUF,
    ERROR_CODE,
    NUM_TABLES,
    JOIN_HASH_TABLES,
    KERN_PARAM_COUNT,
  };

  void initializeDynamicWatchdog(void* native_module, const int device_id) const;

  void initializeRuntimeInterrupter(void* native_module, const int device_id) const;

  std::vector<CUdeviceptr> prepareKernelParams(
      const std::vector<std::vector<const int8_t*>>& col_buffers,
      const std::vector<int8_t>& literal_buff,
      const std::vector<std::vector<int64_t>>& num_rows,
      const std::vector<std::vector<uint64_t>>& frag_offsets,
      const int32_t scan_limit,
      const std::vector<int64_t>& init_agg_vals,
      const std::vector<int32_t>& error_codes,
      const uint32_t num_tables,
      const std::vector<int64_t>& join_hash_tables,
      Data_Namespace::DataMgr* data_mgr,
      const int device_id,
      const bool hoist_literals,
      const bool is_group_by) const;
#endif  // HAVE_CUDA

  ResultSetPtr groupBufferToDeinterleavedResults(const size_t i) const;

  std::unique_ptr<CudaAllocator> gpu_allocator_;

  // TODO(adb): convert to shared_ptr
  QueryMemoryDescriptor query_mem_desc_;
  const Executor* executor_;
  const ExecutorDeviceType device_type_;
  const ExecutorDispatchMode dispatch_mode_;
  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;
  const bool output_columnar_;
  std::unique_ptr<QueryMemoryInitializer> query_buffers_;
  mutable std::unique_ptr<ResultSet> estimator_result_set_;

  friend class Executor;
};

#endif  // QUERYENGINE_QUERYEXECUTIONCONTEXT_H
