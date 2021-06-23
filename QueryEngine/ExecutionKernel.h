/*
 * Copyright 2020 OmniSci, Inc.
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

#include "Logger/Logger.h"
#include "QueryEngine/ColumnFetcher.h"
#include "QueryEngine/Descriptors/QueryCompilationDescriptor.h"

#include "Shared/threadpool.h"

#ifdef HAVE_TBB
#include "tbb/enumerable_thread_specific.h"
#endif

class SharedKernelContext {
 public:
  SharedKernelContext(const std::vector<InputTableInfo>& query_infos)
      : query_infos_(query_infos)
#ifdef HAVE_TBB
      , thread_pool_(nullptr)
#endif
  {
  }

  const std::vector<uint64_t>& getFragOffsets();

  void addDeviceResults(ResultSetPtr&& device_results,
                        std::vector<size_t> outer_table_fragment_ids);

  std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& getFragmentResults();

  const std::vector<InputTableInfo>& getQueryInfos() const { return query_infos_; }

  std::atomic_flag dynamic_watchdog_set = ATOMIC_FLAG_INIT;

#ifdef HAVE_TBB
  auto getThreadPool() { return thread_pool_; }
  void setThreadPool(threadpool::ThreadPool<void>* pool) { thread_pool_ = pool; }
  auto& getTlsExecutionContext() { return tls_execution_context_; }
#endif  // HAVE_TBB

 private:
  std::mutex reduce_mutex_;
  std::vector<std::pair<ResultSetPtr, std::vector<size_t>>> all_fragment_results_;

  std::vector<uint64_t> all_frag_row_offsets_;
  std::mutex all_frag_row_offsets_mutex_;
  const std::vector<InputTableInfo>& query_infos_;
  const RegisteredQueryHint query_hint_;

#ifdef HAVE_TBB
  threadpool::ThreadPool<void>* thread_pool_;
  tbb::enumerable_thread_specific<std::unique_ptr<QueryExecutionContext>>
      tls_execution_context_;
#endif  // HAVE_TBB
};

class ExecutionKernel {
 public:
  ExecutionKernel(const RelAlgExecutionUnit& ra_exe_unit,
                  const ExecutorDeviceType chosen_device_type,
                  int chosen_device_id,
                  const ExecutionOptions& eo,
                  const ColumnFetcher& column_fetcher,
                  const QueryCompilationDescriptor& query_comp_desc,
                  const QueryMemoryDescriptor& query_mem_desc,
                  const FragmentsList& frag_list,
                  const ExecutorDispatchMode kernel_dispatch_mode,
                  RenderInfo* render_info,
                  const int64_t rowid_lookup_key)
      : ra_exe_unit_(ra_exe_unit)
      , chosen_device_type(chosen_device_type)
      , chosen_device_id(chosen_device_id)
      , eo(eo)
      , column_fetcher(column_fetcher)
      , query_comp_desc(query_comp_desc)
      , query_mem_desc(query_mem_desc)
      , frag_list(frag_list)
      , kernel_dispatch_mode(kernel_dispatch_mode)
      , render_info_(render_info)
      , rowid_lookup_key(rowid_lookup_key) {}

  void run(Executor* executor,
           const size_t thread_idx,
           SharedKernelContext& shared_context);

  const RelAlgExecutionUnit& ra_exe_unit_;

 private:
  const ExecutorDeviceType chosen_device_type;
  int chosen_device_id;
  const ExecutionOptions& eo;
  const ColumnFetcher& column_fetcher;
  const QueryCompilationDescriptor& query_comp_desc;
  const QueryMemoryDescriptor& query_mem_desc;
  const FragmentsList frag_list;
  const ExecutorDispatchMode kernel_dispatch_mode;
  RenderInfo* render_info_;
  const int64_t rowid_lookup_key;

  ResultSetPtr device_results_;

  void runImpl(Executor* executor,
               const size_t thread_idx,
               SharedKernelContext& shared_context);

  friend class KernelSubtask;
};

#ifdef HAVE_TBB
class KernelSubtask {
 public:
  KernelSubtask(ExecutionKernel& k,
                SharedKernelContext& shared_context,
                std::shared_ptr<FetchResult> fetch_result,
                std::shared_ptr<std::list<ChunkIter>> chunk_iterators,
                int64_t total_num_input_rows,
                size_t start_rowid,
                size_t num_rows_to_process,
                size_t thread_idx)
      : kernel_(k)
      , shared_context_(shared_context)
      , fetch_result_(fetch_result)
      , chunk_iterators_(chunk_iterators)
      , total_num_input_rows_(total_num_input_rows)
      , start_rowid_(start_rowid)
      , num_rows_to_process_(num_rows_to_process)
      , thread_idx_(thread_idx) {}

  void run(Executor* executor);

 private:
  void runImpl(Executor* executor);

  ExecutionKernel& kernel_;
  SharedKernelContext& shared_context_;
  std::shared_ptr<FetchResult> fetch_result_;
  std::shared_ptr<std::list<ChunkIter>> chunk_iterators_;
  int64_t total_num_input_rows_;
  size_t start_rowid_;
  size_t num_rows_to_process_;
  size_t thread_idx_;
};
#endif  // HAVE_TBB
