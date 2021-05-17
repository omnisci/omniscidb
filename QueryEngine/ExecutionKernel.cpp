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

#include "QueryEngine/ExecutionKernel.h"

#include <mutex>
#include <vector>

#include "QueryEngine/Descriptors/RowSetMemoryOwner.h"
#include "QueryEngine/DynamicWatchdog.h"
#include "QueryEngine/ErrorHandling.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ExternalExecutor.h"
#include "QueryEngine/SerializeToSql.h"

namespace {

bool needs_skip_result(const ResultSetPtr& res) {
  return !res || res->definitelyHasNoRows();
}

inline bool query_has_inner_join(const RelAlgExecutionUnit& ra_exe_unit) {
  return (std::count_if(ra_exe_unit.join_quals.begin(),
                        ra_exe_unit.join_quals.end(),
                        [](const auto& join_condition) {
                          return join_condition.type == JoinType::INNER;
                        }) > 0);
}

// column is part of the target expressions, result set iteration needs it alive.
bool need_to_hold_chunk(const Chunk_NS::Chunk* chunk,
                        const RelAlgExecutionUnit& ra_exe_unit) {
  CHECK(chunk->getColumnDesc());
  const auto chunk_ti = chunk->getColumnDesc()->columnType;
  if (chunk_ti.is_array() ||
      (chunk_ti.is_string() && chunk_ti.get_compression() == kENCODING_NONE)) {
    for (const auto target_expr : ra_exe_unit.target_exprs) {
      const auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(target_expr);
      if (col_var && col_var->get_column_id() == chunk->getColumnDesc()->columnId &&
          col_var->get_table_id() == chunk->getColumnDesc()->tableId) {
        return true;
      }
    }
  }
  return false;
}

bool need_to_hold_chunk(const std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunks,
                        const RelAlgExecutionUnit& ra_exe_unit) {
  for (const auto& chunk : chunks) {
    if (need_to_hold_chunk(chunk.get(), ra_exe_unit)) {
      return true;
    }
  }

  return false;
}

}  // namespace

const std::vector<uint64_t>& SharedKernelContext::getFragOffsets() {
  std::lock_guard<std::mutex> lock(all_frag_row_offsets_mutex_);
  if (all_frag_row_offsets_.empty()) {
    all_frag_row_offsets_.resize(query_infos_.front().info.fragments.size() + 1);
    for (size_t i = 1; i <= query_infos_.front().info.fragments.size(); ++i) {
      all_frag_row_offsets_[i] =
          all_frag_row_offsets_[i - 1] +
          query_infos_.front().info.fragments[i - 1].getNumTuples();
    }
  }
  return all_frag_row_offsets_;
}

void SharedKernelContext::addDeviceResults(ResultSetPtr&& device_results,
                                           std::vector<size_t> outer_table_fragment_ids) {
  std::lock_guard<std::mutex> lock(reduce_mutex_);
  if (!needs_skip_result(device_results)) {
    all_fragment_results_.emplace_back(std::move(device_results),
                                       outer_table_fragment_ids);
  }
}

std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>&
SharedKernelContext::getFragmentResults() {
  return all_fragment_results_;
}

void ExecutionKernel::run(Executor* executor,
                          const size_t thread_idx,
                          SharedKernelContext& shared_context) {
  DEBUG_TIMER("ExecutionKernel::run");
  INJECT_TIMER(kernel_run);
  try {
    runImpl(executor, thread_idx, shared_context);
  } catch (const OutOfHostMemory& e) {
    throw QueryExecutionError(Executor::ERR_OUT_OF_CPU_MEM, e.what());
  } catch (const std::bad_alloc& e) {
    throw QueryExecutionError(Executor::ERR_OUT_OF_CPU_MEM, e.what());
  } catch (const OutOfRenderMemory& e) {
    throw QueryExecutionError(Executor::ERR_OUT_OF_RENDER_MEM, e.what());
  } catch (const OutOfMemory& e) {
    throw QueryExecutionError(
        Executor::ERR_OUT_OF_GPU_MEM,
        e.what(),
        QueryExecutionProperties{
            query_mem_desc.getQueryDescriptionType(),
            kernel_dispatch_mode == ExecutorDispatchMode::MultifragmentKernel});
  } catch (const ColumnarConversionNotSupported& e) {
    throw QueryExecutionError(Executor::ERR_COLUMNAR_CONVERSION_NOT_SUPPORTED, e.what());
  } catch (const TooManyLiterals& e) {
    throw QueryExecutionError(Executor::ERR_TOO_MANY_LITERALS, e.what());
  } catch (const StringConstInResultSet& e) {
    throw QueryExecutionError(Executor::ERR_STRING_CONST_IN_RESULTSET, e.what());
  } catch (const QueryExecutionError& e) {
    throw e;
  }
}

void ExecutionKernel::runImpl(Executor* executor,
                              const size_t thread_idx,
                              SharedKernelContext& shared_context) {
  CHECK(executor);
  const auto memory_level = chosen_device_type == ExecutorDeviceType::GPU
                                ? Data_Namespace::GPU_LEVEL
                                : Data_Namespace::CPU_LEVEL;
  CHECK_GE(frag_list.size(), size_t(1));
  // frag_list[0].table_id is how we tell which query we are running for UNION ALL.
  const int outer_table_id = ra_exe_unit_.union_all
                                 ? frag_list[0].table_id
                                 : ra_exe_unit_.input_descs[0].getTableId();
  CHECK_EQ(frag_list[0].table_id, outer_table_id);
  const auto& outer_tab_frag_ids = frag_list[0].fragment_ids;

  CHECK_GE(chosen_device_id, 0);
  CHECK_LT(chosen_device_id, Executor::max_gpu_count);

  auto catalog = executor->getCatalog();
  CHECK(catalog);

  // need to own them while query executes
  auto chunk_iterators_ptr = std::make_shared<std::list<ChunkIter>>();
  std::list<std::shared_ptr<Chunk_NS::Chunk>> chunks;
  std::unique_ptr<std::lock_guard<std::mutex>> gpu_lock;
  std::unique_ptr<CudaAllocator> device_allocator;
  if (chosen_device_type == ExecutorDeviceType::GPU) {
    gpu_lock.reset(
        new std::lock_guard<std::mutex>(executor->gpu_exec_mutex_[chosen_device_id]));
    device_allocator =
        std::make_unique<CudaAllocator>(&catalog->getDataMgr(), chosen_device_id);
  }
  std::shared_ptr<FetchResult> fetch_result(new FetchResult);
  try {
    std::map<int, const TableFragments*> all_tables_fragments;
    QueryFragmentDescriptor::computeAllTablesFragments(
        all_tables_fragments, ra_exe_unit_, shared_context.getQueryInfos());

    *fetch_result = ra_exe_unit_.union_all
                        ? executor->fetchUnionChunks(column_fetcher,
                                                     ra_exe_unit_,
                                                     chosen_device_id,
                                                     memory_level,
                                                     all_tables_fragments,
                                                     frag_list,
                                                     *catalog,
                                                     *chunk_iterators_ptr,
                                                     chunks,
                                                     device_allocator.get(),
                                                     thread_idx,
                                                     eo.allow_runtime_query_interrupt)
                        : executor->fetchChunks(column_fetcher,
                                                ra_exe_unit_,
                                                chosen_device_id,
                                                memory_level,
                                                all_tables_fragments,
                                                frag_list,
                                                *catalog,
                                                *chunk_iterators_ptr,
                                                chunks,
                                                device_allocator.get(),
                                                thread_idx,
                                                eo.allow_runtime_query_interrupt);
    if (fetch_result->num_rows.empty()) {
      return;
    }
    if (eo.with_dynamic_watchdog &&
        !shared_context.dynamic_watchdog_set.test_and_set(std::memory_order_acquire)) {
      CHECK_GT(eo.dynamic_watchdog_time_limit, 0u);
      auto cycle_budget = dynamic_watchdog_init(eo.dynamic_watchdog_time_limit);
      LOG(INFO) << "Dynamic Watchdog budget: CPU: "
                << std::to_string(eo.dynamic_watchdog_time_limit) << "ms, "
                << std::to_string(cycle_budget) << " cycles";
    }
  } catch (const OutOfMemory&) {
    throw QueryExecutionError(
        memory_level == Data_Namespace::GPU_LEVEL ? Executor::ERR_OUT_OF_GPU_MEM
                                                  : Executor::ERR_OUT_OF_CPU_MEM,
        QueryExecutionProperties{
            query_mem_desc.getQueryDescriptionType(),
            kernel_dispatch_mode == ExecutorDispatchMode::MultifragmentKernel});
    return;
  }

  if (eo.executor_type == ExecutorType::Extern) {
    if (ra_exe_unit_.input_descs.size() > 1) {
      throw std::runtime_error("Joins not supported through external execution");
    }
    const auto query = serialize_to_sql(&ra_exe_unit_, catalog);
    GroupByAndAggregate group_by_and_aggregate(executor,
                                               ExecutorDeviceType::CPU,
                                               ra_exe_unit_,
                                               shared_context.getQueryInfos(),
                                               executor->row_set_mem_owner_,
                                               std::nullopt);
    const auto query_mem_desc =
        group_by_and_aggregate.initQueryMemoryDescriptor(false, 0, 8, nullptr, false);
    device_results_ = run_query_external(
        query,
        *fetch_result,
        executor->plan_state_.get(),
        ExternalQueryOutputSpec{
            *query_mem_desc,
            target_exprs_to_infos(ra_exe_unit_.target_exprs, *query_mem_desc),
            executor});
    shared_context.addDeviceResults(std::move(device_results_), outer_tab_frag_ids);
    return;
  }
  const CompilationResult& compilation_result = query_comp_desc.getCompilationResult();
  std::unique_ptr<QueryExecutionContext> query_exe_context_owned;
  const bool do_render = render_info_ && render_info_->isPotentialInSituRender();

  int64_t total_num_input_rows{-1};
  if (kernel_dispatch_mode == ExecutorDispatchMode::KernelPerFragment &&
      query_mem_desc.getQueryDescriptionType() == QueryDescriptionType::Projection) {
    total_num_input_rows = 0;
    std::for_each(fetch_result->num_rows.begin(),
                  fetch_result->num_rows.end(),
                  [&total_num_input_rows](const std::vector<int64_t>& frag_row_count) {
                    total_num_input_rows = std::accumulate(frag_row_count.begin(),
                                                           frag_row_count.end(),
                                                           total_num_input_rows);
                  });
    VLOG(2) << "total_num_input_rows=" << total_num_input_rows;
    // TODO(adb): we may want to take this early out for all queries, but we are most
    // likely to see this query pattern on the kernel per fragment path (e.g. with HAVING
    // 0=1)
    if (total_num_input_rows == 0) {
      return;
    }

    if (query_has_inner_join(ra_exe_unit_)) {
      total_num_input_rows *= ra_exe_unit_.input_descs.size();
    }
  }

  uint32_t start_rowid{0};
  if (rowid_lookup_key >= 0) {
    if (!frag_list.empty()) {
      const auto& all_frag_row_offsets = shared_context.getFragOffsets();
      start_rowid = rowid_lookup_key -
                    all_frag_row_offsets[frag_list.begin()->fragment_ids.front()];
    }
  }

#ifdef HAVE_TBB
  bool can_run_subkernels = shared_context.getThreadPool() != nullptr;

  // Subfragments are supported for groupby queries and estimators only for now.
  can_run_subkernels = can_run_subkernels &&
                       (!ra_exe_unit_.groupby_exprs.empty() || ra_exe_unit_.estimator);

  // In case some column is lazily fetched, we cannot mix different
  // fragments in a single ResultSet.
  can_run_subkernels =
      can_run_subkernels && !executor->hasLazyFetchColumns(ra_exe_unit_.target_exprs);

  // TODO: Use another structure to hold chunks. Currently,
  // ResultSet holds them, but with subfragments chunk can
  // be referenced by many ResultSets. So, some outer
  // structure to hold all ResultSets and all chunks is
  // required.
  can_run_subkernels = can_run_subkernels && !need_to_hold_chunk(chunks, ra_exe_unit_);

  // TODO: check for literals? We serialize literals before execution
  // and hold them in result sets. Can we simply do it once and hold
  // in an outer structure?

  if (can_run_subkernels) {
    size_t total_rows = fetch_result->num_rows[0][0];
    size_t sub_size = g_subfragment_size;

    for (size_t sub_start = start_rowid; sub_start < total_rows; sub_start += sub_size) {
      sub_size = (sub_start + sub_size > total_rows) ? total_rows - sub_start : sub_size;
      auto subtask = std::make_shared<KernelSubtask>(*this,
                                                     shared_context,
                                                     fetch_result,
                                                     total_num_input_rows,
                                                     sub_start,
                                                     sub_size,
                                                     thread_idx);
      shared_context.getThreadPool()->spawn(
          [subtask, executor] { subtask->run(executor); });
    }

    return;
  }
#endif  // HAVE_TBB

  if (eo.executor_type == ExecutorType::Native) {
    try {
      query_exe_context_owned =
          query_mem_desc.getQueryExecutionContext(ra_exe_unit_,
                                                  executor,
                                                  chosen_device_type,
                                                  kernel_dispatch_mode,
                                                  chosen_device_id,
                                                  total_num_input_rows,
                                                  fetch_result->col_buffers,
                                                  fetch_result->frag_offsets,
                                                  executor->getRowSetMemoryOwner(),
                                                  compilation_result.output_columnar,
                                                  query_mem_desc.sortOnGpu(),
                                                  thread_idx,
                                                  do_render ? render_info_ : nullptr);
    } catch (const OutOfHostMemory& e) {
      throw QueryExecutionError(Executor::ERR_OUT_OF_CPU_MEM);
    }
  }
  QueryExecutionContext* query_exe_context{query_exe_context_owned.get()};
  CHECK(query_exe_context);
  int32_t err{0};

  if (ra_exe_unit_.groupby_exprs.empty()) {
    err = executor->executePlanWithoutGroupBy(ra_exe_unit_,
                                              compilation_result,
                                              query_comp_desc.hoistLiterals(),
                                              &device_results_,
                                              ra_exe_unit_.target_exprs,
                                              chosen_device_type,
                                              fetch_result->col_buffers,
                                              query_exe_context,
                                              fetch_result->num_rows,
                                              fetch_result->frag_offsets,
                                              &catalog->getDataMgr(),
                                              chosen_device_id,
                                              start_rowid,
                                              ra_exe_unit_.input_descs.size(),
                                              eo.allow_runtime_query_interrupt,
                                              do_render ? render_info_ : nullptr);
  } else {
    if (ra_exe_unit_.union_all) {
      VLOG(1) << "outer_table_id=" << outer_table_id
              << " ra_exe_unit_.scan_limit=" << ra_exe_unit_.scan_limit;
    }
    err = executor->executePlanWithGroupBy(ra_exe_unit_,
                                           compilation_result,
                                           query_comp_desc.hoistLiterals(),
                                           &device_results_,
                                           chosen_device_type,
                                           fetch_result->col_buffers,
                                           outer_tab_frag_ids,
                                           query_exe_context,
                                           fetch_result->num_rows,
                                           fetch_result->frag_offsets,
                                           &catalog->getDataMgr(),
                                           chosen_device_id,
                                           outer_table_id,
                                           ra_exe_unit_.scan_limit,
                                           start_rowid,
                                           ra_exe_unit_.input_descs.size(),
                                           eo.allow_runtime_query_interrupt,
                                           do_render ? render_info_ : nullptr);
  }
  if (device_results_) {
    std::list<std::shared_ptr<Chunk_NS::Chunk>> chunks_to_hold;
    for (const auto& chunk : chunks) {
      if (need_to_hold_chunk(chunk.get(), ra_exe_unit_)) {
        chunks_to_hold.push_back(chunk);
      }
    }
    device_results_->holdChunks(chunks_to_hold);
    device_results_->holdChunkIterators(chunk_iterators_ptr);
  } else {
    VLOG(1) << "null device_results.";
  }
  if (err) {
    throw QueryExecutionError(err);
  }
  shared_context.addDeviceResults(std::move(device_results_), outer_tab_frag_ids);
}

#ifdef HAVE_TBB

void KernelSubtask::run(Executor* executor) {
  auto& query_exe_context_owned = shared_context_.getTlsExecutionContext().local();
  const bool do_render =
      kernel_.render_info_ && kernel_.render_info_->isPotentialInSituRender();
  const CompilationResult& compilation_result =
      kernel_.query_comp_desc.getCompilationResult();

  if (!query_exe_context_owned) {
    try {
      // We pass fake col_buffers and frag_offsets. These are not actually used
      // for subtasks but shouldn't pass empty structures to avoid empty results.
      std::vector<std::vector<const int8_t*>> col_buffers(
          fetch_result_->col_buffers.size(),
          std::vector<const int8_t*>(fetch_result_->col_buffers[0].size()));
      std::vector<std::vector<uint64_t>> frag_offsets(
          fetch_result_->frag_offsets.size(),
          std::vector<uint64_t>(fetch_result_->frag_offsets[0].size()));
      query_exe_context_owned = kernel_.query_mem_desc.getQueryExecutionContext(
          kernel_.ra_exe_unit_,
          executor,
          kernel_.chosen_device_type,
          kernel_.kernel_dispatch_mode,
          kernel_.chosen_device_id,
          total_num_input_rows_,
          col_buffers,
          frag_offsets,
          executor->getRowSetMemoryOwner(),
          compilation_result.output_columnar,
          kernel_.query_mem_desc.sortOnGpu(),
          // TODO: use TBB thread id to choose allocator
          thread_idx_,
          do_render ? kernel_.render_info_ : nullptr);
    } catch (const OutOfHostMemory& e) {
      throw QueryExecutionError(Executor::ERR_OUT_OF_CPU_MEM);
    }
  }

  const int outer_table_id = kernel_.ra_exe_unit_.union_all
                                 ? kernel_.frag_list[0].table_id
                                 : kernel_.ra_exe_unit_.input_descs[0].getTableId();
  const auto& outer_tab_frag_ids = kernel_.frag_list[0].fragment_ids;
  auto catalog = executor->getCatalog();
  CHECK(catalog);
  QueryExecutionContext* query_exe_context{query_exe_context_owned.get()};
  CHECK(query_exe_context);
  int32_t err{0};

  // ResultSetPtr results;
  if (kernel_.ra_exe_unit_.groupby_exprs.empty()) {
    err = executor->executePlanWithoutGroupBy(kernel_.ra_exe_unit_,
                                              compilation_result,
                                              kernel_.query_comp_desc.hoistLiterals(),
                                              nullptr,
                                              kernel_.ra_exe_unit_.target_exprs,
                                              kernel_.chosen_device_type,
                                              fetch_result_->col_buffers,
                                              query_exe_context,
                                              fetch_result_->num_rows,
                                              fetch_result_->frag_offsets,
                                              &catalog->getDataMgr(),
                                              kernel_.chosen_device_id,
                                              start_rowid_,
                                              kernel_.ra_exe_unit_.input_descs.size(),
                                              kernel_.eo.allow_runtime_query_interrupt,
                                              do_render ? kernel_.render_info_ : nullptr,
                                              start_rowid_ + num_rows_to_process_);
  } else {
    err = executor->executePlanWithGroupBy(kernel_.ra_exe_unit_,
                                           compilation_result,
                                           kernel_.query_comp_desc.hoistLiterals(),
                                           nullptr,
                                           kernel_.chosen_device_type,
                                           fetch_result_->col_buffers,
                                           outer_tab_frag_ids,
                                           query_exe_context,
                                           fetch_result_->num_rows,
                                           fetch_result_->frag_offsets,
                                           &catalog->getDataMgr(),
                                           kernel_.chosen_device_id,
                                           outer_table_id,
                                           kernel_.ra_exe_unit_.scan_limit,
                                           start_rowid_,
                                           kernel_.ra_exe_unit_.input_descs.size(),
                                           kernel_.eo.allow_runtime_query_interrupt,
                                           do_render ? kernel_.render_info_ : nullptr,
                                           start_rowid_ + num_rows_to_process_);
  }

  if (err) {
    throw QueryExecutionError(err);
  }
}

#endif  // HAVE_TBB
