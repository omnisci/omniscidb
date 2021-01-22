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
 * @file    ResultSet.cpp
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Basic constructors and methods of the row set interface.
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 */

#include "ResultSet.h"
#include "DataMgr/Allocators/CudaAllocator.h"
#include "DataMgr/BufferMgr/BufferMgr.h"
#include "Execute.h"
#include "GpuMemUtils.h"
#include "InPlaceSort.h"
#include "OutputBufferInitialization.h"
#include "RuntimeFunctions.h"
#include "Shared/SqlTypesLayout.h"
#include "Shared/checked_alloc.h"
#include "Shared/likely.h"
#include "Shared/thread_count.h"
#include "Shared/threadpool.h"

#include <algorithm>
#include <bitset>
#include <future>
#include <numeric>
#include "Utils/Threading.h"

extern bool g_use_tbb_pool;

void ResultSet::keepFirstN(const size_t n) {
  CHECK_EQ(-1, cached_row_count_);
  keep_first_ = n;
}

void ResultSet::dropFirstN(const size_t n) {
  CHECK_EQ(-1, cached_row_count_);
  drop_first_ = n;
}

ResultSet::ResultSet(const std::vector<TargetInfo>& targets,
                     const ExecutorDeviceType device_type,
                     const QueryMemoryDescriptor& query_mem_desc,
                     const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                     const Catalog_Namespace::Catalog* catalog,
                     const unsigned block_size,
                     const unsigned grid_size)
    : targets_(targets)
    , device_type_(device_type)
    , device_id_(-1)
    , query_mem_desc_(query_mem_desc)
    , crt_row_buff_idx_(0)
    , fetched_so_far_(0)
    , drop_first_(0)
    , keep_first_(0)
    , row_set_mem_owner_(row_set_mem_owner)
    , catalog_(catalog)
    , block_size_(block_size)
    , grid_size_(grid_size)
    , data_mgr_(nullptr)
    , separate_varlen_storage_valid_(false)
    , just_explain_(false)
    , for_validation_only_(false)
    , cached_row_count_(-1)
    , geo_return_type_(GeoReturnType::WktString) {}

ResultSet::ResultSet(const std::vector<TargetInfo>& targets,
                     const std::vector<ColumnLazyFetchInfo>& lazy_fetch_info,
                     const std::vector<std::vector<const int8_t*>>& col_buffers,
                     const std::vector<std::vector<int64_t>>& frag_offsets,
                     const std::vector<int64_t>& consistent_frag_sizes,
                     const ExecutorDeviceType device_type,
                     const int device_id,
                     const QueryMemoryDescriptor& query_mem_desc,
                     const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                     const Catalog_Namespace::Catalog* catalog,
                     const unsigned block_size,
                     const unsigned grid_size)
    : targets_(targets)
    , device_type_(device_type)
    , device_id_(device_id)
    , query_mem_desc_(query_mem_desc)
    , crt_row_buff_idx_(0)
    , fetched_so_far_(0)
    , drop_first_(0)
    , keep_first_(0)
    , row_set_mem_owner_(row_set_mem_owner)
    , catalog_(catalog)
    , block_size_(block_size)
    , grid_size_(grid_size)
    , lazy_fetch_info_(lazy_fetch_info)
    , col_buffers_{col_buffers}
    , frag_offsets_{frag_offsets}
    , consistent_frag_sizes_{consistent_frag_sizes}
    , data_mgr_(nullptr)
    , separate_varlen_storage_valid_(false)
    , just_explain_(false)
    , for_validation_only_(false)
    , cached_row_count_(-1)
    , geo_return_type_(GeoReturnType::WktString) {}

ResultSet::ResultSet(const std::shared_ptr<const Analyzer::Estimator> estimator,
                     const ExecutorDeviceType device_type,
                     const int device_id,
                     Data_Namespace::DataMgr* data_mgr)
    : device_type_(device_type)
    , device_id_(device_id)
    , query_mem_desc_{}
    , crt_row_buff_idx_(0)
    , estimator_(estimator)
    , data_mgr_(data_mgr)
    , separate_varlen_storage_valid_(false)
    , just_explain_(false)
    , for_validation_only_(false)
    , cached_row_count_(-1)
    , geo_return_type_(GeoReturnType::WktString) {
  if (device_type == ExecutorDeviceType::GPU) {
    device_estimator_buffer_ = CudaAllocator::allocGpuAbstractBuffer(
        data_mgr_, estimator_->getBufferSize(), device_id_);
    data_mgr->getCudaMgr()->zeroDeviceMem(device_estimator_buffer_->getMemoryPtr(),
                                          estimator_->getBufferSize(),
                                          device_id_);
  } else {
    host_estimator_buffer_ =
        static_cast<int8_t*>(checked_calloc(estimator_->getBufferSize(), 1));
  }
}

ResultSet::ResultSet(const std::string& explanation)
    : device_type_(ExecutorDeviceType::CPU)
    , device_id_(-1)
    , fetched_so_far_(0)
    , separate_varlen_storage_valid_(false)
    , explanation_(explanation)
    , just_explain_(true)
    , for_validation_only_(false)
    , cached_row_count_(-1)
    , geo_return_type_(GeoReturnType::WktString) {}

ResultSet::ResultSet(int64_t queue_time_ms,
                     int64_t render_time_ms,
                     const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner)
    : device_type_(ExecutorDeviceType::CPU)
    , device_id_(-1)
    , fetched_so_far_(0)
    , row_set_mem_owner_(row_set_mem_owner)
    , timings_(QueryExecutionTimings{queue_time_ms, render_time_ms, 0, 0})
    , separate_varlen_storage_valid_(false)
    , just_explain_(true)
    , for_validation_only_(false)
    , cached_row_count_(-1)
    , geo_return_type_(GeoReturnType::WktString){};

ResultSet::~ResultSet() {
  if (storage_) {
    if (!storage_->buff_is_provided_) {
      CHECK(storage_->getUnderlyingBuffer());
      free(storage_->getUnderlyingBuffer());
    }
  }
  for (auto& storage : appended_storage_) {
    if (storage && !storage->buff_is_provided_) {
      free(storage->getUnderlyingBuffer());
    }
  }
  if (host_estimator_buffer_) {
    CHECK(device_type_ == ExecutorDeviceType::CPU || device_estimator_buffer_);
    free(host_estimator_buffer_);
  }
  if (device_estimator_buffer_) {
    CHECK(data_mgr_);
    data_mgr_->free(device_estimator_buffer_);
  }
}

ExecutorDeviceType ResultSet::getDeviceType() const {
  return device_type_;
}

const ResultSetStorage* ResultSet::allocateStorage() const {
  CHECK(!storage_);
  CHECK(row_set_mem_owner_);
  auto buff =
      row_set_mem_owner_->allocate(query_mem_desc_.getBufferSizeBytes(device_type_));
  storage_.reset(
      new ResultSetStorage(targets_, query_mem_desc_, buff, /*buff_is_provided=*/true));
  return storage_.get();
}

const ResultSetStorage* ResultSet::allocateStorage(
    int8_t* buff,
    const std::vector<int64_t>& target_init_vals) const {
  CHECK(buff);
  CHECK(!storage_);
  storage_.reset(new ResultSetStorage(targets_, query_mem_desc_, buff, true));
  storage_->target_init_vals_ = target_init_vals;
  return storage_.get();
}

const ResultSetStorage* ResultSet::allocateStorage(
    const std::vector<int64_t>& target_init_vals) const {
  CHECK(!storage_);
  CHECK(row_set_mem_owner_);
  auto buff =
      row_set_mem_owner_->allocate(query_mem_desc_.getBufferSizeBytes(device_type_));
  storage_.reset(
      new ResultSetStorage(targets_, query_mem_desc_, buff, /*buff_is_provided=*/true));
  storage_->target_init_vals_ = target_init_vals;
  return storage_.get();
}

size_t ResultSet::getCurrentRowBufferIndex() const {
  if (crt_row_buff_idx_ == 0) {
    throw std::runtime_error("current row buffer iteration index is undefined");
  }
  return crt_row_buff_idx_ - 1;
}

// Note: that.appended_storage_ does not get appended to this.
void ResultSet::append(ResultSet& that) {
  CHECK_EQ(-1, cached_row_count_);
  if (!that.storage_) {
    return;
  }
  appended_storage_.push_back(std::move(that.storage_));
  query_mem_desc_.setEntryCount(
      query_mem_desc_.getEntryCount() +
      appended_storage_.back()->query_mem_desc_.getEntryCount());
  chunks_.insert(chunks_.end(), that.chunks_.begin(), that.chunks_.end());
  col_buffers_.insert(
      col_buffers_.end(), that.col_buffers_.begin(), that.col_buffers_.end());
  frag_offsets_.insert(
      frag_offsets_.end(), that.frag_offsets_.begin(), that.frag_offsets_.end());
  consistent_frag_sizes_.insert(consistent_frag_sizes_.end(),
                                that.consistent_frag_sizes_.begin(),
                                that.consistent_frag_sizes_.end());
  chunk_iters_.insert(
      chunk_iters_.end(), that.chunk_iters_.begin(), that.chunk_iters_.end());
  if (separate_varlen_storage_valid_) {
    CHECK(that.separate_varlen_storage_valid_);
    serialized_varlen_buffer_.insert(serialized_varlen_buffer_.end(),
                                     that.serialized_varlen_buffer_.begin(),
                                     that.serialized_varlen_buffer_.end());
  }
  for (auto& buff : that.literal_buffers_) {
    literal_buffers_.push_back(std::move(buff));
  }
}

const ResultSetStorage* ResultSet::getStorage() const {
  return storage_.get();
}

size_t ResultSet::colCount() const {
  return just_explain_ ? 1 : targets_.size();
}

SQLTypeInfo ResultSet::getColType(const size_t col_idx) const {
  if (just_explain_) {
    return SQLTypeInfo(kTEXT, false);
  }
  CHECK_LT(col_idx, targets_.size());
  return targets_[col_idx].agg_kind == kAVG ? SQLTypeInfo(kDOUBLE, false)
                                            : targets_[col_idx].sql_type;
}

namespace {

size_t get_truncated_row_count(size_t total_row_count, size_t limit, size_t offset) {
  if (total_row_count < offset) {
    return 0;
  }

  size_t total_truncated_row_count = total_row_count - offset;

  if (limit) {
    return std::min(total_truncated_row_count, limit);
  }

  return total_truncated_row_count;
}

}  // namespace

size_t ResultSet::rowCount(const bool force_parallel) const {
  if (just_explain_) {
    return 1;
  }
  if (!permutation_.empty()) {
    if (drop_first_ > permutation_.size()) {
      return 0;
    }
    const auto limited_row_count = keep_first_ + drop_first_;
    return limited_row_count ? std::min(limited_row_count, permutation_.size())
                             : permutation_.size();
  }
  if (cached_row_count_ != -1) {
    CHECK_GE(cached_row_count_, 0);
    return cached_row_count_;
  }
  if (!storage_) {
    return 0;
  }
  if (permutation_.empty() &&
      query_mem_desc_.getQueryDescriptionType() == QueryDescriptionType::Projection) {
    return binSearchRowCount();
  }
  if (force_parallel || entryCount() > 20000) {
    return parallelRowCount();
  }
  std::lock_guard<std::mutex> lock(row_iteration_mutex_);
  moveToBegin();
  size_t row_count{0};
  while (true) {
    auto crt_row = getNextRowUnlocked(false, false);
    if (crt_row.empty()) {
      break;
    }
    ++row_count;
  }
  moveToBegin();
  return row_count;
}

void ResultSet::setCachedRowCount(const size_t row_count) const {
  CHECK(cached_row_count_ == -1 || cached_row_count_ == static_cast<int64_t>(row_count));
  cached_row_count_ = row_count;
}

size_t ResultSet::binSearchRowCount() const {
  if (!storage_) {
    return 0;
  }

  size_t row_count = storage_->binSearchRowCount();
  for (auto& s : appended_storage_) {
    row_count += s->binSearchRowCount();
  }

  return get_truncated_row_count(row_count, getLimit(), drop_first_);
}

size_t ResultSet::parallelRowCount() const {
  auto execute_parallel_row_count = [this](auto counter_threads) -> size_t {
    const size_t worker_count = cpu_threads();
    for (size_t i = 0,
                start_entry = 0,
                stride = (entryCount() + worker_count - 1) / worker_count;
         i < worker_count && start_entry < entryCount();
         ++i, start_entry += stride) {
      const auto end_entry = std::min(start_entry + stride, entryCount());
      counter_threads.spawn(
          [this](const size_t start, const size_t end) {
            size_t row_count{0};
            for (size_t i = start; i < end; ++i) {
              if (!isRowAtEmpty(i)) {
                ++row_count;
              }
            }
            return row_count;
          },
          start_entry,
          end_entry);
    }
    const auto row_counts = counter_threads.join();
    const size_t row_count = std::accumulate(row_counts.begin(), row_counts.end(), 0);
    return row_count;
  };
  // will fall back to futures threadpool if TBB is not enabled
  const auto row_count =
      g_use_tbb_pool
          ? execute_parallel_row_count(threadpool::ThreadPool<size_t>())
          : execute_parallel_row_count(threadpool::FuturesThreadPool<size_t>());

  return get_truncated_row_count(row_count, getLimit(), drop_first_);
}

bool ResultSet::definitelyHasNoRows() const {
  return !storage_ && !estimator_ && !just_explain_;
}

const QueryMemoryDescriptor& ResultSet::getQueryMemDesc() const {
  CHECK(storage_);
  return storage_->query_mem_desc_;
}

const std::vector<TargetInfo>& ResultSet::getTargetInfos() const {
  return targets_;
}

const std::vector<int64_t>& ResultSet::getTargetInitVals() const {
  CHECK(storage_);
  return storage_->target_init_vals_;
}

int8_t* ResultSet::getDeviceEstimatorBuffer() const {
  CHECK(device_type_ == ExecutorDeviceType::GPU);
  CHECK(device_estimator_buffer_);
  return device_estimator_buffer_->getMemoryPtr();
}

int8_t* ResultSet::getHostEstimatorBuffer() const {
  return host_estimator_buffer_;
}

void ResultSet::syncEstimatorBuffer() const {
  CHECK(device_type_ == ExecutorDeviceType::GPU);
  CHECK(!host_estimator_buffer_);
  CHECK_EQ(size_t(0), estimator_->getBufferSize() % sizeof(int64_t));
  host_estimator_buffer_ =
      static_cast<int8_t*>(checked_calloc(estimator_->getBufferSize(), 1));
  CHECK(device_estimator_buffer_);
  auto device_buffer_ptr = device_estimator_buffer_->getMemoryPtr();
  copy_from_gpu(data_mgr_,
                host_estimator_buffer_,
                reinterpret_cast<CUdeviceptr>(device_buffer_ptr),
                estimator_->getBufferSize(),
                device_id_);
}

void ResultSet::setQueueTime(const int64_t queue_time) {
  timings_.executor_queue_time = queue_time;
}

void ResultSet::setKernelQueueTime(const int64_t kernel_queue_time) {
  timings_.kernel_queue_time = kernel_queue_time;
}

void ResultSet::addCompilationQueueTime(const int64_t compilation_queue_time) {
  timings_.compilation_queue_time += compilation_queue_time;
}

int64_t ResultSet::getQueueTime() const {
  return timings_.executor_queue_time + timings_.kernel_queue_time +
         timings_.compilation_queue_time;
}

int64_t ResultSet::getRenderTime() const {
  return timings_.render_time;
}

void ResultSet::moveToBegin() const {
  crt_row_buff_idx_ = 0;
  fetched_so_far_ = 0;
}

bool ResultSet::isTruncated() const {
  return keep_first_ + drop_first_;
}

bool ResultSet::isExplain() const {
  return just_explain_;
}

void ResultSet::setValidationOnlyRes() {
  for_validation_only_ = true;
}

bool ResultSet::isValidationOnlyRes() const {
  return for_validation_only_;
}

int ResultSet::getDeviceId() const {
  return device_id_;
}

QueryMemoryDescriptor ResultSet::fixupQueryMemoryDescriptor(
    const QueryMemoryDescriptor& query_mem_desc) {
  auto query_mem_desc_copy = query_mem_desc;
  query_mem_desc_copy.resetGroupColWidths(
      std::vector<int8_t>(query_mem_desc_copy.getGroupbyColCount(), 8));
  if (query_mem_desc.didOutputColumnar()) {
    return query_mem_desc_copy;
  }
  query_mem_desc_copy.alignPaddedSlots();
  return query_mem_desc_copy;
}

void ResultSet::sort(const std::list<Analyzer::OrderEntry>& order_entries,
                     const size_t top_n,
                     const Executor* executor) {
  auto timer = DEBUG_TIMER(__func__);

  if (!storage_) {
    return;
  }
  CHECK_EQ(-1, cached_row_count_);
  CHECK(!targets_.empty());
#ifdef HAVE_CUDA
  if (canUseFastBaselineSort(order_entries, top_n)) {
    baselineSort(order_entries, top_n, executor);
    return;
  }
#endif  // HAVE_CUDA
  if (query_mem_desc_.sortOnGpu()) {
    try {
      radixSortOnGpu(order_entries);
    } catch (const OutOfMemory&) {
      LOG(WARNING) << "Out of GPU memory during sort, finish on CPU";
      radixSortOnCpu(order_entries);
    } catch (const std::bad_alloc&) {
      LOG(WARNING) << "Out of GPU memory during sort, finish on CPU";
      radixSortOnCpu(order_entries);
    }
    return;
  }
  // This check isn't strictly required, but allows the index buffer to be 32-bit.
  if (query_mem_desc_.getEntryCount() > std::numeric_limits<uint32_t>::max()) {
    throw RowSortException("Sorting more than 4B elements not supported");
  }

  CHECK(permutation_.empty());

  const bool use_heap{order_entries.size() == 1 && top_n};
  if (use_heap && entryCount() > 100000) {
    if (g_enable_watchdog && (entryCount() > 20000000)) {
      throw WatchdogException("Sorting the result would be too slow");
    }
    parallelTop(order_entries, top_n, executor);
    return;
  }

  if (g_enable_watchdog && (entryCount() > Executor::baseline_threshold)) {
    throw WatchdogException("Sorting the result would be too slow");
  }

  permutation_ = initPermutationBuffer(0, 1);

  auto compare = createComparator(order_entries, use_heap, executor);

  if (use_heap) {
    topPermutation(permutation_, top_n, compare);
  } else {
    sortPermutation(compare);
  }
}

#ifdef HAVE_CUDA
void ResultSet::baselineSort(const std::list<Analyzer::OrderEntry>& order_entries,
                             const size_t top_n,
                             const Executor* executor) {
  auto timer = DEBUG_TIMER(__func__);
  // If we only have on GPU, it's usually faster to do multi-threaded radix sort on CPU
  if (getGpuCount() > 1) {
    try {
      doBaselineSort(ExecutorDeviceType::GPU, order_entries, top_n, executor);
    } catch (...) {
      doBaselineSort(ExecutorDeviceType::CPU, order_entries, top_n, executor);
    }
  } else {
    doBaselineSort(ExecutorDeviceType::CPU, order_entries, top_n, executor);
  }
}
#endif  // HAVE_CUDA

std::vector<uint32_t> ResultSet::initPermutationBuffer(const size_t start,
                                                       const size_t step) {
  auto timer = DEBUG_TIMER(__func__);
  CHECK_NE(size_t(0), step);
  std::vector<uint32_t> permutation;
  const auto total_entries = query_mem_desc_.getEntryCount();
  permutation.reserve(total_entries / step);
  for (size_t i = start; i < total_entries; i += step) {
    const auto storage_lookup_result = findStorage(i);
    const auto lhs_storage = storage_lookup_result.storage_ptr;
    const auto off = storage_lookup_result.fixedup_entry_idx;
    CHECK(lhs_storage);
    if (!lhs_storage->isEmptyEntry(off)) {
      permutation.emplace_back(i);
    }
  }
  return permutation;
}

const std::vector<uint32_t>& ResultSet::getPermutationBuffer() const {
  return permutation_;
}

void ResultSet::parallelTop(const std::list<Analyzer::OrderEntry>& order_entries,
                            const size_t top_n,
                            const Executor* executor) {
  auto timer = DEBUG_TIMER(__func__);
  const size_t step = cpu_threads();
  std::vector<std::vector<uint32_t>> strided_permutations(step);
  std::vector<utils::future<void>> init_futures;
  for (size_t start = 0; start < step; ++start) {
    init_futures.emplace_back(utils::async([this, start, step, &strided_permutations] {
      strided_permutations[start] = initPermutationBuffer(start, step);
    }));
  }
  for (auto& init_future : init_futures) {
    init_future.wait();
  }
  auto compare = createComparator(order_entries, true, executor);
  std::vector<utils::future<void>> top_futures;
  for (auto& strided_permutation : strided_permutations) {
    top_futures.emplace_back(utils::async([&strided_permutation, &compare, top_n] {
      topPermutation(strided_permutation, top_n, compare);
    }));
  }
  for (auto& top_future : top_futures) {
    top_future.wait();
  }
  permutation_.reserve(strided_permutations.size() * top_n);
  for (const auto& strided_permutation : strided_permutations) {
    permutation_.insert(
        permutation_.end(), strided_permutation.begin(), strided_permutation.end());
  }
  topPermutation(permutation_, top_n, compare);
}

std::pair<size_t, size_t> ResultSet::getStorageIndex(const size_t entry_idx) const {
  size_t fixedup_entry_idx = entry_idx;
  auto entry_count = storage_->query_mem_desc_.getEntryCount();
  const bool is_rowwise_layout = !storage_->query_mem_desc_.didOutputColumnar();
  if (fixedup_entry_idx < entry_count) {
    return {0, fixedup_entry_idx};
  }
  fixedup_entry_idx -= entry_count;
  for (size_t i = 0; i < appended_storage_.size(); ++i) {
    const auto& desc = appended_storage_[i]->query_mem_desc_;
    CHECK_NE(is_rowwise_layout, desc.didOutputColumnar());
    entry_count = desc.getEntryCount();
    if (fixedup_entry_idx < entry_count) {
      return {i + 1, fixedup_entry_idx};
    }
    fixedup_entry_idx -= entry_count;
  }
  UNREACHABLE() << "entry_idx = " << entry_idx << ", query_mem_desc_.getEntryCount() = "
                << query_mem_desc_.getEntryCount();
  return {};
}

template struct ResultSet::ResultSetComparator<ResultSet::RowWiseTargetAccessor>;
template struct ResultSet::ResultSetComparator<ResultSet::ColumnWiseTargetAccessor>;

ResultSet::StorageLookupResult ResultSet::findStorage(const size_t entry_idx) const {
  auto [stg_idx, fixedup_entry_idx] = getStorageIndex(entry_idx);
  return {stg_idx ? appended_storage_[stg_idx - 1].get() : storage_.get(),
          fixedup_entry_idx,
          stg_idx};
}

template <typename BUFFER_ITERATOR_TYPE>
void ResultSet::ResultSetComparator<
    BUFFER_ITERATOR_TYPE>::materializeCountDistinctColumns() {
  for (const auto& order_entry : order_entries_) {
    if (is_distinct_target(result_set_->targets_[order_entry.tle_no - 1])) {
      count_distinct_materialized_buffers_.emplace_back(
          materializeCountDistinctColumn(order_entry));
    }
  }
}

template <typename BUFFER_ITERATOR_TYPE>
std::vector<int64_t>
ResultSet::ResultSetComparator<BUFFER_ITERATOR_TYPE>::materializeCountDistinctColumn(
    const Analyzer::OrderEntry& order_entry) const {
  const size_t num_storage_entries = result_set_->query_mem_desc_.getEntryCount();
  std::vector<int64_t> count_distinct_materialized_buffer(num_storage_entries);
  const CountDistinctDescriptor count_distinct_descriptor =
      result_set_->query_mem_desc_.getCountDistinctDescriptor(order_entry.tle_no - 1);
  const size_t num_non_empty_entries = result_set_->permutation_.size();
  const size_t worker_count = cpu_threads();
  // TODO(tlm): Allow use of tbb after we determine how to easily encapsulate the choice
  // between thread pool types
  threadpool::FuturesThreadPool<void> thread_pool;
  for (size_t i = 0,
              start_entry = 0,
              stride = (num_non_empty_entries + worker_count - 1) / worker_count;
       i < worker_count && start_entry < num_non_empty_entries;
       ++i, start_entry += stride) {
    const auto end_entry = std::min(start_entry + stride, num_non_empty_entries);
    thread_pool.spawn(
        [this](const size_t start,
               const size_t end,
               const Analyzer::OrderEntry& order_entry,
               const CountDistinctDescriptor& count_distinct_descriptor,
               std::vector<int64_t>& count_distinct_materialized_buffer) {
          for (size_t i = start; i < end; ++i) {
            const uint32_t permuted_idx = result_set_->permutation_[i];
            const auto storage_lookup_result = result_set_->findStorage(permuted_idx);
            const auto storage = storage_lookup_result.storage_ptr;
            const auto off = storage_lookup_result.fixedup_entry_idx;
            const auto value = buffer_itr_.getColumnInternal(
                storage->buff_, off, order_entry.tle_no - 1, storage_lookup_result);
            count_distinct_materialized_buffer[permuted_idx] =
                count_distinct_set_size(value.i1, count_distinct_descriptor);
          }
        },
        start_entry,
        end_entry,
        std::cref(order_entry),
        std::cref(count_distinct_descriptor),
        std::ref(count_distinct_materialized_buffer));
  }
  thread_pool.join();
  return count_distinct_materialized_buffer;
}

template <typename BUFFER_ITERATOR_TYPE>
bool ResultSet::ResultSetComparator<BUFFER_ITERATOR_TYPE>::operator()(
    const uint32_t lhs,
    const uint32_t rhs) const {
  // NB: The compare function must define a strict weak ordering, otherwise
  // std::sort will trigger a segmentation fault (or corrupt memory).
  const auto lhs_storage_lookup_result = result_set_->findStorage(lhs);
  const auto rhs_storage_lookup_result = result_set_->findStorage(rhs);
  const auto lhs_storage = lhs_storage_lookup_result.storage_ptr;
  const auto rhs_storage = rhs_storage_lookup_result.storage_ptr;
  const auto fixedup_lhs = lhs_storage_lookup_result.fixedup_entry_idx;
  const auto fixedup_rhs = rhs_storage_lookup_result.fixedup_entry_idx;
  size_t materialized_count_distinct_buffer_idx{0};

  for (const auto& order_entry : order_entries_) {
    CHECK_GE(order_entry.tle_no, 1);
    const auto& agg_info = result_set_->targets_[order_entry.tle_no - 1];
    const auto entry_ti = get_compact_type(agg_info);
    bool float_argument_input = takes_float_argument(agg_info);
    // Need to determine if the float value has been stored as float
    // or if it has been compacted to a different (often larger 8 bytes)
    // in distributed case the floats are actually 4 bytes
    // TODO the above takes_float_argument() is widely used wonder if this problem
    // exists elsewhere
    if (entry_ti.get_type() == kFLOAT) {
      const auto is_col_lazy =
          !result_set_->lazy_fetch_info_.empty() &&
          result_set_->lazy_fetch_info_[order_entry.tle_no - 1].is_lazily_fetched;
      if (result_set_->query_mem_desc_.getPaddedSlotWidthBytes(order_entry.tle_no - 1) ==
          sizeof(float)) {
        float_argument_input =
            result_set_->query_mem_desc_.didOutputColumnar() ? !is_col_lazy : true;
      }
    }

    const bool use_desc_cmp = use_heap_ ? !order_entry.is_desc : order_entry.is_desc;

    if (UNLIKELY(is_distinct_target(agg_info))) {
      CHECK_LT(materialized_count_distinct_buffer_idx,
               count_distinct_materialized_buffers_.size());
      const auto& count_distinct_materialized_buffer =
          count_distinct_materialized_buffers_[materialized_count_distinct_buffer_idx];
      const auto lhs_sz = count_distinct_materialized_buffer[lhs];
      const auto rhs_sz = count_distinct_materialized_buffer[rhs];
      ++materialized_count_distinct_buffer_idx;
      if (lhs_sz == rhs_sz) {
        continue;
      }
      return use_desc_cmp ? lhs_sz > rhs_sz : lhs_sz < rhs_sz;
    }

    const auto lhs_v = buffer_itr_.getColumnInternal(lhs_storage->buff_,
                                                     fixedup_lhs,
                                                     order_entry.tle_no - 1,
                                                     lhs_storage_lookup_result);
    const auto rhs_v = buffer_itr_.getColumnInternal(rhs_storage->buff_,
                                                     fixedup_rhs,
                                                     order_entry.tle_no - 1,
                                                     rhs_storage_lookup_result);

    if (UNLIKELY(isNull(entry_ti, lhs_v, float_argument_input) &&
                 isNull(entry_ti, rhs_v, float_argument_input))) {
      return false;
    }
    if (UNLIKELY(isNull(entry_ti, lhs_v, float_argument_input) &&
                 !isNull(entry_ti, rhs_v, float_argument_input))) {
      return use_heap_ ? !order_entry.nulls_first : order_entry.nulls_first;
    }
    if (UNLIKELY(isNull(entry_ti, rhs_v, float_argument_input) &&
                 !isNull(entry_ti, lhs_v, float_argument_input))) {
      return use_heap_ ? order_entry.nulls_first : !order_entry.nulls_first;
    }

    if (LIKELY(lhs_v.isInt())) {
      CHECK(rhs_v.isInt());
      if (UNLIKELY(entry_ti.is_string() &&
                   entry_ti.get_compression() == kENCODING_DICT)) {
        CHECK_EQ(4, entry_ti.get_logical_size());
        CHECK(executor_);
        const auto string_dict_proxy = executor_->getStringDictionaryProxy(
            entry_ti.get_comp_param(), result_set_->row_set_mem_owner_, false);
        auto lhs_str = string_dict_proxy->getString(lhs_v.i1);
        auto rhs_str = string_dict_proxy->getString(rhs_v.i1);
        if (lhs_str == rhs_str) {
          continue;
        }
        return use_desc_cmp ? lhs_str > rhs_str : lhs_str < rhs_str;
      }

      if (lhs_v.i1 == rhs_v.i1) {
        continue;
      }
      if (entry_ti.is_fp()) {
        if (float_argument_input) {
          const auto lhs_dval = *reinterpret_cast<const float*>(may_alias_ptr(&lhs_v.i1));
          const auto rhs_dval = *reinterpret_cast<const float*>(may_alias_ptr(&rhs_v.i1));
          return use_desc_cmp ? lhs_dval > rhs_dval : lhs_dval < rhs_dval;
        } else {
          const auto lhs_dval =
              *reinterpret_cast<const double*>(may_alias_ptr(&lhs_v.i1));
          const auto rhs_dval =
              *reinterpret_cast<const double*>(may_alias_ptr(&rhs_v.i1));
          return use_desc_cmp ? lhs_dval > rhs_dval : lhs_dval < rhs_dval;
        }
      }
      return use_desc_cmp ? lhs_v.i1 > rhs_v.i1 : lhs_v.i1 < rhs_v.i1;
    } else {
      if (lhs_v.isPair()) {
        CHECK(rhs_v.isPair());
        const auto lhs =
            pair_to_double({lhs_v.i1, lhs_v.i2}, entry_ti, float_argument_input);
        const auto rhs =
            pair_to_double({rhs_v.i1, rhs_v.i2}, entry_ti, float_argument_input);
        if (lhs == rhs) {
          continue;
        }
        return use_desc_cmp ? lhs > rhs : lhs < rhs;
      } else {
        CHECK(lhs_v.isStr() && rhs_v.isStr());
        const auto lhs = lhs_v.strVal();
        const auto rhs = rhs_v.strVal();
        if (lhs == rhs) {
          continue;
        }
        return use_desc_cmp ? lhs > rhs : lhs < rhs;
      }
    }
  }
  return false;
}

void ResultSet::topPermutation(
    std::vector<uint32_t>& to_sort,
    const size_t n,
    const std::function<bool(const uint32_t, const uint32_t)> compare) {
  auto timer = DEBUG_TIMER(__func__);
  std::make_heap(to_sort.begin(), to_sort.end(), compare);
  std::vector<uint32_t> permutation_top;
  permutation_top.reserve(n);
  for (size_t i = 0; i < n && !to_sort.empty(); ++i) {
    permutation_top.push_back(to_sort.front());
    std::pop_heap(to_sort.begin(), to_sort.end(), compare);
    to_sort.pop_back();
  }
  to_sort.swap(permutation_top);
}

void ResultSet::sortPermutation(
    const std::function<bool(const uint32_t, const uint32_t)> compare) {
  auto timer = DEBUG_TIMER(__func__);
  std::sort(permutation_.begin(), permutation_.end(), compare);
}

void ResultSet::radixSortOnGpu(
    const std::list<Analyzer::OrderEntry>& order_entries) const {
  auto timer = DEBUG_TIMER(__func__);
  auto data_mgr = &catalog_->getDataMgr();
  const int device_id{0};
  CudaAllocator cuda_allocator(data_mgr, device_id);
  CHECK_GT(block_size_, 0);
  CHECK_GT(grid_size_, 0);
  std::vector<int64_t*> group_by_buffers(block_size_);
  group_by_buffers[0] = reinterpret_cast<int64_t*>(storage_->getUnderlyingBuffer());
  auto dev_group_by_buffers =
      create_dev_group_by_buffers(&cuda_allocator,
                                  group_by_buffers,
                                  query_mem_desc_,
                                  block_size_,
                                  grid_size_,
                                  device_id,
                                  ExecutorDispatchMode::KernelPerFragment,
                                  -1,
                                  true,
                                  true,
                                  false,
                                  nullptr);
  inplace_sort_gpu(
      order_entries, query_mem_desc_, dev_group_by_buffers, data_mgr, device_id);
  copy_group_by_buffers_from_gpu(
      data_mgr,
      group_by_buffers,
      query_mem_desc_.getBufferSizeBytes(ExecutorDeviceType::GPU),
      dev_group_by_buffers.second,
      query_mem_desc_,
      block_size_,
      grid_size_,
      device_id,
      false);
}

void ResultSet::radixSortOnCpu(
    const std::list<Analyzer::OrderEntry>& order_entries) const {
  auto timer = DEBUG_TIMER(__func__);
  CHECK(!query_mem_desc_.hasKeylessHash());
  std::vector<int64_t> tmp_buff(query_mem_desc_.getEntryCount());
  std::vector<int32_t> idx_buff(query_mem_desc_.getEntryCount());
  CHECK_EQ(size_t(1), order_entries.size());
  auto buffer_ptr = storage_->getUnderlyingBuffer();
  for (const auto& order_entry : order_entries) {
    const auto target_idx = order_entry.tle_no - 1;
    const auto sortkey_val_buff = reinterpret_cast<int64_t*>(
        buffer_ptr + query_mem_desc_.getColOffInBytes(target_idx));
    const auto chosen_bytes = query_mem_desc_.getPaddedSlotWidthBytes(target_idx);
    sort_groups_cpu(sortkey_val_buff,
                    &idx_buff[0],
                    query_mem_desc_.getEntryCount(),
                    order_entry.is_desc,
                    chosen_bytes);
    apply_permutation_cpu(reinterpret_cast<int64_t*>(buffer_ptr),
                          &idx_buff[0],
                          query_mem_desc_.getEntryCount(),
                          &tmp_buff[0],
                          sizeof(int64_t));
    for (size_t target_idx = 0; target_idx < query_mem_desc_.getSlotCount();
         ++target_idx) {
      if (static_cast<int>(target_idx) == order_entry.tle_no - 1) {
        continue;
      }
      const auto chosen_bytes = query_mem_desc_.getPaddedSlotWidthBytes(target_idx);
      const auto satellite_val_buff = reinterpret_cast<int64_t*>(
          buffer_ptr + query_mem_desc_.getColOffInBytes(target_idx));
      apply_permutation_cpu(satellite_val_buff,
                            &idx_buff[0],
                            query_mem_desc_.getEntryCount(),
                            &tmp_buff[0],
                            chosen_bytes);
    }
  }
}

size_t ResultSet::getLimit() const {
  return keep_first_;
}

std::shared_ptr<const std::vector<std::string>> ResultSet::getStringDictionaryPayloadCopy(
    const int dict_id) const {
  const auto sdp = row_set_mem_owner_->getOrAddStringDictProxy(
      dict_id, /*with_generation=*/false, catalog_);
  CHECK(sdp);
  return sdp->getDictionary()->copyStrings();
}

/**
 * Determines if it is possible to directly form a ColumnarResults class from this
 * result set, bypassing the default columnarization.
 *
 * NOTE: If there exists a permutation vector (i.e., in some ORDER BY queries), it
 * becomes equivalent to the row-wise columnarization.
 */
bool ResultSet::isDirectColumnarConversionPossible() const {
  if (!g_enable_direct_columnarization) {
    return false;
  } else if (query_mem_desc_.didOutputColumnar()) {
    return permutation_.empty() && (query_mem_desc_.getQueryDescriptionType() ==
                                        QueryDescriptionType::Projection ||
                                    (query_mem_desc_.getQueryDescriptionType() ==
                                         QueryDescriptionType::GroupByPerfectHash ||
                                     query_mem_desc_.getQueryDescriptionType() ==
                                         QueryDescriptionType::GroupByBaselineHash));
  } else {
    return permutation_.empty() && (query_mem_desc_.getQueryDescriptionType() ==
                                        QueryDescriptionType::GroupByPerfectHash ||
                                    query_mem_desc_.getQueryDescriptionType() ==
                                        QueryDescriptionType::GroupByBaselineHash);
  }
}

bool ResultSet::isZeroCopyColumnarConversionPossible(size_t column_idx) const {
  return query_mem_desc_.didOutputColumnar() &&
         query_mem_desc_.getQueryDescriptionType() == QueryDescriptionType::Projection &&
         appended_storage_.empty() && storage_ &&
         (lazy_fetch_info_.empty() || !lazy_fetch_info_[column_idx].is_lazily_fetched);
}

const int8_t* ResultSet::getColumnarBuffer(size_t column_idx) const {
  CHECK(isZeroCopyColumnarConversionPossible(column_idx));
  return storage_->getUnderlyingBuffer() + query_mem_desc_.getColOffInBytes(column_idx);
}

// returns a bitmap (and total number) of all single slot targets
std::tuple<std::vector<bool>, size_t> ResultSet::getSingleSlotTargetBitmap() const {
  std::vector<bool> target_bitmap(targets_.size(), true);
  size_t num_single_slot_targets = 0;
  for (size_t target_idx = 0; target_idx < targets_.size(); target_idx++) {
    const auto& sql_type = targets_[target_idx].sql_type;
    if (targets_[target_idx].is_agg && targets_[target_idx].agg_kind == kAVG) {
      target_bitmap[target_idx] = false;
    } else if (sql_type.is_varlen()) {
      target_bitmap[target_idx] = false;
    } else {
      num_single_slot_targets++;
    }
  }
  return std::make_tuple(std::move(target_bitmap), num_single_slot_targets);
}

/**
 * This function returns a bitmap and population count of it, where it denotes
 * all supported single-column targets suitable for direct columnarization.
 *
 * The final goal is to remove the need for such selection, but at the moment for any
 * target that doesn't qualify for direct columnarization, we use the traditional
 * result set's iteration to handle it (e.g., count distinct, approximate count distinct)
 */
std::tuple<std::vector<bool>, size_t> ResultSet::getSupportedSingleSlotTargetBitmap()
    const {
  CHECK(isDirectColumnarConversionPossible());
  auto [single_slot_targets, num_single_slot_targets] = getSingleSlotTargetBitmap();

  for (size_t target_idx = 0; target_idx < single_slot_targets.size(); target_idx++) {
    const auto& target = targets_[target_idx];
    if (single_slot_targets[target_idx] &&
        (is_distinct_target(target) ||
         (target.is_agg && target.agg_kind == kSAMPLE && target.sql_type == kFLOAT))) {
      single_slot_targets[target_idx] = false;
      num_single_slot_targets--;
    }
  }
  CHECK_GE(num_single_slot_targets, size_t(0));
  return std::make_tuple(std::move(single_slot_targets), num_single_slot_targets);
}

// returns the starting slot index for all targets in the result set
std::vector<size_t> ResultSet::getSlotIndicesForTargetIndices() const {
  std::vector<size_t> slot_indices(targets_.size(), 0);
  size_t slot_index = 0;
  for (size_t target_idx = 0; target_idx < targets_.size(); target_idx++) {
    slot_indices[target_idx] = slot_index;
    slot_index = advance_slot(slot_index, targets_[target_idx], false);
  }
  return slot_indices;
}

// namespace result_set

bool result_set::can_use_parallel_algorithms(const ResultSet& rows) {
  return !rows.isTruncated();
}

bool result_set::use_parallel_algorithms(const ResultSet& rows) {
  return result_set::can_use_parallel_algorithms(rows) && rows.entryCount() >= 20000;
}
