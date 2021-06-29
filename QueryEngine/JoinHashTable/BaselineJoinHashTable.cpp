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

#include "QueryEngine/JoinHashTable/BaselineJoinHashTable.h"

#include <future>

#include "DataMgr/Allocators/CudaAllocator.h"
#include "QueryEngine/CodeGenerator.h"
#include "QueryEngine/ColumnFetcher.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ExpressionRewrite.h"
#include "QueryEngine/JoinHashTable/BaselineHashTable.h"
#include "QueryEngine/JoinHashTable/Builders/BaselineHashTableBuilder.h"
#include "QueryEngine/JoinHashTable/PerfectJoinHashTable.h"
#include "QueryEngine/JoinHashTable/Runtime/HashJoinKeyHandlers.h"
#include "QueryEngine/JoinHashTable/Runtime/JoinHashTableGpuUtils.h"

std::unique_ptr<
    HashTableCache<HashTableCacheKey, BaselineJoinHashTable::HashTableCacheValue>>
    BaselineJoinHashTable::hash_table_cache_ = std::make_unique<
        HashTableCache<HashTableCacheKey, BaselineJoinHashTable::HashTableCacheValue>>();

//! Make hash table from an in-flight SQL query's parse tree etc.
std::shared_ptr<BaselineJoinHashTable> BaselineJoinHashTable::getInstance(
    const std::shared_ptr<Analyzer::BinOper> condition,
    const std::vector<InputTableInfo>& query_infos,
    const Data_Namespace::MemoryLevel memory_level,
    const JoinType join_type,
    const HashType preferred_hash_type,
    const int device_count,
    ColumnCacheMap& column_cache,
    Executor* executor) {
  decltype(std::chrono::steady_clock::now()) ts1, ts2;

  if (VLOGGING(1)) {
    VLOG(1) << "Building keyed hash table " << getHashTypeString(preferred_hash_type)
            << " for qual: " << condition->toString();
    ts1 = std::chrono::steady_clock::now();
  }
  auto inner_outer_pairs = normalize_column_pairs(
      condition.get(), *executor->getCatalog(), executor->getTemporaryTables());

  auto join_hash_table =
      std::shared_ptr<BaselineJoinHashTable>(new BaselineJoinHashTable(condition,
                                                                       join_type,
                                                                       query_infos,
                                                                       memory_level,
                                                                       column_cache,
                                                                       executor,
                                                                       inner_outer_pairs,
                                                                       device_count));
  try {
    join_hash_table->reify(preferred_hash_type);
  } catch (const TableMustBeReplicated& e) {
    // Throw a runtime error to abort the query
    join_hash_table->freeHashBufferMemory();
    throw std::runtime_error(e.what());
  } catch (const HashJoinFail& e) {
    // HashJoinFail exceptions log an error and trigger a retry with a join loop (if
    // possible)
    join_hash_table->freeHashBufferMemory();
    throw HashJoinFail(std::string("Could not build a 1-to-1 correspondence for columns "
                                   "involved in equijoin | ") +
                       e.what());
  } catch (const ColumnarConversionNotSupported& e) {
    throw HashJoinFail(std::string("Could not build hash tables for equijoin | ") +
                       e.what());
  } catch (const OutOfMemory& e) {
    throw HashJoinFail(
        std::string("Ran out of memory while building hash tables for equijoin | ") +
        e.what());
  } catch (const std::exception& e) {
    throw std::runtime_error(
        std::string("Fatal error while attempting to build hash tables for join: ") +
        e.what());
  }
  if (VLOGGING(1)) {
    ts2 = std::chrono::steady_clock::now();
    VLOG(1) << "Built keyed hash table "
            << getHashTypeString(join_hash_table->getHashType()) << " in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(ts2 - ts1).count()
            << " ms";
  }
  return join_hash_table;
}

BaselineJoinHashTable::BaselineJoinHashTable(
    const std::shared_ptr<Analyzer::BinOper> condition,
    const JoinType join_type,
    const std::vector<InputTableInfo>& query_infos,
    const Data_Namespace::MemoryLevel memory_level,
    ColumnCacheMap& column_cache,
    Executor* executor,
    const std::vector<InnerOuter>& inner_outer_pairs,
    const int device_count)
    : condition_(condition)
    , join_type_(join_type)
    , query_infos_(query_infos)
    , memory_level_(memory_level)
    , executor_(executor)
    , column_cache_(column_cache)
    , inner_outer_pairs_(inner_outer_pairs)
    , catalog_(executor->getCatalog())
    , device_count_(device_count) {
  CHECK_GT(device_count_, 0);
  hash_tables_for_device_.resize(std::max(device_count_, 1));
}

size_t BaselineJoinHashTable::getShardCountForCondition(
    const Analyzer::BinOper* condition,
    const Executor* executor,
    const std::vector<InnerOuter>& inner_outer_pairs) {
  for (const auto& inner_outer_pair : inner_outer_pairs) {
    const auto pair_shard_count = get_shard_count(inner_outer_pair, executor);
    if (pair_shard_count) {
      return pair_shard_count;
    }
  }
  return 0;
}

std::string BaselineJoinHashTable::toString(const ExecutorDeviceType device_type,
                                            const int device_id,
                                            bool raw) const {
  auto buffer = getJoinHashBuffer(device_type, device_id);
  CHECK_LT(device_id, hash_tables_for_device_.size());
  auto hash_table = hash_tables_for_device_[device_id];
  CHECK(hash_table);
  auto buffer_size = hash_table->getHashTableBufferSize(device_type);
#ifdef HAVE_CUDA
  std::unique_ptr<int8_t[]> buffer_copy;
  if (device_type == ExecutorDeviceType::GPU) {
    buffer_copy = std::make_unique<int8_t[]>(buffer_size);

    auto& data_mgr = catalog_->getDataMgr();
    auto device_allocator = data_mgr.createGpuAllocator(device_id);
    device_allocator->copyFromDevice(buffer_copy.get(), buffer, buffer_size);
  }
  auto ptr1 = buffer_copy ? buffer_copy.get() : buffer;
#else
  auto ptr1 = buffer;
#endif  // HAVE_CUDA
  auto ptr2 = ptr1 + offsetBufferOff();
  auto ptr3 = ptr1 + countBufferOff();
  auto ptr4 = ptr1 + payloadBufferOff();
  CHECK(hash_table);
  const auto layout = getHashType();
  return HashTable::toString(
      "keyed",
      getHashTypeString(layout),
      getKeyComponentCount() + (layout == HashType::OneToOne ? 1 : 0),
      getKeyComponentWidth(),
      hash_table->getEntryCount(),
      ptr1,
      ptr2,
      ptr3,
      ptr4,
      buffer_size,
      raw);
}

std::set<DecodedJoinHashBufferEntry> BaselineJoinHashTable::toSet(
    const ExecutorDeviceType device_type,
    const int device_id) const {
  auto buffer = getJoinHashBuffer(device_type, device_id);
  auto hash_table = getHashTableForDevice(device_id);
  CHECK(hash_table);
  auto buffer_size = hash_table->getHashTableBufferSize(device_type);
#ifdef HAVE_CUDA
  std::unique_ptr<int8_t[]> buffer_copy;
  if (device_type == ExecutorDeviceType::GPU) {
    buffer_copy = std::make_unique<int8_t[]>(buffer_size);
    auto& data_mgr = catalog_->getDataMgr();
    auto device_allocator = data_mgr.createGpuAllocator(device_id);
    device_allocator->copyFromDevice(buffer_copy.get(), buffer, buffer_size);
  }
  auto ptr1 = buffer_copy ? buffer_copy.get() : buffer;
#else
  auto ptr1 = buffer;
#endif  // HAVE_CUDA
  auto ptr2 = ptr1 + offsetBufferOff();
  auto ptr3 = ptr1 + countBufferOff();
  auto ptr4 = ptr1 + payloadBufferOff();
  const auto layout = hash_table->getLayout();
  return HashTable::toSet(getKeyComponentCount() + (layout == HashType::OneToOne ? 1 : 0),
                          getKeyComponentWidth(),
                          hash_table->getEntryCount(),
                          ptr1,
                          ptr2,
                          ptr3,
                          ptr4,
                          buffer_size);
}

void BaselineJoinHashTable::reify(const HashType preferred_layout) {
  auto timer = DEBUG_TIMER(__func__);
  CHECK_LT(0, device_count_);
  const auto composite_key_info =
      HashJoin::getCompositeKeyInfo(inner_outer_pairs_, executor_);
  const auto type_and_found = HashTypeCache::get(composite_key_info.cache_key_chunks);
  const auto layout = type_and_found.second ? type_and_found.first : preferred_layout;

  HashJoin::checkHashJoinReplicationConstraint(
      getInnerTableId(inner_outer_pairs_),
      BaselineJoinHashTable::getShardCountForCondition(
          condition_.get(), executor_, inner_outer_pairs_),
      executor_);

  if (condition_->is_overlaps_oper()) {
    CHECK_EQ(inner_outer_pairs_.size(), size_t(1));
    HashType layout;

    if (inner_outer_pairs_[0].second->get_type_info().is_array()) {
      layout = HashType::ManyToMany;
    } else {
      layout = HashType::OneToMany;
    }
    try {
      reifyWithLayout(layout);
      return;
    } catch (const std::exception& e) {
      VLOG(1) << "Caught exception while building overlaps baseline hash table: "
              << e.what();
      throw;
    }
  }

  try {
    reifyWithLayout(layout);
  } catch (const std::exception& e) {
    VLOG(1) << "Caught exception while building baseline hash table: " << e.what();
    freeHashBufferMemory();
    HashTypeCache::set(composite_key_info.cache_key_chunks, HashType::OneToMany);
    reifyWithLayout(HashType::OneToMany);
  }
}

void BaselineJoinHashTable::reifyWithLayout(const HashType layout) {
  const auto& query_info = get_inner_query_info(getInnerTableId(), query_infos_).info;
  if (query_info.fragments.empty()) {
    return;
  }

  const auto total_entries = 2 * query_info.getNumTuplesUpperBound();
  if (total_entries > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw TooManyHashEntries();
  }

  auto& data_mgr = catalog_->getDataMgr();
  std::vector<std::unique_ptr<CudaAllocator>> dev_buff_owners;
  if (memory_level_ == Data_Namespace::MemoryLevel::GPU_LEVEL) {
    for (int device_id = 0; device_id < device_count_; ++device_id) {
      dev_buff_owners.emplace_back(std::make_unique<CudaAllocator>(&data_mgr, device_id));
    }
  }
  std::vector<ColumnsForDevice> columns_per_device;
  const auto shard_count = shardCount();
  auto entries_per_device =
      get_entries_per_device(total_entries, shard_count, device_count_, memory_level_);

  for (int device_id = 0; device_id < device_count_; ++device_id) {
    const auto fragments =
        shard_count
            ? only_shards_for_device(query_info.fragments, device_id, device_count_)
            : query_info.fragments;
    const auto columns_for_device =
        fetchColumnsForDevice(fragments,
                              device_id,
                              memory_level_ == Data_Namespace::MemoryLevel::GPU_LEVEL
                                  ? dev_buff_owners[device_id].get()
                                  : nullptr);
    columns_per_device.push_back(columns_for_device);
  }
  size_t emitted_keys_count = 0;
  if (layout == HashType::OneToMany) {
    CHECK(!columns_per_device.front().join_columns.empty());
    emitted_keys_count = columns_per_device.front().join_columns.front().num_elems;
    size_t tuple_count;
    std::tie(tuple_count, std::ignore) = approximateTupleCount(columns_per_device);
    const auto entry_count = 2 * std::max(tuple_count, size_t(1));

    // reset entries per device with one to many info
    entries_per_device =
        get_entries_per_device(entry_count, shard_count, device_count_, memory_level_);
  }
  std::vector<std::future<void>> init_threads;
  for (int device_id = 0; device_id < device_count_; ++device_id) {
    const auto fragments =
        shard_count
            ? only_shards_for_device(query_info.fragments, device_id, device_count_)
            : query_info.fragments;
    init_threads.push_back(std::async(std::launch::async,
                                      &BaselineJoinHashTable::reifyForDevice,
                                      this,
                                      columns_per_device[device_id],
                                      layout,
                                      device_id,
                                      entries_per_device,
                                      emitted_keys_count,
                                      logger::thread_id()));
  }
  for (auto& init_thread : init_threads) {
    init_thread.wait();
  }
  for (auto& init_thread : init_threads) {
    init_thread.get();
  }
}

std::pair<size_t, size_t> BaselineJoinHashTable::approximateTupleCount(
    const std::vector<ColumnsForDevice>& columns_per_device) const {
  const auto effective_memory_level = getEffectiveMemoryLevel(inner_outer_pairs_);
  CountDistinctDescriptor count_distinct_desc{
      CountDistinctImplType::Bitmap,
      0,
      11,
      true,
      effective_memory_level == Data_Namespace::MemoryLevel::GPU_LEVEL
          ? ExecutorDeviceType::GPU
          : ExecutorDeviceType::CPU,
      1};
  const auto padded_size_bytes = count_distinct_desc.bitmapPaddedSizeBytes();

  CHECK(!columns_per_device.empty() && !columns_per_device.front().join_columns.empty());

  if (effective_memory_level == Data_Namespace::MemoryLevel::CPU_LEVEL) {
    const auto composite_key_info =
        HashJoin::getCompositeKeyInfo(inner_outer_pairs_, executor_);
    HashTableCacheKey cache_key{columns_per_device.front().join_columns.front().num_elems,
                                composite_key_info.cache_key_chunks,
                                condition_->get_optype(),
                                join_type_};
    const auto cached_count_info = getApproximateTupleCountFromCache(cache_key);
    if (cached_count_info.first) {
      VLOG(1) << "Using a cached tuple count: " << *cached_count_info.first
              << ", emitted keys count: " << cached_count_info.second;
      return std::make_pair(*cached_count_info.first, cached_count_info.second);
    }
    int thread_count = cpu_threads();
    std::vector<uint8_t> hll_buffer_all_cpus(thread_count * padded_size_bytes);
    auto hll_result = &hll_buffer_all_cpus[0];

    approximate_distinct_tuples(hll_result,
                                count_distinct_desc.bitmap_sz_bits,
                                padded_size_bytes,
                                columns_per_device.front().join_columns,
                                columns_per_device.front().join_column_types,
                                thread_count);
    for (int i = 1; i < thread_count; ++i) {
      hll_unify(hll_result,
                hll_result + i * padded_size_bytes,
                1 << count_distinct_desc.bitmap_sz_bits);
    }
    return std::make_pair(hll_size(hll_result, count_distinct_desc.bitmap_sz_bits), 0);
  }
#ifdef HAVE_CUDA
  auto& data_mgr = catalog_->getDataMgr();
  std::vector<std::vector<uint8_t>> host_hll_buffers(device_count_);
  for (auto& host_hll_buffer : host_hll_buffers) {
    host_hll_buffer.resize(count_distinct_desc.bitmapPaddedSizeBytes());
  }
  std::vector<std::future<void>> approximate_distinct_device_threads;
  for (int device_id = 0; device_id < device_count_; ++device_id) {
    approximate_distinct_device_threads.emplace_back(std::async(
        std::launch::async,
        [device_id,
         &columns_per_device,
         &count_distinct_desc,
         &data_mgr,
         &host_hll_buffers] {
          auto allocator = data_mgr.createGpuAllocator(device_id);
          auto device_hll_buffer =
              allocator->alloc(count_distinct_desc.bitmapPaddedSizeBytes());
          data_mgr.getCudaMgr()->zeroDeviceMem(
              device_hll_buffer, count_distinct_desc.bitmapPaddedSizeBytes(), device_id);
          const auto& columns_for_device = columns_per_device[device_id];
          auto join_columns_gpu = transfer_vector_of_flat_objects_to_gpu(
              columns_for_device.join_columns, *allocator);
          auto join_column_types_gpu = transfer_vector_of_flat_objects_to_gpu(
              columns_for_device.join_column_types, *allocator);
          const auto key_handler =
              GenericKeyHandler(columns_for_device.join_columns.size(),
                                true,
                                join_columns_gpu,
                                join_column_types_gpu,
                                nullptr,
                                nullptr);
          const auto key_handler_gpu =
              transfer_flat_object_to_gpu(key_handler, *allocator);
          approximate_distinct_tuples_on_device(
              reinterpret_cast<uint8_t*>(device_hll_buffer),
              count_distinct_desc.bitmap_sz_bits,
              key_handler_gpu,
              columns_for_device.join_columns[0].num_elems);

          auto& host_hll_buffer = host_hll_buffers[device_id];

          allocator->copyFromDevice(&host_hll_buffer[0],
                                    device_hll_buffer,
                                    count_distinct_desc.bitmapPaddedSizeBytes());
        }));
  }
  for (auto& child : approximate_distinct_device_threads) {
    child.get();
  }
  CHECK_EQ(Data_Namespace::MemoryLevel::GPU_LEVEL, effective_memory_level);
  auto& result_hll_buffer = host_hll_buffers.front();
  auto hll_result = reinterpret_cast<int32_t*>(&result_hll_buffer[0]);
  for (int device_id = 1; device_id < device_count_; ++device_id) {
    auto& host_hll_buffer = host_hll_buffers[device_id];
    hll_unify(hll_result,
              reinterpret_cast<int32_t*>(&host_hll_buffer[0]),
              1 << count_distinct_desc.bitmap_sz_bits);
  }
  return std::make_pair(hll_size(hll_result, count_distinct_desc.bitmap_sz_bits), 0);
#else
  UNREACHABLE();
  return {0, 0};
#endif  // HAVE_CUDA
}

ColumnsForDevice BaselineJoinHashTable::fetchColumnsForDevice(
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments,
    const int device_id,
    DeviceAllocator* dev_buff_owner) {
  const auto effective_memory_level = getEffectiveMemoryLevel(inner_outer_pairs_);

  std::vector<JoinColumn> join_columns;
  std::vector<std::shared_ptr<Chunk_NS::Chunk>> chunks_owner;
  std::vector<JoinColumnTypeInfo> join_column_types;
  std::vector<JoinBucketInfo> join_bucket_info;
  std::vector<std::shared_ptr<void>> malloc_owner;
  for (const auto& inner_outer_pair : inner_outer_pairs_) {
    const auto inner_col = inner_outer_pair.first;
    const auto inner_cd = get_column_descriptor_maybe(
        inner_col->get_column_id(), inner_col->get_table_id(), *catalog_);
    if (inner_cd && inner_cd->isVirtualCol) {
      throw FailedToJoinOnVirtualColumn();
    }
    join_columns.emplace_back(fetchJoinColumn(inner_col,
                                              fragments,
                                              effective_memory_level,
                                              device_id,
                                              chunks_owner,
                                              dev_buff_owner,
                                              malloc_owner,
                                              executor_,
                                              &column_cache_));
    const auto& ti = inner_col->get_type_info();
    join_column_types.emplace_back(JoinColumnTypeInfo{static_cast<size_t>(ti.get_size()),
                                                      0,
                                                      0,
                                                      inline_fixed_encoding_null_val(ti),
                                                      isBitwiseEq(),
                                                      0,
                                                      get_join_column_type_kind(ti)});
  }
  return {join_columns, join_column_types, chunks_owner, join_bucket_info, malloc_owner};
}

void BaselineJoinHashTable::reifyForDevice(const ColumnsForDevice& columns_for_device,
                                           const HashType layout,
                                           const int device_id,
                                           const size_t entry_count,
                                           const size_t emitted_keys_count,
                                           const logger::ThreadId parent_thread_id) {
  DEBUG_TIMER_NEW_THREAD(parent_thread_id);
  const auto effective_memory_level = getEffectiveMemoryLevel(inner_outer_pairs_);
  const auto err = initHashTableForDevice(columns_for_device.join_columns,
                                          columns_for_device.join_column_types,
                                          columns_for_device.join_buckets,
                                          layout,
                                          effective_memory_level,
                                          entry_count,
                                          emitted_keys_count,
                                          device_id);
  if (err) {
    throw HashJoinFail(
        std::string("Unrecognized error when initializing baseline hash table (") +
        std::to_string(err) + std::string(")"));
  }
}

size_t BaselineJoinHashTable::shardCount() const {
  if (memory_level_ != Data_Namespace::GPU_LEVEL) {
    return 0;
  }
  return BaselineJoinHashTable::getShardCountForCondition(
      condition_.get(), executor_, inner_outer_pairs_);
}

size_t BaselineJoinHashTable::getKeyComponentWidth() const {
  for (const auto& inner_outer_pair : inner_outer_pairs_) {
    const auto inner_col = inner_outer_pair.first;
    const auto& inner_col_ti = inner_col->get_type_info();
    if (inner_col_ti.get_logical_size() > 4) {
      CHECK_EQ(8, inner_col_ti.get_logical_size());
      return 8;
    }
  }
  return 4;
}

size_t BaselineJoinHashTable::getKeyComponentCount() const {
  return inner_outer_pairs_.size();
}

Data_Namespace::MemoryLevel BaselineJoinHashTable::getEffectiveMemoryLevel(
    const std::vector<InnerOuter>& inner_outer_pairs) const {
  for (const auto& inner_outer_pair : inner_outer_pairs) {
    if (needs_dictionary_translation(
            inner_outer_pair.first, inner_outer_pair.second, executor_)) {
      return Data_Namespace::CPU_LEVEL;
    }
  }
  return memory_level_;
}

int BaselineJoinHashTable::initHashTableForDevice(
    const std::vector<JoinColumn>& join_columns,
    const std::vector<JoinColumnTypeInfo>& join_column_types,
    const std::vector<JoinBucketInfo>& join_bucket_info,
    const HashType layout,
    const Data_Namespace::MemoryLevel effective_memory_level,
    const size_t entry_count,
    const size_t emitted_keys_count,
    const int device_id) {
  auto timer = DEBUG_TIMER(__func__);
  const auto key_component_count = getKeyComponentCount();
  int err = 0;

  if (effective_memory_level == Data_Namespace::CPU_LEVEL) {
    std::lock_guard<std::mutex> cpu_hash_table_buff_lock(cpu_hash_table_buff_mutex_);

    const auto composite_key_info =
        HashJoin::getCompositeKeyInfo(inner_outer_pairs_, executor_);

    CHECK(!join_columns.empty());
    HashTableCacheKey cache_key{join_columns.front().num_elems,
                                composite_key_info.cache_key_chunks,
                                condition_->get_optype(),
                                join_type_};

    if (memory_level_ == Data_Namespace::MemoryLevel::CPU_LEVEL) {
      CHECK_EQ(device_id, size_t(0));
    }
    CHECK_LT(static_cast<size_t>(device_id), hash_tables_for_device_.size());

    auto hash_table = initHashTableOnCpuFromCache(cache_key);
    if (hash_table) {
      hash_tables_for_device_[device_id] = hash_table;
    } else {
      BaselineJoinHashTableBuilder builder(catalog_);

      const auto key_handler =
          GenericKeyHandler(key_component_count,
                            true,
                            &join_columns[0],
                            &join_column_types[0],
                            &composite_key_info.sd_inner_proxy_per_key[0],
                            &composite_key_info.sd_outer_proxy_per_key[0]);
      err = builder.initHashTableOnCpu(&key_handler,
                                       composite_key_info,
                                       join_columns,
                                       join_column_types,
                                       join_bucket_info,
                                       entry_count,
                                       join_columns.front().num_elems,
                                       layout,
                                       join_type_,
                                       getKeyComponentWidth(),
                                       getKeyComponentCount());
      hash_tables_for_device_[device_id] = builder.getHashTable();

      if (!err) {
        if (getInnerTableId() > 0) {
          putHashTableOnCpuToCache(cache_key, hash_tables_for_device_[device_id]);
        }
      }
    }
    // Transfer the hash table on the GPU if we've only built it on CPU
    // but the query runs on GPU (join on dictionary encoded columns).
    // Don't transfer the buffer if there was an error since we'll bail anyway.
    if (memory_level_ == Data_Namespace::GPU_LEVEL && !err) {
#ifdef HAVE_CUDA
      BaselineJoinHashTableBuilder builder(catalog_);

      builder.allocateDeviceMemory(layout,
                                   getKeyComponentWidth(),
                                   getKeyComponentCount(),
                                   entry_count,
                                   emitted_keys_count,
                                   device_id);

      CHECK_LT(size_t(device_id), hash_tables_for_device_.size());
      auto cpu_source_hash_table = hash_tables_for_device_[device_id];
      CHECK(cpu_source_hash_table);
      auto gpu_target_hash_table = builder.getHashTable();
      CHECK(gpu_target_hash_table);

      const auto gpu_buff = gpu_target_hash_table->getGpuBuffer();
      CHECK(gpu_buff);
      auto& data_mgr = catalog_->getDataMgr();
      auto allocator = data_mgr.createGpuAllocator(device_id);
      allocator->copyToDevice(
          gpu_buff,
          cpu_source_hash_table->getCpuBuffer(),
          cpu_source_hash_table->getHashTableBufferSize(ExecutorDeviceType::CPU));
      hash_tables_for_device_[device_id] = std::move(gpu_target_hash_table);
#else
      CHECK(false);
#endif
    }
  } else {
#ifdef HAVE_CUDA
    BaselineJoinHashTableBuilder builder(catalog_);

    auto& data_mgr = catalog_->getDataMgr();
    CudaAllocator allocator(&data_mgr, device_id);
    auto join_column_types_gpu =
        transfer_vector_of_flat_objects_to_gpu(join_column_types, allocator);
    auto join_columns_gpu =
        transfer_vector_of_flat_objects_to_gpu(join_columns, allocator);
    const auto key_handler = GenericKeyHandler(key_component_count,
                                               true,
                                               join_columns_gpu,
                                               join_column_types_gpu,
                                               nullptr,
                                               nullptr);

    err = builder.initHashTableOnGpu(&key_handler,
                                     join_columns,
                                     layout,
                                     join_type_,
                                     getKeyComponentWidth(),
                                     getKeyComponentCount(),
                                     entry_count,
                                     emitted_keys_count,
                                     device_id);
    CHECK_LT(size_t(device_id), hash_tables_for_device_.size());
    hash_tables_for_device_[device_id] = builder.getHashTable();
#else
    UNREACHABLE();
#endif
  }
  return err;
}

#define LL_CONTEXT executor_->cgen_state_->context_
#define LL_BUILDER executor_->cgen_state_->ir_builder_
#define LL_INT(v) executor_->cgen_state_->llInt(v)
#define LL_FP(v) executor_->cgen_state_->llFp(v)
#define ROW_FUNC executor_->cgen_state_->row_func_

llvm::Value* BaselineJoinHashTable::codegenSlot(const CompilationOptions& co,
                                                const size_t index) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  CHECK(getHashType() == HashType::OneToOne);
  const auto key_component_width = getKeyComponentWidth();
  CHECK(key_component_width == 4 || key_component_width == 8);
  auto key_buff_lv = codegenKey(co);
  const auto hash_ptr = hashPtr(index);
  const auto key_ptr_lv =
      LL_BUILDER.CreatePointerCast(key_buff_lv, llvm::Type::getInt8PtrTy(LL_CONTEXT));
  const auto key_size_lv = LL_INT(getKeyComponentCount() * key_component_width);
  const auto hash_table = getHashTableForDevice(size_t(0));
  return executor_->cgen_state_->emitExternalCall(
      "baseline_hash_join_idx_" + std::to_string(key_component_width * 8),
      get_int_type(64, LL_CONTEXT),
      {hash_ptr, key_ptr_lv, key_size_lv, LL_INT(hash_table->getEntryCount())});
}

HashJoinMatchingSet BaselineJoinHashTable::codegenMatchingSet(
    const CompilationOptions& co,
    const size_t index) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  const auto hash_table = getHashTableForDevice(size_t(0));
  CHECK(hash_table);
  const auto key_component_width = getKeyComponentWidth();
  CHECK(key_component_width == 4 || key_component_width == 8);
  auto key_buff_lv = codegenKey(co);
  CHECK(getHashType() == HashType::OneToMany);
  auto hash_ptr = HashJoin::codegenHashTableLoad(index, executor_);
  const auto composite_dict_ptr_type =
      llvm::Type::getIntNPtrTy(LL_CONTEXT, key_component_width * 8);
  const auto composite_key_dict =
      hash_ptr->getType()->isPointerTy()
          ? LL_BUILDER.CreatePointerCast(hash_ptr, composite_dict_ptr_type)
          : LL_BUILDER.CreateIntToPtr(hash_ptr, composite_dict_ptr_type);
  const auto key_component_count = getKeyComponentCount();
  const auto key = executor_->cgen_state_->emitExternalCall(
      "get_composite_key_index_" + std::to_string(key_component_width * 8),
      get_int_type(64, LL_CONTEXT),
      {key_buff_lv,
       LL_INT(key_component_count),
       composite_key_dict,
       LL_INT(hash_table->getEntryCount())});
  auto one_to_many_ptr = hash_ptr;
  if (one_to_many_ptr->getType()->isPointerTy()) {
    one_to_many_ptr =
        LL_BUILDER.CreatePtrToInt(hash_ptr, llvm::Type::getInt64Ty(LL_CONTEXT));
  } else {
    CHECK(one_to_many_ptr->getType()->isIntegerTy(64));
  }
  const auto composite_key_dict_size = offsetBufferOff();
  one_to_many_ptr =
      LL_BUILDER.CreateAdd(one_to_many_ptr, LL_INT(composite_key_dict_size));
  return HashJoin::codegenMatchingSet(
      {one_to_many_ptr, key, LL_INT(int64_t(0)), LL_INT(hash_table->getEntryCount() - 1)},
      false,
      false,
      false,
      getComponentBufferSize(),
      executor_);
}

size_t BaselineJoinHashTable::offsetBufferOff() const noexcept {
  return getKeyBufferSize();
}

size_t BaselineJoinHashTable::countBufferOff() const noexcept {
  if (layoutRequiresAdditionalBuffers(getHashType())) {
    return offsetBufferOff() + getComponentBufferSize();
  } else {
    return getKeyBufferSize();
  }
}

size_t BaselineJoinHashTable::payloadBufferOff() const noexcept {
  if (layoutRequiresAdditionalBuffers(getHashType())) {
    return countBufferOff() + getComponentBufferSize();
  } else {
    return getKeyBufferSize();
  }
}

size_t BaselineJoinHashTable::getKeyBufferSize() const noexcept {
  const auto key_component_width = getKeyComponentWidth();
  CHECK(key_component_width == 4 || key_component_width == 8);
  const auto key_component_count = getKeyComponentCount();
  auto hash_table = getHashTableForDevice(size_t(0));
  CHECK(hash_table);
  if (layoutRequiresAdditionalBuffers(hash_table->getLayout())) {
    return hash_table->getEntryCount() * key_component_count * key_component_width;
  } else {
    return hash_table->getEntryCount() * (key_component_count + 1) * key_component_width;
  }
}

size_t BaselineJoinHashTable::getComponentBufferSize() const noexcept {
  const auto hash_table = getHashTableForDevice(size_t(0));
  return hash_table->getEntryCount() * sizeof(int32_t);
}

llvm::Value* BaselineJoinHashTable::codegenKey(const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  const auto key_component_width = getKeyComponentWidth();
  CHECK(key_component_width == 4 || key_component_width == 8);
  const auto key_size_lv = LL_INT(getKeyComponentCount() * key_component_width);
  llvm::Value* key_buff_lv{nullptr};
  switch (key_component_width) {
    case 4:
      key_buff_lv =
          LL_BUILDER.CreateAlloca(llvm::Type::getInt32Ty(LL_CONTEXT), key_size_lv);
      break;
    case 8:
      key_buff_lv =
          LL_BUILDER.CreateAlloca(llvm::Type::getInt64Ty(LL_CONTEXT), key_size_lv);
      break;
    default:
      CHECK(false);
  }

  CodeGenerator code_generator(executor_);
  for (size_t i = 0; i < getKeyComponentCount(); ++i) {
    const auto key_comp_dest_lv = LL_BUILDER.CreateGEP(key_buff_lv, LL_INT(i));
    const auto& inner_outer_pair = inner_outer_pairs_[i];
    const auto outer_col = inner_outer_pair.second;
    const auto key_col_var = dynamic_cast<const Analyzer::ColumnVar*>(outer_col);
    const auto val_col_var =
        dynamic_cast<const Analyzer::ColumnVar*>(inner_outer_pair.first);
    if (key_col_var && val_col_var &&
        self_join_not_covered_by_left_deep_tree(
            key_col_var,
            val_col_var,
            get_max_rte_scan_table(executor_->cgen_state_->scan_idx_to_hash_pos_))) {
      throw std::runtime_error(
          "Query execution fails because the query contains not supported self-join "
          "pattern. We suspect the query requires multiple left-deep join tree due to "
          "the join condition of the self-join and is not supported for now. Please "
          "consider rewriting table order in "
          "FROM clause.");
    }
    const auto col_lvs = code_generator.codegen(outer_col, true, co);
    CHECK_EQ(size_t(1), col_lvs.size());
    const auto col_lv = LL_BUILDER.CreateSExt(
        col_lvs.front(), get_int_type(key_component_width * 8, LL_CONTEXT));
    LL_BUILDER.CreateStore(col_lv, key_comp_dest_lv);
  }
  return key_buff_lv;
}

llvm::Value* BaselineJoinHashTable::hashPtr(const size_t index) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  auto hash_ptr = HashJoin::codegenHashTableLoad(index, executor_);
  const auto pi8_type = llvm::Type::getInt8PtrTy(LL_CONTEXT);
  return hash_ptr->getType()->isPointerTy()
             ? LL_BUILDER.CreatePointerCast(hash_ptr, pi8_type)
             : LL_BUILDER.CreateIntToPtr(hash_ptr, pi8_type);
}

#undef ROW_FUNC
#undef LL_INT
#undef LL_BUILDER
#undef LL_CONTEXT

int BaselineJoinHashTable::getInnerTableId() const noexcept {
  try {
    return getInnerTableId(inner_outer_pairs_);
  } catch (...) {
    CHECK(false);
  }
  return 0;
}

int BaselineJoinHashTable::getInnerTableRteIdx() const noexcept {
  CHECK(!inner_outer_pairs_.empty());
  const auto first_inner_col = inner_outer_pairs_.front().first;
  return first_inner_col->get_rte_idx();
}

HashType BaselineJoinHashTable::getHashType() const noexcept {
  auto hash_table = getHashTableForDevice(size_t(0));
  CHECK(hash_table);
  if (layout_override_) {
    return *layout_override_;
  } else {
    return hash_table->getLayout();
  }
}

int BaselineJoinHashTable::getInnerTableId(
    const std::vector<InnerOuter>& inner_outer_pairs) {
  CHECK(!inner_outer_pairs.empty());
  const auto first_inner_col = inner_outer_pairs.front().first;
  return first_inner_col->get_table_id();
}

std::shared_ptr<HashTable> BaselineJoinHashTable::initHashTableOnCpuFromCache(
    const HashTableCacheKey& key) {
  auto timer = DEBUG_TIMER(__func__);
  VLOG(1) << "Checking CPU hash table cache.";
  CHECK(hash_table_cache_);
  auto hash_table_opt = (hash_table_cache_->get(key));
  return hash_table_opt ? *hash_table_opt : nullptr;
}

void BaselineJoinHashTable::putHashTableOnCpuToCache(
    const HashTableCacheKey& key,
    std::shared_ptr<HashTable>& hash_table) {
  for (auto chunk_key : key.chunk_keys) {
    CHECK_GE(chunk_key.size(), size_t(2));
    if (chunk_key[1] < 0) {
      return;
    }
  }
  CHECK(hash_table_cache_);
  hash_table_cache_->insert(key, hash_table);
}

std::pair<std::optional<size_t>, size_t>
BaselineJoinHashTable::getApproximateTupleCountFromCache(
    const HashTableCacheKey& key) const {
  for (auto chunk_key : key.chunk_keys) {
    CHECK_GE(chunk_key.size(), size_t(2));
    if (chunk_key[1] < 0) {
      return std::make_pair(std::nullopt, 0);
      ;
    }
  }

  CHECK(hash_table_cache_);
  auto hash_table_opt = hash_table_cache_->get(key);
  if (hash_table_opt) {
    auto hash_table = *hash_table_opt;
    return std::make_pair(hash_table->getEntryCount() / 2,
                          hash_table->getEmittedKeysCount());
  }
  return std::make_pair(std::nullopt, 0);
}

bool BaselineJoinHashTable::isBitwiseEq() const {
  return condition_->get_optype() == kBW_EQ;
}

std::map<std::vector<ChunkKey>, HashType> HashTypeCache::hash_type_cache_;
std::mutex HashTypeCache::hash_type_cache_mutex_;

void HashTypeCache::set(const std::vector<ChunkKey>& key, const HashType hash_type) {
  for (auto chunk_key : key) {
    CHECK_GE(chunk_key.size(), size_t(2));
    if (chunk_key[1] < 0) {
      return;
    }
  }
  std::lock_guard<std::mutex> hash_type_cache_lock(hash_type_cache_mutex_);
  hash_type_cache_[key] = hash_type;
}

std::pair<HashType, bool> HashTypeCache::get(const std::vector<ChunkKey>& key) {
  std::lock_guard<std::mutex> hash_type_cache_lock(hash_type_cache_mutex_);
  const auto it = hash_type_cache_.find(key);
  if (it == hash_type_cache_.end()) {
    return {HashType::OneToOne, false};
  }
  return {it->second, true};
}

void HashTypeCache::clear() {
  std::lock_guard<std::mutex> hash_type_cache_lock(hash_type_cache_mutex_);
  hash_type_cache_.clear();
}
