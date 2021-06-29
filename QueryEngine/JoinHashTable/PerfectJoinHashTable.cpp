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

#include "QueryEngine/JoinHashTable/PerfectJoinHashTable.h"

#include <atomic>
#include <future>
#include <numeric>
#include <thread>

#include "Logger/Logger.h"
#include "QueryEngine/CodeGenerator.h"
#include "QueryEngine/ColumnFetcher.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ExpressionRewrite.h"
#include "QueryEngine/JoinHashTable/Builders/PerfectHashTableBuilder.h"
#include "QueryEngine/JoinHashTable/Runtime/HashJoinRuntime.h"
#include "QueryEngine/RuntimeFunctions.h"

std::unique_ptr<HashTableCache<PerfectJoinHashTable::JoinHashTableCacheKey,
                               PerfectJoinHashTable::HashTableCacheValue>>
    PerfectJoinHashTable::hash_table_cache_ =
        std::make_unique<HashTableCache<PerfectJoinHashTable::JoinHashTableCacheKey,
                                        PerfectJoinHashTable::HashTableCacheValue>>();

namespace {

InnerOuter get_cols(const Analyzer::BinOper* qual_bin_oper,
                    const Catalog_Namespace::Catalog& cat,
                    const TemporaryTables* temporary_tables) {
  const auto lhs = qual_bin_oper->get_left_operand();
  const auto rhs = qual_bin_oper->get_right_operand();
  return normalize_column_pair(lhs, rhs, cat, temporary_tables);
}

HashEntryInfo get_bucketized_hash_entry_info(SQLTypeInfo const& context_ti,
                                             ExpressionRange const& col_range,
                                             bool const is_bw_eq) {
  using EmptyRangeSize = boost::optional<size_t>;
  auto empty_range_check = [](ExpressionRange const& col_range,
                              bool const is_bw_eq) -> EmptyRangeSize {
    if (col_range.getIntMin() > col_range.getIntMax()) {
      CHECK_EQ(col_range.getIntMin(), int64_t(0));
      CHECK_EQ(col_range.getIntMax(), int64_t(-1));
      if (is_bw_eq) {
        return size_t(1);
      }
      return size_t(0);
    }
    return EmptyRangeSize{};
  };

  auto empty_range = empty_range_check(col_range, is_bw_eq);
  if (empty_range) {
    return {size_t(*empty_range), 1};
  }

  int64_t bucket_normalization =
      context_ti.get_type() == kDATE ? col_range.getBucket() : 1;
  CHECK_GT(bucket_normalization, 0);
  return {size_t(col_range.getIntMax() - col_range.getIntMin() + 1 + (is_bw_eq ? 1 : 0)),
          bucket_normalization};
}

size_t get_hash_entry_count(const ExpressionRange& col_range, const bool is_bw_eq) {
  if (col_range.getIntMin() > col_range.getIntMax()) {
    CHECK_EQ(col_range.getIntMin(), int64_t(0));
    CHECK_EQ(col_range.getIntMax(), int64_t(-1));
    return is_bw_eq ? 1 : 0;
  }
  return col_range.getIntMax() - col_range.getIntMin() + 1 + (is_bw_eq ? 1 : 0);
}

}  // namespace

namespace {

bool shard_count_less_or_equal_device_count(const int inner_table_id,
                                            const Executor* executor) {
  const auto inner_table_info = executor->getTableInfo(inner_table_id);
  std::unordered_set<int> device_holding_fragments;
  auto cuda_mgr = executor->getCatalog()->getDataMgr().getCudaMgr();
  const int device_count = cuda_mgr ? cuda_mgr->getDeviceCount() : 1;
  for (const auto& fragment : inner_table_info.fragments) {
    if (fragment.shard != -1) {
      const auto it_ok = device_holding_fragments.emplace(fragment.shard % device_count);
      if (!it_ok.second) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

size_t get_shard_count(
    std::pair<const Analyzer::ColumnVar*, const Analyzer::Expr*> equi_pair,
    const Executor* executor) {
  const auto inner_col = equi_pair.first;
  const auto outer_col = dynamic_cast<const Analyzer::ColumnVar*>(equi_pair.second);
  if (!outer_col || inner_col->get_table_id() < 0 || outer_col->get_table_id() < 0) {
    return 0;
  }
  if (outer_col->get_rte_idx()) {
    return 0;
  }
  if (inner_col->get_type_info() != outer_col->get_type_info()) {
    return 0;
  }
  const auto catalog = executor->getCatalog();
  const auto inner_td = catalog->getMetadataForTable(inner_col->get_table_id());
  CHECK(inner_td);
  const auto outer_td = catalog->getMetadataForTable(outer_col->get_table_id());
  CHECK(outer_td);
  if (inner_td->shardedColumnId == 0 || outer_td->shardedColumnId == 0 ||
      inner_td->nShards != outer_td->nShards) {
    return 0;
  }
  if (!shard_count_less_or_equal_device_count(inner_td->tableId, executor)) {
    return 0;
  }
  // The two columns involved must be the ones on which the tables have been sharded on.
  return (inner_td->shardedColumnId == inner_col->get_column_id() &&
          outer_td->shardedColumnId == outer_col->get_column_id()) ||
                 (outer_td->shardedColumnId == inner_col->get_column_id() &&
                  inner_td->shardedColumnId == inner_col->get_column_id())
             ? inner_td->nShards
             : 0;
}

//! Make hash table from an in-flight SQL query's parse tree etc.
std::shared_ptr<PerfectJoinHashTable> PerfectJoinHashTable::getInstance(
    const std::shared_ptr<Analyzer::BinOper> qual_bin_oper,
    const std::vector<InputTableInfo>& query_infos,
    const Data_Namespace::MemoryLevel memory_level,
    const JoinType join_type,
    const HashType preferred_hash_type,
    const int device_count,
    ColumnCacheMap& column_cache,
    Executor* executor) {
  decltype(std::chrono::steady_clock::now()) ts1, ts2;
  if (VLOGGING(1)) {
    VLOG(1) << "Building perfect hash table " << getHashTypeString(preferred_hash_type)
            << " for qual: " << qual_bin_oper->toString();
    ts1 = std::chrono::steady_clock::now();
  }
  CHECK(IS_EQUIVALENCE(qual_bin_oper->get_optype()));
  const auto cols =
      get_cols(qual_bin_oper.get(), *executor->getCatalog(), executor->temporary_tables_);
  const auto inner_col = cols.first;
  CHECK(inner_col);
  const auto& ti = inner_col->get_type_info();
  auto col_range =
      getExpressionRange(ti.is_string() ? cols.second : inner_col, query_infos, executor);
  if (col_range.getType() == ExpressionRangeType::Invalid) {
    throw HashJoinFail(
        "Could not compute range for the expressions involved in the equijoin");
  }
  if (ti.is_string()) {
    // The nullable info must be the same as the source column.
    const auto source_col_range = getExpressionRange(inner_col, query_infos, executor);
    if (source_col_range.getType() == ExpressionRangeType::Invalid) {
      throw HashJoinFail(
          "Could not compute range for the expressions involved in the equijoin");
    }
    if (source_col_range.getIntMin() > source_col_range.getIntMax()) {
      // If the inner column expression range is empty, use the inner col range
      CHECK_EQ(source_col_range.getIntMin(), int64_t(0));
      CHECK_EQ(source_col_range.getIntMax(), int64_t(-1));
      col_range = source_col_range;
    } else {
      col_range = ExpressionRange::makeIntRange(
          std::min(source_col_range.getIntMin(), col_range.getIntMin()),
          std::max(source_col_range.getIntMax(), col_range.getIntMax()),
          0,
          source_col_range.hasNulls());
    }
  }
  // We can't allocate more than 2GB contiguous memory on GPU and each entry is 4 bytes.
  const auto max_hash_entry_count =
      memory_level == Data_Namespace::MemoryLevel::GPU_LEVEL
          ? static_cast<size_t>(std::numeric_limits<int32_t>::max() / sizeof(int32_t))
          : static_cast<size_t>(std::numeric_limits<int32_t>::max());

  auto bucketized_entry_count_info = get_bucketized_hash_entry_info(
      ti, col_range, qual_bin_oper->get_optype() == kBW_EQ);
  auto bucketized_entry_count = bucketized_entry_count_info.getNormalizedHashEntryCount();

  if (bucketized_entry_count > max_hash_entry_count) {
    throw TooManyHashEntries();
  }

  if (qual_bin_oper->get_optype() == kBW_EQ &&
      col_range.getIntMax() >= std::numeric_limits<int64_t>::max()) {
    throw HashJoinFail("Cannot translate null value for kBW_EQ");
  }
  auto join_hash_table =
      std::shared_ptr<PerfectJoinHashTable>(new PerfectJoinHashTable(qual_bin_oper,
                                                                     inner_col,
                                                                     query_infos,
                                                                     memory_level,
                                                                     join_type,
                                                                     preferred_hash_type,
                                                                     col_range,
                                                                     column_cache,
                                                                     executor,
                                                                     device_count));
  try {
    join_hash_table->reify();
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
    VLOG(1) << "Built perfect hash table "
            << getHashTypeString(join_hash_table->getHashType()) << " in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(ts2 - ts1).count()
            << " ms";
  }
  return join_hash_table;
}

bool needs_dictionary_translation(const Analyzer::ColumnVar* inner_col,
                                  const Analyzer::Expr* outer_col_expr,
                                  const Executor* executor) {
  const auto catalog = executor->getCatalog();
  CHECK(catalog);
  const auto inner_cd = get_column_descriptor_maybe(
      inner_col->get_column_id(), inner_col->get_table_id(), *catalog);
  const auto& inner_ti = get_column_type(inner_col->get_column_id(),
                                         inner_col->get_table_id(),
                                         inner_cd,
                                         executor->getTemporaryTables());
  // Only strings may need dictionary translation.
  if (!inner_ti.is_string()) {
    return false;
  }
  const auto outer_col = dynamic_cast<const Analyzer::ColumnVar*>(outer_col_expr);
  CHECK(outer_col);
  const auto outer_cd = get_column_descriptor_maybe(
      outer_col->get_column_id(), outer_col->get_table_id(), *catalog);
  // Don't want to deal with temporary tables for now, require translation.
  if (!inner_cd || !outer_cd) {
    return true;
  }
  const auto& outer_ti = get_column_type(outer_col->get_column_id(),
                                         outer_col->get_table_id(),
                                         outer_cd,
                                         executor->getTemporaryTables());
  CHECK_EQ(inner_ti.is_string(), outer_ti.is_string());
  // If the two columns don't share the dictionary, translation is needed.
  return outer_ti.get_comp_param() != inner_ti.get_comp_param();
}

std::vector<Fragmenter_Namespace::FragmentInfo> only_shards_for_device(
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments,
    const int device_id,
    const int device_count) {
  std::vector<Fragmenter_Namespace::FragmentInfo> shards_for_device;
  for (const auto& fragment : fragments) {
    CHECK_GE(fragment.shard, 0);
    if (fragment.shard % device_count == device_id) {
      shards_for_device.push_back(fragment);
    }
  }
  return shards_for_device;
}

void PerfectJoinHashTable::reify() {
  auto timer = DEBUG_TIMER(__func__);
  CHECK_LT(0, device_count_);
  catalog_ = const_cast<Catalog_Namespace::Catalog*>(executor_->getCatalog());
  const auto cols =
      get_cols(qual_bin_oper_.get(), *catalog_, executor_->temporary_tables_);
  const auto inner_col = cols.first;
  HashJoin::checkHashJoinReplicationConstraint(
      inner_col->get_table_id(),
      get_shard_count(qual_bin_oper_.get(), executor_),
      executor_);
  const auto& query_info = getInnerQueryInfo(inner_col).info;
  if (query_info.fragments.empty()) {
    return;
  }
  if (query_info.getNumTuplesUpperBound() >
      static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw TooManyHashEntries();
  }
  std::vector<std::future<void>> init_threads;
  const int shard_count = shardCount();

  inner_outer_pairs_.push_back(cols);
  CHECK_EQ(inner_outer_pairs_.size(), size_t(1));

  std::vector<ColumnsForDevice> columns_per_device;
  std::vector<std::unique_ptr<CudaAllocator>> dev_buff_owners;
  try {
    auto& data_mgr = catalog_->getDataMgr();
    if (memory_level_ == Data_Namespace::MemoryLevel::GPU_LEVEL) {
      for (int device_id = 0; device_id < device_count_; ++device_id) {
        dev_buff_owners.emplace_back(
            std::make_unique<CudaAllocator>(&data_mgr, device_id));
      }
    }
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
      const auto hash_table_key = genHashTableKey(
          fragments, inner_outer_pairs_.front().second, inner_outer_pairs_.front().first);
      init_threads.push_back(std::async(std::launch::async,
                                        &PerfectJoinHashTable::reifyForDevice,
                                        this,
                                        hash_table_key,
                                        columns_per_device[device_id],
                                        hash_type_,
                                        device_id,
                                        logger::thread_id()));
    }
    for (auto& init_thread : init_threads) {
      init_thread.wait();
    }
    for (auto& init_thread : init_threads) {
      init_thread.get();
    }

  } catch (const NeedsOneToManyHash& e) {
    hash_type_ = HashType::OneToMany;
    freeHashBufferMemory();
    init_threads.clear();
    if (memory_level_ == Data_Namespace::MemoryLevel::GPU_LEVEL) {
      CHECK_EQ(dev_buff_owners.size(), size_t(device_count_));
    }
    CHECK_EQ(columns_per_device.size(), size_t(device_count_));
    for (int device_id = 0; device_id < device_count_; ++device_id) {
      const auto fragments =
          shard_count
              ? only_shards_for_device(query_info.fragments, device_id, device_count_)
              : query_info.fragments;
      const auto hash_table_key = genHashTableKey(
          fragments, inner_outer_pairs_.front().second, inner_outer_pairs_.front().first);
      init_threads.push_back(std::async(std::launch::async,
                                        &PerfectJoinHashTable::reifyForDevice,
                                        this,
                                        hash_table_key,
                                        columns_per_device[device_id],
                                        hash_type_,
                                        device_id,
                                        logger::thread_id()));
    }
    for (auto& init_thread : init_threads) {
      init_thread.wait();
    }
    for (auto& init_thread : init_threads) {
      init_thread.get();
    }
  }
}

Data_Namespace::MemoryLevel PerfectJoinHashTable::getEffectiveMemoryLevel(
    const std::vector<InnerOuter>& inner_outer_pairs) const {
  for (const auto& inner_outer_pair : inner_outer_pairs) {
    if (needs_dictionary_translation(
            inner_outer_pair.first, inner_outer_pair.second, executor_)) {
      return Data_Namespace::CPU_LEVEL;
    }
  }
  return memory_level_;
}

ColumnsForDevice PerfectJoinHashTable::fetchColumnsForDevice(
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

void PerfectJoinHashTable::reifyForDevice(const ChunkKey& hash_table_key,
                                          const ColumnsForDevice& columns_for_device,
                                          const HashType layout,
                                          const int device_id,
                                          const logger::ThreadId parent_thread_id) {
  DEBUG_TIMER_NEW_THREAD(parent_thread_id);
  const auto effective_memory_level = getEffectiveMemoryLevel(inner_outer_pairs_);

  CHECK_EQ(columns_for_device.join_columns.size(), size_t(1));
  CHECK_EQ(inner_outer_pairs_.size(), size_t(1));
  auto& join_column = columns_for_device.join_columns.front();
  if (layout == HashType::OneToOne) {
    const auto err = initHashTableForDevice(hash_table_key,
                                            join_column,
                                            inner_outer_pairs_.front(),
                                            layout,
                                            effective_memory_level,
                                            device_id);
    if (err) {
      throw NeedsOneToManyHash();
    }
  } else {
    const auto err = initHashTableForDevice(hash_table_key,
                                            join_column,
                                            inner_outer_pairs_.front(),
                                            HashType::OneToMany,
                                            effective_memory_level,
                                            device_id);
    if (err) {
      throw std::runtime_error("Unexpected error building one to many hash table: " +
                               std::to_string(err));
    }
  }
}

int PerfectJoinHashTable::initHashTableForDevice(
    const ChunkKey& chunk_key,
    const JoinColumn& join_column,
    const InnerOuter& cols,
    const HashType layout,
    const Data_Namespace::MemoryLevel effective_memory_level,
    const int device_id) {
  auto timer = DEBUG_TIMER(__func__);
  const auto inner_col = cols.first;
  CHECK(inner_col);

  auto hash_entry_info = get_bucketized_hash_entry_info(
      inner_col->get_type_info(), col_range_, isBitwiseEq());
  if (!hash_entry_info && layout == HashType::OneToOne) {
    // TODO: what is this for?
    return 0;
  }
#ifndef HAVE_CUDA
  CHECK_EQ(Data_Namespace::CPU_LEVEL, effective_memory_level);
#endif
  int err{0};
  const int32_t hash_join_invalid_val{-1};
  if (effective_memory_level == Data_Namespace::CPU_LEVEL) {
    CHECK(!chunk_key.empty());

    auto hash_table = initHashTableOnCpuFromCache(chunk_key, join_column.num_elems, cols);
    {
      std::lock_guard<std::mutex> cpu_hash_table_buff_lock(cpu_hash_table_buff_mutex_);
      if (!hash_table) {
        PerfectJoinHashTableBuilder builder(executor_->catalog_);
        if (layout == HashType::OneToOne) {
          builder.initOneToOneHashTableOnCpu(join_column,
                                             col_range_,
                                             isBitwiseEq(),
                                             cols,
                                             join_type_,
                                             layout,
                                             hash_entry_info,
                                             hash_join_invalid_val,
                                             executor_);
          hash_table = builder.getHashTable();
        } else {
          builder.initOneToManyHashTableOnCpu(join_column,
                                              col_range_,
                                              isBitwiseEq(),
                                              cols,
                                              hash_entry_info,
                                              hash_join_invalid_val,
                                              executor_);
          hash_table = builder.getHashTable();
        }
      } else {
        if (layout == HashType::OneToOne &&
            hash_table->getHashTableBufferSize(ExecutorDeviceType::CPU) >
                hash_entry_info.getNormalizedHashEntryCount() * sizeof(int32_t)) {
          // TODO: can this ever happen?
          // Too many hash entries, need to retry with a 1:many table
          throw NeedsOneToManyHash();
        }
      }
    }
    if (inner_col->get_table_id() > 0) {
      putHashTableOnCpuToCache(chunk_key, join_column.num_elems, hash_table, cols);
    }
    // Transfer the hash table on the GPU if we've only built it on CPU
    // but the query runs on GPU (join on dictionary encoded columns).
    if (memory_level_ == Data_Namespace::GPU_LEVEL) {
#ifdef HAVE_CUDA
      const auto& ti = inner_col->get_type_info();
      CHECK(ti.is_string());
      auto catalog = executor_->getCatalog();
      CHECK(catalog);
      auto& data_mgr = catalog->getDataMgr();
      std::lock_guard<std::mutex> cpu_hash_table_buff_lock(cpu_hash_table_buff_mutex_);

      PerfectJoinHashTableBuilder gpu_builder(executor_->catalog_);
      gpu_builder.allocateDeviceMemory(join_column,
                                       hash_table->getLayout(),
                                       hash_entry_info,
                                       shardCount(),
                                       device_id,
                                       device_count_);
      std::shared_ptr<PerfectHashTable> gpu_hash_table = gpu_builder.getHashTable();
      CHECK(gpu_hash_table);
      auto gpu_buffer_ptr = gpu_hash_table->getGpuBuffer();
      CHECK(gpu_buffer_ptr);

      CHECK(hash_table);
      // GPU size returns reserved size
      CHECK_LE(hash_table->getHashTableBufferSize(ExecutorDeviceType::CPU),
               gpu_hash_table->getHashTableBufferSize(ExecutorDeviceType::GPU));

      auto device_allocator = data_mgr.createGpuAllocator(device_id);
      device_allocator->copyToDevice(
          gpu_buffer_ptr,
          hash_table->getCpuBuffer(),
          hash_table->getHashTableBufferSize(ExecutorDeviceType::CPU));
      CHECK_LT(size_t(device_id), hash_tables_for_device_.size());
      hash_tables_for_device_[device_id] = std::move(gpu_hash_table);
#else
      UNREACHABLE();
#endif
    } else {
      CHECK(hash_table);
      CHECK_LT(size_t(device_id), hash_tables_for_device_.size());
      hash_tables_for_device_[device_id] = hash_table;
    }
  } else {
#ifdef HAVE_CUDA
    PerfectJoinHashTableBuilder builder(executor_->catalog_);
    CHECK_EQ(Data_Namespace::GPU_LEVEL, effective_memory_level);
    builder.allocateDeviceMemory(
        join_column, layout, hash_entry_info, shardCount(), device_id, device_count_);
    builder.initHashTableOnGpu(chunk_key,
                               join_column,
                               col_range_,
                               isBitwiseEq(),
                               cols,
                               join_type_,
                               layout,
                               hash_entry_info,
                               shardCount(),
                               hash_join_invalid_val,
                               device_id,
                               device_count_,
                               executor_);
    CHECK_LT(size_t(device_id), hash_tables_for_device_.size());
    hash_tables_for_device_[device_id] = builder.getHashTable();
#else
    UNREACHABLE();
#endif
  }

  return err;
}

ChunkKey PerfectJoinHashTable::genHashTableKey(
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments,
    const Analyzer::Expr* outer_col_expr,
    const Analyzer::ColumnVar* inner_col) const {
  ChunkKey hash_table_key{executor_->getCatalog()->getCurrentDB().dbId,
                          inner_col->get_table_id(),
                          inner_col->get_column_id()};
  const auto& ti = inner_col->get_type_info();
  if (ti.is_string()) {
    CHECK_EQ(kENCODING_DICT, ti.get_compression());
    size_t outer_elem_count = 0;
    const auto outer_col = dynamic_cast<const Analyzer::ColumnVar*>(outer_col_expr);
    CHECK(outer_col);
    const auto& outer_query_info = getInnerQueryInfo(outer_col).info;
    for (auto& frag : outer_query_info.fragments) {
      outer_elem_count = frag.getNumTuples();
    }
    hash_table_key.push_back(outer_elem_count);
  }
  if (fragments.size() < 2) {
    hash_table_key.push_back(fragments.front().fragmentId);
  }
  return hash_table_key;
}

std::shared_ptr<PerfectHashTable> PerfectJoinHashTable::initHashTableOnCpuFromCache(
    const ChunkKey& chunk_key,
    const size_t num_elements,
    const InnerOuter& cols) {
  auto timer = DEBUG_TIMER(__func__);
  CHECK_GE(chunk_key.size(), size_t(2));
  if (chunk_key[1] < 0) {
    // Do not cache hash tables over intermediate results
    return nullptr;
  }
  const auto outer_col = dynamic_cast<const Analyzer::ColumnVar*>(cols.second);
  JoinHashTableCacheKey cache_key{col_range_,
                                  *cols.first,
                                  outer_col ? *outer_col : *cols.first,
                                  num_elements,
                                  chunk_key,
                                  qual_bin_oper_->get_optype(),
                                  join_type_};
  auto hash_table_opt = (hash_table_cache_->get(cache_key));
  return hash_table_opt ? *hash_table_opt : nullptr;
}

void PerfectJoinHashTable::putHashTableOnCpuToCache(const ChunkKey& chunk_key,
                                                    const size_t num_elements,
                                                    HashTableCacheValue hash_table,
                                                    const InnerOuter& cols) {
  CHECK_GE(chunk_key.size(), size_t(2));
  if (chunk_key[1] < 0) {
    // Do not cache hash tables over intermediate results
    return;
  }
  const auto outer_col = dynamic_cast<const Analyzer::ColumnVar*>(cols.second);
  JoinHashTableCacheKey cache_key{col_range_,
                                  *cols.first,
                                  outer_col ? *outer_col : *cols.first,
                                  num_elements,
                                  chunk_key,
                                  qual_bin_oper_->get_optype(),
                                  join_type_};
  CHECK(hash_table_cache_);
  CHECK(hash_table && !hash_table->getGpuBuffer());
  hash_table_cache_->insert(cache_key, hash_table);
}

llvm::Value* PerfectJoinHashTable::codegenHashTableLoad(const size_t table_idx) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  const auto hash_ptr = HashJoin::codegenHashTableLoad(table_idx, executor_);
  if (hash_ptr->getType()->isIntegerTy(64)) {
    return hash_ptr;
  }
  CHECK(hash_ptr->getType()->isPointerTy());
  return executor_->cgen_state_->ir_builder_.CreatePtrToInt(
      get_arg_by_name(executor_->cgen_state_->row_func_, "join_hash_tables"),
      llvm::Type::getInt64Ty(executor_->cgen_state_->context_));
}

std::vector<llvm::Value*> PerfectJoinHashTable::getHashJoinArgs(
    llvm::Value* hash_ptr,
    const Analyzer::Expr* key_col,
    const int shard_count,
    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  CodeGenerator code_generator(executor_);
  const auto key_lvs = code_generator.codegen(key_col, true, co);
  CHECK_EQ(size_t(1), key_lvs.size());
  auto const& key_col_ti = key_col->get_type_info();
  auto hash_entry_info =
      get_bucketized_hash_entry_info(key_col_ti, col_range_, isBitwiseEq());

  std::vector<llvm::Value*> hash_join_idx_args{
      hash_ptr,
      executor_->cgen_state_->castToTypeIn(key_lvs.front(), 64),
      executor_->cgen_state_->llInt(col_range_.getIntMin()),
      executor_->cgen_state_->llInt(col_range_.getIntMax())};
  if (shard_count) {
    const auto expected_hash_entry_count =
        get_hash_entry_count(col_range_, isBitwiseEq());
    const auto entry_count_per_shard =
        (expected_hash_entry_count + shard_count - 1) / shard_count;
    hash_join_idx_args.push_back(
        executor_->cgen_state_->llInt<uint32_t>(entry_count_per_shard));
    hash_join_idx_args.push_back(executor_->cgen_state_->llInt<uint32_t>(shard_count));
    hash_join_idx_args.push_back(executor_->cgen_state_->llInt<uint32_t>(device_count_));
  }
  auto key_col_logical_ti = get_logical_type_info(key_col->get_type_info());
  if (!key_col_logical_ti.get_notnull() || isBitwiseEq()) {
    hash_join_idx_args.push_back(executor_->cgen_state_->llInt(
        inline_fixed_encoding_null_val(key_col_logical_ti)));
  }
  auto special_date_bucketization_case = key_col_ti.get_type() == kDATE;
  if (isBitwiseEq()) {
    if (special_date_bucketization_case) {
      hash_join_idx_args.push_back(executor_->cgen_state_->llInt(
          col_range_.getIntMax() / hash_entry_info.bucket_normalization + 1));
    } else {
      hash_join_idx_args.push_back(
          executor_->cgen_state_->llInt(col_range_.getIntMax() + 1));
    }
  }

  if (special_date_bucketization_case) {
    hash_join_idx_args.emplace_back(
        executor_->cgen_state_->llInt(hash_entry_info.bucket_normalization));
  }

  return hash_join_idx_args;
}

HashJoinMatchingSet PerfectJoinHashTable::codegenMatchingSet(const CompilationOptions& co,
                                                             const size_t index) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  const auto cols = get_cols(
      qual_bin_oper_.get(), *executor_->getCatalog(), executor_->temporary_tables_);
  auto key_col = cols.second;
  CHECK(key_col);
  auto val_col = cols.first;
  CHECK(val_col);
  auto pos_ptr = codegenHashTableLoad(index);
  CHECK(pos_ptr);
  const int shard_count = shardCount();
  const auto key_col_var = dynamic_cast<const Analyzer::ColumnVar*>(key_col);
  const auto val_col_var = dynamic_cast<const Analyzer::ColumnVar*>(val_col);
  if (key_col_var && val_col_var &&
      self_join_not_covered_by_left_deep_tree(
          key_col_var,
          val_col_var,
          get_max_rte_scan_table(executor_->cgen_state_->scan_idx_to_hash_pos_))) {
    throw std::runtime_error(
        "Query execution fails because the query contains not supported self-join "
        "pattern. We suspect the query requires multiple left-deep join tree due to "
        "the "
        "join condition of the self-join and is not supported for now. Please consider "
        "rewriting table order in "
        "FROM clause.");
  }
  auto hash_join_idx_args = getHashJoinArgs(pos_ptr, key_col, shard_count, co);
  const int64_t sub_buff_size = getComponentBufferSize();
  const auto& key_col_ti = key_col->get_type_info();

  auto bucketize = (key_col_ti.get_type() == kDATE);
  return HashJoin::codegenMatchingSet(hash_join_idx_args,
                                      shard_count,
                                      !key_col_ti.get_notnull(),
                                      isBitwiseEq(),
                                      sub_buff_size,
                                      executor_,
                                      bucketize);
}

size_t PerfectJoinHashTable::offsetBufferOff() const noexcept {
  return 0;
}

size_t PerfectJoinHashTable::countBufferOff() const noexcept {
  return getComponentBufferSize();
}

size_t PerfectJoinHashTable::payloadBufferOff() const noexcept {
  return 2 * getComponentBufferSize();
}

size_t PerfectJoinHashTable::getComponentBufferSize() const noexcept {
  if (hash_tables_for_device_.empty()) {
    return 0;
  }
  auto hash_table = hash_tables_for_device_.front();
  if (hash_table && hash_table->getLayout() == HashType::OneToMany) {
    return hash_table->getEntryCount() * sizeof(int32_t);
  } else {
    return 0;
  }
}

HashTable* PerfectJoinHashTable::getHashTableForDevice(const size_t device_id) const {
  CHECK_LT(device_id, hash_tables_for_device_.size());
  return hash_tables_for_device_[device_id].get();
}

std::string PerfectJoinHashTable::toString(const ExecutorDeviceType device_type,
                                           const int device_id,
                                           bool raw) const {
  auto buffer = getJoinHashBuffer(device_type, device_id);
  auto buffer_size = getJoinHashBufferSize(device_type, device_id);
  auto hash_table = getHashTableForDevice(device_id);
#ifdef HAVE_CUDA
  std::unique_ptr<int8_t[]> buffer_copy;
  if (device_type == ExecutorDeviceType::GPU) {
    buffer_copy = std::make_unique<int8_t[]>(buffer_size);
    auto& data_mgr = executor_->getCatalog()->getDataMgr();
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
  return HashTable::toString("perfect",
                             getHashTypeString(hash_type_),
                             0,
                             0,
                             hash_table ? hash_table->getEntryCount() : 0,
                             ptr1,
                             ptr2,
                             ptr3,
                             ptr4,
                             buffer_size,
                             raw);
}

std::set<DecodedJoinHashBufferEntry> PerfectJoinHashTable::toSet(
    const ExecutorDeviceType device_type,
    const int device_id) const {
  auto buffer = getJoinHashBuffer(device_type, device_id);
  auto buffer_size = getJoinHashBufferSize(device_type, device_id);
  auto hash_table = getHashTableForDevice(device_id);
#ifdef HAVE_CUDA
  std::unique_ptr<int8_t[]> buffer_copy;
  if (device_type == ExecutorDeviceType::GPU) {
    buffer_copy = std::make_unique<int8_t[]>(buffer_size);
    auto& data_mgr = executor_->getCatalog()->getDataMgr();
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
  return HashTable::toSet(0,
                          0,
                          hash_table ? hash_table->getEntryCount() : 0,
                          ptr1,
                          ptr2,
                          ptr3,
                          ptr4,
                          buffer_size);
}

llvm::Value* PerfectJoinHashTable::codegenSlot(const CompilationOptions& co,
                                               const size_t index) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  using namespace std::string_literals;

  CHECK(getHashType() == HashType::OneToOne);
  const auto cols = get_cols(
      qual_bin_oper_.get(), *executor_->getCatalog(), executor_->temporary_tables_);
  auto key_col = cols.second;
  CHECK(key_col);
  auto val_col = cols.first;
  CHECK(val_col);
  CodeGenerator code_generator(executor_);
  const auto key_col_var = dynamic_cast<const Analyzer::ColumnVar*>(key_col);
  const auto val_col_var = dynamic_cast<const Analyzer::ColumnVar*>(val_col);
  if (key_col_var && val_col_var &&
      self_join_not_covered_by_left_deep_tree(
          key_col_var,
          val_col_var,
          get_max_rte_scan_table(executor_->cgen_state_->scan_idx_to_hash_pos_))) {
    throw std::runtime_error(
        "Query execution fails because the query contains not supported self-join "
        "pattern. We suspect the query requires multiple left-deep join tree due to "
        "the "
        "join condition of the self-join and is not supported for now. Please consider "
        "rewriting table order in "
        "FROM clause.");
  }
  const auto key_lvs = code_generator.codegen(key_col, true, co);
  CHECK_EQ(size_t(1), key_lvs.size());
  auto hash_ptr = codegenHashTableLoad(index);
  CHECK(hash_ptr);
  const int shard_count = shardCount();
  const auto hash_join_idx_args = getHashJoinArgs(hash_ptr, key_col, shard_count, co);

  const auto& key_col_ti = key_col->get_type_info();
  std::string fname((key_col_ti.get_type() == kDATE) ? "bucketized_hash_join_idx"s
                                                     : "hash_join_idx"s);

  if (isBitwiseEq()) {
    fname += "_bitwise";
  }
  if (shard_count) {
    fname += "_sharded";
  }

  if (!isBitwiseEq() && !key_col_ti.get_notnull()) {
    fname += "_nullable";
  }
  return executor_->cgen_state_->emitCall(fname, hash_join_idx_args);
}

const InputTableInfo& PerfectJoinHashTable::getInnerQueryInfo(
    const Analyzer::ColumnVar* inner_col) const {
  return get_inner_query_info(inner_col->get_table_id(), query_infos_);
}

const InputTableInfo& get_inner_query_info(
    const int inner_table_id,
    const std::vector<InputTableInfo>& query_infos) {
  std::optional<size_t> ti_idx;
  for (size_t i = 0; i < query_infos.size(); ++i) {
    if (inner_table_id == query_infos[i].table_id) {
      ti_idx = i;
      break;
    }
  }
  CHECK(ti_idx);
  return query_infos[*ti_idx];
}

size_t get_entries_per_device(const size_t total_entries,
                              const size_t shard_count,
                              const size_t device_count,
                              const Data_Namespace::MemoryLevel memory_level) {
  const auto entries_per_shard =
      shard_count ? (total_entries + shard_count - 1) / shard_count : total_entries;
  size_t entries_per_device = entries_per_shard;
  if (memory_level == Data_Namespace::GPU_LEVEL && shard_count) {
    const auto shards_per_device = (shard_count + device_count - 1) / device_count;
    CHECK_GT(shards_per_device, 0u);
    entries_per_device = entries_per_shard * shards_per_device;
  }
  return entries_per_device;
}

size_t PerfectJoinHashTable::shardCount() const {
  return memory_level_ == Data_Namespace::GPU_LEVEL
             ? get_shard_count(qual_bin_oper_.get(), executor_)
             : 0;
}

bool PerfectJoinHashTable::isBitwiseEq() const {
  return qual_bin_oper_->get_optype() == kBW_EQ;
}
