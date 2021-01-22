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

#include "InputMetadata.h"
#include "Execute.h"

#include "../Fragmenter/Fragmenter.h"

#include <future>

#include "Utils/Threading.h"

InputTableInfoCache::InputTableInfoCache(Executor* executor) : executor_(executor) {}

namespace {

Fragmenter_Namespace::TableInfo copy_table_info(
    const Fragmenter_Namespace::TableInfo& table_info) {
  Fragmenter_Namespace::TableInfo table_info_copy;
  table_info_copy.chunkKeyPrefix = table_info.chunkKeyPrefix;
  table_info_copy.fragments = table_info.fragments;
  table_info_copy.setPhysicalNumTuples(table_info.getPhysicalNumTuples());
  return table_info_copy;
}

Fragmenter_Namespace::TableInfo build_table_info(
    const std::vector<const TableDescriptor*>& shard_tables) {
  size_t total_number_of_tuples{0};
  Fragmenter_Namespace::TableInfo table_info_all_shards;
  for (const TableDescriptor* shard_table : shard_tables) {
    CHECK(shard_table->fragmenter);
    const auto& shard_metainfo = shard_table->fragmenter->getFragmentsForQuery();
    total_number_of_tuples += shard_metainfo.getPhysicalNumTuples();
    table_info_all_shards.fragments.reserve(table_info_all_shards.fragments.size() +
                                            shard_metainfo.fragments.size());
    table_info_all_shards.fragments.insert(table_info_all_shards.fragments.end(),
                                           shard_metainfo.fragments.begin(),
                                           shard_metainfo.fragments.end());
  }
  table_info_all_shards.setPhysicalNumTuples(total_number_of_tuples);
  return table_info_all_shards;
}

}  // namespace

Fragmenter_Namespace::TableInfo InputTableInfoCache::getTableInfo(const int table_id) {
  const auto it = cache_.find(table_id);
  if (it != cache_.end()) {
    const auto& table_info = it->second;
    return copy_table_info(table_info);
  }
  const auto cat = executor_->getCatalog();
  CHECK(cat);
  const auto td = cat->getMetadataForTable(table_id);
  CHECK(td);
  const auto shard_tables = cat->getPhysicalTablesDescriptors(td);
  auto table_info = build_table_info(shard_tables);
  auto it_ok = cache_.emplace(table_id, copy_table_info(table_info));
  CHECK(it_ok.second);
  return copy_table_info(table_info);
}

void InputTableInfoCache::clear() {
  decltype(cache_)().swap(cache_);
}

namespace {

bool uses_int_meta(const SQLTypeInfo& col_ti) {
  return col_ti.is_integer() || col_ti.is_decimal() || col_ti.is_time() ||
         col_ti.is_boolean() ||
         (col_ti.is_string() && col_ti.get_compression() == kENCODING_DICT);
}

Fragmenter_Namespace::TableInfo synthesize_table_info(const ResultSetPtr& rows) {
  std::vector<Fragmenter_Namespace::FragmentInfo> result;
  if (rows) {
    result.resize(1);
    auto& fragment = result.front();
    fragment.fragmentId = 0;
    fragment.deviceIds.resize(3);
    fragment.resultSet = rows.get();
    fragment.resultSetMutex.reset(new std::mutex());
  }
  Fragmenter_Namespace::TableInfo table_info;
  table_info.fragments = result;
  return table_info;
}

void collect_table_infos(std::vector<InputTableInfo>& table_infos,
                         const std::vector<InputDescriptor>& input_descs,
                         Executor* executor) {
  const auto temporary_tables = executor->getTemporaryTables();
  const auto cat = executor->getCatalog();
  CHECK(cat);
  std::unordered_map<int, size_t> info_cache;
  for (const auto& input_desc : input_descs) {
    const auto table_id = input_desc.getTableId();
    const auto cached_index_it = info_cache.find(table_id);
    if (cached_index_it != info_cache.end()) {
      CHECK_LT(cached_index_it->second, table_infos.size());
      table_infos.push_back(
          {table_id, copy_table_info(table_infos[cached_index_it->second].info)});
      continue;
    }
    if (input_desc.getSourceType() == InputSourceType::RESULT) {
      CHECK_LT(table_id, 0);
      CHECK(temporary_tables);
      const auto it = temporary_tables->find(table_id);
      LOG_IF(FATAL, it == temporary_tables->end())
          << "Failed to find previous query result for node " << -table_id;
      table_infos.push_back({table_id, synthesize_table_info(it->second)});
    } else {
      CHECK(input_desc.getSourceType() == InputSourceType::TABLE);
      table_infos.push_back({table_id, executor->getTableInfo(table_id)});
    }
    CHECK(!table_infos.empty());
    info_cache.insert(std::make_pair(table_id, table_infos.size() - 1));
  }
}

}  // namespace

ChunkMetadataMap synthesize_metadata(const ResultSet* rows) {
  rows->moveToBegin();
  std::vector<std::vector<std::unique_ptr<Encoder>>> dummy_encoders;
  const size_t worker_count =
      result_set::use_parallel_algorithms(*rows) ? cpu_threads() : 1;
  for (size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
    dummy_encoders.emplace_back();
    for (size_t i = 0; i < rows->colCount(); ++i) {
      const auto& col_ti = rows->getColType(i);
      dummy_encoders.back().emplace_back(Encoder::Create(nullptr, col_ti));
    }
  }
  const auto do_work = [rows](const std::vector<TargetValue>& crt_row,
                              std::vector<std::unique_ptr<Encoder>>& dummy_encoders) {
    for (size_t i = 0; i < rows->colCount(); ++i) {
      const auto& col_ti = rows->getColType(i);
      const auto& col_val = crt_row[i];
      const auto scalar_col_val = boost::get<ScalarTargetValue>(&col_val);
      CHECK(scalar_col_val);
      if (uses_int_meta(col_ti)) {
        const auto i64_p = boost::get<int64_t>(scalar_col_val);
        CHECK(i64_p);
        dummy_encoders[i]->updateStats(*i64_p, *i64_p == inline_int_null_val(col_ti));
      } else if (col_ti.is_fp()) {
        switch (col_ti.get_type()) {
          case kFLOAT: {
            const auto float_p = boost::get<float>(scalar_col_val);
            CHECK(float_p);
            dummy_encoders[i]->updateStats(*float_p,
                                           *float_p == inline_fp_null_val(col_ti));
            break;
          }
          case kDOUBLE: {
            const auto double_p = boost::get<double>(scalar_col_val);
            CHECK(double_p);
            dummy_encoders[i]->updateStats(*double_p,
                                           *double_p == inline_fp_null_val(col_ti));
            break;
          }
          default:
            CHECK(false);
        }
      } else {
        throw std::runtime_error(col_ti.get_type_name() +
                                 " is not supported in temporary table.");
      }
    }
  };
  if (result_set::use_parallel_algorithms(*rows)) {
    const size_t worker_count = cpu_threads();
    std::vector<utils::future<void>> compute_stats_threads;
    const auto entry_count = rows->entryCount();
    for (size_t i = 0,
                start_entry = 0,
                stride = (entry_count + worker_count - 1) / worker_count;
         i < worker_count && start_entry < entry_count;
         ++i, start_entry += stride) {
      const auto end_entry = std::min(start_entry + stride, entry_count);
      compute_stats_threads.push_back(utils::async(
          [rows, &do_work, &dummy_encoders](
              const size_t start, const size_t end, const size_t worker_idx) {
            for (size_t i = start; i < end; ++i) {
              const auto crt_row = rows->getRowAtNoTranslations(i);
              if (!crt_row.empty()) {
                do_work(crt_row, dummy_encoders[worker_idx]);
              }
            }
          },
          start_entry,
          end_entry,
          i));
    }
    for (auto& child : compute_stats_threads) {
      child.wait();
    }
  } else {
    while (true) {
      auto crt_row = rows->getNextRow(false, false);
      if (crt_row.empty()) {
        break;
      }
      do_work(crt_row, dummy_encoders[0]);
    }
    rows->moveToBegin();
  }
  ChunkMetadataMap metadata_map;
  for (size_t worker_idx = 1; worker_idx < worker_count; ++worker_idx) {
    CHECK_LT(worker_idx, dummy_encoders.size());
    const auto& worker_encoders = dummy_encoders[worker_idx];
    for (size_t i = 0; i < rows->colCount(); ++i) {
      dummy_encoders[0][i]->reduceStats(*worker_encoders[i]);
    }
  }
  for (size_t i = 0; i < rows->colCount(); ++i) {
    const auto it_ok =
        metadata_map.emplace(i, dummy_encoders[0][i]->getMetadata(rows->getColType(i)));
    CHECK(it_ok.second);
  }
  return metadata_map;
}

size_t get_frag_count_of_table(const int table_id, Executor* executor) {
  const auto temporary_tables = executor->getTemporaryTables();
  CHECK(temporary_tables);
  auto it = temporary_tables->find(table_id);
  if (it != temporary_tables->end()) {
    CHECK_GE(int(0), table_id);
    return size_t(1);
  } else {
    const auto table_info = executor->getTableInfo(table_id);
    return table_info.fragments.size();
  }
}

std::vector<InputTableInfo> get_table_infos(
    const std::vector<InputDescriptor>& input_descs,
    Executor* executor) {
  std::vector<InputTableInfo> table_infos;
  collect_table_infos(table_infos, input_descs, executor);
  return table_infos;
}

std::vector<InputTableInfo> get_table_infos(const RelAlgExecutionUnit& ra_exe_unit,
                                            Executor* executor) {
  INJECT_TIMER(get_table_infos);
  std::vector<InputTableInfo> table_infos;
  collect_table_infos(table_infos, ra_exe_unit.input_descs, executor);
  return table_infos;
}

const ChunkMetadataMap& Fragmenter_Namespace::FragmentInfo::getChunkMetadataMap() const {
  if (resultSet && !synthesizedMetadataIsValid) {
    chunkMetadataMap = synthesize_metadata(resultSet);
    synthesizedMetadataIsValid = true;
  }
  return chunkMetadataMap;
}

size_t Fragmenter_Namespace::FragmentInfo::getNumTuples() const {
  std::unique_ptr<std::lock_guard<std::mutex>> lock;
  if (resultSetMutex) {
    lock.reset(new std::lock_guard<std::mutex>(*resultSetMutex));
  }
  CHECK_EQ(!!resultSet, !!resultSetMutex);
  if (resultSet && !synthesizedNumTuplesIsValid) {
    numTuples = resultSet->rowCount();
    synthesizedNumTuplesIsValid = true;
  }
  return numTuples;
}

size_t Fragmenter_Namespace::TableInfo::getNumTuples() const {
  if (!fragments.empty() && fragments.front().resultSet) {
    return fragments.front().getNumTuples();
  }
  return numTuples;
}

size_t Fragmenter_Namespace::TableInfo::getNumTuplesUpperBound() const {
  if (!fragments.empty() && fragments.front().resultSet) {
    return fragments.front().resultSet->entryCount();
  }
  return numTuples;
}

size_t Fragmenter_Namespace::TableInfo::getFragmentNumTuplesUpperBound() const {
  if (!fragments.empty() && fragments.front().resultSet) {
    return fragments.front().resultSet->entryCount();
  }
  size_t fragment_num_tupples_upper_bound = 0;
  for (const auto& fragment : fragments) {
    fragment_num_tupples_upper_bound =
        std::max(fragment.getNumTuples(), fragment_num_tupples_upper_bound);
  }
  return fragment_num_tupples_upper_bound;
}
