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

#include <future>

#include "Fragmenter/InsertOrderFragmenter.h"
#include "LockMgr/LockMgr.h"
#include "QueryEngine/CodeGenerator.h"
#include "QueryEngine/ExternalCacheInvalidators.h"
#include "QueryEngine/InputMetadata.h"
#include "QueryEngine/TargetMetaInfo.h"
#include "Shared/UpdelRoll.h"
#include "Shared/likely.h"
#include "Shared/thread_count.h"

#include <future>

#include "Utils/Threading.h"

template <typename EXECUTOR_TRAITS, typename FRAGMENT_UPDATER = UpdateLogForFragment>
class StorageIOFacility {
 public:
  using ExecutorType = typename EXECUTOR_TRAITS::ExecutorType;
  using CatalogType = typename EXECUTOR_TRAITS::CatalogType;
  using FragmentUpdaterType = FRAGMENT_UPDATER;
  using UpdateCallback = typename FragmentUpdaterType::Callback;

  using TableDescriptorType = typename EXECUTOR_TRAITS::TableDescriptorType;
  using DeleteVictimOffsetList = std::vector<uint64_t>;
  using UpdateTargetOffsetList = std::vector<uint64_t>;
  using UpdateTargetTypeList = std::vector<TargetMetaInfo>;
  using UpdateTargetColumnNamesList = std::vector<std::string>;

  using FragmenterType = Fragmenter_Namespace::InsertOrderFragmenter;
  using TransactionLog = typename FragmenterType::ModifyTransactionTracker;
  using TransactionLogPtr = std::unique_ptr<TransactionLog>;
  using ColumnValidationFunction = std::function<bool(std::string const&)>;

  class TransactionParameters {
   public:
    typename StorageIOFacility::TransactionLog& getTransactionTracker() {
      return transaction_tracker_;
    }
    void finalizeTransaction() { transaction_tracker_.commitUpdate(); }

   private:
    typename StorageIOFacility::TransactionLog transaction_tracker_;
  };

  struct DeleteTransactionParameters : public TransactionParameters {
   public:
    DeleteTransactionParameters(const bool table_is_temporary)
        : table_is_temporary_(table_is_temporary) {}

    auto tableIsTemporary() const { return table_is_temporary_; }

   private:
    DeleteTransactionParameters(DeleteTransactionParameters const& other) = delete;
    DeleteTransactionParameters& operator=(DeleteTransactionParameters const& other) =
        delete;

    bool table_is_temporary_;
  };

  class UpdateTransactionParameters : public TransactionParameters {
   public:
    UpdateTransactionParameters(TableDescriptorType const* table_desc,
                                UpdateTargetColumnNamesList const& update_column_names,
                                UpdateTargetTypeList const& target_types,
                                bool varlen_update_required)
        : table_descriptor_(table_desc)
        , update_column_names_(update_column_names)
        , targets_meta_(target_types)
        , varlen_update_required_(varlen_update_required)
        , table_is_temporary_(table_is_temporary(table_descriptor_)) {}

    auto getUpdateColumnCount() const { return update_column_names_.size(); }
    auto const* getTableDescriptor() const { return table_descriptor_; }
    auto const& getTargetsMetaInfo() const { return targets_meta_; }
    auto getTargetsMetaInfoSize() const { return targets_meta_.size(); }
    auto const& getUpdateColumnNames() const { return update_column_names_; }
    auto isVarlenUpdateRequired() const { return varlen_update_required_; }
    auto tableIsTemporary() const { return table_is_temporary_; }

   private:
    UpdateTransactionParameters(UpdateTransactionParameters const& other) = delete;
    UpdateTransactionParameters& operator=(UpdateTransactionParameters const& other) =
        delete;

    TableDescriptorType const* table_descriptor_;
    UpdateTargetColumnNamesList update_column_names_;
    UpdateTargetTypeList const& targets_meta_;
    bool varlen_update_required_ = false;
    bool table_is_temporary_;
  };

  StorageIOFacility(ExecutorType* executor, CatalogType const& catalog)
      : executor_(executor), catalog_(catalog) {}

  UpdateCallback yieldUpdateCallback(UpdateTransactionParameters& update_parameters);
  UpdateCallback yieldDeleteCallback(DeleteTransactionParameters& delete_parameters);

 private:
  int normalized_cpu_threads() const { return cpu_threads() / 2; }
  static std::unique_ptr<int8_t[]> getRsBufferNoPadding(const ResultSet* rs,
                                                        size_t col_idx,
                                                        const SQLTypeInfo& column_type,
                                                        size_t row_count);

  ExecutorType* executor_;
  CatalogType const& catalog_;
};

template <typename EXECUTOR_TRAITS, typename FRAGMENT_UPDATER>
typename StorageIOFacility<EXECUTOR_TRAITS, FRAGMENT_UPDATER>::UpdateCallback
StorageIOFacility<EXECUTOR_TRAITS, FRAGMENT_UPDATER>::yieldUpdateCallback(
    UpdateTransactionParameters& update_parameters) {
  using OffsetVector = std::vector<uint64_t>;
  using ScalarTargetValueVector = std::vector<ScalarTargetValue>;
  using RowProcessingFuturesVector = std::vector<std::future<uint64_t>>;

  if (update_parameters.isVarlenUpdateRequired()) {
    auto callback = [this,
                     &update_parameters](FragmentUpdaterType const& update_log) -> void {
      std::vector<const ColumnDescriptor*> columnDescriptors;
      std::vector<TargetMetaInfo> sourceMetaInfos;

      for (size_t idx = 0; idx < update_parameters.getUpdateColumnNames().size(); idx++) {
        auto& column_name = update_parameters.getUpdateColumnNames()[idx];
        auto target_column =
            catalog_.getMetadataForColumn(update_log.getPhysicalTableId(), column_name);
        columnDescriptors.push_back(target_column);
        sourceMetaInfos.push_back(update_parameters.getTargetsMetaInfo()[idx]);
      }

      auto td = catalog_.getMetadataForTable(update_log.getPhysicalTableId());
      auto* fragmenter = td->fragmenter.get();
      CHECK(fragmenter);

      fragmenter->updateColumns(
          &catalog_,
          td,
          update_log.getFragmentId(),
          sourceMetaInfos,
          columnDescriptors,
          update_log,
          update_parameters.getUpdateColumnCount(),  // last column of result set
          Data_Namespace::MemoryLevel::CPU_LEVEL,
          update_parameters.getTransactionTracker(),
          executor_);
    };
    return callback;
  } else if (update_parameters.tableIsTemporary()) {
    auto callback = [this,
                     &update_parameters](FragmentUpdaterType const& update_log) -> void {
      auto rs = update_log.getResultSet();
      CHECK(rs->didOutputColumnar());
      CHECK(rs->isDirectColumnarConversionPossible());
      CHECK_EQ(update_parameters.getUpdateColumnCount(), size_t(1));
      CHECK_EQ(rs->colCount(), size_t(1));

      // Temporary table updates require the full projected column
      CHECK_EQ(rs->rowCount(), update_log.getRowCount());

      ChunkKey chunk_key_prefix{catalog_.getCurrentDB().dbId,
                                update_parameters.getTableDescriptor()->tableId};
      const auto table_lock =
          lockmgr::TableDataLockMgr::getWriteLockForTable(chunk_key_prefix);

      auto& fragment_info = update_log.getFragmentInfo();
      const auto td = catalog_.getMetadataForTable(update_log.getPhysicalTableId());
      CHECK(td);
      const auto cd = catalog_.getMetadataForColumn(
          td->tableId, update_parameters.getUpdateColumnNames().front());
      CHECK(cd);
      auto chunk_metadata =
          fragment_info.getChunkMetadataMapPhysical().find(cd->columnId);
      CHECK(chunk_metadata != fragment_info.getChunkMetadataMapPhysical().end());
      ChunkKey chunk_key{catalog_.getCurrentDB().dbId,
                         td->tableId,
                         cd->columnId,
                         fragment_info.fragmentId};
      auto chunk = Chunk_NS::Chunk::getChunk(cd,
                                             &catalog_.getDataMgr(),
                                             chunk_key,
                                             Data_Namespace::MemoryLevel::CPU_LEVEL,
                                             0,
                                             chunk_metadata->second->numBytes,
                                             chunk_metadata->second->numElements);
      CHECK(chunk);
      auto chunk_buffer = chunk->getBuffer();
      CHECK(chunk_buffer);

      auto encoder = chunk_buffer->getEncoder();
      CHECK(encoder);

      auto owned_buffer =
          StorageIOFacility<EXECUTOR_TRAITS, FRAGMENT_UPDATER>::getRsBufferNoPadding(
              rs.get(), 0, cd->columnType, rs->rowCount());
      auto buffer = reinterpret_cast<int8_t*>(owned_buffer.get());

      const auto new_chunk_metadata =
          encoder->appendData(buffer, rs->rowCount(), cd->columnType, false, 0);
      CHECK(new_chunk_metadata);

      auto fragmenter = td->fragmenter.get();
      CHECK(fragmenter);

      // The fragmenter copy of the fragment info differs from the copy used by the query
      // engine. Update metadata in the fragmenter directly.
      auto fragment = fragmenter->getFragmentInfo(fragment_info.fragmentId);
      // TODO: we may want to put this directly in the fragmenter so we are under the
      // fragmenter lock. But, concurrent queries on the same fragmenter should not be
      // allowed in this path.

      fragment->setChunkMetadata(cd->columnId, new_chunk_metadata);
      fragment->shadowChunkMetadataMap =
          fragment->getChunkMetadataMap();  // TODO(adb): needed?

      auto& data_mgr = catalog_.getDataMgr();
      if (data_mgr.gpusPresent()) {
        // flush any GPU copies of the updated chunk
        data_mgr.deleteChunksWithPrefix(chunk_key,
                                        Data_Namespace::MemoryLevel::GPU_LEVEL);
      }
    };
    return callback;
  } else {
    auto callback = [this,
                     &update_parameters](FragmentUpdaterType const& update_log) -> void {
      auto entries_per_column = update_log.getEntryCount();
      auto rows_per_column = update_log.getRowCount();
      if (rows_per_column == 0) {
        return;
      }

      OffsetVector column_offsets(rows_per_column);
      ScalarTargetValueVector scalar_target_values(rows_per_column);

      auto complete_entry_block_size = entries_per_column / normalized_cpu_threads();
      auto partial_row_block_size = entries_per_column % normalized_cpu_threads();
      auto usable_threads = normalized_cpu_threads();
      if (UNLIKELY(rows_per_column < (unsigned)normalized_cpu_threads())) {
        complete_entry_block_size = entries_per_column;
        partial_row_block_size = 0;
        usable_threads = 1;
      }

      std::atomic<size_t> row_idx{0};

      auto process_rows =
          [&update_parameters, &column_offsets, &scalar_target_values, &row_idx](
              auto get_entry_at_func,
              uint64_t column_index,
              uint64_t entry_start,
              uint64_t entry_count) -> uint64_t {
        uint64_t entries_processed = 0;
        for (uint64_t entry_index = entry_start;
             entry_index < (entry_start + entry_count);
             entry_index++) {
          const auto& row = get_entry_at_func(entry_index);
          if (row.empty()) {
            continue;
          }

          entries_processed++;
          size_t row_index = row_idx.fetch_add(1);

          CHECK(row.size() == update_parameters.getUpdateColumnCount() + 1);

          auto terminal_column_iter = std::prev(row.end());
          const auto frag_offset_scalar_tv =
              boost::get<ScalarTargetValue>(&*terminal_column_iter);
          CHECK(frag_offset_scalar_tv);

          column_offsets[row_index] =
              static_cast<uint64_t>(*(boost::get<int64_t>(frag_offset_scalar_tv)));
          scalar_target_values[row_index] =
              boost::get<ScalarTargetValue>(row[column_index]);
        }
        return entries_processed;
      };

      auto get_row_index =
          [complete_entry_block_size](uint64_t thread_index) -> uint64_t {
        return (thread_index * complete_entry_block_size);
      };

      // Iterate over each column
      for (decltype(update_parameters.getUpdateColumnCount()) column_index = 0;
           column_index < update_parameters.getUpdateColumnCount();
           column_index++) {
        row_idx = 0;
        RowProcessingFuturesVector entry_processing_futures;
        entry_processing_futures.reserve(usable_threads);

        auto get_entry_at_func = [&update_log, &column_index](const size_t entry_index) {
          if (UNLIKELY(update_log.getColumnType(column_index).is_string())) {
            return update_log.getTranslatedEntryAt(entry_index);
          } else {
            return update_log.getEntryAt(entry_index);
          }
        };

        for (unsigned i = 0; i < static_cast<unsigned>(usable_threads); i++) {
          entry_processing_futures.emplace_back(
              std::async(std::launch::async,
                         std::forward<decltype(process_rows)>(process_rows),
                         get_entry_at_func,
                         column_index,
                         get_row_index(i),
                         complete_entry_block_size));
        }
        if (partial_row_block_size) {
          entry_processing_futures.emplace_back(
              std::async(std::launch::async,
                         std::forward<decltype(process_rows)>(process_rows),
                         get_entry_at_func,
                         column_index,
                         get_row_index(usable_threads),
                         partial_row_block_size));
        }

        uint64_t entries_processed(0);
        for (auto& t : entry_processing_futures) {
          t.wait();
          entries_processed += t.get();
        }

        CHECK(row_idx == rows_per_column);

        const auto table_id = update_log.getPhysicalTableId();
        auto const* table_descriptor =
            catalog_.getMetadataForTable(update_log.getPhysicalTableId());
        CHECK(table_descriptor);
        const auto fragmenter = table_descriptor->fragmenter;
        CHECK(fragmenter);
        auto const* target_column = catalog_.getMetadataForColumn(
            table_id, update_parameters.getUpdateColumnNames()[column_index]);

        fragmenter->updateColumn(&catalog_,
                                 table_descriptor,
                                 target_column,
                                 update_log.getFragmentId(),
                                 column_offsets,
                                 scalar_target_values,
                                 update_log.getColumnType(column_index),
                                 Data_Namespace::MemoryLevel::CPU_LEVEL,
                                 update_parameters.getTransactionTracker());
      }
    };
    return callback;
  }
}

template <typename EXECUTOR_TRAITS, typename FRAGMENT_UPDATER>
typename StorageIOFacility<EXECUTOR_TRAITS, FRAGMENT_UPDATER>::UpdateCallback
StorageIOFacility<EXECUTOR_TRAITS, FRAGMENT_UPDATER>::yieldDeleteCallback(
    DeleteTransactionParameters& delete_parameters) {
  using RowProcessingFuturesVector = std::vector<utils::future<uint64_t>>;

  if (delete_parameters.tableIsTemporary()) {
    auto callback = [this](FragmentUpdaterType const& update_log) -> void {
      auto rs = update_log.getResultSet();
      CHECK(rs->didOutputColumnar());
      CHECK(rs->isDirectColumnarConversionPossible());
      CHECK_EQ(rs->colCount(), size_t(1));

      // Temporary table updates require the full projected column
      CHECK_EQ(rs->rowCount(), update_log.getRowCount());

      ChunkKey chunk_key_prefix{catalog_.getCurrentDB().dbId,
                                update_log.getPhysicalTableId()};
      const auto table_lock =
          lockmgr::TableDataLockMgr::getWriteLockForTable(chunk_key_prefix);

      auto& fragment_info = update_log.getFragmentInfo();
      const auto td = catalog_.getMetadataForTable(update_log.getPhysicalTableId());
      CHECK(td);
      const auto cd = catalog_.getDeletedColumn(td);
      CHECK(cd);
      CHECK(cd->columnType.get_type() == kBOOLEAN);
      auto chunk_metadata =
          fragment_info.getChunkMetadataMapPhysical().find(cd->columnId);
      CHECK(chunk_metadata != fragment_info.getChunkMetadataMapPhysical().end());
      ChunkKey chunk_key{catalog_.getCurrentDB().dbId,
                         td->tableId,
                         cd->columnId,
                         fragment_info.fragmentId};
      auto chunk = Chunk_NS::Chunk::getChunk(cd,
                                             &catalog_.getDataMgr(),
                                             chunk_key,
                                             Data_Namespace::MemoryLevel::CPU_LEVEL,
                                             0,
                                             chunk_metadata->second->numBytes,
                                             chunk_metadata->second->numElements);
      CHECK(chunk);
      auto chunk_buffer = chunk->getBuffer();
      CHECK(chunk_buffer);

      auto encoder = chunk_buffer->getEncoder();
      CHECK(encoder);

      auto owned_buffer =
          StorageIOFacility<EXECUTOR_TRAITS, FRAGMENT_UPDATER>::getRsBufferNoPadding(
              rs.get(), 0, cd->columnType, rs->rowCount());
      auto buffer = reinterpret_cast<int8_t*>(owned_buffer.get());

      const auto new_chunk_metadata =
          encoder->appendData(buffer, rs->rowCount(), cd->columnType, false, 0);

      auto fragmenter = td->fragmenter.get();
      CHECK(fragmenter);

      // The fragmenter copy of the fragment info differs from the copy used by the query
      // engine. Update metadata in the fragmenter directly.
      auto fragment = fragmenter->getFragmentInfo(fragment_info.fragmentId);
      // TODO: we may want to put this directly in the fragmenter so we are under the
      // fragmenter lock. But, concurrent queries on the same fragmenter should not be
      // allowed in this path.

      fragment->setChunkMetadata(cd->columnId, new_chunk_metadata);
      fragment->shadowChunkMetadataMap =
          fragment->getChunkMetadataMap();  // TODO(adb): needed?

      auto& data_mgr = catalog_.getDataMgr();
      if (data_mgr.gpusPresent()) {
        // flush any GPU copies of the updated chunk
        data_mgr.deleteChunksWithPrefix(chunk_key,
                                        Data_Namespace::MemoryLevel::GPU_LEVEL);
      }
    };
    return callback;
  } else {
    auto callback = [this,
                     &delete_parameters](FragmentUpdaterType const& update_log) -> void {
      auto entries_per_column = update_log.getEntryCount();
      auto rows_per_column = update_log.getRowCount();
      if (rows_per_column == 0) {
        return;
      }
      DeleteVictimOffsetList victim_offsets(rows_per_column);

      auto complete_row_block_size = entries_per_column / normalized_cpu_threads();
      auto partial_row_block_size = entries_per_column % normalized_cpu_threads();
      auto usable_threads = normalized_cpu_threads();

      if (UNLIKELY(rows_per_column < (unsigned)normalized_cpu_threads())) {
        complete_row_block_size = rows_per_column;
        partial_row_block_size = 0;
        usable_threads = 1;
      }

      std::atomic<size_t> row_idx{0};

      auto process_rows = [&update_log, &victim_offsets, &row_idx](
                              uint64_t entry_start, uint64_t entry_count) -> uint64_t {
        uint64_t entries_processed = 0;

        for (uint64_t entry_index = entry_start;
             entry_index < (entry_start + entry_count);
             entry_index++) {
          auto const row(update_log.getEntryAt(entry_index));

          if (row.empty()) {
            continue;
          }

          entries_processed++;
          size_t row_index = row_idx.fetch_add(1);

          auto terminal_column_iter = std::prev(row.end());
          const auto scalar_tv = boost::get<ScalarTargetValue>(&*terminal_column_iter);
          CHECK(scalar_tv);

          uint64_t fragment_offset =
              static_cast<uint64_t>(*(boost::get<int64_t>(scalar_tv)));
          victim_offsets[row_index] = fragment_offset;
        }
        return entries_processed;
      };

      auto get_row_index = [complete_row_block_size](uint64_t thread_index) -> uint64_t {
        return thread_index * complete_row_block_size;
      };

      RowProcessingFuturesVector row_processing_futures;
      row_processing_futures.reserve(usable_threads);

      for (unsigned i = 0; i < (unsigned)usable_threads; i++) {
        row_processing_futures.emplace_back(
            utils::async(std::forward<decltype(process_rows)>(process_rows),
                         get_row_index(i),
                         complete_row_block_size));
      }
      if (partial_row_block_size) {
        row_processing_futures.emplace_back(
            utils::async(std::forward<decltype(process_rows)>(process_rows),
                         get_row_index(usable_threads),
                         partial_row_block_size));
      }

      uint64_t rows_processed(0);
      for (auto& t : row_processing_futures) {
        t.wait();
        rows_processed += t.get();
      }

      auto const* table_descriptor =
          catalog_.getMetadataForTable(update_log.getPhysicalTableId());
      CHECK(!table_is_temporary(table_descriptor));
      auto* fragmenter = table_descriptor->fragmenter.get();
      CHECK(fragmenter);

      auto const* deleted_column_desc = catalog_.getDeletedColumn(table_descriptor);
      CHECK(deleted_column_desc);
      fragmenter->updateColumn(&catalog_,
                               table_descriptor,
                               deleted_column_desc,
                               update_log.getFragmentId(),
                               victim_offsets,
                               ScalarTargetValue(int64_t(1L)),
                               update_log.getColumnType(0),
                               Data_Namespace::MemoryLevel::CPU_LEVEL,
                               delete_parameters.getTransactionTracker());
    };
    return callback;
  }
}

template <typename EXECUTOR_TRAITS, typename FRAGMENT_UPDATER>
std::unique_ptr<int8_t[]>
StorageIOFacility<EXECUTOR_TRAITS, FRAGMENT_UPDATER>::getRsBufferNoPadding(
    const ResultSet* rs,
    size_t col_idx,
    const SQLTypeInfo& column_type,
    size_t row_count) {
  const auto padded_size = rs->getPaddedSlotWidthBytes(col_idx);
  const auto type_size = column_type.is_dict_encoded_string()
                             ? column_type.get_size()
                             : column_type.get_logical_size();

  auto rs_buffer_size = padded_size * row_count;
  auto rs_buffer = std::make_unique<int8_t[]>(rs_buffer_size);
  rs->copyColumnIntoBuffer(col_idx, rs_buffer.get(), rs_buffer_size);

  if (type_size < padded_size) {
    // else we're going to remove padding and we do it inplace in the same buffer
    // we can do updates inplace in the same buffer because type_size < padded_size
    // for some types, like kFLOAT, simple memcpy is not enough
    auto src_ptr = rs_buffer.get();
    auto dst_ptr = rs_buffer.get();
    if (column_type.is_fp()) {
      CHECK(column_type.get_type() == kFLOAT);
      CHECK(padded_size == sizeof(double));
      for (size_t i = 0; i < row_count; i++) {
        const auto old_val = *reinterpret_cast<double*>(may_alias_ptr(src_ptr));
        auto new_val = static_cast<float>(old_val);
        std::memcpy(dst_ptr, &new_val, type_size);
        dst_ptr += type_size;
        src_ptr += padded_size;
      }
    } else {
      // otherwise just take first type_size bytes from the padded value
      for (size_t i = 0; i < row_count; i++) {
        std::memcpy(dst_ptr, src_ptr, type_size);
        dst_ptr += type_size;
        src_ptr += padded_size;
      }
    }
  }
  return rs_buffer;
}
