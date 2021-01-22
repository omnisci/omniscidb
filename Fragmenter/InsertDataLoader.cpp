/*
 * Copyright 2019, OmniSci, Inc.
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

#include <algorithm>
#include <numeric>
#include <vector>

#include "../Shared/shard_key.h"
#include "InsertDataLoader.h"
#include "TargetValueConvertersFactories.h"

#include "Utils/Threading.h"

namespace Fragmenter_Namespace {

struct ShardDataOwner {
  std::vector<std::vector<uint8_t>> rawData;
  std::vector<std::vector<std::string>> stringData;
  std::vector<std::vector<ArrayDatum>> arrayData;
};

template <typename SRC>
std::vector<std::vector<size_t>> computeRowIndicesOfShards(size_t shardCount,
                                                           size_t leafCount,
                                                           size_t rowCount,
                                                           SRC* src) {
  const auto numShardTables = shardCount * leafCount;

  std::vector<std::vector<size_t>> rowIndicesOfShards(numShardTables);

  for (size_t row = 0; row < rowCount; row++) {
    // expecting unsigned data
    // thus, no need for double remainder
    auto shardId = (std::is_unsigned<SRC>::value)
                       ? src[row] % numShardTables
                       : SHARD_FOR_KEY(src[row], numShardTables);
    rowIndicesOfShards[shardId].push_back(row);
  }

  return rowIndicesOfShards;
}

template <typename T>
size_t indexOf(std::vector<T>& vec, T val) {
  typename std::vector<T>::iterator it = std::find(vec.begin(), vec.end(), val);
  CHECK(it != vec.end());
  return std::distance(vec.begin(), it);
}

bool isStringVectorData(const ColumnDescriptor* cd) {
  return (cd->columnType.is_geometry()) ||
         (cd->columnType.is_string() &&
          cd->columnType.get_compression() == kENCODING_NONE);
}

bool isDatumVectorData(const ColumnDescriptor* cd) {
  return cd->columnType.is_array();
}

size_t sizeOfRawColumn(const Catalog_Namespace::Catalog& cat,
                       const ColumnDescriptor* cd) {
  switch (cd->columnType.get_type()) {
    case kPOINT:
    case kLINESTRING:
    case kPOLYGON:
    case kMULTIPOLYGON:
    case kARRAY:
      throw std::runtime_error("geo and array columns have variable length elements");
    case kBOOLEAN:
    case kTINYINT:
    case kSMALLINT:
    case kINT:
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL:
    case kFLOAT:
    case kDOUBLE:
    case kTIMESTAMP:
    case kTIME:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
    case kDATE:
      return cd->columnType.get_logical_size();
    case kTEXT:
    case kVARCHAR:
    case kCHAR:
      if (cd->columnType.get_compression() == kENCODING_NONE) {
        throw std::runtime_error(
            "non encoded string columns have variable length elements");
      }
      return cd->columnType.get_size();
    default:
      throw std::runtime_error("not supported column type: " + cd->columnName + " (" +
                               cd->columnType.get_type_name() + ")");
  }
}

std::vector<std::vector<size_t>> computeRowIndicesOfShards(
    const Catalog_Namespace::Catalog& cat,
    size_t leafCount,
    InsertData& insert_data) {
  const auto* td = cat.getMetadataForTable(insert_data.tableId);
  const auto* shard_cd = cat.getShardColumnMetadataForTable(td);
  auto shardDataBlockIndex = indexOf(insert_data.columnIds, shard_cd->columnId);
  DataBlockPtr& shardDataBlock = insert_data.data[shardDataBlockIndex];
  auto rowCount = insert_data.numRows;
  auto shardCount = td->nShards;

  CHECK(!isStringVectorData(shard_cd));
  CHECK(!isDatumVectorData(shard_cd));

  switch (sizeOfRawColumn(cat, shard_cd)) {
    case 1:
      return computeRowIndicesOfShards(
          shardCount,
          leafCount,
          rowCount,
          reinterpret_cast<uint8_t*>(shardDataBlock.numbersPtr));
    case 2:
      return computeRowIndicesOfShards(
          shardCount,
          leafCount,
          rowCount,
          reinterpret_cast<uint16_t*>(shardDataBlock.numbersPtr));
    case 4:
      return computeRowIndicesOfShards(
          shardCount,
          leafCount,
          rowCount,
          reinterpret_cast<uint32_t*>(shardDataBlock.numbersPtr));
    case 8:
      return computeRowIndicesOfShards(
          shardCount,
          leafCount,
          rowCount,
          reinterpret_cast<uint64_t*>(shardDataBlock.numbersPtr));
  }
  throw std::runtime_error("Unexpected data block element size");
}

template <typename T>
void copyColumnDataOfShard(const std::vector<size_t>& rowIndices, T* src, T* dst) {
  for (size_t row = 0; row < rowIndices.size(); row++) {
    auto srcRowIndex = rowIndices[row];
    dst[row] = src[srcRowIndex];
  }
}

struct BlockWithColumnId {
  int columnId;
  DataBlockPtr block;
};

BlockWithColumnId copyColumnDataOfShard(const Catalog_Namespace::Catalog& cat,
                                        ShardDataOwner& dataOwner,
                                        const std::vector<size_t>& rowIndices,
                                        const ColumnDescriptor* pCol,
                                        size_t columnIndex,
                                        DataBlockPtr dataBlock) {
  DataBlockPtr ret;
  if (isStringVectorData(pCol)) {
    auto& data = dataOwner.stringData[columnIndex];
    data.resize(rowIndices.size());
    copyColumnDataOfShard(rowIndices, &(*(dataBlock.stringsPtr))[0], &data[0]);
    ret.stringsPtr = &data;

  } else if (isDatumVectorData(pCol)) {
    auto& data = dataOwner.arrayData[columnIndex];
    data.resize(rowIndices.size());
    copyColumnDataOfShard(rowIndices, &(*(dataBlock.arraysPtr))[0], &data[0]);
    ret.arraysPtr = &data;

  } else {
    auto rawArrayElementSize = sizeOfRawColumn(cat, pCol);
    auto& data = dataOwner.rawData[columnIndex];
    data.resize(rowIndices.size() * rawArrayElementSize);

    switch (rawArrayElementSize) {
      case 1: {
        copyColumnDataOfShard(rowIndices,
                              reinterpret_cast<uint8_t*>(dataBlock.numbersPtr),
                              reinterpret_cast<uint8_t*>(&data[0]));
        break;
      }
      case 2: {
        copyColumnDataOfShard(rowIndices,
                              reinterpret_cast<uint16_t*>(dataBlock.numbersPtr),
                              reinterpret_cast<uint16_t*>(&data[0]));
        break;
      }
      case 4: {
        copyColumnDataOfShard(rowIndices,
                              reinterpret_cast<uint32_t*>(dataBlock.numbersPtr),
                              reinterpret_cast<uint32_t*>(&data[0]));
        break;
      }
      case 8: {
        copyColumnDataOfShard(rowIndices,
                              reinterpret_cast<uint64_t*>(dataBlock.numbersPtr),
                              reinterpret_cast<uint64_t*>(&data[0]));
        break;
      }
      default:
        throw std::runtime_error("Unexpected data block element size");
    }

    ret.numbersPtr = reinterpret_cast<int8_t*>(&data[0]);
  }

  return {pCol->columnId, ret};
}

InsertData copyDataOfShard(const Catalog_Namespace::Catalog& cat,
                           ShardDataOwner& dataOwner,
                           InsertData& insert_data,
                           int shardTableIndex,
                           const std::vector<size_t>& rowIndices) {
  const auto* td = cat.getMetadataForTable(insert_data.tableId);
  const auto* ptd = cat.getPhysicalTablesDescriptors(td)[shardTableIndex];

  InsertData shardData;
  shardData.databaseId = insert_data.databaseId;
  shardData.tableId = ptd->tableId;
  shardData.numRows = rowIndices.size();

  std::vector<const ColumnDescriptor*> pCols;
  std::vector<int> lCols;

  {
    auto logicalColumns = cat.getAllColumnMetadataForTable(td->tableId, true, true, true);
    for (const auto& cd : logicalColumns) {
      lCols.push_back(cd->columnId);
    }

    auto physicalColumns =
        cat.getAllColumnMetadataForTable(ptd->tableId, true, true, true);
    for (const auto& cd : physicalColumns) {
      pCols.push_back(cd);
    }
  }

  for (size_t col = 0; col < insert_data.columnIds.size(); col++) {
    dataOwner.arrayData.emplace_back();
    dataOwner.rawData.emplace_back();
    dataOwner.stringData.emplace_back();
  }

  auto copycat = [&cat, &dataOwner, &rowIndices, &lCols, &pCols, &insert_data](int col) {
    const auto lColId = insert_data.columnIds[col];
    const auto pCol = pCols[indexOf(lCols, lColId)];
    return copyColumnDataOfShard(
        cat, dataOwner, rowIndices, pCol, col, insert_data.data[col]);
  };

  std::vector<utils::future<BlockWithColumnId>> worker_threads;
  for (size_t col = 0; col < insert_data.columnIds.size(); col++) {
    worker_threads.push_back(utils::async(copycat, col));
  }

  for (auto& child : worker_threads) {
    child.wait();
  }

  for (auto& child : worker_threads) {
    auto shardColumnData = child.get();
    shardData.columnIds.push_back(shardColumnData.columnId);
    shardData.data.push_back(shardColumnData.block);
  }

  return shardData;
}

void InsertDataLoader::insertData(const Catalog_Namespace::SessionInfo& session_info,
                                  InsertData& insert_data) {
  const auto& cat = session_info.getCatalog();
  const auto* td = cat.getMetadataForTable(insert_data.tableId);

  CHECK(td);
  if (td->nShards == 0) {
    connector_.insertDataToLeaf(session_info, current_leaf_index_, insert_data);
  } else {
    // we have a sharded target table, start spreading to physical tables
    auto rowIndicesOfShards =
        computeRowIndicesOfShards(cat, connector_.leafCount(), insert_data);

    auto insertShardData =
        [this, &session_info, &insert_data, &cat, &td, &rowIndicesOfShards](
            size_t shardId) {
          const auto shard_tables = cat.getPhysicalTablesDescriptors(td);
          auto stardTableIdx = shardId % td->nShards;
          auto shardLeafIdx = shardId / td->nShards;

          const auto& rowIndicesOfShard = rowIndicesOfShards[shardId];
          ShardDataOwner shardDataOwner;

          InsertData shardData = copyDataOfShard(
              cat, shardDataOwner, insert_data, stardTableIdx, rowIndicesOfShard);
          connector_.insertDataToLeaf(session_info, shardLeafIdx, shardData);
        };

    std::vector<utils::future<void>> worker_threads;
    for (size_t shardId = 0; shardId < rowIndicesOfShards.size(); shardId++) {
      if (rowIndicesOfShards[shardId].size() > 0) {
        worker_threads.push_back(utils::async(insertShardData, shardId));
      }
    }
    for (auto& child : worker_threads) {
      child.wait();
    }
    for (auto& child : worker_threads) {
      child.get();
    }
  }

  moveToNextLeaf();
}

}  // namespace Fragmenter_Namespace
