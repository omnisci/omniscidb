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

#include "ArrowForeignStorage.h"

#include <arrow/api.h>
#include <arrow/csv/reader.h>
#include <arrow/io/file.h>
#include <arrow/util/decimal.h>
#include <tbb/parallel_for.h>
#include <tbb/task_group.h>
#include <array>
#include <future>
#include <vector>

#include "Catalog/DataframeTableDescriptor.h"
#include "DataMgr/ForeignStorage/ForeignStorageInterface.h"
#include "DataMgr/StringNoneEncoder.h"
#include "Logger/Logger.h"
#include "QueryEngine/ArrowResultSet.h"
#include "Shared/ArrowUtil.h"
#include "Shared/measure.h"

struct Frag {
  size_t first_chunk;         // index of the first chunk assigned to the fragment
  size_t first_chunk_offset;  // offset from the begining of the first chunk
  size_t last_chunk;          // index of the last chunk
  size_t last_chunk_size;     // number of elements in the last chunk
};

struct ArrowFragment {
  int64_t offset{0};
  int64_t sz{0};
  std::vector<std::shared_ptr<arrow::ArrayData>> chunks;
};

class ArrowForeignStorageBase : public PersistentForeignStorageInterface {
 public:
  void append(const std::vector<ForeignStorageColumnBuffer>& column_buffers) override;

  void read(const ChunkKey& chunk_key,
            const SQLTypeInfo& sql_type,
            int8_t* dest,
            const size_t numBytes) override;

  int8_t* tryZeroCopy(const ChunkKey& chunk_key,
                      const SQLTypeInfo& sql_type,
                      const size_t numBytes) override;

  void parseArrowTable(Catalog_Namespace::Catalog* catalog,
                       std::pair<int, int> table_key,
                       const std::string& type,
                       const TableDescriptor& td,
                       const std::list<ColumnDescriptor>& cols,
                       Data_Namespace::AbstractBufferMgr* mgr,
                       const arrow::Table& table);

  std::shared_ptr<arrow::ChunkedArray> createDictionaryEncodedColumn(
      StringDictionary* dict,
      const ColumnDescriptor& c,
      std::shared_ptr<arrow::ChunkedArray> arr_col_chunked_array);

  std::shared_ptr<arrow::ChunkedArray> convertArrowDictionary(
      StringDictionary* dict,
      const ColumnDescriptor& c,
      std::shared_ptr<arrow::ChunkedArray> arr_col_chunked_array);

  template <typename T, typename ChunkType>
  std::shared_ptr<arrow::ChunkedArray> createDecimalColumn(
      const ColumnDescriptor& c,
      std::shared_ptr<arrow::ChunkedArray> arr_col_chunked_array);

  std::shared_ptr<arrow::ChunkedArray> replaceNullValues(
      const SQLTypeInfo& columnType,
      std::shared_ptr<arrow::ChunkedArray> arr_col_chunked_array);

  template <typename T>
  std::shared_ptr<arrow::ChunkedArray> replaceNullValuesImpl(
      std::shared_ptr<arrow::ChunkedArray> arr_col_chunked_array);

  void getSizeAndOffset(const Frag& frag,
                        const std::shared_ptr<arrow::Array>& chunk,
                        size_t i,
                        int& size,
                        int& offset);

  int64_t makeFragment(const Frag& frag,
                       ArrowFragment& arrowFrag,
                       const std::vector<std::shared_ptr<arrow::Array>>& chunks,
                       bool is_varlen);

  std::map<std::array<int, 3>, std::vector<ArrowFragment>> m_columns;
};

std::shared_ptr<arrow::ChunkedArray> ArrowForeignStorageBase::replaceNullValues(
    const SQLTypeInfo& columnType,
    std::shared_ptr<arrow::ChunkedArray> arr_col_chunked_array) {
  const size_t typeSize = columnType.get_size();
  if (columnType.is_integer() || is_datetime(columnType.get_type())) {
    switch (typeSize) {
      case 1:
        return replaceNullValuesImpl<int8_t>(arr_col_chunked_array);
      case 2:
        return replaceNullValuesImpl<int16_t>(arr_col_chunked_array);
      case 4:
        return replaceNullValuesImpl<int32_t>(arr_col_chunked_array);
      case 8:
        return replaceNullValuesImpl<int64_t>(arr_col_chunked_array);
      default:
        // TODO: throw unsupported integer type exception
        CHECK(false);
    }
  } else if (columnType.is_fp()) {
    switch (typeSize) {
      case 4:
        return replaceNullValuesImpl<float>(arr_col_chunked_array);
      case 8:
        return replaceNullValuesImpl<double>(arr_col_chunked_array);
    }
  } else if (columnType.is_boolean()) {
    return replaceNullValuesImpl<bool>(arr_col_chunked_array);
  }
  CHECK(false);
  return nullptr;
}

void convertBoolBitmapBuffer(int8_t* dst,
                             const uint8_t* src,
                             const uint8_t* bitmap,
                             int64_t length,
                             int64_t bitmap_length,
                             int8_t null_value) {
  for (int64_t bitmap_idx = 0; bitmap_idx < bitmap_length - 1; ++bitmap_idx) {
    auto source = src[bitmap_idx];
    auto dest = dst + bitmap_idx * 8;
    auto inversed_bitmap = ~bitmap[bitmap_idx];
    for (int8_t bitmap_offset = 0; bitmap_offset < 8; ++bitmap_offset) {
      auto is_null = (inversed_bitmap >> bitmap_offset) & 1;
      auto val = (source >> bitmap_offset) & 1;
      dest[bitmap_offset] = is_null ? null_value : val;
    }
  }

  for (int64_t j = (bitmap_length - 1) * 8; j < length; ++j) {
    auto is_null = (~bitmap[bitmap_length - 1] >> (j % 8)) & 1;
    auto val = (src[bitmap_length - 1] >> (j % 8)) & 1;
    dst[j] = is_null ? null_value : val;
  }
}

template <
    typename V,
    std::enable_if_t<!std::is_same_v<V, bool> && std::is_integral<V>::value, int> = 0>
inline V get_null_value() {
  return inline_int_null_value<V>();
}

template <typename V, std::enable_if_t<std::is_same_v<V, bool>, int> = 0>
inline int8_t get_null_value() {
  return inline_int_null_value<int8_t>();
}

template <typename V, std::enable_if_t<std::is_floating_point<V>::value, int> = 0>
inline V get_null_value() {
  return inline_fp_null_value<V>();
}

template <typename T>
std::shared_ptr<arrow::ChunkedArray> ArrowForeignStorageBase::replaceNullValuesImpl(
    std::shared_ptr<arrow::ChunkedArray> arr_col_chunked_array) {
  if ((!std::is_same_v<T, bool>)&&(arr_col_chunked_array->null_count() == 0)) {
    // for boolean columns we still need to convert bitmaps to array
    return arr_col_chunked_array;
  }

  auto null_value = get_null_value<T>();

  auto resultBuf =
      arrow::AllocateBuffer(sizeof(T) * arr_col_chunked_array->length()).ValueOrDie();
  auto resultData = reinterpret_cast<T*>(resultBuf->mutable_data());

  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, arr_col_chunked_array->num_chunks()),
      [&](const tbb::blocked_range<size_t>& r) {
        for (size_t c = r.begin(); c != r.end(); ++c) {
          size_t offset = 0;
          for (int i = 0; i < c; i++) {
            offset += arr_col_chunked_array->chunk(i)->length();
          }
          auto resWithOffset = resultData + offset;

          auto chunk = arr_col_chunked_array->chunk(c);

          if (chunk->null_count() == chunk->length()) {
            std::fill(resWithOffset, resWithOffset + chunk->length(), null_value);
            continue;
          }

          auto chunkData = reinterpret_cast<const T*>(chunk->data()->buffers[1]->data());

          if (chunk->null_count() == 0) {
            std::copy(chunkData, chunkData + chunk->length(), resWithOffset);
            continue;
          }

          const uint8_t* bitmap_data = chunk->null_bitmap_data();
          const int64_t length = chunk->length();
          const int64_t bitmap_length = chunk->null_bitmap()->size();

          if constexpr (std::is_same_v<T, bool>) {
            convertBoolBitmapBuffer(reinterpret_cast<int8_t*>(resWithOffset),
                                    reinterpret_cast<const uint8_t*>(chunkData),
                                    bitmap_data,
                                    length,
                                    bitmap_length,
                                    null_value);
          } else {
            for (int64_t bitmap_idx = 0; bitmap_idx < bitmap_length - 1; ++bitmap_idx) {
              auto source = chunkData + bitmap_idx * 8;
              auto dest = resWithOffset + bitmap_idx * 8;
              auto inversed_bitmap = ~bitmap_data[bitmap_idx];
              for (int8_t bitmap_offset = 0; bitmap_offset < 8; ++bitmap_offset) {
                auto is_null = (inversed_bitmap >> bitmap_offset) & 1;
                auto val = is_null ? null_value : source[bitmap_offset];
                dest[bitmap_offset] = val;
              }
            }

            for (int64_t j = (bitmap_length - 1) * 8; j < length; ++j) {
              auto is_null = (~bitmap_data[bitmap_length - 1] >> (j % 8)) & 1;
              auto val = is_null ? null_value : chunkData[j];
              resWithOffset[j] = val;
            }
          }
        }
      });

  using ArrowType = typename arrow::CTypeTraits<T>::ArrowType;
  using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

  auto array =
      std::make_shared<ArrayType>(arr_col_chunked_array->length(), std::move(resultBuf));
  return std::make_shared<arrow::ChunkedArray>(array);
}

void ArrowForeignStorageBase::getSizeAndOffset(const Frag& frag,
                                               const std::shared_ptr<arrow::Array>& chunk,
                                               size_t i,
                                               int& size,
                                               int& offset) {
  offset = (i == frag.first_chunk) ? frag.first_chunk_offset : 0;
  size = (i == frag.last_chunk) ? frag.last_chunk_size : (chunk->length() - offset);
}

int64_t ArrowForeignStorageBase::makeFragment(
    const Frag& frag,
    ArrowFragment& arrowFrag,
    const std::vector<std::shared_ptr<arrow::Array>>& chunks,
    bool is_varlen) {
  int64_t varlen = 0;
  arrowFrag.chunks.resize(frag.last_chunk - frag.first_chunk + 1);
  for (int i = frag.first_chunk, e = frag.last_chunk; i <= e; i++) {
    int size, offset;
    getSizeAndOffset(frag, chunks[i], i, size, offset);
    arrowFrag.offset += offset;
    arrowFrag.sz += size;
    arrowFrag.chunks[i - frag.first_chunk] = chunks[i]->data();
    auto& buffers = chunks[i]->data()->buffers;
    if (is_varlen) {
      if (buffers.size() <= 2) {
        throw std::runtime_error(
            "Importing fixed length arrow array as variable length column");
      }
      auto offsets_buffer = reinterpret_cast<const uint32_t*>(buffers[1]->data());
      varlen += offsets_buffer[offset + size] - offsets_buffer[offset];
    } else if (buffers.size() != 2) {
      throw std::runtime_error(
          "Importing varialbe length arrow array as fixed length column");
    }
  }
  // return length of string buffer if array is none encoded string
  return varlen;
}

std::vector<Frag> calculateFragmentsOffsets(const arrow::ChunkedArray& array,
                                            size_t maxFragRows) {
  std::vector<Frag> fragments;
  size_t sz = 0;
  size_t offset = 0;
  fragments.push_back({0, 0, 0, 0});
  size_t num_chunks = (size_t)array.num_chunks();
  for (size_t i = 0; i < num_chunks;) {
    auto& chunk = *array.chunk(i);
    auto& frag = *fragments.rbegin();
    if (maxFragRows - sz > chunk.length() - offset) {
      sz += chunk.length() - offset;
      if (i == num_chunks - 1) {
        fragments.rbegin()->last_chunk = num_chunks - 1;
        fragments.rbegin()->last_chunk_size =
            array.chunk((int)num_chunks - 1)->length() - offset;
      }
      offset = 0;
      i++;
    } else {
      frag.last_chunk = i;
      frag.last_chunk_size = maxFragRows - sz;
      offset += maxFragRows - sz;
      sz = 0;
      fragments.push_back({i, offset, 0, 0});
    }
  }
  if (fragments.rbegin()->first_chunk == fragments.rbegin()->first_chunk &&
      fragments.rbegin()->last_chunk_size == 0) {
    // remove empty fragment at the end if any
    fragments.pop_back();
  }
  return fragments;
}

void ArrowForeignStorageBase::parseArrowTable(Catalog_Namespace::Catalog* catalog,
                                              std::pair<int, int> table_key,
                                              const std::string& type,
                                              const TableDescriptor& td,
                                              const std::list<ColumnDescriptor>& cols,
                                              Data_Namespace::AbstractBufferMgr* mgr,
                                              const arrow::Table& table) {
  std::map<std::array<int, 3>, StringDictionary*> dictionaries;
  for (auto& c : cols) {
    std::array<int, 3> col_key{table_key.first, table_key.second, c.columnId};
    m_columns[col_key] = {};
    // fsi registerTable runs under SqliteLock which does not allow invoking
    // getMetadataForDict in other threads
    if (c.columnType.is_dict_encoded_string()) {
      auto dictDesc = catalog->getMetadataForDict(c.columnType.get_comp_param());
      dictionaries[col_key] = dictDesc->stringDict.get();
    }
  }

  tbb::task_group tg;

  tbb::parallel_for(
      tbb::blocked_range(0, (int)cols.size()),
      [this, &tg, &table_key, &td, mgr, &table, &cols, &dictionaries](auto range) {
        auto columnIter = std::next(cols.begin(), range.begin());
        for (auto col_idx = range.begin(); col_idx != range.end(); col_idx++) {
          auto& c = *(columnIter++);

          if (c.isSystemCol) {
            continue;  // must be processed by base interface implementation
          }

          // data comes like this - database_id, table_id, column_id, fragment_id
          ChunkKey key{table_key.first, table_key.second, c.columnId, 0};
          std::array<int, 3> col_key{table_key.first, table_key.second, c.columnId};

          if (col_idx >= table.num_columns()) {
            LOG(ERROR) << "Number of columns read from Arrow (" << table.num_columns()
                       << ") mismatch CREATE TABLE request: " << cols.size();
            break;
          }

          auto arr_col_chunked_array = table.column(col_idx);
          auto column_type = c.columnType.get_type();

          if (column_type != kDECIMAL && column_type != kNUMERIC &&
              !c.columnType.is_string()) {
            arr_col_chunked_array = replaceNullValues(column_type, arr_col_chunked_array);
          }

          if (c.columnType.is_dict_encoded_string()) {
            StringDictionary* dict = dictionaries[col_key];

            switch (arr_col_chunked_array->type()->id()) {
              case arrow::Type::STRING:
                arr_col_chunked_array =
                    createDictionaryEncodedColumn(dict, c, arr_col_chunked_array);
                break;
              case arrow::Type::DICTIONARY:
                arr_col_chunked_array =
                    convertArrowDictionary(dict, c, arr_col_chunked_array);
                break;
              default:
                CHECK(false);
            }
          } else if (column_type == kDECIMAL || column_type == kNUMERIC) {
            switch (c.columnType.get_size()) {
              case 2:
                arr_col_chunked_array = createDecimalColumn<int16_t, arrow::Int16Array>(
                    c, arr_col_chunked_array);
                break;
              case 4:
                arr_col_chunked_array = createDecimalColumn<int32_t, arrow::Int32Array>(
                    c, arr_col_chunked_array);
                break;
              case 8:
                arr_col_chunked_array = createDecimalColumn<int64_t, arrow::Int64Array>(
                    c, arr_col_chunked_array);
                break;
              default:
                // TODO: throw unsupported decimal type exception
                CHECK(false);
                break;
            }
          }

          auto fragments =
              calculateFragmentsOffsets(*arr_col_chunked_array, td.maxFragRows);

          auto ctype = c.columnType.get_type();
          auto& col = m_columns[col_key];
          col.resize(fragments.size());

          for (size_t f = 0; f < fragments.size(); f++) {
            key[3] = f;
            auto& frag = col[f];
            bool is_varlen = ctype == kTEXT && !c.columnType.is_dict_encoded_string();
            size_t varlen = makeFragment(
                fragments[f], frag, arr_col_chunked_array->chunks(), is_varlen);

            // create buffer descriptors
            if (ctype == kTEXT && !c.columnType.is_dict_encoded_string()) {
              auto k = key;
              k.push_back(1);
              {
                auto b = mgr->createBuffer(k);
                b->setSize(varlen);
                b->initEncoder(c.columnType);
              }
              k[4] = 2;
              {
                auto b = mgr->createBuffer(k);
                b->setSqlType(SQLTypeInfo(kINT, false));
                b->setSize(frag.sz * b->getSqlType().get_size());
              }
            } else {
              auto b = mgr->createBuffer(key);
              b->setSize(frag.sz * c.columnType.get_size());
              b->initEncoder(c.columnType);
              size_t type_size = c.columnType.get_size();
              tg.run([b, fr = &frag, type_size]() {
                size_t sz = 0;
                for (size_t i = 0; i < fr->chunks.size(); i++) {
                  auto& chunk = fr->chunks[i];
                  int offset = (i == 0) ? fr->offset : 0;
                  size_t size = (i == fr->chunks.size() - 1) ? (fr->sz - sz)
                                                             : (chunk->length - offset);
                  sz += size;
                  auto data = chunk->buffers[1]->data();
                  b->getEncoder()->updateStatsEncoded(
                      (const int8_t*)data + offset * type_size, size);
                }
              });
              b->getEncoder()->setNumElems(frag.sz);
            }
          }
        }
      });  // each col and fragment

  // wait untill all stats have been updated
  tg.wait();
}

void ArrowForeignStorageBase::append(
    const std::vector<ForeignStorageColumnBuffer>& column_buffers) {
  CHECK(false);
}

void ArrowForeignStorageBase::read(const ChunkKey& chunk_key,
                                   const SQLTypeInfo& sql_type,
                                   int8_t* dest,
                                   const size_t numBytes) {
  std::array<int, 3> col_key{chunk_key[0], chunk_key[1], chunk_key[2]};
  auto& frag = m_columns.at(col_key).at(chunk_key[3]);

  CHECK(!frag.chunks.empty() || !chunk_key[3]);
  int64_t sz = 0, copied = 0;
  int varlen_offset = 0;
  size_t read_size = 0;
  for (size_t i = 0; i < frag.chunks.size(); i++) {
    auto& array_data = frag.chunks[i];
    int offset = (i == 0) ? frag.offset : 0;
    size_t size = (i == frag.chunks.size() - 1) ? (frag.sz - read_size)
                                                : (array_data->length - offset);
    read_size += size;
    arrow::Buffer* bp = nullptr;
    if (sql_type.is_dict_encoded_string()) {
      // array_data->buffers[1] stores dictionary indexes
      bp = array_data->buffers[1].get();
    } else if (sql_type.get_type() == kTEXT) {
      CHECK_GE(array_data->buffers.size(), 3UL);
      // array_data->buffers[2] stores string array
      bp = array_data->buffers[2].get();
    } else if (array_data->null_count != array_data->length) {
      // any type except strings (none encoded strings offsets go here as well)
      CHECK_GE(array_data->buffers.size(), 2UL);
      bp = array_data->buffers[1].get();
    }
    CHECK(bp);
    // offset buffer for none encoded strings need to be merged
    if (chunk_key.size() == 5 && chunk_key[4] == 2) {
      auto data = reinterpret_cast<const uint32_t*>(bp->data()) + offset;
      auto dest_ui32 = reinterpret_cast<uint32_t*>(dest);
      // as size contains count of string in chunk slice it would always be one less
      // then offsets array size
      sz = (size + 1) * sizeof(uint32_t);
      if (sz > 0) {
        if (i != 0) {
          // We merge arrow chunks with string offsets into a single contigous fragment.
          // Each string is represented by a pair of offsets, thus size of offset table
          // is num strings + 1. When merging two chunks, the last number in the first
          // chunk duplicates the first number in the second chunk, so we skip it.
          data++;
          sz -= sizeof(uint32_t);
        } else {
          // As we support cases when fragment starts with offset of arrow chunk we need
          // to substract the first element of the first chunk from all elements in that
          // fragment
          varlen_offset -= data[0];
        }
        // We also re-calculate offsets in the second chunk as it is a continuation of
        // the first one.
        std::transform(data,
                       data + (sz / sizeof(uint32_t)),
                       dest_ui32,
                       [varlen_offset](uint32_t val) { return val + varlen_offset; });
        varlen_offset += data[(sz / sizeof(uint32_t)) - 1];
      }
    } else {
      auto fixed_type = dynamic_cast<arrow::FixedWidthType*>(array_data->type.get());
      if (fixed_type) {
        std::memcpy(
            dest,
            bp->data() + (array_data->offset + offset) * (fixed_type->bit_width() / 8),
            sz = size * (fixed_type->bit_width() / 8));
      } else {
        auto offsets_buffer =
            reinterpret_cast<const uint32_t*>(array_data->buffers[1]->data());
        auto string_buffer_offset = offsets_buffer[offset + array_data->offset];
        auto string_buffer_size =
            offsets_buffer[offset + array_data->offset + size] - string_buffer_offset;
        std::memcpy(dest, bp->data() + string_buffer_offset, sz = string_buffer_size);
      }
    }
    dest += sz;
    copied += sz;
  }
  CHECK_EQ(numBytes, size_t(copied));
}

int8_t* ArrowForeignStorageBase::tryZeroCopy(const ChunkKey& chunk_key,
                                             const SQLTypeInfo& sql_type,
                                             const size_t numBytes) {
  std::array<int, 3> col_key{chunk_key[0], chunk_key[1], chunk_key[2]};
  auto& frag = m_columns.at(col_key).at(chunk_key[3]);

  // fragment should be continious to allow zero copy
  if (frag.chunks.size() != 1) {
    return nullptr;
  }

  auto& array_data = frag.chunks[0];
  int offset = frag.offset;

  arrow::Buffer* bp = nullptr;
  if (sql_type.is_dict_encoded_string()) {
    // array_data->buffers[1] stores dictionary indexes
    bp = array_data->buffers[1].get();
  } else if (sql_type.get_type() == kTEXT) {
    CHECK_GE(array_data->buffers.size(), 3UL);
    // array_data->buffers[2] stores string array
    bp = array_data->buffers[2].get();
  } else if (array_data->null_count != array_data->length) {
    // any type except strings (none encoded strings offsets go here as well)
    CHECK_GE(array_data->buffers.size(), 2UL);
    bp = array_data->buffers[1].get();
  }

  // arrow buffer is empty, it means we should fill fragment with null's in read function
  if (!bp) {
    return nullptr;
  }

  auto data = reinterpret_cast<int8_t*>(const_cast<uint8_t*>(bp->data()));

  // if buffer is null encoded string index buffer
  if (chunk_key.size() == 5 && chunk_key[4] == 2) {
    // if offset != 0 we need to recalculate index buffer by adding  offset to each index
    if (offset != 0) {
      return nullptr;
    } else {
      return data;
    }
  }

  auto fixed_type = dynamic_cast<arrow::FixedWidthType*>(array_data->type.get());
  if (fixed_type) {
    return data + (array_data->offset + offset) * (fixed_type->bit_width() / 8);
  }
  // if buffer is none encoded string data buffer
  // then we should find it's offset in offset buffer
  auto offsets_buffer = reinterpret_cast<const uint32_t*>(array_data->buffers[1]->data());
  auto string_buffer_offset = offsets_buffer[offset + array_data->offset];
  return data + string_buffer_offset;
}

std::shared_ptr<arrow::ChunkedArray>
ArrowForeignStorageBase::createDictionaryEncodedColumn(
    StringDictionary* dict,
    const ColumnDescriptor& c,
    std::shared_ptr<arrow::ChunkedArray> arr_col_chunked_array) {
  // calculate offsets for every fragment in bulk
  size_t bulk_size = 0;
  std::vector<int> offsets(arr_col_chunked_array->num_chunks());
  for (int i = 0; i < arr_col_chunked_array->num_chunks(); i++) {
    offsets[i] = bulk_size;
    bulk_size += arr_col_chunked_array->chunk(i)->length();
  }

  std::vector<std::string_view> bulk(bulk_size);

  tbb::parallel_for(
      tbb::blocked_range<int>(0, arr_col_chunked_array->num_chunks()),
      [&bulk, &arr_col_chunked_array, &offsets](const tbb::blocked_range<int>& r) {
        for (int i = r.begin(); i < r.end(); i++) {
          auto chunk = std::static_pointer_cast<arrow::StringArray>(
              arr_col_chunked_array->chunk(i));
          auto offset = offsets[i];
          for (int j = 0; j < chunk->length(); j++) {
            auto view = chunk->GetView(j);
            bulk[offset + j] = std::string_view(view.data(), view.length());
          }
        }
      });

  std::shared_ptr<arrow::Buffer> indices_buf;
  auto res = arrow::AllocateBuffer(bulk_size * sizeof(int32_t));
  CHECK(res.ok());
  indices_buf = std::move(res).ValueOrDie();
  auto raw_data = reinterpret_cast<int*>(indices_buf->mutable_data());
  dict->getOrAddBulk(bulk, raw_data);
  auto array = std::make_shared<arrow::Int32Array>(bulk_size, indices_buf);
  return std::make_shared<arrow::ChunkedArray>(array);
}

std::shared_ptr<arrow::ChunkedArray> ArrowForeignStorageBase::convertArrowDictionary(
    StringDictionary* dict,
    const ColumnDescriptor& c,
    std::shared_ptr<arrow::ChunkedArray> arr_col_chunked_array) {
  // TODO: allocate one big array and split it by fragments as it is done in
  // createDictionaryEncodedColumn
  std::vector<std::shared_ptr<arrow::Array>> converted_chunks;
  for (auto& chunk : arr_col_chunked_array->chunks()) {
    auto dict_array = std::static_pointer_cast<arrow::DictionaryArray>(chunk);
    auto values = std::static_pointer_cast<arrow::StringArray>(dict_array->dictionary());
    std::vector<std::string_view> strings(values->length());
    for (int i = 0; i < values->length(); i++) {
      auto view = values->GetView(i);
      strings[i] = std::string_view(view.data(), view.length());
    }
    auto arrow_indices =
        std::static_pointer_cast<arrow::Int32Array>(dict_array->indices());
    std::vector<int> indices_mapping(values->length());
    dict->getOrAddBulk(strings, indices_mapping.data());

    // create new arrow chunk with remapped indices
    std::shared_ptr<arrow::Buffer> dict_indices_buf;
    auto res = arrow::AllocateBuffer(arrow_indices->length() * sizeof(int32_t));
    CHECK(res.ok());
    dict_indices_buf = std::move(res).ValueOrDie();
    auto raw_data = reinterpret_cast<int32_t*>(dict_indices_buf->mutable_data());

    for (int i = 0; i < arrow_indices->length(); i++) {
      raw_data[i] = indices_mapping[arrow_indices->Value(i)];
    }

    converted_chunks.push_back(
        std::make_shared<arrow::Int32Array>(arrow_indices->length(), dict_indices_buf));
  }
  return std::make_shared<arrow::ChunkedArray>(converted_chunks);
}

template <typename T, typename ChunkType>
std::shared_ptr<arrow::ChunkedArray> ArrowForeignStorageBase::createDecimalColumn(
    const ColumnDescriptor& c,
    std::shared_ptr<arrow::ChunkedArray> arr_col_chunked_array) {
  size_t column_size = 0;
  std::vector<int> offsets(arr_col_chunked_array->num_chunks());
  for (int i = 0; i < arr_col_chunked_array->num_chunks(); i++) {
    offsets[i] = column_size;
    column_size += arr_col_chunked_array->chunk(i)->length();
  }

  std::shared_ptr<arrow::Buffer> result_buffer;
  auto res = arrow::AllocateBuffer(column_size * c.columnType.get_size());
  CHECK(res.ok());
  result_buffer = std::move(res).ValueOrDie();

  T* buffer_data = reinterpret_cast<T*>(result_buffer->mutable_data());
  tbb::parallel_for(
      tbb::blocked_range(0, arr_col_chunked_array->num_chunks()),
      [buffer_data, &offsets, arr_col_chunked_array](auto& range) {
        for (int chunk_idx = range.begin(); chunk_idx < range.end(); chunk_idx++) {
          auto offset = offsets[chunk_idx];
          T* chunk_buffer = buffer_data + offset;

          auto decimalArray = std::static_pointer_cast<arrow::Decimal128Array>(
              arr_col_chunked_array->chunk(chunk_idx));
          auto empty =
              arr_col_chunked_array->null_count() == arr_col_chunked_array->length();
          for (int i = 0; i < decimalArray->length(); i++) {
            if (empty || decimalArray->null_count() == decimalArray->length() ||
                decimalArray->IsNull(i)) {
              chunk_buffer[i] = inline_int_null_value<T>();
            } else {
              arrow::Decimal128 val(decimalArray->GetValue(i));
              chunk_buffer[i] =
                  static_cast<int64_t>(val);  // arrow can cast only to int64_t
            }
          }
        }
      });
  auto array = std::make_shared<ChunkType>(column_size, result_buffer);
  return std::make_shared<arrow::ChunkedArray>(array);
}

class ArrowForeignStorage : public ArrowForeignStorageBase {
 public:
  ArrowForeignStorage() {}

  void prepareTable(const int db_id,
                    const std::string& type,
                    TableDescriptor& td,
                    std::list<ColumnDescriptor>& cols) override;
  void registerTable(Catalog_Namespace::Catalog* catalog,
                     std::pair<int, int> table_key,
                     const std::string& type,
                     const TableDescriptor& td,
                     const std::list<ColumnDescriptor>& cols,
                     Data_Namespace::AbstractBufferMgr* mgr) override;

  std::string getType() const override;

  std::string name;

  static std::map<std::string, std::shared_ptr<arrow::Table>> tables;
};

std::map<std::string, std::shared_ptr<arrow::Table>> ArrowForeignStorage::tables =
    std::map<std::string, std::shared_ptr<arrow::Table>>();

static SQLTypeInfo getOmnisciType(const arrow::DataType& type) {
  using namespace arrow;
  switch (type.id()) {
    case Type::INT8:
      return SQLTypeInfo(kTINYINT, false);
    case Type::INT16:
      return SQLTypeInfo(kSMALLINT, false);
    case Type::INT32:
      return SQLTypeInfo(kINT, false);
    case Type::INT64:
      return SQLTypeInfo(kBIGINT, false);
    case Type::BOOL:
      return SQLTypeInfo(kBOOLEAN, false);
    case Type::FLOAT:
      return SQLTypeInfo(kFLOAT, false);
    case Type::DOUBLE:
      return SQLTypeInfo(kDOUBLE, false);
      // uncomment when arrow 2.0 will be released and modin support for dictionary types
      // in read_csv would be implemented

      // case Type::DICTIONARY: {
      //   auto type = SQLTypeInfo(kTEXT, false, kENCODING_DICT);
      //   // this is needed because createTable forces type.size to be equal to
      //   // comp_param / 8, no matter what type.size you set here
      //   type.set_comp_param(sizeof(uint32_t) * 8);
      //   return type;
      // }
      // case Type::STRING:
      //   return SQLTypeInfo(kTEXT, false, kENCODING_NONE);

    case Type::STRING: {
      auto type = SQLTypeInfo(kTEXT, false, kENCODING_DICT);
      // this is needed because createTable forces type.size to be equal to
      // comp_param / 8, no matter what type.size you set here
      type.set_comp_param(sizeof(uint32_t) * 8);
      return type;
    }
    case Type::DECIMAL: {
      const auto& decimal_type = static_cast<const arrow::DecimalType&>(type);
      return SQLTypeInfo(kDECIMAL, decimal_type.precision(), decimal_type.scale(), false);
    }
    case Type::TIME32:
      return SQLTypeInfo(kTIME, false);
    case Type::TIMESTAMP:
      switch (static_cast<const arrow::TimestampType&>(type).unit()) {
        case TimeUnit::SECOND:
          return SQLTypeInfo(kTIMESTAMP, 0, 0);
        case TimeUnit::MILLI:
          return SQLTypeInfo(kTIMESTAMP, 3, 0);
        case TimeUnit::MICRO:
          return SQLTypeInfo(kTIMESTAMP, 6, 0);
        case TimeUnit::NANO:
          return SQLTypeInfo(kTIMESTAMP, 9, 0);
      }
    default:
      throw std::runtime_error(type.ToString() + " is not yet supported.");
  }
}

void ArrowForeignStorage::prepareTable(const int db_id,
                                       const std::string& name,
                                       TableDescriptor& td,
                                       std::list<ColumnDescriptor>& cols) {
  td.hasDeletedCol = false;
  this->name = name;
  auto table = tables[name];
  for (auto& field : table->schema()->fields()) {
    ColumnDescriptor cd;
    cd.columnName = field->name();
    cd.columnType = getOmnisciType(*field->type());
    cols.push_back(cd);
  }
}

void ArrowForeignStorage::registerTable(Catalog_Namespace::Catalog* catalog,
                                        std::pair<int, int> table_key,
                                        const std::string& info,
                                        const TableDescriptor& td,
                                        const std::list<ColumnDescriptor>& cols,
                                        Data_Namespace::AbstractBufferMgr* mgr) {
  parseArrowTable(catalog, table_key, info, td, cols, mgr, *(tables[name].get()));
}

std::string ArrowForeignStorage::getType() const {
  LOG(INFO) << "CSV backed temporary tables has been activated. Create table `with "
               "(storage_type='CSV:path/to/file.csv');`\n";
  return "ARROW";
}

void setArrowTable(std::string name, std::shared_ptr<arrow::Table> table) {
  ArrowForeignStorage::tables[name] = table;
}

void releaseArrowTable(std::string name) {
  ArrowForeignStorage::tables.erase(name);
}

void registerArrowForeignStorage(std::shared_ptr<ForeignStorageInterface> fsi) {
  fsi->registerPersistentStorageInterface(std::make_unique<ArrowForeignStorage>());
}

class ArrowCsvForeignStorage : public ArrowForeignStorageBase {
 public:
  ArrowCsvForeignStorage() {}

  void prepareTable(const int db_id,
                    const std::string& type,
                    TableDescriptor& td,
                    std::list<ColumnDescriptor>& cols) override;
  void registerTable(Catalog_Namespace::Catalog* catalog,
                     std::pair<int, int> table_key,
                     const std::string& type,
                     const TableDescriptor& td,
                     const std::list<ColumnDescriptor>& cols,
                     Data_Namespace::AbstractBufferMgr* mgr) override;

  std::string getType() const override;
};

void ArrowCsvForeignStorage::prepareTable(const int db_id,
                                          const std::string& type,
                                          TableDescriptor& td,
                                          std::list<ColumnDescriptor>& cols) {
  td.hasDeletedCol = false;
}

// TODO: this overlaps with getArrowType() from ArrowResultSetConverter.cpp but with few
// differences in kTEXT and kDATE
static std::shared_ptr<arrow::DataType> getArrowImportType(const SQLTypeInfo type) {
  using namespace arrow;
  auto ktype = type.get_type();
  if (IS_INTEGER(ktype)) {
    switch (type.get_size()) {
      case 1:
        return int8();
      case 2:
        return int16();
      case 4:
        return int32();
      case 8:
        return int64();
      default:
        CHECK(false);
    }
  }
  switch (ktype) {
    case kBOOLEAN:
      return arrow::boolean();
    case kFLOAT:
      return float32();
    case kDOUBLE:
      return float64();
    case kCHAR:
    case kVARCHAR:
    case kTEXT:
      return utf8();
    case kDECIMAL:
    case kNUMERIC:
      return decimal(type.get_precision(), type.get_scale());
    case kTIME:
      return time32(TimeUnit::SECOND);
    // case kDATE:
    // TODO(wamsi) : Remove date64() once date32() support is added in cuDF. date32()
    // Currently support for date32() is missing in cuDF.Hence, if client requests for
    // date on GPU, return date64() for the time being, till support is added.
    // return device_type_ == ExecutorDeviceType::GPU ? date64() : date32();
    case kTIMESTAMP:
      switch (type.get_precision()) {
        case 0:
          return timestamp(TimeUnit::SECOND);
        case 3:
          return timestamp(TimeUnit::MILLI);
        case 6:
          return timestamp(TimeUnit::MICRO);
        case 9:
          return timestamp(TimeUnit::NANO);
        default:
          throw std::runtime_error("Unsupported timestamp precision for Arrow: " +
                                   std::to_string(type.get_precision()));
      }
    case kARRAY:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
    default:
      throw std::runtime_error(type.get_type_name() + " is not supported in Arrow.");
  }
  return nullptr;
}

void ArrowCsvForeignStorage::registerTable(Catalog_Namespace::Catalog* catalog,
                                           std::pair<int, int> table_key,
                                           const std::string& info,
                                           const TableDescriptor& td,
                                           const std::list<ColumnDescriptor>& cols,
                                           Data_Namespace::AbstractBufferMgr* mgr) {
  const DataframeTableDescriptor* df_td =
      dynamic_cast<const DataframeTableDescriptor*>(&td);
  bool isDataframe = df_td ? true : false;
  std::unique_ptr<DataframeTableDescriptor> df_td_owned;
  if (!isDataframe) {
    df_td_owned = std::make_unique<DataframeTableDescriptor>(td);
    CHECK(df_td_owned);
    df_td = df_td_owned.get();
  }
  auto memory_pool = arrow::default_memory_pool();
  auto arrow_parse_options = arrow::csv::ParseOptions::Defaults();
  arrow_parse_options.quoting = false;
  arrow_parse_options.escaping = false;
  arrow_parse_options.newlines_in_values = false;
  arrow_parse_options.delimiter = *df_td->delimiter.c_str();
  auto arrow_read_options = arrow::csv::ReadOptions::Defaults();
  arrow_read_options.use_threads = true;

  arrow_read_options.block_size = 20 * 1024 * 1024;
  arrow_read_options.autogenerate_column_names = false;
  arrow_read_options.skip_rows =
      df_td->hasHeader ? (df_td->skipRows + 1) : df_td->skipRows;

  auto arrow_convert_options = arrow::csv::ConvertOptions::Defaults();
  arrow_convert_options.check_utf8 = false;
  arrow_convert_options.include_columns = arrow_read_options.column_names;
  arrow_convert_options.strings_can_be_null = true;

  for (auto& c : cols) {
    if (c.isSystemCol) {
      continue;  // must be processed by base interface implementation
    }
    arrow_convert_options.column_types.emplace(c.columnName,
                                               getArrowImportType(c.columnType));
    arrow_read_options.column_names.push_back(c.columnName);
  }

  std::shared_ptr<arrow::io::ReadableFile> inp;
  auto file_result = arrow::io::ReadableFile::Open(info.c_str());
  ARROW_THROW_NOT_OK(file_result.status());
  inp = file_result.ValueOrDie();

  auto table_reader_result = arrow::csv::TableReader::Make(
      memory_pool, inp, arrow_read_options, arrow_parse_options, arrow_convert_options);
  ARROW_THROW_NOT_OK(table_reader_result.status());
  auto table_reader = table_reader_result.ValueOrDie();

  std::shared_ptr<arrow::Table> arrowTable;
  auto time = measure<>::execution([&]() {
    auto arrow_table_result = table_reader->Read();
    ARROW_THROW_NOT_OK(arrow_table_result.status());
    arrowTable = arrow_table_result.ValueOrDie();
  });

  VLOG(1) << "Read Arrow CSV file " << info << " in " << time << "ms";

  arrow::Table& table = *arrowTable.get();
  parseArrowTable(catalog, table_key, info, td, cols, mgr, table);
}

std::string ArrowCsvForeignStorage::getType() const {
  LOG(INFO) << "CSV backed temporary tables has been activated. Create table `with "
               "(storage_type='CSV:path/to/file.csv');`\n";
  return "CSV";
}

void registerArrowCsvForeignStorage(std::shared_ptr<ForeignStorageInterface> fsi) {
  fsi->registerPersistentStorageInterface(std::make_unique<ArrowCsvForeignStorage>());
}
