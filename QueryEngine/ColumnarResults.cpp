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

#include "ColumnarResults.h"
#include "Descriptors/RowSetMemoryOwner.h"
#include "Shared/Intervals.h"
#include "Shared/thread_count.h"

#include <atomic>
#include <future>
#include <numeric>

#include "Utils/Threading.h"

namespace {

inline int64_t fixed_encoding_nullable_val(const int64_t val,
                                           const SQLTypeInfo& type_info) {
  if (type_info.get_compression() != kENCODING_NONE) {
    CHECK(type_info.get_compression() == kENCODING_FIXED ||
          type_info.get_compression() == kENCODING_DICT);
    auto logical_ti = get_logical_type_info(type_info);
    if (val == inline_int_null_val(logical_ti)) {
      return inline_fixed_encoding_null_val(type_info);
    }
  }
  return val;
}

}  // namespace

ColumnarResults::ColumnarResults(std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                 const ResultSet& rows,
                                 const size_t num_columns,
                                 const std::vector<SQLTypeInfo>& target_types,
                                 const bool is_parallel_execution_enforced)
    : column_buffers_(num_columns)
    , num_rows_(result_set::use_parallel_algorithms(rows) ||
                        rows.isDirectColumnarConversionPossible()
                    ? rows.entryCount()
                    : rows.rowCount())
    , target_types_(target_types)
    , parallel_conversion_(is_parallel_execution_enforced
                               ? true
                               : result_set::use_parallel_algorithms(rows))
    , direct_columnar_conversion_(rows.isDirectColumnarConversionPossible()) {
  auto timer = DEBUG_TIMER(__func__);
  column_buffers_.resize(num_columns);
  for (size_t i = 0; i < num_columns; ++i) {
    const bool is_varlen = target_types[i].is_array() ||
                           (target_types[i].is_string() &&
                            target_types[i].get_compression() == kENCODING_NONE) ||
                           target_types[i].is_geometry();
    if (is_varlen) {
      throw ColumnarConversionNotSupported();
    }
    if (!isDirectColumnarConversionPossible() ||
        !rows.isZeroCopyColumnarConversionPossible(i)) {
      column_buffers_[i] =
          row_set_mem_owner->allocate(num_rows_ * target_types[i].get_size());
    }
  }

  if (isDirectColumnarConversionPossible() && rows.entryCount() > 0) {
    materializeAllColumnsDirectly(rows, num_columns);
  } else {
    materializeAllColumnsThroughIteration(rows, num_columns);
  }
}

ColumnarResults::ColumnarResults(std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                 const int8_t* one_col_buffer,
                                 const size_t num_rows,
                                 const SQLTypeInfo& target_type)
    : column_buffers_(1)
    , num_rows_(num_rows)
    , target_types_{target_type}
    , parallel_conversion_(false)
    , direct_columnar_conversion_(false) {
  auto timer = DEBUG_TIMER(__func__);
  const bool is_varlen =
      target_type.is_array() ||
      (target_type.is_string() && target_type.get_compression() == kENCODING_NONE) ||
      target_type.is_geometry();
  if (is_varlen) {
    throw ColumnarConversionNotSupported();
  }
  const auto buf_size = num_rows * target_type.get_size();
  column_buffers_[0] = reinterpret_cast<int8_t*>(row_set_mem_owner->allocate(buf_size));
  memcpy(((void*)column_buffers_[0]), one_col_buffer, buf_size);
}

std::unique_ptr<ColumnarResults> ColumnarResults::mergeResults(
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const std::vector<std::unique_ptr<ColumnarResults>>& sub_results) {
  if (sub_results.empty()) {
    return nullptr;
  }
  const auto total_row_count = std::accumulate(
      sub_results.begin(),
      sub_results.end(),
      size_t(0),
      [](const size_t init, const std::unique_ptr<ColumnarResults>& result) {
        return init + result->size();
      });
  std::unique_ptr<ColumnarResults> merged_results(
      new ColumnarResults(total_row_count, sub_results[0]->target_types_));
  const auto col_count = sub_results[0]->column_buffers_.size();
  const auto nonempty_it = std::find_if(
      sub_results.begin(),
      sub_results.end(),
      [](const std::unique_ptr<ColumnarResults>& needle) { return needle->size(); });
  if (nonempty_it == sub_results.end()) {
    return nullptr;
  }
  for (size_t col_idx = 0; col_idx < col_count; ++col_idx) {
    const auto byte_width = (*nonempty_it)->getColumnType(col_idx).get_size();
    auto write_ptr = row_set_mem_owner->allocate(byte_width * total_row_count);
    merged_results->column_buffers_.push_back(write_ptr);
    for (auto& rs : sub_results) {
      CHECK_EQ(col_count, rs->column_buffers_.size());
      if (!rs->size()) {
        continue;
      }
      CHECK_EQ(byte_width, rs->getColumnType(col_idx).get_size());
      memcpy(write_ptr, rs->column_buffers_[col_idx], rs->size() * byte_width);
      write_ptr += rs->size() * byte_width;
    }
  }
  return merged_results;
}

/**
 * This function iterates through the result set (using the getRowAtNoTranslation and
 * getNextRow family of functions) and writes back the results into output column buffers.
 */
void ColumnarResults::materializeAllColumnsThroughIteration(const ResultSet& rows,
                                                            const size_t num_columns) {
  std::atomic<size_t> row_idx{0};
  const auto do_work = [num_columns, this](const std::vector<TargetValue>& crt_row,
                                           const size_t row_idx) {
    for (size_t i = 0; i < num_columns; ++i) {
      writeBackCell(crt_row[i], row_idx, i);
    }
  };
  if (isParallelConversion()) {
    const size_t worker_count = cpu_threads();
    std::vector<utils::future<void>> conversion_threads;
    for (auto interval : makeIntervals(size_t(0), rows.entryCount(), worker_count)) {
      conversion_threads.push_back(utils::async(
          [&rows, &do_work, &row_idx](const size_t start, const size_t end) {
            for (size_t i = start; i < end; ++i) {
              const auto crt_row = rows.getRowAtNoTranslations(i);
              if (!crt_row.empty()) {
                do_work(crt_row, row_idx.fetch_add(1));
              }
            }
          },
          interval.begin,
          interval.end));
    }
    for (auto& child : conversion_threads) {
      child.wait();
    }

    num_rows_ = row_idx;
    rows.setCachedRowCount(num_rows_);
    return;
  }
  while (true) {
    const auto crt_row = rows.getNextRow(false, false);
    if (crt_row.empty()) {
      break;
    }
    do_work(crt_row, row_idx);
    ++row_idx;
  }
  rows.moveToBegin();
}

/*
 * This function processes and decodes its input TargetValue
 * and write it into its corresponding column buffer's cell (with corresponding
 * row and column indices)
 *
 * NOTE: this is not supposed to be processing varlen types, and they should be
 * handled differently outside this function.
 */
inline void ColumnarResults::writeBackCell(const TargetValue& col_val,
                                           const size_t row_idx,
                                           const size_t column_idx) {
  const auto scalar_col_val = boost::get<ScalarTargetValue>(&col_val);
  CHECK(scalar_col_val);
  auto i64_p = boost::get<int64_t>(scalar_col_val);
  const auto& type_info = target_types_[column_idx];
  if (i64_p) {
    const auto val = fixed_encoding_nullable_val(*i64_p, type_info);
    switch (target_types_[column_idx].get_size()) {
      case 1:
        ((int8_t*)column_buffers_[column_idx])[row_idx] = static_cast<int8_t>(val);
        break;
      case 2:
        ((int16_t*)column_buffers_[column_idx])[row_idx] = static_cast<int16_t>(val);
        break;
      case 4:
        ((int32_t*)column_buffers_[column_idx])[row_idx] = static_cast<int32_t>(val);
        break;
      case 8:
        ((int64_t*)column_buffers_[column_idx])[row_idx] = val;
        break;
      default:
        CHECK(false);
    }
  } else {
    CHECK(target_types_[column_idx].is_fp());
    switch (target_types_[column_idx].get_type()) {
      case kFLOAT: {
        auto float_p = boost::get<float>(scalar_col_val);
        ((float*)column_buffers_[column_idx])[row_idx] = static_cast<float>(*float_p);
        break;
      }
      case kDOUBLE: {
        auto double_p = boost::get<double>(scalar_col_val);
        ((double*)column_buffers_[column_idx])[row_idx] = static_cast<double>(*double_p);
        break;
      }
      default:
        CHECK(false);
    }
  }
}

/**
 * A set of write functions to be used to directly write into final column_buffers_.
 * The read_from_function is used to read from the input result set's storage
 * NOTE: currently only used for direct columnarizations
 */
template <typename DATA_TYPE>
void ColumnarResults::writeBackCellDirect(const ResultSet& rows,
                                          const size_t input_buffer_entry_idx,
                                          const size_t output_buffer_entry_idx,
                                          const size_t target_idx,
                                          const size_t slot_idx,
                                          const ReadFunction& read_from_function) {
  const auto val = static_cast<DATA_TYPE>(fixed_encoding_nullable_val(
      read_from_function(rows, input_buffer_entry_idx, target_idx, slot_idx),
      target_types_[target_idx]));
  reinterpret_cast<DATA_TYPE*>(column_buffers_[target_idx])[output_buffer_entry_idx] =
      val;
}

template <>
void ColumnarResults::writeBackCellDirect<float>(const ResultSet& rows,
                                                 const size_t input_buffer_entry_idx,
                                                 const size_t output_buffer_entry_idx,
                                                 const size_t target_idx,
                                                 const size_t slot_idx,
                                                 const ReadFunction& read_from_function) {
  const int32_t ival =
      read_from_function(rows, input_buffer_entry_idx, target_idx, slot_idx);
  const float fval = *reinterpret_cast<const float*>(may_alias_ptr(&ival));
  reinterpret_cast<float*>(column_buffers_[target_idx])[output_buffer_entry_idx] = fval;
}

template <>
void ColumnarResults::writeBackCellDirect<double>(
    const ResultSet& rows,
    const size_t input_buffer_entry_idx,
    const size_t output_buffer_entry_idx,
    const size_t target_idx,
    const size_t slot_idx,
    const ReadFunction& read_from_function) {
  const int64_t ival =
      read_from_function(rows, input_buffer_entry_idx, target_idx, slot_idx);
  const double dval = *reinterpret_cast<const double*>(may_alias_ptr(&ival));
  reinterpret_cast<double*>(column_buffers_[target_idx])[output_buffer_entry_idx] = dval;
}

/**
 * This function materializes all columns from the main storage and all appended storages
 * and form a single continguous column for each output column. Depending on whether the
 * column is lazily fetched or not, it will treat them differently.
 *
 * NOTE: this function should
 * only be used when the result set is columnar and completely compacted (e.g., in
 * columnar projections).
 */
void ColumnarResults::materializeAllColumnsDirectly(const ResultSet& rows,
                                                    const size_t num_columns) {
  CHECK(isDirectColumnarConversionPossible());
  switch (rows.getQueryDescriptionType()) {
    case QueryDescriptionType::Projection: {
      materializeAllColumnsProjection(rows, num_columns);
      break;
    }
    case QueryDescriptionType::GroupByPerfectHash:
    case QueryDescriptionType::GroupByBaselineHash: {
      materializeAllColumnsGroupBy(rows, num_columns);
      break;
    }
    default:
      UNREACHABLE()
          << "Direct columnar conversion for this query type is not supported yet.";
  }
}

/**
 * This function handles materialization for two types of columns in columnar projections:
 * 1. for all non-lazy columns, it directly copies the results from the result set's
 * storage into the output column buffers
 * 2. for all lazy fetched columns, it uses result set's iterators to decode the proper
 * values before storing them into the output column buffers
 */
void ColumnarResults::materializeAllColumnsProjection(const ResultSet& rows,
                                                      const size_t num_columns) {
  CHECK(rows.query_mem_desc_.didOutputColumnar());
  CHECK(isDirectColumnarConversionPossible() &&
        rows.query_mem_desc_.getQueryDescriptionType() ==
            QueryDescriptionType::Projection);

  const auto& lazy_fetch_info = rows.getLazyFetchInfo();

  // We can directly copy each non-lazy column's content
  copyAllNonLazyColumns(lazy_fetch_info, rows, num_columns);

  // Only lazy columns are iterated through first and then materialized
  materializeAllLazyColumns(lazy_fetch_info, rows, num_columns);
}

/*
 * For all non-lazy columns, we can directly copy back the results of each column's
 * contents from different storages and put them into the corresponding output buffer.
 *
 * This function is parallelized through assigning each column to a CPU thread.
 */
void ColumnarResults::copyAllNonLazyColumns(
    const std::vector<ColumnLazyFetchInfo>& lazy_fetch_info,
    const ResultSet& rows,
    const size_t num_columns) {
  CHECK(isDirectColumnarConversionPossible());
  const auto is_column_non_lazily_fetched = [&lazy_fetch_info](const size_t col_idx) {
    // Saman: make sure when this lazy_fetch_info is empty
    if (lazy_fetch_info.empty()) {
      return true;
    } else {
      return !lazy_fetch_info[col_idx].is_lazily_fetched;
    }
  };

  // parallelized by assigning each column to a thread
  std::vector<utils::future<void>> direct_copy_threads;
  for (size_t col_idx = 0; col_idx < num_columns; col_idx++) {
    if (rows.isZeroCopyColumnarConversionPossible(col_idx)) {
      CHECK(!column_buffers_[col_idx]);
      column_buffers_[col_idx] = const_cast<int8_t*>(rows.getColumnarBuffer(col_idx));
    } else if (is_column_non_lazily_fetched(col_idx)) {
      direct_copy_threads.push_back(utils::async(
          [&rows, this](const size_t column_index) {
            const size_t column_size = num_rows_ * target_types_[column_index].get_size();
            rows.copyColumnIntoBuffer(
                column_index, column_buffers_[column_index], column_size);
          },
          col_idx));
    }
  }

  for (auto& child : direct_copy_threads) {
    child.wait();
  }
}

/**
 * For all lazy fetched columns, we should iterate through the column's content and
 * properly materialize it.
 *
 * This function is parallelized through dividing total rows among all existing threads.
 * Since there's no invalid element in the result set (e.g., columnar projections), the
 * output buffer will have as many rows as there are in the result set, removing the need
 * for atomicly incrementing the output buffer position.
 */
void ColumnarResults::materializeAllLazyColumns(
    const std::vector<ColumnLazyFetchInfo>& lazy_fetch_info,
    const ResultSet& rows,
    const size_t num_columns) {
  CHECK(isDirectColumnarConversionPossible());
  const auto do_work_just_lazy_columns = [num_columns, this](
                                             const std::vector<TargetValue>& crt_row,
                                             const size_t row_idx,
                                             const std::vector<bool>& targets_to_skip) {
    for (size_t i = 0; i < num_columns; ++i) {
      if (!targets_to_skip.empty() && !targets_to_skip[i]) {
        writeBackCell(crt_row[i], row_idx, i);
      }
    }
  };

  const auto contains_lazy_fetched_column =
      [](const std::vector<ColumnLazyFetchInfo>& lazy_fetch_info) {
        for (auto& col_info : lazy_fetch_info) {
          if (col_info.is_lazily_fetched) {
            return true;
          }
        }
        return false;
      };

  // parallelized by assigning a chunk of rows to each thread)
  const bool skip_non_lazy_columns = rows.isPermutationBufferEmpty();
  if (contains_lazy_fetched_column(lazy_fetch_info)) {
    const size_t worker_count =
        result_set::use_parallel_algorithms(rows) ? cpu_threads() : 1;
    std::vector<utils::future<void>> conversion_threads;
    std::vector<bool> targets_to_skip;
    if (skip_non_lazy_columns) {
      CHECK_EQ(lazy_fetch_info.size(), size_t(num_columns));
      targets_to_skip.reserve(num_columns);
      for (size_t i = 0; i < num_columns; i++) {
        // we process lazy columns (i.e., skip non-lazy columns)
        targets_to_skip.push_back(!lazy_fetch_info[i].is_lazily_fetched);
      }
    }
    for (auto interval : makeIntervals(size_t(0), rows.entryCount(), worker_count)) {
      conversion_threads.push_back(utils::async(
          [&rows, &do_work_just_lazy_columns, &targets_to_skip](const size_t start,
                                                                const size_t end) {
            for (size_t i = start; i < end; ++i) {
              const auto crt_row = rows.getRowAtNoTranslations(i, targets_to_skip);
              do_work_just_lazy_columns(crt_row, i, targets_to_skip);
            }
          },
          interval.begin,
          interval.end));
    }

    for (auto& child : conversion_threads) {
      child.wait();
    }
  }
}

/**
 * This function is to directly columnarize a result set for group by queries.
 * Its main difference with the traditional alternative is that it directly reads
 * non-empty entries from the result set, and then writes them into output column buffers,
 * rather than using the result set's iterators.
 */
void ColumnarResults::materializeAllColumnsGroupBy(const ResultSet& rows,
                                                   const size_t num_columns) {
  CHECK(isDirectColumnarConversionPossible());
  CHECK(rows.getQueryDescriptionType() == QueryDescriptionType::GroupByPerfectHash ||
        rows.getQueryDescriptionType() == QueryDescriptionType::GroupByBaselineHash);

  const size_t num_threads = isParallelConversion() ? cpu_threads() : 1;
  const size_t entry_count = rows.entryCount();
  const size_t size_per_thread = (entry_count + num_threads - 1) / num_threads;

  // step 1: compute total non-empty elements and store a bitmap per thread
  std::vector<size_t> non_empty_per_thread(num_threads,
                                           0);  // number of non-empty entries per thread

  ColumnBitmap bitmap(size_per_thread, num_threads);

  locateAndCountEntries(
      rows, bitmap, non_empty_per_thread, entry_count, num_threads, size_per_thread);

  // step 2: go through the generated bitmap and copy/decode corresponding entries
  // into the output buffer
  compactAndCopyEntries(rows,
                        bitmap,
                        non_empty_per_thread,
                        num_columns,
                        entry_count,
                        num_threads,
                        size_per_thread);
}

/**
 * This function goes through all the keys in the result set, and count the total number
 * of non-empty keys. It also store the location of non-empty keys in a bitmap data
 * structure for later faster access.
 */
void ColumnarResults::locateAndCountEntries(const ResultSet& rows,
                                            ColumnBitmap& bitmap,
                                            std::vector<size_t>& non_empty_per_thread,
                                            const size_t entry_count,
                                            const size_t num_threads,
                                            const size_t size_per_thread) const {
  CHECK(isDirectColumnarConversionPossible());
  CHECK(rows.getQueryDescriptionType() == QueryDescriptionType::GroupByPerfectHash ||
        rows.getQueryDescriptionType() == QueryDescriptionType::GroupByBaselineHash);
  CHECK_EQ(num_threads, non_empty_per_thread.size());
  auto locate_and_count_func =
      [&rows, &bitmap, &non_empty_per_thread](
          size_t start_index, size_t end_index, size_t thread_idx) {
        size_t total_non_empty = 0;
        size_t local_idx = 0;
        for (size_t entry_idx = start_index; entry_idx < end_index;
             entry_idx++, local_idx++) {
          if (!rows.isRowAtEmpty(entry_idx)) {
            total_non_empty++;
            bitmap.set(local_idx, thread_idx, true);
          }
        }
        non_empty_per_thread[thread_idx] = total_non_empty;
      };

  std::vector<utils::future<void>> conversion_threads;
  for (size_t thread_idx = 0; thread_idx < num_threads; thread_idx++) {
    const size_t start_entry = thread_idx * size_per_thread;
    const size_t end_entry = std::min(start_entry + size_per_thread, entry_count);
    conversion_threads.push_back(
        utils::async(locate_and_count_func, start_entry, end_entry, thread_idx));
  }

  for (auto& child : conversion_threads) {
    child.wait();
  }
}

/**
 * This function goes through all non-empty elements marked in the bitmap data structure,
 * and store them back into output column buffers. The output column buffers are compacted
 * without any holes in it.
 *
 * TODO(Saman): if necessary, we can look into the distribution of non-empty entries
 * and choose a different load-balanced strategy (assigning equal number of non-empties
 * to each thread) as opposed to equal partitioning of the bitmap
 */
void ColumnarResults::compactAndCopyEntries(
    const ResultSet& rows,
    const ColumnBitmap& bitmap,
    const std::vector<size_t>& non_empty_per_thread,
    const size_t num_columns,
    const size_t entry_count,
    const size_t num_threads,
    const size_t size_per_thread) {
  CHECK(isDirectColumnarConversionPossible());
  CHECK(rows.getQueryDescriptionType() == QueryDescriptionType::GroupByPerfectHash ||
        rows.getQueryDescriptionType() == QueryDescriptionType::GroupByBaselineHash);
  CHECK_EQ(num_threads, non_empty_per_thread.size());

  // compute the exclusive scan over all non-empty totals
  std::vector<size_t> global_offsets(num_threads + 1, 0);
  std::partial_sum(non_empty_per_thread.begin(),
                   non_empty_per_thread.end(),
                   std::next(global_offsets.begin()));

  const auto slot_idx_per_target_idx = rows.getSlotIndicesForTargetIndices();
  const auto [single_slot_targets_to_skip, num_single_slot_targets] =
      rows.getSupportedSingleSlotTargetBitmap();

  // We skip multi-slot targets (e.g., AVG). These skipped targets are treated
  // differently and accessed through result set's iterator
  if (num_single_slot_targets < num_columns) {
    compactAndCopyEntriesWithTargetSkipping(rows,
                                            bitmap,
                                            non_empty_per_thread,
                                            global_offsets,
                                            single_slot_targets_to_skip,
                                            slot_idx_per_target_idx,
                                            num_columns,
                                            entry_count,
                                            num_threads,
                                            size_per_thread);
  } else {
    compactAndCopyEntriesWithoutTargetSkipping(rows,
                                               bitmap,
                                               non_empty_per_thread,
                                               global_offsets,
                                               slot_idx_per_target_idx,
                                               num_columns,
                                               entry_count,
                                               num_threads,
                                               size_per_thread);
  }
}

/**
 * This functions takes a bitmap of non-empty entries within the result set's storage
 * and compact and copy those contents back into the output column_buffers_.
 * In this variation, multi-slot targets (e.g., AVG) are treated with the existing
 * result set's iterations, but everything else is directly columnarized.
 */
void ColumnarResults::compactAndCopyEntriesWithTargetSkipping(
    const ResultSet& rows,
    const ColumnBitmap& bitmap,
    const std::vector<size_t>& non_empty_per_thread,
    const std::vector<size_t>& global_offsets,
    const std::vector<bool>& targets_to_skip,
    const std::vector<size_t>& slot_idx_per_target_idx,
    const size_t num_columns,
    const size_t entry_count,
    const size_t num_threads,
    const size_t size_per_thread) {
  CHECK(isDirectColumnarConversionPossible());
  CHECK(rows.getQueryDescriptionType() == QueryDescriptionType::GroupByPerfectHash ||
        rows.getQueryDescriptionType() == QueryDescriptionType::GroupByBaselineHash);

  const auto [write_functions, read_functions] =
      initAllConversionFunctions(rows, slot_idx_per_target_idx, targets_to_skip);
  CHECK_EQ(write_functions.size(), num_columns);
  CHECK_EQ(read_functions.size(), num_columns);

  auto compact_buffer_func = [this,
                              &rows,
                              &bitmap,
                              &global_offsets,
                              &non_empty_per_thread,
                              &num_columns,
                              &targets_to_skip,
                              &slot_idx_per_target_idx,
                              &write_functions = write_functions,
                              &read_functions = read_functions](const size_t start_index,
                                                                const size_t end_index,
                                                                const size_t thread_idx) {
    const size_t total_non_empty = non_empty_per_thread[thread_idx];
    size_t non_empty_idx = 0;
    size_t local_idx = 0;
    for (size_t entry_idx = start_index; entry_idx < end_index;
         entry_idx++, local_idx++) {
      if (non_empty_idx >= total_non_empty) {
        // all non-empty entries has been written back
        break;
      }
      const size_t output_buffer_row_idx = global_offsets[thread_idx] + non_empty_idx;
      if (bitmap.get(local_idx, thread_idx)) {
        // targets that are recovered from the result set iterators:
        const auto crt_row = rows.getRowAtNoTranslations(entry_idx, targets_to_skip);
        for (size_t column_idx = 0; column_idx < num_columns; ++column_idx) {
          if (!targets_to_skip.empty() && !targets_to_skip[column_idx]) {
            writeBackCell(crt_row[column_idx], output_buffer_row_idx, column_idx);
          }
        }
        // targets that are copied directly without any translation/decoding from
        // result set
        for (size_t column_idx = 0; column_idx < num_columns; column_idx++) {
          if (!targets_to_skip.empty() && !targets_to_skip[column_idx]) {
            continue;
          }
          write_functions[column_idx](rows,
                                      entry_idx,
                                      output_buffer_row_idx,
                                      column_idx,
                                      slot_idx_per_target_idx[column_idx],
                                      read_functions[column_idx]);
        }
        non_empty_idx++;
      } else {
        continue;
      }
    }
  };

  std::vector<utils::future<void>> compaction_threads;
  for (size_t thread_idx = 0; thread_idx < num_threads; thread_idx++) {
    const size_t start_entry = thread_idx * size_per_thread;
    const size_t end_entry = std::min(start_entry + size_per_thread, entry_count);
    compaction_threads.push_back(
        utils::async(compact_buffer_func, start_entry, end_entry, thread_idx));
  }

  for (auto& child : compaction_threads) {
    child.wait();
  }
}

/**
 * This functions takes a bitmap of non-empty entries within the result set's storage
 * and compact and copy those contents back into the output column_buffers_.
 * In this variation, all targets are assumed to be single-slot and thus can be directly
 * columnarized.
 */
void ColumnarResults::compactAndCopyEntriesWithoutTargetSkipping(
    const ResultSet& rows,
    const ColumnBitmap& bitmap,
    const std::vector<size_t>& non_empty_per_thread,
    const std::vector<size_t>& global_offsets,
    const std::vector<size_t>& slot_idx_per_target_idx,
    const size_t num_columns,
    const size_t entry_count,
    const size_t num_threads,
    const size_t size_per_thread) {
  CHECK(isDirectColumnarConversionPossible());
  CHECK(rows.getQueryDescriptionType() == QueryDescriptionType::GroupByPerfectHash ||
        rows.getQueryDescriptionType() == QueryDescriptionType::GroupByBaselineHash);

  const auto [write_functions, read_functions] =
      initAllConversionFunctions(rows, slot_idx_per_target_idx);
  CHECK_EQ(write_functions.size(), num_columns);
  CHECK_EQ(read_functions.size(), num_columns);

  auto compact_buffer_func = [&rows,
                              &bitmap,
                              &global_offsets,
                              &non_empty_per_thread,
                              &num_columns,
                              &slot_idx_per_target_idx,
                              &write_functions = write_functions,
                              &read_functions = read_functions](const size_t start_index,
                                                                const size_t end_index,
                                                                const size_t thread_idx) {
    const size_t total_non_empty = non_empty_per_thread[thread_idx];
    size_t non_empty_idx = 0;
    size_t local_idx = 0;
    for (size_t entry_idx = start_index; entry_idx < end_index;
         entry_idx++, local_idx++) {
      if (non_empty_idx >= total_non_empty) {
        // all non-empty entries has been written back
        break;
      }
      const size_t output_buffer_row_idx = global_offsets[thread_idx] + non_empty_idx;
      if (bitmap.get(local_idx, thread_idx)) {
        for (size_t column_idx = 0; column_idx < num_columns; column_idx++) {
          write_functions[column_idx](rows,
                                      entry_idx,
                                      output_buffer_row_idx,
                                      column_idx,
                                      slot_idx_per_target_idx[column_idx],
                                      read_functions[column_idx]);
        }
        non_empty_idx++;
      } else {
        continue;
      }
    }
  };

  std::vector<utils::future<void>> compaction_threads;
  for (size_t thread_idx = 0; thread_idx < num_threads; thread_idx++) {
    const size_t start_entry = thread_idx * size_per_thread;
    const size_t end_entry = std::min(start_entry + size_per_thread, entry_count);
    compaction_threads.push_back(
        utils::async(compact_buffer_func, start_entry, end_entry, thread_idx));
  }

  for (auto& child : compaction_threads) {
    child.wait();
  }
}

/**
 * Initialize a set of write functions per target (i.e., column). Target types' logical
 * size are used to categorize the correct write function per target. These functions are
 * then used for every row in the result set.
 */
std::vector<ColumnarResults::WriteFunction> ColumnarResults::initWriteFunctions(
    const ResultSet& rows,
    const std::vector<bool>& targets_to_skip) {
  CHECK(isDirectColumnarConversionPossible());
  CHECK(rows.getQueryDescriptionType() == QueryDescriptionType::GroupByPerfectHash ||
        rows.getQueryDescriptionType() == QueryDescriptionType::GroupByBaselineHash);

  std::vector<WriteFunction> result;
  result.reserve(target_types_.size());

  for (size_t target_idx = 0; target_idx < target_types_.size(); target_idx++) {
    if (!targets_to_skip.empty() && !targets_to_skip[target_idx]) {
      result.emplace_back([](const ResultSet& rows,
                             const size_t input_buffer_entry_idx,
                             const size_t output_buffer_entry_idx,
                             const size_t target_idx,
                             const size_t slot_idx,
                             const ReadFunction& read_function) {
        UNREACHABLE() << "Invalid write back function used.";
      });
      continue;
    }

    if (target_types_[target_idx].is_fp()) {
      switch (target_types_[target_idx].get_size()) {
        case 8:
          result.emplace_back(std::bind(&ColumnarResults::writeBackCellDirect<double>,
                                        this,
                                        std::placeholders::_1,
                                        std::placeholders::_2,
                                        std::placeholders::_3,
                                        std::placeholders::_4,
                                        std::placeholders::_5,
                                        std::placeholders::_6));
          break;
        case 4:
          result.emplace_back(std::bind(&ColumnarResults::writeBackCellDirect<float>,
                                        this,
                                        std::placeholders::_1,
                                        std::placeholders::_2,
                                        std::placeholders::_3,
                                        std::placeholders::_4,
                                        std::placeholders::_5,
                                        std::placeholders::_6));
          break;
        default:
          UNREACHABLE() << "Invalid target type encountered.";
          break;
      }
    } else {
      switch (target_types_[target_idx].get_size()) {
        case 8:
          result.emplace_back(std::bind(&ColumnarResults::writeBackCellDirect<int64_t>,
                                        this,
                                        std::placeholders::_1,
                                        std::placeholders::_2,
                                        std::placeholders::_3,
                                        std::placeholders::_4,
                                        std::placeholders::_5,
                                        std::placeholders::_6));
          break;
        case 4:
          result.emplace_back(std::bind(&ColumnarResults::writeBackCellDirect<int32_t>,
                                        this,
                                        std::placeholders::_1,
                                        std::placeholders::_2,
                                        std::placeholders::_3,
                                        std::placeholders::_4,
                                        std::placeholders::_5,
                                        std::placeholders::_6));
          break;
        case 2:
          result.emplace_back(std::bind(&ColumnarResults::writeBackCellDirect<int16_t>,
                                        this,
                                        std::placeholders::_1,
                                        std::placeholders::_2,
                                        std::placeholders::_3,
                                        std::placeholders::_4,
                                        std::placeholders::_5,
                                        std::placeholders::_6));
          break;
        case 1:
          result.emplace_back(std::bind(&ColumnarResults::writeBackCellDirect<int8_t>,
                                        this,
                                        std::placeholders::_1,
                                        std::placeholders::_2,
                                        std::placeholders::_3,
                                        std::placeholders::_4,
                                        std::placeholders::_5,
                                        std::placeholders::_6));
          break;
        default:
          UNREACHABLE() << "Invalid target type encountered.";
          break;
      }
    }
  }
  return result;
}

namespace {

int64_t invalid_read_func(const ResultSet& rows,
                          const size_t input_buffer_entry_idx,
                          const size_t target_idx,
                          const size_t slot_idx) {
  UNREACHABLE() << "Invalid read function used, target should have been skipped.";
  return static_cast<int64_t>(0);
}

template <QueryDescriptionType QUERY_TYPE, bool COLUMNAR_OUTPUT>
int64_t read_float_key_baseline(const ResultSet& rows,
                                const size_t input_buffer_entry_idx,
                                const size_t target_idx,
                                const size_t slot_idx) {
  // float keys in baseline hash are written as doubles in the buffer, so
  // the result should properly be casted before being written in the output
  // columns
  auto fval = static_cast<float>(rows.getEntryAt<double, QUERY_TYPE, COLUMNAR_OUTPUT>(
      input_buffer_entry_idx, target_idx, slot_idx));
  return *reinterpret_cast<int32_t*>(may_alias_ptr(&fval));
}

template <QueryDescriptionType QUERY_TYPE, bool COLUMNAR_OUTPUT>
int64_t read_int64_func(const ResultSet& rows,
                        const size_t input_buffer_entry_idx,
                        const size_t target_idx,
                        const size_t slot_idx) {
  return rows.getEntryAt<int64_t, QUERY_TYPE, COLUMNAR_OUTPUT>(
      input_buffer_entry_idx, target_idx, slot_idx);
}

template <QueryDescriptionType QUERY_TYPE, bool COLUMNAR_OUTPUT>
int64_t read_int32_func(const ResultSet& rows,
                        const size_t input_buffer_entry_idx,
                        const size_t target_idx,
                        const size_t slot_idx) {
  return rows.getEntryAt<int32_t, QUERY_TYPE, COLUMNAR_OUTPUT>(
      input_buffer_entry_idx, target_idx, slot_idx);
}

template <QueryDescriptionType QUERY_TYPE, bool COLUMNAR_OUTPUT>
int64_t read_int16_func(const ResultSet& rows,
                        const size_t input_buffer_entry_idx,
                        const size_t target_idx,
                        const size_t slot_idx) {
  return rows.getEntryAt<int16_t, QUERY_TYPE, COLUMNAR_OUTPUT>(
      input_buffer_entry_idx, target_idx, slot_idx);
}

template <QueryDescriptionType QUERY_TYPE, bool COLUMNAR_OUTPUT>
int64_t read_int8_func(const ResultSet& rows,
                       const size_t input_buffer_entry_idx,
                       const size_t target_idx,
                       const size_t slot_idx) {
  return rows.getEntryAt<int8_t, QUERY_TYPE, COLUMNAR_OUTPUT>(
      input_buffer_entry_idx, target_idx, slot_idx);
}

template <QueryDescriptionType QUERY_TYPE, bool COLUMNAR_OUTPUT>
int64_t read_float_func(const ResultSet& rows,
                        const size_t input_buffer_entry_idx,
                        const size_t target_idx,
                        const size_t slot_idx) {
  auto fval = rows.getEntryAt<float, QUERY_TYPE, COLUMNAR_OUTPUT>(
      input_buffer_entry_idx, target_idx, slot_idx);
  return *reinterpret_cast<int32_t*>(may_alias_ptr(&fval));
}

template <QueryDescriptionType QUERY_TYPE, bool COLUMNAR_OUTPUT>
int64_t read_double_func(const ResultSet& rows,
                         const size_t input_buffer_entry_idx,
                         const size_t target_idx,
                         const size_t slot_idx) {
  auto dval = rows.getEntryAt<double, QUERY_TYPE, COLUMNAR_OUTPUT>(
      input_buffer_entry_idx, target_idx, slot_idx);
  return *reinterpret_cast<int64_t*>(may_alias_ptr(&dval));
}

}  // namespace

/**
 * Initializes a set of read funtions to properly access the contents of the result set's
 * storage buffer. Each particular read function is chosen based on the data type and data
 * size used to store that target in the result set's storage buffer. These functions are
 * then used for each row in the result set.
 */
template <QueryDescriptionType QUERY_TYPE, bool COLUMNAR_OUTPUT>
std::vector<ColumnarResults::ReadFunction> ColumnarResults::initReadFunctions(
    const ResultSet& rows,
    const std::vector<size_t>& slot_idx_per_target_idx,
    const std::vector<bool>& targets_to_skip) {
  CHECK(isDirectColumnarConversionPossible());
  CHECK(COLUMNAR_OUTPUT == rows.didOutputColumnar());
  CHECK(QUERY_TYPE == rows.getQueryDescriptionType());

  std::vector<ReadFunction> read_functions;
  read_functions.reserve(target_types_.size());

  for (size_t target_idx = 0; target_idx < target_types_.size(); target_idx++) {
    if (!targets_to_skip.empty() && !targets_to_skip[target_idx]) {
      // for targets that should be skipped, we use a placeholder function that should
      // never be called. The CHECKs inside it make sure that never happens.
      read_functions.emplace_back(invalid_read_func);
      continue;
    }

    if (QUERY_TYPE == QueryDescriptionType::GroupByBaselineHash) {
      if (rows.getPaddedSlotWidthBytes(slot_idx_per_target_idx[target_idx]) == 0) {
        // for key columns only
        CHECK(rows.query_mem_desc_.getTargetGroupbyIndex(target_idx) >= 0);
        if (target_types_[target_idx].is_fp()) {
          CHECK_EQ(size_t(8), rows.query_mem_desc_.getEffectiveKeyWidth());
          switch (target_types_[target_idx].get_type()) {
            case kFLOAT:
              read_functions.emplace_back(
                  read_float_key_baseline<QUERY_TYPE, COLUMNAR_OUTPUT>);
              break;
            case kDOUBLE:
              read_functions.emplace_back(read_double_func<QUERY_TYPE, COLUMNAR_OUTPUT>);
              break;
            default:
              UNREACHABLE()
                  << "Invalid data type encountered (BaselineHash, floating point key).";
              break;
          }
        } else {
          switch (rows.query_mem_desc_.getEffectiveKeyWidth()) {
            case 8:
              read_functions.emplace_back(read_int64_func<QUERY_TYPE, COLUMNAR_OUTPUT>);
              break;
            case 4:
              read_functions.emplace_back(read_int32_func<QUERY_TYPE, COLUMNAR_OUTPUT>);
              break;
            default:
              UNREACHABLE()
                  << "Invalid data type encountered (BaselineHash, integer key).";
          }
        }
        continue;
      }
    }
    if (target_types_[target_idx].is_fp()) {
      switch (rows.getPaddedSlotWidthBytes(slot_idx_per_target_idx[target_idx])) {
        case 8:
          read_functions.emplace_back(read_double_func<QUERY_TYPE, COLUMNAR_OUTPUT>);
          break;
        case 4:
          read_functions.emplace_back(read_float_func<QUERY_TYPE, COLUMNAR_OUTPUT>);
          break;
        default:
          UNREACHABLE() << "Invalid data type encountered (floating point agg column).";
          break;
      }
    } else {
      switch (rows.getPaddedSlotWidthBytes(slot_idx_per_target_idx[target_idx])) {
        case 8:
          read_functions.emplace_back(read_int64_func<QUERY_TYPE, COLUMNAR_OUTPUT>);
          break;
        case 4:
          read_functions.emplace_back(read_int32_func<QUERY_TYPE, COLUMNAR_OUTPUT>);
          break;
        case 2:
          read_functions.emplace_back(read_int16_func<QUERY_TYPE, COLUMNAR_OUTPUT>);
          break;
        case 1:
          read_functions.emplace_back(read_int8_func<QUERY_TYPE, COLUMNAR_OUTPUT>);
          break;
        default:
          UNREACHABLE() << "Invalid data type encountered (integer agg column).";
          break;
      }
    }
  }
  return read_functions;
}

/**
 * This function goes through all target types in the output, and chooses appropriate
 * write and read functions per target. The goal is then to simply use these functions
 * for each row and per target. Read functions are used to read each cell's data content
 * (particular target in a row), and write functions are used to properly write back the
 * cell's content into the output column buffers.
 */
std::tuple<std::vector<ColumnarResults::WriteFunction>,
           std::vector<ColumnarResults::ReadFunction>>
ColumnarResults::initAllConversionFunctions(
    const ResultSet& rows,
    const std::vector<size_t>& slot_idx_per_target_idx,
    const std::vector<bool>& targets_to_skip) {
  CHECK(isDirectColumnarConversionPossible() &&
        (rows.getQueryDescriptionType() == QueryDescriptionType::GroupByPerfectHash ||
         rows.getQueryDescriptionType() == QueryDescriptionType::GroupByBaselineHash));

  const auto write_functions = initWriteFunctions(rows, targets_to_skip);
  if (rows.getQueryDescriptionType() == QueryDescriptionType::GroupByPerfectHash) {
    if (rows.didOutputColumnar()) {
      return std::make_tuple(
          std::move(write_functions),
          initReadFunctions<QueryDescriptionType::GroupByPerfectHash, true>(
              rows, slot_idx_per_target_idx, targets_to_skip));
    } else {
      return std::make_tuple(
          std::move(write_functions),
          initReadFunctions<QueryDescriptionType::GroupByPerfectHash, false>(
              rows, slot_idx_per_target_idx, targets_to_skip));
    }
  } else {
    if (rows.didOutputColumnar()) {
      return std::make_tuple(
          std::move(write_functions),
          initReadFunctions<QueryDescriptionType::GroupByBaselineHash, true>(
              rows, slot_idx_per_target_idx, targets_to_skip));
    } else {
      return std::make_tuple(
          std::move(write_functions),
          initReadFunctions<QueryDescriptionType::GroupByBaselineHash, false>(
              rows, slot_idx_per_target_idx, targets_to_skip));
    }
  }
}
