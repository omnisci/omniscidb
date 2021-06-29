#include "BufferCompaction.h"
#include "GpuMemUtils.h"
#include "GpuRtConstants.h"
#include "ResultSetBufferAccessors.h"
#include "ResultSetSortImpl.h"
#include "SortUtils.cuh"

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#define FORCE_CPU_VERSION
#include "BufferEntryUtils.h"
#undef FORCE_CPU_VERSION

namespace {

template <class K, class V, class I>
std::vector<uint32_t> do_radix_sort(const ExecutorDeviceType device_type,
                                    ThrustAllocator& thrust_allocator,
                                    const int8_t* groupby_buffer,
                                    V dev_oe_col_buffer_begin,
                                    V dev_oe_col_buffer_end,
                                    I dev_idx_buff_begin,
                                    const size_t dev_idx_buff_size,
                                    const PodOrderEntry& oe,
                                    const GroupByBufferLayoutInfo& layout,
                                    const size_t top_n) {
  if (dev_idx_buff_size == 0) {
    return {};
  }
  if (oe.is_desc) {
    if (device_type == ExecutorDeviceType::GPU) {
      thrust::sort_by_key(thrust::device(thrust_allocator),
                          dev_oe_col_buffer_begin,
                          dev_oe_col_buffer_end,
                          dev_idx_buff_begin,
                          thrust::greater<int64_t>());
    } else {
      thrust::sort_by_key(dev_oe_col_buffer_begin,
                          dev_oe_col_buffer_end,
                          dev_idx_buff_begin,
                          thrust::greater<int64_t>());
    }
  } else {
    if (device_type == ExecutorDeviceType::GPU) {
      thrust::sort_by_key(thrust::device(thrust_allocator),
                          dev_oe_col_buffer_begin,
                          dev_oe_col_buffer_end,
                          dev_idx_buff_begin);
    } else {
      thrust::sort_by_key(
          dev_oe_col_buffer_begin, dev_oe_col_buffer_end, dev_idx_buff_begin);
    }
  }
  // Speculatively transfer only the top_n first, most of the time it'll be enough.
  thrust::host_vector<uint32_t> host_vector_result(
      dev_idx_buff_begin, dev_idx_buff_begin + std::min(top_n, dev_idx_buff_size));
  // Sometimes, radix sort can bring to the front entries which are empty.
  // For example, ascending sort on COUNT(*) will bring non-existent groups
  // to the front of dev_idx_buff since they're 0 in our system. Re-do the
  // transfer in that case to bring the entire dev_idx_buff; existing logic
  // in row iteration will take care of skipping the empty rows.
  for (size_t i = 0; i < host_vector_result.size(); ++i) {
    const auto entry_idx = host_vector_result[i];
    if (is_empty_entry<K>(entry_idx, groupby_buffer, layout.row_bytes)) {
      host_vector_result = thrust::host_vector<uint32_t>(
          dev_idx_buff_begin, dev_idx_buff_begin + dev_idx_buff_size);
      break;
    }
  }
  std::vector<uint32_t> result;
  result.reserve(std::min(top_n, host_vector_result.size()));
  for (size_t i = 0; i < host_vector_result.size(); ++i) {
    const auto entry_idx = host_vector_result[i];
    if (!is_empty_entry<K>(entry_idx, groupby_buffer, layout.row_bytes)) {
      result.push_back(entry_idx);
      if (result.size() >= top_n) {
        break;
      }
    }
  }
  return result;
}

void add_nulls(std::vector<uint32_t>& idx_buff,
               const std::vector<uint32_t>& null_idx_buff,
               const PodOrderEntry& oe) {
  if (null_idx_buff.empty()) {
    return;
  }
  const auto insertion_point = oe.nulls_first ? idx_buff.begin() : idx_buff.end();
  idx_buff.insert(insertion_point, null_idx_buff.begin(), null_idx_buff.end());
}

template <typename T>
thrust::device_ptr<T> get_device_copy_ptr(const thrust::host_vector<T>& host_vec,
                                          ThrustAllocator& thrust_allocator) {
  if (host_vec.empty()) {
    return thrust::device_ptr<T>(static_cast<T*>(nullptr));
  }
  const auto host_vec_bytes = host_vec.size() * sizeof(T);
  T* dev_ptr = reinterpret_cast<T*>(
      thrust_allocator.allocateScopedBuffer(align_to_int64(host_vec_bytes)));
  copy_to_nvidia_gpu(thrust_allocator.getDataMgr(),
                     reinterpret_cast<CUdeviceptr>(dev_ptr),
                     &host_vec[0],
                     host_vec_bytes,
                     thrust_allocator.getDeviceId());
  return thrust::device_ptr<T>(dev_ptr);
}

template <class K>
std::vector<uint32_t> baseline_sort_fp(const ExecutorDeviceType device_type,
                                       const int device_id,
                                       Data_Namespace::DataMgr* data_mgr,
                                       const int8_t* groupby_buffer,
                                       const thrust::host_vector<int64_t>& oe_col_buffer,
                                       const PodOrderEntry& oe,
                                       const GroupByBufferLayoutInfo& layout,
                                       const size_t top_n,
                                       const size_t start,
                                       const size_t step) {
  thrust::host_vector<uint32_t> neg_idx_buff;
  thrust::host_vector<uint32_t> pos_idx_buff;
  std::vector<uint32_t> null_idx_buff;
  thrust::host_vector<int64_t> neg_oe_col_buffer;
  thrust::host_vector<int64_t> pos_oe_col_buffer;
  const auto slice_entry_count =
      layout.entry_count / step + (layout.entry_count % step ? 1 : 0);
  neg_idx_buff.reserve(slice_entry_count);
  pos_idx_buff.reserve(slice_entry_count);
  null_idx_buff.reserve(slice_entry_count);
  neg_oe_col_buffer.reserve(slice_entry_count);
  pos_oe_col_buffer.reserve(slice_entry_count);
  size_t oe_col_buffer_idx = 0;
  const auto& oe_info = layout.oe_target_info;
  const auto col_ti =
      oe_info.agg_kind == kAVG ? SQLTypeInfo(kDOUBLE, false) : oe_info.sql_type;
  // Execlude AVG b/c collect_order_entry_column already makes its pair collapse into a
  // double
  const bool float_argument_input =
      takes_float_argument(oe_info) && oe_info.agg_kind != kAVG;

  auto is_negative =
      float_argument_input ? [](const int64_t v) -> bool { return (v & (1 << 31)) != 0; }
  : [](const int64_t v) -> bool { return v < 0; };

  for (size_t i = start; i < layout.entry_count; i += step, ++oe_col_buffer_idx) {
    if (!is_empty_entry<K>(i, groupby_buffer, layout.row_bytes) &&
        oe_col_buffer[oe_col_buffer_idx] ==
            null_val_bit_pattern(col_ti, float_argument_input)) {
      null_idx_buff.push_back(i);
      continue;
    }
    if (is_negative(oe_col_buffer[oe_col_buffer_idx])) {  // sign bit works the same for
                                                          // integer and floating point
      neg_idx_buff.push_back(i);
      neg_oe_col_buffer.push_back(oe_col_buffer[oe_col_buffer_idx]);
    } else {
      pos_idx_buff.push_back(i);
      pos_oe_col_buffer.push_back(oe_col_buffer[oe_col_buffer_idx]);
    }
  }
  std::vector<uint32_t> pos_result;
  ThrustAllocator thrust_allocator(data_mgr, device_id);
  if (device_type == ExecutorDeviceType::GPU) {
    const auto dev_pos_idx_buff = get_device_copy_ptr(pos_idx_buff, thrust_allocator);
    const auto dev_pos_oe_col_buffer =
        get_device_copy_ptr(pos_oe_col_buffer, thrust_allocator);
    pos_result = do_radix_sort<K>(device_type,
                                  thrust_allocator,
                                  groupby_buffer,
                                  dev_pos_oe_col_buffer,
                                  dev_pos_oe_col_buffer + pos_oe_col_buffer.size(),
                                  dev_pos_idx_buff,
                                  pos_idx_buff.size(),
                                  oe,
                                  layout,
                                  top_n);
  } else {
    CHECK(device_type == ExecutorDeviceType::CPU);
    pos_result = do_radix_sort<K>(device_type,
                                  thrust_allocator,
                                  groupby_buffer,
                                  pos_oe_col_buffer.begin(),
                                  pos_oe_col_buffer.end(),
                                  pos_idx_buff.begin(),
                                  pos_idx_buff.size(),
                                  oe,
                                  layout,
                                  top_n);
  }
  std::vector<uint32_t> neg_result;
  PodOrderEntry reverse_oe{oe.tle_no, !oe.is_desc, oe.nulls_first};
  if (device_type == ExecutorDeviceType::GPU) {
    const auto dev_neg_idx_buff = get_device_copy_ptr(neg_idx_buff, thrust_allocator);
    const auto dev_neg_oe_col_buffer =
        get_device_copy_ptr(neg_oe_col_buffer, thrust_allocator);
    neg_result = do_radix_sort<K>(device_type,
                                  thrust_allocator,
                                  groupby_buffer,
                                  dev_neg_oe_col_buffer,
                                  dev_neg_oe_col_buffer + neg_oe_col_buffer.size(),
                                  dev_neg_idx_buff,
                                  neg_idx_buff.size(),
                                  reverse_oe,
                                  layout,
                                  top_n);
  } else {
    CHECK(device_type == ExecutorDeviceType::CPU);
    neg_result = do_radix_sort<K>(device_type,
                                  thrust_allocator,
                                  groupby_buffer,
                                  neg_oe_col_buffer.begin(),
                                  neg_oe_col_buffer.end(),
                                  neg_idx_buff.begin(),
                                  neg_idx_buff.size(),
                                  reverse_oe,
                                  layout,
                                  top_n);
  }
  if (oe.is_desc) {
    pos_result.insert(pos_result.end(), neg_result.begin(), neg_result.end());
    add_nulls(pos_result, null_idx_buff, oe);
    return pos_result;
  }
  neg_result.insert(neg_result.end(), pos_result.begin(), pos_result.end());
  add_nulls(neg_result, null_idx_buff, oe);
  return neg_result;
}

template <class K>
std::vector<uint32_t> baseline_sort_int(const ExecutorDeviceType device_type,
                                        const int device_id,
                                        Data_Namespace::DataMgr* data_mgr,
                                        const int8_t* groupby_buffer,
                                        const thrust::host_vector<int64_t>& oe_col_buffer,
                                        const PodOrderEntry& oe,
                                        const GroupByBufferLayoutInfo& layout,
                                        const size_t top_n,
                                        const size_t start,
                                        const size_t step) {
  const auto& entry_ti = get_compact_type(layout.oe_target_info);
  std::vector<uint32_t> null_idx_buff;
  thrust::host_vector<uint32_t> notnull_idx_buff;
  const auto slice_entry_count =
      layout.entry_count / step + (layout.entry_count % step ? 1 : 0);
  null_idx_buff.reserve(slice_entry_count);
  notnull_idx_buff.reserve(slice_entry_count);
  thrust::host_vector<int64_t> notnull_oe_col_buffer;
  notnull_oe_col_buffer.reserve(slice_entry_count);
  size_t oe_col_buffer_idx = 0;
  for (size_t i = start; i < layout.entry_count; i += step, ++oe_col_buffer_idx) {
    if (!is_empty_entry<K>(i, groupby_buffer, layout.row_bytes) &&
        oe_col_buffer[oe_col_buffer_idx] == null_val_bit_pattern(entry_ti, false)) {
      null_idx_buff.push_back(i);
    } else {
      notnull_idx_buff.push_back(i);
      notnull_oe_col_buffer.push_back(oe_col_buffer[oe_col_buffer_idx]);
    }
  }
  std::vector<uint32_t> notnull_result;
  ThrustAllocator thrust_allocator(data_mgr, device_id);
  if (device_type == ExecutorDeviceType::GPU) {
    const auto dev_notnull_idx_buff =
        get_device_copy_ptr(notnull_idx_buff, thrust_allocator);
    const auto dev_notnull_oe_col_buffer =
        get_device_copy_ptr(notnull_oe_col_buffer, thrust_allocator);
    notnull_result =
        do_radix_sort<K>(device_type,
                         thrust_allocator,
                         groupby_buffer,
                         dev_notnull_oe_col_buffer,
                         dev_notnull_oe_col_buffer + notnull_oe_col_buffer.size(),
                         dev_notnull_idx_buff,
                         notnull_idx_buff.size(),
                         oe,
                         layout,
                         top_n);
  } else {
    CHECK(device_type == ExecutorDeviceType::CPU);
    notnull_result = do_radix_sort<K>(device_type,
                                      thrust_allocator,
                                      groupby_buffer,
                                      notnull_oe_col_buffer.begin(),
                                      notnull_oe_col_buffer.end(),
                                      notnull_idx_buff.begin(),
                                      notnull_idx_buff.size(),
                                      oe,
                                      layout,
                                      top_n);
  }
  add_nulls(notnull_result, null_idx_buff, oe);
  return notnull_result;
}

template <class K>
thrust::host_vector<int64_t> collect_order_entry_column(
    const int8_t* groupby_buffer,
    const GroupByBufferLayoutInfo& layout,
    const size_t start,
    const size_t step) {
  thrust::host_vector<int64_t> oe_col_buffer;
  const auto row_ptr = groupby_buffer + start * layout.row_bytes;
  auto crt_group_ptr1 = layout.target_groupby_index >= 0
                            ? row_ptr + layout.target_groupby_index * sizeof(K)
                            : row_ptr + layout.col_off;
  const int8_t* crt_group_ptr2{nullptr};
  if (layout.oe_target_info.agg_kind == kAVG) {
    crt_group_ptr2 = crt_group_ptr1 + layout.col_bytes;
  }
  const auto entry_ti = get_compact_type(layout.oe_target_info);
  const bool float_argument_input = takes_float_argument(layout.oe_target_info);
  const auto step_bytes = layout.row_bytes * step;
  const auto col_bytes = float_argument_input ? entry_ti.get_size() : layout.col_bytes;
  for (size_t i = start; i < layout.entry_count; i += step) {
    auto val1 = read_int_from_buff(crt_group_ptr1, col_bytes > 0 ? col_bytes : sizeof(K));
    if (crt_group_ptr2) {
      const auto val2 = read_int_from_buff(crt_group_ptr2, 8);
      const auto avg_val = pair_to_double({val1, val2}, entry_ti, float_argument_input);
      val1 = *reinterpret_cast<const int64_t*>(&avg_val);
    }
    oe_col_buffer.push_back(val1);
    crt_group_ptr1 += step_bytes;
    if (crt_group_ptr2) {
      crt_group_ptr2 += step_bytes;
    }
  }
  return oe_col_buffer;
}

}  // namespace

template <class K>
std::vector<uint32_t> baseline_sort(const ExecutorDeviceType device_type,
                                    const int device_id,
                                    Data_Namespace::DataMgr* data_mgr,
                                    const int8_t* groupby_buffer,
                                    const PodOrderEntry& oe,
                                    const GroupByBufferLayoutInfo& layout,
                                    const size_t top_n,
                                    const size_t start,
                                    const size_t step) {
  auto oe_col_buffer = collect_order_entry_column<K>(groupby_buffer, layout, start, step);
  const auto& entry_ti = get_compact_type(layout.oe_target_info);
  CHECK(entry_ti.is_number());
  if (entry_ti.is_fp() || layout.oe_target_info.agg_kind == kAVG) {
    return baseline_sort_fp<K>(device_type,
                               device_id,
                               data_mgr,
                               groupby_buffer,
                               oe_col_buffer,
                               oe,
                               layout,
                               top_n,
                               start,
                               step);
  }
  // Because of how we represent nulls for integral types, they'd be at the
  // wrong position in these two cases. Separate them into a different buffer.
  if ((oe.is_desc && oe.nulls_first) || (!oe.is_desc && !oe.nulls_first)) {
    return baseline_sort_int<K>(device_type,
                                device_id,
                                data_mgr,
                                groupby_buffer,
                                oe_col_buffer,
                                oe,
                                layout,
                                top_n,
                                start,
                                step);
  }
  ThrustAllocator thrust_allocator(data_mgr, device_id);
  // Fastest path, no need to separate nulls away since they'll end up at the
  // right place as a side effect of how we're representing nulls.
  if (device_type == ExecutorDeviceType::GPU) {
    if (oe_col_buffer.empty()) {
      return {};
    }
    const auto dev_idx_buff =
        get_device_ptr<uint32_t>(oe_col_buffer.size(), thrust_allocator);
    thrust::sequence(dev_idx_buff, dev_idx_buff + oe_col_buffer.size(), start, step);
    const auto dev_oe_col_buffer = get_device_copy_ptr(oe_col_buffer, thrust_allocator);
    return do_radix_sort<K>(device_type,
                            thrust_allocator,
                            groupby_buffer,
                            dev_oe_col_buffer,
                            dev_oe_col_buffer + oe_col_buffer.size(),
                            dev_idx_buff,
                            oe_col_buffer.size(),
                            oe,
                            layout,
                            top_n);
  }
  CHECK(device_type == ExecutorDeviceType::CPU);
  thrust::host_vector<uint32_t> host_idx_buff(oe_col_buffer.size());
  thrust::sequence(host_idx_buff.begin(), host_idx_buff.end(), start, step);
  return do_radix_sort<K>(device_type,
                          thrust_allocator,
                          groupby_buffer,
                          oe_col_buffer.begin(),
                          oe_col_buffer.end(),
                          host_idx_buff.begin(),
                          host_idx_buff.size(),
                          oe,
                          layout,
                          top_n);
}

template std::vector<uint32_t> baseline_sort<int32_t>(
    const ExecutorDeviceType device_type,
    const int device_id,
    Data_Namespace::DataMgr* data_mgr,
    const int8_t* groupby_buffer,
    const PodOrderEntry& oe,
    const GroupByBufferLayoutInfo& layout,
    const size_t top_n,
    const size_t start,
    const size_t step);

template std::vector<uint32_t> baseline_sort<int64_t>(
    const ExecutorDeviceType device_type,
    const int device_id,
    Data_Namespace::DataMgr* data_mgr,
    const int8_t* groupby_buffer,
    const PodOrderEntry& oe,
    const GroupByBufferLayoutInfo& layout,
    const size_t top_n,
    const size_t start,
    const size_t step);
