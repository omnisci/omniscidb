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

#ifndef QUERYENGINE_RUNTIMEFUNCTIONS_H
#define QUERYENGINE_RUNTIMEFUNCTIONS_H

#include "Shared/funcannotations.h"

#include <cassert>
#include <cstdint>
#include <ctime>
#include <limits>
#include <type_traits>

extern "C" int64_t agg_sum(int64_t* agg, const int64_t val);

extern "C" int64_t agg_sum_cpu_shared(int64_t* agg, const int64_t val);

extern "C" void agg_max(int64_t* agg, const int64_t val);

extern "C" void agg_min(int64_t* agg, const int64_t val);

extern "C" void agg_sum_double(int64_t* agg, const double val);

extern "C" void agg_max_double(int64_t* agg, const double val);

extern "C" void agg_min_double(int64_t* agg, const double val);

extern "C" int32_t agg_sum_int32_skip_val(int32_t* agg,
                                          const int32_t val,
                                          const int32_t skip_val);

extern "C" int64_t agg_sum_skip_val(int64_t* agg,
                                    const int64_t val,
                                    const int64_t skip_val);
extern "C" int64_t agg_sum_skip_val_cpu_shared(int64_t* agg,
                                               const int64_t val,
                                               const int64_t skip_val);

extern "C" void agg_max_skip_val(int64_t* agg, const int64_t val, const int64_t skip_val);

extern "C" void agg_min_skip_val(int64_t* agg, const int64_t val, const int64_t skip_val);

extern "C" void agg_sum_float_skip_val(int32_t* agg,
                                       const float val,
                                       const float skip_val);

extern "C" void agg_sum_double_skip_val(int64_t* agg,
                                        const double val,
                                        const double skip_val);

extern "C" void agg_max_double_skip_val(int64_t* agg,
                                        const double val,
                                        const double skip_val);

extern "C" void agg_min_double_skip_val(int64_t* agg,
                                        const double val,
                                        const double skip_val);

extern "C" int32_t agg_sum_int32(int32_t* agg, const int32_t val);
extern "C" int32_t agg_sum_int32_cpu_shared(int32_t* agg, const int32_t val);

extern "C" void agg_max_int32(int32_t* agg, const int32_t val);
extern "C" void agg_max_int16(int16_t* agg, const int16_t val);
extern "C" void agg_max_int8(int8_t* agg, const int8_t val);

extern "C" void agg_min_int32(int32_t* agg, const int32_t val);
extern "C" void agg_min_int16(int16_t* agg, const int16_t val);
extern "C" void agg_min_int8(int8_t* agg, const int8_t val);

extern "C" void agg_sum_float(int32_t* agg, const float val);

extern "C" void agg_max_float(int32_t* agg, const float val);

extern "C" void agg_min_float(int32_t* agg, const float val);

extern "C" void agg_max_int32_skip_val(int32_t* agg,
                                       const int32_t val,
                                       const int32_t skip_val);
extern "C" void agg_max_int16_skip_val(int16_t* agg,
                                       const int16_t val,
                                       const int16_t skip_val);
extern "C" void agg_max_int8_skip_val(int8_t* agg,
                                      const int8_t val,
                                      const int8_t skip_val);

extern "C" void agg_min_int32_skip_val(int32_t* agg,
                                       const int32_t val,
                                       const int32_t skip_val);
extern "C" void agg_min_int16_skip_val(int16_t* agg,
                                       const int16_t val,
                                       const int16_t skip_val);
extern "C" void agg_min_int8_skip_val(int8_t* agg,
                                      const int8_t val,
                                      const int8_t skip_val);

extern "C" void agg_max_float_skip_val(int32_t* agg,
                                       const float val,
                                       const float skip_val);

extern "C" void agg_min_float_skip_val(int32_t* agg,
                                       const float val,
                                       const float skip_val);

extern "C" void agg_count_distinct_bitmap(int64_t* agg,
                                          const int64_t val,
                                          const int64_t min_val);

#define EMPTY_KEY_64 std::numeric_limits<int64_t>::max()
#define EMPTY_KEY_32 std::numeric_limits<int32_t>::max()
#define EMPTY_KEY_16 std::numeric_limits<int16_t>::max()
#define EMPTY_KEY_8 std::numeric_limits<int8_t>::max()

extern "C" RUNTIME_EXPORT uint32_t key_hash(const int64_t* key,
                                            const uint32_t key_qw_count,
                                            const uint32_t key_byte_width);

extern "C" RUNTIME_EXPORT int64_t* get_group_value(
    int64_t* groups_buffer,
    const uint32_t groups_buffer_entry_count,
    const int64_t* key,
    const uint32_t key_count,
    const uint32_t key_width,
    const uint32_t row_size_quad);

enum RuntimeInterruptFlags { INT_CHECK = 0, INT_ABORT = -1, INT_RESET = -2 };

extern "C" bool check_interrupt();

extern "C" bool check_interrupt_init(unsigned command);

extern "C" RUNTIME_EXPORT int64_t* get_group_value_with_watchdog(
    int64_t* groups_buffer,
    const uint32_t groups_buffer_entry_count,
    const int64_t* key,
    const uint32_t key_count,
    const uint32_t key_width,
    const uint32_t row_size_quad);

extern "C" RUNTIME_EXPORT int64_t* get_group_value_columnar(
    int64_t* groups_buffer,
    const uint32_t groups_buffer_entry_count,
    const int64_t* key,
    const uint32_t key_qw_count);

extern "C" RUNTIME_EXPORT int64_t* get_group_value_columnar_with_watchdog(
    int64_t* groups_buffer,
    const uint32_t groups_buffer_entry_count,
    const int64_t* key,
    const uint32_t key_qw_count);

extern "C" RUNTIME_EXPORT int64_t* get_group_value_fast(int64_t* groups_buffer,
                                                        const int64_t key,
                                                        const int64_t min_key,
                                                        const int64_t bucket,
                                                        const uint32_t row_size_quad);

extern "C" RUNTIME_EXPORT int64_t* get_group_value_fast_with_original_key(
    int64_t* groups_buffer,
    const int64_t key,
    const int64_t orig_key,
    const int64_t min_key,
    const int64_t bucket,
    const uint32_t row_size_quad);

extern "C" RUNTIME_EXPORT uint32_t get_columnar_group_bin_offset(int64_t* key_base_ptr,
                                                                 const int64_t key,
                                                                 const int64_t min_key,
                                                                 const int64_t bucket);

extern "C" int64_t* get_matching_group_value_perfect_hash(int64_t* groups_buffer,
                                                          const uint32_t h,
                                                          const int64_t* key,
                                                          const uint32_t key_qw_count,
                                                          const uint32_t row_size_quad);

extern "C" int64_t* get_matching_group_value_perfect_hash_keyless(
    int64_t* groups_buffer,
    const uint32_t hashed_index,
    const uint32_t row_size_quad);

extern "C" int32_t* get_bucketized_hash_slot(int32_t* buff,
                                             const int64_t key,
                                             const int64_t min_key,
                                             const int64_t bucket_normalization = 1);

extern "C" int32_t* get_hash_slot(int32_t* buff,
                                  const int64_t key,
                                  const int64_t min_key);

extern "C" int32_t* get_hash_slot_sharded(int32_t* buff,
                                          const int64_t key,
                                          const int64_t min_key,
                                          const uint32_t entry_count_per_shard,
                                          const uint32_t num_shards,
                                          const uint32_t device_count);

extern "C" int32_t* get_bucketized_hash_slot_sharded(int32_t* buff,
                                                     const int64_t key,
                                                     const int64_t min_key,
                                                     const uint32_t entry_count_per_shard,
                                                     const uint32_t num_shards,
                                                     const uint32_t device_count,
                                                     const int64_t bucket_normalization);

extern "C" int32_t* get_hash_slot_sharded_opt(int32_t* buff,
                                              const int64_t key,
                                              const int64_t min_key,
                                              const uint32_t entry_count_per_shard,
                                              const uint32_t shard,
                                              const uint32_t num_shards,
                                              const uint32_t device_count);

extern "C" int32_t* get_bucketized_hash_slot_sharded_opt(
    int32_t* buff,
    const int64_t key,
    const int64_t min_key,
    const uint32_t entry_count_per_shard,
    const uint32_t shard,
    const uint32_t num_shards,
    const uint32_t device_count,
    const int64_t bucket_normalization);

extern "C" void linear_probabilistic_count(uint8_t* bitmap,
                                           const uint32_t bitmap_bytes,
                                           const uint8_t* key_bytes,
                                           const uint32_t key_len);

// Regular fixed_width_*_decode are only available from the JIT,
// we need to call them for lazy fetch columns -- create wrappers.

extern "C" int64_t fixed_width_int_decode_noinline(const int8_t* byte_stream,
                                                   const int32_t byte_width,
                                                   const int64_t pos);

extern "C" int64_t fixed_width_unsigned_decode_noinline(const int8_t* byte_stream,
                                                        const int32_t byte_width,
                                                        const int64_t pos);

extern "C" float fixed_width_float_decode_noinline(const int8_t* byte_stream,
                                                   const int64_t pos);

extern "C" double fixed_width_double_decode_noinline(const int8_t* byte_stream,
                                                     const int64_t pos);

extern "C" int64_t fixed_width_small_date_decode_noinline(const int8_t* byte_stream,
                                                          const int32_t byte_width,
                                                          const int32_t null_val,
                                                          const int64_t ret_null_val,
                                                          const int64_t pos);

extern "C" int8_t* extract_str_ptr_noinline(const uint64_t str_and_len);

extern "C" int32_t extract_str_len_noinline(const uint64_t str_and_len);

template <typename T = int64_t>
inline T get_empty_key() {
  static_assert(std::is_same<T, int64_t>::value,
                "Unsupported template parameter other than int64_t for now");
  return EMPTY_KEY_64;
}

template <>
inline int32_t get_empty_key() {
  return EMPTY_KEY_32;
}

#endif  // QUERYENGINE_RUNTIMEFUNCTIONS_H
