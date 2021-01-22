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

#include "StringDictionary/StringDictionary.h"
#include "Utils/Threading.h"

#include <tbb/parallel_for.h>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/sort/spreadsort/string_sort.hpp>
#include <future>
#include <iostream>
#include <string_view>
#include <thread>

// TODO(adb): fixup
#ifdef _MSC_VER
#include <fcntl.h>
#else
#include <sys/fcntl.h>
#endif

#include "Logger/Logger.h"
#include "OSDependent/omnisci_fs.h"
#include "Shared/sqltypes.h"
#include "Shared/thread_count.h"
#include "StringDictionaryClient.h"
#include "Utils/Regexp.h"
#include "Utils/StringLike.h"

#include "LeafHostInfo.h"

bool g_cache_string_hash{true};

namespace {

const int SYSTEM_PAGE_SIZE = omnisci::get_page_size();

int checked_open(const char* path, const bool recover) {
  auto fd = omnisci::open(path, O_RDWR | O_CREAT | (recover ? O_APPEND : O_TRUNC), 0644);
  if (fd > 0) {
    return fd;
  }
  auto err = std::string("Dictionary path ") + std::string(path) +
             std::string(" does not exist.");
  LOG(ERROR) << err;
  throw DictPayloadUnavailable(err);
}

const uint64_t round_up_p2(const uint64_t num) {
  uint64_t in = num;
  in--;
  in |= in >> 1;
  in |= in >> 2;
  in |= in >> 4;
  in |= in >> 8;
  in |= in >> 16;
  in++;
  // TODO MAT deal with case where filesize has been increased but reality is
  // we are constrained to 2^31.
  // In that situation this calculation will wrap to zero
  if (in == 0 || (in > (UINT32_MAX))) {
    in = UINT32_MAX;
  }
  return in;
}

uint32_t rk_hash(const std::string_view& str) {
  uint32_t str_hash = 1;
  // rely on fact that unsigned overflow is defined and wraps
  for (size_t i = 0; i < str.size(); ++i) {
    str_hash = str_hash * 997 + str[i];
  }
  return str_hash;
}
}  // namespace

bool g_enable_stringdict_parallel{false};
constexpr int32_t StringDictionary::INVALID_STR_ID;
constexpr size_t StringDictionary::MAX_STRLEN;
constexpr size_t StringDictionary::MAX_STRCOUNT;

StringDictionary::StringDictionary(const std::string& folder,
                                   const bool isTemp,
                                   const bool recover,
                                   const bool materializeHashes,
                                   size_t initial_capacity)
    : str_count_(0)
    , string_id_hash_table_(initial_capacity, INVALID_STR_ID)
    , rk_hashes_(initial_capacity)
    , isTemp_(isTemp)
    , materialize_hashes_(materializeHashes)
    , payload_fd_(-1)
    , offset_fd_(-1)
    , offset_map_(nullptr)
    , payload_map_(nullptr)
    , offset_file_size_(0)
    , payload_file_size_(0)
    , payload_file_off_(0)
    , strings_cache_(nullptr) {
  if (!isTemp && folder.empty()) {
    return;
  }

  // initial capacity must be a power of two for efficient bucket computation
  CHECK_EQ(size_t(0), (initial_capacity & (initial_capacity - 1)));
  if (!isTemp_) {
    boost::filesystem::path storage_path(folder);
    offsets_path_ = (storage_path / boost::filesystem::path("DictOffsets")).string();
    const auto payload_path =
        (storage_path / boost::filesystem::path("DictPayload")).string();
    payload_fd_ = checked_open(payload_path.c_str(), recover);
    offset_fd_ = checked_open(offsets_path_.c_str(), recover);
    payload_file_size_ = omnisci::file_size(payload_fd_);
    offset_file_size_ = omnisci::file_size(offset_fd_);
  }
  bool storage_is_empty = false;
  if (payload_file_size_ == 0) {
    storage_is_empty = true;
    addPayloadCapacity();
  }
  if (offset_file_size_ == 0) {
    addOffsetCapacity();
  }
  if (!isTemp_) {  // we never mmap or recover temp dictionaries
    payload_map_ =
        reinterpret_cast<char*>(omnisci::checked_mmap(payload_fd_, payload_file_size_));
    offset_map_ = reinterpret_cast<StringIdxEntry*>(
        omnisci::checked_mmap(offset_fd_, offset_file_size_));
    if (recover) {
      const size_t bytes = omnisci::file_size(offset_fd_);
      if (bytes % sizeof(StringIdxEntry) != 0) {
        LOG(WARNING) << "Offsets " << offsets_path_ << " file is truncated";
      }
      const uint64_t str_count =
          storage_is_empty ? 0 : getNumStringsFromStorage(bytes / sizeof(StringIdxEntry));
      collisions_ = 0;
      // at this point we know the size of the StringDict we need to load
      // so lets reallocate the vector to the correct size
      const uint64_t max_entries =
          std::max(round_up_p2(str_count * 2 + 1),
                   round_up_p2(std::max(initial_capacity, static_cast<size_t>(1))));
      std::vector<int32_t> new_str_ids(max_entries, INVALID_STR_ID);
      string_id_hash_table_.swap(new_str_ids);
      if (materialize_hashes_) {
        std::vector<uint32_t> new_rk_hashes(max_entries / 2);
        rk_hashes_.swap(new_rk_hashes);
      }
      // Bail early if we know we don't have strings to add (i.e. a new or empty
      // dictionary)
      if (str_count == 0) {
        return;
      }

      unsigned string_id = 0;
      mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);

      uint32_t thread_inits = 0;
      const auto thread_count = std::thread::hardware_concurrency();
      const uint32_t items_per_thread = std::max<uint32_t>(
          2000, std::min<uint32_t>(200000, (str_count / thread_count) + 1));
      std::vector<utils::future<std::vector<std::pair<uint32_t, unsigned int>>>>
          dictionary_futures;
      for (string_id = 0; string_id < str_count; string_id += items_per_thread) {
        dictionary_futures.emplace_back(
            utils::async([string_id, str_count, items_per_thread, this] {
              std::vector<std::pair<uint32_t, unsigned int>> hashVec;
              for (uint32_t curr_id = string_id;
                   curr_id < string_id + items_per_thread && curr_id < str_count;
                   curr_id++) {
                const auto recovered = getStringFromStorage(curr_id);
                if (recovered.canary) {
                  // hit the canary, recovery finished
                  break;
                } else {
                  std::string temp(recovered.c_str_ptr, recovered.size);
                  hashVec.emplace_back(std::make_pair(rk_hash(temp), temp.size()));
                }
              }
              return hashVec;
            }));
        thread_inits++;
        if (thread_inits % thread_count == 0) {
          processDictionaryFutures(dictionary_futures);
        }
      }
      // gather last few threads
      if (dictionary_futures.size() != 0) {
        processDictionaryFutures(dictionary_futures);
      }
      VLOG(1) << "Opened string dictionary " << folder << " # Strings: " << str_count_
              << " Hash table size: " << string_id_hash_table_.size() << " Fill rate: "
              << static_cast<double>(str_count_) * 100.0 / string_id_hash_table_.size()
              << "% Collisions: " << collisions_;
    }
  }
}

void StringDictionary::processDictionaryFutures(
    std::vector<utils::future<std::vector<std::pair<uint32_t, unsigned int>>>>&
        dictionary_futures) {
  for (auto& dictionary_future : dictionary_futures) {
    dictionary_future.wait();
    auto hashVec = dictionary_future.get();
    for (auto& hash : hashVec) {
      uint32_t bucket = computeUniqueBucketWithHash(hash.first, string_id_hash_table_);
      payload_file_off_ += hash.second;
      string_id_hash_table_[bucket] = static_cast<int32_t>(str_count_);
      if (materialize_hashes_) {
        rk_hashes_[str_count_] = hash.first;
      }
      ++str_count_;
    }
  }
  dictionary_futures.clear();
}

/**
 * Method to retrieve number of strings in storage via a binary search for the first
 * canary
 * @param storage_slots number of storage entries we should search to find the minimum
 * canary
 * @return number of strings in storage
 */
size_t StringDictionary::getNumStringsFromStorage(const size_t storage_slots) const
    noexcept {
  if (storage_slots == 0) {
    return 0;
  }
  // Must use signed integers since final binary search step can wrap to max size_t value
  // if dictionary is empty
  int64_t min_bound = 0;
  int64_t max_bound = storage_slots - 1;
  int64_t guess{0};
  while (min_bound <= max_bound) {
    guess = (max_bound + min_bound) / 2;
    CHECK_GE(guess, 0);
    if (getStringFromStorage(guess).canary) {
      max_bound = guess - 1;
    } else {
      min_bound = guess + 1;
    }
  }
  CHECK_GE(guess + (min_bound > guess ? 1 : 0), 0);
  return guess + (min_bound > guess ? 1 : 0);
}

StringDictionary::StringDictionary(const LeafHostInfo& host, const DictRef dict_ref)
    : strings_cache_(nullptr)
    , client_(new StringDictionaryClient(host, dict_ref, true))
    , client_no_timeout_(new StringDictionaryClient(host, dict_ref, false)) {}

StringDictionary::~StringDictionary() noexcept {
  free(CANARY_BUFFER);
  if (client_) {
    return;
  }
  if (payload_map_) {
    if (!isTemp_) {
      CHECK(offset_map_);
      omnisci::checked_munmap(payload_map_, payload_file_size_);
      omnisci::checked_munmap(offset_map_, offset_file_size_);
      CHECK_GE(payload_fd_, 0);
      omnisci::close(payload_fd_);
      CHECK_GE(offset_fd_, 0);
      omnisci::close(offset_fd_);
    } else {
      CHECK(offset_map_);
      free(payload_map_);
      free(offset_map_);
    }
  }
}

int32_t StringDictionary::getOrAdd(const std::string& str) noexcept {
  if (client_) {
    std::vector<int32_t> string_ids;
    client_->get_or_add_bulk(string_ids, std::vector<std::string>{str});
    CHECK_EQ(size_t(1), string_ids.size());
    return string_ids.front();
  }
  return getOrAddImpl(str);
}

namespace {

template <class T>
void log_encoding_error(std::string_view str) {
  LOG(ERROR) << "Could not encode string: " << str
             << ", the encoded value doesn't fit in " << sizeof(T) * 8
             << " bits. Will store NULL instead.";
}

}  // namespace

template <class String>
void StringDictionary::getOrAddBulkArray(
    const std::vector<std::vector<String>>& string_array_vec,
    std::vector<std::vector<int32_t>>& ids_array_vec) {
  ids_array_vec.resize(string_array_vec.size());
  for (size_t i = 0; i < string_array_vec.size(); i++) {
    auto& strings = string_array_vec[i];
    auto& ids = ids_array_vec[i];
    ids.resize(strings.size());
    getOrAddBulk(strings, &ids[0]);
  }
}

template void StringDictionary::getOrAddBulkArray(
    const std::vector<std::vector<std::string>>& string_array_vec,
    std::vector<std::vector<int32_t>>& ids_array_vec);

/**
 * Method to rk_hash a vector of strings in parallel.
 * @param string_vec input vector of strings to be hashed
 * @param hashes space for the output - should be pre-sized to match string_vec size
 */
template <class String>
void StringDictionary::hashStrings(const std::vector<String>& string_vec,
                                   std::vector<uint32_t>& hashes) const noexcept {
  CHECK_EQ(string_vec.size(), hashes.size());

  utils::parallel_for(utils::blocked_range<size_t>(0, string_vec.size()),
                    [&string_vec, &hashes](const utils::blocked_range<size_t>& r) {
                      for (size_t curr_id = r.begin(); curr_id != r.end(); ++curr_id) {
                        if (string_vec[curr_id].empty()) {
                          continue;
                        }
                        hashes[curr_id] = rk_hash(string_vec[curr_id]);
                      }
                    });
}

template <class T, class String>
void StringDictionary::getOrAddBulk(const std::vector<String>& input_strings,
                                    T* output_string_ids) {
  if (g_enable_stringdict_parallel) {
    getOrAddBulkParallel(input_strings, output_string_ids);
    return;
  }
  // Single-thread path.
  if (client_no_timeout_) {
    getOrAddBulkRemote(input_strings, output_string_ids);
    return;
  }
  size_t out_idx{0};
  mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);

  for (const auto& str : input_strings) {
    if (str.empty()) {
      output_string_ids[out_idx++] = inline_int_null_value<T>();
      continue;
    }
    CHECK(str.size() <= MAX_STRLEN);
    uint32_t bucket;
    const uint32_t hash = rk_hash(str);
    bucket = computeBucket(hash, str, string_id_hash_table_);
    if (string_id_hash_table_[bucket] != INVALID_STR_ID) {
      output_string_ids[out_idx++] = string_id_hash_table_[bucket];
      continue;
    }
    // need to add record to dictionary
    // check there is room
    if (str_count_ == static_cast<size_t>(max_valid_int_value<T>())) {
      log_encoding_error<T>(str);
      output_string_ids[out_idx++] = inline_int_null_value<T>();
      continue;
    }
    if (string_id_hash_table_[bucket] == INVALID_STR_ID) {
      CHECK_LT(str_count_, MAX_STRCOUNT)
          << "Maximum number (" << str_count_
          << ") of Dictionary encoded Strings reached for this column, offset path "
             "for column is  "
          << offsets_path_;
      if (fillRateIsHigh(str_count_)) {
        // resize when more than 50% is full
        increaseCapacity();
        bucket = computeBucket(hash, str, string_id_hash_table_);
      }
      appendToStorage(str);

      string_id_hash_table_[bucket] = static_cast<int32_t>(str_count_);
      if (materialize_hashes_) {
        rk_hashes_[str_count_] = hash;
      }
      ++str_count_;
    }
    output_string_ids[out_idx++] = string_id_hash_table_[bucket];
  }
  invalidateInvertedIndex();
}

template <class T, class String>
void StringDictionary::getOrAddBulkParallel(const std::vector<String>& input_strings,
                                            T* output_string_ids) {
  if (client_no_timeout_) {
    getOrAddBulkRemote(input_strings, output_string_ids);
    return;
  }
  // Run rk_hash on the input strings up front, and in parallel,
  // as the string hashing does not need to be behind the subsequent write_lock
  std::vector<uint32_t> input_strings_rk_hashes(input_strings.size());
  hashStrings(input_strings, input_strings_rk_hashes);

  mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);
  size_t shadow_str_count =
      str_count_;  // Need to shadow str_count_ now with bulk add methods
  const size_t storage_high_water_mark = shadow_str_count;
  std::vector<size_t> string_memory_ids;
  size_t sum_new_string_lengths = 0;
  string_memory_ids.reserve(input_strings.size());
  size_t input_string_idx{0};
  for (const auto& input_string : input_strings) {
    // Currently we make empty strings null
    if (input_string.empty()) {
      output_string_ids[input_string_idx++] = inline_int_null_value<T>();
      continue;
    }
    // TODO: Recover gracefully if an input string is too long
    CHECK(input_string.size() <= MAX_STRLEN);

    if (fillRateIsHigh(shadow_str_count)) {
      // resize when more than 50% is full
      increaseCapacityFromStorageAndMemory(storage_high_water_mark,
                                           input_strings,
                                           string_memory_ids,
                                           input_strings_rk_hashes);
    }
    // Get the rk_hash for this input_string
    const uint32_t input_string_rk_hash = input_strings_rk_hashes[input_string_idx];

    uint32_t hash_bucket = computeBucketFromStorageAndMemory(input_string_rk_hash,
                                                             input_string,
                                                             string_id_hash_table_,
                                                             storage_high_water_mark,
                                                             input_strings,
                                                             string_memory_ids);

    // If the hash bucket is not empty, that is our string id
    // (computeBucketFromStorageAndMemory) already checked to ensure the input string and
    // bucket string are equal)
    if (string_id_hash_table_[hash_bucket] != INVALID_STR_ID) {
      output_string_ids[input_string_idx++] = string_id_hash_table_[hash_bucket];
      continue;
    }
    // Did not find string, so need to add record to dictionary
    // First check there is room
    if (shadow_str_count == static_cast<size_t>(max_valid_int_value<T>())) {
      log_encoding_error<T>(input_string);
      output_string_ids[input_string_idx++] = inline_int_null_value<T>();
      continue;
    }
    CHECK_LT(shadow_str_count, MAX_STRCOUNT)
        << "Maximum number (" << shadow_str_count
        << ") of Dictionary encoded Strings reached for this column, offset path "
           "for column is  "
        << offsets_path_;

    string_memory_ids.push_back(input_string_idx);
    sum_new_string_lengths += input_string.size();
    string_id_hash_table_[hash_bucket] = static_cast<int32_t>(shadow_str_count);
    if (materialize_hashes_) {
      rk_hashes_[shadow_str_count] = input_string_rk_hash;
    }
    output_string_ids[input_string_idx++] = shadow_str_count++;
  }
  appendToStorageBulk(input_strings, string_memory_ids, sum_new_string_lengths);
  str_count_ = shadow_str_count;

  invalidateInvertedIndex();
}
template void StringDictionary::getOrAddBulk(const std::vector<std::string>& string_vec,
                                             uint8_t* encoded_vec);
template void StringDictionary::getOrAddBulk(const std::vector<std::string>& string_vec,
                                             uint16_t* encoded_vec);
template void StringDictionary::getOrAddBulk(const std::vector<std::string>& string_vec,
                                             int32_t* encoded_vec);

template void StringDictionary::getOrAddBulk(
    const std::vector<std::string_view>& string_vec,
    uint8_t* encoded_vec);
template void StringDictionary::getOrAddBulk(
    const std::vector<std::string_view>& string_vec,
    uint16_t* encoded_vec);
template void StringDictionary::getOrAddBulk(
    const std::vector<std::string_view>& string_vec,
    int32_t* encoded_vec);

template <class T, class String>
void StringDictionary::getOrAddBulkRemote(const std::vector<String>& string_vec,
                                          T* encoded_vec) {
  CHECK(client_no_timeout_);
  std::vector<int32_t> string_ids;
  client_no_timeout_->get_or_add_bulk(string_ids, string_vec);
  size_t out_idx{0};
  for (size_t i = 0; i < string_ids.size(); ++i) {
    const auto string_id = string_ids[i];
    const bool invalid = string_id > max_valid_int_value<T>();
    if (invalid || string_id == inline_int_null_value<int32_t>()) {
      if (invalid) {
        log_encoding_error<T>(string_vec[i]);
      }
      encoded_vec[out_idx++] = inline_int_null_value<T>();
      continue;
    }
    encoded_vec[out_idx++] = string_id;
  }
}

template void StringDictionary::getOrAddBulkRemote(
    const std::vector<std::string>& string_vec,
    uint8_t* encoded_vec);
template void StringDictionary::getOrAddBulkRemote(
    const std::vector<std::string>& string_vec,
    uint16_t* encoded_vec);
template void StringDictionary::getOrAddBulkRemote(
    const std::vector<std::string>& string_vec,
    int32_t* encoded_vec);

template void StringDictionary::getOrAddBulkRemote(
    const std::vector<std::string_view>& string_vec,
    uint8_t* encoded_vec);
template void StringDictionary::getOrAddBulkRemote(
    const std::vector<std::string_view>& string_vec,
    uint16_t* encoded_vec);
template void StringDictionary::getOrAddBulkRemote(
    const std::vector<std::string_view>& string_vec,
    int32_t* encoded_vec);

int32_t StringDictionary::getIdOfString(const std::string& str) const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  if (client_) {
    return client_->get(str);
  }
  return getUnlocked(str);
}

int32_t StringDictionary::getUnlocked(const std::string& str) const noexcept {
  const uint32_t hash = rk_hash(str);
  auto str_id = string_id_hash_table_[computeBucket(hash, str, string_id_hash_table_)];
  return str_id;
}

std::string StringDictionary::getString(int32_t string_id) const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  if (client_) {
    std::string ret;
    client_->get_string(ret, string_id);
    return ret;
  }
  return getStringUnlocked(string_id);
}

std::string StringDictionary::getStringUnlocked(int32_t string_id) const noexcept {
  CHECK_LT(string_id, static_cast<int32_t>(str_count_));
  return getStringChecked(string_id);
}

std::pair<char*, size_t> StringDictionary::getStringBytes(int32_t string_id) const
    noexcept {
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  CHECK(!client_);
  CHECK_LE(0, string_id);
  CHECK_LT(string_id, static_cast<int32_t>(str_count_));
  return getStringBytesChecked(string_id);
}

size_t StringDictionary::storageEntryCount() const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  if (client_) {
    return client_->storage_entry_count();
  }
  return str_count_;
}

namespace {

bool is_like(const std::string& str,
             const std::string& pattern,
             const bool icase,
             const bool is_simple,
             const char escape) {
  return icase
             ? (is_simple ? string_ilike_simple(
                                str.c_str(), str.size(), pattern.c_str(), pattern.size())
                          : string_ilike(str.c_str(),
                                         str.size(),
                                         pattern.c_str(),
                                         pattern.size(),
                                         escape))
             : (is_simple ? string_like_simple(
                                str.c_str(), str.size(), pattern.c_str(), pattern.size())
                          : string_like(str.c_str(),
                                        str.size(),
                                        pattern.c_str(),
                                        pattern.size(),
                                        escape));
}

}  // namespace

std::vector<int32_t> StringDictionary::getLike(const std::string& pattern,
                                               const bool icase,
                                               const bool is_simple,
                                               const char escape,
                                               const size_t generation) const {
  mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);
  if (client_) {
    return client_->get_like(pattern, icase, is_simple, escape, generation);
  }
  const auto cache_key = std::make_tuple(pattern, icase, is_simple, escape);
  const auto it = like_cache_.find(cache_key);
  if (it != like_cache_.end()) {
    return it->second;
  }
  std::vector<int32_t> result;
  std::vector<utils::future<void>> workers;

  int worker_count = cpu_threads();
  CHECK_GT(worker_count, 0);
  std::vector<std::vector<int32_t>> worker_results(worker_count);
  CHECK_LE(generation, str_count_);
  for (int worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
    workers.push_back(utils::async([&worker_results,
                                    &pattern,
                                    generation,
                                    icase,
                                    is_simple,
                                    escape,
                                    worker_idx,
                                    worker_count,
                                    this]() {
      for (size_t string_id = worker_idx; string_id < generation;
           string_id += worker_count) {
        const auto str = getStringUnlocked(string_id);
        if (is_like(str, pattern, icase, is_simple, escape)) {
          worker_results[worker_idx].push_back(string_id);
        }
      }
    }));
  }
  for (auto& thread : workers) {
    thread.wait();
  }
  for (const auto& worker_result : worker_results) {
    result.insert(result.end(), worker_result.begin(), worker_result.end());
  }
  // place result into cache for reuse if similar query
  const auto it_ok = like_cache_.insert(std::make_pair(cache_key, result));

  CHECK(it_ok.second);

  return result;
}

std::vector<int32_t> StringDictionary::getEquals(std::string pattern,
                                                 std::string comp_operator,
                                                 size_t generation) {
  std::vector<int32_t> result;
  auto eq_id_itr = equal_cache_.find(pattern);
  int32_t eq_id = MAX_STRLEN + 1;
  int32_t cur_size = str_count_;
  if (eq_id_itr != equal_cache_.end()) {
    auto eq_id = eq_id_itr->second;
    if (comp_operator == "=") {
      result.push_back(eq_id);
    } else {
      for (int32_t idx = 0; idx <= cur_size; idx++) {
        if (idx == eq_id) {
          continue;
        }
        result.push_back(idx);
      }
    }
  } else {
    std::vector<utils::future<void>> workers;
    int worker_count = cpu_threads();
    CHECK_GT(worker_count, 0);
    std::vector<std::vector<int32_t>> worker_results(worker_count);
    CHECK_LE(generation, str_count_);
    for (int worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
      workers.push_back(utils::async(
          [&worker_results, &pattern, generation, worker_idx, worker_count, this]() {
            for (size_t string_id = worker_idx; string_id < generation;
                 string_id += worker_count) {
              const auto str = getStringUnlocked(string_id);
              if (str == pattern) {
                worker_results[worker_idx].push_back(string_id);
              }
            }
          }));
    }
    for (auto& thread : workers) {
      thread.wait();
    }
    for (const auto& worker_result : worker_results) {
      result.insert(result.end(), worker_result.begin(), worker_result.end());
    }
    if (result.size() > 0) {
      const auto it_ok = equal_cache_.insert(std::make_pair(pattern, result[0]));
      CHECK(it_ok.second);
      eq_id = result[0];
    }
    if (comp_operator == "<>") {
      for (int32_t idx = 0; idx <= cur_size; idx++) {
        if (idx == eq_id) {
          continue;
        }
        result.push_back(idx);
      }
    }
  }
  return result;
}

std::vector<int32_t> StringDictionary::getCompare(const std::string& pattern,
                                                  const std::string& comp_operator,
                                                  const size_t generation) {
  mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);
  if (client_) {
    return client_->get_compare(pattern, comp_operator, generation);
  }
  std::vector<int32_t> ret;
  if (str_count_ == 0) {
    return ret;
  }
  if (sorted_cache.size() < str_count_) {
    if (comp_operator == "=" || comp_operator == "<>") {
      return getEquals(pattern, comp_operator, generation);
    }

    buildSortedCache();
  }
  auto cache_index = compare_cache_.get(pattern);

  if (!cache_index) {
    cache_index = std::make_shared<StringDictionary::compare_cache_value_t>();
    const auto cache_itr = std::lower_bound(
        sorted_cache.begin(),
        sorted_cache.end(),
        pattern,
        [this](decltype(sorted_cache)::value_type const& a, decltype(pattern)& b) {
          auto a_str = this->getStringFromStorage(a);
          return string_lt(a_str.c_str_ptr, a_str.size, b.c_str(), b.size());
        });

    if (cache_itr == sorted_cache.end()) {
      cache_index->index = sorted_cache.size() - 1;
      cache_index->diff = 1;
    } else {
      const auto cache_str = getStringFromStorage(*cache_itr);
      if (!string_eq(
              cache_str.c_str_ptr, cache_str.size, pattern.c_str(), pattern.size())) {
        cache_index->index = cache_itr - sorted_cache.begin() - 1;
        cache_index->diff = 1;
      } else {
        cache_index->index = cache_itr - sorted_cache.begin();
        cache_index->diff = 0;
      }
    }

    compare_cache_.put(pattern, cache_index);
  }

  // since we have a cache in form of vector of ints which is sorted according to
  // corresponding strings in the dictionary all we need is the index of the element
  // which equal to the pattern that we are trying to match or the index of “biggest”
  // element smaller than the pattern, to perform all the comparison operators over
  // string. The search function guarantees we have such index so now it is just the
  // matter to include all the elements in the result vector.

  // For < operator if the index that we have points to the element which is equal to
  // the pattern that we are searching for we simply get all the elements less than the
  // index. If the element pointed by the index is not equal to the pattern we are
  // comparing with we also need to include that index in result vector, except when the
  // index points to 0 and the pattern is lesser than the smallest value in the string
  // dictionary.

  if (comp_operator == "<") {
    size_t idx = cache_index->index;
    if (cache_index->diff) {
      idx = cache_index->index + 1;
      if (cache_index->index == 0 && cache_index->diff > 0) {
        idx = cache_index->index;
      }
    }
    for (size_t i = 0; i < idx; i++) {
      ret.push_back(sorted_cache[i]);
    }

    // For <= operator if the index that we have points to the element which is equal to
    // the pattern that we are searching for we want to include the element pointed by
    // the index in the result set. If the element pointed by the index is not equal to
    // the pattern we are comparing with we just want to include all the ids with index
    // less than the index that is cached, except when pattern that we are searching for
    // is smaller than the smallest string in the dictionary.

  } else if (comp_operator == "<=") {
    size_t idx = cache_index->index + 1;
    if (cache_index == 0 && cache_index->diff > 0) {
      idx = cache_index->index;
    }
    for (size_t i = 0; i < idx; i++) {
      ret.push_back(sorted_cache[i]);
    }

    // For > operator we want to get all the elements with index greater than the index
    // that we have except, when the pattern we are searching for is lesser than the
    // smallest string in the dictionary we also want to include the id of the index
    // that we have.

  } else if (comp_operator == ">") {
    size_t idx = cache_index->index + 1;
    if (cache_index->index == 0 && cache_index->diff > 0) {
      idx = cache_index->index;
    }
    for (size_t i = idx; i < sorted_cache.size(); i++) {
      ret.push_back(sorted_cache[i]);
    }

    // For >= operator when the indexed element that we have points to element which is
    // equal to the pattern we are searching for we want to include that in the result
    // vector. If the index that we have does not point to the string which is equal to
    // the pattern we are searching we don’t want to include that id into the result
    // vector except when the index is 0.

  } else if (comp_operator == ">=") {
    size_t idx = cache_index->index;
    if (cache_index->diff) {
      idx = cache_index->index + 1;
      if (cache_index->index == 0 && cache_index->diff > 0) {
        idx = cache_index->index;
      }
    }
    for (size_t i = idx; i < sorted_cache.size(); i++) {
      ret.push_back(sorted_cache[i]);
    }
  } else if (comp_operator == "=") {
    if (!cache_index->diff) {
      ret.push_back(sorted_cache[cache_index->index]);
    }

    // For <> operator it is simple matter of not including id of string which is equal
    // to pattern we are searching for.
  } else if (comp_operator == "<>") {
    if (!cache_index->diff) {
      size_t idx = cache_index->index;
      for (size_t i = 0; i < idx; i++) {
        ret.push_back(sorted_cache[i]);
      }
      ++idx;
      for (size_t i = idx; i < sorted_cache.size(); i++) {
        ret.push_back(sorted_cache[i]);
      }
    } else {
      for (size_t i = 0; i < sorted_cache.size(); i++) {
        ret.insert(ret.begin(), sorted_cache.begin(), sorted_cache.end());
      }
    }

  } else {
    std::runtime_error("Unsupported string comparison operator");
  }
  return ret;
}

namespace {

bool is_regexp_like(const std::string& str,
                    const std::string& pattern,
                    const char escape) {
  return regexp_like(str.c_str(), str.size(), pattern.c_str(), pattern.size(), escape);
}

}  // namespace

std::vector<int32_t> StringDictionary::getRegexpLike(const std::string& pattern,
                                                     const char escape,
                                                     const size_t generation) const {
  mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);
  if (client_) {
    return client_->get_regexp_like(pattern, escape, generation);
  }
  const auto cache_key = std::make_pair(pattern, escape);
  const auto it = regex_cache_.find(cache_key);
  if (it != regex_cache_.end()) {
    return it->second;
  }
  std::vector<int32_t> result;
  std::vector<utils::future<void>> workers;
  int worker_count = cpu_threads();
  CHECK_GT(worker_count, 0);
  std::vector<std::vector<int32_t>> worker_results(worker_count);
  CHECK_LE(generation, str_count_);
  for (int worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
    workers.push_back(utils::async([&worker_results,
                                    &pattern,
                                    generation,
                                    escape,
                                    worker_idx,
                                    worker_count,
                                    this]() {
      for (size_t string_id = worker_idx; string_id < generation;
           string_id += worker_count) {
        const auto str = getStringUnlocked(string_id);
        if (is_regexp_like(str, pattern, escape)) {
          worker_results[worker_idx].push_back(string_id);
        }
      }
    }));
  }
  for (auto& thread : workers) {
    thread.wait();
  }
  for (const auto& worker_result : worker_results) {
    result.insert(result.end(), worker_result.begin(), worker_result.end());
  }
  const auto it_ok = regex_cache_.insert(std::make_pair(cache_key, result));
  CHECK(it_ok.second);

  return result;
}

std::shared_ptr<const std::vector<std::string>> StringDictionary::copyStrings() const {
  mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);
  if (client_) {
    // TODO(miyu): support remote string dictionary
    throw std::runtime_error(
        "copying dictionaries from remote server is not supported yet.");
  }

  if (strings_cache_) {
    return strings_cache_;
  }

  strings_cache_ = std::make_shared<std::vector<std::string>>();
  strings_cache_->reserve(str_count_);
  const bool multithreaded = str_count_ > 10000;
  const auto worker_count =
      multithreaded ? static_cast<size_t>(cpu_threads()) : size_t(1);
  CHECK_GT(worker_count, 0UL);
  std::vector<std::vector<std::string>> worker_results(worker_count);
  auto copy = [this](std::vector<std::string>& str_list,
                     const size_t start_id,
                     const size_t end_id) {
    CHECK_LE(start_id, end_id);
    str_list.reserve(end_id - start_id);
    for (size_t string_id = start_id; string_id < end_id; ++string_id) {
      str_list.push_back(getStringUnlocked(string_id));
    }
  };
  if (multithreaded) {
    std::vector<utils::future<void>> workers;
    const auto stride = (str_count_ + (worker_count - 1)) / worker_count;
    for (size_t worker_idx = 0, start = 0, end = std::min(start + stride, str_count_);
         worker_idx < worker_count && start < str_count_;
         ++worker_idx, start += stride, end = std::min(start + stride, str_count_)) {
      workers.push_back(
          utils::async(copy, std::ref(worker_results[worker_idx]), start, end));
    }
    for (auto& worker : workers) {
      worker.get();
    }
  } else {
    CHECK_EQ(worker_results.size(), size_t(1));
    copy(worker_results[0], 0, str_count_);
  }

  for (const auto& worker_result : worker_results) {
    strings_cache_->insert(
        strings_cache_->end(), worker_result.begin(), worker_result.end());
  }
  return strings_cache_;
}

bool StringDictionary::fillRateIsHigh(const size_t num_strings) const noexcept {
  return string_id_hash_table_.size() <= num_strings * 2;
}

void StringDictionary::increaseCapacity() noexcept {
  std::vector<int32_t> new_str_ids(string_id_hash_table_.size() * 2, INVALID_STR_ID);

  if (materialize_hashes_) {
    for (size_t i = 0; i < string_id_hash_table_.size(); ++i) {
      if (string_id_hash_table_[i] != INVALID_STR_ID) {
        const uint32_t hash = rk_hashes_[string_id_hash_table_[i]];
        uint32_t bucket = computeUniqueBucketWithHash(hash, new_str_ids);
        new_str_ids[bucket] = string_id_hash_table_[i];
      }
    }
    rk_hashes_.resize(rk_hashes_.size() * 2);
  } else {
    for (size_t i = 0; i < str_count_; ++i) {
      const auto str = getStringChecked(i);
      const uint32_t hash = rk_hash(str);
      uint32_t bucket = computeUniqueBucketWithHash(hash, new_str_ids);
      new_str_ids[bucket] = i;
    }
  }
  string_id_hash_table_.swap(new_str_ids);
}

template <class String>
void StringDictionary::increaseCapacityFromStorageAndMemory(
    const size_t storage_high_water_mark,
    const std::vector<String>& input_strings,
    const std::vector<size_t>& string_memory_ids,
    const std::vector<uint32_t>& input_strings_rk_hashes) noexcept {
  std::vector<int32_t> new_str_ids(string_id_hash_table_.size() * 2, INVALID_STR_ID);
  if (materialize_hashes_) {
    for (size_t i = 0; i < string_id_hash_table_.size(); ++i) {
      if (string_id_hash_table_[i] != INVALID_STR_ID) {
        const uint32_t hash = rk_hashes_[string_id_hash_table_[i]];
        uint32_t bucket = computeUniqueBucketWithHash(hash, new_str_ids);
        new_str_ids[bucket] = string_id_hash_table_[i];
      }
    }
    rk_hashes_.resize(rk_hashes_.size() * 2);
  } else {
    for (size_t storage_idx = 0; storage_idx != storage_high_water_mark; ++storage_idx) {
      const auto storage_string = getStringChecked(storage_idx);
      const uint32_t hash = rk_hash(storage_string);
      uint32_t bucket = computeUniqueBucketWithHash(hash, new_str_ids);
      new_str_ids[bucket] = storage_idx;
    }
    for (size_t memory_idx = 0; memory_idx != string_memory_ids.size(); ++memory_idx) {
      size_t string_memory_id = string_memory_ids[memory_idx];
      uint32_t bucket = computeUniqueBucketWithHash(
          input_strings_rk_hashes[string_memory_id], new_str_ids);
      new_str_ids[bucket] = storage_high_water_mark + memory_idx;
    }
  }
  string_id_hash_table_.swap(new_str_ids);
}

int32_t StringDictionary::getOrAddImpl(const std::string& str) noexcept {
  // @TODO(wei) treat empty string as NULL for now
  if (str.size() == 0) {
    return inline_int_null_value<int32_t>();
  }
  CHECK(str.size() <= MAX_STRLEN);
  uint32_t bucket;
  const uint32_t hash = rk_hash(str);
  {
    mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
    bucket = computeBucket(hash, str, string_id_hash_table_);
    if (string_id_hash_table_[bucket] != INVALID_STR_ID) {
      return string_id_hash_table_[bucket];
    }
  }
  mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);
  // need to recalculate the bucket in case it changed before
  // we got the lock
  bucket = computeBucket(hash, str, string_id_hash_table_);
  if (string_id_hash_table_[bucket] == INVALID_STR_ID) {
    CHECK_LT(str_count_, MAX_STRCOUNT)
        << "Maximum number (" << str_count_
        << ") of Dictionary encoded Strings reached for this column, offset path "
           "for column is  "
        << offsets_path_;
    if (fillRateIsHigh(str_count_)) {
      // resize when more than 50% is full
      increaseCapacity();
      bucket = computeBucket(hash, str, string_id_hash_table_);
    }
    appendToStorage(str);
    string_id_hash_table_[bucket] = static_cast<int32_t>(str_count_);
    if (materialize_hashes_) {
      rk_hashes_[str_count_] = hash;
    }
    ++str_count_;
    invalidateInvertedIndex();
  }
  return string_id_hash_table_[bucket];
}

std::string StringDictionary::getStringChecked(const int string_id) const noexcept {
  const auto str_canary = getStringFromStorage(string_id);
  CHECK(!str_canary.canary);
  return std::string(str_canary.c_str_ptr, str_canary.size);
}

std::pair<char*, size_t> StringDictionary::getStringBytesChecked(
    const int string_id) const noexcept {
  const auto str_canary = getStringFromStorage(string_id);
  CHECK(!str_canary.canary);
  return std::make_pair(str_canary.c_str_ptr, str_canary.size);
}

template <class String>
uint32_t StringDictionary::computeBucket(const uint32_t hash,
                                         const String& str,
                                         const std::vector<int32_t>& data) const
    noexcept {
  auto bucket = hash & (data.size() - 1);
  while (true) {
    const int32_t candidate_string_id = data[bucket];
    if (candidate_string_id ==
        INVALID_STR_ID) {  // In this case it means the slot is available for use
      break;
    }
    if (!materialize_hashes_ ||
        (materialize_hashes_ && hash == rk_hashes_[candidate_string_id])) {
      const auto old_str = getStringFromStorageFast(candidate_string_id);
      if (str.size() == old_str.size() &&
          !memcmp(str.data(), old_str.data(), str.size())) {
        // found the string
        break;
      }
    }
    // wrap around
    if (++bucket == data.size()) {
      bucket = 0;
    }
  }
  return bucket;
}

template <class String>
uint32_t StringDictionary::computeBucketFromStorageAndMemory(
    const uint32_t input_string_rk_hash,
    const String& input_string,
    const std::vector<int32_t>& string_id_hash_table,
    const size_t storage_high_water_mark,
    const std::vector<String>& input_strings,
    const std::vector<size_t>& string_memory_ids) const noexcept {
  auto bucket = input_string_rk_hash & (string_id_hash_table.size() - 1);
  while (true) {
    const int32_t candidate_string_id = string_id_hash_table[bucket];
    if (candidate_string_id ==
        INVALID_STR_ID) {  // In this case it means the slot is available for use
      break;
    }
    if (!materialize_hashes_ ||
        (input_string_rk_hash == rk_hashes_[candidate_string_id])) {
      if (candidate_string_id > 0 &&
          static_cast<size_t>(candidate_string_id) >= storage_high_water_mark) {
        // The candidate string is not in storage yet but in our string_memory_ids temp
        // buffer
        size_t memory_offset =
            static_cast<size_t>(candidate_string_id - storage_high_water_mark);
        const String candidate_string = input_strings[string_memory_ids[memory_offset]];
        if (input_string.size() == candidate_string.size() &&
            !memcmp(input_string.data(), candidate_string.data(), input_string.size())) {
          // found the string in the temp memory buffer
          break;
        }
      } else {
        // The candidate string is in storage, need to fetch it for comparison
        const auto candidate_storage_string =
            getStringFromStorageFast(candidate_string_id);
        if (input_string.size() == candidate_storage_string.size() &&
            !memcmp(input_string.data(),
                    candidate_storage_string.data(),
                    input_string.size())) {
          //! memcmp(input_string.data(), candidate_storage_string.c_str_ptr,
          //! input_string.size())) {
          // found the string in storage
          break;
        }
      }
    }
    if (++bucket == string_id_hash_table.size()) {
      bucket = 0;
    }
  }
  return bucket;
}

uint32_t StringDictionary::computeUniqueBucketWithHash(
    const uint32_t hash,
    const std::vector<int32_t>& data) noexcept {
  auto bucket = hash & (data.size() - 1);
  while (true) {
    if (data[bucket] ==
        INVALID_STR_ID) {  // In this case it means the slot is available for use
      break;
    }
    collisions_++;
    // wrap around
    if (++bucket == data.size()) {
      bucket = 0;
    }
  }
  return bucket;
}

void StringDictionary::checkAndConditionallyIncreasePayloadCapacity(
    const size_t write_length) {
  if (payload_file_off_ + write_length > payload_file_size_) {
    const size_t min_capacity_needed =
        write_length - (payload_file_size_ - payload_file_off_);
    if (!isTemp_) {
      CHECK_GE(payload_fd_, 0);
      omnisci::checked_munmap(payload_map_, payload_file_size_);
      addPayloadCapacity(min_capacity_needed);
      CHECK(payload_file_off_ + write_length <= payload_file_size_);
      payload_map_ =
          reinterpret_cast<char*>(omnisci::checked_mmap(payload_fd_, payload_file_size_));
    } else {
      addPayloadCapacity(min_capacity_needed);
      CHECK(payload_file_off_ + write_length <= payload_file_size_);
    }
  }
}

void StringDictionary::checkAndConditionallyIncreaseOffsetCapacity(
    const size_t write_length) {
  size_t offset_file_off = str_count_ * sizeof(StringIdxEntry);
  if (offset_file_off + write_length >= offset_file_size_) {
    const size_t min_capacity_needed =
        write_length - (offset_file_size_ - offset_file_off);
    if (!isTemp_) {
      CHECK_GE(offset_fd_, 0);
      omnisci::checked_munmap(offset_map_, offset_file_size_);
      addOffsetCapacity(min_capacity_needed);
      CHECK(offset_file_off + write_length <= offset_file_size_);
      offset_map_ = reinterpret_cast<StringIdxEntry*>(
          omnisci::checked_mmap(offset_fd_, offset_file_size_));
    } else {
      addOffsetCapacity(min_capacity_needed);
      CHECK(offset_file_off + write_length <= offset_file_size_);
    }
  }
}

template <class String>
void StringDictionary::appendToStorage(String str) noexcept {
  // write the payload
  checkAndConditionallyIncreasePayloadCapacity(str.size());
  memcpy(payload_map_ + payload_file_off_, str.data(), str.size());

  // write the offset and length
  StringIdxEntry str_meta{static_cast<uint64_t>(payload_file_off_), str.size()};
  payload_file_off_ += str.size();  // Need to increment after we've defined str_meta

  checkAndConditionallyIncreaseOffsetCapacity(sizeof(str_meta));
  memcpy(offset_map_ + str_count_, &str_meta, sizeof(str_meta));
}

template <class String>
void StringDictionary::appendToStorageBulk(
    const std::vector<String>& input_strings,
    const std::vector<size_t>& string_memory_ids,
    const size_t sum_new_strings_lengths) noexcept {
  const size_t num_strings = string_memory_ids.size();

  checkAndConditionallyIncreasePayloadCapacity(sum_new_strings_lengths);
  checkAndConditionallyIncreaseOffsetCapacity(sizeof(StringIdxEntry) * num_strings);

  for (size_t i = 0; i < num_strings; ++i) {
    const size_t string_idx = string_memory_ids[i];
    const String str = input_strings[string_idx];
    const size_t str_size(str.size());
    memcpy(payload_map_ + payload_file_off_, str.data(), str_size);
    StringIdxEntry str_meta{static_cast<uint64_t>(payload_file_off_), str_size};
    payload_file_off_ += str_size;  // Need to increment after we've defined str_meta
    memcpy(offset_map_ + str_count_ + i, &str_meta, sizeof(str_meta));
  }
}

std::string_view StringDictionary::getStringFromStorageFast(const int string_id) const
    noexcept {
  const StringIdxEntry* str_meta = offset_map_ + string_id;
  return {payload_map_ + str_meta->off, str_meta->size};
}

StringDictionary::PayloadString StringDictionary::getStringFromStorage(
    const int string_id) const noexcept {
  if (!isTemp_) {
    CHECK_GE(payload_fd_, 0);
    CHECK_GE(offset_fd_, 0);
  }
  CHECK_GE(string_id, 0);
  const StringIdxEntry* str_meta = offset_map_ + string_id;
  if (str_meta->size == 0xffff) {
    // hit the canary
    return {nullptr, 0, true};
  }
  return {payload_map_ + str_meta->off, str_meta->size, false};
}

void StringDictionary::addPayloadCapacity(const size_t min_capacity_requested) noexcept {
  if (!isTemp_) {
    payload_file_size_ += addStorageCapacity(payload_fd_, min_capacity_requested);
  } else {
    payload_map_ = static_cast<char*>(
        addMemoryCapacity(payload_map_, payload_file_size_, min_capacity_requested));
  }
}

void StringDictionary::addOffsetCapacity(const size_t min_capacity_requested) noexcept {
  if (!isTemp_) {
    offset_file_size_ += addStorageCapacity(offset_fd_, min_capacity_requested);
  } else {
    offset_map_ = static_cast<StringIdxEntry*>(
        addMemoryCapacity(offset_map_, offset_file_size_, min_capacity_requested));
  }
}

size_t StringDictionary::addStorageCapacity(
    int fd,
    const size_t min_capacity_requested) noexcept {
  const size_t canary_buff_size_to_add =
      std::max(static_cast<size_t>(1024 * SYSTEM_PAGE_SIZE),
               (min_capacity_requested / SYSTEM_PAGE_SIZE + 1) * SYSTEM_PAGE_SIZE);

  if (canary_buffer_size != canary_buff_size_to_add) {
    CANARY_BUFFER = static_cast<char*>(realloc(CANARY_BUFFER, canary_buff_size_to_add));
    canary_buffer_size = canary_buff_size_to_add;
  }
  CHECK(CANARY_BUFFER);
  memset(CANARY_BUFFER, 0xff, canary_buff_size_to_add);

  CHECK_NE(lseek(fd, 0, SEEK_END), -1);
  const auto write_return = write(fd, CANARY_BUFFER, canary_buff_size_to_add);
  CHECK(write_return > 0 &&
        (static_cast<size_t>(write_return) == canary_buff_size_to_add));
  return canary_buff_size_to_add;
}

void* StringDictionary::addMemoryCapacity(void* addr,
                                          size_t& mem_size,
                                          const size_t min_capacity_requested) noexcept {
  const size_t canary_buff_size_to_add =
      std::max(static_cast<size_t>(1024 * SYSTEM_PAGE_SIZE),
               (min_capacity_requested / SYSTEM_PAGE_SIZE + 1) * SYSTEM_PAGE_SIZE);
  if (canary_buffer_size != canary_buff_size_to_add) {
    CANARY_BUFFER =
        reinterpret_cast<char*>(realloc(CANARY_BUFFER, canary_buff_size_to_add));
    canary_buffer_size = canary_buff_size_to_add;
  }
  CHECK(CANARY_BUFFER);
  memset(CANARY_BUFFER, 0xff, canary_buff_size_to_add);
  void* new_addr = realloc(addr, mem_size + canary_buff_size_to_add);
  CHECK(new_addr);
  void* write_addr = reinterpret_cast<void*>(static_cast<char*>(new_addr) + mem_size);
  CHECK(memcpy(write_addr, CANARY_BUFFER, canary_buff_size_to_add));
  mem_size += canary_buff_size_to_add;
  return new_addr;
}

void StringDictionary::invalidateInvertedIndex() noexcept {
  if (!like_cache_.empty()) {
    decltype(like_cache_)().swap(like_cache_);
  }
  if (!regex_cache_.empty()) {
    decltype(regex_cache_)().swap(regex_cache_);
  }
  if (!equal_cache_.empty()) {
    decltype(equal_cache_)().swap(equal_cache_);
  }
  compare_cache_.invalidateInvertedIndex();
}

bool StringDictionary::checkpoint() noexcept {
  if (client_) {
    try {
      return client_->checkpoint();
    } catch (...) {
      return false;
    }
  }
  CHECK(!isTemp_);
  bool ret = true;
  ret = ret &&
        (omnisci::msync((void*)offset_map_, offset_file_size_, /*async=*/false) == 0);
  ret = ret &&
        (omnisci::msync((void*)payload_map_, payload_file_size_, /*async=*/false) == 0);
  ret = ret && (omnisci::fsync(offset_fd_) == 0);
  ret = ret && (omnisci::fsync(payload_fd_) == 0);
  return ret;
}

void StringDictionary::buildSortedCache() {
  // This method is not thread-safe.
  const auto cur_cache_size = sorted_cache.size();
  std::vector<int32_t> temp_sorted_cache;
  for (size_t i = cur_cache_size; i < str_count_; i++) {
    temp_sorted_cache.push_back(i);
  }
  sortCache(temp_sorted_cache);
  mergeSortedCache(temp_sorted_cache);
}

void StringDictionary::sortCache(std::vector<int32_t>& cache) {
  // This method is not thread-safe.

  // this boost sort is creating some problems when we use UTF-8 encoded strings.
  // TODO (vraj): investigate What is wrong with boost sort and try to mitigate it.

  std::sort(cache.begin(), cache.end(), [this](int32_t a, int32_t b) {
    auto a_str = this->getStringFromStorage(a);
    auto b_str = this->getStringFromStorage(b);
    return string_lt(a_str.c_str_ptr, a_str.size, b_str.c_str_ptr, b_str.size);
  });
}

void StringDictionary::mergeSortedCache(std::vector<int32_t>& temp_sorted_cache) {
  // this method is not thread safe
  std::vector<int32_t> updated_cache(temp_sorted_cache.size() + sorted_cache.size());
  size_t t_idx = 0, s_idx = 0, idx = 0;
  for (; t_idx < temp_sorted_cache.size() && s_idx < sorted_cache.size(); idx++) {
    auto t_string = getStringFromStorage(temp_sorted_cache[t_idx]);
    auto s_string = getStringFromStorage(sorted_cache[s_idx]);
    const auto insert_from_temp_cache =
        string_lt(t_string.c_str_ptr, t_string.size, s_string.c_str_ptr, s_string.size);
    if (insert_from_temp_cache) {
      updated_cache[idx] = temp_sorted_cache[t_idx++];
    } else {
      updated_cache[idx] = sorted_cache[s_idx++];
    }
  }
  while (t_idx < temp_sorted_cache.size()) {
    updated_cache[idx++] = temp_sorted_cache[t_idx++];
  }
  while (s_idx < sorted_cache.size()) {
    updated_cache[idx++] = sorted_cache[s_idx++];
  }
  sorted_cache.swap(updated_cache);
}

void StringDictionary::populate_string_ids(
    std::vector<int32_t>& dest_ids,
    StringDictionary* dest_dict,
    const std::vector<int32_t>& source_ids,
    const StringDictionary* source_dict,
    const std::map<int32_t, std::string> transient_mapping) {
  std::vector<std::string> strings;

  for (const int32_t source_id : source_ids) {
    if (source_id == std::numeric_limits<int32_t>::min()) {
      strings.emplace_back("");
    } else if (source_id < 0) {
      if (auto string_itr = transient_mapping.find(source_id);
          string_itr != transient_mapping.end()) {
        strings.emplace_back(string_itr->second);
      } else {
        throw std::runtime_error("Unexpected negative source ID");
      }
    } else {
      strings.push_back(source_dict->getString(source_id));
    }
  }

  dest_ids.resize(strings.size());
  dest_dict->getOrAddBulk(strings, &dest_ids[0]);
}

void StringDictionary::populate_string_array_ids(
    std::vector<std::vector<int32_t>>& dest_array_ids,
    StringDictionary* dest_dict,
    const std::vector<std::vector<int32_t>>& source_array_ids,
    const StringDictionary* source_dict) {
  dest_array_ids.resize(source_array_ids.size());

  std::atomic<size_t> row_idx{0};
  auto processor = [&row_idx, &dest_array_ids, dest_dict, &source_array_ids, source_dict](
                       int thread_id) {
    for (;;) {
      auto row = row_idx.fetch_add(1);

      if (row >= dest_array_ids.size()) {
        return;
      }
      const auto& source_ids = source_array_ids[row];
      auto& dest_ids = dest_array_ids[row];
      populate_string_ids(dest_ids, dest_dict, source_ids, source_dict);
    }
  };

  const int num_worker_threads = std::thread::hardware_concurrency();

  if (source_array_ids.size() / num_worker_threads > 10) {
    std::vector<utils::future<void>> worker_threads;
    for (int i = 0; i < num_worker_threads; ++i) {
      worker_threads.push_back(utils::async(processor, i));
    }

    for (auto& child : worker_threads) {
      child.wait();
    }
    for (auto& child : worker_threads) {
      child.get();
    }
  } else {
    processor(0);
  }
}

void translate_string_ids(std::vector<int32_t>& dest_ids,
                          const LeafHostInfo& dict_server_host,
                          const DictRef dest_dict_ref,
                          const std::vector<int32_t>& source_ids,
                          const DictRef source_dict_ref,
                          const int32_t dest_generation) {
  DictRef temp_dict_ref(-1, -1);
  StringDictionaryClient string_client(dict_server_host, temp_dict_ref, false);
  string_client.translate_string_ids(
      dest_ids, dest_dict_ref, source_ids, source_dict_ref, dest_generation);
}
