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

#ifndef STRINGDICTIONARY_STRINGDICTIONARY_H
#define STRINGDICTIONARY_STRINGDICTIONARY_H

#include "../Shared/mapd_shared_mutex.h"
#include "../Utils/Threading.h"
#include "DictRef.h"
#include "DictionaryCache.hpp"
#include "LeafHostInfo.h"

#include <future>
#include <map>
#include <string>
#include <tuple>
#include <vector>

extern bool g_enable_stringdict_parallel;

class StringDictionaryClient;

class DictPayloadUnavailable : public std::runtime_error {
 public:
  DictPayloadUnavailable() : std::runtime_error("DictPayloadUnavailable") {}

  DictPayloadUnavailable(const std::string& err) : std::runtime_error(err) {}
};

class StringDictionary {
 public:
  StringDictionary(const std::string& folder,
                   const bool isTemp,
                   const bool recover,
                   const bool materializeHashes = false,
                   size_t initial_capacity = 256);
  StringDictionary(const LeafHostInfo& host, const DictRef dict_ref);
  ~StringDictionary() noexcept;

  int32_t getOrAdd(const std::string& str) noexcept;
  template <class T, class String>
  void getOrAddBulk(const std::vector<String>& string_vec, T* encoded_vec);
  template <class T, class String>
  void getOrAddBulkParallel(const std::vector<String>& string_vec, T* encoded_vec);
  template <class String>
  void getOrAddBulkArray(const std::vector<std::vector<String>>& string_array_vec,
                         std::vector<std::vector<int32_t>>& ids_array_vec);
  int32_t getIdOfString(const std::string& str) const;
  std::string getString(int32_t string_id) const;
  std::pair<char*, size_t> getStringBytes(int32_t string_id) const noexcept;
  size_t storageEntryCount() const;

  std::vector<int32_t> getLike(const std::string& pattern,
                               const bool icase,
                               const bool is_simple,
                               const char escape,
                               const size_t generation) const;

  std::vector<int32_t> getCompare(const std::string& pattern,
                                  const std::string& comp_operator,
                                  const size_t generation);

  std::vector<int32_t> getRegexpLike(const std::string& pattern,
                                     const char escape,
                                     const size_t generation) const;

  std::shared_ptr<const std::vector<std::string>> copyStrings() const;

  bool checkpoint() noexcept;

  /**
   * @brief Populates provided \p dest_ids vector with string ids corresponding to given
   * source strings
   *
   * Given a vector of source string ids and corresponding source dictionary, this method
   * populates a vector of destination string ids by either returning the string id of
   * matching strings in the destination dictionary or creating new entries in the
   * dictionary. Source string ids can also be transient if they were created by a
   * function (e.g LOWER/UPPER functions). A map of transient string ids to string values
   * is provided in order to handle this use case.
   *
   * @param dest_ids - vector of destination string ids to be populated
   * @param dest_dict - destination dictionary
   * @param source_ids - vector of source string ids for which destination ids are needed
   * @param source_dict - source dictionary
   * @param transient_mapping - map of transient source string ids to string values
   */
  static void populate_string_ids(
      std::vector<int32_t>& dest_ids,
      StringDictionary* dest_dict,
      const std::vector<int32_t>& source_ids,
      const StringDictionary* source_dict,
      const std::map<int32_t, std::string> transient_mapping = {});

  static void populate_string_array_ids(
      std::vector<std::vector<int32_t>>& dest_array_ids,
      StringDictionary* dest_dict,
      const std::vector<std::vector<int32_t>>& source_array_ids,
      const StringDictionary* source_dict);

  static constexpr int32_t INVALID_STR_ID = -1;
  static constexpr size_t MAX_STRLEN = (1 << 15) - 1;
  static constexpr size_t MAX_STRCOUNT = (1U << 31) - 1;

 private:
  struct StringIdxEntry {
    uint64_t off : 48;
    uint64_t size : 16;
  };

  // In the compare_cache_value_t index represents the index of the sorted cache.
  // The diff component represents whether the index the cache is pointing to is equal to
  // the pattern it is cached for. We want to use diff so we don't have compare string
  // again when we are retrieving it from the cache.
  struct compare_cache_value_t {
    int32_t index;
    int32_t diff;
  };

  struct PayloadString {
    char* c_str_ptr;
    size_t size;
    bool canary;
  };

  void processDictionaryFutures(
      std::vector<utils::future<std::vector<std::pair<uint32_t, unsigned int>>>>&
          dictionary_futures);
  size_t getNumStringsFromStorage(const size_t storage_slots) const noexcept;
  bool fillRateIsHigh(const size_t num_strings) const noexcept;
  void increaseCapacity() noexcept;
  template <class String>
  void increaseCapacityFromStorageAndMemory(
      const size_t storage_high_water_mark,
      const std::vector<String>& input_strings,
      const std::vector<size_t>& string_memory_ids,
      const std::vector<uint32_t>& input_strings_rk_hashes) noexcept;
  int32_t getOrAddImpl(const std::string& str) noexcept;
  template <class String>
  void hashStrings(const std::vector<String>& string_vec,
                   std::vector<uint32_t>& hashes) const noexcept;
  template <class T, class String>
  void getOrAddBulkRemote(const std::vector<String>& string_vec, T* encoded_vec);
  int32_t getUnlocked(const std::string& str) const noexcept;
  std::string getStringUnlocked(int32_t string_id) const noexcept;
  std::string getStringChecked(const int string_id) const noexcept;
  std::pair<char*, size_t> getStringBytesChecked(const int string_id) const noexcept;
  template <class String>
  uint32_t computeBucket(const uint32_t hash,
                         const String& str,
                         const std::vector<int32_t>& data) const noexcept;
  template <class String>
  uint32_t computeBucketFromStorageAndMemory(
      const uint32_t input_string_rk_hash,
      const String& input_string,
      const std::vector<int32_t>& string_id_hash_table,
      const size_t storage_high_water_mark,
      const std::vector<String>& input_strings,
      const std::vector<size_t>& string_memory_ids) const noexcept;
  uint32_t computeUniqueBucketWithHash(const uint32_t hash,
                                       const std::vector<int32_t>& data) noexcept;
  void checkAndConditionallyIncreasePayloadCapacity(const size_t write_length);
  void checkAndConditionallyIncreaseOffsetCapacity(const size_t write_length);

  template <class String>
  void appendToStorage(String str) noexcept;
  template <class String>
  void appendToStorageBulk(const std::vector<String>& input_strings,
                           const std::vector<size_t>& string_memory_ids,
                           const size_t sum_new_strings_lengths) noexcept;
  PayloadString getStringFromStorage(const int string_id) const noexcept;
  std::string_view getStringFromStorageFast(const int string_id) const noexcept;
  void addPayloadCapacity(const size_t min_capacity_requested = 0) noexcept;
  void addOffsetCapacity(const size_t min_capacity_requested = 0) noexcept;
  size_t addStorageCapacity(int fd, const size_t min_capacity_requested = 0) noexcept;
  void* addMemoryCapacity(void* addr,
                          size_t& mem_size,
                          const size_t min_capacity_requested = 0) noexcept;
  void invalidateInvertedIndex() noexcept;
  std::vector<int32_t> getEquals(std::string pattern,
                                 std::string comp_operator,
                                 size_t generation);
  void buildSortedCache();
  void insertInSortedCache(std::string str, int32_t str_id);
  void sortCache(std::vector<int32_t>& cache);
  void mergeSortedCache(std::vector<int32_t>& temp_sorted_cache);
  compare_cache_value_t* binary_search_cache(const std::string& pattern) const;

  size_t str_count_;
  size_t collisions_;
  std::vector<int32_t> string_id_hash_table_;
  std::vector<uint32_t> rk_hashes_;
  std::vector<int32_t> sorted_cache;
  bool isTemp_;
  bool materialize_hashes_;
  std::string offsets_path_;
  int payload_fd_;
  int offset_fd_;
  StringIdxEntry* offset_map_;
  char* payload_map_;
  size_t offset_file_size_;
  size_t payload_file_size_;
  size_t payload_file_off_;
  mutable mapd_shared_mutex rw_mutex_;
  mutable std::map<std::tuple<std::string, bool, bool, char>, std::vector<int32_t>>
      like_cache_;
  mutable std::map<std::pair<std::string, char>, std::vector<int32_t>> regex_cache_;
  mutable std::map<std::string, int32_t> equal_cache_;
  mutable DictionaryCache<std::string, compare_cache_value_t> compare_cache_;
  mutable std::shared_ptr<std::vector<std::string>> strings_cache_;
  std::unique_ptr<StringDictionaryClient> client_;
  std::unique_ptr<StringDictionaryClient> client_no_timeout_;

  char* CANARY_BUFFER{nullptr};
  size_t canary_buffer_size = 0;
};

int32_t truncate_to_generation(const int32_t id, const size_t generation);

void translate_string_ids(std::vector<int32_t>& dest_ids,
                          const LeafHostInfo& dict_server_host,
                          const DictRef dest_dict_ref,
                          const std::vector<int32_t>& source_ids,
                          const DictRef source_dict_ref,
                          const int32_t dest_generation);

#endif  // STRINGDICTIONARY_STRINGDICTIONARY_H
