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

#include <memory>
#include <vector>

#include "QueryEngine/JoinHashTable/HashTable.h"

class PerfectHashTable : public HashTable {
 public:
  // CPU + GPU constructor
  PerfectHashTable(const Catalog_Namespace::Catalog* catalog,
                   const HashType layout,
                   const ExecutorDeviceType device_type,
                   const size_t entry_count,
                   const size_t emitted_keys_count)
      : catalog_(catalog)
      , layout_(layout)
      , entry_count_(entry_count)
      , emitted_keys_count_(emitted_keys_count) {
    if (device_type == ExecutorDeviceType::CPU) {
      cpu_hash_table_buff_size_ = layout_ == HashType::OneToOne
                                      ? entry_count_
                                      : 2 * entry_count_ + emitted_keys_count_;
      cpu_hash_table_buff_.reset(new int32_t[cpu_hash_table_buff_size_]);
    }
  }

  ~PerfectHashTable() {
    CHECK(catalog_);
    auto& data_mgr = catalog_->getDataMgr();
    if (gpu_hash_table_buff_) {
      data_mgr.free(gpu_hash_table_buff_);
    }
  }

  size_t gpuReservedSize() const {
    CHECK(gpu_hash_table_buff_);
    return gpu_hash_table_buff_->reservedSize();
  }

  void allocateGpuMemory(const size_t entries, const int device_id) {
    CHECK(catalog_);
    auto& data_mgr = catalog_->getDataMgr();
    CHECK_GE(device_id, 0);
    CHECK(!gpu_hash_table_buff_);
    gpu_hash_table_buff_ = CudaAllocator::allocGpuAbstractBuffer(
        &data_mgr, entries * sizeof(int32_t), device_id);
  }

  size_t getHashTableBufferSize(const ExecutorDeviceType device_type) const override {
    if (device_type == ExecutorDeviceType::CPU) {
      return cpu_hash_table_buff_size_ *
             sizeof(decltype(cpu_hash_table_buff_)::element_type);
    } else {
      return gpu_hash_table_buff_ ? gpu_hash_table_buff_->reservedSize() : 0;
    }
  }

  HashType getLayout() const override { return layout_; }

  int8_t* getCpuBuffer() override {
    return reinterpret_cast<int8_t*>(cpu_hash_table_buff_.get());
  }

  int8_t* getGpuBuffer() const override {
    return gpu_hash_table_buff_ ? gpu_hash_table_buff_->getMemoryPtr() : nullptr;
  }

  size_t getEntryCount() const override { return entry_count_; }

  size_t getEmittedKeysCount() const override { return emitted_keys_count_; }

 private:
  Data_Namespace::AbstractBuffer* gpu_hash_table_buff_{nullptr};
  const Catalog_Namespace::Catalog* catalog_;
  std::unique_ptr<int32_t[]> cpu_hash_table_buff_;
  size_t cpu_hash_table_buff_size_;

  HashType layout_;
  size_t entry_count_;         // number of keys in the hash table
  size_t emitted_keys_count_;  // number of keys emitted across all rows
};
