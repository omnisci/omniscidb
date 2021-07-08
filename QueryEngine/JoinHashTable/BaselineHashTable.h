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

#include "Catalog/Catalog.h"
#include "DataMgr/AbstractBuffer.h"
#include "DataMgr/Allocators/CudaAllocator.h"
#include "QueryEngine/CompilationOptions.h"
#include "QueryEngine/JoinHashTable/HashJoin.h"

#include "QueryEngine/JoinHashTable/HashTable.h"

class BaselineHashTable : public HashTable {
 public:
  // CPU constructor
  BaselineHashTable(const Catalog_Namespace::Catalog* catalog,
                    HashType layout,
                    const size_t entry_count,
                    const size_t emitted_keys_count,
                    const size_t hash_table_size)
      : cpu_hash_table_buff_size_(hash_table_size)
      , gpu_hash_table_buff_(nullptr)
#ifdef HAVE_CUDA
      , device_id_(0)
      , catalog_(catalog)
#endif
      , layout_(layout)
      , entry_count_(entry_count)
      , emitted_keys_count_(emitted_keys_count) {
    cpu_hash_table_buff_.reset(new int8_t[cpu_hash_table_buff_size_]);
  }

  // GPU constructor
  BaselineHashTable(const Catalog_Namespace::Catalog* catalog,
                    HashType layout,
                    const size_t entry_count,
                    const size_t emitted_keys_count,
                    const size_t hash_table_size,
                    const size_t device_id)
      : gpu_hash_table_buff_(nullptr)
#ifdef HAVE_CUDA
      , device_id_(device_id)
      , catalog_(catalog)
#endif
      , layout_(layout)
      , entry_count_(entry_count)
      , emitted_keys_count_(emitted_keys_count) {
#ifdef HAVE_CUDA
    CHECK(catalog_);
    auto& data_mgr = catalog_->getDataMgr();
    gpu_hash_table_buff_ =
        CudaAllocator::allocGpuAbstractBuffer(&data_mgr, hash_table_size, device_id_);
#else
    UNREACHABLE();
#endif
  }

  ~BaselineHashTable() {
#ifdef HAVE_CUDA
    CHECK(catalog_);
    auto& data_mgr = catalog_->getDataMgr();
    if (gpu_hash_table_buff_) {
      data_mgr.free(gpu_hash_table_buff_);
    }
#endif
  }

  int8_t* getGpuBuffer() const override {
    return gpu_hash_table_buff_ ? gpu_hash_table_buff_->getMemoryPtr() : nullptr;
  }

  size_t getHashTableBufferSize(const ExecutorDeviceType device_type) const override {
    if (device_type == ExecutorDeviceType::CPU) {
      return cpu_hash_table_buff_size_ *
             sizeof(decltype(cpu_hash_table_buff_)::element_type);
    } else {
      return gpu_hash_table_buff_ ? gpu_hash_table_buff_->reservedSize() : 0;
    }
  }

  int8_t* getCpuBuffer() override {
    return reinterpret_cast<int8_t*>(cpu_hash_table_buff_.get());
  }

  HashType getLayout() const override { return layout_; }
  size_t getEntryCount() const override { return entry_count_; }
  size_t getEmittedKeysCount() const override { return emitted_keys_count_; }

 private:
  std::unique_ptr<int8_t[]> cpu_hash_table_buff_;
  size_t cpu_hash_table_buff_size_;
  Data_Namespace::AbstractBuffer* gpu_hash_table_buff_;

#ifdef HAVE_CUDA
  const size_t device_id_;
  const Catalog_Namespace::Catalog* catalog_;
#endif

  HashType layout_;
  size_t entry_count_;         // number of keys in the hash table
  size_t emitted_keys_count_;  // number of keys emitted across all rows
};
