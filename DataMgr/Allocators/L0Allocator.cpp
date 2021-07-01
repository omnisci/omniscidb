/*
 * Copyright 2021 MapD Technologies, Inc.
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

#include "DataMgr/Allocators/L0Allocator.h"

#include <DataMgr/DataMgr.h>
#include <L0Mgr/L0Mgr.h>
#include <Logger/Logger.h>
#include <Shared/types.h>

L0Allocator::L0Allocator(Data_Namespace::DataMgr* data_mgr, const int device_id)
    : data_mgr_(data_mgr), device_id_(device_id) {
  CHECK(data_mgr_);
}

L0Allocator::~L0Allocator() {
  CHECK(data_mgr_);
  for (auto& buffer_ptr : owned_buffers_) {
    data_mgr_->free(buffer_ptr);
  }
}

Data_Namespace::AbstractBuffer* L0Allocator::allocGpuAbstractBuffer(
    Data_Namespace::DataMgr* data_mgr,
    const size_t num_bytes,
    const int device_id) {
  CHECK(data_mgr);
  auto ab = data_mgr->alloc(Data_Namespace::GPU_LEVEL, device_id, num_bytes);
  CHECK_EQ(ab->getPinCount(), 1);
  return ab;
}

void L0Allocator::freeGpuAbstractBuffer(Data_Namespace::DataMgr* data_mgr,
                                        Data_Namespace::AbstractBuffer* ab) {
  CHECK(data_mgr);
  data_mgr->free(ab);
}

int8_t* L0Allocator::alloc(const size_t num_bytes) {
  CHECK(data_mgr_);
  owned_buffers_.emplace_back(
      L0Allocator::allocGpuAbstractBuffer(data_mgr_, num_bytes, device_id_));
  return owned_buffers_.back()->getMemoryPtr();
}

void L0Allocator::free(Data_Namespace::AbstractBuffer* ab) const {
  data_mgr_->free(ab);
}

void L0Allocator::copyToDevice(int8_t* device_dst,
                               const int8_t* host_src,
                               const size_t num_bytes) const {
  const auto l0_mgr = data_mgr_->getL0Mgr();
  CHECK(l0_mgr);
  l0_mgr->copyHostToDevice(device_dst, host_src, num_bytes, device_id_);
}

void L0Allocator::copyFromDevice(int8_t* host_dst,
                                 const int8_t* device_src,
                                 const size_t num_bytes) const {
  const auto l0_mgr = data_mgr_->getL0Mgr();
  CHECK(l0_mgr);
  l0_mgr->copyDeviceToHost(host_dst, device_src, num_bytes, device_id_);
}

void L0Allocator::zeroDeviceMem(int8_t* device_ptr, const size_t num_bytes) const {
  const auto l0_mgr = data_mgr_->getL0Mgr();
  CHECK(l0_mgr);
  l0_mgr->zeroDeviceMem(device_ptr, num_bytes, device_id_);
}

void L0Allocator::setDeviceMem(int8_t* device_ptr,
                               unsigned char uc,
                               const size_t num_bytes) const {
  const auto l0_mgr = data_mgr_->getL0Mgr();
  CHECK(l0_mgr);
  l0_mgr->setDeviceMem(device_ptr, uc, num_bytes, device_id_);
}
