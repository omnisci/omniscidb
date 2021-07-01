/*
 * Copyright 2021 OmniSci, Inc.
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

#include "L0Mgr.h"
#include "Logger/Logger.h"

namespace l0 {

const std::vector<std::shared_ptr<L0Device>>& L0Driver::devices() const {
  CHECK(false);
  return {};
}

std::shared_ptr<L0CommandQueue> L0Device::command_queue() const {
  CHECK(false);
  return nullptr;
}
std::unique_ptr<L0CommandList> L0Device::create_command_list() const {
  CHECK(false);
  return nullptr;
}

std::shared_ptr<L0Module> L0Device::create_module(uint8_t* code,
                                                  size_t len,
                                                  bool log) const {
  CHECK(false);
  return nullptr;
}

std::shared_ptr<L0Kernel> L0Module::create_kernel(const char* name,
                                                  uint32_t x,
                                                  uint32_t y,
                                                  uint32_t z) const {
  CHECK(false);
  return nullptr;
}

void L0CommandList::copy(void* dst, const void* src, const size_t num_bytes) {
  CHECK(false);
}

void L0CommandList::submit(L0CommandQueue& queue) {
  CHECK(false);
}

void* allocate_device_mem(const size_t num_bytes, L0Device& device) {
  CHECK(false);
  return nullptr;
}

L0Manager::L0Manager() {}

void L0Manager::copyHostToDevice(int8_t* device_ptr,
                                 const int8_t* host_ptr,
                                 const size_t num_bytes,
                                 const int device_num) {
  CHECK(false);
}
void L0Manager::copyDeviceToHost(int8_t* host_ptr,
                                 const int8_t* device_ptr,
                                 const size_t num_bytes,
                                 const int device_num) {
  CHECK(false);
}
void L0Manager::copyDeviceToDevice(int8_t* dest_ptr,
                                   int8_t* src_ptr,
                                   const size_t num_bytes,
                                   const int dest_device_num,
                                   const int src_device_num) {
  CHECK(false);
}

int8_t* L0Manager::allocatePinnedHostMem(const size_t num_bytes) {
  CHECK(false);
  return nullptr;
}
int8_t* L0Manager::allocateDeviceMem(const size_t num_bytes, const int device_num) {
  CHECK(false);
  return nullptr;
}
void L0Manager::freePinnedHostMem(int8_t* host_ptr) {
  CHECK(false);
}
void L0Manager::freeDeviceMem(int8_t* device_ptr) {
  CHECK(false);
}
void L0Manager::zeroDeviceMem(int8_t* device_ptr,
                              const size_t num_bytes,
                              const int device_num) {
  CHECK(false);
}
void L0Manager::setDeviceMem(int8_t* device_ptr,
                             const unsigned char uc,
                             const size_t num_bytes,
                             const int device_num) {
  CHECK(false);
}

void L0Manager::synchronizeDevices() const {
  CHECK(false);
}

const std::vector<std::shared_ptr<L0Driver>>& L0Manager::drivers() const {
  return drivers_;
}
}  // namespace l0