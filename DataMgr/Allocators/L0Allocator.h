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

#pragma once

#include <cstdint>
#include <cstdlib>

#include "DataMgr/Allocators/DeviceAllocator.h"

namespace Data_Namespace {
class AbstractBuffer;
class DataMgr;
}  // namespace Data_Namespace

class L0Allocator : public DeviceAllocator {
 public:
  L0Allocator(Data_Namespace::DataMgr* data_mgr, const int device_id);

  ~L0Allocator() override;

  static Data_Namespace::AbstractBuffer* allocGpuAbstractBuffer(
      Data_Namespace::DataMgr* data_mgr,
      const size_t num_bytes,
      const int device_id);

  static void freeGpuAbstractBuffer(Data_Namespace::DataMgr* data_mgr,
                                    Data_Namespace::AbstractBuffer* ab);

  int8_t* alloc(const size_t num_bytes) override;

  void free(Data_Namespace::AbstractBuffer* ab) const override;

  void copyToDevice(int8_t* device_dst,
                    const int8_t* host_src,
                    const size_t num_bytes) const override;

  void copyFromDevice(int8_t* host_dst,
                      const int8_t* device_src,
                      const size_t num_bytes) const override;

  void zeroDeviceMem(int8_t* device_ptr, const size_t num_bytes) const override;

  void setDeviceMem(int8_t* device_ptr,
                    unsigned char uc,
                    const size_t num_bytes) const override;

 private:
  std::vector<Data_Namespace::AbstractBuffer*> owned_buffers_;

  Data_Namespace::DataMgr* data_mgr_;
  int device_id_;
};
