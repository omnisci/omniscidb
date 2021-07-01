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

#pragma once

#include "DataMgr/BufferMgr/Buffer.h"

namespace l0 {
class L0Manager;
}

namespace Buffer_Namespace {

class GpuL0Buffer : public Buffer {
 public:
  GpuL0Buffer(BufferMgr* bm,
              BufferList::iterator seg_it,
              const int device_id,
              l0::L0Manager* l0_mgr,
              const size_t page_size = 512,
              const size_t num_bytes = 0);
  inline Data_Namespace::MemoryLevel getType() const override { return GPU_LEVEL; }

 private:
  void readData(int8_t* const dst,
                const size_t num_bytes,
                const size_t offset = 0,
                const MemoryLevel dst_buffer_type = CPU_LEVEL,
                const int dst_devic_id = -1) override;
  void writeData(int8_t* const src,
                 const size_t num_bytes,
                 const size_t offset = 0,
                 const MemoryLevel src_buffer_type = CPU_LEVEL,
                 const int src_device_id = -1) override;

  l0::L0Manager* l0_mgr_;
};
}  // namespace Buffer_Namespace
