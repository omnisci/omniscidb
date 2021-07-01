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

#include "DataMgr/BufferMgr/BufferMgr.h"

namespace l0 {
class L0Manager;
}

namespace Buffer_Namespace {

class GpuL0BufferMgr : public BufferMgr {
 public:
  GpuL0BufferMgr(const int device_id,
                 const size_t max_buffer_pool_size,
                 l0::L0Manager* l0_mgr,
                 const size_t min_slab_size,
                 const size_t max_slab_size,
                 const size_t page_size,
                 AbstractBufferMgr* parent_mgr = 0);
  inline MgrType getMgrType() override { return GPU_MGR; }
  inline std::string getStringMgrType() override { return ToString(GPU_MGR); }
  ~GpuL0BufferMgr() override;

 private:
  void addSlab(const size_t slab_size) override;
  void freeAllMem() override;
  void allocateBuffer(BufferList::iterator seg_it,
                      const size_t page_size,
                      const size_t initial_size) override;
  l0::L0Manager* l0_mgr_;
};

}  // namespace Buffer_Namespace
