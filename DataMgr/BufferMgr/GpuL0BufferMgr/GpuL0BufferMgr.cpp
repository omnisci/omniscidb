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

#include "DataMgr/BufferMgr/GpuL0BufferMgr/GpuL0BufferMgr.h"

#include "DataMgr/BufferMgr/GpuL0BufferMgr/GpuL0Buffer.h"
#include "L0Mgr/L0Mgr.h"
#include "Logger/Logger.h"

namespace Buffer_Namespace {

GpuL0BufferMgr::GpuL0BufferMgr(const int device_id,
                               const size_t max_buffer_pool_size,
                               l0::L0Manager* l0_mgr,
                               const size_t min_slab_size,
                               const size_t max_slab_size,
                               const size_t page_size,
                               AbstractBufferMgr* parent_mgr)
    : BufferMgr(device_id,
                max_buffer_pool_size,
                min_slab_size,
                max_slab_size,
                page_size,
                parent_mgr)
    , l0_mgr_(l0_mgr) {}

GpuL0BufferMgr::~GpuL0BufferMgr() {
  try {
    l0_mgr_->synchronizeDevices();
    freeAllMem();
  } catch (const std::exception& e) {
    LOG(ERROR) << "L0 Error: " << e.what();
  }
}

void GpuL0BufferMgr::addSlab(const size_t slab_size) {
  slabs_.resize(slabs_.size() + 1);
  try {
    slabs_.back() = l0_mgr_->allocateDeviceMem(slab_size, device_id_);
  } catch (std::exception& error) {
    slabs_.resize(slabs_.size() - 1);
    throw FailedToCreateSlab(slab_size);
  }
  slab_segments_.resize(slab_segments_.size() + 1);
  slab_segments_[slab_segments_.size() - 1].push_back(
      BufferSeg(0, slab_size / page_size_));
}

void GpuL0BufferMgr::freeAllMem() {
  for (auto buf_it = slabs_.begin(); buf_it != slabs_.end(); ++buf_it) {
    l0_mgr_->freeDeviceMem(*buf_it);
  }
}

void GpuL0BufferMgr::allocateBuffer(BufferList::iterator seg_it,
                                    const size_t page_size,
                                    const size_t initial_size) {
  new GpuL0Buffer(this, seg_it, device_id_, l0_mgr_, page_size, initial_size);
}

}  // namespace Buffer_Namespace
