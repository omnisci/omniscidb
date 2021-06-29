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

#include <memory>

class CompilationContext;

class DeviceClock {
 public:
  virtual void start() = 0;
  virtual int stop() = 0;
};

class DeviceKernel {
 public:
  virtual void launch(unsigned int gridDimX,
                      unsigned int gridDimY,
                      unsigned int gridDimZ,
                      unsigned int blockDimX,
                      unsigned int blockDimY,
                      unsigned int blockDimZ,
                      unsigned int sharedMemBytes,
                      void** kernelParams) = 0;

  virtual void initializeDynamicWatchdog(bool could_interrupt, uint64_t cycle_budget) = 0;
  virtual void initializeRuntimeInterrupter() = 0;

  virtual std::unique_ptr<DeviceClock> make_clock() = 0;
};

std::unique_ptr<DeviceKernel> create_device_kernel(const CompilationContext* ctx,
                                                   int device_id);
