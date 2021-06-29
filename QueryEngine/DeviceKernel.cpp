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

#include "DeviceKernel.h"

#include "CompilationContext.h"
#include "NvidiaKernel.h"

#ifdef HAVE_CUDA
class CudaEventClock : public DeviceClock {
 public:
  CudaEventClock() {
    cuEventCreate(&start_, 0);
    cuEventCreate(&stop_, 0);
  }
  virtual void start() override { cuEventRecord(start_, 0); }
  virtual int stop() override {
    cuEventRecord(stop_, 0);
    cuEventSynchronize(stop_);
    float ms = 0;
    cuEventElapsedTime(&ms, start_, stop_);
    return ms;
  }

 private:
  CUevent start_, stop_;  // preparation
};

class NvidiaKernel : public DeviceKernel {
 public:
  NvidiaKernel(const CompilationContext* ctx, int device_id) : device_id(device_id) {
    auto cuda_ctx = dynamic_cast<const GpuCompilationContext*>(ctx);
    CHECK(cuda_ctx);
    const auto native_code = cuda_ctx->getNativeCode(device_id);
    function_ptr = static_cast<CUfunction>(native_code.first);
    module_ptr = static_cast<CUmodule>(native_code.second);
  }

  void launch(unsigned int gridDimX,
              unsigned int gridDimY,
              unsigned int gridDimZ,
              unsigned int blockDimX,
              unsigned int blockDimY,
              unsigned int blockDimZ,
              unsigned int sharedMemBytes,
              void** kernelParams) override {
    checkCudaErrors(cuLaunchKernel(function_ptr,
                                   gridDimX,
                                   gridDimY,
                                   gridDimZ,
                                   blockDimX,
                                   blockDimY,
                                   blockDimZ,
                                   sharedMemBytes,
                                   nullptr,
                                   kernelParams,
                                   nullptr));
  }

  void initializeDynamicWatchdog(bool could_interrupt, uint64_t cycle_budget) override {
    CHECK(module_ptr);
    CUevent start, stop;
    cuEventCreate(&start, 0);
    cuEventCreate(&stop, 0);
    cuEventRecord(start, 0);

    CUdeviceptr dw_cycle_budget;
    size_t dw_cycle_budget_size;
    // Translate milliseconds to device cycles
    if (device_id == 0) {
      LOG(INFO) << "Dynamic Watchdog budget: GPU: "
                << std::to_string(g_dynamic_watchdog_time_limit) << "ms, "
                << std::to_string(cycle_budget) << " cycles";
    }
    checkCudaErrors(cuModuleGetGlobal(
        &dw_cycle_budget, &dw_cycle_budget_size, module_ptr, "dw_cycle_budget"));
    CHECK_EQ(dw_cycle_budget_size, sizeof(uint64_t));
    checkCudaErrors(cuMemcpyHtoD(
        dw_cycle_budget, reinterpret_cast<void*>(&cycle_budget), sizeof(uint64_t)));

    CUdeviceptr dw_sm_cycle_start;
    size_t dw_sm_cycle_start_size;
    checkCudaErrors(cuModuleGetGlobal(
        &dw_sm_cycle_start, &dw_sm_cycle_start_size, module_ptr, "dw_sm_cycle_start"));
    CHECK_EQ(dw_sm_cycle_start_size, 128 * sizeof(uint64_t));
    checkCudaErrors(cuMemsetD32(dw_sm_cycle_start, 0, 128 * 2));

    if (!could_interrupt) {
      // Executor is not marked as interrupted, make sure dynamic watchdog doesn't block
      // execution
      CUdeviceptr dw_abort;
      size_t dw_abort_size;
      checkCudaErrors(
          cuModuleGetGlobal(&dw_abort, &dw_abort_size, module_ptr, "dw_abort"));
      CHECK_EQ(dw_abort_size, sizeof(uint32_t));
      checkCudaErrors(cuMemsetD32(dw_abort, 0, 1));
    }

    cuEventRecord(stop, 0);
    cuEventSynchronize(stop);
    float milliseconds = 0;
    cuEventElapsedTime(&milliseconds, start, stop);
    VLOG(1) << "Device " << std::to_string(device_id)
            << ": launchGpuCode: dynamic watchdog init: " << std::to_string(milliseconds)
            << " ms\n";
  }

  void initializeRuntimeInterrupter() override {
    CHECK(module_ptr);
    CUevent start, stop;
    cuEventCreate(&start, 0);
    cuEventCreate(&stop, 0);
    cuEventRecord(start, 0);

    CUdeviceptr runtime_interrupt_flag;
    size_t runtime_interrupt_flag_size;
    checkCudaErrors(cuModuleGetGlobal(&runtime_interrupt_flag,
                                      &runtime_interrupt_flag_size,
                                      module_ptr,
                                      "runtime_interrupt_flag"));
    CHECK_EQ(runtime_interrupt_flag_size, sizeof(uint32_t));
    checkCudaErrors(cuMemsetD32(runtime_interrupt_flag, 0, 1));

    cuEventRecord(stop, 0);
    cuEventSynchronize(stop);
    float milliseconds = 0;
    cuEventElapsedTime(&milliseconds, start, stop);
    VLOG(1) << "Device " << std::to_string(device_id)
            << ": launchGpuCode: runtime query interrupter init: "
            << std::to_string(milliseconds) << " ms";
  }

  std::unique_ptr<DeviceClock> make_clock() override {
    return std::make_unique<CudaEventClock>();
  }

 private:
  CUfunction function_ptr;
  CUmodule module_ptr;
  int device_id;
};
#endif

std::unique_ptr<DeviceKernel> create_device_kernel(const CompilationContext* ctx,
                                                   int device_id) {
#ifdef HAVE_CUDA
  return std::make_unique<NvidiaKernel>(ctx, device_id);
#else
  return nullptr;
#endif
}
