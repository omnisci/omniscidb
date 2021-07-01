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
#include "L0Exception.h"

#include <level_zero/ze_api.h>

namespace l0 {
L0Exception::L0Exception(L0result status) : status_(status) {}

const char* L0Exception::what() const noexcept {
  switch (status_) {
    case ZE_RESULT_NOT_READY:
      return "L0 error: synchronization primitive not signaled";
    case ZE_RESULT_ERROR_DEVICE_LOST:
      return "L0 error: device hung, reset, was removed, or driver update occurred";
    case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
      return "L0 error: insufficient host memory to satisfy call";
    case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
      return "L0 error: insufficient device memory to satisfy call";
    case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE:
      return "L0 error: error occurred when building module, see build log for details";
    case ZE_RESULT_ERROR_MODULE_LINK_FAILURE:
      return "L0 error: error occurred when linking modules, see build log for details";
    case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS:
      return "L0 error: access denied due to permission level";
    case ZE_RESULT_ERROR_NOT_AVAILABLE:
      return "L0 error: resource already in use and simultaneous access not allowed or "
             "resource was removed";
    case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE:
      return "L0 error: external required dependency is unavailable or missing";
    case ZE_RESULT_ERROR_UNINITIALIZED:
      return "L0 error: driver is not initialized";
    case ZE_RESULT_ERROR_UNSUPPORTED_VERSION:
      return "L0 error: generic error code for unsupported versions";
    case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
      return "L0 error: generic error code for unsupported features";
    case ZE_RESULT_ERROR_INVALID_ARGUMENT:
      return "L0 error: generic error code for invalid arguments";
    case ZE_RESULT_ERROR_INVALID_NULL_HANDLE:
      return "L0 error: handle argument is not valid";
    case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
      return "L0 error: object pointed to by handle still in-use by device";
    case ZE_RESULT_ERROR_INVALID_NULL_POINTER:
      return "L0 error: pointer argument may not be nullptr";
    case ZE_RESULT_ERROR_INVALID_SIZE:
      return "L0 error: size argument is invalid (e.g., must not be zero)";
    case ZE_RESULT_ERROR_UNSUPPORTED_SIZE:
      return "L0 error: size argument is not supported by the device (e.g., too large)";
    case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
      return "L0 error: alignment argument is not supported by the device";
    case ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
      return "L0 error: synchronization object in invalid state";
    case ZE_RESULT_ERROR_INVALID_ENUMERATION:
      return "L0 error: enumerator argument is not valid";
    case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
      return "L0 error: enumerator argument is not supported by the device";
    case ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
      return "L0 error: image format is not supported by the device";
    case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY:
      return "L0 error: native binary is not supported by the device";
    case ZE_RESULT_ERROR_INVALID_GLOBAL_NAME:
      return "L0 error: global variable is not found in the module";
    case ZE_RESULT_ERROR_INVALID_KERNEL_NAME:
      return "L0 error: kernel name is not found in the module";
    case ZE_RESULT_ERROR_INVALID_FUNCTION_NAME:
      return "L0 error: function name is not found in the module";
    case ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
      return "L0 error: group size dimension is not valid for the kernel or device";
    case ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
      return "L0 error: global width dimension is not valid for the kernel or device";
    case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
      return "L0 error: kernel argument index is not valid for kernel";
    case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
      return "L0 error: kernel argument size does not match kernel";
    case ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
      return "L0 error: value of kernel attribute is not valid for the kernel or device";
    case ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED:
      return "L0 error: module with imports needs to be linked before kernels can be "
             "created from it";
    case ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE:
      return "L0 error: command list type does not match command queue type";
    case ZE_RESULT_ERROR_OVERLAPPING_REGIONS:
      return "L0 error: copy operations do not support overlapping regions of memory";
    case ZE_RESULT_ERROR_UNKNOWN:
      return "L0 error: unknown or internal error";
    default:
      return "L0 unexpected error code";
  }
}
}  // namespace l0