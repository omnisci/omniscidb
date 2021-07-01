#include "QueryEngine/L0Kernel.h"

#include "L0Mgr/L0Exception.h"
#include "L0Mgr/Utils.h"
#include "Logger/Logger.h"  // CHECK

#include <iostream>

L0BinResult spv_to_bin(const std::string& spv,
                       const std::string& name,
                       const unsigned block_size,
                       const l0::L0Manager* mgr) {
  CHECK(!spv.empty());
  CHECK(mgr);

  auto driver = mgr->drivers()[0];
  auto device = driver->devices()[0];

  CHECK(driver);
  CHECK(device);

  std::ofstream out("complete.spv", std::ios::binary);
  out.write((char*)spv.data(), spv.size());

  auto module = device->create_module((uint8_t*)spv.data(), spv.size(), true);
  auto kernel = module->create_kernel(name.c_str(), 1, 1, 1);

  return {kernel, module};
}

L0DeviceCompilationContext::L0DeviceCompilationContext(
    std::shared_ptr<l0::L0Kernel> kernel,
    std::shared_ptr<l0::L0Module> module,
    const l0::L0Manager* l0_mgr,
    const int device_id,
    unsigned int num_options,
    void** option_vals)
    : kernel_(kernel), module_(module), l0_mgr_(l0_mgr), device_id_(device_id) {}

L0DeviceCompilationContext::~L0DeviceCompilationContext() {}