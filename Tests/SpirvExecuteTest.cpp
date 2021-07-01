#include <gtest/gtest.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

#include <LLVMSPIRVLib/LLVMSPIRVLib.h>
#include <level_zero/ze_api.h>

#include "L0Mgr/L0Mgr.h"
#include "TestHelpers.h"

#define L0_SAFE_CALL(call)                                               \
  {                                                                      \
    auto status = (call);                                                \
    if (status) {                                                        \
      std::cerr << "L0 error: " << std::hex << (int)status << " ";       \
      std::cerr << std::dec << __FILE__ << ":" << __LINE__ << std::endl; \
      exit(status);                                                      \
    }                                                                    \
  }

template <typename T, size_t N>
struct alignas(4096) AlignedArray {
  T data[N];
};

class SPIRVExecuteTest : public ::testing::Test {
 protected:
  std::string generateSimpleSPIRV();
};

static void L0InitContext(ze_driver_handle_t& hDriver,
                          ze_device_handle_t& hDevice,
                          ze_module_handle_t& hModule,
                          ze_command_queue_handle_t& hCommandQueue,
                          ze_context_handle_t& hContext,
                          std::string& spv) {
  ze_init_flag_t init_flag;
  L0_SAFE_CALL(zeInit(0));

  // Discover all the driver instances
  uint32_t driverCount = 0;
  L0_SAFE_CALL(zeDriverGet(&driverCount, nullptr));

  ze_driver_handle_t* allDrivers =
      (ze_driver_handle_t*)malloc(driverCount * sizeof(ze_driver_handle_t));
  L0_SAFE_CALL(zeDriverGet(&driverCount, allDrivers));

  // Find a driver instance with a GPU device
  for (uint32_t i = 0; i < driverCount; ++i) {
    uint32_t deviceCount = 0;
    hDriver = allDrivers[i];
    L0_SAFE_CALL(zeDeviceGet(hDriver, &deviceCount, nullptr));
    ze_device_handle_t* allDevices =
        (ze_device_handle_t*)malloc(deviceCount * sizeof(ze_device_handle_t));
    L0_SAFE_CALL(zeDeviceGet(hDriver, &deviceCount, allDevices));
    for (uint32_t d = 0; d < deviceCount; ++d) {
      ze_device_properties_t device_properties;
      L0_SAFE_CALL(zeDeviceGetProperties(allDevices[d], &device_properties));
      if (ZE_DEVICE_TYPE_GPU == device_properties.type) {
        hDevice = allDevices[d];
        break;
      }
    }
    free(allDevices);
    if (nullptr != hDevice) {
      break;
    }
  }
  free(allDrivers);
  assert(hDriver);
  assert(hDevice);

  // Create context
  ze_context_desc_t ctxtDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  zeContextCreate(hDriver, &ctxtDesc, &hContext);
  // Create a command queue
  ze_command_queue_desc_t commandQueueDesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                              nullptr,
                                              0,  // computeQueueGroupOrdinal
                                              0,  // index
                                              0,  // flags
                                              ZE_COMMAND_QUEUE_MODE_DEFAULT,
                                              ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  L0_SAFE_CALL(
      zeCommandQueueCreate(hContext, hDevice, &commandQueueDesc, &hCommandQueue));

  size_t codeSize = spv.size();
  assert(codeSize != 0 && "Code size is 0.");
  unsigned char* codeBin = new unsigned char[codeSize];
  std::copy(spv.data(), spv.data() + codeSize, codeBin);

  std::ofstream out("complete.spv", std::ios::binary);
  out.write((char*)codeBin, codeSize);

  assert(codeSize && "CodeBin is empty");

  ze_module_desc_t moduleDesc;
  moduleDesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  moduleDesc.pNext = nullptr;
  moduleDesc.pBuildFlags = "";
  moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  moduleDesc.inputSize = codeSize;
  moduleDesc.pConstants = nullptr;
  moduleDesc.pInputModule = (uint8_t*)codeBin;

  ze_module_build_log_handle_t buildlog = nullptr;
  L0_SAFE_CALL(zeModuleCreate(hContext, hDevice, &moduleDesc, &hModule, &buildlog));
  size_t szLog = 0;
  L0_SAFE_CALL(zeModuleBuildLogGetString(buildlog, &szLog, nullptr));
  std::cout << "Got build log size " << szLog << std::endl;
  char* strLog = (char*)malloc(szLog);
  L0_SAFE_CALL(zeModuleBuildLogGetString(buildlog, &szLog, strLog));
  std::fstream log;
  log.open("log.txt", std::ios::app);
  if (!log.good()) {
    std::cerr << "Unable to open log file" << std::endl;
    exit(1);
  }
  log << strLog;
  log.close();
}

std::string SPIRVExecuteTest::generateSimpleSPIRV() {
  using namespace llvm;
  // See source at https://github.com/kurapov-peter/L0Snippets
  LLVMContext ctx;
  std::unique_ptr<Module> module = std::make_unique<Module>("code_generated", ctx);
  module->setTargetTriple("spir-unknown-unknown");
  IRBuilder<> builder(ctx);

  std::vector<Type*> args{Type::getFloatPtrTy(ctx, 1), Type::getFloatPtrTy(ctx, 1)};
  FunctionType* f_type = FunctionType::get(Type::getVoidTy(ctx), args, false);
  Function* f = Function::Create(f_type,
                                 GlobalValue::LinkageTypes::ExternalLinkage,
                                 "sum_first_elements",
                                 module.get());
  f->setCallingConv(CallingConv::SPIR_KERNEL);

  // get_global_id
  FunctionType* ggi_type =
      FunctionType::get(Type::getInt32Ty(ctx), {Type::getInt32Ty(ctx)}, false);
  Function* get_global_idj = Function::Create(ggi_type,
                                              GlobalValue::LinkageTypes::ExternalLinkage,
                                              "_Z13get_global_idj",
                                              module.get());
  get_global_idj->setCallingConv(CallingConv::SPIR_FUNC);

  BasicBlock* entry = BasicBlock::Create(ctx, "entry", f);

  builder.SetInsertPoint(entry);
  Constant* zero = ConstantInt::get(Type::getInt32Ty(ctx), 0);
  Value* idx = builder.CreateCall(get_global_idj, zero, "idx");
  auto argit = f->args().begin();
  Value* firstElemSrc = builder.CreateGEP(f->args().begin(), idx, "src.idx");
  Value* firstElemDst = builder.CreateGEP(++argit, idx, "dst.idx");
  Value* ldSrc = builder.CreateLoad(Type::getFloatTy(ctx), firstElemSrc, "ld.src");
  Value* ldDst = builder.CreateLoad(Type::getFloatTy(ctx), firstElemDst, "ld.dst");

  Value* result = builder.CreateFAdd(ldSrc, ldDst, "result");
  builder.CreateStore(result, firstElemDst);
  builder.CreateRetVoid();

  // set metadata -- pretend we're opencl (see
  // https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/master/docs/SPIRVRepresentationInLLVM.rst#spir-v-instructions-mapped-to-llvm-metadata)
  Metadata* spirv_src_ops[] = {
      ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(ctx), 3 /*OpenCL_C*/)),
      ConstantAsMetadata::get(
          ConstantInt::get(Type::getInt32Ty(ctx), 102000 /*OpenCL ver 1.2*/))};
  NamedMDNode* spirv_src = module->getOrInsertNamedMetadata("spirv.Source");
  spirv_src->addOperand(MDNode::get(ctx, spirv_src_ops));

  module->print(errs(), nullptr);

  SPIRV::TranslatorOpts opts;
  opts.enableAllExtensions();
  opts.setDesiredBIsRepresentation(SPIRV::BIsRepresentation::OpenCL12);
  opts.setDebugInfoEIS(SPIRV::DebugInfoEIS::OpenCL_DebugInfo_100);

  std::ostringstream ss;
  std::string err;
  auto success = writeSpirv(module.get(), opts, ss, err);
  assert(success);

  return ss.str();
}

TEST_F(SPIRVExecuteTest, TranslateSimple) {
  auto spv = generateSimpleSPIRV();

  ze_device_handle_t hDevice = nullptr;
  ze_module_handle_t hModule = nullptr;
  ze_driver_handle_t hDriver = nullptr;
  ze_command_queue_handle_t hCommandQueue = nullptr;
  ze_context_handle_t hContext = nullptr;
  L0InitContext(hDriver, hDevice, hModule, hCommandQueue, hContext, spv);

  ze_command_list_handle_t hCommandList;
  ze_kernel_handle_t hKernel;

  ze_command_list_desc_t commandListDesc;
  commandListDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
  commandListDesc.pNext = nullptr;
  commandListDesc.flags = 0;
  commandListDesc.commandQueueGroupOrdinal = 0;

  L0_SAFE_CALL(zeCommandListCreate(hContext, hDevice, &commandListDesc, &hCommandList));

  ze_kernel_desc_t kernelDesc;
  kernelDesc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
  kernelDesc.pNext = nullptr;
  kernelDesc.flags = 0;
  kernelDesc.pKernelName = "sum_first_elements";

  L0_SAFE_CALL(zeKernelCreate(hModule, &kernelDesc, &hKernel));

  constexpr int a_size = 32;
  AlignedArray<float, a_size> a, b;
  for (auto i = 0; i < a_size; ++i) {
    a.data[i] = 5;
    b.data[i] = 6;
  }

  ze_device_mem_alloc_desc_t alloc_desc;
  alloc_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  alloc_desc.pNext = nullptr;
  alloc_desc.flags = 0;
  alloc_desc.ordinal = 0;

  const float copy_size = a_size * sizeof(float);
  void* dA = nullptr;
  L0_SAFE_CALL(
      zeMemAllocDevice(hContext, &alloc_desc, copy_size, 0 /*align*/, hDevice, &dA));
  void* dB = nullptr;
  L0_SAFE_CALL(
      zeMemAllocDevice(hContext, &alloc_desc, copy_size, 0 /*align*/, hDevice, &dB));

  int sz = 32;
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(void*), &dA));
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 1, sizeof(void*), &dB));

  void* a_void = a.data;
  L0_SAFE_CALL(zeCommandListAppendMemoryCopy(
      hCommandList, dA, a_void, copy_size, nullptr, 0, nullptr));
  void* b_void = b.data;
  L0_SAFE_CALL(zeCommandListAppendMemoryCopy(
      hCommandList, dB, b_void, copy_size, nullptr, 0, nullptr));

  L0_SAFE_CALL(zeCommandListAppendBarrier(hCommandList, nullptr, 0, nullptr));
  L0_SAFE_CALL(zeKernelSetGroupSize(hKernel, 1, 1, 1));
  ze_group_count_t dispatchTraits = {1, 1, 1};

  L0_SAFE_CALL(zeCommandListAppendLaunchKernel(
      hCommandList, hKernel, &dispatchTraits, nullptr, 0, nullptr));

  L0_SAFE_CALL(zeCommandListAppendBarrier(hCommandList, nullptr, 0, nullptr));
  L0_SAFE_CALL(zeCommandListAppendMemoryCopy(
      hCommandList, b_void, dB, copy_size, nullptr, 0, nullptr));

  L0_SAFE_CALL(zeCommandListClose(hCommandList));
  L0_SAFE_CALL(
      zeCommandQueueExecuteCommandLists(hCommandQueue, 1, &hCommandList, nullptr));
  L0_SAFE_CALL(
      zeCommandQueueSynchronize(hCommandQueue, std::numeric_limits<uint32_t>::max()));

  for (int i = 0; i < a_size; ++i) {
    std::cout << b.data[i] << " ";
  }
  std::cout << std::endl;

  ASSERT_EQ(b.data[0], 11);
  ASSERT_EQ(b.data[1], 6);
  ASSERT_EQ(b.data[2], 6);

  L0_SAFE_CALL(zeMemFree(hContext, dA));
  L0_SAFE_CALL(zeMemFree(hContext, dB));
}

TEST_F(SPIRVExecuteTest, TranslateSimpleWithL0Wrapper) {
  auto mgr = std::make_shared<l0::L0Manager>();
  auto driver = mgr->drivers()[0];
  auto device = driver->devices()[0];

  auto spv = generateSimpleSPIRV();

  auto module = device->create_module((uint8_t*)spv.data(), spv.length());

  auto command_queue = device->command_queue();
  auto command_list = device->create_command_list();

  constexpr int a_size = 32;
  AlignedArray<float, a_size> a, b;
  for (auto i = 0; i < a_size; ++i) {
    a.data[i] = 5;
    b.data[i] = 6;
  }

  ze_device_mem_alloc_desc_t alloc_desc;
  alloc_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  alloc_desc.pNext = nullptr;
  alloc_desc.flags = 0;
  alloc_desc.ordinal = 0;

  const float copy_size = a_size * sizeof(float);

  void* dA = l0::allocate_device_mem(copy_size, *device);
  void* dB = l0::allocate_device_mem(copy_size, *device);

  void* a_void = a.data;
  void* b_void = b.data;

  command_list->copy(dA, a_void, copy_size);
  command_list->copy(dB, b_void, copy_size);

  auto kernel = module->create_kernel("sum_first_elements", 1, 1, 1);

  command_list->launch(*kernel, &dA, &dB);
  command_list->launch(*kernel, &dA, &dB);

  command_list->copy(b_void, dB, copy_size);

  command_list->submit(*command_queue);

  for (int i = 0; i < a_size; ++i) {
    std::cout << b.data[i] << " ";
  }
  std::cout << std::endl;

  ASSERT_EQ(b.data[0], 16);
  ASSERT_EQ(b.data[1], 6);
  ASSERT_EQ(b.data[2], 6);

  L0_SAFE_CALL(zeMemFree(device->ctx(), dA));
  L0_SAFE_CALL(zeMemFree(device->ctx(), dB));
}

TEST_F(SPIRVExecuteTest, TranslateSimpleWithL0Manager) {
  auto mgr = std::make_shared<l0::L0Manager>();
  auto driver = mgr->drivers()[0];
  auto device = driver->devices()[0];

  auto spv = generateSimpleSPIRV();

  auto module = device->create_module((uint8_t*)spv.data(), spv.length());

  auto command_queue = device->command_queue();
  auto command_list = device->create_command_list();

  constexpr int a_size = 32;
  AlignedArray<float, a_size> a, b;
  for (auto i = 0; i < a_size; ++i) {
    a.data[i] = 5;
    b.data[i] = 6;
  }

  const float copy_size = a_size * sizeof(float);
  void* dA = mgr->allocateDeviceMem(copy_size, 0);
  void* dB = mgr->allocateDeviceMem(copy_size, 0);

  void* a_void = a.data;
  void* b_void = b.data;

  mgr->copyHostToDevice((int8_t*)dA, (int8_t*)a_void, copy_size, 0);
  mgr->copyHostToDevice((int8_t*)dB, (int8_t*)b_void, copy_size, 0);

  auto kernel = module->create_kernel("sum_first_elements", 1, 1, 1);

  command_list->launch(*kernel, &dA, &dB);
  command_list->launch(*kernel, &dA, &dB);

  command_list->copy(b_void, dB, copy_size);

  command_list->submit(*command_queue);

  for (int i = 0; i < a_size; ++i) {
    std::cout << b.data[i] << " ";
  }
  std::cout << std::endl;

  ASSERT_EQ(b.data[0], 16);
  ASSERT_EQ(b.data[1], 6);
  ASSERT_EQ(b.data[2], 6);

  mgr->freeDeviceMem((int8_t*)dA);
  mgr->freeDeviceMem((int8_t*)dB);
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  int err = RUN_ALL_TESTS();
  return err;
}
