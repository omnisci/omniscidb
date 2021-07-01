cmake_minimum_required(VERSION 3.16)

# Required for LLVM / JIT, alternative to setting CMP0065 to OLD
set(CMAKE_ENABLE_EXPORTS True)

set(ENABLE_CONDA OFF)
if(DEFINED ENV{CONDA_PREFIX})
  set(ENABLE_CONDA ON)
  list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
  # resolves link issue for zlib 
  link_directories("$ENV{CONDA_PREFIX}/lib")
  # various fixes and workarounds
  add_definitions("-Dsecure_getenv=getenv")
  # fixes `undefined reference to `boost::system::detail::system_category_instance'`:
  add_definitions("-DBOOST_ERROR_CODE_HEADER_ONLY")
  # Adding formating macros
  add_definitions("-D__STDC_FORMAT_MACROS=1")
  # fixes always_inline attribute errors
  add_compile_options("-fno-semantic-interposition")
  # Adding `--sysroot=...` resolves `no member named 'signbit' in the global namespace` error:
  set(CMAKE_SYSROOT "$ENV{CONDA_BUILD_SYSROOT}")
endif(DEFINED ENV{CONDA_PREFIX})

# force `Release` build type if left unspecified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
string(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_LOWER)

if("${CMAKE_VERSION}" VERSION_GREATER 3.11.999)
  cmake_policy(SET CMP0074 NEW)
endif()

find_program(CCACHE_EXE ccache)
if(CCACHE_EXE)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_EXE}")
endif()

option(ENABLE_IWYU "Enable include-what-you-use" OFF)
if(ENABLE_IWYU)
  find_program(IWYU_EXE include-what-you-use)
  if(IWYU_EXE)
    set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE "${IWYU_EXE}")
  endif()
endif()

project(omnisci)

if(NOT EXISTS "${CMAKE_SOURCE_DIR}/Rendering")
  set(MAPD_EDITION "OS")
elseif(NOT DEFINED MAPD_EDITION)
  set(MAPD_EDITION "EE")
endif()
set(MAPD_EDITION "${MAPD_EDITION}" CACHE STRING "MapD edition" FORCE)
set_property(CACHE MAPD_EDITION PROPERTY STRINGS "EE" "CE" "OS")
add_definitions("-DMAPD_EDITION_${MAPD_EDITION}")
string(TOLOWER "${MAPD_EDITION}" MAPD_EDITION_LOWER)

set(MAPD_VERSION_MAJOR "5")
set(MAPD_VERSION_MINOR "7")
set(MAPD_VERSION_PATCH "0")
set(MAPD_VERSION_EXTRA "dev")
set(MAPD_VERSION_RAW "${MAPD_VERSION_MAJOR}.${MAPD_VERSION_MINOR}.${MAPD_VERSION_PATCH}${MAPD_VERSION_EXTRA}")
set(MAPD_IMMERSE_URL "http://builds.mapd.com/frontend/mapd2-dashboard-v2-137-release-prod.zip")
string(TIMESTAMP MAPD_BUILD_DATE "%Y%m%d")

if($ENV{BUILD_NUMBER})
  set(MAPD_BUILD_NUMBER "$ENV{BUILD_NUMBER}")
else()
  set(MAPD_BUILD_NUMBER "dev")
endif()
set(MAPD_VERSION "${MAPD_VERSION_RAW}-${MAPD_BUILD_NUMBER}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

add_custom_target(clean-all
  COMMAND ${CMAKE_BUILD_TOOL} clean
 )

option(PREFER_STATIC_LIBS "Prefer linking against static libraries" OFF)
if(PREFER_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(Arrow_USE_STATIC_LIBS ON)
  set(Boost_USE_STATIC_LIBS ON)
  set(OPENSSL_USE_STATIC_LIBS ON)
  set(Thrift_USE_STATIC_LIBS ON)
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
  endif()

  set(CUDA_USE_STATIC_CUDA_RUNTIME ON CACHE STRING "Use static CUDA runtime")

  # On ppc, build failures occur for targets that depend on locale related functions due to unresolved symbols that are
  # present in the stdc++ library. Add the library flag to these targets to be used in resolving these symbols.
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "ppc64le")
    set(LOCALE_LINK_FLAG "-lstdc++")
  endif()
else()
  add_definitions("-DBOOST_LOG_DYN_LINK")
endif()

# Required for macOS with Boost 1.71.0+
# See https://gitlab.kitware.com/cmake/cmake/issues/19714
set(Boost_NO_BOOST_CMAKE 1)

option(ENABLE_JAVA_REMOTE_DEBUG "Enable Java Remote Debug" OFF )
if(ENABLE_JAVA_REMOTE_DEBUG)
  add_definitions("-DENABLE_JAVA_REMOTE_DEBUG")
endif()

option(ENABLE_L0 "Enable level zero support" OFF)
if(ENABLE_L0)
  find_package(LevelZero REQUIRED COMPONENTS ${LevelZero_COMPONENTS})
  add_definitions("-DHAVE_L0")
endif()

option(ENABLE_CUDA "Enable CUDA support" ON)
if(ENABLE_CUDA)
  find_package(CUDA REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS})
  list(APPEND CUDA_LIBRARIES ${CUDA_CUDA_LIBRARY})
  add_definitions("-DHAVE_CUDA")

  set(MAPD_HOST_COMPILER "" CACHE STRING "Host compiler to use with nvcc")
  if(NOT "${MAPD_HOST_COMPILER}" STREQUAL "")
      set(MAPD_HOST_COMPILER_FLAG "-ccbin=${MAPD_HOST_COMPILER}")
  endif()
else()
  set(CUDA_LIBRARIES "")
  set(MAPD_PACKAGE_FLAGS "${MAPD_PACKAGE_FLAGS}-cpu")
endif()

# CUDA architecture flags
if("${CMAKE_BUILD_TYPE_LOWER}" STREQUAL "debug")
  option(ENABLE_ONLY_ONE_ARCH "Enable quicker building for only one GPU arch" ON)
else()
  option(ENABLE_ONLY_ONE_ARCH "Enable quicker building for only one GPU arch" OFF)
endif()
if(ENABLE_CUDA)
  if(ENABLE_ONLY_ONE_ARCH)
    execute_process(
      COMMAND cmake -S ${CMAKE_SOURCE_DIR}/NvidiaComputeCapability -B NvidiaComputeCapability
      OUTPUT_QUIET
      ERROR_QUIET
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    execute_process(
      COMMAND cmake --build NvidiaComputeCapability
      OUTPUT_FILE ${CMAKE_BINARY_DIR}/NvidiaComputeCapability/build.out.txt
      ERROR_FILE ${CMAKE_BINARY_DIR}/NvidiaComputeCapability/build.err.txt
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    set(NVIDIA_COMPUTE_CAPABILITY "")
    if (EXISTS ${CMAKE_BINARY_DIR}/NvidiaComputeCapability.txt)
      file(STRINGS ${CMAKE_BINARY_DIR}/NvidiaComputeCapability.txt NVIDIA_COMPUTE_CAPABILITY)
    endif()
  endif()
  if (ENABLE_ONLY_ONE_ARCH AND NOT "${NVIDIA_COMPUTE_CAPABILITY}" STREQUAL "")
    set (CUDA_COMPILATION_ARCH
      -gencode=arch=compute_${NVIDIA_COMPUTE_CAPABILITY},code=compute_${NVIDIA_COMPUTE_CAPABILITY}
      -Wno-deprecated-gpu-targets
    )
    add_custom_target(clean_nvidia_compute_capability
      COMMAND ${CMAKE_BUILD_TOOL} clean
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/NvidiaComputeCapability
    )
    add_dependencies(clean-all clean_nvidia_compute_capability)
    message(STATUS "CUDA_COMPILATION_ARCH: ${CUDA_COMPILATION_ARCH}")
  else()
    set (CUDA_COMPILATION_ARCH
      -gencode=arch=compute_35,code=compute_35
      -gencode=arch=compute_50,code=compute_50
      -gencode=arch=compute_60,code=compute_60
      -gencode=arch=compute_70,code=compute_70
      -gencode=arch=compute_75,code=compute_75
      -Wno-deprecated-gpu-targets
    )
    if(ENABLE_ONLY_ONE_ARCH)
      message(STATUS "ENABLE_ONLY_ONE_ARCH ignored because NvidiaComputeCapability.txt not found or not readable")
      message(STATUS "CUDA_COMPILATION_ARCH: ${CUDA_COMPILATION_ARCH}")
    endif()
  endif()
endif()

option(ENABLE_NVTX "Enable NVidia Tools Extension library" OFF)
if(ENABLE_NVTX)
  if(NOT ENABLE_CUDA)
    set(ENABLE_NVTX OFF CACHE BOOL "Enable NVidia Tools Extension library" FORCE)
    message(STATUS "Cuda must be enabled to use NVTX, disabling NVTX support.")
  else()
    find_package(NVTX)
    if (NVTX_FOUND)
      message(STATUS "Using NVTX profiling markers")
      add_definitions("-DHAVE_NVTX")
    else()
      set(ENABLE_NVTX OFF CACHE BOOL "Enable NVidia Tools Extension library" FORCE)
      message(STATUS "NVTX not found, disabling NVTX support.")
    endif()
  endif()
endif()

option(SUPPRESS_NULL_LOGGER_DEPRECATION_WARNINGS "Suppress NullLogger deprecated warnings.")
if (SUPPRESS_NULL_LOGGER_DEPRECATION_WARNINGS)
  add_definitions("-DSUPPRESS_NULL_LOGGER_DEPRECATION_WARNINGS")
endif()

option(ENABLE_CUDA_KERNEL_DEBUG "Enable debugging symbols for CUDA device Kernels" OFF)

option(ENABLE_JIT_DEBUG "Enable debugging symbols for the JIT" OFF)
if (ENABLE_JIT_DEBUG)
  add_definitions("-DWITH_JIT_DEBUG")
endif()

if(XCODE)
  if(ENABLE_CUDA)
    set(CMAKE_EXE_LINKER_FLAGS "-F/Library/Frameworks -framework CUDA")
  endif()
  add_definitions("-DXCODE")
endif()

option(ENABLE_GEOS "Enable GEOS Support" ON)
if (ENABLE_GEOS)
  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(ENABLE_GEOS OFF CACHE BOOL "Enable GEOS support" FORCE)
    message(STATUS "GEOS functionality is not supported on macOS.")
  else()
    find_package(GEOS)
    if(GEOS_NOTFOUND)
      set(ENABLE_GEOS OFF CACHE BOOL "Enable GEOS support" FORCE)
      message(STATUS "GEOS not found, disabling support.")
    else()
      set(GEOS_LIBRARY_FILENAME '"${GEOS_LIBRARY}"')
      add_definitions("-DENABLE_GEOS -DGEOS_LIBRARY_FILENAME=${GEOS_LIBRARY_FILENAME}")
      set(GEOS_RT_DEFINITIONS "-DENABLE_GEOS")
    endif()
  endif()
endif()

# fixme: hack works for Homebrew, might not work for Conda
if(ENABLE_CONDA)
  set(OPENSSL_ROOT_DIR "$ENV{CONDA_PREFIX}")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin" )
  set(OPENSSL_ROOT_DIR "/usr/local/opt/openssl/")
endif()

find_package(OpenSSL REQUIRED)
include_directories(${OPENSSL_INCLUDE_DIR})

if(MSVC)
  find_package(Thrift CONFIG REQUIRED)
  set(Thrift_INCLUDE_DIRS ${THRIFT_INCLUDE_DIR})
  set(Thrift_EXECUTABLE "${THRIFT_BIN_DIR}/thrift.exe")
  set(Thrift_LIBRARIES ${THRIFT_LIBRARIES})
else()
  find_package(Thrift REQUIRED)
endif()
include_directories(${Thrift_INCLUDE_DIRS})
if("${Thrift_VERSION}" VERSION_LESS "0.11.0")
  add_definitions("-DHAVE_THRIFT_BOOST_SHAREDPTR")
else()
  add_definitions("-DHAVE_THRIFT_STD_SHAREDPTR")
endif()
if("${Thrift_VERSION}" VERSION_LESS "0.13.0")
  add_definitions("-DHAVE_THRIFT_PLATFORMTHREADFACTORY")
else()
  add_definitions("-DHAVE_THRIFT_THREADFACTORY")
endif()

find_package(Git)
find_package(Glog REQUIRED)
find_package(PNG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(GDAL REQUIRED)
find_package(GDALExtra REQUIRED)
find_package(BLOSC REQUIRED)
list(APPEND GDAL_LIBRARIES ${PNG_LIBRARIES} ${GDALExtra_LIBRARIES})
include_directories(${GDAL_INCLUDE_DIRS})

option(ENABLE_FOLLY "Use Folly" ON)
if(ENABLE_FOLLY)
  find_package(Folly)
  if(NOT Folly_FOUND)
    set(ENABLE_FOLLY OFF CACHE BOOL "Use Folly" FORCE)
  else()
    include_directories(${Folly_INCLUDE_DIRS})
    add_definitions("-DHAVE_FOLLY")
    list(APPEND Folly_LIBRARIES ${Glog_LIBRARIES})
    # TODO: use Folly::folly_deps?
    if(MSVC)
      find_package(Libevent REQUIRED)
      list(APPEND Folly_LIBRARIES libevent::core)
    endif()
  endif()
endif()

option(ENABLE_MLPACK "Use mlpack" OFF)
if(ENABLE_MLPACK)
  find_package(OpenMP REQUIRED)
  find_package(Armadillo REQUIRED)
  find_package(Boost COMPONENTS serialization REQUIRED)
  find_package(MLPACK REQUIRED)
  include_directories(${MLPACK_INCLUDE_DIRS})
  add_definitions("-DHAVE_MLPACK")
endif()

if(MSVC)
  include_directories(include_directories("${LIBS_PATH}/include/pdcurses"))
else()
  find_package(Curses)
  include_directories(${CURSES_INCLUDE_DIRS})
  if (CURSES_HAVE_NCURSES_CURSES_H AND NOT CURSES_HAVE_CURSES_H)
    include_directories(${CURSES_INCLUDE_DIRS}/ncurses/)
  endif()
endif()

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /permissive- /EHsc /std:c++17 /D NOMINMAX /D WIN32_LEAN_AND_MEAN")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-local-typedefs -fdiagnostics-color=auto -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS")
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-register")
  endif()
  if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
  endif()
endif()

if (ENABLE_MLPACK)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}") 
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-declarations")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
endif()

# address and thread sanitizer
option(ENABLE_STANDALONE_CALCITE "Require standalone Calcite server" OFF)
option(ENABLE_ASAN "Enable address sanitizer" OFF)
option(ENABLE_TSAN "Enable thread sanitizer" OFF)
option(ENABLE_UBSAN "Enable undefined behavior sanitizer" OFF)
if(ENABLE_ASAN)
  set(SAN_FLAGS "-fsanitize=address -O1 -fno-omit-frame-pointer")
  add_definitions("-DWITH_DECODERS_BOUNDS_CHECKING")
elseif(ENABLE_TSAN)
  add_definitions("-DHAVE_TSAN")
  # Copy the config directory to the build dir for TSAN suppressions
  file(COPY config DESTINATION ${CMAKE_BINARY_DIR})

  set(SAN_FLAGS "-fsanitize=thread -fPIC -O1 -fno-omit-frame-pointer")
  # required for older GCC, see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=64354
  add_definitions("-D__SANITIZE_THREAD__")
elseif(ENABLE_UBSAN)
  set(SAN_FLAGS "-fsanitize=undefined -fPIC -O1 -fno-omit-frame-pointer")
endif()
if(ENABLE_ASAN OR ENABLE_TSAN OR ENABLE_UBSAN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SAN_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SAN_FLAGS}")
  set(ENABLE_STANDALONE_CALCITE ON)
endif()

# Embedded database
option(ENABLE_DBE "Enable embedded database" OFF)
if(ENABLE_DBE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  set(ENABLE_STANDALONE_CALCITE ON)
  add_definitions("-DENABLE_EMBEDDED_DATABASE")
  add_definitions("-DDBEngine_LIBNAME=\"${CMAKE_SHARED_LIBRARY_PREFIX}DBEngine${CMAKE_SHARED_LIBRARY_SUFFIX}\"")
endif()

# Code coverage
option(ENABLE_CODE_COVERAGE "Enable compile time code coverage" OFF)
if(ENABLE_CODE_COVERAGE)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(COVERAGE_FLAGS "-fprofile-instr-generate -fcoverage-mapping")
  else()
    message(FATAL_ERROR "Code coverage currently only supported with Clang compiler")
  endif()
  set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE ON)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COVERAGE_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COVERAGE_FLAGS}")
endif()

option(ENABLE_DECODERS_BOUNDS_CHECKING "Enable bounds checking for column decoding" OFF)

if(ENABLE_STANDALONE_CALCITE)
  add_definitions("-DSTANDALONE_CALCITE")
endif()

include_directories(${CMAKE_SOURCE_DIR}
                    ${CMAKE_SOURCE_DIR}/Parser
                    ${CMAKE_CURRENT_BINARY_DIR})

## Dependencies

# LLVM
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  list(APPEND CMAKE_PREFIX_PATH "/usr/local/opt/llvm")
endif()
find_package(LLVM CONFIG REQUIRED)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

find_library(CLANG_LIB clang-cpp)
find_library(LLVM_LIB LLVM)
# Deps builds use separate libs for each clang component, while some distros now bundle into a single lib
if (${CMAKE_SYSTEM_NAME} STREQUAL "Darwin" OR NOT LLVM_LIB)
  set(LLVM_COMPONENTS support mcjit core irreader option linker)

  option(ENABLE_INTEL_JIT_LISTENER "Enable Intel Vtune JIT Listener" OFF)
  if(ENABLE_INTEL_JIT_LISTENER)
    list(APPEND LLVM_COMPONENTS inteljitevents)
  endif()

  llvm_map_components_to_libnames(llvm_libs ${LLVM_TARGETS_TO_BUILD} ${LLVM_COMPONENTS})
  set(clang_libs
      clangFrontend
      clangSerialization
      clangDriver
      clangTooling
      clangParse
      clangSema
      clangAnalysis
      clangEdit
      clangAST
      clangLex
      clangBasic
      clangRewrite
      clangRewriteFrontend)

  # LLVMSupport explicitly lists tinfo in its INTERFACE_LINK_LIBRARIES, even
  # though we provide it in our build of ncurses. Since LLVMSupport is listed
  # as a requirement for other llvm libs, we need to walk through the entire
  # list in order to remove all instances of tinfo.
  foreach(lib ${llvm_libs})
    get_target_property(interface_libs ${lib} INTERFACE_LINK_LIBRARIES)
    list(REMOVE_ITEM interface_libs tinfo z rt pthread -lpthread m dl)
    set_target_properties(${lib} PROPERTIES INTERFACE_LINK_LIBRARIES "${interface_libs}")
  endforeach()

  list(APPEND llvm_libs ${CURSES_NCURSES_LIBRARY})
else()
  if(NOT CLANG_LIB)
    message(FATAL_ERROR "Could not find CLANG library.")
  endif()

  set(clang_libs ${CLANG_LIB})
  set(llvm_libs ${LLVM_LIB})
endif()

# Clang (UDF Compiler)
find_package(Clang REQUIRED)
include_directories(${CLANG_INCLUDE_DIRS})
add_definitions(${CLANG_DEFINITIONS})


# Boost
find_package(Boost COMPONENTS log log_setup filesystem program_options regex system thread timer locale iostreams REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

# Allow explicit include statements to access third party headers directly.
# Ex: raft/canonical/include/raft.h
include_directories(ThirdParty/)

# EGL
include_directories(ThirdParty/egl)

# Google Test and Google Mock
if(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
  add_definitions("-DGTEST_USE_OWN_TR1_TUPLE=0")
endif()
include_directories(ThirdParty/googletest)
add_subdirectory(ThirdParty/googletest)

# Google Benchmark
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)
if(WIN32)
  set(HAVE_POSIX_REGEX 0)
endif()
add_subdirectory(ThirdParty/googlebenchmark)

# aws-sdk
option(ENABLE_AWS_S3 "Enable AWS S3 support" ON)
if(ENABLE_AWS_S3)
  find_package(LibAwsS3)
  if(NOT LibAwsS3_FOUND)
    set(ENABLE_AWS_S3 OFF CACHE BOOL "Enable AWS S3 support" FORCE)
  else()
    add_definitions("-DHAVE_AWS_S3")
  endif()
endif()

# Arrow
find_package(Arrow REQUIRED)
add_definitions("-DARROW_NO_DEPRECATED_API")
include_directories(${Arrow_INCLUDE_DIRS})

option(ENABLE_IMPORT_PARQUET "Enable Parquet Importer support" ON)
if(ENABLE_IMPORT_PARQUET)
  find_package(Parquet)
  if(NOT Parquet_FOUND)
    set(ENABLE_IMPORT_PARQUET OFF CACHE BOOL "Enable Parquet Importer support" FORCE)
    message(STATUS "Parquet not found. Disabling Parquet Importer support.")
  else()
    add_definitions("-DENABLE_IMPORT_PARQUET")
    # when we found libparquet it means we're using arrow 11+
    # and deps scripts must have built parquet as well as snappy
    find_package(Snappy REQUIRED)
  endif()
endif()

list(APPEND Arrow_LIBRARIES ${Snappy_LIBRARIES})
if(ENABLE_AWS_S3)
  list(INSERT Arrow_LIBRARIES 0 ${LibAwsS3_LIBRARIES})
endif()
if (ENABLE_CUDA)
  list(INSERT Arrow_LIBRARIES 0 ${Arrow_GPU_CUDA_LIBRARIES})
endif()

# RapidJSON
include_directories(ThirdParty/rapidjson)
add_definitions(-DRAPIDJSON_HAS_STDSTRING)

# Linenoise
add_subdirectory(ThirdParty/linenoise)

# SQLite
include_directories(ThirdParty/sqlite3)
add_subdirectory(ThirdParty/sqlite3)

# raft/canonical
option(ENABLE_CANONICAL_RAFT "Enable Canonical Raft" OFF)
if(ENABLE_CANONICAL_RAFT)
  add_subdirectory(ThirdParty/raft/canonical)
endif()

# rdkafka
find_package(RdKafka REQUIRED)
include_directories(${RdKafka_INCLUDE_DIRS})

# libarchive
find_package(LibArchive REQUIRED)
include_directories(${LibArchive_INCLUDE_DIRS})

find_package(CURL REQUIRED QUIET)
if(CURL_FOUND)
  set(CURL_LIBRARIES ${LibAwsS3_SUPPORT_LIBRARIES})
endif()

# bcrypt
include_directories(ThirdParty/bcrypt)
add_subdirectory(ThirdParty/bcrypt)

# PicoSHA2
include_directories(ThirdParty/PicoSHA2)

if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
# opensaml
  option(ENABLE_SAML "Enable SAML support" ON)
  if(ENABLE_SAML)
    find_package(OpenSaml)
    if(NOT OpenSaml_FOUND)
      set(ENABLE_SAML OFF CACHE BOOL "Enable SAML support" FORCE)
    else()
      add_definitions("-DHAVE_SAML")
    endif()
  endif()
endif()

# TBB
set(TBB_LIBS "")
find_package(TBB)
if(TBB_FOUND)
  message(STATUS "Building with TBB support")
  add_definitions("-DHAVE_TBB")
  list(APPEND TBB_LIBS ${TBB_LIBRARIES})
endif()

set(gen_cpp_files
    ${CMAKE_BINARY_DIR}/gen-cpp/OmniSci.cpp
    ${CMAKE_BINARY_DIR}/gen-cpp/OmniSci.h
    ${CMAKE_BINARY_DIR}/gen-cpp/omnisci_types.cpp
    ${CMAKE_BINARY_DIR}/gen-cpp/common_types.cpp
    ${CMAKE_BINARY_DIR}/gen-cpp/completion_hints_types.cpp
    ${CMAKE_BINARY_DIR}/gen-cpp/serialized_result_set_types.cpp
    ${CMAKE_BINARY_DIR}/gen-cpp/extension_functions_types.cpp
    ${CMAKE_BINARY_DIR}/gen-cpp/extension_functions_types.h
)
add_custom_command(
  DEPENDS
    ${CMAKE_SOURCE_DIR}/omnisci.thrift
    ${CMAKE_SOURCE_DIR}/common.thrift
    ${CMAKE_SOURCE_DIR}/completion_hints.thrift
    ${CMAKE_SOURCE_DIR}/QueryEngine/serialized_result_set.thrift
    ${CMAKE_SOURCE_DIR}/QueryEngine/extension_functions.thrift
  OUTPUT ${gen_cpp_files}
  COMMAND ${Thrift_EXECUTABLE}
  ARGS -gen cpp -r -o ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/omnisci.thrift)
list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/gen-cpp/)

add_custom_target(thrift_gen DEPENDS ${gen_cpp_files})

add_library(mapd_thrift ${gen_cpp_files})

target_compile_options(mapd_thrift PRIVATE -fPIC)
target_link_libraries(mapd_thrift ${Thrift_LIBRARIES})

if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
  include_directories(Catalog/ee)
  include_directories(Distributed/ee)
else()
  include_directories(Catalog/os)
  include_directories(Distributed/os)
endif()

set(MAPD_RENDERING_LIBRARIES "")
if(NOT "${MAPD_EDITION_LOWER}" STREQUAL "os")
  option(ENABLE_RENDERING "Build backend renderer" OFF)
  if(ENABLE_RENDERING)
    # muparserx - currently render-only
    include_directories(ThirdParty/muparserx)
    add_subdirectory(ThirdParty/muparserx)

    option(ENABLE_RENDER_TESTS "Build backend renderer tests" ON)

    add_definitions("-DHAVE_RENDERING")
    add_subdirectory(Rendering)
    add_subdirectory(QueryRenderer)

    include_directories(${RENDERING_INCLUDE_DIRS})
    set(MAPD_RENDERING_LIBRARIES QueryRenderer Rendering)

    set(MAPD_PACKAGE_FLAGS "${MAPD_PACKAGE_FLAGS}-render")
    if(${RENDERER_CONTEXT_TYPE} STREQUAL "GLX")
      set(MAPD_PACKAGE_FLAGS "${MAPD_PACKAGE_FLAGS}-glx")
    endif()
  endif()
else()
  set(ENABLE_RENDERING OFF CACHE BOOL "Build backend renderer" FORCE)
endif()

set(TIME_LIMITED_NUMBER_OF_DAYS "30" CACHE STRING "Number of days this build is valid for if build is time limited")

option(TIME_LIMITED_BUILD "Build Time Limited Build" OFF)
if(TIME_LIMITED_BUILD)
  list(APPEND TIME_LIMITED_DEFINITIONS "TIME_LIMITED_BUILD")
  list(APPEND TIME_LIMITED_DEFINITIONS "TIME_LIMITED_NUMBER_OF_DAYS=${TIME_LIMITED_NUMBER_OF_DAYS}")
  set(MAPD_PACKAGE_FLAGS "${MAPD_PACKAGE_FLAGS}-${TIME_LIMITED_NUMBER_OF_DAYS}d")
endif()

option(ENABLE_PROFILER "Enable google perftools" OFF)
if(ENABLE_PROFILER)
  find_package(Gperftools REQUIRED COMPONENTS TCMALLOC PROFILER)
  set(PROFILER_LIBS ${Gperftools_TCMALLOC} ${Gperftools_PROFILER})
  add_definitions("-DHAVE_PROFILER")
else()
  set(PROFILER_LIBS "")
endif()

add_subdirectory(SqliteConnector)

add_subdirectory(StringDictionary)
add_subdirectory(Calcite)
get_target_property(CalciteThrift_BINARY_DIR calciteserver_thrift BINARY_DIR)
include_directories(${CalciteThrift_BINARY_DIR})

add_subdirectory(Catalog)
add_subdirectory(Parser)
add_subdirectory(Analyzer)
add_subdirectory(ImportExport)
add_subdirectory(QueryEngine)
add_subdirectory(DataMgr)
add_subdirectory(CudaMgr)
add_subdirectory(L0Mgr)
add_subdirectory(LockMgr)
add_subdirectory(Logger)
add_subdirectory(MigrationMgr)
add_subdirectory(Fragmenter)
add_subdirectory(Shared)
add_subdirectory(OSDependent)
add_subdirectory(Utils)
add_subdirectory(QueryRunner)
add_subdirectory(SQLFrontend)
add_subdirectory(TableArchiver)
add_subdirectory(ThriftHandler)
add_subdirectory(Geospatial)
add_subdirectory(Distributed)

if(ENABLE_DBE)
  add_subdirectory(Embedded)
endif()

option(ENABLE_ODBC "Build ODBC driver" OFF)
if(ENABLE_ODBC)
  add_subdirectory(ODBC)
endif()

set(MAPD_LIBRARIES OSDependent Shared Catalog SqliteConnector MigrationMgr TableArchiver Parser Analyzer ImportExport QueryRunner QueryEngine QueryState LockMgr DataMgr Fragmenter Logger Geospatial)

if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
  list(APPEND MAPD_LIBRARIES Distributed)
  if(ENABLE_DISTRIBUTED_5_0)
    list(APPEND MAPD_LIBRARIES StringDictionaryThread)
  endif()
endif()

list(APPEND MAPD_LIBRARIES Calcite)

if(ENABLE_RENDERING)
  add_dependencies(QueryRenderer Parser)
  list(APPEND MAPD_LIBRARIES "${MAPD_RENDERING_LIBRARIES}")
endif()

list(APPEND MAPD_LIBRARIES ${Arrow_LIBRARIES})

if(ENABLE_FOLLY)
  list(APPEND MAPD_LIBRARIES ${Folly_LIBRARIES})
endif()

if(ENABLE_MLPACK)
  list(APPEND MAPD_LIBRARIES ${MLPACK_LIBRARIES} ${ARMADILLO_LIBRARIES})
endif()

if(ENABLE_LICENSING_AWS)
  list(APPEND MAPD_LIBRARIES AWSMarketplace)
endif()

list(APPEND MAPD_LIBRARIES ${TBB_LIBS})

if(ENABLE_CANONICAL_RAFT)
  list(APPEND MAPD_LIBRARIES raft_canonical)
endif()

option(ENABLE_TESTS "Build unit tests" ON)
if (ENABLE_TESTS)
  enable_testing()
  add_subdirectory(Tests)
  add_subdirectory(SampleCode)
endif()

if(ENABLE_RENDERING AND ENABLE_RENDER_TESTS)
  enable_testing()
  add_subdirectory(Tests/RenderTests)
endif()

set(initdb_source_files initdb.cpp)
add_executable(initdb ${initdb_source_files})

set(omnisci_server_source_files MapDServer.cpp ${CMAKE_BINARY_DIR}/MapDRelease.h)

add_executable(omnisci_server ${omnisci_server_source_files})
set_target_properties(omnisci_server PROPERTIES COMPILE_DEFINITIONS "${TIME_LIMITED_DEFINITIONS}")

add_custom_command(
    DEPENDS ${CMAKE_SOURCE_DIR}/omnisci.thrift
    OUTPUT
        ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/com/omnisci/thrift/server/OmniSci.java
        ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/com/omnisci/thrift/server/TRow.java
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/java/thrift/src/gen
    COMMAND ${Thrift_EXECUTABLE}
    ARGS -gen java -r -out ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/ ${CMAKE_SOURCE_DIR}/omnisci.thrift)

add_custom_command(
    DEPENDS ${CMAKE_SOURCE_DIR}/common.thrift
    OUTPUT
        ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/com/omnisci/thrift/server/common.java
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/java/thrift/src/gen
    COMMAND ${Thrift_EXECUTABLE}
    ARGS -gen java -r -out ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/ ${CMAKE_SOURCE_DIR}/common.thrift)

add_custom_command(
    DEPENDS ${CMAKE_SOURCE_DIR}/QueryEngine/serialized_result_set.thrift
    OUTPUT
        ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/com/omnisci/thrift/server/serialized_result_set.java
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/java/thrift/src/gen
    COMMAND ${Thrift_EXECUTABLE}
    ARGS -gen java -r -out ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/ ${CMAKE_SOURCE_DIR}/QueryEngine/serialized_result_set.thrift)

add_custom_command(
    DEPENDS ${CMAKE_SOURCE_DIR}/java/thrift/calciteserver.thrift ${CMAKE_SOURCE_DIR}/completion_hints.thrift ${CMAKE_SOURCE_DIR}/QueryEngine/extension_functions.thrift
    OUTPUT ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/com/omnisci/thrift/calciteserver/CalciteServer.java
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/java/thrift/src/gen
    COMMAND ${Thrift_EXECUTABLE}
    ARGS -gen java -r -I ${CMAKE_SOURCE_DIR} -out ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/ ${CMAKE_SOURCE_DIR}/java/thrift/calciteserver.thrift)

list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/)

execute_process(
  COMMAND ${GIT_EXECUTABLE} rev-parse --short=10 HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE MAPD_GIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )
file(WRITE ${CMAKE_BINARY_DIR}/MAPD_GIT_HASH.txt "${MAPD_GIT_HASH}\n")
file(STRINGS ${CMAKE_BINARY_DIR}/MAPD_GIT_HASH.txt MAPD_GIT_HASH)
set(CPACK_PACKAGE_VERSION "${MAPD_VERSION_RAW}-${MAPD_BUILD_DATE}-${MAPD_GIT_HASH}")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/Shared/release.h"
  "${CMAKE_BINARY_DIR}/MapDRelease.h"
  @ONLY
  )
list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/MAPD_GIT_HASH.txt)
list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/MapDRelease.h)
# required to force regen of MAPD_GIT_HASH.txt, MapDRelease.h
add_custom_target(rerun_cmake ALL
  COMMAND cmake .
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
add_dependencies(omnisci_server rerun_cmake)

target_link_libraries(omnisci_server mapd_thrift thrift_handler ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES} ${PROFILER_LIBS} ${ZLIB_LIBRARIES} ${LOCALE_LINK_FLAG})

target_link_libraries(initdb mapd_thrift DataMgr ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES} ${ZLIB_LIBRARIES})

macro(set_dpkg_arch arch_in arch_out)
  if("${arch_in}" STREQUAL "x86_64")
    set(${arch_out} "amd64")
  elseif("${arch_in}" STREQUAL "aarch64")
    set(${arch_out} "arm64")
  elseif("${arch_in}" STREQUAL "ppc64le")
    set(${arch_out} "ppc64el")
  else()
    set(${arch_out} "${arch_in}")
  endif()
endmacro()

# clang-tidy
find_program(JQ_EXECUTABLE NAMES jq)
if (NOT ${JQ_EXECUTABLE} STREQUAL "JQ_EXECUTABLE-NOTFOUND")
  file(WRITE ${CMAKE_BINARY_DIR}/jq.filter "map(select(.file | test(\".*/(build|ThirdParty)/.*\") | not))")
  add_custom_target(run-clang-tidy
    COMMAND mkdir -p clang-tidy
    COMMAND ${JQ_EXECUTABLE} -f jq.filter ${CMAKE_BINARY_DIR}/compile_commands.json > clang-tidy/compile_commands.json
    COMMAND cd clang-tidy && ${CMAKE_SOURCE_DIR}/ThirdParty/clang/run-clang-tidy.py -quiet -format -fix -header-filter="${CMAKE_SOURCE_DIR}/.*" 2> /dev/null
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
else()
  message(STATUS "jq not found, disabling run-clang-tidy target")
endif()

# doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
  include(${CMAKE_CURRENT_SOURCE_DIR}/docs/CMakeLists.txt)
endif(DOXYGEN_FOUND)

# Packaging

if(NOT "${CMAKE_BUILD_TYPE_LOWER}" STREQUAL "debug" AND NOT "${CMAKE_BUILD_TYPE_LOWER}" STREQUAL "relwithdebinfo")
  set(CPACK_STRIP_FILES ON)
else()
  set(MAPD_PACKAGE_FLAGS "${MAPD_PACKAGE_FLAGS}-debug")
endif()
set(CPACK_PACKAGE_VENDOR "OmniSci, Inc.")
set(CPACK_PACKAGE_CONTACT "support@omnisci.com")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "OmniSci Core Database")
set(CPACK_PROJECT_CONFIG_FILE ${CMAKE_SOURCE_DIR}/CMakePackaging.txt)
set(CPACK_DEBIAN_PACKAGE_DEPENDS "default-jre-headless | openjdk-8-jre-headless | java8-runtime-headless, bsdmainutils, curl | wget")
set(CPACK_RPM_PACKAGE_REQUIRES "java-headless, util-linux, curl")
set(CPACK_RPM_PACKAGE_AUTOREQ OFF)
set(CPACK_RPM_SPEC_MORE_DEFINE "%define __jar_repack %{nil}")
if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}, libldap-2.4-2")
endif()
if(ENABLE_RENDERING)
  set(CPACK_RPM_PACKAGE_REQUIRES "${CPACK_RPM_PACKAGE_REQUIRES}, libX11, libXext")
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}, libx11-6, libxext6")
endif()

set_dpkg_arch(${CMAKE_SYSTEM_PROCESSOR} CPACK_DEBIAN_PACKAGE_ARCHITECTURE)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/licenses" DESTINATION "ThirdParty" COMPONENT "doc")

# OmniSciTypes.h local includes (for UDF/UDTF)
install(FILES Shared/funcannotations.h DESTINATION "Shared/" COMPONENT "include")
install(FILES Shared/InlineNullValues.h DESTINATION "Shared/" COMPONENT "include")
install(FILES Logger/Logger.h DESTINATION "Logger/" COMPONENT "include")

# Frontend
option(MAPD_IMMERSE_DOWNLOAD "Download OmniSci Immerse for packaging" OFF)
set(MAPD_IMMERSE_URL ${MAPD_IMMERSE_URL} CACHE STRING "URL to bundled frontend")
if(MAPD_IMMERSE_DOWNLOAD)
  include(ExternalProject)
  externalproject_add(frontend
    URL ${MAPD_IMMERSE_URL}
    PREFIX external
    CONFIGURE_COMMAND ""
    UPDATE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    LOG_DOWNLOAD on
    )
  externalproject_get_property(frontend source_dir)

  install(DIRECTORY ${source_dir}/ DESTINATION "frontend/" PATTERN .git EXCLUDE PATTERN node_modules EXCLUDE)
  add_custom_command(TARGET frontend COMMAND ${CMAKE_COMMAND} -E copy_directory ${source_dir} frontend)
  list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/frontend)

  ## Go web server
  if("${MAPD_EDITION_LOWER}" STREQUAL "ee" AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Immerse")
    add_subdirectory(Immerse)
  endif()
endif()

add_subdirectory(ThirdParty/generate_cert)

# systemd
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
    install(FILES systemd/omnisci_sd_server.service.in systemd/omnisci_sd_server@.service.in DESTINATION systemd)
    install(FILES systemd/omnisci-sds.conf.in DESTINATION systemd)
  endif()
  install(FILES systemd/omnisci_server.service.in systemd/omnisci_server@.service.in DESTINATION systemd)
  install(FILES systemd/omnisci.conf.in DESTINATION systemd)
  install(PROGRAMS systemd/install_omnisci_systemd.sh DESTINATION systemd)
endif()

## mvn process for java code
find_program(MVN_EXECUTABLE NAMES mvn)
if(NOT MVN_EXECUTABLE)
  message(FATAL_ERROR "mvn not found. Install Apache Maven.")
endif()
file(GLOB_RECURSE JAVA_POM RELATIVE ${CMAKE_SOURCE_DIR} java/**/pom.xml)
file(GLOB_RECURSE JAVA_FTL RELATIVE ${CMAKE_SOURCE_DIR} java/calcite/src/main/codegen/includes/*.ftl)
file(GLOB_RECURSE JAVA_SOURCES RELATIVE ${CMAKE_SOURCE_DIR} java/**/*.java)
list(FILTER JAVA_SOURCES EXCLUDE REGEX ".*/gen/.*")
list(FILTER JAVA_SOURCES EXCLUDE REGEX ".*/generated-sources/.*")

set(OMNISCI_JAR_RELEASE_VERSION "${MAPD_VERSION_MAJOR}.${MAPD_VERSION_MINOR}.${MAPD_VERSION_PATCH}")
if("${MAPD_VERSION_EXTRA}" STREQUAL "dev")
  set (OMNISCI_JAR_RELEASE_VERSION "${OMNISCI_JAR_RELEASE_VERSION}-SNAPSHOT")
endif()

set (OMNISCI_JDBC_JAR "omnisci-jdbc-${OMNISCI_JAR_RELEASE_VERSION}.jar")
set (OMNISCI_UTILITY_JAR "omnisci-utility-${OMNISCI_JAR_RELEASE_VERSION}.jar")

set(MVN_PATH_COMMAND "")
if(MSVC) 
    set(MVN_PATH_COMMAND set "MVNPATH=${CMAKE_SOURCE_DIR}/java" &&)
else()
    set(MVN_PATH_COMMAND "MVNPATH=${CMAKE_SOURCE_DIR}/java")
endif()

add_custom_command(
  OUTPUT
    ${CMAKE_BINARY_DIR}/bin/${OMNISCI_UTILITY_JAR}
    ${CMAKE_BINARY_DIR}/bin/${OMNISCI_JDBC_JAR}
    ${CMAKE_BINARY_DIR}/bin/calcite-1.0-SNAPSHOT-jar-with-dependencies.jar
    COMMAND ${MVN_PATH_COMMAND} ${MVN_EXECUTABLE} -l ${CMAKE_BINARY_DIR}/mvn_build.log -e clean install -Dthrift.version="${Thrift_VERSION}" -Dmaven.compiler.showDeprecation=true -Dmaven.compiler.showWarnings=true -Domnisci.release.version="${OMNISCI_JAR_RELEASE_VERSION}" -Djava.net.preferIPv4Stack=true -Dmaven.wagon.http.retryHandler.count=3
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/java/mapd/target/${OMNISCI_UTILITY_JAR} ${CMAKE_BINARY_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/java/omniscijdbc/target/${OMNISCI_JDBC_JAR} ${CMAKE_BINARY_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/java/calcite/target/calcite-1.0-SNAPSHOT-jar-with-dependencies.jar ${CMAKE_BINARY_DIR}/bin
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/java
  DEPENDS
    ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/com/omnisci/thrift/server/OmniSci.java
    ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/com/omnisci/thrift/calciteserver/CalciteServer.java
    ${CMAKE_SOURCE_DIR}/java/calcite/src/main/codegen/config.fmpp
    ${CMAKE_SOURCE_DIR}/java/pom.xml
    ${CMAKE_SOURCE_DIR}/omnisci.thrift
    ${JAVA_POM}
    ${JAVA_SOURCES}
    ${JAVA_FTL}
  )
add_custom_target(mapd_java_components ALL DEPENDS
  ${CMAKE_BINARY_DIR}/bin/${OMNISCI_UTILITY_JAR}
  ${CMAKE_BINARY_DIR}/bin/${OMNISCI_JDBC_JAR}
  ${CMAKE_BINARY_DIR}/bin/calcite-1.0-SNAPSHOT-jar-with-dependencies.jar)
add_custom_target(mapd_java_clean
  COMMAND ${MVN_PATH_COMMAND} ${MVN_EXECUTABLE} -q clean -Domnisci.release.version="${OMNISCI_JAR_RELEASE_VERSION}"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/java
  )
add_dependencies(clean-all mapd_java_clean)
install(FILES ${CMAKE_SOURCE_DIR}/java/mapd/target/${OMNISCI_UTILITY_JAR} DESTINATION bin COMPONENT "jar")
install(FILES ${CMAKE_SOURCE_DIR}/java/omniscijdbc/target/${OMNISCI_JDBC_JAR} DESTINATION bin COMPONENT "jar")
install(FILES ${CMAKE_SOURCE_DIR}/java/calcite/target/calcite-1.0-SNAPSHOT-jar-with-dependencies.jar DESTINATION bin COMPONENT "jar")

add_custom_target(maven_populate_cache
  COMMAND ${MVN_PATH_COMMAND} ${MVN_EXECUTABLE} -q verify -Domnisci.release.version="${OMNISCI_JAR_RELEASE_VERSION}"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/java
  )

set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "${ADDITIONAL_MAKE_CLEAN_FILES}")

install(TARGETS initdb omnisci_server DESTINATION bin COMPONENT "exe")
install(FILES ${CMAKE_BINARY_DIR}/MAPD_GIT_HASH.txt DESTINATION "." COMPONENT "doc")
if(ENABLE_CUDA)
  install(FILES ${CMAKE_BINARY_DIR}/QueryEngine/cuda_mapd_rt.fatbin DESTINATION QueryEngine COMPONENT "exe")
endif()
install(FILES completion_hints.thrift DESTINATION "." COMPONENT "thrift")
install(FILES omnisci.thrift DESTINATION "." COMPONENT "thrift")
install(FILES common.thrift DESTINATION "." COMPONENT "thrift")
install(FILES QueryEngine/serialized_result_set.thrift DESTINATION "QueryEngine/" COMPONENT "thrift")
install(FILES QueryEngine/extension_functions.thrift DESTINATION "QueryEngine/" COMPONENT "thrift")

if(NOT PREFER_STATIC_LIBS AND NOT ENABLE_CONDA)
  install(FILES ${Boost_LIBRARIES} DESTINATION ThirdParty/lib)
endif()

if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
  set(EULA_FILE "${CMAKE_SOURCE_DIR}/EULA-EE.txt")
else()
  set(EULA_FILE "${CMAKE_SOURCE_DIR}/LICENSE.md")
endif()

if("${MAPD_EDITION_LOWER}" STREQUAL "os")
  install(FILES LICENSE.md DESTINATION "." COMPONENT "doc")
endif()

set(CPACK_RESOURCE_FILE_LICENSE "${EULA_FILE}")
install(FILES "${EULA_FILE}" DESTINATION "."  COMPONENT "doc")

install(PROGRAMS startomnisci DESTINATION ".")
install(PROGRAMS scripts/inneromnisci DESTINATION "scripts")
install(PROGRAMS insert_sample_data DESTINATION ".")

if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
  install(DIRECTORY docker/sds DESTINATION "docker")
endif()
install(FILES docker/README.md DESTINATION "docker")
if(ENABLE_CUDA OR ENABLE_RENDERING)
  install(FILES docker/Dockerfile.cuda RENAME Dockerfile DESTINATION "docker")
else()
  install(FILES docker/Dockerfile.cpu RENAME Dockerfile DESTINATION "docker")
endif()

exec_program(uname ARGS -m OUTPUT_VARIABLE MAPD_HOST_SYSTEM_ARCHITECTURE) # does not account for cross-compiling or Windows
set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${MAPD_EDITION_LOWER}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}-${MAPD_HOST_SYSTEM_ARCHITECTURE}${MAPD_PACKAGE_FLAGS}")

set(CPACK_GENERATOR "STGZ")

include(CPack)

if(DOXYGEN_FOUND)
  add_custom_target(sphinx
    COMMAND python3 -m venv sphinx-env
    COMMAND . sphinx-env/bin/activate && pip install -r requirements.txt
    COMMAND rm -rf build
    COMMAND . sphinx-env/bin/activate && make html SPHINXOPTS="-D version=${MAPD_VERSION_MAJOR}.${MAPD_VERSION_MINOR}.${MAPD_VERSION_PATCH}"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/docs
    )

  add_dependencies(sphinx doxygen)
endif(DOXYGEN_FOUND)
