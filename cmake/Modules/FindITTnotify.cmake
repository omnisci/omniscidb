# Find ITTnotify ittnotify library
# Defines:
#   ITTnotify_FOUND
#   ITTnotify_INCLUDE_DIRS
#   ITTnotify_LIBRARIES
set(dirs
  "/opt/intel/vtune_profiler/"
  "$ENV{VTUNE_PROFILER_2020_DIR}/"
  "$ENV{VTUNE_PROFILER_2019_DIR}/"
  "$ENV{VTUNE_AMPLIFIER_XE_2013_DIR}/"
  "$ENV{VTUNE_AMPLIFIER_XE_2011_DIR}/"
  "$ENV{CONDA_PREFIX}/"
  )
find_path(ITTnotify_INCLUDE_DIRS ittnotify.h
    PATHS ${dirs}
    PATH_SUFFIXES include)
if (CMAKE_SIZEOF_VOID_P MATCHES "8")
  set(vtune_lib_dir lib64)
else()
  set(vtune_lib_dir lib32)
endif()
find_library(ITTnotify_LIBRARIES ittnotify
    HINTS "${ITTnotify_INCLUDE_DIRS}/.."
    PATHS ${dirs}
    PATH_SUFFIXES ${vtune_lib_dir})
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    ITTnotify DEFAULT_MSG ITTnotify_LIBRARIES ITTnotify_INCLUDE_DIRS)
