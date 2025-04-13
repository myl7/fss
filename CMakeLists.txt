cmake_minimum_required(VERSION 3.18) # For CUDA
project(
  fss
  VERSION 0.7.1
  DESCRIPTION "Function secret sharing (FSS) primitives including distributed point functions (DPF) and distributed comparison functions (DCF)"
  HOMEPAGE_URL "https://github.com/myl7/fss"
  LANGUAGES C
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(OpenMP REQUIRED)
find_package(Sodium REQUIRED)

add_library(dpf STATIC src/dpf.c)
target_include_directories(dpf PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(dpf PUBLIC OpenMP::OpenMP_C)

add_library(dcf STATIC src/dcf.c)
target_include_directories(dcf PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(cw_mac_bytes STATIC src/cw_mac_bytes.c)
target_include_directories(cw_mac_bytes PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(cw_mac_bytes PUBLIC sodium)

option(WITH_TEST "Build with tests" ON)
if(WITH_TEST)
  include(cmake/test.cmake)
endif()

option(WITH_CUDA "Build with CUDA" ON)
if(WITH_CUDA)
  include(cmake/cuda.cmake)
endif()

option(WITH_SAMPLES "Build with samples" ON)
if(WITH_SAMPLES)
  include(cmake/samples.cmake)
endif()
