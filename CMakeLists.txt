cmake_minimum_required(VERSION 3.18) # For CUDA
project(
  fss
  VERSION 0.7.0
  DESCRIPTION "Function secret sharing (FSS) primitives including distributed point functions (DPF) and distributed comparison functions (DCF)"
  HOMEPAGE_URL "https://github.com/myl7/fss"
  LANGUAGES C
)

add_library(dpf STATIC src/dpf.c)
target_include_directories(dpf PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(dcf STATIC src/dcf.c)
target_include_directories(dcf PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_executable(
  sample_dpf_u128_le_aes_mmo samples/dpf_u128_le.c
  src/group/u128_le.c
  src/prg/aes_mmo.c src/prg/torchcsprng/aes.c
)
target_link_libraries(sample_dpf_u128_le_aes_mmo PRIVATE dpf)

add_executable(
  sample_dpf_u128_le_aes_mmo_ni samples/dpf_u128_le.c
  src/group/u128_le.c
  src/prg/aes_mmo_ni.c src/prg/torchcsprng/aes.c
)
target_link_libraries(sample_dpf_u128_le_aes_mmo_ni PRIVATE dpf)
target_compile_options(sample_dpf_u128_le_aes_mmo_ni PRIVATE -msse2 -maes)

option(WITH_TEST "Build with tests" ON)
if(WITH_TEST)
  include(cmake/test.cmake)
endif()

option(WITH_CUDA "Build with CUDA" ON)
if(WITH_CUDA)
  include(cmake/cuda.cmake)
endif()
