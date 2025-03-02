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

add_executable(
  sample_dpf_i128_aes_mmo samples/dpf_i128.c
  src/group/i128.c
  src/prg/aes_mmo.c src/prg/torchcsprng/aes.c
)
target_link_libraries(sample_dpf_i128_aes_mmo PRIVATE dpf)

add_executable(
  sample_dpf_i128_aes_mmo_ni samples/dpf_i128.c
  src/group/i128.c
  src/prg/aes_mmo_ni.c src/prg/torchcsprng/aes.c
)
target_link_libraries(sample_dpf_i128_aes_mmo_ni PRIVATE dpf)
target_compile_options(sample_dpf_i128_aes_mmo_ni PRIVATE -msse2 -maes)

option(WITH_TEST "Build with tests" OFF)
if(WITH_TEST)
  include(test.cmake)
endif()

option(WITH_CUDA "Build with CUDA" OFF)
if(WITH_CUDA)
  include(cuda.cmake)
endif()
