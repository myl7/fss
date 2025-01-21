# For Ubuntu 20.04 focal
cmake_minimum_required(VERSION 3.16)
project(
  fss
  VERSION 0.7.0
  DESCRIPTION "Function secret sharing (FSS) primitives including distributed point functions (DPF) and distributed comparison functions (DCF)"
  HOMEPAGE_URL "https://github.com/myl7/fss"
  LANGUAGES C CUDA
)

add_library(
  dpf STATIC
  src/dpf.c include/dpf.h
  include/dpf_api.h
  src/utils.h
)
target_include_directories(dpf PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(dpf PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_library(
  dpf_cuda STATIC
  src/dpf.cu include/dpf.h
  include/dpf_api.h
  src/utils.h
)
set_target_properties(dpf_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(dpf_cuda PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(dpf_cuda PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_executable(
  sample_dpf_i128_aes_mmo
  samples/dpf_i128.c
  samples/group/i128.c
  samples/prg/aes_mmo.c samples/prg/torchcsprng/aes.c
)
target_include_directories(sample_dpf_i128_aes_mmo PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(sample_dpf_i128_aes_mmo PRIVATE dpf)

add_executable(
  sample_dpf_i128_aes_mmo_ni
  samples/dpf_i128.c
  samples/group/i128.c
  samples/prg/aes_mmo_ni.c samples/prg/torchcsprng/aes.c
)
target_include_directories(sample_dpf_i128_aes_mmo_ni PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(sample_dpf_i128_aes_mmo_ni PRIVATE dpf)
target_compile_options(sample_dpf_i128_aes_mmo_ni PRIVATE -msse2 -maes)

add_executable(
  sample_dpf_cuda_i128_salsa20
  samples/dpf_i128.cu
  samples/group/i128.cu
  samples/prg/salsa20.cu
)
set_target_properties(sample_dpf_cuda_i128_salsa20 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(sample_dpf_cuda_i128_salsa20 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(sample_dpf_cuda_i128_salsa20 PRIVATE dpf_cuda)
