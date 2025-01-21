# For Ubuntu 20.04 focal
cmake_minimum_required(VERSION 3.16)
project(
  fss
  VERSION 0.7.0
  DESCRIPTION "Function secret sharing (FSS) primitives including distributed point functions (DPF) and distributed comparison functions (DCF)"
  HOMEPAGE_URL "https://github.com/myl7/fss"
  LANGUAGES C CUDA
)

add_library(dpf STATIC src/dpf.c)
target_include_directories(dpf PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(dpf_cuda STATIC src/dpf.cu)
set_target_properties(dpf_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(dpf_cuda PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_executable(
  sample_dpf_i128_aes_mmo
  samples/dpf_i128.c
  src/group/i128.c
  src/prg/aes_mmo.c src/prg/torchcsprng/aes.c
)
target_include_directories(sample_dpf_i128_aes_mmo PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(sample_dpf_i128_aes_mmo PRIVATE dpf)

add_executable(
  sample_dpf_i128_aes_mmo_ni
  samples/dpf_i128.c
  src/group/i128.c
  src/prg/aes_mmo_ni.c src/prg/torchcsprng/aes.c
)
target_include_directories(sample_dpf_i128_aes_mmo_ni PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(sample_dpf_i128_aes_mmo_ni PRIVATE dpf)
target_compile_options(sample_dpf_i128_aes_mmo_ni PRIVATE -msse2 -maes)

add_executable(
  sample_dpf_cuda_i128_salsa20
  samples/dpf_i128.cu
  src/group/i128.cu
  src/prg/salsa20.cu
)
set_target_properties(sample_dpf_cuda_i128_salsa20 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(sample_dpf_cuda_i128_salsa20 PRIVATE dpf_cuda)
