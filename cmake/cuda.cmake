enable_language(CUDA)

set_source_files_properties(src/dpf.c PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/group/i128.c PROPERTIES LANGUAGE CUDA)

add_library(dpf_cuda STATIC src/dpf.c)
set_target_properties(dpf_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(dpf_cuda PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_executable(
  sample_dpf_cuda_i128_salsa20 samples/dpf_i128.cu
  src/group/i128.c
  src/prg/salsa20.cu
)
set_target_properties(sample_dpf_cuda_i128_salsa20 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(sample_dpf_cuda_i128_salsa20 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-x cu>)
target_link_libraries(sample_dpf_cuda_i128_salsa20 PRIVATE dpf_cuda)
