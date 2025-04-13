enable_language(CUDA)

set_source_files_properties(src/dpf.c PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/dcf.c PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/group/u128_le.c PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/group/bytes.c PROPERTIES LANGUAGE CUDA)

add_library(dpf_cuda STATIC src/dpf.c)
set_target_properties(dpf_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(dpf_cuda PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
