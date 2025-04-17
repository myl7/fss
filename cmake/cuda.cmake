enable_language(CUDA)

set(_FSS_CUDA_SOURCES
  src/dpf.c
  src/dcf.c
  src/group/u128_le.c
  src/group/bytes.c
)
set_source_files_properties(${_FSS_CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
unset(_FSS_CUDA_SOURCES)

add_library(cudpf STATIC src/dpf.c)
set_target_properties(cudpf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(cudpf PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
