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

add_executable(
  sample_dpf_full_domain_u128_le_aes_mmo_ni samples/dpf_full_domain_u128_le.c
  src/group/u128_le.c
  src/prg/aes_mmo_ni.c src/prg/torchcsprng/aes.c
)
target_link_libraries(sample_dpf_full_domain_u128_le_aes_mmo_ni PRIVATE dpf)
target_compile_options(sample_dpf_full_domain_u128_le_aes_mmo_ni PRIVATE -msse2 -maes)

if(WITH_CUDA)
  add_executable(
    sample_dpf_cuda_u128_le_salsa20 samples/dpf_u128_le.cu
    src/group/u128_le.c
    src/prg/salsa20.cu
  )
  set_target_properties(sample_dpf_cuda_u128_le_salsa20 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_options(sample_dpf_cuda_u128_le_salsa20 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-x cu>)
  target_link_libraries(sample_dpf_cuda_u128_le_salsa20 PRIVATE dpf_cuda)
endif()
