add_executable(
  sample_dpf_u128_le_aes128_mmo samples/dpf_u128_le.c
  src/group/u128_le.c
  src/prg/aes128_mmo.c
)
target_link_libraries(sample_dpf_u128_le_aes128_mmo PRIVATE dpf OpenSSL::Crypto)

add_executable(
  sample_dpf_full_domain_u128_le_aes128_mmo samples/dpf_full_domain_u128_le.c
  src/group/u128_le.c
  src/prg/aes128_mmo.c
)
target_link_libraries(sample_dpf_full_domain_u128_le_aes128_mmo PRIVATE dpf OpenSSL::Crypto)

if(BUILD_WITH_CUDA)
  add_executable(
    sample_dpf_cu_u128_le_salsa12 samples/dpf_u128_le.cu
    src/group/u128_le.c
    src/prg/salsa.cu
  )
  target_compile_options(sample_dpf_cu_u128_le_salsa12 PUBLIC -DkRounds=12)
  set_target_properties(sample_dpf_cu_u128_le_salsa12 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(sample_dpf_cu_u128_le_salsa12 PRIVATE dpf_cu OpenSSL::Crypto)

  add_executable(
    sample_dpf_cu_u128_le_chacha8 samples/dpf_u128_le.cu
    src/group/u128_le.c
    src/prg/chacha.cu
  )
  target_compile_options(sample_dpf_cu_u128_le_chacha8 PUBLIC -DkRounds=8)
  set_target_properties(sample_dpf_cu_u128_le_chacha8 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(sample_dpf_cu_u128_le_chacha8 PRIVATE dpf_cu OpenSSL::Crypto)
endif()
