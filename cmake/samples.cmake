add_executable(
  sample_dpf_u128_le_aes128_mmo samples/dpf_u128_le.c
  src/group/u128_le.c
  src/prg/aes128_mmo.c src/prg/torchcsprng/aes.c
)
target_link_libraries(sample_dpf_u128_le_aes128_mmo PRIVATE dpf)

if(BUILD_WITH_AES_NI)
  add_executable(
    sample_dpf_u128_le_aes128_mmo_ni samples/dpf_u128_le.c
    src/group/u128_le.c
    src/prg/aes128_mmo_ni.c
  )
  target_link_libraries(sample_dpf_u128_le_aes128_mmo_ni PRIVATE dpf)
  target_compile_options(sample_dpf_u128_le_aes128_mmo_ni PRIVATE -msse2 -maes)

  add_executable(
    sample_dpf_full_domain_u128_le_aes128_mmo_ni samples/dpf_full_domain_u128_le.c
    src/group/u128_le.c
    src/prg/aes128_mmo_ni.c
  )
  target_link_libraries(sample_dpf_full_domain_u128_le_aes128_mmo_ni PRIVATE dpf)
  target_compile_options(sample_dpf_full_domain_u128_le_aes128_mmo_ni PRIVATE -msse2 -maes)

  add_executable(
    sample_dpf_full_domain_u128_le_aes128_r9_mmo_ni samples/dpf_full_domain_u128_le.c
    src/group/u128_le.c
    src/prg/aes128_mmo_ni.c
  )
  target_compile_definitions(sample_dpf_full_domain_u128_le_aes128_r9_mmo_ni PUBLIC -DkRounds=9)
  target_link_libraries(sample_dpf_full_domain_u128_le_aes128_r9_mmo_ni PRIVATE dpf)
  target_compile_options(sample_dpf_full_domain_u128_le_aes128_r9_mmo_ni PRIVATE -msse2 -maes)
endif()

if(BUILD_WITH_OPENSSL)
  add_executable(
    sample_dpf_full_domain_u128_le_aes128_mmo_openssl samples/dpf_full_domain_u128_le.c
    src/group/u128_le.c
    src/prg/aes128_mmo_openssl.c
  )
  target_link_libraries(sample_dpf_full_domain_u128_le_aes128_mmo_openssl PRIVATE dpf OpenSSL::Crypto)
endif()

if(BUILD_WITH_CUDA)
  add_executable(
    sample_dpf_cu_u128_le_salsa12 samples/dpf_u128_le.cu
    src/group/u128_le.c
    src/prg/salsa.cu
  )
  target_compile_options(sample_dpf_cu_u128_le_salsa12 PUBLIC -DkRounds=12)
  set_target_properties(sample_dpf_cu_u128_le_salsa12 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(sample_dpf_cu_u128_le_salsa12 PRIVATE dpf_cu)

  add_executable(
    sample_dpf_cu_u128_le_chacha8 samples/dpf_u128_le.cu
    src/group/u128_le.c
    src/prg/chacha.cu
  )
  target_compile_options(sample_dpf_cu_u128_le_chacha8 PUBLIC -DkRounds=8)
  set_target_properties(sample_dpf_cu_u128_le_chacha8 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(sample_dpf_cu_u128_le_chacha8 PRIVATE dpf_cu)
endif()
