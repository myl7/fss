cmake_minimum_required(VERSION 3.18) # For CUDA
project(fss_samples LANGUAGES C)

find_package(OpenSSL REQUIRED)

set(BUILD_TESTING OFF CACHE BOOL "")
set(BUILD_WITH_CUDA OFF CACHE BOOL "")
add_subdirectory(.. "${CMAKE_CURRENT_BINARY_DIR}/fss")

add_executable(
  sample_dpf_full_domain_bytes dpf_full_domain_bytes.c
  ../src/group/bytes.c
  ../src/prg/aes128_mmo.c
)
target_link_libraries(sample_dpf_full_domain_bytes PRIVATE dpf OpenSSL::Crypto)

add_executable(
  sample_dpf_bytes dpf_bytes.c
  ../src/group/bytes.c
  ../src/prg/aes128_mmo.c
)
target_link_libraries(sample_dpf_bytes PRIVATE dpf OpenSSL::Crypto)

add_executable(
  sample_dcf_full_domain_bytes dcf_full_domain_bytes.c
  ../src/group/bytes.c
  ../src/prg/aes128_mmo.c
)
target_compile_definitions(sample_dcf_full_domain_bytes PRIVATE -DkBlocks=4)
target_link_libraries(sample_dcf_full_domain_bytes PRIVATE dcf OpenSSL::Crypto)

add_executable(
  sample_dcf_bytes dcf_bytes.c
  ../src/group/bytes.c
  ../src/prg/aes128_mmo.c
)
target_compile_definitions(sample_dcf_bytes PRIVATE -DkBlocks=4)
target_link_libraries(sample_dcf_bytes PRIVATE dcf OpenSSL::Crypto)
