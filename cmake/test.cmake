enable_language(CXX)
include(CTest)
enable_testing()
add_subdirectory(third_party/googletest)
include(GoogleTest)

add_executable(
  dpf_u128_le_test src/dpf_test.cc
  src/group/u128_le.c
  src/prg/aes_mmo.c src/prg/torchcsprng/aes.c
)
target_link_libraries(dpf_u128_le_test GTest::gtest_main dpf)
gtest_discover_tests(dpf_u128_le_test)

add_executable(
  dpf_bytes_test src/dpf_test.cc
  src/group/bytes.c
  src/prg/aes_mmo.c src/prg/torchcsprng/aes.c
)
target_link_libraries(dpf_bytes_test GTest::gtest_main dpf)
gtest_discover_tests(dpf_bytes_test)

add_executable(
  dcf_u128_le_test src/dcf_test.cc
  src/group/u128_le.c
  src/prg/aes_mmo.c src/prg/torchcsprng/aes.c
)
target_compile_definitions(dcf_u128_le_test PRIVATE -DBLOCK_NUM=4)
target_link_libraries(dcf_u128_le_test GTest::gtest_main dcf)
gtest_discover_tests(dcf_u128_le_test)

add_executable(
  dcf_bytes_test src/dcf_test.cc
  src/group/bytes.c
  src/prg/aes_mmo.c src/prg/torchcsprng/aes.c
)
target_compile_definitions(dcf_bytes_test PRIVATE -DBLOCK_NUM=4)
target_link_libraries(dcf_bytes_test GTest::gtest_main dcf)
gtest_discover_tests(dcf_bytes_test)

add_executable(
  cw_mac_bytes_test src/cw_mac_bytes_test.cc
  src/group/bytes.c
  src/prg/aes_mmo.c src/prg/torchcsprng/aes.c
)
target_compile_definitions(cw_mac_bytes_test PRIVATE -DBLOCK_NUM=4)
target_link_libraries(cw_mac_bytes_test GTest::gtest_main cw_mac_bytes dpf dcf)
gtest_discover_tests(cw_mac_bytes_test)
