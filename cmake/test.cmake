enable_language(CXX)
include(CTest)
enable_testing()
add_subdirectory(test/googletest)
include(GoogleTest)

add_executable(
  dpf_test src/dpf_test.cc
  src/group/i128.c
  src/prg/aes_mmo.c src/prg/torchcsprng/aes.c
)
target_link_libraries(dpf_test GTest::gtest_main dpf)
gtest_discover_tests(dpf_test)
