cmake_minimum_required(VERSION 3.22) # For Ubuntu 22.04
project(fss
    VERSION 0.8.0
    DESCRIPTION "Function secret sharing (FSS) primitives including distributed point functions (DPFs) and distributed comparison functions (DCFs)"
    HOMEPAGE_URL "https://github.com/myl7/fss"
    LANGUAGES C CXX)

include(CTest)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(OpenMP)
if(NOT OpenMP_FOUND)
    message(WARNING "OpenMP not found. No multi-threading.")
endif()

include(FetchContent)
if(BUILD_TESTING)
    FetchContent_Declare(GTest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG 56efe3983185e3f37e43415d1afa97e3860f187f)
endif()

if(BUILD_TESTING)
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        # For Windows: Prevent overriding the parent project's compiler/linker settings
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_MakeAvailable(GTest)
    include(GoogleTest)
endif()

add_library(dpf STATIC src/dpf.c)
target_include_directories(dpf PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
if(OpenMP_FOUND)
    target_link_libraries(dpf PUBLIC OpenMP::OpenMP_C)
endif()

add_library(dcf STATIC src/dcf.c)
target_include_directories(dcf PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
if(OpenMP_FOUND)
    target_link_libraries(dcf PUBLIC OpenMP::OpenMP_C)
endif()

# set_target_properties(dpf_cu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(BUILD_TESTING)
    find_package(OpenSSL REQUIRED)

    add_executable(dpf_bytes_test src/dpf_test.cc src/group/bytes.c src/prg/aes128_mmo.c)
    target_link_libraries(dpf_bytes_test GTest::gtest_main dpf OpenSSL::Crypto)
    gtest_discover_tests(dpf_bytes_test)

    add_executable(dcf_bytes_test src/dcf_test.cc src/group/bytes.c src/prg/aes128_mmo.c)
    target_compile_definitions(dcf_bytes_test PRIVATE -DkBlocks=4)
    target_link_libraries(dcf_bytes_test GTest::gtest_main dcf OpenSSL::Crypto)
    gtest_discover_tests(dcf_bytes_test)

    add_executable(dpf_u128_le_test src/dpf_test.cc src/group/u128_le.c src/prg/aes128_mmo.c)
    target_link_libraries(dpf_u128_le_test GTest::gtest_main dpf OpenSSL::Crypto)
    gtest_discover_tests(dpf_u128_le_test)

    add_executable(dcf_u128_le_test src/dcf_test.cc src/group/u128_le.c src/prg/aes128_mmo.c)
    target_compile_definitions(dcf_u128_le_test PRIVATE -DkBlocks=4)
    target_link_libraries(dcf_u128_le_test GTest::gtest_main dcf OpenSSL::Crypto)
    gtest_discover_tests(dcf_u128_le_test)
endif()
