cmake_minimum_required(VERSION 3.22)
project(bench_libfss LANGUAGES CXX)

if(NOT TARGET benchmark::benchmark_main)
    include(FetchContent)
    FetchContent_Declare(benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.5)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
endif()

find_package(OpenSSL REQUIRED)

# openssl-aes.h conflicts with OpenSSL 3.x; skip the system header.
# aesni_* are internal OpenSSL symbols removed in 3.x; alias to standard API.
add_library(libfss_static STATIC
    libfss/cpp/fss-common.cpp
    libfss/cpp/fss-client.cpp
    libfss/cpp/fss-server.cpp)
target_include_directories(libfss_static PUBLIC libfss/cpp)
target_compile_options(libfss_static PRIVATE -maes -msse4.2 -Wno-deprecated-declarations)
target_compile_definitions(libfss_static PUBLIC OPENSSL_AES_H)
target_compile_definitions(libfss_static PRIVATE
    "aesni_set_encrypt_key=AES_set_encrypt_key"
    "aesni_encrypt=AES_encrypt")
target_compile_features(libfss_static PRIVATE cxx_std_11)
target_link_libraries(libfss_static PUBLIC OpenSSL::Crypto gmp gmpxx pthread)

add_executable(bench_dpf_libfss bench.cc)
set_target_properties(bench_dpf_libfss PROPERTIES OUTPUT_NAME bench)
target_link_libraries(bench_dpf_libfss PRIVATE benchmark::benchmark_main libfss_static)
