cmake_minimum_required(VERSION 3.22)
project(bench_fss LANGUAGES CUDA CXX C)

if(NOT TARGET benchmark::benchmark_main)
    include(FetchContent)
    FetchContent_Declare(benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.5)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
endif()

set(FSS_BENCH_DOMAIN_BITS 20 CACHE STRING "input domain bit count")
set(FSS_BENCH_NUM_KEYS 1048576 CACHE STRING "independent point/gen keys")
set(FSS_BENCH_THREADS_PER_BLOCK 256 CACHE STRING "GPU threads per block")

find_package(OpenSSL REQUIRED)

if(NOT TARGET fss)
    add_library(fss INTERFACE)
    target_include_directories(fss INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
    target_compile_features(fss INTERFACE cxx_std_20 cuda_std_20)
endif()

add_executable(bench_fss bench.cu)
set_target_properties(bench_fss PROPERTIES OUTPUT_NAME bench)
target_link_libraries(bench_fss PRIVATE benchmark::benchmark_main fss OpenSSL::Crypto)
target_compile_options(bench_fss PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
target_link_options(bench_fss PRIVATE -fopenmp)

set(FSS_BENCH_CPU_PRG "openssl" CACHE STRING "CPU PRG: openssl or aes-ni")
set_property(CACHE FSS_BENCH_CPU_PRG PROPERTY STRINGS openssl aes-ni)
target_compile_definitions(bench_fss PRIVATE
    FSS_BENCH_DOMAIN_BITS=${FSS_BENCH_DOMAIN_BITS}
    FSS_BENCH_NUM_KEYS=${FSS_BENCH_NUM_KEYS}
    FSS_BENCH_THREADS_PER_BLOCK=${FSS_BENCH_THREADS_PER_BLOCK})
if(FSS_BENCH_CPU_PRG STREQUAL "aes-ni")
    target_compile_definitions(bench_fss PRIVATE FSS_BENCH_AES_NI)
    target_compile_options(bench_fss PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-maes,-msse4.1>)
elseif(NOT FSS_BENCH_CPU_PRG STREQUAL "openssl")
    message(FATAL_ERROR "unsupported CPU PRG: ${FSS_BENCH_CPU_PRG}")
endif()
