cmake_minimum_required(VERSION 3.22)
project(fss_third_party LANGUAGES CUDA CXX C)

# Pre-fetch Google Benchmark once; subdirectories skip via NOT TARGET guard.
include(FetchContent)
FetchContent_Declare(benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.5)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

find_package(OpenSSL REQUIRED)

add_subdirectory(fss)
add_subdirectory(libfss)
add_subdirectory(GPU-DPF)
add_subdirectory(torchcsprng)
add_subdirectory(EzPC)
add_subdirectory(fss-v0.7.0)
