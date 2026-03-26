cmake_minimum_required(VERSION 3.22)
project(fss_third_party LANGUAGES CUDA CXX)

add_library(fss INTERFACE)
target_include_directories(fss INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_compile_features(fss INTERFACE cxx_std_20 cuda_std_20)

include(FetchContent)
FetchContent_Declare(benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.5)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

add_executable(bench_aes128_soft bench_aes128_soft.cu)
target_include_directories(bench_aes128_soft PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bench_aes128_soft PRIVATE benchmark::benchmark_main fss)
