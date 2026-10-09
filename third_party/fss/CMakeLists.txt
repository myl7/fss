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
