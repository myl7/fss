cmake_minimum_required(VERSION 3.22)
project(bench_gpu_dpf LANGUAGES CXX)

if(NOT TARGET benchmark::benchmark_main)
    include(FetchContent)
    FetchContent_Declare(benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.5)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
endif()

add_executable(bench_dpf_gpu_dpf bench.cc)
set_target_properties(bench_dpf_gpu_dpf PROPERTIES OUTPUT_NAME bench)
target_include_directories(bench_dpf_gpu_dpf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/GPU-DPF)
target_link_libraries(bench_dpf_gpu_dpf PRIVATE benchmark::benchmark_main)
target_compile_features(bench_dpf_gpu_dpf PRIVATE cxx_std_17)
target_compile_options(bench_dpf_gpu_dpf PRIVATE -O3)

set(FSS_BENCH_DOMAIN_BITS 20 CACHE STRING "benchmark domain bit count")
target_compile_definitions(bench_dpf_gpu_dpf PRIVATE
    FSS_BENCH_DOMAIN_BITS=${FSS_BENCH_DOMAIN_BITS})

option(FSS_BENCH_BUILD_GPU_FULL "build native CUDA full-domain benchmark" OFF)
if(FSS_BENCH_BUILD_GPU_FULL)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "cuda compiler required for GPU full-domain benchmark")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_executable(bench_dpf_gpu_dpf_full bench_full.cu)
    target_include_directories(bench_dpf_gpu_dpf_full PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/GPU-DPF)
    target_link_libraries(bench_dpf_gpu_dpf_full PRIVATE
        benchmark::benchmark_main CUDA::cudart)
    target_compile_features(bench_dpf_gpu_dpf_full PRIVATE cxx_std_17 cuda_std_17)
    target_compile_options(bench_dpf_gpu_dpf_full PRIVATE -O3)
    target_compile_definitions(bench_dpf_gpu_dpf_full PRIVATE
        FSS_BENCH_DOMAIN_BITS=${FSS_BENCH_DOMAIN_BITS})
endif()
