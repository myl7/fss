cmake_minimum_required(VERSION 3.22)
project(bench_ezpc LANGUAGES CUDA CXX)

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
find_package(CUDAToolkit REQUIRED)

# curand may not be in the CUDAToolkit sysroot (e.g. separate dpkg package
# or pip-installed nvidia-curand). Fall back to find_library/find_path.
if(NOT TARGET CUDA::curand)
    find_library(CURAND_LIB curand
        HINTS ${CUDAToolkit_LIBRARY_DIR} ENV LD_LIBRARY_PATH)
    if(CURAND_LIB)
        add_library(CUDA::curand SHARED IMPORTED)
        set_target_properties(CUDA::curand PROPERTIES IMPORTED_LOCATION "${CURAND_LIB}")
    else()
        message(FATAL_ERROR "could not find libcurand")
    endif()
endif()

# curand.h may be in a non-standard location (e.g. pip nvidia-curand package).
find_path(CURAND_INCLUDE_DIR curand.h
    HINTS ${CUDAToolkit_INCLUDE_DIRS} ENV CPATH)

set(EZPC_GPU_MPC "${CMAKE_CURRENT_SOURCE_DIR}/EzPC/GPU-MPC")
set(EZPC_SYTORCH "${EZPC_GPU_MPC}/ext/sytorch")
set(EZPC_SYTORCH_BUILD "${EZPC_SYTORCH}/build")

if(EXISTS "${EZPC_SYTORCH_BUILD}/libsytorch.a")
    add_executable(bench_dpf_ezpc
        bench.cu
        gpu_mem_bench.cu
        ${EZPC_GPU_MPC}/utils/gpu_file_utils.cpp)
    set_target_properties(bench_dpf_ezpc PROPERTIES OUTPUT_NAME bench)
    target_include_directories(bench_dpf_ezpc PRIVATE
        ${EZPC_GPU_MPC}
        ${EZPC_GPU_MPC}/utils
        ${EZPC_GPU_MPC}/ext/cutlass/include
        ${EZPC_GPU_MPC}/ext/cutlass/tools/util/include
        ${EZPC_SYTORCH}/include
        ${EZPC_SYTORCH}/ext/llama/include
        ${EZPC_SYTORCH}/ext/cryptoTools
        ${CURAND_INCLUDE_DIR})
    find_package(Eigen3 3.3 QUIET)
    if(Eigen3_FOUND)
        target_link_libraries(bench_dpf_ezpc PRIVATE Eigen3::Eigen)
        # sytorch includes <eigen3/Eigen/Dense>, so we need the parent of
        # the eigen3 include directory on the search path.
        get_target_property(_eigen_inc Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
        foreach(_d ${_eigen_inc})
            target_include_directories(bench_dpf_ezpc PRIVATE "${_d}/..")
        endforeach()
    endif()
    target_link_libraries(bench_dpf_ezpc PRIVATE
        benchmark::benchmark_main
        ${EZPC_SYTORCH_BUILD}/libsytorch.a
        ${EZPC_SYTORCH_BUILD}/ext/llama/libLLAMA.a
        ${EZPC_SYTORCH_BUILD}/ext/cryptoTools/libcryptoTools.a
        ${EZPC_SYTORCH_BUILD}/ext/bitpack/libbitpack.a
        CUDA::cudart CUDA::curand
        OpenSSL::Crypto pthread)
    target_compile_options(bench_dpf_ezpc PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-template-body>)
else()
    message(STATUS "EzPC sytorch libs not found, skipping bench. "
        "build EzPC first: cd EzPC/GPU-MPC && bash setup.sh")
endif()
