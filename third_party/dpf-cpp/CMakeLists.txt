cmake_minimum_required(VERSION 3.22)
project(bench_dpf_cpp LANGUAGES C CXX)

# The PRG and the packed-bit expansion are AES-NI/PCLMUL/AVX2 intrinsic code.
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    message(FATAL_ERROR "dpf-cpp requires an x86-64 host with AES-NI, PCLMUL, and AVX2")
endif()

if(NOT TARGET benchmark::benchmark_main)
    include(FetchContent)
    FetchContent_Declare(benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.5)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
endif()

# Upstream is research-purpose code pinned as a submodule at dpf-cpp/. Compile
# its plain sources directly instead of importing its CMake targets, skipping
# its own main/bench/test drivers and hashdatastore.cpp (PIR-answer experiments
# only, and the sole source whose headers include <malloc.h>, which would need
# a guard fix on BSDs). gsl-lite is vendored at dpf-cpp/gsl.
file(GLOB DPF_CPP_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/dpf-cpp/*.cpp)
list(FILTER DPF_CPP_SOURCES EXCLUDE REGEX "/(main|bench|test|hashdatastore)\\.cpp$")

add_library(dpf_cpp STATIC ${DPF_CPP_SOURCES})
target_include_directories(dpf_cpp PUBLIC dpf-cpp)
target_compile_options(dpf_cpp PRIVATE -maes -msse4.1 -mavx2 -mpclmul
                                       -Wno-ignored-attributes -Wno-strict-overflow)
set_target_properties(dpf_cpp PROPERTIES CXX_STANDARD 14)

add_executable(bench_dpf_cpp bench.cc)
set_target_properties(bench_dpf_cpp PROPERTIES OUTPUT_NAME bench CXX_STANDARD 14)
target_compile_options(bench_dpf_cpp PRIVATE -maes -msse4.1 -mavx2 -mpclmul)
target_link_libraries(bench_dpf_cpp PRIVATE benchmark::benchmark_main dpf_cpp)
