cmake_minimum_required(VERSION 3.22)
project(bench_dpfplusplus LANGUAGES CXX)

# Upstream is fetched at bench time into the gitignored build directory (never
# committed; see doc/bench-dpfplusplus.md). The wrapper consumes only the
# headers: no add_subdirectory on the fetched tree.
if(NOT DEFINED DPFPP_INCLUDE_DIR)
    message(FATAL_ERROR
        "set DPFPP_INCLUDE_DIR to the include/ tree fetched by "
        "'bash third_party/dpfplusplus-bench/fetch.sh <builddir>'")
endif()
get_filename_component(DPFPP_INCLUDE "${DPFPP_INCLUDE_DIR}" ABSOLUTE)
if(NOT EXISTS "${DPFPP_INCLUDE}/dpf.hpp")
    message(FATAL_ERROR
        "${DPFPP_INCLUDE}/dpf.hpp not found; run "
        "third_party/dpfplusplus-bench/fetch.sh first")
endif()

# The upstream headers use SSE/AVX/AES intrinsics unconditionally.
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    message(FATAL_ERROR "dpf++ headers require an x86_64 host")
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

add_executable(bench_dpfplusplus bench.cc)
set_target_properties(bench_dpfplusplus PROPERTIES OUTPUT_NAME bench)
target_compile_features(bench_dpfplusplus PRIVATE cxx_std_17)
target_include_directories(bench_dpfplusplus PRIVATE "${DPFPP_INCLUDE}")
# The header-only library is compiled into this translation unit. -O2 is
# appended after the build-type flags, so it also pins -O2 for Release builds.
target_compile_options(bench_dpfplusplus PRIVATE -maes -mavx2 -O2)
target_link_libraries(bench_dpfplusplus PRIVATE benchmark::benchmark_main)
