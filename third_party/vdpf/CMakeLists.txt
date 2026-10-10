cmake_minimum_required(VERSION 3.22)
project(bench_vdpf_servan LANGUAGES C CXX)

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

# Upstream is research-purpose code pinned as a submodule at vdpf/.
# -Wno-deprecated-declarations: OpenSSL 3.x deprecates the EVP entry points used.
add_library(vdpf_servan STATIC
    vdpf/src/common.c
    vdpf/src/dpf.c
    vdpf/src/mmo.c
    vdpf/src/sha256.c
    vdpf/src/vdpf.c)
target_include_directories(vdpf_servan PUBLIC vdpf/include)
target_compile_options(vdpf_servan PRIVATE -Wno-deprecated-declarations)
target_link_libraries(vdpf_servan PUBLIC OpenSSL::Crypto)

add_executable(bench_vdpf_servan bench.cc)
set_target_properties(bench_vdpf_servan PROPERTIES OUTPUT_NAME bench)
target_link_libraries(bench_vdpf_servan PRIVATE benchmark::benchmark_main vdpf_servan)

set(FSS_BENCH_DOMAIN_BITS 20 CACHE STRING "benchmark domain bit count")
target_compile_definitions(bench_vdpf_servan PRIVATE
    FSS_BENCH_DOMAIN_BITS=${FSS_BENCH_DOMAIN_BITS})
