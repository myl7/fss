cmake_minimum_required(VERSION 3.22)
project(fss_samples LANGUAGES CUDA CXX)

# Header-only library for CUDA C++ API
add_library(fss INTERFACE)
target_include_directories(fss INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_compile_features(fss INTERFACE cxx_std_20 cuda_std_20)

# CUDA C++ API benchmarks (2^20 parallel gen/eval)
add_executable(dpf_bench dpf_bench.cu)
target_link_libraries(dpf_bench PRIVATE fss)

add_executable(dcf_bench dcf_bench.cu)
target_link_libraries(dcf_bench PRIVATE fss)
