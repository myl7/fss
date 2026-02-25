cmake_minimum_required(VERSION 3.22)
project(fss_samples LANGUAGES CUDA CXX)

add_library(fss INTERFACE)
target_include_directories(fss INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_compile_features(fss INTERFACE cxx_std_20 cuda_std_20)

find_package(OpenSSL REQUIRED)

add_executable(dpf_dcf_cpu dpf_dcf_cpu.cu)
target_link_libraries(dpf_dcf_cpu PRIVATE fss OpenSSL::Crypto)

add_executable(dpf_dcf_gpu dpf_dcf_gpu.cu)
target_link_libraries(dpf_dcf_gpu PRIVATE fss)
