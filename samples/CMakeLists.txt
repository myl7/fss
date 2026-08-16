cmake_minimum_required(VERSION 3.22)
project(fss_samples LANGUAGES CUDA CXX)

add_library(fss INTERFACE)
target_include_directories(fss INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_compile_features(fss INTERFACE cxx_std_20 cuda_std_20)

find_package(OpenSSL REQUIRED)

# EvalAll uses OpenMP on the host side; nvcc needs the flags passed through
find_package(OpenMP)
if(OpenMP_FOUND)
    target_link_libraries(fss INTERFACE OpenMP::OpenMP_CXX)
    target_compile_options(fss INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
endif()

add_executable(dpf_dcf_cpu dpf_dcf_cpu.cu)
target_link_libraries(dpf_dcf_cpu PRIVATE fss OpenSSL::Crypto)

add_executable(dpf_dcf_gpu dpf_dcf_gpu.cu)
target_link_libraries(dpf_dcf_gpu PRIVATE fss)

add_executable(half_tree_dpf_cpu half_tree_dpf_cpu.cu)
target_link_libraries(half_tree_dpf_cpu PRIVATE fss OpenSSL::Crypto)

add_executable(grotto_dcf_cpu grotto_dcf_cpu.cu)
target_link_libraries(grotto_dcf_cpu PRIVATE fss OpenSSL::Crypto)

add_executable(vdpf_cpu vdpf_cpu.cu)
target_link_libraries(vdpf_cpu PRIVATE fss OpenSSL::Crypto)

add_executable(vdmpf_cpu vdmpf_cpu.cu)
target_link_libraries(vdmpf_cpu PRIVATE fss OpenSSL::Crypto)
