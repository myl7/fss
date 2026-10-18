cmake_minimum_required(VERSION 3.22)
project(fss_released_consumer LANGUAGES CUDA CXX)

find_package(fss 1.2 REQUIRED)
find_package(OpenSSL REQUIRED)
add_executable(released_consumer ../dpf_dcf_cpu.cu)
target_link_libraries(released_consumer PRIVATE fss::fss OpenSSL::Crypto)
