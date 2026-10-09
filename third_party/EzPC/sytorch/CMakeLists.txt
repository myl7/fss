cmake_minimum_required(VERSION 3.22)
project(ezpc_benchmark_dependencies LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(EZPC_SYTORCH_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../EzPC/GPU-MPC/ext/sytorch"
    CACHE PATH "pinned sytorch source directory")
if(NOT EXISTS "${EZPC_SYTORCH_SOURCE}/src/sytorch/random.cpp")
    message(FATAL_ERROR "sytorch sources are missing; initialize EzPC recursively")
endif()
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# These are the pinned upstream libraries. Binary directories keep all generated
# files out of the source tree, including bitpack's configured test target.
add_subdirectory("${EZPC_SYTORCH_SOURCE}/ext/cryptoTools" ext/cryptoTools)
add_subdirectory("${EZPC_SYTORCH_SOURCE}/ext/bitpack" ext/bitpack)
add_subdirectory("${EZPC_SYTORCH_SOURCE}/ext/llama" ext/llama)

# GPU-MPC uses sytorch's authentic random globals and rand_float only. Neural
# network backends and SCI/SEAL are not dependencies of this benchmark.
add_library(sytorch STATIC "${EZPC_SYTORCH_SOURCE}/src/sytorch/random.cpp")
target_include_directories(sytorch PUBLIC "${EZPC_SYTORCH_SOURCE}/include")
target_compile_options(sytorch PRIVATE
    -Wno-write-strings -Wno-unused-result -maes -Wno-ignored-attributes
    -march=native -Wno-deprecated-declarations)
target_link_libraries(sytorch PUBLIC Eigen3::Eigen Threads::Threads
    OpenMP::OpenMP_CXX LLAMA cryptoTools bitpack)
