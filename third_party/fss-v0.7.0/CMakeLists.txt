cmake_minimum_required(VERSION 3.22)
project(bench_fss_v070 LANGUAGES CXX C)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit)
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

set(FSS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/fss-v0.7.0)

# --- CPU object libraries (compiled as C, BLOCK_NUM=4 for DCF compat) ---

add_library(fss070_prg OBJECT ${FSS_SRC}/src/prg/aes_mmo_ni.c)
target_include_directories(fss070_prg PUBLIC ${FSS_SRC}/include ${FSS_SRC}/src/prg)
target_compile_definitions(fss070_prg PRIVATE BLOCK_NUM=4)
target_compile_options(fss070_prg PRIVATE -msse2 -maes)

add_library(fss070_dpf OBJECT ${FSS_SRC}/src/dpf.c)
target_include_directories(fss070_dpf PUBLIC ${FSS_SRC}/include)
target_compile_definitions(fss070_dpf PRIVATE BLOCK_NUM=4)

add_library(fss070_dcf OBJECT ${FSS_SRC}/src/dcf.c)
target_include_directories(fss070_dcf PUBLIC ${FSS_SRC}/include)
target_compile_definitions(fss070_dcf PRIVATE BLOCK_NUM=4)

add_library(fss070_group_uint OBJECT ${FSS_SRC}/src/group/u128_le.c)
target_include_directories(fss070_group_uint PUBLIC ${FSS_SRC}/include)

add_library(fss070_group_bytes OBJECT ${FSS_SRC}/src/group/bytes.c)
target_include_directories(fss070_group_bytes PUBLIC ${FSS_SRC}/include)

# --- CPU bench targets (one per group type, both DPF and DCF) ---

function(add_fss070_cpu_bench name group_type)
    add_executable(${name} bench.cc)
    target_include_directories(${name} PRIVATE ${FSS_SRC}/include)
    target_compile_definitions(${name} PRIVATE BLOCK_NUM=4)
    target_compile_definitions(${name} PRIVATE
        "BENCH_NAME_PREFIX=\"fss-v0.7.0/CPU/${group_type}\"")
    target_link_libraries(${name} PRIVATE
        $<TARGET_OBJECTS:fss070_dpf>
        $<TARGET_OBJECTS:fss070_dcf>
        $<TARGET_OBJECTS:fss070_prg>)
    if(group_type STREQUAL "uint")
        target_link_libraries(${name} PRIVATE $<TARGET_OBJECTS:fss070_group_uint>)
    else()
        target_link_libraries(${name} PRIVATE $<TARGET_OBJECTS:fss070_group_bytes>)
    endif()
    target_link_libraries(${name} PRIVATE benchmark::benchmark_main)
endfunction()

add_fss070_cpu_bench(bench_cpu_uint uint)
add_fss070_cpu_bench(bench_cpu_bytes bytes)

# --- GPU bench targets (Salsa20 PRG, BLOCK_NUM=2 for DCF compat) ---
# To compile .c sources as CUDA without directory-scoped LANGUAGE leaking to CPU
# targets, we copy them to .cu in the build directory.

if(CMAKE_CUDA_COMPILER AND CUDAToolkit_FOUND)
    set(GPU_C_SOURCES
        dpf:${FSS_SRC}/src/dpf.c
        dcf:${FSS_SRC}/src/dcf.c
        u128_le:${FSS_SRC}/src/group/u128_le.c
        bytes:${FSS_SRC}/src/group/bytes.c)

    foreach(entry ${GPU_C_SOURCES})
        string(REPLACE ":" ";" parts ${entry})
        list(GET parts 0 name)
        list(GET parts 1 src)
        configure_file(${src} ${CMAKE_CURRENT_BINARY_DIR}/gpu_${name}.cu COPYONLY)
    endforeach()
    # Copy utils.h so ../utils.h resolves correctly from the build dir
    configure_file(${FSS_SRC}/src/utils.h ${CMAKE_CURRENT_SOURCE_DIR}/utils.h COPYONLY)

    add_library(fss070_salsa20 OBJECT ${FSS_SRC}/src/prg/salsa20.cu)
    target_include_directories(fss070_salsa20 PUBLIC ${FSS_SRC}/include)
    target_compile_definitions(fss070_salsa20 PRIVATE BLOCK_NUM=2)
    set_target_properties(fss070_salsa20 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    add_library(fss070_dpf_cuda OBJECT ${CMAKE_CURRENT_BINARY_DIR}/gpu_dpf.cu)
    target_include_directories(fss070_dpf_cuda PUBLIC ${FSS_SRC}/include ${FSS_SRC}/src)
    target_compile_definitions(fss070_dpf_cuda PRIVATE BLOCK_NUM=2)
    set_target_properties(fss070_dpf_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    add_library(fss070_dcf_cuda OBJECT ${CMAKE_CURRENT_BINARY_DIR}/gpu_dcf.cu)
    target_include_directories(fss070_dcf_cuda PUBLIC ${FSS_SRC}/include ${FSS_SRC}/src)
    target_compile_definitions(fss070_dcf_cuda PRIVATE BLOCK_NUM=2)
    set_target_properties(fss070_dcf_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    add_library(fss070_group_uint_cuda OBJECT ${CMAKE_CURRENT_BINARY_DIR}/gpu_u128_le.cu)
    target_include_directories(fss070_group_uint_cuda PUBLIC ${FSS_SRC}/include ${FSS_SRC}/src)
    set_target_properties(fss070_group_uint_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    add_library(fss070_group_bytes_cuda OBJECT ${CMAKE_CURRENT_BINARY_DIR}/gpu_bytes.cu)
    target_include_directories(fss070_group_bytes_cuda PUBLIC ${FSS_SRC}/include ${FSS_SRC}/src)
    set_target_properties(fss070_group_bytes_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    function(add_fss070_gpu_bench name group_type)
        add_executable(${name} bench_gpu.cu)
        target_include_directories(${name} PRIVATE ${FSS_SRC}/include)
        target_compile_definitions(${name} PRIVATE BLOCK_NUM=2)
        target_compile_definitions(${name} PRIVATE
            "BENCH_NAME_PREFIX=\"fss-v0.7.0/GPU/${group_type}\"")
        set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_link_libraries(${name} PRIVATE
            $<TARGET_OBJECTS:fss070_dpf_cuda>
            $<TARGET_OBJECTS:fss070_dcf_cuda>
            $<TARGET_OBJECTS:fss070_salsa20>)
        if(group_type STREQUAL "uint")
            target_link_libraries(${name} PRIVATE $<TARGET_OBJECTS:fss070_group_uint_cuda>)
        else()
            target_link_libraries(${name} PRIVATE $<TARGET_OBJECTS:fss070_group_bytes_cuda>)
        endif()
        target_link_libraries(${name} PRIVATE benchmark::benchmark_main CUDA::cudart)
    endfunction()

    add_fss070_gpu_bench(bench_gpu_uint uint)
    add_fss070_gpu_bench(bench_gpu_bytes bytes)
endif()
