cmake_minimum_required(VERSION 3.22) # For Ubuntu 22.04
project(fss
    VERSION 1.2.0
    DESCRIPTION "Function secret sharing (FSS) primitives including distributed point/comparison function (DPF/DCF)"
    HOMEPAGE_URL "https://github.com/myl7/fss"
    LANGUAGES CUDA CXX)

include(CTest)
include(GNUInstallDirs)

find_package(OpenMP)
if(NOT OpenMP_FOUND)
    message(WARNING "OpenMP not found. No multi-threading.")
endif()

include(FetchContent)
if(BUILD_TESTING)
    FetchContent_Declare(GTest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG 56efe3983185e3f37e43415d1afa97e3860f187f)
endif()

if(BUILD_TESTING)
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        # For Windows: Prevent overriding the parent project's compiler/linker settings
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_MakeAvailable(GTest)
    include(GoogleTest)
endif()

add_library(fss INTERFACE)
target_include_directories(fss INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(fss INTERFACE cxx_std_20 cuda_std_20)
set_target_properties(fss PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
if(OpenMP_FOUND)
    target_link_libraries(fss INTERFACE OpenMP::OpenMP_CXX)
    # Nvcc doesn't recognize OpenMP flags, so fix to manually pass them.
    target_compile_options(fss INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
endif()

if(BUILD_TESTING)
    find_package(OpenSSL REQUIRED)

    add_executable(group_test src/group_test.cu)
    target_link_libraries(group_test GTest::gtest_main fss)
    gtest_discover_tests(group_test)

    add_executable(dpf_bytes_test src/dpf_test.cu)
    target_link_libraries(dpf_bytes_test GTest::gtest_main fss OpenSSL::Crypto)
    target_compile_options(dpf_bytes_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-maes,-msse4.2>)
    gtest_discover_tests(dpf_bytes_test)

    add_executable(dcf_bytes_test src/dcf_test.cu)
    target_compile_definitions(dcf_bytes_test PRIVATE -DkBlocks=4)
    target_link_libraries(dcf_bytes_test GTest::gtest_main fss OpenSSL::Crypto)
    target_compile_options(dcf_bytes_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-maes,-msse4.2>)
    gtest_discover_tests(dcf_bytes_test)

    add_executable(dpf_u128_le_test src/dpf_test.cu)
    target_compile_definitions(dpf_u128_le_test PRIVATE TEST_GROUP_U128)
    target_link_libraries(dpf_u128_le_test GTest::gtest_main fss OpenSSL::Crypto)
    target_compile_options(dpf_u128_le_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-maes,-msse4.2>)
    gtest_discover_tests(dpf_u128_le_test)

    add_executable(dcf_u128_le_test src/dcf_test.cu)
    target_compile_definitions(dcf_u128_le_test PRIVATE -DkBlocks=4 TEST_GROUP_U128)
    target_link_libraries(dcf_u128_le_test GTest::gtest_main fss OpenSSL::Crypto)
    target_compile_options(dcf_u128_le_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-maes,-msse4.2>)
    gtest_discover_tests(dcf_u128_le_test)

    add_executable(vdpf_test src/vdpf_test.cu)
    target_link_libraries(vdpf_test GTest::gtest_main fss)
    gtest_discover_tests(vdpf_test)

    add_executable(eval_all_gpu_test src/eval_all_gpu_test.cu)
    target_link_libraries(eval_all_gpu_test GTest::gtest_main fss)
    gtest_discover_tests(eval_all_gpu_test)

    add_executable(point_eval_gpu_test src/point_eval_gpu_test.cu)
    target_link_libraries(point_eval_gpu_test GTest::gtest_main fss)
    gtest_discover_tests(point_eval_gpu_test)

    add_executable(half_tree_dpf_test src/half_tree_dpf_test.cu)
    target_link_libraries(half_tree_dpf_test GTest::gtest_main fss OpenSSL::Crypto)
    target_compile_options(half_tree_dpf_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-maes,-msse4.2>)
    gtest_discover_tests(half_tree_dpf_test)

    add_executable(grotto_dcf_test src/grotto_dcf_test.cu)
    target_link_libraries(grotto_dcf_test GTest::gtest_main fss)
    gtest_discover_tests(grotto_dcf_test)

    add_executable(vdmpf_test src/vdmpf_test.cu)
    target_link_libraries(vdmpf_test GTest::gtest_main fss OpenSSL::Crypto)
    gtest_discover_tests(vdmpf_test)

    add_executable(dmpf_test src/dmpf_test.cu)
    target_link_libraries(dmpf_test GTest::gtest_main fss OpenSSL::Crypto)
    gtest_discover_tests(dmpf_test)

    add_executable(chacha_feistel_test src/chacha_feistel_test.cu)
    target_link_libraries(chacha_feistel_test GTest::gtest_main fss)
    gtest_discover_tests(chacha_feistel_test)

    add_executable(dmpf_gpu_test src/dmpf_gpu_test.cu)
    target_link_libraries(dmpf_gpu_test GTest::gtest_main fss)
    gtest_discover_tests(dmpf_gpu_test)

    add_executable(vdmpf_gpu_test src/vdmpf_gpu_test.cu)
    target_link_libraries(vdmpf_gpu_test GTest::gtest_main fss)
    gtest_discover_tests(vdmpf_gpu_test)

    add_executable(packed_half_tree_dpf_test src/packed_half_tree_dpf_test.cu)
    target_link_libraries(packed_half_tree_dpf_test GTest::gtest_main fss OpenSSL::Crypto)
    target_compile_options(packed_half_tree_dpf_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-maes,-msse4.2>)
    gtest_discover_tests(packed_half_tree_dpf_test)

    add_executable(aes128_mmo_raw_test src/aes128_mmo_raw_test.cu)
    target_link_libraries(aes128_mmo_raw_test GTest::gtest_main fss)
    target_compile_options(aes128_mmo_raw_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-maes,-msse4.2>)
    gtest_discover_tests(aes128_mmo_raw_test)

    add_executable(aes128_feistel_raw_test src/aes128_feistel_raw_test.cu)
    target_link_libraries(aes128_feistel_raw_test GTest::gtest_main fss OpenSSL::Crypto)
    target_compile_options(aes128_feistel_raw_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-maes,-msse4.2>)
    gtest_discover_tests(aes128_feistel_raw_test)

    add_executable(point_eval_cpu_test src/point_eval_cpu_test.cu)
    target_link_libraries(point_eval_cpu_test GTest::gtest_main fss)
    target_compile_options(point_eval_cpu_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-maes,-msse4.2>)
    gtest_discover_tests(point_eval_cpu_test)

    add_executable(par_depth_test src/par_depth_test.cu)
    target_link_libraries(par_depth_test GTest::gtest_main fss)
    gtest_discover_tests(par_depth_test)

    add_executable(release_compat_test src/release_compat_test.cu)
    target_link_libraries(release_compat_test GTest::gtest_main fss OpenSSL::Crypto)
    target_compile_options(release_compat_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-msse4.2>)
    gtest_discover_tests(release_compat_test)

    add_executable(word_arithmetic_test src/word_arithmetic_test.cu)
    target_link_libraries(word_arithmetic_test GTest::gtest_main fss)
    gtest_discover_tests(word_arithmetic_test)

    add_executable(binding_prg_test test/binding_prg_test.cu)
    target_include_directories(binding_prg_test PRIVATE fss_crypto/_csrc)
    target_link_libraries(binding_prg_test PRIVATE fss OpenSSL::Crypto)
    add_test(NAME binding_prg_test COMMAND binding_prg_test)

    # Keep these released consumers unchanged so API repairs cannot update the
    # expected calls together with the library implementation.
    foreach(sample dpf_dcf_cpu half_tree_dpf_cpu grotto_dcf_cpu vdpf_cpu)
        add_executable(compat_v1_2_${sample} src/compat_v1_2/${sample}.cu)
        target_link_libraries(compat_v1_2_${sample} PRIVATE fss OpenSSL::Crypto)
        add_test(NAME compat_v1_2_${sample}
            COMMAND ${CMAKE_COMMAND}
                -DEXECUTABLE=$<TARGET_FILE:compat_v1_2_${sample}>
                -P ${CMAKE_CURRENT_SOURCE_DIR}/src/compat_v1_2/check_sample.cmake)
        set_tests_properties(compat_v1_2_${sample} PROPERTIES
            ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
    # The released VDMPF sample used t=8 despite its documented t>=30
    # precondition. Compile its API calls unchanged and use the valid-domain
    # legacy oracle for numerical compatibility.
    add_executable(compat_v1_2_vdmpf_cpu src/compat_v1_2/vdmpf_cpu.cu)
    target_link_libraries(compat_v1_2_vdmpf_cpu PRIVATE fss OpenSSL::Crypto)
    add_executable(compat_v1_2_oracle src/compat_v1_2/legacy_oracle.cu)
    target_link_libraries(compat_v1_2_oracle PRIVATE fss OpenSSL::Crypto)
    add_executable(compat_v1_2_dpf_dcf_gpu src/compat_v1_2/gpu_sample_runner.cu)
    target_link_libraries(compat_v1_2_dpf_dcf_gpu PRIVATE fss)
    add_test(NAME compat_v1_2_dpf_dcf_gpu
        COMMAND ${CMAKE_COMMAND}
            -DEXECUTABLE=$<TARGET_FILE:compat_v1_2_dpf_dcf_gpu>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/src/compat_v1_2/check_sample.cmake)
    set_tests_properties(compat_v1_2_dpf_dcf_gpu PROPERTIES
        SKIP_REGULAR_EXPRESSION "no CUDA device")
endif()

option(BUILD_BENCH "Build benchmarks" OFF)
if(BUILD_BENCH)
    FetchContent_Declare(benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.5)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)

    find_package(OpenSSL)
    if(OpenSSL_FOUND)
        add_executable(bench_cpu src/bench_cpu.cu)
        target_link_libraries(bench_cpu benchmark::benchmark_main fss OpenSSL::Crypto)
        target_compile_options(bench_cpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-maes,-msse4.2>)
    endif()

    add_executable(bench_gpu src/bench_gpu.cu)
    target_link_libraries(bench_gpu benchmark::benchmark_main fss)
endif()

install(TARGETS fss EXPORT fssTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT fssTargets
    FILE fssTargets.cmake
    NAMESPACE fss::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fss)

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/fssConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/fssConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fss)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/fssConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/fssConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/fssConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fss)
