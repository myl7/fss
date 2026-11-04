cmake_minimum_required(VERSION 3.22) # For Ubuntu 22.04
project(fss
    VERSION 1.3.0
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

    add_executable(cuckoo_hash_test src/cuckoo_hash_test.cu)
    target_link_libraries(cuckoo_hash_test GTest::gtest_main fss OpenSSL::Crypto)
    gtest_discover_tests(cuckoo_hash_test)

    add_executable(group_test src/group_test.cu)
    target_link_libraries(group_test GTest::gtest_main fss)
    gtest_discover_tests(group_test)

    add_executable(dpf_test src/dpf_test.cu)
    target_link_libraries(dpf_test GTest::gtest_main fss OpenSSL::Crypto)
    target_compile_options(dpf_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-maes,-msse4.2>)
    gtest_discover_tests(dpf_test)

    add_executable(dcf_test src/dcf_test.cu)
    target_link_libraries(dcf_test GTest::gtest_main fss OpenSSL::Crypto)
    target_compile_options(dcf_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-maes,-msse4.2>)
    gtest_discover_tests(dcf_test)

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

    add_executable(half_tree_grotto_dcf_test src/half_tree_grotto_dcf_test.cu)
    target_link_libraries(half_tree_grotto_dcf_test GTest::gtest_main fss)
    gtest_discover_tests(half_tree_grotto_dcf_test)

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
    # The bucket-bound tests include the package-private binding validator.
    target_include_directories(dmpf_gpu_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/fss_crypto/_csrc)
    gtest_discover_tests(dmpf_gpu_test)

    add_executable(vdmpf_gpu_test src/vdmpf_gpu_test.cu)
    target_link_libraries(vdmpf_gpu_test GTest::gtest_main fss)
    gtest_discover_tests(vdmpf_gpu_test)

    add_executable(packed_half_tree_dpf_test src/packed_half_tree_dpf_test.cu)
    target_link_libraries(packed_half_tree_dpf_test GTest::gtest_main fss OpenSSL::Crypto)
    target_compile_options(packed_half_tree_dpf_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-maes,-msse4.2>)
    gtest_discover_tests(packed_half_tree_dpf_test)

    add_executable(aes128_mmo_raw_test src/aes128_mmo_raw_test.cu)
    target_link_libraries(aes128_mmo_raw_test GTest::gtest_main fss OpenSSL::Crypto)
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

    # Linker wrapping gives deterministic OpenSSL failures without an injection
    # hook in the public code. GNU ld and lld support --wrap, so this is Linux-only.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_executable(openssl_failure_cleanup_test src/openssl_failure_cleanup_test.cu)
        target_link_libraries(openssl_failure_cleanup_test GTest::gtest_main fss OpenSSL::Crypto)
        target_link_options(openssl_failure_cleanup_test PRIVATE
            -Wl,--wrap=EVP_CIPHER_CTX_new
            -Wl,--wrap=EVP_EncryptInit_ex2
            -Wl,--wrap=EVP_CIPHER_CTX_set_padding
            -Wl,--wrap=EVP_CIPHER_CTX_free)
        gtest_discover_tests(openssl_failure_cleanup_test)
    endif()

    add_executable(word_arithmetic_test src/word_arithmetic_test.cu)
    target_link_libraries(word_arithmetic_test GTest::gtest_main fss OpenSSL::Crypto)
    gtest_discover_tests(word_arithmetic_test)

    # Compile-only coverage for the wide Uint127 Group in CUDA host stubs.
    add_library(uint127_launch_test OBJECT src/uint127_launch_test.cu)
    target_link_libraries(uint127_launch_test PRIVATE fss)
    add_custom_target(uint127_launch_test_compile ALL DEPENDS uint127_launch_test)

    # Host-only backends must fail to link when device code reaches them. Each test compiles
    # one source with nvcc and passes only on the expected outcome. The control compiles the
    # same host code with no device use and must stay free of #20011-D and #20014-D.
    if(CMAKE_CUDA_COMPILER)
        get_target_property(host_only_ssl_dirs OpenSSL::Crypto INTERFACE_INCLUDE_DIRECTORIES)
        set(host_only_dirs "${CMAKE_CURRENT_SOURCE_DIR}/include")
        if(host_only_ssl_dirs)
            list(APPEND host_only_dirs ${host_only_ssl_dirs})
        endif()
        list(GET CMAKE_CUDA_ARCHITECTURES 0 host_only_arch)

        set(host_only_cases CONTROL PERMU COPY)
        foreach(case IN LISTS host_only_cases)
            string(TOLOWER ${case} case_lower)
            set(host_only_args -DNVCC=${CMAKE_CUDA_COMPILER}
                -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/src/host_only_device_use_test.cu
                -DOBJECT=${CMAKE_CURRENT_BINARY_DIR}/host_only_${case_lower}.o
                "-DINCLUDE_DIRS=${host_only_dirs}" -DARCH=${host_only_arch})
            if(case STREQUAL CONTROL)
                set(test_name host_only_device_use_control_builds)
            else()
                set(test_name host_only_device_${case_lower}_fails)
                list(APPEND host_only_args -DCASE=FSS_HOST_ONLY_DEVICE_${case}
                    "-DEXPECT=(Unresolved extern function|Undefined reference to)[^\n]*fss_host_only_Aes128Feistel_used_in_device_code")
            endif()
            add_test(NAME ${test_name}
                COMMAND ${CMAKE_COMMAND} ${host_only_args} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/HostOnlyDeviceUseCheck.cmake)
        endforeach()
    endif()

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
