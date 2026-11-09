cmake_minimum_required(VERSION 3.22) # For Ubuntu 22.04
project(fss
    VERSION 2.0.0
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

    add_executable(api_contract_test src/api_contract_test.cu)
    target_link_libraries(api_contract_test GTest::gtest_main fss)
    gtest_discover_tests(api_contract_test)

    # Behavior coverage of the deprecated Cw-form entries. The source builds an
    # empty suite when FSS_ENABLE_V1_COMPAT is 0, so both switch configurations
    # link the same target.
    add_executable(cw_compat_test src/cw_compat_test.cu)
    target_link_libraries(cw_compat_test GTest::gtest_main fss)
    gtest_discover_tests(cw_compat_test)

    # Behavior coverage of the deprecated aggregate-Key entries and legacy
    # names. The source builds an empty suite when FSS_ENABLE_V1_COMPAT is 0,
    # so both switch configurations link the same target.
    add_executable(key_compat_test src/key_compat_test.cu)
    target_link_libraries(key_compat_test GTest::gtest_main fss OpenSSL::Crypto)
    gtest_discover_tests(key_compat_test)

    # Every removed public Cw overload argument sequence must fail to compile.
    # One direct-nvcc case per sequence; see cmake/ApiRemovedCwCheck.cmake.
    if(CMAKE_CUDA_COMPILER)
        set(api_removed_cw_cases
            DPF_GEN DPF_EVAL DPF_EVALALL DPF_RANGE DPF_RANGE_GPU DPF_EVALALL_GPU DPF_BATCH
            DCF_GEN DCF_EVAL DCF_EVALALL DCF_RANGE DCF_RANGE_GPU DCF_EVALALL_GPU DCF_BATCH
            HT_GEN HT_EVAL HT_EVALALL HT_RANGE HT_RANGE_GPU HT_EVALALL_GPU HT_BATCH
            PHT_GEN PHT_EVAL PHT_EVALALL PHT_RANGE PHT_EVALALL_GPU PHT_BATCH
            GROTTO_GEN GROTTO_EVAL GROTTO_PREPROCESS GROTTO_PREPROCESSOWNED
            GROTTO_EVALALL GROTTO_PREPROCESS_GPU GROTTO_EVALALL_GPU
            HTG_GEN HTG_EVAL HTG_PREPROCESS HTG_PREPROCESSOWNED
            HTG_EVALALL HTG_PREPROCESS_GPU HTG_EVALALL_GPU
            VDPF_GEN VDPF_EVAL VDPF_EVALALL VDPF_RANGE VDPF_BATCH
            VDPF_BATCH_MERKLE VDPF_EVALALL_MERKLE VDPF_RANGE_MERKLE
            VDPF_EVALALL_MERKLE_GPU VDPF_RANGE_MERKLE_GPU VDPF_BATCH_MERKLE_GPU)
        foreach(case IN LISTS api_removed_cw_cases)
            string(TOLOWER ${case} case_lower)
            add_test(NAME api_removed_cw_${case_lower}_fails
                COMMAND ${CMAKE_COMMAND}
                -DNVCC=${CMAKE_CUDA_COMPILER}
                -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/src/api_removed_cw_compile_fail.cu
                -DOBJECT=${CMAKE_CURRENT_BINARY_DIR}/api_removed_cw_${case_lower}.o
                "-DINCLUDE_DIRS=${CMAKE_CURRENT_SOURCE_DIR}/include"
                -DARCH=${CMAKE_CUDA_ARCHITECTURES}
                -DCASE=FSS_API_REMOVED_CW_${case}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ApiRemovedCwCheck.cmake)
        endforeach()

        # The deprecated v1.3.0 aggregate-Key entries and legacy names must be
        # absent with the switch off, and every restored entry must produce a
        # real deprecation diagnostic with the switch on. One direct-nvcc case
        # per entry and check; see cmake/ApiRemovedCwCheck.cmake (which drives
        # the shared removal harness) and cmake/KeyCompatDeprecatedCheck.cmake.
        set(key_compat_cases
            DPF_KEY_GEN DPF_KEY_EVAL DPF_KEY_EVALALL DPF_KEY_KPARDEPTH
            DCF_KEY_GEN DCF_KEY_EVAL DCF_KEY_EVALALL
            HT_KEY_GEN HT_KEY_EVAL HT_KEY_EVALALL HT_KEY_KPARDEPTH
            PHT_KEY_GEN PHT_KEY_EVAL PHT_KEY_EVALALL PHT_KBETAMAX PHT_KTOPDEPTH
            GROTTO_KEY_GEN GROTTO_KEY_PREPROCESS GROTTO_KEY_EVAL_PARITYTREE
            GROTTO_KEY_EVAL_PREPROCESSED GROTTO_KEY_EVALALL
            VDPF_KEY_EVALRESULT VDPF_KEY_EVAL VDPF_KEY_EVALALL VDPF_KEY_EVALALL_MERKLE
            DMPF_KEY_BATCHEVAL DMPF_KEY_EVALALL
            DMPF_CONST_M DMPF_CONST_N DMPF_CONST_B_SIZE
            VDMPF_KEY_GEN VDMPF_KEY_BATCHEVAL_STD VDMPF_KEY_BATCHEVAL_EXTENT
            VDMPF_KEY_BATCHEVAL_MERKLE
            VDMPF_CONST_M VDMPF_CONST_N VDMPF_CONST_B_SIZE VDMPF_CONST_MERKLE_CHUNK)
        foreach(case IN LISTS key_compat_cases)
            string(TOLOWER ${case} case_lower)
            add_test(NAME api_removed_key_${case_lower}_fails
                COMMAND ${CMAKE_COMMAND}
                -DNVCC=${CMAKE_CUDA_COMPILER}
                -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/src/api_removed_key_compile_fail.cu
                -DOBJECT=${CMAKE_CURRENT_BINARY_DIR}/api_removed_key_${case_lower}.o
                "-DINCLUDE_DIRS=${CMAKE_CURRENT_SOURCE_DIR}/include"
                -DARCH=${CMAKE_CUDA_ARCHITECTURES}
                -DCASE=FSS_API_REMOVED_KEY_${case}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ApiRemovedCwCheck.cmake)
            add_test(NAME key_compat_deprecated_${case_lower}_warns
                COMMAND ${CMAKE_COMMAND}
                -DNVCC=${CMAKE_CUDA_COMPILER}
                -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/src/key_compat_deprecated_warn.cu
                -DOBJECT=${CMAKE_CURRENT_BINARY_DIR}/key_compat_deprecated_${case_lower}.o
                "-DINCLUDE_DIRS=${CMAKE_CURRENT_SOURCE_DIR}/include"
                -DARCH=${CMAKE_CUDA_ARCHITECTURES}
                -DCASE=FSS_KEY_COMPAT_WARN_${case}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/KeyCompatDeprecatedCheck.cmake)
        endforeach()
    endif()

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
            -Wl,--wrap=EVP_CIPHER_CTX_free
            -Wl,--wrap=EVP_EncryptUpdate
            -Wl,--wrap=EVP_Digest
            -Wl,--wrap=EVP_MD_CTX_new
            -Wl,--wrap=EVP_MD_CTX_free
            -Wl,--wrap=EVP_MD_CTX_reset
            -Wl,--wrap=EVP_DigestInit_ex
            -Wl,--wrap=EVP_DigestUpdate
            -Wl,--wrap=EVP_DigestFinal_ex)
        gtest_discover_tests(openssl_failure_cleanup_test)
    endif()

    add_executable(word_arithmetic_test src/word_arithmetic_test.cu)
    target_link_libraries(word_arithmetic_test GTest::gtest_main fss OpenSSL::Crypto)
    gtest_discover_tests(word_arithmetic_test)

    # Compile-only coverage for the Uint128 Group in CUDA host stubs.
    add_library(uint128_launch_test OBJECT src/uint128_launch_test.cu)
    target_link_libraries(uint128_launch_test PRIVATE fss)
    add_custom_target(uint128_launch_test_compile ALL DEPENDS uint128_launch_test)

    # Compile-only coverage for the deprecated wide-storage Uint compat Group
    # in CUDA host stubs. The source builds an empty file when
    # FSS_ENABLE_V1_COMPAT is 0, so the same target compiles in both switch
    # configurations.
    add_library(uint_compat_launch_test OBJECT src/uint_compat_launch_test.cu)
    target_link_libraries(uint_compat_launch_test PRIVATE fss)
    add_custom_target(uint_compat_launch_test_compile ALL DEPENDS uint_compat_launch_test)

    # Uint storage and Uint128 modulus contracts reject at compile time. Each
    # case compiles one rejected group type with nvcc and must fail on the
    # unsatisfied requires-clause.
    if(CMAKE_CUDA_COMPILER)
        set(uint_reject_cases
            UINT128_STORAGE UINT128_STORAGE_MOD UINT128_MOD_ABOVE_U64 MOD_ZERO MOD_ABOVE_2_127)
        foreach(case IN LISTS uint_reject_cases)
            string(TOLOWER ${case} case_lower)
            add_test(NAME uint_reject_${case_lower}_fails
                COMMAND ${CMAKE_COMMAND}
                -DNVCC=${CMAKE_CUDA_COMPILER}
                -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/src/uint_reject_compile_fail.cu
                -DOBJECT=${CMAKE_CURRENT_BINARY_DIR}/uint_reject_${case_lower}.o
                "-DINCLUDE_DIRS=${CMAKE_CURRENT_SOURCE_DIR}/include"
                -DARCH=${CMAKE_CUDA_ARCHITECTURES}
                -DCASE=FSS_UINT_REJECT_${case}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/UintRejectCheck.cmake)
        endforeach()
        # The same strict rejection with the compatibility entries compiled out.
        add_test(NAME uint_reject_uint128_small_mod_strict_fails
            COMMAND ${CMAKE_COMMAND}
            -DNVCC=${CMAKE_CUDA_COMPILER}
            -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/src/uint_reject_compile_fail.cu
            -DOBJECT=${CMAKE_CURRENT_BINARY_DIR}/uint_reject_uint128_small_mod_strict.o
            "-DINCLUDE_DIRS=${CMAKE_CURRENT_SOURCE_DIR}/include"
            -DARCH=${CMAKE_CUDA_ARCHITECTURES}
            -DCASE=FSS_UINT_REJECT_UINT128_SMALL_MOD_STRICT
            -DCOMPAT=0
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/UintRejectCheck.cmake)

        # The deprecated wide-storage Uint entry must compile with its
        # deprecation diagnostic on direct spellings, and current-only use
        # must stay free of it.
        add_test(NAME uint_compat_deprecated_warns
            COMMAND ${CMAKE_COMMAND}
            -DNVCC=${CMAKE_CUDA_COMPILER}
            -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/src/uint_compat_compile_check.cu
            -DOBJECT=${CMAKE_CURRENT_BINARY_DIR}/uint_compat_deprecated.o
            "-DINCLUDE_DIRS=${CMAKE_CURRENT_SOURCE_DIR}/include"
            -DARCH=${CMAKE_CUDA_ARCHITECTURES}
            -DCASE=FSS_UINT_COMPAT_CHECK_DEPRECATED
            -DREQUIRE_DEPRECATED=1
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/UintCompatCheck.cmake)
        add_test(NAME uint_compat_modern_no_warn
            COMMAND ${CMAKE_COMMAND}
            -DNVCC=${CMAKE_CUDA_COMPILER}
            -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/src/uint_compat_compile_check.cu
            -DOBJECT=${CMAKE_CURRENT_BINARY_DIR}/uint_compat_modern.o
            "-DINCLUDE_DIRS=${CMAKE_CURRENT_SOURCE_DIR}/include"
            -DARCH=${CMAKE_CUDA_ARCHITECTURES}
            -DCASE=FSS_UINT_COMPAT_CHECK_MODERN
            -DREQUIRE_DEPRECATED=0
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/UintCompatCheck.cmake)
    endif()

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
