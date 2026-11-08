# SPDX-License-Identifier: Apache-2.0
#
# Compile src/uint_compat_compile_check.cu in one mode and require the compile
# to succeed, with the deprecation diagnostic present (DEPRECATED mode) or
# absent (MODERN mode, host warning promoted to an error). Run as
# `cmake -DNVCC=... -DSOURCE=... -DOBJECT=... -DINCLUDE_DIRS=... -DARCH=... -DCASE=... -DREQUIRE_DEPRECATED=1|0 -P`.
#
# This runs nvcc directly, like UintRejectCheck.cmake, so the checked outcome
# is the raw toolchain behavior.

set(flags -std=c++20 -O2 -DNDEBUG)
if(ARCH MATCHES "^([0-9]+)")
    list(APPEND flags "--generate-code=arch=compute_${CMAKE_MATCH_1},code=[compute_${CMAKE_MATCH_1},sm_${CMAKE_MATCH_1}]")
endif()
foreach(dir IN LISTS INCLUDE_DIRS)
    list(APPEND flags "-I${dir}")
endforeach()
list(APPEND flags "-D${CASE}")
if(REQUIRE_DEPRECATED)
    set(expect_deprecated TRUE)
else()
    set(expect_deprecated FALSE)
    list(APPEND flags "-Xcompiler=-Werror=deprecated-declarations")
endif()

execute_process(
    COMMAND ${NVCC} ${flags} -c ${SOURCE} -o ${OBJECT}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
set(log "${out}${err}")
message("${log}")

if(NOT result EQUAL 0)
    message(FATAL_ERROR "expected the compile of ${CASE} to succeed, but it failed with rc=${result}")
endif()
if(expect_deprecated AND NOT log MATCHES "deprecated")
    message(FATAL_ERROR "expected the compile of ${CASE} to report the deprecation diagnostic, but none appeared")
endif()
if(NOT expect_deprecated AND log MATCHES "deprecated")
    message(FATAL_ERROR "expected the compile of ${CASE} to stay free of deprecation diagnostics")
endif()
