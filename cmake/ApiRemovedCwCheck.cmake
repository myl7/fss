# SPDX-License-Identifier: Apache-2.0
#
# Compile src/api_removed_cw_compile_fail.cu with one case macro and require
# the compile to fail on the removed Cw overload argument sequences.
# Run as `cmake -DNVCC=... -DSOURCE=... -DOBJECT=... -DINCLUDE_DIRS=... -DARCH=... -DCASE=... -P`.
#
# The deprecated Cw-form entries come back under FSS_ENABLE_V1_COMPAT, so the
# probes pin the switch to 0: they check that the strict modern API alone
# rejects every removed argument sequence.
#
# This runs nvcc directly instead of nesting a build tool in the test, because
# the failing compile is meant to fail; see HostOnlyDeviceUseCheck.cmake.

set(flags -std=c++20 -O2 -DNDEBUG -DFSS_ENABLE_V1_COMPAT=0)
if(ARCH MATCHES "^([0-9]+)")
    list(APPEND flags "--generate-code=arch=compute_${CMAKE_MATCH_1},code=[compute_${CMAKE_MATCH_1},sm_${CMAKE_MATCH_1}]")
endif()
foreach(dir IN LISTS INCLUDE_DIRS)
    list(APPEND flags "-I${dir}")
endforeach()
list(APPEND flags "-D${CASE}")

execute_process(
    COMMAND ${NVCC} ${flags} -c ${SOURCE} -o ${OBJECT}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
set(log "${out}${err}")
message("${log}")

if(result EQUAL 0)
    message(FATAL_ERROR "expected the compile of ${CASE} to fail, but it succeeded")
endif()
if(NOT log MATCHES "no instance|argument list|no member|cannot be called|too few arguments|matches the call")
    message(FATAL_ERROR "the compile of ${CASE} failed, but not on overload resolution")
endif()
