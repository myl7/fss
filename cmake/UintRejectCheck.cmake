# SPDX-License-Identifier: Apache-2.0
#
# Compile src/uint_reject_compile_fail.cu with one case macro and require the
# compile to fail on the unsatisfied requires-clause of the group contract.
# Run as `cmake -DNVCC=... -DSOURCE=... -DOBJECT=... -DINCLUDE_DIRS=... -DARCH=... -DCASE=... -P`.
#
# This runs nvcc directly instead of nesting a build tool in the test, because
# the failing compile is meant to fail; see HostOnlyDeviceUseCheck.cmake.

set(flags -std=c++20 -O2 -DNDEBUG)
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
if(NOT log MATCHES "constraint")
    message(FATAL_ERROR "the compile of ${CASE} failed, but not on the group contract constraint")
endif()
