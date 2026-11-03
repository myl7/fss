# SPDX-License-Identifier: Apache-2.0
#
# Compile src/host_only_device_use_test.cu with one case macro and check the outcome.
# Run as `cmake -DNVCC=... -DSOURCE=... -DINCLUDE_DIRS=... -DARCH=... [-DCASE=...] [-DEXPECT=<regex>] -P`.
# With EXPECT, the compile must fail and its output must match the regex.
# Without EXPECT, the compile must succeed without nvcc #20011-D or #20014-D.
#
# This runs nvcc directly instead of nesting a build tool in the test, because the failing
# compile is meant to fail and a nested `cmake --build` of it was seen to take the whole
# container down with SIGKILL when run as a child of ctest.

set(flags -std=c++20 -O2 -DNDEBUG -Xcompiler=-maes,-msse4.2)
if(ARCH MATCHES "^([0-9]+)")
    list(APPEND flags "--generate-code=arch=compute_${CMAKE_MATCH_1},code=[compute_${CMAKE_MATCH_1},sm_${CMAKE_MATCH_1}]")
endif()
foreach(dir IN LISTS INCLUDE_DIRS)
    list(APPEND flags "-I${dir}")
endforeach()
if(CASE)
    list(APPEND flags "-D${CASE}")
endif()

execute_process(
    COMMAND ${NVCC} ${flags} -c ${SOURCE} -o ${OBJECT}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
set(log "${out}${err}")
message("${log}")

if(EXPECT)
    if(result EQUAL 0)
        message(FATAL_ERROR "expected the compile to fail, but it succeeded")
    endif()
    if(NOT log MATCHES "${EXPECT}")
        message(FATAL_ERROR "the compile failed, but its output does not match: ${EXPECT}")
    endif()
else()
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "the compile failed with ${result}")
    endif()
    if(log MATCHES "#20011-D|#20014-D")
        message(FATAL_ERROR "the compile emitted nvcc #20011-D or #20014-D")
    endif()
endif()
