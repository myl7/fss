# SPDX-License-Identifier: Apache-2.0
#
# Compile src/key_compat_deprecated_warn.cu with one case macro, the
# compatibility switch at its default 1, and deprecation diagnostics promoted
# to errors. The compile must fail on the deprecation itself, proving the
# entry carries a real [[deprecated]] attribute that legacy callers observe.
# Run as `cmake -DNVCC=... -DSOURCE=... -DOBJECT=... -DINCLUDE_DIRS=... -DARCH=... -DCASE=... -P`.

set(flags -std=c++20 -O1 -DFSS_ENABLE_V1_COMPAT=1 -Xcompiler=-Werror=deprecated-declarations)
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
    message(FATAL_ERROR "expected the compile of ${CASE} to warn, but it succeeded")
endif()
if(NOT log MATCHES "deprecated")
    message(FATAL_ERROR "the compile of ${CASE} failed, but not on a deprecation diagnostic")
endif()
