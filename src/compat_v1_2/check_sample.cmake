execute_process(COMMAND "${EXECUTABLE}"
    RESULT_VARIABLE status
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
    TIMEOUT 60)
if(status EQUAL 77)
    message("no CUDA device")
    return()
endif()
if(NOT status EQUAL 0)
    message(FATAL_ERROR "released sample failed: ${status}\n${output}\n${error}")
endif()
if(output MATCHES "(\\?[ \t]+NO|REJECT|mismatches[^\n]*: [1-9][0-9]*)")
    message(FATAL_ERROR "released sample reported incorrect results:\n${output}")
endif()
string(REGEX MATCHALL "Verification: [0-9]+/[0-9]+ correct" checks "${output}")
foreach(check IN LISTS checks)
    string(REGEX REPLACE "Verification: ([0-9]+)/([0-9]+) correct" "\\1;\\2" counts "${check}")
    list(GET counts 0 actual)
    list(GET counts 1 expected)
    if(NOT actual EQUAL expected)
        message(FATAL_ERROR "released GPU sample reported incorrect results: ${check}")
    endif()
endforeach()
