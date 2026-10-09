cmake_minimum_required(VERSION 3.22)
project(fss_third_party LANGUAGES CXX C)

# Each wrapper also supports a standalone configure in its own build directory.
set(FSS_THIRD_PARTY_LIBRARIES "fss;libfss;GPU-DPF;torchcsprng;EzPC;fss-v0.7.0"
    CACHE STRING "native benchmark wrappers to configure")
set(_fss_native_libraries fss libfss GPU-DPF torchcsprng EzPC fss-v0.7.0)
foreach(_library IN LISTS FSS_THIRD_PARTY_LIBRARIES)
    if(NOT _library IN_LIST _fss_native_libraries)
        message(FATAL_ERROR "unknown native benchmark wrapper: ${_library}")
    endif()
    add_subdirectory(${_library})
endforeach()
