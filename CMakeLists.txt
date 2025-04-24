cmake_minimum_required(VERSION 3.18) # For CUDA
project(
  fss
  VERSION 0.7.1
  DESCRIPTION "Function secret sharing (FSS) primitives including distributed point functions (DPF) and distributed comparison functions (DCF)"
  HOMEPAGE_URL "https://github.com/myl7/fss"
  LANGUAGES C
)

# Option `BUILD_TESTING` available
include(CTest)

set(FSS_kLambda 16 CACHE STRING "Custom kLambda")
option(BUILD_WITH_CUDA "Build with CUDA" ON)

if(BUILD_TESTING)
  enable_language(CXX)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(OpenMP REQUIRED)
find_package(Sodium REQUIRED)
# OpenSSL 3 for bundled AES MMO PRG
find_package(OpenSSL)

include(FetchContent)
if(BUILD_TESTING)
  FetchContent_Declare(
    GTest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG e90fe2485641bab0d6af4500192dc503384950d1
  )
endif()

if(BUILD_TESTING)
  if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  endif()
  FetchContent_MakeAvailable(GTest)
  include(GoogleTest)
endif()

add_library(dpf STATIC src/dpf.c)
target_compile_definitions(dpf PUBLIC kLambda=${FSS_kLambda})
target_include_directories(dpf PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(dpf PUBLIC OpenMP::OpenMP_C)

add_library(dcf STATIC src/dcf.c)
target_compile_definitions(dcf PUBLIC kLambda=${FSS_kLambda})
target_include_directories(dcf PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(dcf PUBLIC OpenMP::OpenMP_C)

add_library(cw_mac_bytes STATIC src/cw_mac_bytes.c)
target_compile_definitions(cw_mac_bytes PUBLIC kLambda=${FSS_kLambda})
target_include_directories(cw_mac_bytes PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(cw_mac_bytes PUBLIC sodium)

if(BUILD_WITH_CUDA)
  enable_language(CUDA)

  set(_FSS_CUDA_SOURCES
    src/dpf.c
    src/dcf.c
    src/group/u128_le.c
    src/group/bytes.c
  )
  set_source_files_properties(${_FSS_CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
  unset(_FSS_CUDA_SOURCES)

  add_library(dpf_cu STATIC src/dpf.c)
  target_compile_definitions(dpf_cu PUBLIC kLambda=${FSS_kLambda})
  target_include_directories(dpf_cu PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
  set_target_properties(dpf_cu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  add_library(dcf_cu STATIC src/dcf.c)
  target_compile_definitions(dcf_cu PUBLIC kLambda=${FSS_kLambda})
  target_include_directories(dcf_cu PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
  set_target_properties(dcf_cu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

if(BUILD_TESTING)
  include(cmake/test.cmake)
endif()
