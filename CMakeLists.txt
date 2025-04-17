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

option(BUILD_WITH_CUDA "Build with CUDA" ON)
option(BUILD_SAMPLES "Build with samples" ON)

if(BUILD_TESTING)
  enable_language(CXX)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(OpenMP REQUIRED)
find_package(Sodium REQUIRED)

include(FetchContent)
if(BUILD_TESTING)
  FetchContent_Declare(
    GTest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG e90fe2485641bab0d6af4500192dc503384950d1
  )
  if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  endif()
endif()
if(BUILD_TESTING)
  FetchContent_MakeAvailable(GTest)
  include(GoogleTest)
endif()

add_library(dpf STATIC src/dpf.c)
target_include_directories(dpf PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(dpf PUBLIC OpenMP::OpenMP_C)

add_library(dcf STATIC src/dcf.c)
target_include_directories(dcf PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(dcf PUBLIC OpenMP::OpenMP_C)

add_library(cw_mac_bytes STATIC src/cw_mac_bytes.c)
target_include_directories(cw_mac_bytes PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(cw_mac_bytes PUBLIC sodium)

if(BUILD_TESTING)
  include(cmake/test.cmake)
endif()

if(BUILD_WITH_CUDA)
  include(cmake/cuda.cmake)
endif()

if(BUILD_SAMPLES)
  include(cmake/samples.cmake)
endif()
