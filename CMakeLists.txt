# For Ubuntu 20.04 focal
cmake_minimum_required(VERSION 3.16)
project(
  fss
  VERSION 0.7.0
  DESCRIPTION "Function secret sharing (FSS) primitives including distributed point functions (DPF) and distributed comparison functions (DCF)"
  HOMEPAGE_URL "https://github.com/myl7/fss"
  LANGUAGES C
)

add_library(
  dpf STATIC
  src/dpf.c include/dpf.h
  include/dpf_api.h
  src/utils.h
)
target_include_directories(dpf PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(dpf PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
