add_library(cw_mac_bytes STATIC src/cw_mac_bytes.c)
target_include_directories(cw_mac_bytes PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(cw_mac_bytes PRIVATE sodium)
