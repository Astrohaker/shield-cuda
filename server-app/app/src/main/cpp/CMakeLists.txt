cmake_minimum_required(VERSION 3.18.1)
project(cudausbserver C)

# The same server as build/android/cudausb-server, built as a library (no main()).
set(SERVER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../server)
add_library(cudausbserver SHARED jni.c ${SERVER_DIR}/server.c)
target_compile_definitions(cudausbserver PRIVATE CUDAUSB_LIBRARY)
target_include_directories(cudausbserver PRIVATE ${CUDA_HOME}/include ${SERVER_DIR})
target_link_libraries(cudausbserver log dl)
