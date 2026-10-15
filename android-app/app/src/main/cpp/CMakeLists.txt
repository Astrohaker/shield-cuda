cmake_minimum_required(VERSION 3.18.1)
project(nstcuda C)

add_library(nstcuda SHARED nstcuda.c)
target_include_directories(nstcuda PRIVATE ${CUDA_HOME}/include)
target_link_libraries(nstcuda android log dl m)
