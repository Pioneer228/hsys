cmake_minimum_required(VERSION 3.24)
project(cudagh LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)

add_library(cudagh INTERFACE)
add_library(work1::cudagh ALIAS cudagh)

target_include_directories(cudagh INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

target_compile_features(cudagh INTERFACE
    cxx_std_20
)

target_link_libraries(cudagh INTERFACE
    CUDA::cudart
)
