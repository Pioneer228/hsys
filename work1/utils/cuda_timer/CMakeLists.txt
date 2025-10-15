cmake_minimum_required(VERSION 3.24)
project(cuda_timer LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_library(cuda_timer STATIC ${SRC})
add_library(work1::cuda_timer ALIAS cuda_timer)

target_include_directories(cuda_timer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

target_compile_features(cuda_timer PRIVATE
    cxx_std_20
)

target_link_libraries(cuda_timer PRIVATE
    CUDA::cudart
)
