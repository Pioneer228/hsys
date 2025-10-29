cmake_minimum_required(VERSION 3.31)
project(work2_core LANGUAGES CUDA)

include(../../cmake/configs/cuda_arch.cmake)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

add_library(work2_core INTERFACE)
add_library(work2::core ALIAS work2_core)

set_target_minimum_cuda_arch(work2_core 75)

target_compile_features(work2_core INTERFACE
    cuda_std_20
)

target_compile_options(work2_core INTERFACE
    -Wall
    -O3
    -Wno-deprecated-gpu-targets
    -Wno-unknown-pragmas
)

target_include_directories(work2_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

target_link_libraries(work2_core INTERFACE
    work2::cudagh
)
