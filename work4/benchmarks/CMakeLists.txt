cmake_minimum_required(VERSION 3.31)
project(work4_benchmarks LANGUAGES CUDA)

include(../../cmake/configs/cuda_arch.cmake)
include(../../cmake/deps/gbench.cmake)
include(../../cmake/deps/eigen.cmake)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_executable(work4_benchmarks ${SRC})

set_target_minimum_cuda_arch(work4_benchmarks 75)

target_compile_features(work4_benchmarks PRIVATE
    cuda_std_20
)

target_compile_options(work4_benchmarks PRIVATE
    -Wall -O3
)

target_link_libraries(work4_benchmarks PRIVATE
    benchmark::benchmark
    Eigen3::Eigen
    work4::core
    work4::cuda_timer
)
