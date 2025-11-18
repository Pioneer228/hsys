cmake_minimum_required(VERSION 3.31)
project(work3_benchmarks LANGUAGES CUDA)

include(../../cmake/configs/cuda_arch.cmake)
include(../../cmake/deps/gbench.cmake)
include(../../cmake/deps/eigen.cmake)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_executable(work3_benchmarks ${SRC})

set_target_minimum_cuda_arch(work3_benchmarks 75)

target_compile_features(work3_benchmarks PRIVATE
    cuda_std_20
)

target_compile_options(work3_benchmarks PRIVATE
    -Wall -O3
)

target_link_libraries(work3_benchmarks PRIVATE
    benchmark::benchmark
    Eigen3::Eigen
    work3::core
    work3::cuda_timer
)
