cmake_minimum_required(VERSION 3.31)
project(work5_benchmarks LANGUAGES CUDA)

include(../../cmake/configs/cuda_arch.cmake)
include(../../cmake/deps/gbench.cmake)
include(../../cmake/deps/eigen.cmake)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_executable(work5_benchmarks ${SRC})

set_target_minimum_cuda_arch(work5_benchmarks 75)

target_compile_features(work5_benchmarks PRIVATE
    cuda_std_20
)

target_compile_options(work5_benchmarks PRIVATE
    -Wall -O3
)

target_link_libraries(work5_benchmarks PRIVATE
    benchmark::benchmark
    Eigen3::Eigen
    work5::core
    work5::cuda_timer
)
