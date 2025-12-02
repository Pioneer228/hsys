cmake_minimum_required(VERSION 3.31)
project(work3_tests LANGUAGES CUDA)

include(../../cmake/configs/cuda_arch.cmake)
include(../../cmake/deps/gtest.cmake)
include(../../cmake/deps/eigen.cmake)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_executable(work3_tests ${SRC})

set_target_minimum_cuda_arch(work3_tests 75)

target_compile_features(work3_tests PRIVATE
    cuda_std_20
)

target_compile_options(work3_tests PRIVATE
    -Wall -O3
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -Wno-deprecated-gpu-targets
    >
)

target_link_libraries(work3_tests PRIVATE
    GTest::gtest_main
    Eigen3::Eigen
    work3::core
)

enable_testing()
gtest_discover_tests(work3_tests)
