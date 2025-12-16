cmake_minimum_required(VERSION 3.31)
project(work4_tests LANGUAGES CUDA)

include(../../cmake/configs/cuda_arch.cmake)
include(../../cmake/deps/gtest.cmake)
include(../../cmake/deps/eigen.cmake)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_executable(work4_tests ${SRC})

set_target_minimum_cuda_arch(work4_tests 75)

target_compile_features(work4_tests PRIVATE
    cuda_std_20
)

target_compile_options(work4_tests PRIVATE
    -Wall -O3
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -Wno-deprecated-gpu-targets
    >
)

target_link_libraries(work4_tests PRIVATE
    GTest::gtest_main
    Eigen3::Eigen
    work4::core
)

enable_testing()
gtest_discover_tests(work4_tests)
