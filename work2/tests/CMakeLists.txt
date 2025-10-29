cmake_minimum_required(VERSION 3.31)
project(work2_tests LANGUAGES CUDA)

include(../../cmake/configs/cuda_arch.cmake)
include(../../cmake/deps/gtest.cmake)
include(../../cmake/deps/eigen.cmake)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_executable(work2_tests ${SRC})

set_target_minimum_cuda_arch(work2_tests 75)

target_compile_features(work2_tests PRIVATE
    cuda_std_20
)

target_compile_options(work2_tests PRIVATE
    -Wall -O3
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -Wno-deprecated-gpu-targets
    >
)

target_link_libraries(work2_tests PRIVATE
    GTest::gtest_main
    Eigen3::Eigen
    work2::core
)

enable_testing()
gtest_discover_tests(work2_tests)
