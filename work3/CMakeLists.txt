cmake_minimum_required(VERSION 3.24)
project(work2 LANGUAGES CXX)

option(ENABLE_TESTS "Enable tests" ON)
option(ENABLE_BENCH "Enable benchmarks" ON)

add_subdirectory(core)
add_subdirectory(utils/cuda_timer)
add_subdirectory(utils/cudagh)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

if(ENABLE_BENCH)
    add_subdirectory(benchmarks)
endif()
