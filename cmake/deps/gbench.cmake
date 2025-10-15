include(FetchContent)

FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.2
)
FetchContent_MakeAvailable(benchmark)
