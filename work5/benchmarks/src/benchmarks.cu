#include <benchmark/benchmark.h>
#include <cuda_timer.hpp>
#include <cuda_runtime.h>
#include <work5/vector.cuh>
#include <work5/vector_operators.cuh>

#include <cstddef>
#include <cstdint>

namespace {

constexpr std::size_t kSafetyMarginBytes = 64ULL * 1024ULL * 1024ULL;
constexpr auto kUnit = benchmark::kMillisecond;

bool has_enough_device_memory(std::size_t vector_size) {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);

  const auto required_bytes = vector_size * sizeof(float) + sizeof(float);
  return required_bytes + kSafetyMarginBytes < free_bytes;
}

void add_work5_sizes(benchmark::internal::Benchmark* benchmark) {
  std::int64_t size = 8;
  for (int power = 0; power <= 28; ++power) {
    benchmark->Arg(size);
    size *= 2;
  }
}

template <class Reducer>
void run_vector_reduction_benchmark(benchmark::State& state, Reducer reducer) {
  const auto size = static_cast<std::size_t>(state.range(0));

  if (!has_enough_device_memory(size)) {
    state.SkipWithError("Not enough device memory for benchmark input");
    return;
  }

  auto input = hsys::Vector<float>(size);
  auto result = hsys::Vector<float>(1);

  cudaMemset(input.data().data(), 0, size * sizeof(float));
  cudaMemset(result.data().data(), 0, sizeof(float));

  for (auto _ : state) {
    float elapsed_time = 0;
    {
      CUDATimer timer(elapsed_time);
      reducer(result, input);
    }

    benchmark::DoNotOptimize(result.data().data());
    benchmark::ClobberMemory();
    state.SetIterationTime(elapsed_time);
  }
}

}  // namespace

static void BM_CUDAVecRedNoBroadcast(benchmark::State& state) {
  run_vector_reduction_benchmark(
      state,
      [](auto& result, const auto& input) { hsys::vecred_nobr(result, input); });
}

static void BM_CUDAVecRedBroadcast(benchmark::State& state) {
  run_vector_reduction_benchmark(
      state,
      [](auto& result, const auto& input) { hsys::vecred_br(result, input); });
}

BENCHMARK(BM_CUDAVecRedNoBroadcast)
    ->Name("CUDA Vector Reduction (Shared Memory)")
    ->Apply(add_work5_sizes)
    ->Unit(kUnit)
    ->UseManualTime();

BENCHMARK(BM_CUDAVecRedBroadcast)
    ->Name("CUDA Vector Reduction (Warp Shuffle)")
    ->Apply(add_work5_sizes)
    ->Unit(kUnit)
    ->UseManualTime();

BENCHMARK_MAIN();  // NOLINT
