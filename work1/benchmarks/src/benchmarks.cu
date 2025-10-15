#include <cudagh.hpp>
#include <work1/kernels/kernel_vector_add.cuh>
#include <work1/vector.cuh>

#define EIGEN_NO_CUDA

#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <cuda_timer.hpp>

static void BM_EigenVectorAddCPU(benchmark::State& state) {
  auto len = state.range(0);

  Eigen::VectorXf a = Eigen::VectorXf(len);
  Eigen::VectorXf b = Eigen::VectorXf(len);
  Eigen::VectorXf result(len);

  for (auto _ : state) {
    result = a + b;  // lazy RHS
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
}

static void BM_OurVectorAddGPU(benchmark::State& state) {
  auto size = state.range(0);

  auto a = hsys::Vector<float>(size);
  auto b = hsys::Vector<float>(size);
  auto c = hsys::Vector<float>(size);

  for (auto _ : state) {
    float elapsed_time = 0;

    {
      CUDATimer timer(elapsed_time);
      hsys::kernel_vector_add<<<cudagh::cover(size, 128), 128>>>(
          c.accessor(), a.accessor(), b.accessor());
    }

    benchmark::DoNotOptimize(elapsed_time);
    benchmark::ClobberMemory();

    state.SetIterationTime(elapsed_time);
  }
}

constexpr const int multiplier = 8;
constexpr const auto range = std::make_pair(8, 1 << 26);
constexpr const auto unit = benchmark::kMillisecond;

BENCHMARK(BM_EigenVectorAddCPU)
    ->Name("Eigen Vector Addition (CPU)")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseRealTime()
    ->MeasureProcessCPUTime();

BENCHMARK(BM_OurVectorAddGPU)
    ->Name("CUDA Vector Addition (GPU)")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK_MAIN();  // NOLINT
