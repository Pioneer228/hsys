#include <cudagh.hpp>
//#include <utils/cudagh/include/cudagh.hpp>
//#include <work2/utils/cudagh/include/cudagh.hpp>
#include <work2/kernels/kernel_matmul_naive.cuh>
#include <work2/matrix.cuh>
#include <work2/matrix_operators.cuh>

#define EIGEN_NO_CUDA
#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <cuda_timer.hpp>
//#include <work2/utils/cuda_timer/include/cuda_timer.hpp>

static void BM_EigenMatrixMulCPU(benchmark::State& state) {
  auto N = state.range(0);

  Eigen::MatrixXf A = Eigen::MatrixXf::Random(N, N);
  Eigen::MatrixXf B = Eigen::MatrixXf::Random(N, N);
  Eigen::MatrixXf C(N, N);

  for (auto _ : state) {
    C = A * B; 
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }
}

static void BM_OurMatrixMulGPU(benchmark::State& state) {
  auto N = state.range(0);

  auto a = hsys::Matrix<float>(N, N);
  auto b = hsys::Matrix<float>(N, N);
  auto c = hsys::Matrix<float>(N, N);

  for (auto _ : state) {
    float elapsed_time = 0;
    {
      CUDATimer timer(elapsed_time);
      auto c = a * b;
    }

    benchmark::DoNotOptimize(elapsed_time);
    benchmark::ClobberMemory();

    state.SetIterationTime(elapsed_time);
  }
}

constexpr const int multiplier = 2;
//constexpr const auto range = std::make_pair(8, 1 << 26);
constexpr auto range = std::make_pair(8, 8192);
constexpr const auto unit = benchmark::kMillisecond;

BENCHMARK(BM_EigenMatrixMulCPU)
    ->Name("Eigen Matrix Multiplication (CPU)")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseRealTime()
    ->MeasureProcessCPUTime();

BENCHMARK(BM_OurMatrixMulGPU)
    ->Name("CUDA Matrix Multiplication (GPU)")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK_MAIN();  // NOLINT
