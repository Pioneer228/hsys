#include <cudagh.hpp>

#include <work3/kernels/kernel_matmul_shmem.cuh>
#include <work3/matrix.cuh>
#include <work3/matrix_operators.cuh>

#define EIGEN_NO_CUDA
#include <Eigen/Dense>

#include <benchmark/benchmark.h>
#include <cuda_timer.hpp>

// ================= CPU =================

static void BM_EigenMatrixMulCPU(benchmark::State& state) {
  const int N = state.range(0);

  Eigen::MatrixXf A = Eigen::MatrixXf::Random(N, N);
  Eigen::MatrixXf B = Eigen::MatrixXf::Random(N, N);
  Eigen::MatrixXf C(N, N);

  for (auto _ : state) {
    C = A * B;
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }
}

// ================= GPU (Shared Memory) =================

static void BM_MatMul_Shmem(benchmark::State& state) {
  const int N = state.range(0);

  auto A = hsys::Matrix<float>(N, N);
  auto B = hsys::Matrix<float>(N, N);

  for (auto _ : state) {
    float elapsed_time = 0.0f;
    {
      CUDATimer timer(elapsed_time);
      auto C = A * B;
    }

    benchmark::DoNotOptimize(elapsed_time);
    benchmark::ClobberMemory();
    state.SetIterationTime(elapsed_time);
  }
}

// ================= Benchmark config =================

constexpr int multiplier = 2;
constexpr auto range = std::make_pair(8, 8192);
constexpr auto unit = benchmark::kMillisecond;

// ---------- CPU ----------
BENCHMARK(BM_EigenMatrixMulCPU)
    ->Name("Eigen Matrix Multiplication (CPU)")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseRealTime()
    ->MeasureProcessCPUTime();

// ---------- GPU ----------
BENCHMARK(BM_MatMul_Shmem)
    ->Name("CUDA Matrix Multiplication (Shared Memory)")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK_MAIN();
