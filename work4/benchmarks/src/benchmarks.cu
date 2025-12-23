#include <cudagh.hpp>

#include <work4/kernels/kernel_matmul_wmma.cuh>
#include <work4/matrix.cuh>
#include <work4/matrix_operators.cuh>

#define EIGEN_NO_CUDA
#include <Eigen/Dense>

#include <benchmark/benchmark.h>
#include <cuda_fp16.h>
#include <cuda_timer.hpp>
#include <vector>

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

// ================= GPU (WMMA) =================

static void BM_MatMul_WMMA(benchmark::State& state) {
  const int N = state.range(0);

  auto A = hsys::Matrix<half>(N, N);
  auto B = hsys::Matrix<half>(N, N);

  std::vector<half> host(N * N, __float2half(1.0f));
  A.data().copy_from_host(host.data());
  B.data().copy_from_host(host.data());

  for (auto _ : state) {
    float elapsed_time = 0.0f;
    {
      CUDATimer timer(elapsed_time);
      auto C = A * B;  // Matrix<float>
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
BENCHMARK(BM_MatMul_WMMA)
    ->Name("CUDA Matrix Multiplication (WMMA)")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK_MAIN();
