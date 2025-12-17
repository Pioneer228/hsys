#include <cudagh.hpp>
// #include <utils/cudagh/include/cudagh.hpp>
// #include <work3/utils/cudagh/include/cudagh.hpp>

#include <work4/kernels/kernel_matmul_wmma.cuh>
#include <work4/matrix.cuh>
#include <work4/matrix_operators.cuh>

#define EIGEN_NO_CUDA
#include <Eigen/Dense>

#include <benchmark/benchmark.h>
#include <cuda_fp16.h>
#include <cuda_timer.hpp>
#include <vector>

// ================= CPU (Eigen) =================

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

// ================= GPU (WMMA) =================

static void BM_OurMatrixMulGPU(benchmark::State& state) {
  auto N = state.range(0);

  // --- данные half для WMMA ---
  auto a = hsys::Matrix<half>(N, N);
  auto b = hsys::Matrix<half>(N, N);
  auto c = hsys::Matrix<half>(N, N);

  std::vector<half> host(N * N, __float2half(1.0f));
  a.data().copy_from_host(host.data());
  b.data().copy_from_host(host.data());

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

// ================= Benchmark config =================

constexpr const int multiplier = 2;
// constexpr const auto range = std::make_pair(8, 1 << 26);

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
    ->Name("CUDA Matrix Multiplication (WMMA)")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK_MAIN();  // NOLINT
