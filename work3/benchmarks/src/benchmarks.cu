#include <cudagh.hpp>

#include <work3/kernels/kernel_matmul_shmem.cuh>
#include <work3/matrix.cuh>
#include <work3/matrix_operators.cuh>

#define EIGEN_NO_CUDA
#include <Eigen/Dense>

#include <benchmark/benchmark.h>
#include <cuda_timer.hpp>

// ================= CPU =================

static void BM_EigenMatrixMulCPU(benchmark::State& state, int N) {
  Eigen::MatrixXf A = Eigen::MatrixXf::Random(N, N);
  Eigen::MatrixXf B = Eigen::MatrixXf::Random(N, N);
  Eigen::MatrixXf C(N, N);

  for (auto _ : state) {
    C = A * B;
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }
}

// ================= GPU (shared memory) =================

static void BM_MatMul_Shmem(benchmark::State& state, int N) {
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

constexpr const auto unit = benchmark::kMillisecond;

// ---------- CPU ----------
BENCHMARK_CAPTURE(BM_EigenMatrixMulCPU, CPU_16, 16)
    ->Name("Eigen Matrix Multiplication (CPU)/16")
    ->Unit(unit);

BENCHMARK_CAPTURE(BM_EigenMatrixMulCPU, CPU_32, 32)
    ->Name("Eigen Matrix Multiplication (CPU)/32")
    ->Unit(unit);

BENCHMARK_CAPTURE(BM_EigenMatrixMulCPU, CPU_64, 64)
    ->Name("Eigen Matrix Multiplication (CPU)/64")
    ->Unit(unit);

BENCHMARK_CAPTURE(BM_EigenMatrixMulCPU, CPU_128, 128)
    ->Name("Eigen Matrix Multiplication (CPU)/128")
    ->Unit(unit);

// ---------- GPU ----------
BENCHMARK_CAPTURE(BM_MatMul_Shmem, SHMEM_16, 16)
    ->Name("CUDA Matrix Multiplication (Shared Memory)/16")
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK_CAPTURE(BM_MatMul_Shmem, SHMEM_32, 32)
    ->Name("CUDA Matrix Multiplication (Shared Memory)/32")
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK_CAPTURE(BM_MatMul_Shmem, SHMEM_64, 64)
    ->Name("CUDA Matrix Multiplication (Shared Memory)/64")
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK_CAPTURE(BM_MatMul_Shmem, SHMEM_128, 128)
    ->Name("CUDA Matrix Multiplication (Shared Memory)/128")
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK_MAIN();
