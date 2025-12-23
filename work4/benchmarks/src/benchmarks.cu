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

// ================= GPU (WMMA) =================

static void BM_MatMul_WMMA(benchmark::State& state, int N) {
  auto A = hsys::Matrix<half>(N, N);
  auto B = hsys::Matrix<half>(N, N);

  std::vector<half> host(N * N, __float2half(1.0f));
  A.data().copy_from_host(host.data());
  B.data().copy_from_host(host.data());

  for (auto _ : state) {
    float elapsed_time = 0.0f;
    {
      CUDATimer timer(elapsed_time);
      auto C = A * B;  // returns Matrix<float>
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
BENCHMARK_CAPTURE(BM_MatMul_WMMA, WMMA_16, 16)
    ->Name("CUDA Matrix Multiplication (WMMA)/16")
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK_CAPTURE(BM_MatMul_WMMA, WMMA_32, 32)
    ->Name("CUDA Matrix Multiplication (WMMA)/32")
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK_CAPTURE(BM_MatMul_WMMA, WMMA_64, 64)
    ->Name("CUDA Matrix Multiplication (WMMA)/64")
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK_CAPTURE(BM_MatMul_WMMA, WMMA_128, 128)
    ->Name("CUDA Matrix Multiplication (WMMA)/128")
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK_MAIN();
