#ifndef HSYS_MATRIX_OPERATORS_CUH
#define HSYS_MATRIX_OPERATORS_CUH

#include "kernels/kernel_matmul_wmma.cuh"
#include "matrix.cuh"

#include <cassert>
#include <cuda_runtime.h>
#include <stdexcept>

namespace hsys {

// WMMA: half × half → float
inline Matrix<float> operator*(const Matrix<half>& A, const Matrix<half>& B) {
  assert(A.ncols() == B.nrows() && "Matrix dimensions incompatible for multiplication");

  const std::size_t m = A.nrows();
  const std::size_t n = B.ncols();

  Matrix<float> C(m, n);

  constexpr int WMMA_TILE = 16;
  constexpr int WARPS_PER_BLOCK = 4;
  constexpr int WARP_SIZE = 32;

  dim3 block(WARPS_PER_BLOCK * WARP_SIZE);

  int tiles = ((m + WMMA_TILE - 1) / WMMA_TILE) * ((n + WMMA_TILE - 1) / WMMA_TILE);

  dim3 grid((tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

  kernel_matmul_wmma<half><<<grid, block>>>(C.view(), A.view(), B.view());

#ifdef DEBUG
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "kernel_matmul_wmma failed: " + std::string(cudaGetErrorString(err)));
  }
#endif

  return C;  // RVO
}

}  // namespace hsys

#endif  // HSYS_MATRIX_OPERATORS_CUH
