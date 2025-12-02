#ifndef HSYS_MATRIX_OPERATORS_CUH
#define HSYS_MATRIX_OPERATORS_CUH

#include "kernels/kernel_matmul_shmem.cuh"
#include "matrix.cuh"

#include <cassert>

namespace hsys {

template <AtomKind AtomT>
Matrix<AtomT> operator*(const Matrix<AtomT>& A, const Matrix<AtomT>& B) {
  assert(A.ncols() == B.nrows() && "Matrix dimensions incompatible for multiplication");

  const std::size_t m = A.nrows();
  const std::size_t k = A.ncols();
  const std::size_t n = B.ncols();

  Matrix<AtomT> C(m, n);

  constexpr int TILE_SIZE = 16;
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);

  kernel_matmul_shmem<<<grid, block>>>(C.view(), A.view(), B.view());

#ifdef DEBUG
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "kernel_matmul_shmem failed: " + std::string(cudaGetErrorString(err)));
  }
#endif

  return C;  // RVO/NRVO — эффективно
}

}  // namespace hsys

#endif  // HSYS_MATRIX_OPERATORS_CUH