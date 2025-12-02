#ifndef HSYS_KERNEL_MATMUL_SHMEM
#define HSYS_KERNEL_MATMUL_SHMEM

#include "../matrix_view.cuh"

namespace hsys {

constexpr int TILE_SIZE = 16;

template <AtomKind AtomT>
__global__ void kernel_matmul_shmem(
    MatrixView<AtomT> C, const MatrixView<AtomT> A, const MatrixView<AtomT> B) {
  __shared__ AtomT As[TILE_SIZE][TILE_SIZE];
  __shared__ AtomT Bs[TILE_SIZE][TILE_SIZE];

  int i = blockIdx.y * TILE_SIZE + threadIdx.y;
  int j = blockIdx.x * TILE_SIZE + threadIdx.x;

  AtomT sum = 0;

  for (int tile = 0; tile < (A.ncols() + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
    int ak = tile * TILE_SIZE + threadIdx.x;
    int ai = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (ai < A.nrows() && ak < A.ncols()) {
      As[threadIdx.y][threadIdx.x] = A(ai, ak);
    } else {
      As[threadIdx.y][threadIdx.x] = AtomT{0};
    }

    int bk = tile * TILE_SIZE + threadIdx.y;
    int bj = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (bk < B.nrows() && bj < B.ncols()) {
      Bs[threadIdx.y][threadIdx.x] = B(bk, bj);
    } else {
      Bs[threadIdx.y][threadIdx.x] = AtomT{0};
    }

    __syncthreads();

    for (int k = 0; k < TILE_SIZE; ++k) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (i < C.nrows() && j < C.ncols()) {
    C(i, j) = sum;
  }
}

}  // namespace hsys

#endif  // HSYS_KERNEL_MATMUL_SHMEM