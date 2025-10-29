#ifndef HSYS_KERNEL_MATMUL_NAIVE
#define HSYS_KERNEL_MATMUL_NAIVE

#include "../matrix_view.cuh"

namespace hsys {

template <AtomKind AtomT>
__global__ void kernel_matmul_naive(
    MatrixView<AtomT> C, const MatrixView<AtomT> A, const MatrixView<AtomT> B) {
  std::size_t i = blockIdx.y * blockDim.y + threadIdx.y;
  std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < C.nrows() && j < C.ncols()) {
    AtomT sum = 0;
    for (std::size_t k = 0; k < A.ncols(); ++k) {
      sum += A(i, k) * B(k, j);
    }
    C(i, j) = sum;
  }
}

}  // namespace hsys

#endif  // HSYS_KERNEL_MATMUL_NAIVE