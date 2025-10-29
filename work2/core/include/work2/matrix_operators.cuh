#ifndef HSYS_MATRIX_OPERATORS_CUH
#define HSYS_MATRIX_OPERATORS_CUH

#include "kernels/kernel_matmul_naive.cuh"
#include "matrix.cuh"

#include <cassert>
#include <cudagh.hpp>

namespace hsys {

template <AtomKind AtomT>
Matrix<AtomT> operator*(const Matrix<AtomT>& A, const Matrix<AtomT>& B) {
  assert(A.ncols() == B.nrows() && "Matrix dimensions incompatible for multiplication");

  const std::size_t nrows = A.nrows();
  const std::size_t ncols = B.ncols();

  Matrix<AtomT> C(nrows, ncols);

  constexpr int BLOCK_SIZE = 16;
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((ncols + block.x - 1) / block.x, (nrows + block.y - 1) / block.y);

  kernel_matmul_naive<<<grid, block>>>(C.view(), A.view(), B.view());

  return C;  // RVO/NRVO — эффективно
}

}  // namespace hsys

#endif  // HSYS_MATRIX_OPERATORS_CUH