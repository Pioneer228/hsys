#ifndef HSYS_KERNEL_MATMUL_WMMA
#define HSYS_KERNEL_MATMUL_WMMA

#include <cuda_fp16.h>
#include <mma.h>
#include <type_traits>

#include "../kinds.cuh"
#include "../matrix_view.cuh"

namespace hsys {

using namespace nvcuda;

constexpr int WMMA_TILE = 16;

template <AtomKind AtomT>
__global__ void kernel_matmul_wmma(
    MatrixView<float> C, const MatrixView<AtomT> A, const MatrixView<AtomT> B) {

  static_assert(std::is_same_v<AtomT, half>, "WMMA kernel supports only half precision");

  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

  int tiles_per_row = (C.ncols() + WMMA_TILE - 1) / WMMA_TILE;

  int tile_i = warp_id / tiles_per_row;
  int tile_j = warp_id % tiles_per_row;

  int row = tile_i * WMMA_TILE;
  int col = tile_j * WMMA_TILE;

  if (row >= C.nrows() || col >= C.ncols()) return;

  wmma::fragment<wmma::matrix_a, WMMA_TILE, WMMA_TILE, WMMA_TILE, half, wmma::row_major>
      a_frag;

  wmma::fragment<wmma::matrix_b, WMMA_TILE, WMMA_TILE, WMMA_TILE, half, wmma::row_major>
      b_frag;

  wmma::fragment<wmma::accumulator, WMMA_TILE, WMMA_TILE, WMMA_TILE, float> c_frag;

  wmma::fill_fragment(c_frag, 0.0f);

  for (int k = 0; k < A.ncols(); k += WMMA_TILE) {
    const half* a_ptr = &A(row, k);
    const half* b_ptr = &B(k, col);

    wmma::load_matrix_sync(a_frag, a_ptr, A.ncols());
    wmma::load_matrix_sync(b_frag, b_ptr, B.ncols());

    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  float* c_ptr = &C(row, col);
  wmma::store_matrix_sync(c_ptr, c_frag, C.ncols(), wmma::mem_row_major);
}

}  // namespace hsys

#endif
