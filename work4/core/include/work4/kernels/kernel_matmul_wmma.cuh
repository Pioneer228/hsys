#ifndef HSYS_KERNEL_MATMUL_WMMA
#define HSYS_KERNEL_MATMUL_WMMA

#include <cuda_fp16.h>
#include <mma.h>
#include <type_traits>

#include "../kinds.cuh"
#include "../matrix_view.cuh"

namespace hsys {

using namespace nvcuda;

constexpr int wmma_tile = 16;

template <AtomKind AtomT>
__global__ void kernel_matmul_wmma(
    MatrixView<float> C, const MatrixView<AtomT> A, const MatrixView<AtomT> B) {

  static_assert(std::is_same_v<AtomT, half>, "WMMA kernel supports only half precision");

  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

  int tiles_per_row = (C.ncols() + wmma_tile - 1) / wmma_tile;

  int tile_i = warp_id / tiles_per_row;
  int tile_j = warp_id % tiles_per_row;

  int row = tile_i * wmma_tile;
  int col = tile_j * wmma_tile;

  if (row >= C.nrows() || col >= C.ncols()) return;

  wmma::fragment<wmma::matrix_a, wmma_tile, wmma_tile, wmma_tile, half, wmma::row_major>
      a_frag;

  wmma::fragment<wmma::matrix_b, wmma_tile, wmma_tile, wmma_tile, half, wmma::row_major>
      b_frag;

  wmma::fragment<wmma::accumulator, wmma_tile, wmma_tile, wmma_tile, float> c_frag;

  wmma::fill_fragment(c_frag, 0.0f);

  for (int k = 0; k < A.ncols(); k += wmma_tile) {
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
