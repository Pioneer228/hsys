#ifndef HSYS_KERNEL_VECRED_BR_CUH
#define HSYS_KERNEL_VECRED_BR_CUH

#include "../vector_view.cuh"

namespace hsys {

inline constexpr std::size_t kWarpWidth = 32;

template <class AtomT>
__device__ AtomT warp_reduce_sum(AtomT value) {
  for (int offset = static_cast<int>(kWarpWidth / 2); offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }

  return value;
}

template <AtomKind AtomT, std::size_t BlockSize>
__global__ void kernel_vecred_br(VectorView<AtomT> result, const VectorView<AtomT> vector) {
  static_assert(BlockSize % kWarpWidth == 0, "BlockSize must be divisible by warp width");

  constexpr std::size_t warp_count = BlockSize / kWarpWidth;
  __shared__ AtomT warp_sums[warp_count];

  const auto global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto stride = gridDim.x * blockDim.x;

  AtomT local_sum = AtomT{0};
  for (std::size_t i = global_idx; i < vector.size(); i += stride) {
    local_sum += vector[i];
  }

  local_sum = warp_reduce_sum(local_sum);

  const auto lane = threadIdx.x % kWarpWidth;
  const auto warp_id = threadIdx.x / kWarpWidth;

  if (lane == 0) {
    warp_sums[warp_id] = local_sum;
  }
  __syncthreads();

  if (warp_id == 0) {
    AtomT block_sum = AtomT{0};
    if (threadIdx.x < warp_count) {
      block_sum = warp_sums[threadIdx.x];
    }

    block_sum = warp_reduce_sum(block_sum);

    if (lane == 0) {
      atomicAdd(&result[0], block_sum);
    }
  }
}

}  // namespace hsys

#endif  // HSYS_KERNEL_VECRED_BR_CUH
