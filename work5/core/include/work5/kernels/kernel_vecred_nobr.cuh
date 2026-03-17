#ifndef HSYS_KERNEL_VECRED_NOBR_CUH
#define HSYS_KERNEL_VECRED_NOBR_CUH

#include "../vector_view.cuh"

namespace hsys {

template <AtomKind AtomT, std::size_t BlockSize>
__global__ void kernel_vecred_nobr(VectorView<AtomT> result, const VectorView<AtomT> vector) {
  __shared__ AtomT partial_sums[BlockSize];

  const auto global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto stride = gridDim.x * blockDim.x;

  AtomT local_sum = AtomT{0};
  for (std::size_t i = global_idx; i < vector.size(); i += stride) {
    local_sum += vector[i];
  }

  partial_sums[threadIdx.x] = local_sum;
  __syncthreads();

  for (std::size_t offset = BlockSize / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      partial_sums[threadIdx.x] += partial_sums[threadIdx.x + offset];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(&result[0], partial_sums[0]);
  }
}

}  // namespace hsys

#endif  // HSYS_KERNEL_VECRED_NOBR_CUH
