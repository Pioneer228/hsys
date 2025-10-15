#ifndef HSYS_KERNEL_VECTOR_ADD
#define HSYS_KERNEL_VECTOR_ADD

#include "../vector_accessor.cuh"

namespace hsys {

template <AtomKind AtomT>
__global__ void kernel_vector_add(VectorAccessor<AtomT> c,
    const VectorAccessor<AtomT> a,
    const VectorAccessor<AtomT> b) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < a.size()) c[i] = a[i] + b[i];
}

}  // namespace hsys

#endif  // HSYS_KERNEL_VECTOR_ADD
