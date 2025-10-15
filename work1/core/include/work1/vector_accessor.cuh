#ifndef HSYS_VECTOR_ACCESSOR
#define HSYS_VECTOR_ACCESSOR

#include "kinds.cuh"

namespace hsys {

template <AtomKind AtomT>
struct VectorAccessor {
  struct hsys_vector_accessor_feature {};

 public:
  using atom_t = AtomT;

 private:
  atom_t* data_;
  std::size_t size_;

 public:
  __host__ __device__ VectorAccessor(atom_t* data, std::size_t size)
      : data_(data)
      , size_(size) {}

  [[nodiscard]] __host__ __device__ std::size_t size() const {
    return size_;
  }

  __host__ __device__ atom_t& operator[](std::size_t i) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return data_[i];
  }

  __host__ __device__ const atom_t& operator[](std::size_t i) const {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return data_[i];
  }
};

}  // namespace hsys

#endif  // HSYS_VECTOR_ACCESSOR
