#ifndef HSYS_VECTOR_VIEW_CUH
#define HSYS_VECTOR_VIEW_CUH

#include "kinds.cuh"
#include <cstddef>

namespace hsys {

template <AtomKind AtomT>
struct VectorView {
  struct hsys_vector_view_feature {};

 public:
  using atom_t = AtomT;

 private:
  atom_t* data_;
  std::size_t size_;

 public:
  __host__ __device__ VectorView(atom_t* data, std::size_t size)
      : data_(data)
      , size_(size) {}

  __host__ __device__ VectorView(const atom_t* data, std::size_t size)
      : data_(const_cast<atom_t*>(data))
      , size_(size) {}

  [[nodiscard]] __host__ __device__ std::size_t size() const {
    return size_;
  }

  [[nodiscard]] __host__ __device__ atom_t* data() {
    return data_;
  }

  [[nodiscard]] __host__ __device__ const atom_t* data() const {
    return data_;
  }

  __host__ __device__ atom_t& operator[](std::size_t i) {
    return data_[i];
  }

  __host__ __device__ const atom_t& operator[](std::size_t i) const {
    return data_[i];
  }
};

}  // namespace hsys

#endif  // HSYS_VECTOR_VIEW_CUH
