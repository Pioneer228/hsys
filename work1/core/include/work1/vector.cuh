#ifndef HSYS_VECTOR_CUH
#define HSYS_VECTOR_CUH

#include "data_block.cuh"
#include "vector_accessor.cuh"

namespace hsys {

template <AtomKind AtomT>
struct Vector {
  struct hsys_vector_feature {};

 private:
  DataBlock<AtomT> block_;

 public:
  Vector(std::size_t size)
      : block_(size) {}

  [[nodiscard]] std::size_t size() const {
    return block_.size();
  }

  DataBlock<AtomT>& block() {
    return block_;
  }

  const DataBlock<AtomT>& block() const {
    return block_;
  }

  VectorAccessor<AtomT> accessor() {
    return VectorAccessor<AtomT>(block_.data(), size());
  }

  const VectorAccessor<AtomT> accessor() const {
    return VectorAccessor<AtomT>(const_cast<AtomT*>(block_.data()), size());
  }
};

}  // namespace hsys

#endif  // HSYS_VECTOR_CUH
