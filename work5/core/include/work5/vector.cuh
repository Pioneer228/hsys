#ifndef HSYS_VECTOR_CUH
#define HSYS_VECTOR_CUH

#include "data.cuh"
#include "vector_view.cuh"
#include <memory>

namespace hsys {

template <AtomKind AtomT>
struct Vector {
  struct hsys_vector_feature {};

 private:
  std::shared_ptr<Data<AtomT>> data_;
  std::size_t size_;
  VectorView<AtomT> view_;

 public:
  explicit Vector(std::size_t size)
      : data_(std::make_shared<Data<AtomT>>(size))
      , size_(size)
      , view_(data_->data(), size_) {}

  [[nodiscard]] std::size_t size() const {
    return size_;
  }

  Data<AtomT>& data() {
    return *data_;
  }

  const Data<AtomT>& data() const {
    return *data_;
  }

  VectorView<AtomT>& view() {
    return view_;
  }

  const VectorView<AtomT>& view() const {
    return view_;
  }
};

}  // namespace hsys

#endif  // HSYS_VECTOR_CUH
