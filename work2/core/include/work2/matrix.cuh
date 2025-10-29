#ifndef HSYS_MATRIX_CUH
#define HSYS_MATRIX_CUH

#include "data.cuh"
#include "matrix_view.cuh"
#include <memory>

namespace hsys {

template <AtomKind AtomT>
struct Matrix {
  struct hsys_matrix_feature {};

 private:
  std::shared_ptr<Data<AtomT>> data_;
  std::size_t nrows_;
  std::size_t ncols_;
  MatrixView<AtomT> view_;

 public:
  Matrix(std::size_t nrows, std::size_t ncols)
      : data_(std::make_shared<Data<AtomT>>(nrows * ncols))
      , nrows_(nrows)
      , ncols_(ncols)
      , view_(data_->data(), nrows_, ncols_) {}

  [[nodiscard]] std::size_t size() const {
    return nrows_ * ncols_;
  }

  [[nodiscard]] std::size_t nrows() const {
    return nrows_;
  }

  [[nodiscard]] std::size_t ncols() const {
    return ncols_;
  }

  Data<AtomT>& data() {
    return *data_;
  }

  const Data<AtomT>& data() const {
    return *data_;
  }

  MatrixView<AtomT>& view() {
    return view_;
  }

  const MatrixView<AtomT>& view() const {
    return view_;
  }
};

}  // namespace hsys

#endif  // HSYS_MATRIX_CUH