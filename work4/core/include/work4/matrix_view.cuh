#ifndef HSYS_MATRIX_VIEW
#define HSYS_MATRIX_VIEW

#include "kinds.cuh"
#include <cstddef>

namespace hsys {

template <AtomKind AtomT>
struct MatrixView {
  struct hsys_matrix_view_feature {};

 public:
  using atom_t = AtomT;

 private:
  atom_t* data_;
  const atom_t* cdata_;
  std::size_t nrows_;
  std::size_t ncols_;

 public:
  // ===== mutable view =====
  __host__ __device__ MatrixView(atom_t* data, std::size_t nrows, std::size_t ncols)
      : data_(data)
      , cdata_(data)
      , nrows_(nrows)
      , ncols_(ncols) {}

  // ===== const view =====
  __host__ __device__ MatrixView(const atom_t* data, std::size_t nrows, std::size_t ncols)
      : data_(nullptr)
      , cdata_(data)
      , nrows_(nrows)
      , ncols_(ncols) {}

  [[nodiscard]] __host__ __device__ std::size_t size() const {
    return nrows_ * ncols_;
  }

  [[nodiscard]] __host__ __device__ std::size_t nrows() const {
    return nrows_;
  }

  [[nodiscard]] __host__ __device__ std::size_t ncols() const {
    return ncols_;
  }

  // ===== mutable access =====
  __host__ __device__ atom_t& operator()(std::size_t i, std::size_t j) {
    return data_[i * ncols_ + j];
  }

  // ===== const access =====
  __host__ __device__ const atom_t& operator()(std::size_t i, std::size_t j) const {
    return cdata_[i * ncols_ + j];
  }
};

}  // namespace hsys

#endif  // HSYS_MATRIX_VIEW
