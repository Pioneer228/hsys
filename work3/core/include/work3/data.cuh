#ifndef HSYS_DATA_CUH
#define HSYS_DATA_CUH

#include "kinds.cuh"

// TODO: check CUDA errors
namespace hsys {

template <AtomKind AtomT>
struct Data {
  struct hsys_data_feature {};

 private:
  std::size_t size_;
  AtomT* data_;

 public:
  using atom_t = AtomT;

  explicit __host__ Data(std::size_t size)
      : size_(size)
      , data_(nullptr) {
    cudaMalloc(&data_, size * sizeof(AtomT));
  }

  __host__ Data(const Data& other)
      : size_(other.size_)
      , data_(nullptr) {
    cudaMalloc(&data_, size_ * sizeof(AtomT));
    cudaMemcpy(data_, other.data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToDevice);
  }

  __host__ Data(Data&& other) noexcept
      : size_(other.size_)
      , data_(other.data_) {
    other.data_ = nullptr;
  }

  __host__ Data& operator=(const Data& other) {
    if (this != &other) {
      if (data_) cudaFree(data_);
      size_ = other.size_;
      cudaMalloc(&data_, size_ * sizeof(AtomT));
      cudaMemcpy(data_, other.data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToDevice);
    }
    return *this;
  }

  __host__ Data& operator=(Data&& other) noexcept {
    if (this != &other) {
      if (data_) cudaFree(data_);
      size_ = other.size_;
      data_ = other.data_;
      other.data_ = nullptr;
    }
    return *this;
  }

  __host__ AtomT* data() {
    return data_;
  }

  __host__ const AtomT* data() const {
    return data_;
  }

  [[nodiscard]] __host__ std::size_t size() const {
    return size_;
  }

  __host__ void copy_to_host(AtomT* host_ptr) const {
    cudaMemcpy(host_ptr, data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToHost);
  }

  __host__ void copy_from_host(const AtomT* host_ptr) {
    cudaMemcpy(data_, host_ptr, size_ * sizeof(AtomT), cudaMemcpyHostToDevice);
  }

  __host__ ~Data() noexcept {
    if (data_) cudaFree(data_);
  }
};

}  // namespace hsys

#endif  // HSYS_DATA_CUH