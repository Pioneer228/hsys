#ifndef HSYS_DATA_BLOCK
#define HSYS_DATA_BLOCK

#include "kinds.cuh"

// TODO: check CUDA errors

namespace hsys {

template <AtomKind AtomT>
struct DataBlock {
  struct hsys_data_block_feature {};

 private:
  std::size_t size_;
  AtomT* data_;

 public:
  using atom_t = AtomT;

  DataBlock(std::size_t size)
      : size_(size)
      , data_(nullptr) {
    cudaMalloc(&data_, size * sizeof(AtomT));
  }

  DataBlock(const DataBlock& other)
      : size_(other.size_)
      , data_(nullptr) {
    cudaMalloc(&data_, size_ * sizeof(AtomT));
    cudaMemcpy(data_, other.data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToDevice);
  }

  DataBlock(DataBlock&& other) noexcept
      : size_(other.size_)
      , data_(other.data_) {
    other.data_ = nullptr;
  }

  DataBlock& operator=(const DataBlock& other) {
    if (this != &other) {
      if (data_) cudaFree(data_);
      size_ = other.size_;
      cudaMalloc(&data_, size_ * sizeof(AtomT));
      cudaMemcpy(data_, other.data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToDevice);
    }
    return *this;
  }

  DataBlock& operator=(DataBlock&& other) noexcept {
    if (this != &other) {
      if (data_) cudaFree(data_);
      size_ = other.size_;
      data_ = other.data_;
      other.data_ = nullptr;
    }
    return *this;
  }

  AtomT* data() {
    return data_;
  }

  const AtomT* data() const {
    return data_;
  }

  [[nodiscard]] std::size_t size() const {
    return size_;
  }

  void copy_to_host(AtomT* host_ptr) const {
    cudaMemcpy(host_ptr, data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToHost);
  }

  void copy_from_host(const AtomT* host_ptr) {
    cudaMemcpy(data_, host_ptr, size_ * sizeof(AtomT), cudaMemcpyHostToDevice);
  }

  ~DataBlock() noexcept {
    if (data_) cudaFree(data_);
  }
};

}  // namespace hsys

#endif  // HSYS_DATA_BLOCK
