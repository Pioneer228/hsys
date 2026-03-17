#ifndef HSYS_VECTOR_OPERATORS_CUH
#define HSYS_VECTOR_OPERATORS_CUH

#include "kernels/kernel_vecred_br.cuh"
#include "kernels/kernel_vecred_nobr.cuh"
#include "vector.cuh"

#include <algorithm>
#include <cstdint>
#include <cudagh.hpp>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace hsys {

namespace detail {

inline std::uint32_t make_vecred_grid_size(std::size_t size, std::size_t block_size) {
  if (size == 0) {
    return 1;
  }

  const auto covered = static_cast<std::size_t>(
      cudagh::cover(static_cast<std::uint32_t>(size), block_size));

  return static_cast<std::uint32_t>(std::max<std::size_t>(
      1, std::min<std::size_t>(covered, 1024)));
}

#ifdef DEBUG
inline void throw_on_kernel_error(const char* kernel_name) {
  cudaDeviceSynchronize();
  const auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string(kernel_name) + " failed: " + cudaGetErrorString(err));
  }
}
#endif

}  // namespace detail

template <AtomKind AtomT>
void vecred_nobr(Vector<AtomT>& result, const Vector<AtomT>& vector) {
  if (result.size() != 1) {
    throw std::invalid_argument("vecred_nobr expects a result vector of size 1");
  }

  cudaMemset(result.data().data(), 0, sizeof(AtomT));

  if (vector.size() == 0) {
    return;
  }

  constexpr std::size_t block_size = 256;
  const auto grid_size = detail::make_vecred_grid_size(vector.size(), block_size);

  kernel_vecred_nobr<AtomT, block_size><<<grid_size, block_size>>>(result.view(), vector.view());

#ifdef DEBUG
  detail::throw_on_kernel_error("kernel_vecred_nobr");
#endif
}

template <AtomKind AtomT>
void vecred_br(Vector<AtomT>& result, const Vector<AtomT>& vector) {
  if (result.size() != 1) {
    throw std::invalid_argument("vecred_br expects a result vector of size 1");
  }

  cudaMemset(result.data().data(), 0, sizeof(AtomT));

  if (vector.size() == 0) {
    return;
  }

  constexpr std::size_t block_size = 256;
  const auto grid_size = detail::make_vecred_grid_size(vector.size(), block_size);

  kernel_vecred_br<AtomT, block_size><<<grid_size, block_size>>>(result.view(), vector.view());

#ifdef DEBUG
  detail::throw_on_kernel_error("kernel_vecred_br");
#endif
}

template <AtomKind AtomT>
AtomT vecred_nobr(const Vector<AtomT>& vector) {
  if (vector.size() == 0) {
    return AtomT{0};
  }

  Vector<AtomT> result(1);
  vecred_nobr(result, vector);

  AtomT host_result{};
  result.data().copy_to_host(&host_result);

  return host_result;
}

template <AtomKind AtomT>
AtomT vecred_br(const Vector<AtomT>& vector) {
  if (vector.size() == 0) {
    return AtomT{0};
  }

  Vector<AtomT> result(1);
  vecred_br(result, vector);

  AtomT host_result{};
  result.data().copy_to_host(&host_result);

  return host_result;
}

}  // namespace hsys

#endif  // HSYS_VECTOR_OPERATORS_CUH
