#ifndef _CUDAGH_HPP_
#define _CUDAGH_HPP_

#include <cstdint>
#include <cuda_runtime.h>

namespace cudagh {

inline std::uint32_t cover(std::uint32_t work_size, std::size_t block_size) {
  return (work_size + block_size - 1) / block_size;
}

}  // namespace cudagh

#endif  // _CUDAGH_HPP_
