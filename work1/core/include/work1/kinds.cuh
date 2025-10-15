#ifndef HSYS_KINDS
#define HSYS_KINDS

#include <concepts>
#include <cuda_fp16.h>

namespace hsys {

template <class T>
concept AtomKind = std::floating_point<T> || std::integral<T> || std::same_as<T, half>;

template <class T>
concept VectorKind
    = requires { typename T::hsys_vector_feature; };  // TODO: add more traits

template <class T>
concept VectorAccessorKind = requires { typename T::hsys_vector_accessor_feature; };

template <class T>
concept DataBlockKind = requires { typename T::hsys_data_block_feature; };

}  // namespace hsys

#endif  // HSYS_KINDS
