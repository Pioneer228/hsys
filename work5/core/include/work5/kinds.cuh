#ifndef HSYS_KINDS_CUH
#define HSYS_KINDS_CUH

#include <concepts>
#include <cuda_fp16.h>

namespace hsys {

template <class T>
concept AtomKind = std::floating_point<T> || std::integral<T> || std::same_as<T, half>;

template <class T>
concept VectorKind = requires { typename T::hsys_vector_feature; };

template <class T>
concept VectorViewKind = requires { typename T::hsys_vector_view_feature; };

template <class T>
concept DataKind = requires { typename T::hsys_data_feature; };

}  // namespace hsys

#endif  // HSYS_KINDS_CUH
