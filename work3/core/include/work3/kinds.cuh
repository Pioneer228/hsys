#ifndef HSYS_KINDS
#define HSYS_KINDS

#include <concepts>
#include <cuda_fp16.h>

namespace hsys {

template <class T>
concept AtomKind = std::floating_point<T> || std::integral<T> || std::same_as<T, half>;

template <class T>
concept MatrixKind = requires { typename T::hsys_matrix_feature; };

template <class T>
concept MatrixViewKind = requires { typename T::hsys_matrix_view_feature; };

template <class T>
concept DataKind = requires { typename T::hsys_data_feature; };

}  // namespace hsys

#endif  // HSYS_KINDS
