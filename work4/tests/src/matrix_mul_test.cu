#include <work4/matrix.cuh>
#include <work4/matrix_operators.cuh>

#define EIGEN_NO_CUDA
#include <Eigen/Dense>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using RowMajorFloat
    = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class MatrixMulWMMATest : public ::testing::TestWithParam<std::tuple<int, int, int>> {};

TEST_P(MatrixMulWMMATest, WMMA_MatMul) {
  auto [m, k, n] = GetParam();

  // ===== Генерация float-данных =====
  RowMajorFloat A_f(m, k);
  RowMajorFloat B_f(k, n);
  A_f.setRandom();
  B_f.setRandom();

  // ===== Эталон =====
  RowMajorFloat C_ref = A_f * B_f;

  // ===== Подготовка half-данных =====
  std::vector<half> A_h(m * k);
  std::vector<half> B_h(k * n);

  for (int i = 0; i < m * k; ++i) A_h[i] = __float2half(A_f.data()[i]);

  for (int i = 0; i < k * n; ++i) B_h[i] = __float2half(B_f.data()[i]);

  // ===== Matrix<half> =====
  hsys::Matrix<half> A(m, k);
  hsys::Matrix<half> B(k, n);

  A.data().copy_from_host(A_h.data());
  B.data().copy_from_host(B_h.data());

  // ===== operator* (WMMA) =====
  auto C = A * B;

  ASSERT_EQ(C.nrows(), m);
  ASSERT_EQ(C.ncols(), n);

  // ===== Копирование результата (float!) =====
  std::vector<float> C_h(m * n);
  C.data().copy_to_host(C_h.data());

  RowMajorFloat C_f(m, n);
  std::copy(C_h.begin(), C_h.end(), C_f.data());

  // ===== Проверка =====
  EXPECT_TRUE(C_ref.isApprox(C_f, 1e-2f));
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    WMMATests,
    MatrixMulWMMATest,
    ::testing::Values(
        std::make_tuple(16, 16, 16),
        std::make_tuple(32, 32, 32),
        std::make_tuple(64, 64, 64),
        std::make_tuple(128, 128, 128),
        std::make_tuple(256, 256, 256),
        std::make_tuple(512, 512, 512)
    )
);
// clang-format on
