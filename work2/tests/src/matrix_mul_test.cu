#include <work2/matrix.cuh>
#include <work2/matrix_operators.cuh>

#define EIGEN_NO_CUDA

#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

// Определяем row-major матрицу для согласованности с MatrixView
using RowMajorMatrixXf
    = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class MatrixMulTest : public ::testing::TestWithParam<
                          std::tuple<std::size_t, std::size_t, std::size_t, float>> {
 protected:
  bool matmul_test_impl(std::size_t m, std::size_t k, std::size_t n, float tol) {
    // Генерируем случайные матрицы в ROW-MAJOR порядке
    RowMajorMatrixXf A_target(m, k);
    A_target.setRandom();

    RowMajorMatrixXf B_target(k, n);
    B_target.setRandom();

    RowMajorMatrixXf C_target = A_target * B_target;  // эталонное произведение

    // Создаём матрицы на устройстве
    auto A = hsys::Matrix<float>(m, k);
    A.data().copy_from_host(A_target.data());

    auto B = hsys::Matrix<float>(k, n);
    B.data().copy_from_host(B_target.data());

    // Выполняем умножение через operator*
    auto C = A * B;

    // Проверяем размер результата
    if (C.nrows() != m || C.ncols() != n) {
      return false;
    }

    // Копируем результат обратно на хост
    RowMajorMatrixXf C_from_device(m, n);
    C.data().copy_to_host(C_from_device.data());

    // Сравниваем с эталоном
    return C_target.isApprox(C_from_device, tol);
  }
};

TEST_P(MatrixMulTest, matmul_test) {
  auto [m, k, n, tol] = GetParam();
  EXPECT_TRUE(matmul_test_impl(m, k, n, tol));
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    MatrixMulTestSuite,
    MatrixMulTest,
    ::testing::Values(
        std::make_tuple(1, 1, 1, 1e-5f),
        std::make_tuple(2, 3, 4, 1e-5f),
        std::make_tuple(10, 5, 8, 1e-5f),
        std::make_tuple(16, 16, 16, 1e-5f),
        std::make_tuple(32, 64, 128, 1e-5f),
        std::make_tuple(100, 50, 200, 1e-5f),
        std::make_tuple(127, 63, 255, 1e-5f),
        std::make_tuple(128, 128, 128, 1e-5f)
    )
);
// clang-format on