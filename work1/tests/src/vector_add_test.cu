#include <work1/vector.cuh>
#include <work1/vector_operators.cuh>

#define EIGEN_NO_CUDA

#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

class VectorAddTest : public ::testing::TestWithParam<std::pair<std::size_t, float>> {
 protected:
  bool vadd_test_impl(std::size_t size, float tol) {
    Eigen::VectorXf a_target = Eigen::VectorXf::Random(
        size);  // NOLINT(cppcoreguidelines-narrowing-conversions)

    Eigen::VectorXf b_target = Eigen::VectorXf::Random(
        size);  // NOLINT(cppcoreguidelines-narrowing-conversions)

    Eigen::VectorXf c_target = a_target + b_target;

    auto a = hsys::Vector<float>(size);
    a.block().copy_from_host(a_target.data());

    auto b = hsys::Vector<float>(size);
    b.block().copy_from_host(b_target.data());

    auto c = a + b;

    if (c.size() != size) return false;

    Eigen::VectorXf c_from_device = Eigen::VectorXf(size);
    c.block().copy_to_host(c_from_device.data());

    return c_target.isApprox(c_from_device, tol);
  }
};

TEST_P(VectorAddTest, vadd_test) {  // It's Ok, dumb clangd!
  auto [size, tol] = GetParam();
  EXPECT_TRUE(vadd_test_impl(size, tol));
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    VectorAddTestSuite,
    VectorAddTest,
    ::testing::Values(
        std::make_pair(1, 1e-6),
        std::make_pair(2, 1e-6),
        std::make_pair(3, 1e-6),
        std::make_pair(127, 1e-6),
        std::make_pair(128, 1e-6),
        std::make_pair(129, 1e-6),
        std::make_pair(512, 1e-6),
        std::make_pair(513, 1e-6),
        std::make_pair(1023, 1e-6),
        std::make_pair(1024, 1e-6)
    )
);
// clang-format on
