#include <work5/vector.cuh>
#include <work5/vector_operators.cuh>

#define EIGEN_NO_CUDA

#include <Eigen/Dense>
#include <gtest/gtest.h>

class VectorReductionTest : public ::testing::TestWithParam<std::size_t> {};

TEST_P(VectorReductionTest, BothReductionsMatchEigenSum) {
  const auto size = GetParam();
  const auto eigen_size = static_cast<Eigen::Index>(size);

  Eigen::VectorXf host_vec = Eigen::VectorXf::Random(eigen_size);
  const float expected_sum = host_vec.sum();

  hsys::Vector<float> device_vec(size);
  device_vec.data().copy_from_host(host_vec.data());

  EXPECT_NEAR(hsys::vecred_nobr(device_vec), expected_sum, 1e-4f);
  EXPECT_NEAR(hsys::vecred_br(device_vec), expected_sum, 1e-4f);
}

TEST(VectorReductionStandaloneTest, EmptyVectorReturnsZero) {
  hsys::Vector<float> device_vec(0);

  EXPECT_NEAR(hsys::vecred_nobr(device_vec), 0.0f, 1e-4f);
  EXPECT_NEAR(hsys::vecred_br(device_vec), 0.0f, 1e-4f);
}

INSTANTIATE_TEST_SUITE_P(VectorSizes,
    VectorReductionTest,
    ::testing::Values(
        1UL,
        2UL,
        3UL,
        127UL,
        129UL,
        512UL,
        541UL,
        1037UL));
