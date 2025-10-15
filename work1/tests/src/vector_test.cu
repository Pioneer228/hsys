#include <gtest/gtest.h>
#include <work1/vector.cuh>

/* AI generated code: begin */

using namespace hsys;

TEST(VectorTest, ConstructorAndSize) {
  const std::size_t test_size = 100;
  Vector<float> vec(test_size);

  EXPECT_EQ(vec.size(), test_size);
}

TEST(VectorTest, ZeroSizeConstructor) {
  Vector<double> vec(0);
  EXPECT_EQ(vec.size(), 0);
}

TEST(VectorTest, BlockAccess) {
  const std::size_t test_size = 50;
  Vector<int> vec(test_size);

  auto block = vec.block();
  EXPECT_EQ(block.size(), test_size);

  const auto& const_vec = vec;
  auto const_block = const_vec.block();
  EXPECT_EQ(const_block.size(), test_size);
}

TEST(VectorTest, AccessorAccess) {
  const std::size_t test_size = 75;
  Vector<float> vec(test_size);

  auto accessor = vec.accessor();
  EXPECT_EQ(accessor.size(), test_size);

  const auto& const_vec = vec;
  auto const_accessor = const_vec.accessor();
  EXPECT_EQ(const_accessor.size(), test_size);
}

TEST(VectorTest, AccessorDataConsistency) {
  const std::size_t test_size = 30;
  Vector<double> vec(test_size);

  auto accessor = vec.accessor();
  auto block = vec.block();

  EXPECT_EQ(accessor.size(), block.size());
}

TEST(VectorTest, DifferentAtomTypes) {
  const std::size_t test_size = 25;

  Vector<float> float_vec(test_size);
  Vector<double> double_vec(test_size);
  Vector<int> int_vec(test_size);

  EXPECT_EQ(float_vec.size(), test_size);
  EXPECT_EQ(double_vec.size(), test_size);
  EXPECT_EQ(int_vec.size(), test_size);
}

TEST(VectorTest, ConstCorrectness) {
  const std::size_t test_size = 40;
  const Vector<float> const_vec(test_size);

  [[maybe_unused]] auto size = const_vec.size();
  [[maybe_unused]] auto block = const_vec.block();
  [[maybe_unused]] auto accessor = const_vec.accessor();

  EXPECT_EQ(size, test_size);
}

TEST(VectorTest, InstanceIndependence) {
  const std::size_t size1 = 10;
  const std::size_t size2 = 20;

  Vector<float> vec1(size1);
  Vector<float> vec2(size2);

  EXPECT_EQ(vec1.size(), size1);
  EXPECT_EQ(vec2.size(), size2);
  EXPECT_NE(vec1.size(), vec2.size());
}

class VectorSizeTest : public ::testing::TestWithParam<std::size_t> {};

TEST_P(VectorSizeTest, SizeParameterized) {
  const std::size_t test_size = GetParam();
  Vector<int> vec(test_size);

  EXPECT_EQ(vec.size(), test_size);
}

INSTANTIATE_TEST_SUITE_P(
    VectorSizes, VectorSizeTest, ::testing::Values(0, 1, 10, 100, 1000));

TEST(VectorTest, FeatureTypeExists) {
  [[maybe_unused]] Vector<float>::hsys_vector_feature feature;
  SUCCEED();
}

/* AI generated code: end */
