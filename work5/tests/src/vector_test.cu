#include <work5/kinds.cuh>
#include <work5/vector.cuh>

#include <gtest/gtest.h>

using namespace hsys;

TEST(VectorTest, ConstructorAndSize) {
  const std::size_t size = 100;
  Vector<float> vec(size);

  EXPECT_EQ(vec.size(), size);
  EXPECT_EQ(vec.data().size(), size);
}

TEST(VectorTest, ZeroSizeConstructor) {
  Vector<double> vec(0);

  EXPECT_EQ(vec.size(), 0);
  EXPECT_EQ(vec.data().size(), 0);
}

TEST(VectorTest, ViewAccess) {
  const std::size_t size = 75;
  Vector<float> vec(size);

  auto view = vec.view();
  EXPECT_EQ(view.size(), size);

  const auto& const_vec = vec;
  auto const_view = const_vec.view();
  EXPECT_EQ(const_view.size(), size);
}

TEST(VectorTest, DifferentAtomTypes) {
  const std::size_t size = 25;

  Vector<float> float_vec(size);
  Vector<double> double_vec(size);
  Vector<int> int_vec(size);

  EXPECT_EQ(float_vec.size(), size);
  EXPECT_EQ(double_vec.size(), size);
  EXPECT_EQ(int_vec.size(), size);
}

TEST(VectorTest, ConstCorrectness) {
  const std::size_t size = 40;
  const Vector<float> const_vec(size);

  [[maybe_unused]] const auto& data = const_vec.data();
  [[maybe_unused]] const auto& view = const_vec.view();

  EXPECT_EQ(const_vec.size(), size);
  EXPECT_EQ(data.size(), size);
  EXPECT_EQ(view.size(), size);
}

TEST(VectorTest, InstanceIndependence) {
  Vector<float> vec1(10);
  Vector<float> vec2(20);

  EXPECT_EQ(vec1.size(), 10);
  EXPECT_EQ(vec2.size(), 20);
  EXPECT_NE(vec1.size(), vec2.size());
}

class VectorSizeTest : public ::testing::TestWithParam<std::size_t> {};

TEST_P(VectorSizeTest, SizeParameterized) {
  const auto size = GetParam();
  Vector<int> vec(size);

  EXPECT_EQ(vec.size(), size);
  EXPECT_EQ(vec.data().size(), size);
}

INSTANTIATE_TEST_SUITE_P(
    VectorSizes, VectorSizeTest, ::testing::Values(0, 1, 10, 100, 1000));

TEST(VectorTest, FeatureTypeExists) {
  [[maybe_unused]] hsys::VectorView<float>::hsys_vector_view_feature feature;
  SUCCEED();
}
