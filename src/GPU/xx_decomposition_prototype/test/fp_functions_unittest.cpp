#include <gtest/gtest.h>


extern "C" int FloatCompare(float, float);

TEST(FPFunctionsFloatCompare, OneEqualsOne) {
  EXPECT_EQ(FloatCompare(1.0f, 1.0f), 0);
}
