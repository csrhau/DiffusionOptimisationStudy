#include <gtest/gtest.h>

#include "fp_functions.h"

TEST(FPFunctionsFloatCompare, OneEqualsOne) {
  EXPECT_EQ(FloatCompare(1.0f, 1.0f), 0);
}
