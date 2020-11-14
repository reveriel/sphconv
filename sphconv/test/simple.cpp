#include <gtest/gtest.h>

TEST(Simple, simple) {
    EXPECT_TRUE(true);
}

TEST(Simple, fail) {
    EXPECT_TRUE(false);
}