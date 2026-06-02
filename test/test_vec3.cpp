// test/test_vec3.cpp

#include <gtest/gtest.h>
#include "vec3.cuh"

// Teste les opérations de base sur Vec3
TEST(Vec3Test, BasicOperations) {
    Vec3 v1(1.0f, 2.0f, 3.0f);
    Vec3 v2(4.0f, 5.0f, 6.0f);

    Vec3 v_add = v1 + v2;
    EXPECT_FLOAT_EQ(v_add.x, 5.0f);
    EXPECT_FLOAT_EQ(v_add.y, 7.0f);
    EXPECT_FLOAT_EQ(v_add.z, 9.0f);

    Vec3 v_sub = v2 - v1;
    EXPECT_FLOAT_EQ(v_sub.x, 3.0f);
    EXPECT_FLOAT_EQ(v_sub.y, 3.0f);
    EXPECT_FLOAT_EQ(v_sub.z, 3.0f);

    Vec3 v_mul = v1 * 2.0f;
    EXPECT_FLOAT_EQ(v_mul.x, 2.0f);
    EXPECT_FLOAT_EQ(v_mul.y, 4.0f);
    EXPECT_FLOAT_EQ(v_mul.z, 6.0f);
}

// Teste les produits scalaire et vectoriel
TEST(Vec3Test, Products) {
    Vec3 v1(1.0f, 0.0f, 0.0f);
    Vec3 v2(0.0f, 1.0f, 0.0f);

    EXPECT_FLOAT_EQ(dot(v1, v2), 0.0f);
    EXPECT_FLOAT_EQ(dot(v1, v1), 1.0f);

    Vec3 v_cross = cross(v1, v2);
    EXPECT_FLOAT_EQ(v_cross.x, 0.0f);
    EXPECT_FLOAT_EQ(v_cross.y, 0.0f);
    EXPECT_FLOAT_EQ(v_cross.z, 1.0f);
}

// Teste la longueur et la normalisation
TEST(Vec3Test, LengthAndNormalization) {
    Vec3 v(3.0f, 4.0f, 0.0f);
    EXPECT_FLOAT_EQ(v.length_squared(), 25.0f);
    EXPECT_FLOAT_EQ(v.length(), 5.0f);

    Vec3 v_norm = normalize(v);
    EXPECT_FLOAT_EQ(v_norm.x, 0.6f);
    EXPECT_FLOAT_EQ(v_norm.y, 0.8f);
    EXPECT_FLOAT_EQ(v_norm.z, 0.0f);
    EXPECT_FLOAT_EQ(v_norm.length(), 1.0f);
}
