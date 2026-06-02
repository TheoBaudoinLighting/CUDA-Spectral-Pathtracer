// test/test_camera.cpp

#include <gtest/gtest.h>
#include "camera.cuh"

TEST(CameraTest, Constructor) {
    Point3 lookfrom(0, 0, 1);
    Point3 lookat(0, 0, 0);
    Vec3 vup(0, 1, 0);
    float vfov = 90.0f;
    float aspect_ratio = 1.0f;

    Camera cam(lookfrom, lookat, vup, vfov, aspect_ratio, 0.0f, 1.0f);

    // w doit pointer de lookat vers lookfrom
    EXPECT_FLOAT_EQ(cam.w.x, 0.0f);
    EXPECT_FLOAT_EQ(cam.w.y, 0.0f);
    EXPECT_FLOAT_EQ(cam.w.z, 1.0f);

    // u doit être horizontal (axe X)
    EXPECT_FLOAT_EQ(cam.u.x, 1.0f);
    EXPECT_FLOAT_EQ(cam.u.y, 0.0f);
    EXPECT_FLOAT_EQ(cam.u.z, 0.0f);

    // v doit être vertical (axe Y)
    EXPECT_FLOAT_EQ(cam.v.x, 0.0f);
    EXPECT_FLOAT_EQ(cam.v.y, 1.0f);
    EXPECT_FLOAT_EQ(cam.v.z, 0.0f);

    // Test du lower_left_corner
    // viewport_height = 2 * tan(theta/2) = 2 * tan(45deg) = 2.0
    // viewport_width = aspect_ratio * viewport_height = 1.0 * 2.0 = 2.0
    // horizontal = focus_dist * viewport_width * u = 1.0 * 2.0 * (1,0,0) = (2,0,0)
    // vertical = focus_dist * viewport_height * v = 1.0 * 2.0 * (0,1,0) = (0,2,0)
    // lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist*w
    //                   = (0,0,1) - (1,0,0) - (0,1,0) - (0,0,1) = (-1, -1, 0)
    EXPECT_FLOAT_EQ(cam.lower_left_corner.x, -1.0f);
    EXPECT_FLOAT_EQ(cam.lower_left_corner.y, -1.0f);
    EXPECT_FLOAT_EQ(cam.lower_left_corner.z, 0.0f);
}