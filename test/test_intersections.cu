// test/test_intersections.cu

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "hittable.cuh"
#include "ray.cuh"
#include "cuda_utils.cuh"

// Kernel pour tester l'intersection avec une sphère
__global__ void sphere_hit_test_kernel(
    const Ray r, const SphereData sphere, bool* d_hit, HitRecord* d_rec)
{
    *d_hit = hit_sphere(r, 0.001f, FLT_MAX, *d_rec, sphere);
}


class IntersectionTest : public ::testing::Test {
protected:
    bool* d_hit_result;
    HitRecord* d_rec;
    HitRecord h_rec;

    void SetUp() override {
        // Allouer la mémoire sur le GPU pour les résultats
        CHECK_CUDA_ERROR(cudaMalloc(&d_hit_result, sizeof(bool)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_rec, sizeof(HitRecord)));
    }

    void TearDown() override {
        // Libérer la mémoire
        cudaFree(d_hit_result);
        cudaFree(d_rec);
    }

    // Fonction pour récupérer les résultats depuis le GPU
    bool get_hit_result() {
        bool hit;
        CHECK_CUDA_ERROR(cudaMemcpy(&hit, d_hit_result, sizeof(bool), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(&h_rec, d_rec, sizeof(HitRecord), cudaMemcpyDeviceToHost));
        return hit;
    }
};

TEST_F(IntersectionTest, SphereShouldHit) {
    // Scénario : un rayon pointant directement vers le centre d'une sphère
    SphereData sphere = { Point3(0, 0, -5), 1.0f, 0 };
    Ray ray(Point3(0, 0, 0), Vec3(0, 0, -1));

    sphere_hit_test_kernel<<<1, 1>>>(ray, sphere, d_hit_result, d_rec);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    bool hit = get_hit_result();

    ASSERT_TRUE(hit);
    EXPECT_FLOAT_EQ(h_rec.t, 4.0f); // Le rayon touche le devant de la sphère à z=-4
    EXPECT_FLOAT_EQ(h_rec.p.x, 0.0f);
    EXPECT_FLOAT_EQ(h_rec.p.y, 0.0f);
    EXPECT_FLOAT_EQ(h_rec.p.z, -4.0f);
    EXPECT_FLOAT_EQ(h_rec.normal.z, 1.0f); // La normale pointe vers l'origine
}

TEST_F(IntersectionTest, SphereShouldMiss) {
    // Scénario : un rayon parallèle à la sphère, qui ne la touche pas
    SphereData sphere = { Point3(0, 0, -5), 1.0f, 0 };
    Ray ray(Point3(2, 0, 0), Vec3(0, 0, -1));

    sphere_hit_test_kernel<<<1, 1>>>(ray, sphere, d_hit_result, d_rec);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    bool hit = get_hit_result();
    ASSERT_FALSE(hit);
}
