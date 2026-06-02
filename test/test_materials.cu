// test/test_materials.cu

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "hittable.cuh" // Pour HitRecord
#include "material.cuh"
#include "cuda_utils.cuh"

__global__ void init_test_rand_state(curandState* rand_state)
{
    curand_init(1234, 0, 0, rand_state);
}

// Kernel pour tester un scatter
__global__ void scatter_test_kernel(
    const Ray r_in, const HitRecord rec, const MaterialData mat,
    bool* d_scattered, Ray* d_scattered_ray, float* d_attenuation, bool* d_is_delta, curandState* rand_state)
{
    *d_scattered = scatter(r_in, rec, mat, *d_attenuation, *d_scattered_ray, *d_is_delta, rand_state);
}

class MaterialTest : public ::testing::Test {
protected:
    bool* d_did_scatter;
    Ray* d_scattered_ray;
    float* d_attenuation;
    bool* d_is_delta;
    curandState* d_rand_state;

    bool h_did_scatter;
    Ray h_scattered_ray;
    float h_attenuation;
    bool h_is_delta;

    void SetUp() override {
        CHECK_CUDA_ERROR(cudaMalloc(&d_did_scatter, sizeof(bool)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_scattered_ray, sizeof(Ray)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_attenuation, sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_is_delta, sizeof(bool)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_rand_state, sizeof(curandState)));
        init_test_rand_state<<<1, 1>>>(d_rand_state);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    void TearDown() override {
        cudaFree(d_did_scatter);
        cudaFree(d_scattered_ray);
        cudaFree(d_attenuation);
        cudaFree(d_is_delta);
        cudaFree(d_rand_state);
    }

    void get_results() {
        CHECK_CUDA_ERROR(cudaMemcpy(&h_did_scatter, d_did_scatter, sizeof(bool), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(&h_scattered_ray, d_scattered_ray, sizeof(Ray), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(&h_attenuation, d_attenuation, sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(&h_is_delta, d_is_delta, sizeof(bool), cudaMemcpyDeviceToHost));
    }
};

TEST_F(MaterialTest, MetalScatterDeterministic) {
    // Scénario : réflexion parfaite sur un métal (fuzz = 0)
    Ray r_in(Point3(0, 1, 1), Vec3(0, 0, -1));
    HitRecord rec;
    rec.p = Point3(0, 1, 0);
    rec.normal = Vec3(0, 0, 1);
    MaterialData mat = {};
    mat.type = METAL;
    mat.albedo = Color(0.8f, 0.8f, 0.8f);
    mat.fuzz = 0.0f;
    mat.spectral_function_id = SPECTRAL_PROFILE_RGB_RECONSTRUCTION;

    scatter_test_kernel<<<1, 1>>>(r_in, rec, mat, d_did_scatter, d_scattered_ray, d_attenuation, d_is_delta, d_rand_state);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    get_results();

    ASSERT_TRUE(h_did_scatter);

    // Vérifier l'atténuation
    EXPECT_FLOAT_EQ(h_attenuation, 0.8f);
    EXPECT_TRUE(h_is_delta);

    // Vérifier le rayon réfléchi
    Vec3 expected_reflected_dir = reflect(normalize(r_in.direction), rec.normal);
    EXPECT_FLOAT_EQ(h_scattered_ray.origin.x, rec.p.x);
    EXPECT_FLOAT_EQ(h_scattered_ray.origin.y, rec.p.y);
    EXPECT_FLOAT_EQ(h_scattered_ray.origin.z, rec.p.z);
    EXPECT_FLOAT_EQ(h_scattered_ray.direction.x, expected_reflected_dir.x);
    EXPECT_FLOAT_EQ(h_scattered_ray.direction.y, expected_reflected_dir.y);
    EXPECT_FLOAT_EQ(h_scattered_ray.direction.z, expected_reflected_dir.z);
}

TEST_F(MaterialTest, DielectricTotalInternalReflection) {
    // Scénario : un rayon sortant du verre vers l'air avec un angle > angle critique
    Ray r_in(Point3(0, 0, 0), Vec3(1, 1, 0)); // Rayon à 45 degrés
    r_in.direction = normalize(r_in.direction);

    HitRecord rec;
    rec.p = Point3(0, 0, 0);
    rec.normal = Vec3(0, -1, 0); // Normale orientée contre le rayon sortant
    rec.front_face = false; // Le rayon vient de l'intérieur du matériau

    MaterialData mat = {};
    mat.type = DIELECTRIC;
    mat.ior = 1.5f; // Verre
    mat.dispersive = false;

    scatter_test_kernel<<<1, 1>>>(r_in, rec, mat, d_did_scatter, d_scattered_ray, d_attenuation, d_is_delta, d_rand_state);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    get_results();

    ASSERT_TRUE(h_did_scatter);

    // L'angle d'incidence est 45deg. sin(45) = 0.707.
    // L'indice de réfraction est 1.5. Loi de Snell: n1*sin(t1) = n2*sin(t2)
    // 1.5 * sin(45) = 1.0 * sin(t2) => 1.5 * 0.707 = 1.06 > 1.0.
    // Il doit y avoir une réflexion interne totale. Le rayon doit être réfléchi.
    Vec3 expected_reflected_dir = reflect(r_in.direction, rec.normal);
    EXPECT_FLOAT_EQ(h_scattered_ray.direction.x, expected_reflected_dir.x);
    EXPECT_FLOAT_EQ(h_scattered_ray.direction.y, expected_reflected_dir.y);
    EXPECT_NEAR(h_scattered_ray.direction.z, expected_reflected_dir.z, 1e-6);
    EXPECT_TRUE(h_is_delta);
}
