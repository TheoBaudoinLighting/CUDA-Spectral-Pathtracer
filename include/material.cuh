#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include "ray.cuh"
#include "spectrum.cuh"
#include <curand_kernel.h>

struct HitRecord;

enum MaterialType {
    LAMBERTIAN,
    METAL,
    DIELECTRIC,
    EMISSIVE,
    SPECTRAL
};

struct MaterialData {
    MaterialType type;

    Color albedo;
    float fuzz;
    float ior;
    bool dispersive;
    Color emission;
    int spectral_function_id;

    int emission_spectrum_id;
    float dispersion_A;
    float dispersion_B;
    float dispersion_C;
    float absorption_density;
    Color absorption_color;
    int absorption_spectrum_id;
};

__host__ __device__ inline float material_ior_at_wavelength(const MaterialData& mat_data, float wavelength) {
    if (mat_data.dispersive) {
        return cauchy_dispersion(wavelength, mat_data.dispersion_A, mat_data.dispersion_B, mat_data.dispersion_C);
    }
    return mat_data.ior;
}

__host__ __device__ inline float material_reflectance_at_wavelength(const MaterialData& mat_data, float wavelength) {
    return evaluate_spectral_profile(mat_data.spectral_function_id, mat_data.albedo, wavelength);
}

__host__ __device__ inline float material_emission_at_wavelength(const MaterialData& mat_data, float wavelength) {
    if (mat_data.type != EMISSIVE) {
        return 0.0f;
    }

    if (mat_data.emission_spectrum_id != SPECTRAL_PROFILE_RGB_RECONSTRUCTION) {
        float strength = fmaxf(mat_data.emission.x, fmaxf(mat_data.emission.y, mat_data.emission.z));
        return strength * evaluate_spectral_profile(mat_data.emission_spectrum_id, Color(1.0f, 1.0f, 1.0f), wavelength);
    }

    return rgb_illuminant_to_spectrum(mat_data.emission, wavelength);
}

__host__ __device__ inline float material_absorption_at_wavelength(const MaterialData& mat_data, float wavelength) {
    if (mat_data.absorption_density <= 0.0f) {
        return 0.0f;
    }

    float profile = evaluate_spectral_profile(mat_data.absorption_spectrum_id, mat_data.absorption_color, wavelength);
    return fmaxf(0.0f, mat_data.absorption_density * profile);
}

__device__ inline float schlick_reflectance(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
}

__device__ inline bool scatter_lambertian(
    const Ray& r_in,
    const HitRecord& rec,
    const MaterialData& mat_data,
    float& attenuation,
    Ray& scattered,
    bool& is_delta,
    curandState* rand_state
) {
    Vec3 scatter_direction = random_cosine_direction(rec.normal, rand_state);
    scattered = Ray(rec.p, scatter_direction, r_in.wavelength);
    attenuation = material_reflectance_at_wavelength(mat_data, r_in.wavelength);
    is_delta = false;
    return attenuation > 0.0f;
}

__device__ inline bool scatter_metal(
    const Ray& r_in,
    const HitRecord& rec,
    const MaterialData& mat_data,
    float& attenuation,
    Ray& scattered,
    bool& is_delta,
    curandState* rand_state
) {
    Vec3 reflected = reflect(normalize(r_in.direction), rec.normal);
    scattered = Ray(
        rec.p,
        reflected + mat_data.fuzz * random_in_unit_sphere(rand_state),
        r_in.wavelength
    );
    attenuation = material_reflectance_at_wavelength(mat_data, r_in.wavelength);
    is_delta = true;
    return attenuation > 0.0f && dot(scattered.direction, rec.normal) > 0.0f;
}

__device__ inline bool scatter_dielectric(
    const Ray& r_in,
    const HitRecord& rec,
    const MaterialData& mat_data,
    float& attenuation,
    Ray& scattered,
    bool& is_delta,
    curandState* rand_state
) {
    float index = material_ior_at_wavelength(mat_data, r_in.wavelength);
    float refraction_ratio = rec.front_face ? (1.0f / index) : index;

    Vec3 unit_direction = normalize(r_in.direction);
    float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0f);
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));

    bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
    Vec3 direction;

    if (cannot_refract || schlick_reflectance(cos_theta, refraction_ratio) > curand_uniform(rand_state)) {
        direction = reflect(unit_direction, rec.normal);
    } else {
        direction = refract(unit_direction, rec.normal, refraction_ratio);
    }

    float sigma_a = material_absorption_at_wavelength(mat_data, r_in.wavelength);
    attenuation = rec.front_face ? 1.0f : expf(-sigma_a * fmaxf(rec.t, 0.0f));
    scattered = Ray(rec.p, direction, r_in.wavelength);
    is_delta = true;
    return true;
}

__device__ inline bool scatter_emissive(
    const Ray& r_in,
    const HitRecord& rec,
    const MaterialData& mat_data,
    float& attenuation,
    Ray& scattered,
    bool& is_delta,
    curandState* rand_state
) {
    (void)r_in;
    (void)rec;
    (void)mat_data;
    (void)scattered;
    (void)rand_state;
    attenuation = 0.0f;
    is_delta = false;
    return false;
}

__device__ inline bool scatter_spectral(
    const Ray& r_in,
    const HitRecord& rec,
    const MaterialData& mat_data,
    float& attenuation,
    Ray& scattered,
    bool& is_delta,
    curandState* rand_state
) {
    Vec3 scatter_direction = random_cosine_direction(rec.normal, rand_state);
    scattered = Ray(rec.p, scatter_direction, r_in.wavelength);
    attenuation = material_reflectance_at_wavelength(mat_data, r_in.wavelength);
    is_delta = false;
    return attenuation > 0.0f;
}

__device__ inline bool scatter(
    const Ray& r_in,
    const HitRecord& rec,
    const MaterialData& mat_data,
    float& attenuation,
    Ray& scattered,
    bool& is_delta,
    curandState* rand_state
) {
    switch (mat_data.type) {
        case LAMBERTIAN:
            return scatter_lambertian(r_in, rec, mat_data, attenuation, scattered, is_delta, rand_state);
        case METAL:
            return scatter_metal(r_in, rec, mat_data, attenuation, scattered, is_delta, rand_state);
        case DIELECTRIC:
            return scatter_dielectric(r_in, rec, mat_data, attenuation, scattered, is_delta, rand_state);
        case EMISSIVE:
            return scatter_emissive(r_in, rec, mat_data, attenuation, scattered, is_delta, rand_state);
        case SPECTRAL:
            return scatter_spectral(r_in, rec, mat_data, attenuation, scattered, is_delta, rand_state);
        default:
            return false;
    }
}

__device__ inline float emit(const MaterialData& mat_data, float u, float v, const Point3& p, float wavelength) {
    (void)u;
    (void)v;
    (void)p;
    return material_emission_at_wavelength(mat_data, wavelength);
}

#endif // MATERIAL_CUH
