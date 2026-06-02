#ifndef VEC3_CUH
#define VEC3_CUH

#include <cuda_runtime.h>
#include <math.h>
#include <curand_kernel.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

class Vec3 {
public:
    float x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    __host__ __device__ inline Vec3 operator-() const { return Vec3(-x, -y, -z); }
    __host__ __device__ inline Vec3& operator+=(const Vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    __host__ __device__ inline Vec3& operator*=(float t) { x *= t; y *= t; z *= t; return *this; }
    __host__ __device__ inline Vec3& operator/=(float t) { return *this *= 1/t; }
    
    __host__ __device__ inline Vec3& operator*=(const Vec3& v) { x *= v.x; y *= v.y; z *= v.z; return *this; }

    __host__ __device__ inline float length() const { return sqrtf(length_squared()); }
    __host__ __device__ inline float length_squared() const { return x*x + y*y + z*z; }
    __host__ __device__ inline bool near_zero() const {
        constexpr float s = 1e-8f;
        return (fabsf(x) < s) && (fabsf(y) < s) && (fabsf(z) < s);
    }

    __device__ static Vec3 random(curandState* state) {
        return Vec3(curand_uniform(state), curand_uniform(state), curand_uniform(state));
    }

    __device__ static Vec3 random(curandState* state, float min, float max) {
        return Vec3(
            min + (max-min)*curand_uniform(state),
            min + (max-min)*curand_uniform(state),
            min + (max-min)*curand_uniform(state)
        );
    }
};

using Point3 = Vec3;
using Color = Vec3;

__host__ __device__ inline Vec3 operator+(const Vec3& u, const Vec3& v) {
    return Vec3(u.x + v.x, u.y + v.y, u.z + v.z);
}

__host__ __device__ inline Vec3 operator-(const Vec3& u, const Vec3& v) {
    return Vec3(u.x - v.x, u.y - v.y, u.z - v.z);
}

__host__ __device__ inline Vec3 operator*(const Vec3& u, const Vec3& v) {
    return Vec3(u.x * v.x, u.y * v.y, u.z * v.z);
}

__host__ __device__ inline Vec3 operator*(float t, const Vec3& v) {
    return Vec3(t * v.x, t * v.y, t * v.z);
}

__host__ __device__ inline Vec3 operator*(const Vec3& v, float t) {
    return t * v;
}

__host__ __device__ inline Vec3 operator/(const Vec3& v, float t) {
    return (1/t) * v;
}

__host__ __device__ inline float dot(const Vec3& u, const Vec3& v) {
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

__host__ __device__ inline Vec3 cross(const Vec3& u, const Vec3& v) {
    return Vec3(u.y * v.z - u.z * v.y,
                u.z * v.x - u.x * v.z,
                u.x * v.y - u.y * v.x);
}

__host__ __device__ inline Vec3 normalize(const Vec3& v) {
    return v / v.length();
}

__device__ inline Vec3 random_in_unit_sphere(curandState* state) {
    while (true) {
        auto p = Vec3::random(state, -1, 1);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

__device__ inline Vec3 random_unit_vector(curandState* state) {
    return normalize(random_in_unit_sphere(state));
}

__device__ inline Vec3 random_cosine_direction(const Vec3& normal, curandState* state) {
    float r1 = curand_uniform(state);
    float r2 = curand_uniform(state);
    float phi = 2.0f * M_PI * r1;
    float sqrt_r2 = sqrtf(r2);

    Vec3 w = normalize(normal);
    Vec3 a = fabsf(w.x) > 0.9f ? Vec3(0.0f, 1.0f, 0.0f) : Vec3(1.0f, 0.0f, 0.0f);
    Vec3 v = normalize(cross(w, a));
    Vec3 u = cross(v, w);

    return normalize(
        cosf(phi) * sqrt_r2 * u +
        sinf(phi) * sqrt_r2 * v +
        sqrtf(fmaxf(0.0f, 1.0f - r2)) * w
    );
}

__device__ inline Vec3 random_in_unit_disk(curandState* state) {
    while (true) {
        auto p = Vec3(
            2.0f*curand_uniform(state)-1.0f,
            2.0f*curand_uniform(state)-1.0f,
            0.0f
        );
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

__host__ __device__ inline Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - 2*dot(v,n)*n;
}

__host__ __device__ inline Vec3 refract(const Vec3& uv, const Vec3& n, float etai_over_etat) {
    auto cos_theta = fminf(dot(-uv, n), 1.0f);
    Vec3 r_out_perp = etai_over_etat * (uv + cos_theta*n);
    Vec3 r_out_parallel = -sqrtf(fabsf(1.0f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

#endif // VEC3_CUH
