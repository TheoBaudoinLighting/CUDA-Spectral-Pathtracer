#ifndef HITTABLE_CUH
#define HITTABLE_CUH

#include "ray.cuh"
class Vec3;
struct MaterialData;

enum HittableType {
    SPHERE,
    TRIANGLE,
    RECTANGLE_XY,
    RECTANGLE_XZ,
    RECTANGLE_YZ,
    TRIANGULAR_PRISM,
    BOX
};

struct HitRecord {
    Point3 p;
    Vec3 normal;
    float t;
    float u, v;
    bool front_face;
    int material_id;
    int object_id;

    __device__ inline void set_face_normal(const Ray& r, const Vec3& outward_normal) {
        front_face = dot(r.direction, outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

#include "material.cuh"

struct SphereData {
    Point3 center;
    float radius;
    int material_id;
};

struct TriangleData {
    Point3 v0, v1, v2;
    Vec3 normal;
    int material_id;
};

struct RectangleXYData {
    float x0, x1, y0, y1, k;
    int material_id;
};

struct RectangleXZData {
    float x0, x1, z0, z1, k;
    int material_id;
};

struct RectangleYZData {
    float y0, y1, z0, z1, k;
    int material_id;
};

struct TriangularPrismData {
    Point3 base_center;
    float base_size;
    float height;
    int material_id;
};

struct BoxData {
    Point3 min;
    Point3 max;
    int material_id;
};

struct HittableData {
    HittableType type;
    int data_index;
};

__device__ inline bool hit_sphere(
    const Ray& r,
    float t_min,
    float t_max,
    HitRecord& rec,
    const SphereData& sphere
) {
    Vec3 oc = r.origin - sphere.center;
    auto a = r.direction.length_squared();
    auto half_b = dot(oc, r.direction);
    auto c = oc.length_squared() - sphere.radius * sphere.radius;

    auto discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return false;

    auto sqrtd = sqrtf(discriminant);

    auto root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root) {
            return false;
        }
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    Vec3 outward_normal = (rec.p - sphere.center) / sphere.radius;
    rec.set_face_normal(r, outward_normal);

    auto phi = atan2f(outward_normal.z, outward_normal.x);
    auto theta = asinf(outward_normal.y);
    rec.u = 1.0f - (phi + M_PI) / (2.0f * M_PI);
    rec.v = (theta + M_PI / 2.0f) / M_PI;

    rec.material_id = sphere.material_id;

    return true;
}

__device__ inline bool hit_triangle(
    const Ray& r,
    float t_min,
    float t_max,
    HitRecord& rec,
    const TriangleData& triangle
) {
    const float EPSILON = 0.0000001f;

    Vec3 edge1 = triangle.v1 - triangle.v0;
    Vec3 edge2 = triangle.v2 - triangle.v0;
    Vec3 h = cross(r.direction, edge2);
    float a = dot(edge1, h);

    if (a > -EPSILON && a < EPSILON) return false;

    float f = 1.0f / a;
    Vec3 s = r.origin - triangle.v0;
    float u = f * dot(s, h);

    if (u < 0.0f || u > 1.0f) return false;

    Vec3 q = cross(s, edge1);
    float v = f * dot(r.direction, q);

    if (v < 0.0f || u + v > 1.0f) return false;

    float t = f * dot(edge2, q);

    if (t < t_min || t > t_max) return false;

    rec.t = t;
    rec.p = r.at(t);
    rec.set_face_normal(r, triangle.normal);
    rec.u = u;
    rec.v = v;
    rec.material_id = triangle.material_id;

    return true;
}

__device__ inline bool hit_rectangle_xy(
    const Ray& r,
    float t_min,
    float t_max,
    HitRecord& rec,
    const RectangleXYData& rect
) {
    auto t = (rect.k - r.origin.z) / r.direction.z;
    if (t < t_min || t > t_max) return false;

    auto x = r.origin.x + t * r.direction.x;
    auto y = r.origin.y + t * r.direction.y;

    if (x < rect.x0 || x > rect.x1 || y < rect.y0 || y > rect.y1) return false;

    rec.u = (x - rect.x0) / (rect.x1 - rect.x0);
    rec.v = (y - rect.y0) / (rect.y1 - rect.y0);
    rec.t = t;

    Vec3 outward_normal = Vec3(0, 0, 1);
    rec.set_face_normal(r, outward_normal);
    rec.p = r.at(t);
    rec.material_id = rect.material_id;

    return true;
}

__device__ inline bool hit_rectangle_xz(
    const Ray& r,
    float t_min,
    float t_max,
    HitRecord& rec,
    const RectangleXZData& rect
) {
    auto t = (rect.k - r.origin.y) / r.direction.y;
    if (t < t_min || t > t_max) return false;

    auto x = r.origin.x + t * r.direction.x;
    auto z = r.origin.z + t * r.direction.z;

    if (x < rect.x0 || x > rect.x1 || z < rect.z0 || z > rect.z1) return false;

    rec.u = (x - rect.x0) / (rect.x1 - rect.x0);
    rec.v = (z - rect.z0) / (rect.z1 - rect.z0);
    rec.t = t;

    Vec3 outward_normal = Vec3(0, 1, 0);
    rec.set_face_normal(r, outward_normal);
    rec.p = r.at(t);
    rec.material_id = rect.material_id;

    return true;
}

__device__ inline bool hit_rectangle_yz(
    const Ray& r,
    float t_min,
    float t_max,
    HitRecord& rec,
    const RectangleYZData& rect
) {
    auto t = (rect.k - r.origin.x) / r.direction.x;
    if (t < t_min || t > t_max) return false;

    auto y = r.origin.y + t * r.direction.y;
    auto z = r.origin.z + t * r.direction.z;

    if (y < rect.y0 || y > rect.y1 || z < rect.z0 || z > rect.z1) return false;

    rec.u = (y - rect.y0) / (rect.y1 - rect.y0);
    rec.v = (z - rect.z0) / (rect.z1 - rect.z0);
    rec.t = t;

    Vec3 outward_normal = Vec3(1, 0, 0);
    rec.set_face_normal(r, outward_normal);
    rec.p = r.at(t);
    rec.material_id = rect.material_id;

    return true;
}

__device__ inline bool hit_objects(
    const Ray& r,
    float t_min,
    float t_max,
    HitRecord& rec,
    const HittableData* objects,
    int num_objects,
    const SphereData* spheres,
    const TriangleData* triangles,
    const RectangleXYData* rect_xy,
    const RectangleXZData* rect_xz,
    const RectangleYZData* rect_yz
) {
    HitRecord temp_rec;
    bool hit_anything = false;
    auto closest_so_far = t_max;

    for (int i = 0; i < num_objects; i++) {
        const HittableData& object = objects[i];
        bool hit_object = false;

        switch (object.type) {
            case SPHERE:
                hit_object = hit_sphere(r, t_min, closest_so_far, temp_rec, spheres[object.data_index]);
                break;
            case TRIANGLE:
                hit_object = hit_triangle(r, t_min, closest_so_far, temp_rec, triangles[object.data_index]);
                break;
            case RECTANGLE_XY:
                hit_object = hit_rectangle_xy(r, t_min, closest_so_far, temp_rec, rect_xy[object.data_index]);
                break;
            case RECTANGLE_XZ:
                hit_object = hit_rectangle_xz(r, t_min, closest_so_far, temp_rec, rect_xz[object.data_index]);
                break;
            case RECTANGLE_YZ:
                hit_object = hit_rectangle_yz(r, t_min, closest_so_far, temp_rec, rect_yz[object.data_index]);
                break;
            default:
                break;
        }

        if (hit_object) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            temp_rec.object_id = i;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

#endif // HITTABLE_CUH
