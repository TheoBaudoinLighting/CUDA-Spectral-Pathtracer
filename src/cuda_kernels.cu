#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <cfloat>
#include <stdio.h>

#include "vec3.cuh"
#include "ray.cuh"
#include "hittable.cuh"
#include "material.cuh"
#include "camera.cuh"
#include "spectrum.cuh"
#include "cuda_utils.cuh"

__global__ void init_rand_state(curandState* rand_state, int width, int height, unsigned long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    int pixel_index = j * width + i;
    curand_init(seed, pixel_index, 0, &rand_state[pixel_index]);
}

__device__ inline int material_id_for_object(
    const HittableData& object,
    const SphereData* spheres,
    const TriangleData* triangles,
    const RectangleXYData* rect_xy,
    const RectangleXZData* rect_xz,
    const RectangleYZData* rect_yz
) {
    switch (object.type) {
        case SPHERE:
            return spheres[object.data_index].material_id;
        case TRIANGLE:
            return triangles[object.data_index].material_id;
        case RECTANGLE_XY:
            return rect_xy[object.data_index].material_id;
        case RECTANGLE_XZ:
            return rect_xz[object.data_index].material_id;
        case RECTANGLE_YZ:
            return rect_yz[object.data_index].material_id;
        default:
            return -1;
    }
}

__device__ inline bool sample_object_surface(
    const HittableData& object,
    const SphereData* spheres,
    const TriangleData* triangles,
    const RectangleXYData* rect_xy,
    const RectangleXZData* rect_xz,
    const RectangleYZData* rect_yz,
    Point3& position,
    Vec3& normal,
    float& area,
    curandState* rand_state
) {
    switch (object.type) {
        case SPHERE: {
            const SphereData& sphere = spheres[object.data_index];
            normal = random_unit_vector(rand_state);
            position = sphere.center + sphere.radius * normal;
            area = 4.0f * M_PI * sphere.radius * sphere.radius;
            return area > 0.0f;
        }
        case TRIANGLE: {
            const TriangleData& triangle = triangles[object.data_index];
            float r1 = curand_uniform(rand_state);
            float r2 = curand_uniform(rand_state);
            float sr1 = sqrtf(r1);
            float b0 = 1.0f - sr1;
            float b1 = sr1 * (1.0f - r2);
            float b2 = sr1 * r2;
            position = b0 * triangle.v0 + b1 * triangle.v1 + b2 * triangle.v2;
            normal = triangle.normal;
            area = 0.5f * cross(triangle.v1 - triangle.v0, triangle.v2 - triangle.v0).length();
            return area > 0.0f;
        }
        case RECTANGLE_XY: {
            const RectangleXYData& rect = rect_xy[object.data_index];
            float x = rect.x0 + curand_uniform(rand_state) * (rect.x1 - rect.x0);
            float y = rect.y0 + curand_uniform(rand_state) * (rect.y1 - rect.y0);
            position = Point3(x, y, rect.k);
            normal = Vec3(0.0f, 0.0f, 1.0f);
            area = fabsf((rect.x1 - rect.x0) * (rect.y1 - rect.y0));
            return area > 0.0f;
        }
        case RECTANGLE_XZ: {
            const RectangleXZData& rect = rect_xz[object.data_index];
            float x = rect.x0 + curand_uniform(rand_state) * (rect.x1 - rect.x0);
            float z = rect.z0 + curand_uniform(rand_state) * (rect.z1 - rect.z0);
            position = Point3(x, rect.k, z);
            normal = Vec3(0.0f, 1.0f, 0.0f);
            area = fabsf((rect.x1 - rect.x0) * (rect.z1 - rect.z0));
            return area > 0.0f;
        }
        case RECTANGLE_YZ: {
            const RectangleYZData& rect = rect_yz[object.data_index];
            float y = rect.y0 + curand_uniform(rand_state) * (rect.y1 - rect.y0);
            float z = rect.z0 + curand_uniform(rand_state) * (rect.z1 - rect.z0);
            position = Point3(rect.k, y, z);
            normal = Vec3(1.0f, 0.0f, 0.0f);
            area = fabsf((rect.y1 - rect.y0) * (rect.z1 - rect.z0));
            return area > 0.0f;
        }
        default:
            return false;
    }
}

__device__ inline float object_surface_area(
    const HittableData& object,
    const SphereData* spheres,
    const TriangleData* triangles,
    const RectangleXYData* rect_xy,
    const RectangleXZData* rect_xz,
    const RectangleYZData* rect_yz
) {
    switch (object.type) {
        case SPHERE: {
            const SphereData& sphere = spheres[object.data_index];
            return 4.0f * M_PI * sphere.radius * sphere.radius;
        }
        case TRIANGLE: {
            const TriangleData& triangle = triangles[object.data_index];
            return 0.5f * cross(triangle.v1 - triangle.v0, triangle.v2 - triangle.v0).length();
        }
        case RECTANGLE_XY: {
            const RectangleXYData& rect = rect_xy[object.data_index];
            return fabsf((rect.x1 - rect.x0) * (rect.y1 - rect.y0));
        }
        case RECTANGLE_XZ: {
            const RectangleXZData& rect = rect_xz[object.data_index];
            return fabsf((rect.x1 - rect.x0) * (rect.z1 - rect.z0));
        }
        case RECTANGLE_YZ: {
            const RectangleYZData& rect = rect_yz[object.data_index];
            return fabsf((rect.y1 - rect.y0) * (rect.z1 - rect.z0));
        }
        default:
            return 0.0f;
    }
}

__device__ inline float power_heuristic(float pdf_a, float pdf_b) {
    float a2 = pdf_a * pdf_a;
    float b2 = pdf_b * pdf_b;
    float denom = a2 + b2;
    return denom > 0.0f ? a2 / denom : 0.0f;
}

__device__ inline float light_pdf_solid_angle(
    const HitRecord& light_hit,
    const Ray& ray_to_light,
    const HittableData* objects,
    const SphereData* spheres,
    const TriangleData* triangles,
    const RectangleXYData* rect_xy,
    const RectangleXZData* rect_xz,
    const RectangleYZData* rect_yz
) {
    if (light_hit.object_id < 0) {
        return 0.0f;
    }

    float area = object_surface_area(
        objects[light_hit.object_id],
        spheres,
        triangles,
        rect_xy,
        rect_xz,
        rect_yz
    );
    if (area <= 0.0f) {
        return 0.0f;
    }

    float distance_squared = ray_to_light.direction.length_squared() * light_hit.t * light_hit.t;
    Vec3 wi = normalize(ray_to_light.direction);
    float cos_light = fmaxf(0.0f, dot(light_hit.normal, -wi));
    if (cos_light <= 0.0f) {
        return 0.0f;
    }

    return distance_squared / (cos_light * area);
}

__device__ inline bool supports_direct_lighting(const MaterialData& mat_data) {
    return mat_data.type == LAMBERTIAN || mat_data.type == SPECTRAL;
}

__device__ float estimate_direct_lighting(
    const HitRecord& rec,
    const MaterialData& surface_material,
    const HittableData* objects,
    int num_objects,
    const SphereData* spheres,
    const TriangleData* triangles,
    const RectangleXYData* rect_xy,
    const RectangleXZData* rect_xz,
    const RectangleYZData* rect_yz,
    const MaterialData* materials,
    float wavelength,
    curandState* local_rand_state
) {
    if (!supports_direct_lighting(surface_material)) {
        return 0.0f;
    }

    float direct = 0.0f;
    float reflectance = material_reflectance_at_wavelength(surface_material, wavelength);
    if (reflectance <= 0.0f) {
        return 0.0f;
    }

    for (int light_index = 0; light_index < num_objects; ++light_index) {
        const HittableData& object = objects[light_index];
        int material_id = material_id_for_object(object, spheres, triangles, rect_xy, rect_xz, rect_yz);
        if (material_id < 0 || materials[material_id].type != EMISSIVE) {
            continue;
        }

        Point3 light_position;
        Vec3 light_normal;
        float light_area;
        if (!sample_object_surface(
                object,
                spheres,
                triangles,
                rect_xy,
                rect_xz,
                rect_yz,
                light_position,
                light_normal,
                light_area,
                local_rand_state
            )) {
            continue;
        }

        Vec3 to_light = light_position - rec.p;
        float distance_squared = to_light.length_squared();
        if (distance_squared <= 1.0e-8f) {
            continue;
        }

        float distance = sqrtf(distance_squared);
        Vec3 wi = to_light / distance;
        float cos_surface = dot(rec.normal, wi);
        float cos_light = fabsf(dot(light_normal, -wi));
        if (cos_surface <= 0.0f || cos_light <= 0.0f) {
            continue;
        }

        Ray shadow_ray(rec.p + 0.001f * rec.normal, wi, wavelength);
        HitRecord shadow_rec;
        if (hit_objects(
                shadow_ray,
                0.001f,
                fmaxf(0.001f, distance - 0.002f),
                shadow_rec,
                objects,
                num_objects,
                spheres,
                triangles,
                rect_xy,
                rect_xz,
                rect_yz
            )) {
            continue;
        }

        float li = emit(materials[material_id], 0.0f, 0.0f, light_position, wavelength);
        float lambertian_brdf = reflectance / M_PI;
        float light_pdf = distance_squared / fmaxf(cos_light * light_area, 1.0e-8f);
        float bsdf_pdf = cos_surface / M_PI;
        float mis_weight = power_heuristic(light_pdf, bsdf_pdf);
        direct += mis_weight * li * lambertian_brdf * cos_surface * cos_light * light_area / distance_squared;
    }

    return direct;
}

__device__ inline float background_radiance(float wavelength, const Vec3& direction) {
    Vec3 unit_direction = normalize(direction);
    float t = 0.5f * (unit_direction.y + 1.0f);
    Color background_rgb = (1.0f - t) * Color(0.04f, 0.04f, 0.04f) + t * Color(0.35f, 0.45f, 0.70f);
    return rgb_illuminant_to_spectrum(background_rgb, wavelength);
}

__device__ float ray_spectral_radiance(
    const Ray& r,
    const HittableData* objects,
    int num_objects,
    const SphereData* spheres,
    const TriangleData* triangles,
    const RectangleXYData* rect_xy,
    const RectangleXZData* rect_xz,
    const RectangleYZData* rect_yz,
    const MaterialData* materials,
    int depth,
    curandState* local_rand_state
) {
    Ray cur_ray = r;
    float throughput = 1.0f;
    float radiance = 0.0f;
    bool previous_bounce_delta = true;
    float previous_bsdf_pdf = 0.0f;

    for (int bounce = 0; bounce < depth; ++bounce) {
        HitRecord rec;

        if (hit_objects(
                cur_ray, 0.001f, FLT_MAX, rec,
                objects, num_objects,
                spheres, triangles, rect_xy, rect_xz, rect_yz
            )) {

            const MaterialData& material = materials[rec.material_id];
            float emitted = emit(material, rec.u, rec.v, rec.p, cur_ray.wavelength);
            if (emitted > 0.0f) {
                if (bounce == 0 || previous_bounce_delta) {
                    radiance += throughput * emitted;
                } else {
                    float light_pdf = light_pdf_solid_angle(
                        rec,
                        cur_ray,
                        objects,
                        spheres,
                        triangles,
                        rect_xy,
                        rect_xz,
                        rect_yz
                    );
                    radiance += throughput * emitted * power_heuristic(previous_bsdf_pdf, light_pdf);
                }
            }

            radiance += throughput * estimate_direct_lighting(
                rec,
                material,
                objects,
                num_objects,
                spheres,
                triangles,
                rect_xy,
                rect_xz,
                rect_yz,
                materials,
                cur_ray.wavelength,
                local_rand_state
            );

            Ray scattered;
            float attenuation;
            bool is_delta;

            if (scatter(cur_ray, rec, material, attenuation, scattered, is_delta, local_rand_state)) {
                throughput *= attenuation;
                previous_bsdf_pdf = 0.0f;
                if (!is_delta && supports_direct_lighting(material)) {
                    float cos_scatter = fmaxf(0.0f, dot(rec.normal, normalize(scattered.direction)));
                    previous_bsdf_pdf = cos_scatter / M_PI;
                }
                cur_ray = scattered;
                previous_bounce_delta = is_delta;
            } else {
                return radiance;
            }
        } else {
            return radiance + throughput * background_radiance(cur_ray.wavelength, cur_ray.direction);
        }

        if (bounce >= 3) {
            float survival_probability = fminf(0.95f, fmaxf(0.05f, throughput));
            if (curand_uniform(local_rand_state) > survival_probability) {
                return radiance;
            }
            throughput /= survival_probability;
        }

        if (throughput <= 0.0f) {
            return radiance;
        }
    }

    return radiance;
}

__global__ void render_kernel(
    Color* framebuffer,
    int width,
    int height,
    int samples_per_pixel,
    int max_depth,
    int spectral_samples,
    float lambda_min,
    float lambda_max,
    Camera* camera,
    HittableData* objects,
    int num_objects,
    SphereData* spheres,
    TriangleData* triangles,
    RectangleXYData* rect_xy,
    RectangleXZData* rect_xz,
    RectangleYZData* rect_yz,
    MaterialData* materials,
    curandState* rand_state
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    int pixel_index = j * width + i;
    curandState local_rand_state = rand_state[pixel_index];

    Color pixel_color(0.0f, 0.0f, 0.0f);
    int wavelength_samples = spectral_samples > 0 ? spectral_samples : 1;
    float wavelength_range = lambda_max - lambda_min;
    float delta_lambda = wavelength_range / float(wavelength_samples);

    for (int s = 0; s < samples_per_pixel; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(width - 1);
        float v = float(j + curand_uniform(&local_rand_state)) / float(height - 1);

        Color xyz(0.0f, 0.0f, 0.0f);
        for (int w = 0; w < wavelength_samples; w++) {
            float lambda = lambda_min + (float(w) + curand_uniform(&local_rand_state)) * delta_lambda;

            Ray r = camera->get_ray(u, v, &local_rand_state, lambda);
            float spectral_radiance = ray_spectral_radiance(
                r, objects, num_objects,
                spheres, triangles, rect_xy, rect_xz, rect_yz,
                materials, max_depth, &local_rand_state
            );

            Color cmf = cie_1931_xyz_from_wavelength(lambda);
            xyz += spectral_radiance * cmf * delta_lambda / CIE_Y_NORMALIZATION_380_780;
        }

        pixel_color += xyz_to_linear_srgb(xyz);
    }

    rand_state[pixel_index] = local_rand_state;
    framebuffer[pixel_index] = pixel_color;
}

extern "C" void launch_init_rand_states(
    curandState* rand_state,
    const ImageProperties& img_props,
    unsigned long seed
) {
    dim3 blocks(16, 16);
    dim3 grid(
        (img_props.width + blocks.x - 1) / blocks.x,
        (img_props.height + blocks.y - 1) / blocks.y
    );

    init_rand_state<<<grid, blocks>>>(
        rand_state,
        img_props.width,
        img_props.height,
        seed
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

extern "C" void launch_render_kernel(
    Color* framebuffer,
    const ImageProperties& img_props,
    Camera* dev_camera,
    HittableData* dev_objects,
    int num_objects,
    SphereData* dev_spheres,
    TriangleData* dev_triangles,
    RectangleXYData* dev_rect_xy,
    RectangleXZData* dev_rect_xz,
    RectangleYZData* dev_rect_yz,
    MaterialData* dev_materials,
    curandState* rand_state
) {
    dim3 blocks(16, 16);
    dim3 grid(
        (img_props.width + blocks.x - 1) / blocks.x,
        (img_props.height + blocks.y - 1) / blocks.y
    );

    render_kernel<<<grid, blocks>>>(
        framebuffer,
        img_props.width,
        img_props.height,
        img_props.samples_per_pixel,
        img_props.max_depth,
        img_props.spectral_samples,
        img_props.lambda_min,
        img_props.lambda_max,
        dev_camera,
        dev_objects,
        num_objects,
        dev_spheres,
        dev_triangles,
        dev_rect_xy,
        dev_rect_xz,
        dev_rect_yz,
        dev_materials,
        rand_state
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
