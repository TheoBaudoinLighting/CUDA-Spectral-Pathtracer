#include "scene_setup.h"
#include "cuda_utils.cuh"
#include <cmath>
#include <cstdlib>

namespace {

MaterialData make_default_material(MaterialType type) {
    MaterialData material = {};
    material.type = type;
    material.albedo = Color(1.0f, 1.0f, 1.0f);
    material.fuzz = 0.0f;
    material.ior = 1.0f;
    material.dispersive = false;
    material.emission = Color(0.0f, 0.0f, 0.0f);
    material.spectral_function_id = SPECTRAL_PROFILE_RGB_RECONSTRUCTION;
    material.emission_spectrum_id = SPECTRAL_PROFILE_D65_WHITE;
    material.dispersion_A = 1.0f;
    material.dispersion_B = 0.0f;
    material.dispersion_C = 0.0f;
    material.absorption_density = 0.0f;
    material.absorption_color = Color(1.0f, 1.0f, 1.0f);
    material.absorption_spectrum_id = SPECTRAL_PROFILE_CONSTANT;
    return material;
}

bool nearly_neutral(const Color& value) {
    return std::fabs(value.x - value.y) < 1.0e-4f &&
           std::fabs(value.x - value.z) < 1.0e-4f;
}

}

int add_sphere(Scene& scene, const Point3& center, float radius, int material_id) {
    int index = static_cast<int>(scene.spheres.size());
    
    SphereData sphere;
    sphere.center = center;
    sphere.radius = radius;
    sphere.material_id = material_id;
    
    scene.spheres.push_back(sphere);
    
    HittableData object;
    object.type = SPHERE;
    object.data_index = index;
    
    scene.objects.push_back(object);
    
    return index;
}

int add_rectangle_xy(Scene& scene, float x0, float x1, float y0, float y1, float k, int material_id) {
    int index = static_cast<int>(scene.rect_xy.size());
    
    RectangleXYData rect;
    rect.x0 = x0;
    rect.x1 = x1;
    rect.y0 = y0;
    rect.y1 = y1;
    rect.k = k;
    rect.material_id = material_id;
    
    scene.rect_xy.push_back(rect);
    
    HittableData object;
    object.type = RECTANGLE_XY;
    object.data_index = index;
    
    scene.objects.push_back(object);
    
    return index;
}

int add_rectangle_xz(Scene& scene, float x0, float x1, float z0, float z1, float k, int material_id) {
    int index = static_cast<int>(scene.rect_xz.size());
    
    RectangleXZData rect;
    rect.x0 = x0;
    rect.x1 = x1;
    rect.z0 = z0;
    rect.z1 = z1;
    rect.k = k;
    rect.material_id = material_id;
    
    scene.rect_xz.push_back(rect);
    
    HittableData object;
    object.type = RECTANGLE_XZ;
    object.data_index = index;
    
    scene.objects.push_back(object);
    
    return index;
}

int add_rectangle_yz(Scene& scene, float y0, float y1, float z0, float z1, float k, int material_id) {
    int index = static_cast<int>(scene.rect_yz.size());
    
    RectangleYZData rect;
    rect.y0 = y0;
    rect.y1 = y1;
    rect.z0 = z0;
    rect.z1 = z1;
    rect.k = k;
    rect.material_id = material_id;
    
    scene.rect_yz.push_back(rect);
    
    HittableData object;
    object.type = RECTANGLE_YZ;
    object.data_index = index;
    
    scene.objects.push_back(object);
    
    return index;
}

int add_triangle(Scene& scene, const Point3& v0, const Point3& v1, const Point3& v2, int material_id) {
    int index = static_cast<int>(scene.triangles.size());
    
    TriangleData triangle;
    triangle.v0 = v0;
    triangle.v1 = v1;
    triangle.v2 = v2;
    
    Vec3 edge1 = v1 - v0;
    Vec3 edge2 = v2 - v0;
    triangle.normal = normalize(cross(edge1, edge2));
    
    triangle.material_id = material_id;
    
    scene.triangles.push_back(triangle);
    
    HittableData object;
    object.type = TRIANGLE;
    object.data_index = index;
    
    scene.objects.push_back(object);
    
    return index;
}

void add_triangular_prism(Scene& scene, const Point3& base_center, float base_size, float height, int material_id) {
    const float h = base_size * std::sqrt(3) / 2.0f;
    const float r = base_size / std::sqrt(3.0f);
    
    Point3 base_vertices[3];
    base_vertices[0] = Point3(base_center.x, base_center.y, base_center.z + r);
    base_vertices[1] = Point3(base_center.x + base_size/2.0f, base_center.y, base_center.z - h/3.0f);
    base_vertices[2] = Point3(base_center.x - base_size/2.0f, base_center.y, base_center.z - h/3.0f);
    
    Point3 top_vertices[3];
    for (int i = 0; i < 3; i++) {
        top_vertices[i] = Point3(base_vertices[i].x, base_vertices[i].y + height, base_vertices[i].z);
    }
    
    add_triangle(scene, base_vertices[0], base_vertices[2], base_vertices[1], material_id);
    
    add_triangle(scene, top_vertices[0], top_vertices[1], top_vertices[2], material_id);
    
    for (int i = 0; i < 3; i++) {
        int j = (i + 1) % 3;
        add_triangle(scene, base_vertices[i], base_vertices[j], top_vertices[i], material_id);
        add_triangle(scene, base_vertices[j], top_vertices[j], top_vertices[i], material_id);
    }
}

void add_box(Scene& scene, const Point3& p0, const Point3& p1, int material_id) {
    add_rectangle_xy(scene, p0.x, p1.x, p0.y, p1.y, p1.z, material_id);
    add_rectangle_xy(scene, p0.x, p1.x, p0.y, p1.y, p0.z, material_id);
    
    add_rectangle_xz(scene, p0.x, p1.x, p0.z, p1.z, p1.y, material_id);
    add_rectangle_xz(scene, p0.x, p1.x, p0.z, p1.z, p0.y, material_id);
    
    add_rectangle_yz(scene, p0.y, p1.y, p0.z, p1.z, p1.x, material_id);
    add_rectangle_yz(scene, p0.y, p1.y, p0.z, p1.z, p0.x, material_id);
}

int add_material(Scene& scene, const MaterialData& material) {
    int index = static_cast<int>(scene.materials.size());
    scene.materials.push_back(material);
    return index;
}

int add_lambertian_material(Scene& scene, const Color& albedo) {
    MaterialData material = make_default_material(LAMBERTIAN);
    material.albedo = albedo;
    return add_material(scene, material);
}

int add_metal_material(Scene& scene, const Color& albedo, float fuzz) {
    MaterialData material = make_default_material(METAL);
    material.albedo = albedo;
    material.fuzz = std::fmin(std::fmax(fuzz, 0.0f), 1.0f);
    return add_material(scene, material);
}

int add_dielectric_material(Scene& scene, float ior, bool dispersive) {
    MaterialData material = make_default_material(DIELECTRIC);
    material.ior = ior;
    material.dispersive = dispersive;
    material.dispersion_A = ior;
    material.dispersion_B = 0.0f;
    material.dispersion_C = 0.0f;
    if (dispersive) {
        material.dispersion_A = 1.5837f;
        material.dispersion_B = 10800.0f;
        material.dispersion_C = 0.0f;
    }
    return add_material(scene, material);
}

int add_cauchy_dielectric_material(Scene& scene, float A, float B, float C, float absorption_density) {
    MaterialData material = make_default_material(DIELECTRIC);
    material.ior = A;
    material.dispersive = true;
    material.dispersion_A = A;
    material.dispersion_B = B;
    material.dispersion_C = C;
    material.absorption_density = absorption_density;
    material.absorption_color = Color(1.0f, 1.0f, 1.0f);
    material.absorption_spectrum_id = SPECTRAL_PROFILE_CONSTANT;
    return add_material(scene, material);
}

int add_emissive_material(Scene& scene, const Color& emission) {
    MaterialData material = make_default_material(EMISSIVE);
    material.emission = emission;
    material.emission_spectrum_id = nearly_neutral(emission)
        ? SPECTRAL_PROFILE_D65_WHITE
        : SPECTRAL_PROFILE_RGB_RECONSTRUCTION;
    return add_material(scene, material);
}

int add_emissive_spectral_material(Scene& scene, float strength, int spectral_function_id) {
    MaterialData material = make_default_material(EMISSIVE);
    material.emission = Color(strength, strength, strength);
    material.emission_spectrum_id = spectral_function_id;
    return add_material(scene, material);
}

int add_spectral_material(Scene& scene, const Color& albedo, int spectral_function_id) {
    MaterialData material = make_default_material(SPECTRAL);
    material.albedo = albedo;
    material.spectral_function_id = spectral_function_id;
    return add_material(scene, material);
}

Scene create_prism_showcase_scene(const ImageProperties& img_props) {
    Point3 lookfrom(0, 8, -20);
    Point3 lookat(0, 2, 0);
    Vec3 vup(0, 1, 0);
    float dist_to_focus = 20.0f;
    float aperture = 0.1f;
    float aspect_ratio = static_cast<float>(img_props.width) / img_props.height;
    
    Camera cam(lookfrom, lookat, vup, 45, aspect_ratio, aperture, dist_to_focus);
    
    Scene scene(cam, img_props);
    
    int ground_material = add_lambertian_material(scene, Color(0.2f, 0.3f, 0.1f));
    int prism_material = add_cauchy_dielectric_material(scene, 1.5837f, 10800.0f, 0.0f, 0.0f);
    int light_material = add_emissive_material(scene, Color(15, 15, 15));
    int metal_material = add_metal_material(scene, Color(0.8f, 0.8f, 0.8f), 0.05f);
    int glass_material = add_dielectric_material(scene, 1.5f, false);
    int diffuse_material = add_lambertian_material(scene, Color(0.4f, 0.2f, 0.1f));
    int spectral_material = add_spectral_material(scene, Color(0.8f, 0.8f, 0.8f), SPECTRAL_PROFILE_GREEN_FILTER);
    
    add_rectangle_xz(scene, -1000, 1000, -1000, 1000, -1, ground_material);
    
    add_triangular_prism(scene, Point3(0, 2, -1), 3, 3, prism_material);
    
    add_sphere(scene, Point3(-10, 15, -10), 5, light_material);
    
    add_sphere(scene, Point3(-5, 2, 5), 2, metal_material);
    add_sphere(scene, Point3(5, 2, 5), 2, glass_material);
    add_sphere(scene, Point3(-5, 2, -5), 2, diffuse_material);
    add_sphere(scene, Point3(5, 2, -5), 2, spectral_material);
    
    int white_diffuse = add_lambertian_material(scene, Color(0.9f, 0.9f, 0.9f));
    for (int i = -5; i <= 5; i++) {
        for (int j = -5; j <= 5; j++) {
            add_sphere(scene, Point3(i*0.7f, 0, j*0.7f), 0.3f, white_diffuse);
        }
    }
    
    return scene;
}

Scene create_material_showcase_scene(const ImageProperties& img_props) {
    Point3 lookfrom(13, 2, 3);
    Point3 lookat(0, 0, 0);
    Vec3 vup(0, 1, 0);
    float dist_to_focus = 10.0f;
    float aperture = 0.1f;
    float aspect_ratio = static_cast<float>(img_props.width) / img_props.height;
    
    Camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);
    
    Scene scene(cam, img_props);
    
    int ground_material = add_lambertian_material(scene, Color(0.5f, 0.5f, 0.5f));
    add_sphere(scene, Point3(0, -1000, 0), 1000, ground_material);
    
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = static_cast<float>(rand()) / RAND_MAX;
            Point3 center(a + 0.9f * static_cast<float>(rand()) / RAND_MAX, 0.2f, b + 0.9f * static_cast<float>(rand()) / RAND_MAX);
            
            if ((center - Point3(4, 0.2f, 0)).length() > 0.9f) {
                int sphere_material;
                
                if (choose_mat < 0.8f) {
                    Color albedo = Color(
                        static_cast<float>(rand()) / RAND_MAX * static_cast<float>(rand()) / RAND_MAX,
                        static_cast<float>(rand()) / RAND_MAX * static_cast<float>(rand()) / RAND_MAX,
                        static_cast<float>(rand()) / RAND_MAX * static_cast<float>(rand()) / RAND_MAX
                    );
                    sphere_material = add_lambertian_material(scene, albedo);
                } else if (choose_mat < 0.95f) {
                    Color albedo = Color(
                        0.5f * (1.0f + static_cast<float>(rand()) / RAND_MAX),
                        0.5f * (1.0f + static_cast<float>(rand()) / RAND_MAX),
                        0.5f * (1.0f + static_cast<float>(rand()) / RAND_MAX)
                    );
                    float fuzz = 0.5f * static_cast<float>(rand()) / RAND_MAX;
                    sphere_material = add_metal_material(scene, albedo, fuzz);
                } else {
                    if (choose_mat > 0.98f) {
                        sphere_material = add_cauchy_dielectric_material(scene, 1.5837f, 10800.0f, 0.0f, 0.0f);
                    } else {
                        sphere_material = add_dielectric_material(scene, 1.5f, false);
                    }
                }
                
                add_sphere(scene, center, 0.2f, sphere_material);
            }
        }
    }
    
    int material1 = add_cauchy_dielectric_material(scene, 1.5837f, 10800.0f, 0.0f, 0.0f);
    add_sphere(scene, Point3(0, 1, 0), 1.0f, material1);
    
    int material2 = add_lambertian_material(scene, Color(0.4f, 0.2f, 0.1f));
    add_sphere(scene, Point3(-4, 1, 0), 1.0f, material2);
    
    int material3 = add_metal_material(scene, Color(0.7f, 0.6f, 0.5f), 0.0f);
    add_sphere(scene, Point3(4, 1, 0), 1.0f, material3);
    
    return scene;
}

Scene create_cornell_box_scene(const ImageProperties& img_props) {
    Point3 lookfrom(278, 278, -800);
    Point3 lookat(278, 278, 0);
    Vec3 vup(0, 1, 0);
    float dist_to_focus = 10.0f;
    float aperture = 0.0f;
    float aspect_ratio = static_cast<float>(img_props.width) / img_props.height;
    
    Camera cam(lookfrom, lookat, vup, 40, aspect_ratio, aperture, dist_to_focus);
    
    Scene scene(cam, img_props);
    
    int red = add_spectral_material(scene, Color(1.0f, 1.0f, 1.0f), SPECTRAL_PROFILE_CORNELL_RED);
    int white = add_spectral_material(scene, Color(1.0f, 1.0f, 1.0f), SPECTRAL_PROFILE_CORNELL_WHITE);
    int green = add_spectral_material(scene, Color(1.0f, 1.0f, 1.0f), SPECTRAL_PROFILE_CORNELL_GREEN);
    int light = add_emissive_spectral_material(scene, 1.0f, SPECTRAL_PROFILE_CORNELL_LIGHT);
    
    add_rectangle_yz(scene, 0, 555, 0, 555, 555, green);
    add_rectangle_yz(scene, 0, 555, 0, 555, 0, red);
    add_rectangle_xz(scene, 0, 555, 0, 555, 555, white);
    add_rectangle_xz(scene, 0, 555, 0, 555, 0, white);
    add_rectangle_xy(scene, 0, 555, 0, 555, 555, white);
    
    add_rectangle_xz(scene, 213, 343, 227, 332, 554, light);
    
    int prism_material = add_cauchy_dielectric_material(scene, 1.5837f, 10800.0f, 0.0f, 0.0f);
    add_triangular_prism(scene, Point3(278, 165, 278), 80, 165, prism_material);
    
    return scene;
}

void allocate_scene_on_gpu(
    const Scene& scene,
    HittableData** dev_objects,
    SphereData** dev_spheres,
    TriangleData** dev_triangles,
    RectangleXYData** dev_rect_xy,
    RectangleXZData** dev_rect_xz,
    RectangleYZData** dev_rect_yz,
    MaterialData** dev_materials,
    Camera** dev_camera
) {
    if (!scene.objects.empty()) {
        *dev_objects = cuda_alloc<HittableData>(scene.objects.size());
        cuda_copy_to_device<HittableData>(scene.objects.data(), *dev_objects, scene.objects.size());
    }
    
    if (!scene.spheres.empty()) {
        *dev_spheres = cuda_alloc<SphereData>(scene.spheres.size());
        cuda_copy_to_device<SphereData>(scene.spheres.data(), *dev_spheres, scene.spheres.size());
    }
    
    if (!scene.triangles.empty()) {
        *dev_triangles = cuda_alloc<TriangleData>(scene.triangles.size());
        cuda_copy_to_device<TriangleData>(scene.triangles.data(), *dev_triangles, scene.triangles.size());
    }
    
    if (!scene.rect_xy.empty()) {
        *dev_rect_xy = cuda_alloc<RectangleXYData>(scene.rect_xy.size());
        cuda_copy_to_device<RectangleXYData>(scene.rect_xy.data(), *dev_rect_xy, scene.rect_xy.size());
    }
    
    if (!scene.rect_xz.empty()) {
        *dev_rect_xz = cuda_alloc<RectangleXZData>(scene.rect_xz.size());
        cuda_copy_to_device<RectangleXZData>(scene.rect_xz.data(), *dev_rect_xz, scene.rect_xz.size());
    }
    
    if (!scene.rect_yz.empty()) {
        *dev_rect_yz = cuda_alloc<RectangleYZData>(scene.rect_yz.size());
        cuda_copy_to_device<RectangleYZData>(scene.rect_yz.data(), *dev_rect_yz, scene.rect_yz.size());
    }
    
    if (!scene.materials.empty()) {
        *dev_materials = cuda_alloc<MaterialData>(scene.materials.size());
        cuda_copy_to_device<MaterialData>(scene.materials.data(), *dev_materials, scene.materials.size());
    }
    
    *dev_camera = cuda_alloc<Camera>(1);
    cuda_copy_to_device<Camera>(&scene.camera, *dev_camera, 1);
}

void free_scene_on_gpu(
    HittableData* dev_objects,
    SphereData* dev_spheres,
    TriangleData* dev_triangles,
    RectangleXYData* dev_rect_xy,
    RectangleXZData* dev_rect_xz,
    RectangleYZData* dev_rect_yz,
    MaterialData* dev_materials,
    Camera* dev_camera
) {
    if (dev_objects) cuda_free(dev_objects);
    if (dev_spheres) cuda_free(dev_spheres);
    if (dev_triangles) cuda_free(dev_triangles);
    if (dev_rect_xy) cuda_free(dev_rect_xy);
    if (dev_rect_xz) cuda_free(dev_rect_xz);
    if (dev_rect_yz) cuda_free(dev_rect_yz);
    if (dev_materials) cuda_free(dev_materials);
    if (dev_camera) cuda_free(dev_camera);
}
