#ifndef SPECTRUM_CUH
#define SPECTRUM_CUH

#include "vec3.cuh"
#include "spectral_tables.cuh"
#include "cornell_spectral_data.cuh"
#include <cuda_runtime.h>

constexpr float LAMBDA_MIN = 380.0f;
constexpr float LAMBDA_MAX = 780.0f;
constexpr float CIE_Y_NORMALIZATION_380_780 = CIE_Y_INTEGRAL_380_780;

struct SpectralSampling {
    int num_samples;
    float lambda_min;
    float lambda_max;
    float delta_lambda;
};

enum SpectralProfileId {
    SPECTRAL_PROFILE_RGB_RECONSTRUCTION = 0,
    SPECTRAL_PROFILE_CONSTANT = 1,
    SPECTRAL_PROFILE_GREEN_FILTER = 2,
    SPECTRAL_PROFILE_RED_FILTER = 3,
    SPECTRAL_PROFILE_D65_WHITE = 4,
    SPECTRAL_PROFILE_WARM_BLACKBODY = 5,
    SPECTRAL_PROFILE_CORNELL_WHITE = 6,
    SPECTRAL_PROFILE_CORNELL_GREEN = 7,
    SPECTRAL_PROFILE_CORNELL_RED = 8,
    SPECTRAL_PROFILE_CORNELL_LIGHT = 9
};

__host__ __device__ inline float clamp_spectral(float value, float min_value = 0.0f, float max_value = 1.0f) {
    return fminf(fmaxf(value, min_value), max_value);
}

__host__ __device__ inline float gaussian_spectrum(float wavelength, float center, float sigma) {
    float x = (wavelength - center) / sigma;
    return expf(-0.5f * x * x);
}

__host__ __device__ inline float rgb_reflectance_to_spectrum(const Color& rgb, float wavelength) {
    float neutral = fminf(rgb.x, fminf(rgb.y, rgb.z));
    float red = fmaxf(0.0f, rgb.x - neutral);
    float green = fmaxf(0.0f, rgb.y - neutral);
    float blue = fmaxf(0.0f, rgb.z - neutral);

    float value =
        neutral +
        red * gaussian_spectrum(wavelength, 625.0f, 58.0f) +
        green * gaussian_spectrum(wavelength, 535.0f, 50.0f) +
        blue * gaussian_spectrum(wavelength, 455.0f, 42.0f);

    return clamp_spectral(value, 0.0f, 0.999f);
}

__host__ __device__ inline float rgb_illuminant_to_spectrum(const Color& rgb, float wavelength) {
    float neutral = fminf(rgb.x, fminf(rgb.y, rgb.z));
    float red = fmaxf(0.0f, rgb.x - neutral);
    float green = fmaxf(0.0f, rgb.y - neutral);
    float blue = fmaxf(0.0f, rgb.z - neutral);

    return
        neutral +
        red * gaussian_spectrum(wavelength, 625.0f, 70.0f) +
        green * gaussian_spectrum(wavelength, 535.0f, 62.0f) +
        blue * gaussian_spectrum(wavelength, 455.0f, 50.0f);
}

__host__ __device__ inline float blackbody_relative_spectrum(float wavelength_nm, float temperature_kelvin) {
    constexpr float c2 = 1.4387769e-2f;
    float lambda_m = wavelength_nm * 1.0e-9f;
    float reference_m = 560.0e-9f;

    float exponent = fminf(c2 / (lambda_m * temperature_kelvin), 80.0f);
    float reference_exponent = fminf(c2 / (reference_m * temperature_kelvin), 80.0f);
    float spectral = 1.0f / (powf(lambda_m, 5.0f) * (expf(exponent) - 1.0f));
    float reference = 1.0f / (powf(reference_m, 5.0f) * (expf(reference_exponent) - 1.0f));

    return spectral / reference;
}

__host__ __device__ inline float d65_white_spectrum(float wavelength) {
#ifdef __CUDA_ARCH__
    const float* table = CIE_D65_RELATIVE_DEVICE;
#else
    const float* table = CIE_D65_RELATIVE_HOST;
#endif

    if (wavelength <= float(SPECTRAL_TABLE_START_NM)) {
        return table[0];
    }
    if (wavelength >= float(SPECTRAL_TABLE_END_NM)) {
        return table[SPECTRAL_TABLE_SIZE - 1];
    }

    float position = wavelength - float(SPECTRAL_TABLE_START_NM);
    int index = int(floorf(position));
    float t = position - float(index);
    float a = table[index];
    float b = table[index + 1];
    return (1.0f - t) * a + t * b;
}

__host__ __device__ inline float lookup_regular_measured_spectrum(
    const float* table,
    int table_size,
    float start_nm,
    float step_nm,
    float wavelength
) {
    float end_nm = start_nm + step_nm * float(table_size - 1);
    if (wavelength < start_nm || wavelength > end_nm) {
        return 0.0f;
    }

    float position = (wavelength - start_nm) / step_nm;
    int index = int(floorf(position));
    if (index >= table_size - 1) {
        return table[table_size - 1];
    }

    float t = position - float(index);
    return (1.0f - t) * table[index] + t * table[index + 1];
}

__host__ __device__ inline float lookup_piecewise_measured_spectrum(
    const float* wavelengths,
    const float* values,
    int table_size,
    float wavelength
) {
    if (wavelength < wavelengths[0] || wavelength > wavelengths[table_size - 1]) {
        return 0.0f;
    }

    for (int i = 0; i < table_size - 1; ++i) {
        float wl0 = wavelengths[i];
        float wl1 = wavelengths[i + 1];
        if (wavelength >= wl0 && wavelength <= wl1) {
            float t = (wavelength - wl0) / (wl1 - wl0);
            return (1.0f - t) * values[i] + t * values[i + 1];
        }
    }

    return values[table_size - 1];
}

__host__ __device__ inline float cornell_reflectance_spectrum(int profile_id, float wavelength) {
#ifdef __CUDA_ARCH__
    const float* table = CORNELL_WHITE_REFLECTANCE_DEVICE;
#else
    const float* table = CORNELL_WHITE_REFLECTANCE_HOST;
#endif

    switch (profile_id) {
        case SPECTRAL_PROFILE_CORNELL_GREEN:
#ifdef __CUDA_ARCH__
            table = CORNELL_GREEN_REFLECTANCE_DEVICE;
#else
            table = CORNELL_GREEN_REFLECTANCE_HOST;
#endif
            break;
        case SPECTRAL_PROFILE_CORNELL_RED:
#ifdef __CUDA_ARCH__
            table = CORNELL_RED_REFLECTANCE_DEVICE;
#else
            table = CORNELL_RED_REFLECTANCE_HOST;
#endif
            break;
        case SPECTRAL_PROFILE_CORNELL_WHITE:
        default:
            break;
    }

    return lookup_regular_measured_spectrum(
        table,
        CORNELL_REFLECTANCE_TABLE_SIZE,
        CORNELL_REFLECTANCE_START_NM,
        CORNELL_REFLECTANCE_STEP_NM,
        wavelength
    );
}

__host__ __device__ inline float cornell_light_spectrum(float wavelength) {
#ifdef __CUDA_ARCH__
    const float* wavelengths = CORNELL_LIGHT_WAVELENGTHS_DEVICE;
    const float* values = CORNELL_LIGHT_SPECTRUM_DEVICE;
#else
    const float* wavelengths = CORNELL_LIGHT_WAVELENGTHS_HOST;
    const float* values = CORNELL_LIGHT_SPECTRUM_HOST;
#endif

    return lookup_piecewise_measured_spectrum(wavelengths, values, CORNELL_LIGHT_TABLE_SIZE, wavelength);
}

__host__ __device__ inline float evaluate_spectral_profile(
    int profile_id,
    const Color& rgb,
    float wavelength
) {
    switch (profile_id) {
        case SPECTRAL_PROFILE_CONSTANT:
            return clamp_spectral(rgb.x, 0.0f, 0.999f);
        case SPECTRAL_PROFILE_GREEN_FILTER:
            return (wavelength >= 490.0f && wavelength <= 570.0f) ? 0.82f : 0.04f;
        case SPECTRAL_PROFILE_RED_FILTER:
            return (wavelength >= 600.0f) ? 0.90f : 0.03f;
        case SPECTRAL_PROFILE_D65_WHITE:
            return d65_white_spectrum(wavelength);
        case SPECTRAL_PROFILE_WARM_BLACKBODY:
            return blackbody_relative_spectrum(wavelength, 3000.0f);
        case SPECTRAL_PROFILE_CORNELL_WHITE:
        case SPECTRAL_PROFILE_CORNELL_GREEN:
        case SPECTRAL_PROFILE_CORNELL_RED:
            return cornell_reflectance_spectrum(profile_id, wavelength);
        case SPECTRAL_PROFILE_CORNELL_LIGHT:
            return cornell_light_spectrum(wavelength);
        case SPECTRAL_PROFILE_RGB_RECONSTRUCTION:
        default:
            return rgb_reflectance_to_spectrum(rgb, wavelength);
    }
}

__host__ __device__ inline float cauchy_dispersion(float wavelength, float A, float B, float C) {
    float lambda_squared = wavelength * wavelength;
    return A + B / lambda_squared + C / (lambda_squared * lambda_squared);
}

__host__ __device__ inline float sellmeier_ior_um(float wavelength_nm, float b1, float b2, float b3, float c1, float c2, float c3) {
    float lambda_um = wavelength_nm * 0.001f;
    float lambda2 = lambda_um * lambda_um;
    float n2 = 1.0f +
        (b1 * lambda2) / (lambda2 - c1) +
        (b2 * lambda2) / (lambda2 - c2) +
        (b3 * lambda2) / (lambda2 - c3);
    return sqrtf(fmaxf(n2, 1.0f));
}

__host__ __device__ inline float glass_ior(float wavelength) {
    return cauchy_dispersion(wavelength, 1.5168f, 4320.0f, 0.0f);
}

__host__ __device__ inline float flint_glass_ior(float wavelength) {
    return cauchy_dispersion(wavelength, 1.5837f, 10800.0f, 0.0f);
}

__host__ __device__ inline float diamond_ior(float wavelength) {
    return cauchy_dispersion(wavelength, 2.38f, 19470.0f, 0.0f);
}

__host__ __device__ inline float water_ior(float wavelength) {
    return cauchy_dispersion(wavelength, 1.319f, 3370.0f, 0.0f);
}

__host__ __device__ inline Color cie_1931_xyz_from_wavelength(float wavelength) {
#ifdef __CUDA_ARCH__
    const float (*table)[3] = CIE_XYZ_1931_2DEG_DEVICE;
#else
    const float (*table)[3] = CIE_XYZ_1931_2DEG_HOST;
#endif

    if (wavelength < LAMBDA_MIN || wavelength > LAMBDA_MAX) {
        return Color(0.0f, 0.0f, 0.0f);
    }

    if (wavelength <= float(SPECTRAL_TABLE_START_NM)) {
        return Color(table[0][0], table[0][1], table[0][2]);
    }
    if (wavelength >= float(SPECTRAL_TABLE_END_NM)) {
        int last = SPECTRAL_TABLE_SIZE - 1;
        return Color(table[last][0], table[last][1], table[last][2]);
    }

    float position = wavelength - float(SPECTRAL_TABLE_START_NM);
    int index = int(floorf(position));
    float t = position - float(index);
    Color a(table[index][0], table[index][1], table[index][2]);
    Color b(table[index + 1][0], table[index + 1][1], table[index + 1][2]);
    return (1.0f - t) * a + t * b;
}

__host__ __device__ inline Color xyz_to_linear_srgb(const Color& xyz) {
    return Color(
        3.2406f * xyz.x - 1.5372f * xyz.y - 0.4986f * xyz.z,
       -0.9689f * xyz.x + 1.8758f * xyz.y + 0.0415f * xyz.z,
        0.0557f * xyz.x - 0.2040f * xyz.y + 1.0570f * xyz.z
    );
}

#endif // SPECTRUM_CUH
