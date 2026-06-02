#ifndef CORNELL_SPECTRAL_DATA_CUH
#define CORNELL_SPECTRAL_DATA_CUH

// Measured Cornell Box wall reflectance and light spectrum, sampled from 400-700 nm.
// Source mirror: Raysect Cornell Box demo, which references Cornell's physical box data.
constexpr int CORNELL_REFLECTANCE_TABLE_SIZE = 76;
constexpr float CORNELL_REFLECTANCE_START_NM = 400.0f;
constexpr float CORNELL_REFLECTANCE_END_NM = 700.0f;
constexpr float CORNELL_REFLECTANCE_STEP_NM = 4.0f;

static constexpr float CORNELL_WHITE_REFLECTANCE_HOST[CORNELL_REFLECTANCE_TABLE_SIZE] = {
    0.343f, 0.445f, 0.551f, 0.624f, 0.665f, 0.687f, 0.708f, 0.723f,
    0.715f, 0.710f, 0.745f, 0.758f, 0.739f, 0.767f, 0.777f, 0.765f,
    0.751f, 0.745f, 0.748f, 0.729f, 0.745f, 0.757f, 0.753f, 0.750f,
    0.746f, 0.747f, 0.735f, 0.732f, 0.739f, 0.734f, 0.725f, 0.721f,
    0.733f, 0.725f, 0.732f, 0.743f, 0.744f, 0.748f, 0.728f, 0.716f,
    0.733f, 0.726f, 0.713f, 0.740f, 0.754f, 0.764f, 0.752f, 0.736f,
    0.734f, 0.741f, 0.740f, 0.732f, 0.745f, 0.755f, 0.751f, 0.744f,
    0.731f, 0.733f, 0.744f, 0.731f, 0.712f, 0.708f, 0.729f, 0.730f,
    0.727f, 0.707f, 0.703f, 0.729f, 0.750f, 0.760f, 0.751f, 0.739f,
    0.724f, 0.730f, 0.740f, 0.737f
};

static constexpr float CORNELL_GREEN_REFLECTANCE_HOST[CORNELL_REFLECTANCE_TABLE_SIZE] = {
    0.092f, 0.096f, 0.098f, 0.097f, 0.098f, 0.095f, 0.095f, 0.097f,
    0.095f, 0.094f, 0.097f, 0.098f, 0.096f, 0.101f, 0.103f, 0.104f,
    0.107f, 0.109f, 0.112f, 0.115f, 0.125f, 0.140f, 0.160f, 0.187f,
    0.229f, 0.285f, 0.343f, 0.390f, 0.435f, 0.464f, 0.472f, 0.476f,
    0.481f, 0.462f, 0.447f, 0.441f, 0.426f, 0.406f, 0.373f, 0.347f,
    0.337f, 0.314f, 0.285f, 0.277f, 0.266f, 0.250f, 0.230f, 0.207f,
    0.186f, 0.171f, 0.160f, 0.148f, 0.141f, 0.136f, 0.130f, 0.126f,
    0.123f, 0.121f, 0.122f, 0.119f, 0.114f, 0.115f, 0.117f, 0.117f,
    0.118f, 0.120f, 0.122f, 0.128f, 0.132f, 0.139f, 0.144f, 0.146f,
    0.150f, 0.152f, 0.157f, 0.159f
};

static constexpr float CORNELL_RED_REFLECTANCE_HOST[CORNELL_REFLECTANCE_TABLE_SIZE] = {
    0.040f, 0.046f, 0.048f, 0.053f, 0.049f, 0.050f, 0.053f, 0.055f,
    0.057f, 0.056f, 0.059f, 0.057f, 0.061f, 0.061f, 0.060f, 0.062f,
    0.062f, 0.062f, 0.061f, 0.062f, 0.060f, 0.059f, 0.057f, 0.058f,
    0.058f, 0.058f, 0.056f, 0.055f, 0.056f, 0.059f, 0.057f, 0.055f,
    0.059f, 0.059f, 0.058f, 0.059f, 0.061f, 0.061f, 0.063f, 0.063f,
    0.067f, 0.068f, 0.072f, 0.080f, 0.090f, 0.099f, 0.124f, 0.154f,
    0.192f, 0.255f, 0.287f, 0.349f, 0.402f, 0.443f, 0.487f, 0.513f,
    0.558f, 0.584f, 0.620f, 0.606f, 0.609f, 0.651f, 0.612f, 0.610f,
    0.650f, 0.638f, 0.627f, 0.620f, 0.630f, 0.628f, 0.642f, 0.639f,
    0.657f, 0.639f, 0.635f, 0.642f
};

constexpr int CORNELL_LIGHT_TABLE_SIZE = 4;

static constexpr float CORNELL_LIGHT_WAVELENGTHS_HOST[CORNELL_LIGHT_TABLE_SIZE] = {
    400.0f, 500.0f, 600.0f, 700.0f
};

static constexpr float CORNELL_LIGHT_SPECTRUM_HOST[CORNELL_LIGHT_TABLE_SIZE] = {
    0.0f, 8.0f, 15.6f, 18.4f
};

#ifdef __CUDACC__
inline __device__ __constant__ float CORNELL_WHITE_REFLECTANCE_DEVICE[CORNELL_REFLECTANCE_TABLE_SIZE] = {
    0.343f, 0.445f, 0.551f, 0.624f, 0.665f, 0.687f, 0.708f, 0.723f,
    0.715f, 0.710f, 0.745f, 0.758f, 0.739f, 0.767f, 0.777f, 0.765f,
    0.751f, 0.745f, 0.748f, 0.729f, 0.745f, 0.757f, 0.753f, 0.750f,
    0.746f, 0.747f, 0.735f, 0.732f, 0.739f, 0.734f, 0.725f, 0.721f,
    0.733f, 0.725f, 0.732f, 0.743f, 0.744f, 0.748f, 0.728f, 0.716f,
    0.733f, 0.726f, 0.713f, 0.740f, 0.754f, 0.764f, 0.752f, 0.736f,
    0.734f, 0.741f, 0.740f, 0.732f, 0.745f, 0.755f, 0.751f, 0.744f,
    0.731f, 0.733f, 0.744f, 0.731f, 0.712f, 0.708f, 0.729f, 0.730f,
    0.727f, 0.707f, 0.703f, 0.729f, 0.750f, 0.760f, 0.751f, 0.739f,
    0.724f, 0.730f, 0.740f, 0.737f
};

inline __device__ __constant__ float CORNELL_GREEN_REFLECTANCE_DEVICE[CORNELL_REFLECTANCE_TABLE_SIZE] = {
    0.092f, 0.096f, 0.098f, 0.097f, 0.098f, 0.095f, 0.095f, 0.097f,
    0.095f, 0.094f, 0.097f, 0.098f, 0.096f, 0.101f, 0.103f, 0.104f,
    0.107f, 0.109f, 0.112f, 0.115f, 0.125f, 0.140f, 0.160f, 0.187f,
    0.229f, 0.285f, 0.343f, 0.390f, 0.435f, 0.464f, 0.472f, 0.476f,
    0.481f, 0.462f, 0.447f, 0.441f, 0.426f, 0.406f, 0.373f, 0.347f,
    0.337f, 0.314f, 0.285f, 0.277f, 0.266f, 0.250f, 0.230f, 0.207f,
    0.186f, 0.171f, 0.160f, 0.148f, 0.141f, 0.136f, 0.130f, 0.126f,
    0.123f, 0.121f, 0.122f, 0.119f, 0.114f, 0.115f, 0.117f, 0.117f,
    0.118f, 0.120f, 0.122f, 0.128f, 0.132f, 0.139f, 0.144f, 0.146f,
    0.150f, 0.152f, 0.157f, 0.159f
};

inline __device__ __constant__ float CORNELL_RED_REFLECTANCE_DEVICE[CORNELL_REFLECTANCE_TABLE_SIZE] = {
    0.040f, 0.046f, 0.048f, 0.053f, 0.049f, 0.050f, 0.053f, 0.055f,
    0.057f, 0.056f, 0.059f, 0.057f, 0.061f, 0.061f, 0.060f, 0.062f,
    0.062f, 0.062f, 0.061f, 0.062f, 0.060f, 0.059f, 0.057f, 0.058f,
    0.058f, 0.058f, 0.056f, 0.055f, 0.056f, 0.059f, 0.057f, 0.055f,
    0.059f, 0.059f, 0.058f, 0.059f, 0.061f, 0.061f, 0.063f, 0.063f,
    0.067f, 0.068f, 0.072f, 0.080f, 0.090f, 0.099f, 0.124f, 0.154f,
    0.192f, 0.255f, 0.287f, 0.349f, 0.402f, 0.443f, 0.487f, 0.513f,
    0.558f, 0.584f, 0.620f, 0.606f, 0.609f, 0.651f, 0.612f, 0.610f,
    0.650f, 0.638f, 0.627f, 0.620f, 0.630f, 0.628f, 0.642f, 0.639f,
    0.657f, 0.639f, 0.635f, 0.642f
};

inline __device__ __constant__ float CORNELL_LIGHT_WAVELENGTHS_DEVICE[CORNELL_LIGHT_TABLE_SIZE] = {
    400.0f, 500.0f, 600.0f, 700.0f
};

inline __device__ __constant__ float CORNELL_LIGHT_SPECTRUM_DEVICE[CORNELL_LIGHT_TABLE_SIZE] = {
    0.0f, 8.0f, 15.6f, 18.4f
};
#endif

#endif // CORNELL_SPECTRAL_DATA_CUH
