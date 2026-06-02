#include <gtest/gtest.h>
#include "spectrum.cuh"

TEST(SpectrumTest, CieResponseIsPositiveNearGreen) {
    Color xyz = cie_1931_xyz_from_wavelength(555.0f);

    EXPECT_GT(xyz.x, 0.0f);
    EXPECT_GT(xyz.y, 0.0f);
    EXPECT_GT(xyz.z, 0.0f);
    EXPECT_GT(xyz.y, xyz.z);
}

TEST(SpectrumTest, FlintGlassDispersionDecreasesWithWavelength) {
    EXPECT_GT(flint_glass_ior(400.0f), flint_glass_ior(700.0f));
}

TEST(SpectrumTest, NeutralRgbReflectanceIsFlat) {
    Color neutral(0.73f, 0.73f, 0.73f);

    EXPECT_NEAR(rgb_reflectance_to_spectrum(neutral, 420.0f), 0.73f, 1.0e-5f);
    EXPECT_NEAR(rgb_reflectance_to_spectrum(neutral, 550.0f), 0.73f, 1.0e-5f);
    EXPECT_NEAR(rgb_reflectance_to_spectrum(neutral, 700.0f), 0.73f, 1.0e-5f);
}

TEST(SpectrumTest, CornellReflectanceUsesMeasuredTables) {
    EXPECT_NEAR(evaluate_spectral_profile(SPECTRAL_PROFILE_CORNELL_WHITE, Color(1.0f, 1.0f, 1.0f), 400.0f), 0.343f, 1.0e-6f);
    EXPECT_NEAR(evaluate_spectral_profile(SPECTRAL_PROFILE_CORNELL_GREEN, Color(1.0f, 1.0f, 1.0f), 500.0f), 0.285f, 1.0e-6f);
    EXPECT_NEAR(evaluate_spectral_profile(SPECTRAL_PROFILE_CORNELL_RED, Color(1.0f, 1.0f, 1.0f), 700.0f), 0.642f, 1.0e-6f);
}

TEST(SpectrumTest, CornellLightInterpolatesMeasuredSpectrum) {
    EXPECT_NEAR(evaluate_spectral_profile(SPECTRAL_PROFILE_CORNELL_LIGHT, Color(1.0f, 1.0f, 1.0f), 400.0f), 0.0f, 1.0e-6f);
    EXPECT_NEAR(evaluate_spectral_profile(SPECTRAL_PROFILE_CORNELL_LIGHT, Color(1.0f, 1.0f, 1.0f), 550.0f), 11.8f, 1.0e-5f);
    EXPECT_NEAR(evaluate_spectral_profile(SPECTRAL_PROFILE_CORNELL_LIGHT, Color(1.0f, 1.0f, 1.0f), 700.0f), 18.4f, 1.0e-6f);
}
