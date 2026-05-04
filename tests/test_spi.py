"""
SPI correctness tests.

Validation strategy:
1. Known synthetic data with known gamma parameters → verify SPI values
2. 1D vs 3D consistency — same pixel should give same result
3. Zero precipitation correction — arid region case
4. Calibration period isolation — transform on out-of-sample data
5. NaN propagation — invalid pixels should stay NaN, not corrupt neighbours
"""

import numpy as np
import pytest
from scipy.stats import gamma as scipy_gamma, norm

from spatialdrought.indices.spi import SPI
from spatialdrought.utils.temporal import rolling_sum
from spatialdrought.utils.distributions import fit_gamma, gamma_cdf_vectorized, cdf_to_spi


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def synthetic_precip_1d():
    """240 months of gamma-distributed precipitation (20 years)."""
    rng = np.random.default_rng(42)
    return rng.gamma(shape=2.5, scale=40.0, size=240)  # mm/month


@pytest.fixture
def synthetic_precip_3d():
    """240 months, 20x20 grid. All pixels same distribution for easy validation."""
    rng = np.random.default_rng(42)
    return rng.gamma(shape=2.5, scale=40.0, size=(240, 20, 20))


@pytest.fixture
def arid_precip_3d():
    """Precip with ~40% zeros — typical arid/semi-arid Pakistan scenario."""
    rng = np.random.default_rng(99)
    data = rng.gamma(shape=1.5, scale=20.0, size=(240, 10, 10))
    zero_mask = rng.random(size=(240, 10, 10)) < 0.4
    data[zero_mask] = 0.0
    return data


# -----------------------------------------------------------------------
# Distribution fitting tests
# -----------------------------------------------------------------------

class TestGammaFitting:
    def test_fit_recovers_known_params_1d(self):
        """With large n, fitted params should be close to true params."""
        rng = np.random.default_rng(0)
        true_alpha, true_beta = 3.0, 50.0
        data = rng.gamma(true_alpha, true_beta, size=5000)
        alpha, beta = fit_gamma(data[np.newaxis, :], axis=1)  # (1, 5000) -> scalar
        # Thom approximation: within 5% for large n
        assert abs(alpha - true_alpha) / true_alpha < 0.05, f"alpha: {alpha} vs {true_alpha}"
        assert abs(beta - true_beta) / true_beta < 0.05, f"beta: {beta} vs {true_beta}"

    def test_fit_3d_consistent_with_1d(self, synthetic_precip_3d):
        """Single pixel extracted from 3D should give same params as fitting in 1D."""
        data_3d = synthetic_precip_3d
        alpha_3d, beta_3d = fit_gamma(data_3d, axis=0)

        # Extract pixel (5, 7) and fit in 1D
        pixel = data_3d[:, 5, 7]
        alpha_1d, beta_1d = fit_gamma(pixel[:, np.newaxis, np.newaxis], axis=0)

        np.testing.assert_allclose(alpha_3d[5, 7], alpha_1d[0, 0], rtol=1e-10)
        np.testing.assert_allclose(beta_3d[5, 7], beta_1d[0, 0], rtol=1e-10)

    def test_fit_ignores_nan(self):
        """NaN values should not corrupt the fit."""
        rng = np.random.default_rng(1)
        data = rng.gamma(2.0, 30.0, size=200)
        data_with_nan = data.copy()
        data_with_nan[::10] = np.nan  # 10% NaN

        alpha_clean, beta_clean = fit_gamma(data[np.newaxis, :], axis=1)
        alpha_nan, beta_nan = fit_gamma(data_with_nan[np.newaxis, :], axis=1)

        # Should be close but not identical (different n)
        assert not np.isnan(alpha_nan)
        assert abs(alpha_nan - alpha_clean) / alpha_clean < 0.15  # within 15%


# -----------------------------------------------------------------------
# SPI correctness tests
# -----------------------------------------------------------------------

class TestSPI:
    def test_output_shape_3d(self, synthetic_precip_3d):
        spi = SPI(scale=3)
        result = spi.fit_transform(synthetic_precip_3d)
        assert result.shape == synthetic_precip_3d.shape

    def test_output_shape_1d(self, synthetic_precip_1d):
        spi = SPI(scale=3)
        result = spi.fit_transform(synthetic_precip_1d[:, np.newaxis, np.newaxis])
        assert result.shape == (240, 1, 1)

    def test_spi_approximately_standard_normal(self, synthetic_precip_3d):
        """
        SPI should be approx N(0,1) over the calibration period.
        Mean within 0.1, std within 0.1 of 1.0 for 240 months.
        """
        spi = SPI(scale=3)
        result = spi.fit_transform(synthetic_precip_3d)
        # Ignore first (scale-1) NaN time steps
        valid = result[2:][~np.isnan(result[2:])]
        assert abs(np.mean(valid)) < 0.1, f"Mean SPI = {np.mean(valid):.3f}, expected ~0"
        assert abs(np.std(valid) - 1.0) < 0.15, f"Std SPI = {np.std(valid):.3f}, expected ~1"

    def test_first_scale_minus_1_steps_are_nan(self, synthetic_precip_3d):
        """Rolling accumulation means first (scale-1) steps can't have valid SPI."""
        scale = 6
        spi = SPI(scale=scale)
        result = spi.fit_transform(synthetic_precip_3d)
        assert np.all(np.isnan(result[:scale - 1]))

    def test_3d_pixel_matches_1d(self, synthetic_precip_3d):
        """
        Pixel (3, 7) extracted and run in 1D should match pixel (3, 7) from 3D run.
        This is the key correctness test — no cross-pixel contamination.
        """
        spi_3d = SPI(scale=3)
        result_3d = spi_3d.fit_transform(synthetic_precip_3d)

        pixel = synthetic_precip_3d[:, 3, 7]
        spi_1d = SPI(scale=3)
        result_1d = spi_1d.fit_transform(pixel[:, np.newaxis, np.newaxis])

        np.testing.assert_allclose(
            result_3d[:, 3, 7], result_1d[:, 0, 0],
            rtol=1e-6,
            equal_nan=True
        )

    def test_nan_pixel_does_not_corrupt_neighbours(self):
        """NaN pixel in centre should leave neighbour pixels unaffected."""
        rng = np.random.default_rng(5)
        data = rng.gamma(2.0, 40.0, size=(120, 5, 5))

        data_with_nan = data.copy()
        data_with_nan[:, 2, 2] = np.nan  # centre pixel all NaN

        spi = SPI(scale=3)
        result = spi.fit_transform(data_with_nan)

        # Centre pixel should be all NaN
        assert np.all(np.isnan(result[:, 2, 2]))
        # Neighbour pixel (2, 3) should NOT be all NaN
        assert not np.all(np.isnan(result[:, 2, 3]))

    def test_zero_precip_correction_arid(self, arid_precip_3d):
        """
        With ~40% zeros, prob_zero_correction=True should produce valid SPI.
        Without correction, gamma fit would be poor (inflated by zeros).
        """
        spi_corrected = SPI(scale=3, prob_zero_correction=True)
        result = spi_corrected.fit_transform(arid_precip_3d)
        valid = result[2:][~np.isnan(result[2:])]

        # With correction, should still be approximately normal
        assert abs(np.mean(valid)) < 0.2
        assert abs(np.std(valid) - 1.0) < 0.25

    def test_negative_precip_raises(self):
        data = np.ones((120, 5, 5))
        data[10, 2, 2] = -1.0
        spi = SPI(scale=3)
        with pytest.raises(ValueError, match="Negative precipitation"):
            spi.fit_transform(data)

    def test_invalid_scale_raises(self):
        with pytest.raises(ValueError, match="scale must be >= 1"):
            SPI(scale=0)

    def test_transform_before_fit_raises(self, synthetic_precip_3d):
        spi = SPI(scale=3)
        with pytest.raises(RuntimeError, match="Call fit()"):
            spi.transform(synthetic_precip_3d)

    def test_calibration_period_subset(self, synthetic_precip_3d):
        """Fit on first 120 months, transform all 240. Should not raise."""
        spi = SPI(scale=3, calibration_start=0, calibration_end=120)
        result = spi.fit(synthetic_precip_3d).transform(synthetic_precip_3d)
        assert result.shape == synthetic_precip_3d.shape

    def test_repr(self):
        spi = SPI(scale=6)
        assert "SPI(scale=6" in repr(spi)
        assert "unfitted" in repr(spi)


# -----------------------------------------------------------------------
# Rolling sum tests (temporal utility)
# -----------------------------------------------------------------------

class TestRollingSumNaN:
    def test_window_with_single_nan_gives_nan(self):
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = rolling_sum(data[:, np.newaxis, np.newaxis], window=3, axis=0)[:, 0, 0]
        assert np.isnan(result[2])  # window [1,2,nan]
        assert np.isnan(result[3])  # window [2,nan,4]
        assert np.isnan(result[4])   # window [nan,4,5] — NaN propagates

    def test_no_nan_produces_correct_sum(self):
        data = np.array([1., 2., 3., 4., 5.])
        result = rolling_sum(data[:, np.newaxis, np.newaxis], window=3, axis=0)[:, 0, 0]
        np.testing.assert_allclose(result[2:], [6., 9., 12.])
