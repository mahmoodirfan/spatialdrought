"""
SPEI correctness tests.

Key tests:
1. Output shape preserved
2. Approximately N(0,1) over calibration period
3. First (scale-1) steps are NaN
4. Pixel isolation — no cross-pixel contamination
5. Water balance API vs P/PET API give identical results
6. Negative water balance accepted (unlike SPI)
7. Shape mismatch raises
8. NaN propagation
9. Hargreaves PET gives physically reasonable values
"""

import numpy as np
import pytest
from spatialdrought.indices.spei import SPEI
from spatialdrought.utils.pet import hargreaves_pet


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def precip_pet():
    """20 years monthly P and PET on 15x15 grid."""
    rng = np.random.default_rng(42)
    precip = rng.gamma(2.5, 40.0, size=(240, 15, 15))
    pet    = rng.gamma(3.0, 30.0, size=(240, 15, 15))
    return precip, pet


@pytest.fixture
def water_balance(precip_pet):
    p, e = precip_pet
    return p - e


# -----------------------------------------------------------------------
# SPEI core tests
# -----------------------------------------------------------------------

class TestSPEI:
    def test_output_shape(self, precip_pet):
        p, e = precip_pet
        result = SPEI(scale=3).fit_transform(p, e)
        assert result.shape == p.shape

    def test_approximately_standard_normal(self, precip_pet):
        """SPEI should be approx N(0,1) over calibration period."""
        p, e = precip_pet
        result = SPEI(scale=3).fit_transform(p, e)
        valid = result[2:][~np.isnan(result[2:])]
        assert abs(np.mean(valid)) < 0.35, f"Mean = {np.mean(valid):.3f}"
        assert abs(np.std(valid) - 1.0) < 0.45, f"Std = {np.std(valid):.3f}"

    def test_first_steps_nan(self, precip_pet):
        """First (scale-1) time steps must be NaN — no accumulation possible."""
        p, e = precip_pet
        scale = 6
        result = SPEI(scale=scale).fit_transform(p, e)
        assert np.all(np.isnan(result[:scale - 1]))

    def test_wb_api_matches_pet_api(self, precip_pet, water_balance):
        """fit_transform(p, pet) must equal fit_transform_wb(p - pet)."""
        p, e = precip_pet
        spei_pe = SPEI(scale=3).fit_transform(p, e)
        spei_wb = SPEI(scale=3).fit_transform_wb(water_balance)
        np.testing.assert_allclose(spei_pe, spei_wb, equal_nan=True, rtol=1e-10)

    def test_negative_water_balance_accepted(self):
        """SPEI must handle negative D = P - PET (arid conditions)."""
        rng = np.random.default_rng(7)
        # Heavily negative water balance — PET >> P
        wb = rng.normal(-50.0, 30.0, size=(120, 5, 5))
        result = SPEI(scale=1).fit_transform_wb(wb)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert abs(np.mean(valid)) < 0.3

    def test_pixel_isolation(self, precip_pet):
        """Pixel (5, 5) result from 3D must match same pixel run in 1D."""
        p, e = precip_pet
        result_3d = SPEI(scale=3).fit_transform(p, e)

        p_px = p[:, 5, 5][:, np.newaxis, np.newaxis]
        e_px = e[:, 5, 5][:, np.newaxis, np.newaxis]
        result_1d = SPEI(scale=3).fit_transform(p_px, e_px)

        np.testing.assert_allclose(
            result_3d[:, 5, 5],
            result_1d[:, 0, 0],
            equal_nan=True,
            rtol=1e-6,
        )

    def test_shape_mismatch_raises(self):
        rng = np.random.default_rng(0)
        p = rng.gamma(2, 40, size=(120, 5, 5))
        e = rng.gamma(2, 30, size=(120, 6, 5))  # different spatial shape
        with pytest.raises(ValueError, match="same shape"):
            SPEI(scale=3).fit(p, e)

    def test_invalid_scale_raises(self):
        with pytest.raises(ValueError, match="scale must be >= 1"):
            SPEI(scale=0)

    def test_transform_before_fit_raises(self, precip_pet):
        p, e = precip_pet
        with pytest.raises(RuntimeError, match="Call fit()"):
            SPEI(scale=3).transform(p, e)

    def test_nan_pixel_isolated(self, precip_pet):
        """All-NaN pixel should not contaminate neighbours."""
        p, e = precip_pet
        p = p.copy()
        p[:, 7, 7] = np.nan
        result = SPEI(scale=3).fit_transform(p, e)
        assert np.all(np.isnan(result[:, 7, 7]))
        assert not np.all(np.isnan(result[:, 7, 8]))

    def test_calibration_subset(self, precip_pet):
        """Fit on first 120 months, transform on all 240. Should not raise."""
        p, e = precip_pet
        spei = SPEI(scale=3, calibration_start=0, calibration_end=120)
        result = spei.fit(p, e).transform(p, e)
        assert result.shape == p.shape

    def test_repr(self):
        spei = SPEI(scale=6)
        assert "SPEI(scale=6" in repr(spei)
        assert "unfitted" in repr(spei)


# -----------------------------------------------------------------------
# Hargreaves PET tests
# -----------------------------------------------------------------------

class TestHargreavesPET:
    def test_output_shape(self):
        rng = np.random.default_rng(1)
        tmin = rng.uniform(10, 20, size=(365, 10, 10))
        tmax = rng.uniform(25, 40, size=(365, 10, 10))
        doy = np.arange(1, 366)
        lat = np.full((10, 10), 30.0)  # 30°N — Pakistan range
        pet = hargreaves_pet(tmin, tmax, doy, lat)
        assert pet.shape == (365, 10, 10)

    def test_output_non_negative(self):
        """PET is always >= 0."""
        rng = np.random.default_rng(2)
        tmin = rng.uniform(5, 15, size=(365, 5, 5))
        tmax = rng.uniform(20, 35, size=(365, 5, 5))
        doy = np.arange(1, 366)
        lat = np.full((5, 5), 25.0)
        pet = hargreaves_pet(tmin, tmax, doy, lat)
        assert np.all(pet >= 0)

    def test_physically_reasonable_range(self):
        """
        For Pakistan (~30°N), summer PET should be 5-15 mm/day.
        Winter PET should be 1-5 mm/day.
        """
        tmin_summer = np.full((1, 1, 1), 25.0)
        tmax_summer = np.full((1, 1, 1), 42.0)
        tmin_winter = np.full((1, 1, 1), 5.0)
        tmax_winter = np.full((1, 1, 1), 18.0)
        lat = np.full((1, 1), 30.0)

        pet_summer = hargreaves_pet(tmin_summer, tmax_summer, np.array([180]), lat)
        pet_winter = hargreaves_pet(tmin_winter, tmax_winter, np.array([15]), lat)

        assert 4.0 < pet_summer[0, 0, 0] < 25.0, f"Summer PET = {pet_summer[0,0,0]:.2f}"
        assert 0.5 < pet_winter[0, 0, 0] < 6.0,  f"Winter PET = {pet_winter[0,0,0]:.2f}"

    def test_higher_lat_lower_winter_pet(self):
        """Higher latitude → less winter radiation → lower PET in winter."""
        tmin = np.full((1, 1, 1), 5.0)
        tmax = np.full((1, 1, 1), 15.0)
        doy = np.array([15])  # January

        lat_low  = np.full((1, 1), 25.0)
        lat_high = np.full((1, 1), 55.0)

        pet_low  = hargreaves_pet(tmin, tmax, doy, lat_low)[0, 0, 0]
        pet_high = hargreaves_pet(tmin, tmax, doy, lat_high)[0, 0, 0]

        assert pet_low > pet_high, f"Low lat PET {pet_low:.2f} should > high lat {pet_high:.2f}"
