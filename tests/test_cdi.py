"""
CDI correctness tests.

Key tests:
1. Output dtype int8, shape preserved
2. Hierarchical logic — watch only where SPI<-1, warning only where watch, etc.
3. No drought where all inputs good
4. NaN any input → -1
5. Drought area fraction correctness
6. Custom thresholds
7. Transform before fit raises
8. Shape mismatch raises
"""

import numpy as np
import pytest
from spatialdrought.indices.cdi import CDI, CDI_LEVELS


@pytest.fixture
def inputs():
    """240 months, 20x20 grid. Controlled synthetic inputs."""
    rng = np.random.default_rng(42)
    spi  = rng.normal(0, 1, size=(240, 20, 20))
    vhi  = rng.uniform(0, 100, size=(240, 20, 20))
    ndvi = rng.uniform(0.1, 0.8, size=(240, 20, 20))
    return spi, vhi, ndvi


class TestCDI:
    def test_output_dtype(self, inputs):
        spi, vhi, ndvi = inputs
        result = CDI().fit_transform(spi, vhi, ndvi)
        assert result.dtype == np.int8

    def test_output_shape(self, inputs):
        spi, vhi, ndvi = inputs
        result = CDI().fit_transform(spi, vhi, ndvi)
        assert result.shape == spi.shape

    def test_valid_classes_only(self, inputs):
        spi, vhi, ndvi = inputs
        result = CDI().fit_transform(spi, vhi, ndvi)
        unique = np.unique(result)
        assert set(unique).issubset({-1, 0, 1, 2, 3})

    def test_no_drought_when_spi_positive(self, inputs):
        """Where SPI >= 0, CDI must be 0 (no drought)."""
        spi, vhi, ndvi = inputs
        result = CDI().fit_transform(spi, vhi, ndvi)
        no_watch = spi >= 0
        assert np.all(result[no_watch] == 0)

    def test_watch_requires_low_spi(self):
        """Watch should trigger only where SPI < -1."""
        spi  = np.full((120, 5, 5), -2.0)   # all SPI = -2 → all watch
        vhi  = np.full((120, 5, 5), 80.0)   # VHI fine → no warning
        ndvi = np.full((120, 5, 5), 0.5)
        ndvi += np.random.default_rng(1).normal(0, 0.05, size=ndvi.shape)  # add variance

        result = CDI().fit_transform(spi, vhi, ndvi)
        valid = result >= 0
        # All valid pixels should be watch (1) or higher
        assert np.all(result[valid] >= 1)

    def test_warning_requires_watch_and_low_vhi(self):
        """Warning only where SPI<-1 AND VHI<40."""
        spi  = np.full((120, 5, 5), -2.0)
        vhi  = np.full((120, 5, 5), 20.0)   # low VHI → warning
        ndvi = np.full((120, 5, 5), 0.5)
        ndvi += np.random.default_rng(2).normal(0, 0.05, size=ndvi.shape)

        result = CDI().fit_transform(spi, vhi, ndvi)
        valid = result >= 0
        assert np.all(result[valid] >= 2)

    def test_alert_requires_all_three(self):
        """Alert only where all three conditions met."""
        rng = np.random.default_rng(3)
        spi  = np.full((120, 5, 5), -2.0)
        vhi  = np.full((120, 5, 5), 20.0)

        # Calibration NDVI: realistic values with variance
        ndvi_cal = rng.normal(0.6, 0.08, size=(120, 5, 5))

        # Fit on calibration data
        cdi = CDI()
        cdi.fit(ndvi_cal)

        # Transform NDVI: far below calibration mean → z-score << -1 → alert
        ndvi_low = np.full((120, 5, 5), 0.1)

        result = cdi.transform(spi, vhi, ndvi_low)
        valid = result >= 0
        assert np.all(result[valid] >= 3)

    def test_hierarchy_respected(self, inputs):
        """
        No pixel should be warning (2) unless it's also watch (1 condition met).
        No pixel should be alert (3) unless it's also warning (2 condition met).
        """
        spi, vhi, ndvi = inputs
        result = CDI().fit_transform(spi, vhi, ndvi)

        # Alert implies warning condition (SPI<-1 AND VHI<40)
        alert_mask = result == 3
        assert np.all(spi[alert_mask] < -1.0)
        assert np.all(vhi[alert_mask] < 40.0)

        # Warning implies watch condition (SPI<-1)
        warning_mask = result == 2
        assert np.all(spi[warning_mask] < -1.0)

    def test_nan_input_gives_minus_one(self, inputs):
        """Any NaN in any input → CDI = -1 for that pixel/time."""
        spi, vhi, ndvi = inputs
        spi_nan = spi.copy()
        spi_nan[5, 3, 3] = np.nan

        result = CDI().fit_transform(spi_nan, vhi, ndvi)
        assert result[5, 3, 3] == -1

    def test_shape_mismatch_raises(self):
        rng = np.random.default_rng(0)
        spi  = rng.normal(0, 1, size=(120, 5, 5))
        vhi  = rng.uniform(0, 100, size=(120, 6, 5))  # different shape
        ndvi = rng.uniform(0.1, 0.8, size=(120, 5, 5))
        ndvi_cdi = rng.uniform(0.1, 0.8, size=(120, 5, 5))
        cdi = CDI()
        cdi.fit(ndvi_cdi)
        with pytest.raises(ValueError, match="same shape"):
            cdi.transform(spi, vhi, ndvi)

    def test_transform_before_fit_raises(self, inputs):
        spi, vhi, ndvi = inputs
        with pytest.raises(RuntimeError, match="Call fit()"):
            CDI().transform(spi, vhi, ndvi)

    def test_custom_thresholds(self, inputs):
        """Stricter SPI threshold → fewer watch pixels."""
        spi, vhi, ndvi = inputs
        cdi_default = CDI(spi_threshold=-1.0)
        cdi_strict  = CDI(spi_threshold=-2.0)

        result_default = cdi_default.fit_transform(spi, vhi, ndvi)
        result_strict  = cdi_strict.fit_transform(spi, vhi, ndvi)

        n_watch_default = np.sum(result_default >= 1)
        n_watch_strict  = np.sum(result_strict >= 1)
        assert n_watch_strict < n_watch_default

    def test_drought_area_fraction_range(self, inputs):
        """Fraction must be in [0, 1] for all time steps."""
        spi, vhi, ndvi = inputs
        result = CDI().fit_transform(spi, vhi, ndvi)
        cdi = CDI()
        frac = cdi.drought_area_fraction(result, level=1)
        assert np.all((frac >= 0) & (frac <= 1))
        assert frac.shape == (spi.shape[0],)

    def test_drought_area_fraction_alert_le_warning_le_watch(self, inputs):
        """Higher level drought → smaller or equal area fraction."""
        spi, vhi, ndvi = inputs
        result = CDI().fit_transform(spi, vhi, ndvi)
        cdi = CDI()
        f1 = cdi.drought_area_fraction(result, level=1)
        f2 = cdi.drought_area_fraction(result, level=2)
        f3 = cdi.drought_area_fraction(result, level=3)
        assert np.all(f3 <= f2)
        assert np.all(f2 <= f1)

    def test_repr(self):
        cdi = CDI()
        assert "CDI" in repr(cdi)
        assert "unfitted" in repr(cdi)
