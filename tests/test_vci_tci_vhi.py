"""
Tests for VCI, TCI, VHI.

Correctness checks:
- Output range [0, 100]
- Boundary conditions (min pixel → 0, max pixel → 100)
- Inversion: TCI is inverted relative to VCI
- VHI weighted combination
- NaN propagation
- Calendar month isolation
"""

import numpy as np
import pytest
from spatialdrought.indices.vci import VCI
from spatialdrought.indices.tci import TCI
from spatialdrought.indices.vhi import VHI


@pytest.fixture
def ndvi_3d():
    rng = np.random.default_rng(42)
    return rng.uniform(0.1, 0.8, size=(240, 20, 20))


@pytest.fixture
def lst_3d():
    rng = np.random.default_rng(42)
    return rng.uniform(280.0, 320.0, size=(240, 20, 20))


# -----------------------------------------------------------------------
# VCI tests
# -----------------------------------------------------------------------

class TestVCI:
    def test_output_shape(self, ndvi_3d):
        result = VCI().fit_transform(ndvi_3d)
        assert result.shape == ndvi_3d.shape

    def test_output_range(self, ndvi_3d):
        result = VCI().fit_transform(ndvi_3d)
        valid = result[~np.isnan(result)]
        assert valid.min() >= 0.0
        assert valid.max() <= 100.0

    def test_min_pixel_gives_zero(self):
        """Pixel that equals the historical minimum should give VCI=0."""
        rng = np.random.default_rng(1)
        data = rng.uniform(0.2, 0.7, size=(120, 5, 5))
        # Force pixel (2,2) at time 0 to be the minimum for its month
        data[0, 2, 2] = -0.5  # guaranteed minimum
        vci = VCI()
        result = vci.fit_transform(data)
        assert result[0, 2, 2] == pytest.approx(0.0, abs=1e-6)

    def test_max_pixel_gives_100(self):
        """Pixel that equals the historical maximum should give VCI=100."""
        rng = np.random.default_rng(2)
        data = rng.uniform(0.2, 0.7, size=(120, 5, 5))
        data[0, 2, 2] = 1.5  # guaranteed maximum
        vci = VCI()
        result = vci.fit_transform(data)
        assert result[0, 2, 2] == pytest.approx(100.0, abs=1e-6)

    def test_constant_pixel_gives_nan(self):
        """Pixel with no variance (all same NDVI) → VHI undefined."""
        data = np.ones((120, 5, 5)) * 0.5
        vci = VCI()
        result = vci.fit_transform(data)
        assert np.all(np.isnan(result[:, 2, 2]))

    def test_nan_propagation(self, ndvi_3d):
        data = ndvi_3d.copy()
        data[:, 3, 3] = np.nan
        result = VCI().fit_transform(data)
        assert np.all(np.isnan(result[:, 3, 3]))
        assert not np.all(np.isnan(result[:, 3, 4]))

    def test_transform_before_fit_raises(self, ndvi_3d):
        with pytest.raises(RuntimeError):
            VCI().transform(ndvi_3d)


# -----------------------------------------------------------------------
# TCI tests
# -----------------------------------------------------------------------

class TestTCI:
    def test_output_shape(self, lst_3d):
        result = TCI().fit_transform(lst_3d)
        assert result.shape == lst_3d.shape

    def test_output_range(self, lst_3d):
        result = TCI().fit_transform(lst_3d)
        valid = result[~np.isnan(result)]
        assert valid.min() >= 0.0
        assert valid.max() <= 100.0

    def test_inversion_vs_vci(self):
        """
        TCI is inverted relative to temperature: highest LST → TCI=0.
        Lowest LST → TCI=100. Opposite of VCI.
        """
        rng = np.random.default_rng(5)
        lst = rng.uniform(290, 310, size=(120, 5, 5))
        lst[0, 2, 2] = 350.0  # max LST → TCI should be 0
        lst[1, 2, 2] = 200.0  # min LST → TCI should be 100

        tci = TCI()
        result = tci.fit_transform(lst)
        assert result[0, 2, 2] == pytest.approx(0.0, abs=1e-6)
        assert result[1, 2, 2] == pytest.approx(100.0, abs=1e-6)

    def test_constant_pixel_gives_nan(self):
        lst = np.ones((120, 5, 5)) * 300.0
        result = TCI().fit_transform(lst)
        assert np.all(np.isnan(result))


# -----------------------------------------------------------------------
# VHI tests
# -----------------------------------------------------------------------

class TestVHI:
    def test_output_shape(self, ndvi_3d, lst_3d):
        result = VHI().fit_transform(ndvi_3d, lst_3d)
        assert result.shape == ndvi_3d.shape

    def test_output_range(self, ndvi_3d, lst_3d):
        result = VHI().fit_transform(ndvi_3d, lst_3d)
        valid = result[~np.isnan(result)]
        assert valid.min() >= 0.0
        assert valid.max() <= 100.0

    def test_alpha_weighting(self, ndvi_3d, lst_3d):
        """VHI with alpha=1 should equal VCI. Alpha=0 should equal TCI."""
        vhi_a1 = VHI(alpha=1.0).fit_transform(ndvi_3d, lst_3d)
        vhi_a0 = VHI(alpha=0.0).fit_transform(ndvi_3d, lst_3d)

        vci = VCI().fit_transform(ndvi_3d)
        tci = TCI().fit_transform(lst_3d)

        np.testing.assert_allclose(vhi_a1, vci, equal_nan=True, rtol=1e-10)
        np.testing.assert_allclose(vhi_a0, tci, equal_nan=True, rtol=1e-10)

    def test_from_vci_tci(self, ndvi_3d, lst_3d):
        """from_vci_tci should give same result as fit_transform."""
        vhi = VHI(alpha=0.5)
        result_full = vhi.fit_transform(ndvi_3d, lst_3d)

        vci = VCI().fit_transform(ndvi_3d)
        tci = TCI().fit_transform(lst_3d)
        result_direct = VHI(alpha=0.5).from_vci_tci(vci, tci)

        np.testing.assert_allclose(result_full, result_direct, equal_nan=True, rtol=1e-10)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            VHI(alpha=1.5)

    def test_classify(self):
        vhi_vals = np.array([5.0, 15.0, 25.0, 35.0, 50.0, np.nan])
        classes = VHI.classify(vhi_vals)
        assert classes[0] == 4   # extreme
        assert classes[1] == 3   # severe
        assert classes[2] == 2   # moderate
        assert classes[3] == 1   # mild
        assert classes[4] == 0   # no drought
        assert classes[5] == -1  # NaN

    def test_transform_before_fit_raises(self, ndvi_3d, lst_3d):
        with pytest.raises(RuntimeError):
            VHI().transform(ndvi_3d, lst_3d)
