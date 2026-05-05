"""
Vegetation Health Index (VHI) — Kogan (1995).

VHI = alpha * VCI + (1 - alpha) * TCI

Combines vegetation and thermal condition into a single drought indicator.
Default alpha=0.5 gives equal weight to both. In moisture-limited systems
(e.g. Pakistan's rainfed areas) alpha > 0.5 weights VCI more. In
energy-limited systems alpha < 0.5 weights TCI more.

Range: 0 (extreme drought) to 100 (no drought).

Kogan classification:
    < 10  : extreme drought
    10-20 : severe drought
    20-30 : moderate drought
    30-40 : mild drought
    > 40  : no drought
"""

import numpy as np
from typing import Optional, Union

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

from spatialdrought.indices.vci import VCI
from spatialdrought.indices.tci import TCI


# Kogan (1995) classification thresholds
VHI_CLASSES = {
    "extreme": (0, 10),
    "severe": (10, 20),
    "moderate": (20, 30),
    "mild": (30, 40),
    "no_drought": (40, 100),
}


class VHI:
    """
    Compute Vegetation Health Index from NDVI and LST raster stacks.

    Wraps VCI and TCI — can accept pre-computed VCI/TCI arrays or raw
    NDVI/LST and compute everything internally.

    Parameters
    ----------
    alpha : float
        Weight for VCI. Must be in [0, 1]. Default 0.5.
    calibration_start : int
        Reference period start index.
    calibration_end : int, optional
        Reference period end index (exclusive).

    Examples
    --------
    >>> import numpy as np
    >>> from spatialdrought import VHI
    >>> ndvi = np.random.uniform(0.1, 0.8, size=(240, 50, 50))
    >>> lst  = np.random.uniform(280, 320, size=(240, 50, 50))
    >>> vhi = VHI(alpha=0.5)
    >>> result = vhi.fit_transform(ndvi, lst)
    >>> result.shape
    (240, 50, 50)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        calibration_start: int = 0,
        calibration_end: Optional[int] = None,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.alpha = alpha
        self._vci = VCI(calibration_start=calibration_start, calibration_end=calibration_end)
        self._tci = TCI(calibration_start=calibration_start, calibration_end=calibration_end)
        self._fitted = False

    def fit(
        self,
        ndvi: Union[np.ndarray, "xr.DataArray"],
        lst: Union[np.ndarray, "xr.DataArray"],
        start_month: int = 1,
    ) -> "VHI":
        """Fit VCI and TCI on NDVI and LST respectively."""
        self._vci.fit(ndvi, start_month=start_month)
        self._tci.fit(lst, start_month=start_month)
        self._fitted = True
        return self

    def transform(
        self,
        ndvi: Union[np.ndarray, "xr.DataArray"],
        lst: Union[np.ndarray, "xr.DataArray"],
        start_month: int = 1,
    ) -> Union[np.ndarray, "xr.DataArray"]:
        """
        Transform NDVI and LST to VHI.

        Returns
        -------
        vhi : np.ndarray or xr.DataArray
            VHI values in [0, 100].
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform(). Or use fit_transform().")

        vci = self._vci.transform(ndvi, start_month=start_month)
        tci = self._tci.transform(lst, start_month=start_month)

        # Handle xarray vs numpy transparently
        if HAS_XARRAY and isinstance(vci, xr.DataArray):
            vhi = self.alpha * vci + (1.0 - self.alpha) * tci
            vhi.attrs = {
                "long_name": "VHI",
                "units": "%",
                "valid_range": [0, 100],
                "alpha": self.alpha,
            }
        else:
            vhi = self.alpha * vci + (1.0 - self.alpha) * tci

        return vhi

    def fit_transform(
        self,
        ndvi: Union[np.ndarray, "xr.DataArray"],
        lst: Union[np.ndarray, "xr.DataArray"],
        start_month: int = 1,
    ) -> Union[np.ndarray, "xr.DataArray"]:
        return self.fit(ndvi, lst, start_month).transform(ndvi, lst, start_month)

    def from_vci_tci(
        self,
        vci: np.ndarray,
        tci: np.ndarray,
    ) -> np.ndarray:
        """
        Compute VHI directly from pre-computed VCI and TCI arrays.
        No fitting required — use when you already have VCI/TCI.

        Parameters
        ----------
        vci, tci : np.ndarray
            Arrays of same shape, values in [0, 100].
        """
        return self.alpha * vci + (1.0 - self.alpha) * tci

    @staticmethod
    def classify(vhi: np.ndarray) -> np.ndarray:
        """
        Classify VHI values into Kogan drought categories.

        Returns integer array:
            0 = no drought
            1 = mild
            2 = moderate
            3 = severe
            4 = extreme
        """
        out = np.zeros_like(vhi, dtype=np.int8)
        out = np.where(vhi < 40, 1, out)   # mild
        out = np.where(vhi < 30, 2, out)   # moderate
        out = np.where(vhi < 20, 3, out)   # severe
        out = np.where(vhi < 10, 4, out)   # extreme
        out = np.where(np.isnan(vhi), -1, out)
        return out

    def __repr__(self):
        status = "fitted" if self._fitted else "unfitted"
        return f"VHI(alpha={self.alpha}) [{status}]"
