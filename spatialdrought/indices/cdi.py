"""
Combined Drought Indicator (CDI).

Adapted from the JRC European Drought Observatory (Sepulcre-Canto et al. 2012)
for open satellite data availability:

    Level 0 - No drought
    Level 1 - Watch     : SPI-3 < -1
    Level 2 - Warning   : SPI-3 < -1  AND  VHI < 40
    Level 3 - Alert     : SPI-3 < -1  AND  VHI < 40  AND  NDVI anomaly < threshold

NDVI anomaly = (NDVI - NDVI_mean) / NDVI_std  (z-score, per pixel per month)

All inputs must be pre-computed and spatially aligned (same shape).

Reference:
    Sepulcre-Canto et al. (2012) Nat. Hazards Earth Syst. Sci. 12:3519-3531
"""

import numpy as np
from typing import Optional, Union

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

from spatialdrought.utils.temporal import month_of_year_indices

# CDI class labels
CDI_LEVELS = {
    0: "no_drought",
    1: "watch",
    2: "warning",
    3: "alert",
    -1: "insufficient_data",
}

# Default thresholds (can be overridden)
DEFAULT_SPI_THRESHOLD   = -1.0   # SPI below this → watch
DEFAULT_VHI_THRESHOLD   = 40.0   # VHI below this → warning
DEFAULT_NDVI_THRESHOLD  = -1.0   # NDVI z-score below this → alert


class CDI:
    """
    Compute Combined Drought Indicator from SPI, VHI, and NDVI anomaly.

    CDI is hierarchical: each level requires all previous conditions to be met.
    Returns integer class array (0=none, 1=watch, 2=warning, 3=alert).

    Parameters
    ----------
    spi_threshold : float
        SPI value below which Watch is triggered. Default -1.0.
    vhi_threshold : float
        VHI value below which Warning is triggered. Default 40.0.
    ndvi_threshold : float
        NDVI z-score below which Alert is triggered. Default -1.0.

    Examples
    --------
    >>> import numpy as np
    >>> from spatialdrought import CDI
    >>> spi  = np.random.normal(0, 1, size=(240, 50, 50))
    >>> vhi  = np.random.uniform(0, 100, size=(240, 50, 50))
    >>> ndvi = np.random.uniform(0.1, 0.8, size=(240, 50, 50))
    >>> cdi = CDI()
    >>> result = cdi.fit_transform(spi, vhi, ndvi)
    >>> result.shape
    (240, 50, 50)
    >>> np.unique(result[result >= 0])
    array([0, 1, 2, 3], dtype=int8)
    """

    def __init__(
        self,
        spi_threshold: float  = DEFAULT_SPI_THRESHOLD,
        vhi_threshold: float  = DEFAULT_VHI_THRESHOLD,
        ndvi_threshold: float = DEFAULT_NDVI_THRESHOLD,
    ):
        self.spi_threshold  = spi_threshold
        self.vhi_threshold  = vhi_threshold
        self.ndvi_threshold = ndvi_threshold

        # NDVI climatology (per pixel per calendar month)
        self._ndvi_mean = None   # (12, rows, cols)
        self._ndvi_std  = None   # (12, rows, cols)
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        ndvi: Union[np.ndarray, "xr.DataArray"],
        calibration_start: int = 0,
        calibration_end: Optional[int] = None,
        start_month: int = 1,
    ) -> "CDI":
        """
        Compute NDVI climatology (mean + std per pixel per calendar month).

        Parameters
        ----------
        ndvi : np.ndarray or xr.DataArray
            Shape (time, rows, cols). NDVI values.
        calibration_start : int
            Start index of reference period.
        calibration_end : int, optional
            End index of reference period (exclusive).
        start_month : int
            Calendar month of first time step.
        """
        arr, _ = self._parse(ndvi)

        cal_end = calibration_end or arr.shape[0]
        cal_data = arr[calibration_start:cal_end]
        n_cal = cal_data.shape[0]
        months = month_of_year_indices(n_cal, start_month=start_month)

        spatial = arr.shape[1:]
        self._ndvi_mean = np.full((12, *spatial), np.nan)
        self._ndvi_std  = np.full((12, *spatial), np.nan)

        for m in range(1, 13):
            idx = np.where(months == m)[0]
            if len(idx) == 0:
                continue
            month_data = cal_data[idx]
            n_valid = np.sum(~np.isnan(month_data), axis=0)
            with np.errstate(all="ignore"):
                self._ndvi_mean[m - 1] = np.where(n_valid >= 3, np.nanmean(month_data, axis=0), np.nan)
                self._ndvi_std[m - 1]  = np.where(n_valid >= 3, np.nanstd(month_data, axis=0, ddof=1), np.nan)

        self._fitted = True
        self._start_month = start_month
        return self

    def transform(
        self,
        spi: Union[np.ndarray, "xr.DataArray"],
        vhi: Union[np.ndarray, "xr.DataArray"],
        ndvi: Union[np.ndarray, "xr.DataArray"],
        start_month: int = 1,
    ) -> np.ndarray:
        """
        Compute CDI class for each pixel and time step.

        Parameters
        ----------
        spi : np.ndarray or xr.DataArray
            Pre-computed SPI values. Shape (time, rows, cols).
        vhi : np.ndarray or xr.DataArray
            Pre-computed VHI values [0, 100]. Same shape.
        ndvi : np.ndarray or xr.DataArray
            Raw NDVI values. Same shape. Used to compute anomaly.
        start_month : int
            Calendar month of first time step.

        Returns
        -------
        cdi : np.ndarray, dtype int8
            CDI class. Shape (time, rows, cols).
            0=no drought, 1=watch, 2=warning, 3=alert, -1=no data.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform(). Or use fit_transform().")

        spi_arr, _  = self._parse(spi)
        vhi_arr, _  = self._parse(vhi)
        ndvi_arr, _ = self._parse(ndvi)

        shapes = {spi_arr.shape, vhi_arr.shape, ndvi_arr.shape}
        if len(shapes) > 1:
            raise ValueError(f"spi, vhi, ndvi must have same shape. Got {shapes}")

        ndvi_z = self._ndvi_anomaly(ndvi_arr, start_month=start_month)

        return self._classify(spi_arr, vhi_arr, ndvi_z)

    def fit_transform(
        self,
        spi: Union[np.ndarray, "xr.DataArray"],
        vhi: Union[np.ndarray, "xr.DataArray"],
        ndvi: Union[np.ndarray, "xr.DataArray"],
        calibration_start: int = 0,
        calibration_end: Optional[int] = None,
        start_month: int = 1,
    ) -> np.ndarray:
        """Fit NDVI climatology and compute CDI in one call."""
        return (
            self.fit(ndvi, calibration_start, calibration_end, start_month)
                .transform(spi, vhi, ndvi, start_month)
        )

    def drought_area_fraction(self, cdi: np.ndarray, level: int = 1) -> np.ndarray:
        """
        Compute fraction of spatial domain in drought at each time step.

        Parameters
        ----------
        cdi : np.ndarray
            Shape (time, rows, cols). CDI class array.
        level : int
            Minimum CDI level to count as drought (1=watch, 2=warning, 3=alert).

        Returns
        -------
        frac : np.ndarray
            Shape (time,). Fraction of valid pixels at or above given level.
        """
        n_time = cdi.shape[0]
        frac = np.full(n_time, np.nan)
        for t in range(n_time):
            valid = cdi[t] >= 0
            n_valid = np.sum(valid)
            if n_valid > 0:
                frac[t] = np.sum(cdi[t] >= level) / n_valid
        return frac

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ndvi_anomaly(self, ndvi: np.ndarray, start_month: int) -> np.ndarray:
        """
        Compute per-pixel per-calendar-month NDVI z-score.
        anomaly = (NDVI - monthly_mean) / monthly_std
        """
        n_time = ndvi.shape[0]
        months = month_of_year_indices(n_time, start_month=start_month)
        anomaly = np.full_like(ndvi, np.nan, dtype=np.float64)

        for m in range(1, 13):
            idx = np.where(months == m)[0]
            if len(idx) == 0:
                continue
            mean = self._ndvi_mean[m - 1][np.newaxis, ...]
            std  = self._ndvi_std[m - 1][np.newaxis, ...]
            month_data = ndvi[idx]
            with np.errstate(invalid="ignore", divide="ignore"):
                anomaly[idx] = np.where(std > 0, (month_data - mean) / std, np.nan)

        return anomaly

    def _classify(
        self,
        spi: np.ndarray,
        vhi: np.ndarray,
        ndvi_z: np.ndarray,
    ) -> np.ndarray:
        """
        Apply hierarchical CDI classification rules.

        Level 0 — no drought (default)
        Level 1 — watch:   SPI < spi_threshold
        Level 2 — warning: level 1 AND VHI < vhi_threshold
        Level 3 — alert:   level 2 AND NDVI_z < ndvi_threshold
        Level -1 — no data: any input is NaN
        """
        out = np.zeros(spi.shape, dtype=np.int8)

        any_nan = np.isnan(spi) | np.isnan(vhi) | np.isnan(ndvi_z)

        watch   = spi < self.spi_threshold
        warning = watch & (vhi < self.vhi_threshold)
        alert   = warning & (ndvi_z < self.ndvi_threshold)

        out = np.where(watch,   np.int8(1), out)
        out = np.where(warning, np.int8(2), out)
        out = np.where(alert,   np.int8(3), out)
        out = np.where(any_nan, np.int8(-1), out)

        return out

    def _parse(self, data):
        if HAS_XARRAY and isinstance(data, xr.DataArray):
            return data.values, data.coords
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim not in (1, 3):
            raise ValueError(f"Input must be 1D or 3D, got shape {arr.shape}")
        return arr, None

    def __repr__(self):
        status = "fitted" if self._fitted else "unfitted"
        return (
            f"CDI(spi_thr={self.spi_threshold}, "
            f"vhi_thr={self.vhi_threshold}, "
            f"ndvi_thr={self.ndvi_threshold}) [{status}]"
        )
