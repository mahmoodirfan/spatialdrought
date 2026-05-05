"""
Vegetation Condition Index (VCI) — Kogan (1995).

VCI = (NDVI - NDVI_min) / (NDVI_max - NDVI_min) * 100

Min/max computed per pixel per calendar month over a reference period.
This removes the seasonal signal — you're comparing this January to all
historical Januaries, not to July.

Range: 0 (extreme drought) to 100 (optimal vegetation condition).
"""

import numpy as np
from typing import Optional, Union

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

from spatialdrought.utils.temporal import month_of_year_indices


class VCI:
    """
    Compute Vegetation Condition Index on gridded NDVI raster stacks.

    Parameters
    ----------
    calibration_start : int
        Index into time axis where reference period starts. Default 0.
    calibration_end : int, optional
        Index into time axis where reference period ends (exclusive). Default None (all data).
    smooth : bool
        Apply 3-month running mean to NDVI before computing VCI.
        Reduces noise in sparse vegetation areas. Default False.

    Examples
    --------
    >>> import numpy as np
    >>> from spatialdrought import VCI
    >>> ndvi = np.random.uniform(0.1, 0.8, size=(240, 100, 100))
    >>> vci = VCI()
    >>> result = vci.fit_transform(ndvi)
    >>> result.shape
    (240, 100, 100)
    """

    def __init__(
        self,
        calibration_start: int = 0,
        calibration_end: Optional[int] = None,
        smooth: bool = False,
    ):
        self.calibration_start = calibration_start
        self.calibration_end = calibration_end
        self.smooth = smooth

        self._ndvi_min = None   # (12, rows, cols)
        self._ndvi_max = None   # (12, rows, cols)
        self._fitted = False

    def fit(
        self,
        ndvi: Union[np.ndarray, "xr.DataArray"],
        start_month: int = 1,
    ) -> "VCI":
        """
        Compute per-pixel per-calendar-month NDVI min and max.

        Parameters
        ----------
        ndvi : np.ndarray or xr.DataArray
            Shape (time, rows, cols). NDVI values, typically in [-1, 1].
        start_month : int
            Calendar month of first time step. Default 1 (January).
        """
        arr, _ = self._parse_input(ndvi)

        if self.smooth:
            arr = self._smooth3(arr)

        cal_end = self.calibration_end or arr.shape[0]
        cal_data = arr[self.calibration_start:cal_end]
        n_cal = cal_data.shape[0]
        months = month_of_year_indices(n_cal, start_month=start_month)

        spatial_shape = arr.shape[1:]
        self._ndvi_min = np.full((12, *spatial_shape), np.nan)
        self._ndvi_max = np.full((12, *spatial_shape), np.nan)

        for m in range(1, 13):
            idx = np.where(months == m)[0]
            if len(idx) == 0:
                continue
            month_data = cal_data[idx]
            # Require at least 3 observations — less than that is meaningless
            n_valid = np.sum(~np.isnan(month_data), axis=0)
            with np.errstate(all="ignore"):
                self._ndvi_min[m - 1] = np.where(n_valid >= 3, np.nanmin(month_data, axis=0), np.nan)
                self._ndvi_max[m - 1] = np.where(n_valid >= 3, np.nanmax(month_data, axis=0), np.nan)


        self._fitted = True
        self._start_month = start_month
        return self

    def transform(
        self,
        ndvi: Union[np.ndarray, "xr.DataArray"],
        start_month: int = 1,
    ) -> Union[np.ndarray, "xr.DataArray"]:
        """
        Transform NDVI to VCI values.

        Returns
        -------
        vci : np.ndarray or xr.DataArray
            VCI values in [0, 100]. NaN where NDVI_max == NDVI_min (no variance)
            or insufficient calibration data.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform(). Or use fit_transform().")

        arr, xr_coords = self._parse_input(ndvi)
        start_month = getattr(self, "_start_month", start_month)

        if self.smooth:
            arr = self._smooth3(arr)

        n_time = arr.shape[0]
        months = month_of_year_indices(n_time, start_month=start_month)
        vci_out = np.full_like(arr, np.nan, dtype=np.float64)

        for m in range(1, 13):
            idx = np.where(months == m)[0]
            if len(idx) == 0:
                continue

            ndvi_min = self._ndvi_min[m - 1][np.newaxis, ...]  # (1, rows, cols)
            ndvi_max = self._ndvi_max[m - 1][np.newaxis, ...]

            month_data = arr[idx]
            denom = ndvi_max - ndvi_min

            with np.errstate(invalid="ignore", divide="ignore"):
                vci = (month_data - ndvi_min) / denom * 100.0

            # Where max == min (constant NDVI pixel), result is undefined
            vci = np.where(denom <= 0, np.nan, vci)
            # Clip to [0, 100] — values outside range are sensor artifacts
            vci = np.clip(vci, 0.0, 100.0)

            vci_out[idx] = vci

        if HAS_XARRAY and xr_coords is not None:
            vci_out = xr.DataArray(
                vci_out,
                coords=xr_coords,
                attrs={"long_name": "VCI", "units": "%", "valid_range": [0, 100]},
            )

        return vci_out

    def fit_transform(
        self,
        ndvi: Union[np.ndarray, "xr.DataArray"],
        start_month: int = 1,
    ) -> Union[np.ndarray, "xr.DataArray"]:
        return self.fit(ndvi, start_month).transform(ndvi, start_month)

    def _smooth3(self, arr: np.ndarray) -> np.ndarray:
        """3-month running mean, NaN-aware."""
        from spatialdrought.utils.temporal import rolling_sum
        count = np.sum(~np.isnan(arr), axis=0, keepdims=True)  # not right for rolling
        # Use rolling sum / rolling count
        filled = np.where(np.isnan(arr), 0.0, arr)
        valid = (~np.isnan(arr)).astype(np.float64)
        sum3 = rolling_sum(filled, window=3, axis=0)
        cnt3 = rolling_sum(valid, window=3, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cnt3 > 0, sum3 / cnt3, np.nan)

    def _parse_input(self, data):
        if HAS_XARRAY and isinstance(data, xr.DataArray):
            return data.values, data.coords
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim not in (1, 3):
            raise ValueError(f"Input must be 1D or 3D, got shape {arr.shape}")
        return arr, None

    def __repr__(self):
        status = "fitted" if self._fitted else "unfitted"
        return f"VCI(smooth={self.smooth}) [{status}]"
