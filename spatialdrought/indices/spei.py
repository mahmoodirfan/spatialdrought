"""
Standardized Precipitation Evapotranspiration Index (SPEI).
Vicente-Serrano, Begueria & Lopez-Moreno (2010).

Uses 3-parameter log-logistic distribution fitted via L-moments.
Follows the SPEI R package implementation exactly.
"""

import numpy as np
import warnings
from typing import Optional, Union

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

from scipy.stats import norm
from spatialdrought.utils.temporal import aggregate_to_scale, month_of_year_indices


class SPEI:
    """
    Compute SPEI on gridded raster data. Vectorized, no pixel loops.

    Parameters
    ----------
    scale : int
        Accumulation period in months.
    calibration_start : int
        Reference period start index.
    calibration_end : int, optional
        Reference period end index (exclusive).

    Examples
    --------
    >>> import numpy as np
    >>> from spatialdrought import SPEI
    >>> p   = np.random.gamma(2, 50, size=(240, 50, 50))
    >>> pet = np.random.gamma(3, 30, size=(240, 50, 50))
    >>> result = SPEI(scale=3).fit_transform(p, pet)
    >>> result.shape
    (240, 50, 50)
    """

    def __init__(self, scale: int, calibration_start: int = 0, calibration_end: Optional[int] = None):
        if scale < 1:
            raise ValueError(f"scale must be >= 1, got {scale}")
        self.scale = scale
        self.calibration_start = calibration_start
        self.calibration_end = calibration_end
        self._shape  = None
        self._scale  = None
        self._loc    = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, precip, pet, start_month: int = 1) -> "SPEI":
        p, _ = self._parse(precip)
        e, _ = self._parse(pet)
        if p.shape != e.shape:
            raise ValueError(f"precip and pet must have same shape. Got {p.shape} vs {e.shape}")
        return self._fit_wb(p - e, start_month)

    def fit_wb(self, water_balance, start_month: int = 1) -> "SPEI":
        arr, _ = self._parse(water_balance)
        return self._fit_wb(arr, start_month)

    def transform(self, precip, pet, start_month: int = 1):
        if not self._fitted:
            raise RuntimeError("Call fit() before transform(). Or use fit_transform().")
        p, _ = self._parse(precip)
        e, xr_coords = self._parse(pet)
        return self._transform_wb(p - e, xr_coords, start_month)

    def transform_wb(self, water_balance, start_month: int = 1):
        if not self._fitted:
            raise RuntimeError("Call fit_wb() before transform_wb().")
        arr, xr_coords = self._parse(water_balance)
        return self._transform_wb(arr, xr_coords, start_month)

    def fit_transform(self, precip, pet, start_month: int = 1):
        return self.fit(precip, pet, start_month).transform(precip, pet, start_month)

    def fit_transform_wb(self, water_balance, start_month: int = 1):
        return self.fit_wb(water_balance, start_month).transform_wb(water_balance, start_month)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _fit_wb(self, wb: np.ndarray, start_month: int) -> "SPEI":
        accumulated = aggregate_to_scale(wb, self.scale, axis=0)
        cal_end = self.calibration_end or wb.shape[0]
        cal_data = accumulated[self.calibration_start:cal_end]
        n_cal = cal_data.shape[0]
        months = month_of_year_indices(n_cal, start_month=start_month)

        spatial = wb.shape[1:] if wb.ndim > 1 else ()
        self._shape = np.full((12, *spatial), np.nan)
        self._scale = np.full((12, *spatial), np.nan)
        self._loc   = np.full((12, *spatial), np.nan)

        for m in range(1, 13):
            idx = np.where(months == m)[0]
            if len(idx) == 0:
                continue
            month_data = cal_data[idx]
            n_valid = np.sum(~np.isnan(month_data), axis=0)
            mask = n_valid >= 10
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shape, scale, loc = self._fit_loglogistic(month_data)
            self._shape[m - 1] = np.where(mask, shape, np.nan)
            self._scale[m - 1] = np.where(mask, scale, np.nan)
            self._loc[m - 1]   = np.where(mask, loc,   np.nan)

        self._fitted = True
        self._start_month = start_month
        return self

    def _fit_loglogistic(self, data: np.ndarray):
        """
        Fit 3-parameter log-logistic following the SPEI R package (Begueria 2014).

        Step 1: Fix origin at just below min(data) — guarantees all data in support.
        Step 2: Fit 2-parameter log-logistic on (data - origin) via L-moments.

        For 2-parameter log-logistic:
            l1 = scale * (pi/c) / sin(pi/c)
            t2 = l2/l1 = 1 - 2^(-1/c)
            => c = -1 / log2(1 - t2)
        """
        n = data.shape[0]
        data_sorted = np.sort(data, axis=0)  # (n, rows, cols)

        # Step 1: origin just below minimum (R-package formula)
        x_min = data_sorted[0]
        x_2nd = data_sorted[1] if n > 1 else data_sorted[0]
        gap = np.abs(x_2nd - x_min)
        loc = x_min - 0.001 * np.where(gap > 0, gap, 0.001)

        # Step 2: shift and fit 2-parameter log-logistic
        shifted_sorted = data_sorted - loc[np.newaxis, ...]

        i = np.arange(n, dtype=np.float64)
        b0 = np.nanmean(shifted_sorted, axis=0)
        w1 = i / np.maximum(n - 1, 1)
        b1 = np.nanmean(shifted_sorted * w1[:, np.newaxis, np.newaxis], axis=0)

        l1 = b0
        l2 = 2.0 * b1 - b0

        with np.errstate(invalid="ignore", divide="ignore"):
            t2 = np.where(l1 > 0, l2 / l1, np.nan)
            t2 = np.clip(t2, 1e-6, 1.0 - 1e-6)
            shape = np.where(
                (l1 > 0) & (l2 > 0),
                -1.0 / np.log2(1.0 - t2),
                np.nan
            )
            shape = np.clip(shape, 0.01, 50.0)
            pic = np.pi / shape
            ratio = pic / np.sin(pic)
            scale = np.where(shape > 0, l1 / ratio, np.nan)

        return shape, scale, loc

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def _transform_wb(self, wb, xr_coords, start_month):
        accumulated = aggregate_to_scale(wb, self.scale, axis=0)
        n_time = accumulated.shape[0]
        months = month_of_year_indices(n_time, start_month=start_month)
        spei_out = np.full_like(accumulated, np.nan, dtype=np.float64)

        for m in range(1, 13):
            idx = np.where(months == m)[0]
            if len(idx) == 0:
                continue
            month_data = accumulated[idx]
            shape = self._shape[m - 1][np.newaxis, ...]
            scale = self._scale[m - 1][np.newaxis, ...]
            loc   = self._loc[m - 1][np.newaxis, ...]

            cdf = self._loglogistic_cdf(month_data, shape, scale, loc)
            valid = ~np.isnan(self._shape[m - 1])[np.newaxis, ...]
            with np.errstate(invalid="ignore"):
                spei_vals = np.where(valid, norm.ppf(np.clip(cdf, 1e-6, 1.0 - 1e-6)), np.nan)
            spei_out[idx] = spei_vals

        if HAS_XARRAY and xr_coords is not None:
            return xr.DataArray(
                spei_out, coords=xr_coords,
                attrs={"long_name": f"SPEI-{self.scale}", "units": "dimensionless"}
            )
        return spei_out

    def _loglogistic_cdf(self, x, shape, scale, loc):
        """
        F(x) = 1 / (1 + (scale / (x - loc))^shape)
        Numerically stable log-space implementation.
        """
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            shifted = x - loc
            valid = (shifted > 0) & (scale > 0) & (shape > 0)
            log_ratio = np.where(
                valid,
                shape * (np.log(np.where(valid, scale, 1.0)) - np.log(np.where(valid, shifted, 1.0))),
                0.0
            )
            cdf = np.where(valid, 1.0 / (1.0 + np.exp(log_ratio)), 0.0)
        # Propagate NaN input as NaN output (don't return 0 for NaN x)
        cdf = np.where(np.isnan(x), np.nan, cdf)
        return np.clip(cdf, 0.0, 1.0)

    # ------------------------------------------------------------------

    def _parse(self, data):
        if HAS_XARRAY and isinstance(data, xr.DataArray):
            return data.values, data.coords
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim not in (1, 3):
            raise ValueError(f"Input must be 1D or 3D, got shape {arr.shape}")
        return arr, None

    def __repr__(self):
        status = "fitted" if self._fitted else "unfitted"
        return f"SPEI(scale={self.scale}) [{status}]"
