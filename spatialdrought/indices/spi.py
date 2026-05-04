"""
Standardized Precipitation Index (SPI) — McKee et al. (1993).

Spatial implementation: pixel-wise gamma distribution fitting on raster stacks.
No loops. Works on (time, rows, cols) numpy arrays or xarray DataArrays.

Key design: calibration period is fit ONCE per pixel per calendar month,
then applied to the full time series. This is how operational SPI is computed
(e.g., CHIRPS-based global SPI by USGS/FEWS NET).
"""

import numpy as np
from typing import Optional, Union
import warnings

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

from spatialdrought.utils.distributions import fit_gamma, gamma_cdf_vectorized, cdf_to_spi
from spatialdrought.utils.temporal import aggregate_to_scale, month_of_year_indices


class SPI:
    """
    Compute Standardized Precipitation Index on gridded raster data.

    Fits a gamma distribution per pixel per calendar month during a calibration
    period, then maps all time steps to SPI values via the normal inverse CDF.

    Parameters
    ----------
    scale : int
        Accumulation period in months (e.g., 1, 3, 6, 12, 24, 48).
    calibration_start : int, optional
        Index into the time axis where calibration period starts. Default 0.
    calibration_end : int, optional
        Index into the time axis where calibration period ends (exclusive).
        Default None (use all data).
    prob_zero_correction : bool
        Whether to apply zero-precipitation correction (recommended for arid regions).
        Default True. Especially important for Pakistan/South Asia.

    Examples
    --------
    >>> import numpy as np
    >>> from spatialdrought import SPI
    >>> precip = np.random.gamma(2, 50, size=(240, 100, 100))  # 20yr monthly, 100x100
    >>> spi = SPI(scale=3)
    >>> result = spi.fit_transform(precip)
    >>> result.shape
    (240, 100, 100)

    With xarray:
    >>> import xarray as xr
    >>> da = xr.DataArray(precip, dims=["time", "y", "x"])
    >>> result_da = spi.fit_transform(da)
    """

    def __init__(
        self,
        scale: int,
        calibration_start: int = 0,
        calibration_end: Optional[int] = None,
        prob_zero_correction: bool = True,
    ):
        if scale < 1:
            raise ValueError(f"scale must be >= 1, got {scale}")
        self.scale = scale
        self.calibration_start = calibration_start
        self.calibration_end = calibration_end
        self.prob_zero_correction = prob_zero_correction

        # Fitted parameters — populated by fit()
        self._alpha = None     # shape (12, rows, cols)
        self._beta = None      # shape (12, rows, cols)
        self._p_zero = None    # shape (12, rows, cols) — prob of zero precip
        self._fitted = False

    def fit(
        self,
        data: Union[np.ndarray, "xr.DataArray"],
        start_month: int = 1,
    ) -> "SPI":
        """
        Fit gamma distribution per pixel per calendar month.

        Parameters
        ----------
        data : np.ndarray or xr.DataArray
            Precipitation raster stack. Shape (time, rows, cols) or (time,).
            Units: any (mm/month typical). Must be non-negative.
        start_month : int
            Calendar month of first time step (1=Jan, 2=Feb, ...). Default 1.

        Returns
        -------
        self
        """
        arr, _ = self._parse_input(data)

        # Rolling accumulation
        accumulated = aggregate_to_scale(arr, self.scale, axis=0)

        cal_end = self.calibration_end if self.calibration_end else arr.shape[0]
        cal_data = accumulated[self.calibration_start:cal_end]
        n_cal = cal_data.shape[0]

        months = month_of_year_indices(n_cal, start_month=start_month)

        spatial_shape = arr.shape[1:] if arr.ndim > 1 else ()

        # Pre-allocate fitted parameter arrays
        self._alpha = np.full((12, *spatial_shape), np.nan)
        self._beta  = np.full((12, *spatial_shape), np.nan)
        self._p_zero = np.full((12, *spatial_shape), 0.0)

        for m in range(1, 13):
            idx = np.where(months == m)[0]
            if len(idx) == 0:
                continue

            month_data = cal_data[idx]  # (n_months_this_cal, rows, cols)

            if self.prob_zero_correction:
                # p_zero = fraction of months with zero precipitation
                zero_mask = (month_data == 0) | np.isnan(month_data)
                n_valid = np.sum(~np.isnan(month_data), axis=0)
                n_zero = np.sum(month_data == 0, axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    p_zero = np.where(n_valid > 0, n_zero / n_valid, np.nan)
                self._p_zero[m - 1] = p_zero

                # Fit only on nonzero values — replace zeros with NaN for fitting
                nonzero_data = np.where(month_data == 0, np.nan, month_data)
            else:
                nonzero_data = month_data

            # Require at least 10 valid values per pixel for reliable fit
            n_valid_nonzero = np.sum(~np.isnan(nonzero_data), axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                alpha, beta = fit_gamma(nonzero_data, axis=0)

            # Mask pixels with insufficient data
            alpha = np.where(n_valid_nonzero >= 10, alpha, np.nan)
            beta = np.where(n_valid_nonzero >= 10, beta, np.nan)

            self._alpha[m - 1] = alpha
            self._beta[m - 1] = beta

        self._fitted = True
        self._start_month = start_month
        return self

    def transform(
        self,
        data: Union[np.ndarray, "xr.DataArray"],
        start_month: int = 1,
    ) -> Union[np.ndarray, "xr.DataArray"]:
        """
        Transform precipitation data to SPI values using fitted parameters.

        Parameters
        ----------
        data : np.ndarray or xr.DataArray
            Precipitation data. Same spatial shape as calibration data.
        start_month : int
            Calendar month of first time step.

        Returns
        -------
        spi : np.ndarray or xr.DataArray
            SPI values. Same shape as input.
            NaN where: insufficient calibration data, negative precip, or invalid pixels.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform(). Or use fit_transform().")

        arr, xr_coords = self._parse_input(data)
        start_month = start_month if not hasattr(self, "_start_month") else self._start_month

        accumulated = aggregate_to_scale(arr, self.scale, axis=0)
        n_time = accumulated.shape[0]
        months = month_of_year_indices(n_time, start_month=start_month)

        spi_out = np.full_like(accumulated, np.nan, dtype=np.float64)

        for m in range(1, 13):
            idx = np.where(months == m)[0]
            if len(idx) == 0:
                continue

            month_data = accumulated[idx]  # (k, rows, cols)
            alpha = self._alpha[m - 1]     # (rows, cols)
            beta  = self._beta[m - 1]
            p_zero = self._p_zero[m - 1]

            # Gamma CDF per pixel
            cdf = gamma_cdf_vectorized(month_data, alpha, beta)

            if self.prob_zero_correction:
                # Adjust CDF for zero-inflation:
                # H(x) = p_zero + (1 - p_zero) * G(x)
                p_zero_bc = p_zero[np.newaxis, ...]  # broadcast over time
                cdf = p_zero_bc + (1.0 - p_zero_bc) * cdf
                # For zero precip, assign minimum CDF
                cdf = np.where(month_data == 0, p_zero_bc, cdf)

            spi_out[idx] = cdf_to_spi(cdf)

        if HAS_XARRAY and xr_coords is not None:
            spi_out = xr.DataArray(
                spi_out,
                coords=xr_coords,
                attrs={"long_name": f"SPI-{self.scale}", "units": "dimensionless"},
            )

        return spi_out

    def fit_transform(
        self,
        data: Union[np.ndarray, "xr.DataArray"],
        start_month: int = 1,
    ) -> Union[np.ndarray, "xr.DataArray"]:
        """Fit and transform in one call. Calibration = full input data."""
        return self.fit(data, start_month=start_month).transform(data, start_month=start_month)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_input(self, data):
        """
        Validate and convert input to numpy. Returns (arr, xr_coords_or_None).
        xr_coords preserved if input was xarray, for round-trip output.
        """
        if HAS_XARRAY and isinstance(data, xr.DataArray):
            xr_coords = data.coords
            arr = data.values
        else:
            xr_coords = None
            arr = np.asarray(data, dtype=np.float64)

        if arr.ndim not in (1, 3):
            raise ValueError(
                f"Input must be 1D (time,) or 3D (time, rows, cols), got shape {arr.shape}"
            )
        if np.any(arr[~np.isnan(arr)] < 0):
            raise ValueError(
                "Negative precipitation values found. SPI requires non-negative input."
            )

        return arr, xr_coords

    def __repr__(self):
        status = "fitted" if self._fitted else "unfitted"
        return f"SPI(scale={self.scale}, prob_zero_correction={self.prob_zero_correction}) [{status}]"
