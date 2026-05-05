"""
Temperature Condition Index (TCI) — Kogan (1995).

TCI = (LST_max - LST) / (LST_max - LST_min) * 100

Inverted relative to VCI — high LST means thermal stress, so higher LST
gives lower TCI. Per pixel, per calendar month.

Range: 0 (extreme thermal stress) to 100 (cool, optimal conditions).
"""

import numpy as np
from typing import Optional, Union

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

from spatialdrought.utils.temporal import month_of_year_indices


class TCI:
    """
    Compute Temperature Condition Index on gridded LST raster stacks.

    Parameters
    ----------
    calibration_start : int
        Start index of reference period. Default 0.
    calibration_end : int, optional
        End index of reference period (exclusive). Default None.

    Examples
    --------
    >>> import numpy as np
    >>> from spatialdrought import TCI
    >>> lst = np.random.uniform(280, 320, size=(240, 100, 100))  # Kelvin
    >>> tci = TCI()
    >>> result = tci.fit_transform(lst)
    """

    def __init__(
        self,
        calibration_start: int = 0,
        calibration_end: Optional[int] = None,
    ):
        self.calibration_start = calibration_start
        self.calibration_end = calibration_end

        self._lst_min = None   # (12, rows, cols)
        self._lst_max = None   # (12, rows, cols)
        self._fitted = False

    def fit(
        self,
        lst: Union[np.ndarray, "xr.DataArray"],
        start_month: int = 1,
    ) -> "TCI":
        """
        Compute per-pixel per-calendar-month LST min and max.

        Parameters
        ----------
        lst : np.ndarray or xr.DataArray
            Shape (time, rows, cols). LST in Kelvin or Celsius — units don't
            matter as long as they're consistent between fit and transform.
        start_month : int
            Calendar month of first time step.
        """
        arr, _ = self._parse_input(lst)

        cal_end = self.calibration_end or arr.shape[0]
        cal_data = arr[self.calibration_start:cal_end]
        n_cal = cal_data.shape[0]
        months = month_of_year_indices(n_cal, start_month=start_month)

        spatial_shape = arr.shape[1:]
        self._lst_min = np.full((12, *spatial_shape), np.nan)
        self._lst_max = np.full((12, *spatial_shape), np.nan)

        for m in range(1, 13):
            idx = np.where(months == m)[0]
            if len(idx) == 0:
                continue
            month_data = cal_data[idx]
            n_valid = np.sum(~np.isnan(month_data), axis=0)
            self._lst_min[m - 1] = np.where(n_valid >= 3, np.nanmin(month_data, axis=0), np.nan)
            self._lst_max[m - 1] = np.where(n_valid >= 3, np.nanmax(month_data, axis=0), np.nan)

        self._fitted = True
        self._start_month = start_month
        return self

    def transform(
        self,
        lst: Union[np.ndarray, "xr.DataArray"],
        start_month: int = 1,
    ) -> Union[np.ndarray, "xr.DataArray"]:
        """
        Transform LST to TCI values.

        Returns
        -------
        tci : np.ndarray or xr.DataArray
            TCI values in [0, 100].
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform(). Or use fit_transform().")

        arr, xr_coords = self._parse_input(lst)
        start_month = getattr(self, "_start_month", start_month)

        n_time = arr.shape[0]
        months = month_of_year_indices(n_time, start_month=start_month)
        tci_out = np.full_like(arr, np.nan, dtype=np.float64)

        for m in range(1, 13):
            idx = np.where(months == m)[0]
            if len(idx) == 0:
                continue

            lst_min = self._lst_min[m - 1][np.newaxis, ...]
            lst_max = self._lst_max[m - 1][np.newaxis, ...]
            month_data = arr[idx]
            denom = lst_max - lst_min

            with np.errstate(invalid="ignore", divide="ignore"):
                tci = (lst_max - month_data) / denom * 100.0

            tci = np.where(denom <= 0, np.nan, tci)
            tci = np.clip(tci, 0.0, 100.0)
            tci_out[idx] = tci

        if HAS_XARRAY and xr_coords is not None:
            tci_out = xr.DataArray(
                tci_out,
                coords=xr_coords,
                attrs={"long_name": "TCI", "units": "%", "valid_range": [0, 100]},
            )

        return tci_out

    def fit_transform(
        self,
        lst: Union[np.ndarray, "xr.DataArray"],
        start_month: int = 1,
    ) -> Union[np.ndarray, "xr.DataArray"]:
        return self.fit(lst, start_month).transform(lst, start_month)

    def _parse_input(self, data):
        if HAS_XARRAY and isinstance(data, xr.DataArray):
            return data.values, data.coords
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim not in (1, 3):
            raise ValueError(f"Input must be 1D or 3D, got shape {arr.shape}")
        return arr, None

    def __repr__(self):
        status = "fitted" if self._fitted else "unfitted"
        return f"TCI() [{status}]"
