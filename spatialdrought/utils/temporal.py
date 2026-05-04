"""
Temporal aggregation utilities for raster time series.
All operations preserve spatial dimensions.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def rolling_sum(data: np.ndarray, window: int, axis: int = 0) -> np.ndarray:
    """
    Rolling sum along time axis. NaN-aware. No loops.

    Uses cumsum trick: O(n) time, O(n) space.
    NaN handling: if any value in window is NaN, output is NaN.

    Parameters
    ----------
    data : np.ndarray
        Shape (time, rows, cols) or (time,).
    window : int
        Number of time steps to sum (e.g., 3 for SPI-3).
    axis : int
        Time axis. Default 0.

    Returns
    -------
    result : np.ndarray
        Same shape as input. First (window-1) time steps are NaN.
    """
    if window == 1:
        return data.copy()

    data = np.moveaxis(data, axis, 0)
    n = data.shape[0]
    out = np.full_like(data, np.nan, dtype=np.float64)

    # NaN-aware cumsum: replace NaN with 0 for sum, track NaN mask separately
    nan_mask = np.isnan(data)
    data_filled = np.where(nan_mask, 0.0, data)

    cumsum = np.cumsum(data_filled, axis=0)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    out[window - 1:] = cumsum[window - 1:]

    # Count NaNs in each window — if any NaN present, result is NaN
    nan_count = np.cumsum(nan_mask.astype(int), axis=0)
    nan_count[window:] = nan_count[window:] - nan_count[:-window]
    has_nan = nan_count[window - 1:] > 0
    out[window - 1:] = np.where(has_nan, np.nan, out[window - 1:])

    return np.moveaxis(out, 0, axis)


def aggregate_to_scale(data: np.ndarray, scale: int, axis: int = 0) -> np.ndarray:
    """
    Aggregate precipitation/ET to SPI/SPEI timescale (e.g., SPI-3, SPI-12).
    Alias for rolling_sum — kept separate for semantic clarity in calling code.
    """
    return rolling_sum(data, window=scale, axis=axis)


def month_of_year_indices(n_months: int, start_month: int = 1) -> np.ndarray:
    """
    Return array of calendar month indices (1–12) for a time series of length n_months.
    Used for calibration-period subsetting by calendar month.
    """
    months = np.arange(start_month, start_month + n_months) % 12
    months[months == 0] = 12
    return months
