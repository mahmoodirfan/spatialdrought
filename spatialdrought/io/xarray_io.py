"""
xarray integration for spatialdrought.

Converts between xarray DataArrays and the (time, rows, cols) numpy arrays
expected by the index classes. Preserves coordinates, CRS, and attributes.

Also provides a convenience layer: read GeoTIFF → xarray DataArray with
proper time coordinates, ready to pass directly to SPI/SPEI/VCI etc.
"""

import numpy as np
from typing import Optional, Union
from pathlib import Path

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def _require_xarray():
    if not HAS_XARRAY:
        raise ImportError("xarray is required. Install with: pip install xarray")


def _require_rasterio():
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required. Install with: pip install rasterio")


def stack_to_dataarray(
    data: np.ndarray,
    meta: dict,
    time_coords=None,
    name: str = "data",
    attrs: Optional[dict] = None,
) -> "xr.DataArray":
    """
    Convert (time, rows, cols) numpy array + rasterio metadata to xarray DataArray.

    Assigns spatial coordinates (y, x) from the rasterio affine transform,
    and optionally time coordinates.

    Parameters
    ----------
    data : np.ndarray
        Shape (time, rows, cols).
    meta : dict
        Rasterio metadata from read_stack(). Must contain 'transform'.
    time_coords : array-like, optional
        Time coordinate values. Length must match data.shape[0].
        Can be pandas DatetimeIndex, numpy datetime64, or plain integers.
    name : str
        Name for the DataArray. Default 'data'.
    attrs : dict, optional
        Additional attributes (units, description, etc.).

    Returns
    -------
    da : xr.DataArray
        Shape (time, y, x) with spatial coordinates from transform.
    """
    _require_xarray()

    transform = meta.get("transform")
    height, width = data.shape[1], data.shape[2]

    # Build y (latitude) and x (longitude) coordinates from affine transform
    # transform.c = x origin (top-left), transform.f = y origin (top-left)
    # transform.a = pixel width, transform.e = pixel height (negative)
    if transform is not None:
        x_coords = transform.c + transform.a * (np.arange(width) + 0.5)
        y_coords = transform.f + transform.e * (np.arange(height) + 0.5)
    else:
        x_coords = np.arange(width, dtype=np.float64)
        y_coords = np.arange(height, dtype=np.float64)

    coords = {"y": y_coords, "x": x_coords}

    if time_coords is not None:
        coords["time"] = time_coords
        dims = ["time", "y", "x"]
    else:
        coords["time"] = np.arange(data.shape[0])
        dims = ["time", "y", "x"]

    da_attrs = {}
    if meta.get("crs") is not None:
        da_attrs["crs"] = str(meta["crs"])
    if attrs:
        da_attrs.update(attrs)

    return xr.DataArray(data, coords=coords, dims=dims, name=name, attrs=da_attrs)


def read_stack_as_dataarray(
    path: Union[str, Path],
    time_coords=None,
    name: str = "data",
    band_indices: Optional[list] = None,
) -> "xr.DataArray":
    """
    Read a multi-band GeoTIFF directly as an xarray DataArray.

    Parameters
    ----------
    path : str or Path
        GeoTIFF path.
    time_coords : array-like, optional
        Time coordinate values for each band.
    name : str
        DataArray name. Default 'data'.
    band_indices : list of int, optional
        1-based band indices to read. Default = all.

    Returns
    -------
    da : xr.DataArray
        Shape (time, y, x).
    """
    _require_xarray()
    _require_rasterio()

    from spatialdrought.io.raster_io import read_stack
    data, meta = read_stack(path, band_indices=band_indices)
    return stack_to_dataarray(data, meta, time_coords=time_coords, name=name)


def dataarray_to_stack(da: "xr.DataArray") -> tuple[np.ndarray, dict]:
    """
    Extract numpy array and minimal metadata dict from an xarray DataArray.

    Inverse of stack_to_dataarray. Reconstructs the affine transform
    from y/x coordinates if available.

    Returns
    -------
    data : np.ndarray
        Shape (time, rows, cols).
    meta : dict
        Minimal metadata with transform and crs if available.
    """
    _require_xarray()
    data = da.values

    meta = {}
    if "crs" in da.attrs:
        try:
            from rasterio.crs import CRS
            meta["crs"] = CRS.from_string(da.attrs["crs"])
        except Exception:
            meta["crs"] = None

    # Reconstruct affine transform from coordinates
    if "x" in da.coords and "y" in da.coords:
        x = da.coords["x"].values
        y = da.coords["y"].values
        if len(x) > 1 and len(y) > 1:
            dx = x[1] - x[0]
            dy = y[1] - y[0]
            try:
                from rasterio.transform import Affine
                # x[0] - dx/2 = left edge of first pixel
                meta["transform"] = Affine(dx, 0.0, x[0] - dx / 2,
                                           0.0, dy, y[0] - dy / 2)
            except Exception:
                meta["transform"] = None
        meta["height"] = len(y)
        meta["width"] = len(x)

    return data, meta
