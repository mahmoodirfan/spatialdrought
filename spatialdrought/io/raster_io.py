"""
Rasterio-based I/O for spatialdrought.

Read:  GeoTIFF stacks → numpy arrays + metadata dict
Write: numpy arrays + metadata → GeoTIFF stacks

Design decisions:
- Metadata travels with the array as a separate dict, not embedded in xarray.
  This keeps the core indices free of rasterio dependency.
- Multi-band GeoTIFF = time stack (band 1 = t=0, band 2 = t=1, ...).
- Single-file stack is the expected format (CHIRPS, MODIS MOD13A3 etc.).
- CRS, transform, nodata all preserved exactly through read → process → write.
- nodata values are converted to NaN on read, restored on write.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union

try:
    import rasterio
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def _require_rasterio():
    if not HAS_RASTERIO:
        raise ImportError(
            "rasterio is required for I/O operations. "
            "Install it with: pip install rasterio"
        )


def read_stack(
    path: Union[str, Path],
    band_indices: Optional[list] = None,
    masked: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Read a multi-band GeoTIFF as a (time, rows, cols) numpy array.

    Parameters
    ----------
    path : str or Path
        Path to GeoTIFF file. Each band = one time step.
    band_indices : list of int, optional
        1-based band indices to read. Default None = all bands.
    masked : bool
        If True, convert nodata values to NaN. Default True.

    Returns
    -------
    data : np.ndarray
        Shape (time, rows, cols), dtype float64.
        nodata → NaN if masked=True.
    meta : dict
        Rasterio metadata dict. Keys include:
        - driver, dtype, nodata, width, height, count, crs, transform
        - band_count: total bands in file
        - spatial_shape: (height, width)
    """
    _require_rasterio()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with rasterio.open(path) as src:
        meta = dict(src.meta)
        meta["band_count"] = src.count
        meta["spatial_shape"] = (src.height, src.width)
        meta["crs"] = src.crs
        meta["transform"] = src.transform
        meta["nodata"] = src.nodata

        indices = band_indices if band_indices else list(range(1, src.count + 1))
        data = src.read(indices).astype(np.float64)  # (bands, rows, cols)

    if masked and meta["nodata"] is not None:
        data = np.where(data == meta["nodata"], np.nan, data)

    return data, meta


def write_stack(
    data: np.ndarray,
    path: Union[str, Path],
    meta: dict,
    dtype: str = "float32",
    nodata: Optional[float] = -9999.0,
    descriptions: Optional[list] = None,
) -> Path:
    """
    Write a (time, rows, cols) numpy array as a multi-band GeoTIFF.

    Parameters
    ----------
    data : np.ndarray
        Shape (time, rows, cols).
    path : str or Path
        Output file path.
    meta : dict
        Metadata dict from read_stack() or manually constructed.
        Must contain: crs, transform, height, width.
    dtype : str
        Output dtype. Default 'float32' (half the size of float64).
    nodata : float, optional
        NoData value to write. NaN → nodata on write. Default -9999.0.
    descriptions : list of str, optional
        Band descriptions (e.g. date strings). Length must match time dimension.

    Returns
    -------
    path : Path
        Path to written file.
    """
    _require_rasterio()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 2:
        data = data[np.newaxis, ...]  # treat 2D as single band

    n_bands, height, width = data.shape

    # Replace NaN with nodata value for writing
    out = data.astype(np.float64).copy()
    if nodata is not None:
        out = np.where(np.isnan(out), nodata, out)
    out = out.astype(dtype)

    out_meta = {
        "driver": "GTiff",
        "dtype": dtype,
        "nodata": nodata,
        "width": width,
        "height": height,
        "count": n_bands,
        "crs": meta.get("crs"),
        "transform": meta.get("transform"),
        "compress": "lzw",      # lossless, good compression for float rasters
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    with rasterio.open(path, "w", **out_meta) as dst:
        dst.write(out)
        if descriptions:
            if len(descriptions) != n_bands:
                raise ValueError(
                    f"descriptions length ({len(descriptions)}) "
                    f"must match number of bands ({n_bands})"
                )
            for i, desc in enumerate(descriptions, 1):
                dst.update_tags(i, description=str(desc))

    return path


def read_single_band(
    path: Union[str, Path],
    band: int = 1,
    masked: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Read a single band from a GeoTIFF as a (rows, cols) array.
    Useful for reading masks, DEM, land cover etc.
    """
    _require_rasterio()
    data, meta = read_stack(path, band_indices=[band], masked=masked)
    return data[0], meta


def stack_from_files(
    file_list: list,
    masked: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Build a time stack from a list of single-band GeoTIFFs.
    Common pattern: one file per month/year (e.g. CHIRPS monthly files).

    All files must have identical CRS, transform, and spatial dimensions.

    Parameters
    ----------
    file_list : list of str or Path
        Ordered list of single-band raster files. Order = time order.
    masked : bool
        Convert nodata to NaN. Default True.

    Returns
    -------
    data : np.ndarray
        Shape (len(file_list), rows, cols).
    meta : dict
        Metadata from first file. Includes 'source_files' key.
    """
    _require_rasterio()
    if len(file_list) == 0:
        raise ValueError("file_list is empty")

    bands = []
    meta_ref = None

    for i, fpath in enumerate(file_list):
        band, meta = read_single_band(fpath, masked=masked)
        if meta_ref is None:
            meta_ref = meta
        else:
            # Validate spatial consistency
            if band.shape != (meta_ref["height"], meta_ref["width"]):
                raise ValueError(
                    f"File {fpath} has shape {band.shape}, "
                    f"expected ({meta_ref['height']}, {meta_ref['width']})"
                )
        bands.append(band)

    meta_ref["source_files"] = [str(f) for f in file_list]
    meta_ref["band_count"] = len(bands)
    return np.stack(bands, axis=0), meta_ref


def get_raster_info(path: Union[str, Path]) -> dict:
    """
    Print/return key metadata for a raster file without loading data.
    Useful for quick inspection before processing.

    Returns
    -------
    info : dict
        crs, transform, shape, dtype, nodata, band_count, bounds, resolution
    """
    _require_rasterio()
    with rasterio.open(path) as src:
        res_x = abs(src.transform.a)
        res_y = abs(src.transform.e)
        info = {
            "path": str(path),
            "crs": str(src.crs),
            "shape": (src.count, src.height, src.width),
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
            "band_count": src.count,
            "bounds": src.bounds,
            "resolution_deg": (res_x, res_y),
            "resolution_m": (res_x * 111320, res_y * 111320) if src.crs and src.crs.is_geographic else None,
            "transform": src.transform,
        }
    return info
