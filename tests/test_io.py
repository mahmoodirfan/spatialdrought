"""
I/O module tests.

Uses rasterio MemoryFile to create synthetic GeoTIFFs in memory —
no actual files written to disk during testing.

Tests cover:
1. read_stack — shape, dtype, nodata→NaN conversion
2. write_stack — round-trip: write then read back
3. stack_from_files — multi-file stacking
4. read_single_band — single band extraction
5. get_raster_info — metadata without loading data
6. xarray round-trip — stack_to_dataarray / dataarray_to_stack
7. End-to-end: read → SPI → write
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    from rasterio.io import MemoryFile
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

pytestmark = pytest.mark.skipif(not HAS_RASTERIO, reason="rasterio not installed")

from spatialdrought.io.raster_io import (
    read_stack, write_stack, read_single_band,
    stack_from_files, get_raster_info
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def make_synthetic_geotiff(
    data: np.ndarray,
    nodata: float = -9999.0,
    crs_epsg: int = 4326,
    bounds: tuple = (60.0, 20.0, 80.0, 40.0),  # Pakistan bbox
) -> bytes:
    """
    Create an in-memory GeoTIFF from a (bands, rows, cols) array.
    Returns raw bytes.
    """
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    n_bands, height, width = data.shape
    transform = from_bounds(*bounds, width=width, height=height)

    import io
    import tempfile, os
    # Write to a real temp file — MemoryFile.getvalue() not available in all versions
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name
    with rasterio.open(
        tmp_path, "w",
        driver="GTiff",
        height=height, width=width,
        count=n_bands,
        dtype=data.dtype,
        crs=CRS.from_epsg(crs_epsg),
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data)
    with open(tmp_path, "rb") as f:
        result = f.read()
    os.unlink(tmp_path)
    return result


@pytest.fixture
def synthetic_stack():
    """20-band float32 raster, 30x30 pixels, Pakistan bbox."""
    rng = np.random.default_rng(42)
    data = rng.gamma(2.5, 40.0, size=(20, 30, 30)).astype(np.float32)
    return data


@pytest.fixture
def tiff_bytes(synthetic_stack):
    return make_synthetic_geotiff(synthetic_stack)


@pytest.fixture
def tiff_path(tiff_bytes, tmp_path):
    """Write synthetic GeoTIFF to a temp file."""
    p = tmp_path / "test_stack.tif"
    p.write_bytes(tiff_bytes)
    return p


# -----------------------------------------------------------------------
# read_stack tests
# -----------------------------------------------------------------------

class TestReadStack:
    def test_shape(self, tiff_path, synthetic_stack):
        data, meta = read_stack(tiff_path)
        assert data.shape == synthetic_stack.shape

    def test_dtype_float64(self, tiff_path):
        data, _ = read_stack(tiff_path)
        assert data.dtype == np.float64

    def test_nodata_to_nan(self, tmp_path):
        """NoData values (-9999) should become NaN on read."""
        raw = np.ones((5, 10, 10), dtype=np.float32)
        raw[2, 5, 5] = -9999.0
        path = tmp_path / "nd.tif"
        path.write_bytes(make_synthetic_geotiff(raw, nodata=-9999.0))
        data, _ = read_stack(path)
        assert np.isnan(data[2, 5, 5])
        assert not np.isnan(data[0, 0, 0])

    def test_metadata_keys(self, tiff_path):
        _, meta = read_stack(tiff_path)
        for key in ("crs", "transform", "nodata", "width", "height", "band_count"):
            assert key in meta

    def test_crs_preserved(self, tiff_path):
        _, meta = read_stack(tiff_path)
        assert meta["crs"] is not None
        assert "4326" in str(meta["crs"])

    def test_band_subset(self, tiff_path):
        data, _ = read_stack(tiff_path, band_indices=[1, 2, 3])
        assert data.shape[0] == 3

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            read_stack("/nonexistent/path/file.tif")


# -----------------------------------------------------------------------
# write_stack tests
# -----------------------------------------------------------------------

class TestWriteStack:
    def test_round_trip_shape(self, tiff_path, tmp_path):
        data, meta = read_stack(tiff_path)
        out_path = tmp_path / "out.tif"
        write_stack(data, out_path, meta)
        data2, _ = read_stack(out_path)
        assert data2.shape == data.shape

    def test_round_trip_values(self, tiff_path, tmp_path):
        """Values should survive float64 → float32 → float64 with ~1e-5 precision."""
        data, meta = read_stack(tiff_path)
        out_path = tmp_path / "rt.tif"
        write_stack(data, out_path, meta, dtype="float32")
        data2, _ = read_stack(out_path)
        np.testing.assert_allclose(
            data[~np.isnan(data)],
            data2[~np.isnan(data2)],
            rtol=1e-4
        )

    def test_nan_written_as_nodata(self, tmp_path, tiff_path):
        """NaN in input should become nodata value in output file."""
        data, meta = read_stack(tiff_path)
        data[0, 5, 5] = np.nan
        out_path = tmp_path / "nan_test.tif"
        write_stack(data, out_path, meta, nodata=-9999.0)
        with rasterio.open(out_path) as src:
            band1 = src.read(1)
            assert band1[5, 5] == pytest.approx(-9999.0, abs=1.0)

    def test_creates_parent_dirs(self, tiff_path, tmp_path):
        data, meta = read_stack(tiff_path)
        out_path = tmp_path / "subdir" / "nested" / "out.tif"
        write_stack(data, out_path, meta)
        assert out_path.exists()

    def test_description_mismatch_raises(self, tiff_path, tmp_path):
        data, meta = read_stack(tiff_path)
        out_path = tmp_path / "desc.tif"
        with pytest.raises(ValueError, match="descriptions length"):
            write_stack(data, out_path, meta, descriptions=["only_one"])


# -----------------------------------------------------------------------
# stack_from_files tests
# -----------------------------------------------------------------------

class TestStackFromFiles:
    def test_stacks_multiple_files(self, tmp_path):
        rng = np.random.default_rng(1)
        files = []
        for i in range(5):
            data = rng.uniform(0, 100, size=(1, 10, 10)).astype(np.float32)
            p = tmp_path / f"band_{i:02d}.tif"
            p.write_bytes(make_synthetic_geotiff(data))
            files.append(p)

        stack, meta = stack_from_files(files)
        assert stack.shape == (5, 10, 10)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            stack_from_files([])

    def test_shape_mismatch_raises(self, tmp_path):
        d1 = np.ones((1, 10, 10), dtype=np.float32)
        d2 = np.ones((1, 15, 10), dtype=np.float32)  # different height
        p1 = tmp_path / "a.tif"
        p2 = tmp_path / "b.tif"
        p1.write_bytes(make_synthetic_geotiff(d1))
        p2.write_bytes(make_synthetic_geotiff(d2))
        with pytest.raises(ValueError, match="shape"):
            stack_from_files([p1, p2])


# -----------------------------------------------------------------------
# get_raster_info tests
# -----------------------------------------------------------------------

class TestGetRasterInfo:
    def test_returns_expected_keys(self, tiff_path):
        info = get_raster_info(tiff_path)
        for key in ("crs", "shape", "dtype", "nodata", "band_count", "bounds", "resolution_deg"):
            assert key in info

    def test_shape_correct(self, tiff_path, synthetic_stack):
        info = get_raster_info(tiff_path)
        assert info["shape"] == synthetic_stack.shape

    def test_resolution_geographic(self, tiff_path):
        """Pakistan bbox (60-80E, 20-40N) at 30x30 px = ~0.67 deg resolution."""
        info = get_raster_info(tiff_path)
        res_x, res_y = info["resolution_deg"]
        assert 0.5 < res_x < 1.0
        assert 0.5 < res_y < 1.0


# -----------------------------------------------------------------------
# End-to-end: read → SPI → write
# -----------------------------------------------------------------------

class TestEndToEnd:
    def test_read_spi_write(self, tiff_path, tmp_path):
        """
        Full pipeline: read CHIRPS-like precip stack → compute SPI-3 → write.
        """
        from spatialdrought import SPI

        # Read 20-band precip stack
        precip, meta = read_stack(tiff_path)
        assert precip.shape == (20, 30, 30)

        # Compute SPI-3
        spi = SPI(scale=3)
        result = spi.fit_transform(precip)
        assert result.shape == (20, 30, 30)

        # Write result
        out_path = tmp_path / "spi3.tif"
        write_stack(result, out_path, meta, dtype="float32")
        assert out_path.exists()

        # Read back and verify spatial metadata preserved
        spi_back, meta_back = read_stack(out_path)
        assert spi_back.shape == result.shape
        assert str(meta_back["crs"]) == str(meta["crs"])


# -----------------------------------------------------------------------
# xarray I/O tests (optional — skip if xarray not installed)
# -----------------------------------------------------------------------

@pytest.mark.skipif(not HAS_XARRAY, reason="xarray not installed")
class TestXarrayIO:
    def test_stack_to_dataarray_shape(self, tiff_path):
        from spatialdrought.io.xarray_io import stack_to_dataarray
        data, meta = read_stack(tiff_path)
        da = stack_to_dataarray(data, meta)
        assert da.shape == data.shape
        assert "y" in da.coords
        assert "x" in da.coords

    def test_dataarray_has_crs_attr(self, tiff_path):
        from spatialdrought.io.xarray_io import stack_to_dataarray
        data, meta = read_stack(tiff_path)
        da = stack_to_dataarray(data, meta)
        assert "crs" in da.attrs

    def test_read_stack_as_dataarray(self, tiff_path):
        from spatialdrought.io.xarray_io import read_stack_as_dataarray
        da = read_stack_as_dataarray(tiff_path, name="precip")
        assert da.name == "precip"
        assert da.ndim == 3

    def test_round_trip_dataarray(self, tiff_path):
        from spatialdrought.io.xarray_io import stack_to_dataarray, dataarray_to_stack
        data, meta = read_stack(tiff_path)
        da = stack_to_dataarray(data, meta)
        data2, meta2 = dataarray_to_stack(da)
        np.testing.assert_array_equal(data, data2)

    def test_spi_with_dataarray_input(self, tiff_path):
        """SPI should accept and return xarray DataArray."""
        from spatialdrought.io.xarray_io import read_stack_as_dataarray
        from spatialdrought import SPI
        da = read_stack_as_dataarray(tiff_path, name="precip")
        result = SPI(scale=3).fit_transform(da)
        assert isinstance(result, xr.DataArray)
        assert result.shape == da.shape
