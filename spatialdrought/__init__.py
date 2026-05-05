"""
spatialdrought: Pixel-wise spatial drought index computation on gridded remote sensing data.

Indices: SPI, SPEI, VCI, TCI, VHI, CDI
Input:   numpy arrays, xarray DataArrays, or GeoTIFF files
Output:  numpy arrays, xarray DataArrays, or GeoTIFF files
"""

__version__ = "0.1.0"

from spatialdrought.indices.spi import SPI
from spatialdrought.indices.spei import SPEI
from spatialdrought.indices.vci import VCI
from spatialdrought.indices.tci import TCI
from spatialdrought.indices.vhi import VHI
from spatialdrought.indices.cdi import CDI

__all__ = ["SPI", "SPEI", "VCI", "TCI", "VHI", "CDI"]
