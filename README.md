# spatialdrought

[![tests](https://github.com/mahmoodirfan/spatialdrought/actions/workflows/tests.yml/badge.svg)](https://github.com/mahmoodirfan/spatialdrought/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://pypi.org/project/spatialdrought/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Pixel-wise spatial drought index computation on gridded remote sensing data.

Computes **SPI, SPEI, VCI, TCI, VHI, and CDI** directly on raster stacks —
no pixel loops, no GIS software required.

## Why spatialdrought?

Every existing drought index implementation works on 1D time series (point data).
`spatialdrought` operates natively on `(time, rows, cols)` numpy arrays and
xarray DataArrays, making it suitable for:

- CHIRPS precipitation rasters → SPI/SPEI
- MODIS MOD13A3 NDVI → VCI
- MODIS MOD11A2 LST → TCI
- Combined → VHI, CDI

## Installation

```bash
pip install spatialdrought
```

With rasterio I/O support:
```bash
pip install spatialdrought[io]
```

## Quick start

```python
import numpy as np
from spatialdrought import SPI, SPEI, VCI, TCI, VHI, CDI

# SPI on a 20-year monthly precipitation raster (240 months, 100x100 grid)
precip = np.random.gamma(2.5, 40.0, size=(240, 100, 100))
spi = SPI(scale=3)
spi_result = spi.fit_transform(precip)  # shape: (240, 100, 100)

# SPEI (requires PET)
from spatialdrought.utils import hargreaves_pet
# ... compute or load PET, then:
# spei_result = SPEI(scale=3).fit_transform(precip, pet)

# VHI from NDVI and LST
ndvi = np.random.uniform(0.1, 0.8, size=(240, 100, 100))
lst  = np.random.uniform(280, 320, size=(240, 100, 100))
vhi_result = VHI(alpha=0.5).fit_transform(ndvi, lst)

# CDI (composite)
cdi_result = CDI().fit_transform(spi_result, vhi_result, ndvi)
# Returns: 0=no drought, 1=watch, 2=warning, 3=alert
```

## Reading GeoTIFFs

```python
from spatialdrought.io import read_stack, write_stack

# Read CHIRPS monthly stack
precip, meta = read_stack("chirps_monthly.tif")

# Compute SPI
spi_result = SPI(scale=3).fit_transform(precip)

# Write result with CRS and transform preserved
write_stack(spi_result, "spi3_output.tif", meta)
```

## Indices

| Index | Input | Method |
|-------|-------|--------|
| SPI   | Precipitation | Gamma distribution (Thom 1958 MLE) |
| SPEI  | P - PET | Log-logistic via L-moments (Hosking 1990) |
| VCI   | NDVI | Min-max rescaling (Kogan 1995) |
| TCI   | LST  | Inverted min-max rescaling (Kogan 1995) |
| VHI   | NDVI + LST | Weighted VCI+TCI composite |
| CDI   | SPI + VHI + NDVI | Hierarchical watch/warning/alert |

All indices operate per pixel per calendar month — January is compared
to historical Januaries, not to the full annual distribution.

## Citation

If you use this library in published research, please cite:

> Mahmood, I. et al. (2025). spatialdrought: A Python library for
> pixel-wise spatial drought index computation on gridded remote sensing data.
> *SoftwareX* (under preparation).

## License

MIT
<img width="4015" height="2989" alt="pakistan_spi3_final" src="https://github.com/user-attachments/assets/315873f6-1309-48f6-aa06-a22170df367e" />
