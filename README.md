# spatialdrought

Pixel-wise spatial drought index computation on gridded remote sensing data.

Computes SPI, SPEI, VCI, TCI, VHI, and CDI directly on raster stacks (numpy arrays or xarray DataArrays) — no pixel loops.

## Status
Under active development. SPI module complete with full test suite.

## Installation
```bash
pip install -e .
```

## Quick start
```python
import numpy as np
from spatialdrought import SPI

precip = np.random.gamma(2.5, 40.0, size=(240, 100, 100))  # (time, rows, cols)
spi = SPI(scale=3)
result = spi.fit_transform(precip)  # shape: (240, 100, 100)
```
