from spatialdrought.io.raster_io import (
    read_stack,
    write_stack,
    read_single_band,
    stack_from_files,
    get_raster_info,
)

try:
    from spatialdrought.io.xarray_io import (
        stack_to_dataarray,
        read_stack_as_dataarray,
        dataarray_to_stack,
    )
    HAS_XARRAY_IO = True
except ImportError:
    HAS_XARRAY_IO = False

__all__ = [
    "read_stack",
    "write_stack",
    "read_single_band",
    "stack_from_files",
    "get_raster_info",
    "stack_to_dataarray",
    "read_stack_as_dataarray",
    "dataarray_to_stack",
]
