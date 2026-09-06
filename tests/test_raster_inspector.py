from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from rsfusion_agent.tools.raster_inspector import inspect_raster


def test_inspect_raster_returns_structured_metadata(tmp_path: Path) -> None:
    raster_path = tmp_path / "sample.tif"
    pixels = np.ones((2, 8, 10), dtype=np.uint16)

    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        width=10,
        height=8,
        count=2,
        dtype=pixels.dtype,
        crs="EPSG:4326",
        transform=from_origin(120.0, 30.0, 0.5, 0.5),
        nodata=0,
    ) as dataset:
        dataset.write(pixels)

    result = inspect_raster(raster_path)

    assert result.filename == "sample.tif"
    assert result.driver == "GTiff"
    assert result.width == 10
    assert result.height == 8
    assert result.band_count == 2
    assert result.dtypes == ["uint16", "uint16"]
    assert result.crs == "EPSG:4326"
    assert result.resolution == (0.5, 0.5)
    assert result.is_georeferenced is True


def test_inspect_raster_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Raster does not exist"):
        inspect_raster(tmp_path / "missing.tif")

