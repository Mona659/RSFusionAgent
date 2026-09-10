from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from rsfusion_agent.tools.raster_preview import render_raster_rgb


def _write_raster(path: Path, *, bands: int = 4) -> None:
    data = np.stack(
        [np.arange(120, dtype=np.float32).reshape(10, 12) + index for index in range(bands)]
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=12,
        count=bands,
        dtype="float32",
        transform=from_origin(100, 200, 10, 10),
        crs="EPSG:32650",
    ) as dataset:
        dataset.write(data)


def test_render_raster_rgb_reads_only_requested_bands(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    _write_raster(source)

    preview = render_raster_rgb(source, bands=(2, 1, 0), max_dimension=8)

    assert preview.shape == (7, 8, 3)
    assert preview.dtype == np.uint8


def test_render_raster_rgb_rejects_invalid_band(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    _write_raster(source)

    with pytest.raises(ValueError, match="invalid"):
        render_raster_rgb(source, bands=(0, 1, 4))
