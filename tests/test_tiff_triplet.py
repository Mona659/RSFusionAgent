from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from rsfusion_agent.tools.tiff_triplet import inspect_tiff_triplet


def _write_raster(
    path: Path,
    *,
    bands: int,
    width: int,
    height: int,
    resolution: float,
) -> None:
    pixels = np.ones((bands, height, width), dtype=np.uint16)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=bands,
        dtype=pixels.dtype,
        crs="EPSG:4326",
        transform=from_origin(120.0, 30.0, resolution, resolution),
    ) as dataset:
        dataset.write(pixels)


def test_inspect_tiff_triplet_accepts_expected_yre_geometry(tmp_path: Path) -> None:
    auxiliary_ms = tmp_path / "aux_ms.tif"
    auxiliary_hs = tmp_path / "aux_hs.tif"
    target_ms = tmp_path / "target_ms.tif"
    _write_raster(auxiliary_ms, bands=4, width=12, height=9, resolution=0.5)
    _write_raster(auxiliary_hs, bands=151, width=4, height=3, resolution=1.5)
    _write_raster(target_ms, bands=4, width=12, height=9, resolution=0.5)

    result = inspect_tiff_triplet(auxiliary_ms, auxiliary_hs, target_ms)

    assert result.is_ready_for_preprocessing is True
    assert result.blocking_issues == []
    assert len(result.warnings) == 1


def test_inspect_tiff_triplet_reports_grid_and_band_failures(tmp_path: Path) -> None:
    auxiliary_ms = tmp_path / "aux_ms.tif"
    auxiliary_hs = tmp_path / "aux_hs.tif"
    target_ms = tmp_path / "target_ms.tif"
    _write_raster(auxiliary_ms, bands=4, width=12, height=9, resolution=0.5)
    _write_raster(auxiliary_hs, bands=150, width=4, height=3, resolution=1.5)
    _write_raster(target_ms, bands=4, width=10, height=9, resolution=0.5)

    result = inspect_tiff_triplet(auxiliary_ms, auxiliary_hs, target_ms)

    assert result.is_ready_for_preprocessing is False
    assert any("150 bands" in issue for issue in result.blocking_issues)
    assert any("share CRS, bounds, dimensions" in issue for issue in result.blocking_issues)
