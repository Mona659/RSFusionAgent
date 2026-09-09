from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from rsfusion_agent.tools.tiff_crop import (
    CUSTOM_PROFILE,
    YRE_LEGACY_TEST_PROFILE,
    crop_tiff_triplet,
    resolve_crop_windows,
)


def _write_raster(path: Path, *, bands: int, width: int, height: int, resolution: float) -> None:
    pixels = np.arange(bands * height * width, dtype=np.int16).reshape(bands, height, width)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=bands,
        dtype=pixels.dtype,
        crs="EPSG:32650",
        transform=from_origin(1000.0, 2000.0, resolution, resolution),
    ) as dataset:
        dataset.write(pixels)


def test_legacy_yre_profile_matches_database_test_window() -> None:
    ms_window, hs_window = resolve_crop_windows(profile=YRE_LEGACY_TEST_PROFILE)

    assert ms_window.model_dump() == {
        "row_offset": 0,
        "col_offset": 360,
        "height": 540,
        "width": 540,
    }
    assert hs_window.model_dump() == {
        "row_offset": 0,
        "col_offset": 120,
        "height": 180,
        "width": 180,
    }


def test_crop_tiff_triplet_writes_custom_native_windows(tmp_path: Path) -> None:
    auxiliary_ms = tmp_path / "aux_ms.tif"
    auxiliary_hs = tmp_path / "aux_hs.tif"
    target_ms = tmp_path / "target_ms.tif"
    _write_raster(auxiliary_ms, bands=4, width=12, height=12, resolution=10.0)
    _write_raster(auxiliary_hs, bands=151, width=4, height=4, resolution=30.0)
    _write_raster(target_ms, bands=4, width=12, height=12, resolution=10.0)

    result = crop_tiff_triplet(
        auxiliary_ms,
        auxiliary_hs,
        target_ms,
        tmp_path / "crops",
        profile=CUSTOM_PROFILE,
        ms_row_offset=3,
        ms_col_offset=6,
        window_height=6,
        window_width=6,
        hs_row_offset=1,
        hs_col_offset=2,
    )

    assert Path(result.manifest_path).is_file()
    with rasterio.open(result.auxiliary_ms_path) as dataset:
        assert (dataset.count, dataset.width, dataset.height) == (4, 6, 6)
        assert tuple(dataset.transform)[:6] == (10.0, 0.0, 1060.0, 0.0, -10.0, 1970.0)
    with rasterio.open(result.auxiliary_hs_path) as dataset:
        assert (dataset.count, dataset.width, dataset.height) == (151, 2, 2)
        assert tuple(dataset.transform)[:6] == (30.0, 0.0, 1060.0, 0.0, -30.0, 1970.0)
