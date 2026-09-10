"""Memory-bounded RGB previews for raw remote-sensing rasters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError

from rsfusion_agent.tools.artifacts import hyperspectral_rgb


def render_raster_rgb(
    image_path: str | Path,
    *,
    bands: tuple[int, int, int],
    max_dimension: int = 1024,
) -> np.ndarray:
    """Read three zero-based bands and return a percentile-stretched RGB preview.

    The source raster is resampled while reading, so an input check does not need
    to load a complete multi-band scene into memory.
    """

    if max_dimension < 1:
        raise ValueError("max_dimension must be positive")
    if len(bands) != 3:
        raise ValueError("Exactly three RGB band indices are required")

    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Raster does not exist: {path}")
    try:
        with rasterio.open(path) as dataset:
            if any(index < 0 or index >= dataset.count for index in bands):
                raise ValueError(f"RGB bands {bands} are invalid for {dataset.count} bands")
            scale = min(1.0, max_dimension / max(dataset.width, dataset.height))
            output_height = max(1, round(dataset.height * scale))
            output_width = max(1, round(dataset.width * scale))
            selected = dataset.read(
                indexes=tuple(index + 1 for index in bands),
                out_shape=(3, output_height, output_width),
                out_dtype="float32",
                resampling=Resampling.bilinear,
            )
    except RasterioIOError as exc:
        raise ValueError(f"Cannot open raster '{path}': {exc}") from exc

    return hyperspectral_rgb(selected, bands=(0, 1, 2))
