"""Create a tiny deterministic GeoTIFF for the quick-start example."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


def main() -> None:
    output_path = Path(__file__).parent / "data" / "demo_multispectral.tif"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
    bands = np.stack((base, np.flipud(base), np.fliplr(base)))

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        width=32,
        height=32,
        count=3,
        dtype=bands.dtype,
        crs="EPSG:4326",
        transform=from_origin(120.0, 30.0, 0.0001, 0.0001),
        nodata=0,
        compress="lzw",
    ) as dataset:
        dataset.write(bands)

    print(output_path.resolve())


if __name__ == "__main__":
    main()

