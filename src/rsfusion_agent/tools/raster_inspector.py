"""Raster metadata inspection tool.

The tool intentionally reads metadata only. Large remote-sensing rasters are not
loaded into memory, which keeps it safe to call during an agent planning step.
"""

from __future__ import annotations

from pathlib import Path

import rasterio
from pydantic import BaseModel, Field
from rasterio.errors import RasterioIOError


class RasterBounds(BaseModel):
    """Spatial bounds in the raster coordinate reference system."""

    left: float
    bottom: float
    right: float
    top: float


class RasterInspectionResult(BaseModel):
    """Structured metadata returned to an agent or API caller."""

    path: str
    filename: str
    driver: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    band_count: int = Field(gt=0)
    dtypes: list[str]
    crs: str | None
    bounds: RasterBounds
    resolution: tuple[float, float]
    nodata: float | None
    compression: str | None
    file_size_bytes: int = Field(ge=0)
    is_georeferenced: bool


def inspect_raster(image_path: str | Path) -> RasterInspectionResult:
    """Inspect a GDAL-readable raster and return JSON-friendly metadata.

    Args:
        image_path: Local path to a GeoTIFF or another format supported by Rasterio.

    Raises:
        FileNotFoundError: If ``image_path`` does not exist or is not a file.
        ValueError: If Rasterio cannot open the file as a raster.
    """

    path = Path(image_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Raster does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Raster path is not a file: {path}")

    resolved_path = path.resolve()
    try:
        with rasterio.open(resolved_path) as dataset:
            bounds = dataset.bounds
            compression = dataset.compression.value if dataset.compression else None
            crs = dataset.crs.to_string() if dataset.crs else None

            return RasterInspectionResult(
                path=str(resolved_path),
                filename=resolved_path.name,
                driver=dataset.driver,
                width=dataset.width,
                height=dataset.height,
                band_count=dataset.count,
                dtypes=list(dataset.dtypes),
                crs=crs,
                bounds=RasterBounds(
                    left=bounds.left,
                    bottom=bounds.bottom,
                    right=bounds.right,
                    top=bounds.top,
                ),
                resolution=(abs(dataset.transform.a), abs(dataset.transform.e)),
                nodata=dataset.nodata,
                compression=compression,
                file_size_bytes=resolved_path.stat().st_size,
                is_georeferenced=dataset.crs is not None,
            )
    except RasterioIOError as exc:
        raise ValueError(f"Cannot open raster '{resolved_path}': {exc}") from exc

