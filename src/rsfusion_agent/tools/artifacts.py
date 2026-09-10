"""Artifact writers for YRE-151 single-patch inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from rasterio import Affine
from rasterio.transform import from_origin


def save_prediction_tiff(
    prediction: np.ndarray,
    output_path: str | Path,
    *,
    transform: Affine | None = None,
    crs: str | None = None,
) -> Path:
    """Save normalized CHW float32 data with optional source georeferencing."""

    array = np.asarray(prediction, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Prediction must use CHW layout; got {array.shape}")
    bands, height, width = array.shape
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype="float32",
        transform=transform or from_origin(0, 0, 1, 1),
        crs=crs,
    ) as dataset:
        dataset.write(array)
    return path


RgbStretchBounds = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


def hyperspectral_rgb_stretch_bounds(
    *images: np.ndarray,
    bands: tuple[int, int, int] = (28, 18, 9),
) -> RgbStretchBounds:
    """Calculate one shared percentile stretch for comparable HS RGB previews."""

    if not images:
        raise ValueError("At least one HSI is required to calculate RGB stretch bounds")
    arrays = [np.asarray(image, dtype=np.float32) for image in images]
    for array in arrays:
        if array.ndim != 3:
            raise ValueError(f"HSI must use CHW layout; got {array.shape}")
        if any(index < 0 or index >= array.shape[0] for index in bands):
            raise ValueError(f"RGB bands {bands} are invalid for {array.shape[0]} bands")

    bounds: list[tuple[float, float]] = []
    for band_index in bands:
        values = np.concatenate([array[band_index].reshape(-1) for array in arrays])
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            raise ValueError(f"RGB band {band_index} contains no finite values")
        low, high = np.percentile(finite_values, (1, 99))
        bounds.append((float(low), float(high)))
    return (bounds[0], bounds[1], bounds[2])


def hyperspectral_rgb(
    hsi: np.ndarray,
    bands: tuple[int, int, int] = (28, 18, 9),
    *,
    stretch_bounds: RgbStretchBounds | None = None,
) -> np.ndarray:
    """Create a percentile-stretched RGB preview from one CHW hyperspectral image."""

    array = np.asarray(hsi, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"HSI must use CHW layout; got {array.shape}")
    if any(index < 0 or index >= array.shape[0] for index in bands):
        raise ValueError(f"RGB bands {bands} are invalid for {array.shape[0]} bands")

    bounds = stretch_bounds or hyperspectral_rgb_stretch_bounds(array, bands=bands)
    rgb = np.zeros((array.shape[1], array.shape[2], 3), dtype=np.float32)
    for output_index, band_index in enumerate(bands):
        band = array[band_index]
        low, high = bounds[output_index]
        if high > low:
            band = (band - low) / (high - low)
        else:
            band = np.zeros_like(band)
        rgb[..., output_index] = np.clip(band, 0.0, 1.0)
    return np.round(rgb * 255.0).astype(np.uint8)


def save_rgb_preview(
    hsi: np.ndarray,
    output_path: str | Path,
    bands: tuple[int, int, int] = (28, 18, 9),
    *,
    stretch_bounds: RgbStretchBounds | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(hyperspectral_rgb(hsi, bands, stretch_bounds=stretch_bounds)).save(path)
    return path


def _sam_colormap(normalized: np.ndarray) -> np.ndarray:
    anchors = np.asarray(
        [
            [0, 0, 128],
            [30, 92, 255],
            [0, 212, 212],
            [85, 255, 0],
            [255, 153, 0],
            [255, 255, 0],
        ],
        dtype=np.float32,
    )
    positions = np.asarray([0.0, 0.15, 0.35, 0.55, 0.80, 1.0], dtype=np.float32)
    flat = normalized.reshape(-1)
    output = np.empty((flat.size, 3), dtype=np.float32)
    for index in range(len(positions) - 1):
        mask = (flat >= positions[index]) & (flat <= positions[index + 1])
        weight = (flat[mask] - positions[index]) / (positions[index + 1] - positions[index])
        output[mask] = anchors[index] + weight[:, None] * (anchors[index + 1] - anchors[index])
    return np.clip(output.reshape((*normalized.shape, 3)), 0, 255).astype(np.uint8)


def save_sam_heatmap(
    sam_map: np.ndarray,
    output_path: str | Path,
    *,
    maximum_radians: float = 1.0,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = np.clip(np.asarray(sam_map, dtype=np.float32) / maximum_radians, 0.0, 1.0)
    Image.fromarray(_sam_colormap(normalized)).save(path)
    return path


def write_json(data: Any, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
