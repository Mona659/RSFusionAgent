"""Artifact writers for YRE-151 single-patch inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin


def save_prediction_tiff(prediction: np.ndarray, output_path: str | Path) -> Path:
    """Save normalized CHW float32 data using the legacy synthetic transform."""

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
        transform=from_origin(0, 0, 1, 1),
    ) as dataset:
        dataset.write(array)
    return path


def hyperspectral_rgb(hsi: np.ndarray, bands: tuple[int, int, int] = (28, 18, 9)) -> np.ndarray:
    """Create the same percentile-stretched RGB preview used by the legacy test."""

    array = np.asarray(hsi, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"HSI must use CHW layout; got {array.shape}")
    if any(index < 0 or index >= array.shape[0] for index in bands):
        raise ValueError(f"RGB bands {bands} are invalid for {array.shape[0]} bands")

    rgb = np.zeros((array.shape[1], array.shape[2], 3), dtype=np.float32)
    for output_index, band_index in enumerate(bands):
        band = array[band_index]
        band_min, band_max = float(np.min(band)), float(np.max(band))
        if band_max > band_min:
            band = (band - band_min) / (band_max - band_min)
        low, high = np.percentile(band, (1, 99))
        if high > low:
            band = (band - low) / (high - low)
        rgb[..., output_index] = np.clip(band, 0.0, 1.0)
    return np.round(rgb * 255.0).astype(np.uint8)


def save_rgb_preview(
    hsi: np.ndarray,
    output_path: str | Path,
    bands: tuple[int, int, int] = (28, 18, 9),
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(hyperspectral_rgb(hsi, bands)).save(path)
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

