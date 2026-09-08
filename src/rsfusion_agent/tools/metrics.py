"""Full-reference metrics compatible with the legacy YRE test script."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from skimage.metrics import structural_similarity


class FusionMetrics(BaseModel):
    psnr: float
    rmse: float
    sam: float
    ergas: float
    ssim: float
    cc: float


def _validate_pair(target: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape:
        raise ValueError(f"Metric inputs must have the same shape: {target.shape} != {prediction.shape}")
    if target.ndim != 3:
        raise ValueError(f"Metric inputs must use CHW layout; got shape {target.shape}")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Metric inputs contain NaN or Inf")
    return target, prediction


def calculate_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    scale: int = 3,
) -> FusionMetrics:
    """Calculate the six metrics reported by ``test+sam+rgb.py``."""

    target, prediction = _validate_pair(target, prediction)
    epsilon = np.finfo(np.float64).eps
    difference = target - prediction
    mse = float(np.mean(difference**2))
    target_max = float(np.max(target))
    psnr = float("inf") if mse <= epsilon else float(10.0 * np.log10(target_max**2 / mse))

    normalized_scale = max(abs(target_max), epsilon)
    rmse = float(np.sqrt(np.mean((target / normalized_scale - prediction / normalized_scale) ** 2)))

    target_flat = target.reshape(target.shape[0], -1)
    prediction_flat = prediction.reshape(prediction.shape[0], -1)
    target_norm = np.linalg.norm(target_flat, axis=0)
    prediction_norm = np.linalg.norm(prediction_flat, axis=0)
    spectral_denominator = target_norm * prediction_norm
    cosine = np.divide(
        np.sum(target_flat * prediction_flat, axis=0),
        spectral_denominator,
        out=np.ones_like(spectral_denominator),
        where=spectral_denominator > epsilon,
    )
    sam = float(np.mean(np.arccos(np.clip(cosine, -1.0, 1.0))) * 180.0 / np.pi)

    rmse_per_band = np.sqrt(np.mean(difference.reshape(target.shape[0], -1) ** 2, axis=1))
    mean_per_band = np.mean(np.abs(target_flat), axis=1)
    ergas = float(100.0 / scale * np.sqrt(np.mean((rmse_per_band / (mean_per_band + epsilon)) ** 2)))

    ssim_values = [
        structural_similarity(target[index], prediction[index], data_range=1.0)
        for index in range(target.shape[0])
    ]
    ssim_value = float(np.mean(ssim_values))

    target_centered = target_flat - np.mean(target_flat, axis=1, keepdims=True)
    prediction_centered = prediction_flat - np.mean(prediction_flat, axis=1, keepdims=True)
    numerator = np.sum(target_centered * prediction_centered, axis=1)
    denominator = np.sqrt(
        np.sum(target_centered**2, axis=1) * np.sum(prediction_centered**2, axis=1)
    )
    cc = float(np.mean(numerator / (denominator + 1e-8)))
    return FusionMetrics(psnr=psnr, rmse=rmse, sam=sam, ergas=ergas, ssim=ssim_value, cc=cc)


def calculate_sam_map(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Return a per-pixel spectral-angle map in radians."""

    target, prediction = _validate_pair(target, prediction)
    target_hwc = np.moveaxis(target, 0, -1)
    prediction_hwc = np.moveaxis(prediction, 0, -1)
    numerator = np.sum(target_hwc * prediction_hwc, axis=-1)
    denominator = np.linalg.norm(target_hwc, axis=-1) * np.linalg.norm(
        prediction_hwc, axis=-1
    )
    cosine = np.divide(
        numerator,
        denominator,
        out=np.ones_like(denominator),
        where=denominator > 1e-8,
    )
    return np.arccos(np.clip(cosine, -1.0, 1.0)).astype(np.float32)
