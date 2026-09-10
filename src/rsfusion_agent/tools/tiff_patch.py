"""Prepare one YRE-151 model patch directly from a metadata-validated TIFF triplet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from pydantic import BaseModel, Field
from rasterio.enums import Resampling
from rasterio.windows import Window

from rsfusion_agent.tools.h5_patch import HS_BANDS, MS_BANDS, NORM_FACTOR, SCALE
from rsfusion_agent.tools.raster_inspector import inspect_raster
from rsfusion_agent.tools.tiff_crop import TiffCropResult, load_crop_manifest
from rsfusion_agent.tools.tiff_triplet import TiffTripletInspection, inspect_tiff_triplet


class TiffPatchSpatialMetadata(BaseModel):
    """Target-MS spatial metadata inherited by the fused TIFF output."""

    crs: str
    transform: tuple[float, float, float, float, float, float]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    alignment_mode: str = "strict_geospatial"


@dataclass(frozen=True)
class PreparedTiffPatch:
    """Normalized CHW tensors and output georeferencing for one raw TIFF crop."""

    auxiliary_ms: np.ndarray
    auxiliary_hs_interpolated: np.ndarray
    target_ms: np.ndarray
    target_hs_reference_interpolated: np.ndarray | None
    inspection: TiffTripletInspection
    spatial_metadata: TiffPatchSpatialMetadata
    row_offset: int
    col_offset: int
    crop_manifest_path: str | None = None

    def save_npz(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            auxiliary_ms=self.auxiliary_ms,
            auxiliary_hs_interpolated=self.auxiliary_hs_interpolated,
            target_ms=self.target_ms,
            **(
                {"target_hs_reference_interpolated": self.target_hs_reference_interpolated}
                if self.target_hs_reference_interpolated is not None
                else {}
            ),
        )
        return output


def _validate_patch_window(
    *,
    width: int,
    height: int,
    patch_size: int,
    row_offset: int,
    col_offset: int,
) -> None:
    if patch_size <= 0 or patch_size % SCALE:
        raise ValueError(f"Patch size must be positive and divisible by {SCALE}")
    if row_offset < 0 or col_offset < 0:
        raise ValueError("Patch row and column offsets must be non-negative")
    if row_offset % SCALE or col_offset % SCALE:
        raise ValueError(f"Patch row and column offsets must be divisible by {SCALE}")
    if row_offset + patch_size > height or col_offset + patch_size > width:
        raise ValueError(
            f"Patch window row={row_offset}, col={col_offset}, size={patch_size} exceeds "
            f"MS extent {width}x{height}"
        )


def _read_window(path: str, window: Window, *, out_height: int, out_width: int) -> np.ndarray:
    with rasterio.open(path) as dataset:
        array = dataset.read(
            window=window,
            out_shape=(dataset.count, out_height, out_width),
            resampling=Resampling.bilinear,
            out_dtype="float32",
        )
    if not np.isfinite(array).all():
        raise ValueError(f"Raster patch contains NaN or Inf: {path}")
    return np.ascontiguousarray(array, dtype=np.float32)


def inspect_manifest_crop_triplet(crop_manifest_path: str | Path) -> tuple[TiffTripletInspection, TiffCropResult]:
    """Validate a manifest's three crop artifacts under an explicit pixel-alignment rule.

    A crop manifest records the source-pixel correspondence inherited from the legacy
    data script. It is therefore an explicit opt-in alternative to strict bounds/CRS
    equality, never an automatic registration result.
    """

    crop = load_crop_manifest(crop_manifest_path)
    auxiliary_ms = inspect_raster(crop.auxiliary_ms_path)
    auxiliary_hs = inspect_raster(crop.auxiliary_hs_path)
    target_ms = inspect_raster(crop.target_ms_path)
    target_hs_reference = (
        inspect_raster(crop.target_hs_reference_path)
        if crop.target_hs_reference_path is not None
        else None
    )
    issues: list[str] = []
    if auxiliary_ms.band_count != MS_BANDS or target_ms.band_count != MS_BANDS:
        issues.append(f"Manifest MS crops must each contain {MS_BANDS} bands.")
    if auxiliary_hs.band_count != HS_BANDS:
        issues.append(f"Manifest auxiliary HS crop must contain {HS_BANDS} bands.")
    if target_hs_reference is not None:
        if target_hs_reference.band_count != HS_BANDS:
            issues.append(f"Manifest target HS reference crop must contain {HS_BANDS} bands.")
        if (
            target_hs_reference.width != auxiliary_hs.width
            or target_hs_reference.height != auxiliary_hs.height
        ):
            issues.append("Manifest target HS reference crop must match auxiliary HS crop dimensions.")
    if auxiliary_ms.width != target_ms.width or auxiliary_ms.height != target_ms.height:
        issues.append("Manifest auxiliary and target MS crops must have matching dimensions.")
    if auxiliary_ms.width != auxiliary_hs.width * SCALE or auxiliary_ms.height != auxiliary_hs.height * SCALE:
        issues.append(
            f"Manifest auxiliary HS dimensions must be 1/{SCALE} of auxiliary MS dimensions."
        )
    if auxiliary_ms.width % SCALE or auxiliary_ms.height % SCALE:
        issues.append(f"Manifest MS crop dimensions must be divisible by {SCALE}.")
    if issues:
        raise ValueError("Crop manifest is not model-compatible: " + " ".join(issues))
    warnings = list(crop.warnings)
    warnings.append(
        "The crop manifest authorizes source-pixel correspondence for this run; "
        "native CRS/bounds equality was intentionally not used as a registration test."
    )
    return (
        TiffTripletInspection(
            auxiliary_ms=auxiliary_ms,
            auxiliary_hs=auxiliary_hs,
            target_ms=target_ms,
            scale=SCALE,
            expected_ms_bands=MS_BANDS,
            expected_hs_bands=HS_BANDS,
            grid_tolerance=0.0,
            is_ready_for_preprocessing=True,
            blocking_issues=[],
            warnings=warnings,
        ),
        crop,
    )


def _prepare_from_inspection(
    inspection: TiffTripletInspection,
    *,
    patch_size: int,
    row_offset: int,
    col_offset: int,
    alignment_mode: str,
    crop_manifest_path: str | None = None,
    target_hs_reference_path: str | None = None,
) -> PreparedTiffPatch:
    _validate_patch_window(
        width=inspection.target_ms.width,
        height=inspection.target_ms.height,
        patch_size=patch_size,
        row_offset=row_offset,
        col_offset=col_offset,
    )
    high_window = Window(col_offset, row_offset, patch_size, patch_size)
    low_window = Window(
        col_offset // SCALE,
        row_offset // SCALE,
        patch_size // SCALE,
        patch_size // SCALE,
    )
    auxiliary_ms = _read_window(
        inspection.auxiliary_ms.path,
        high_window,
        out_height=patch_size,
        out_width=patch_size,
    )
    auxiliary_hs = _read_window(
        inspection.auxiliary_hs.path,
        low_window,
        out_height=patch_size,
        out_width=patch_size,
    )
    target_ms = _read_window(
        inspection.target_ms.path,
        high_window,
        out_height=patch_size,
        out_width=patch_size,
    )
    target_hs_reference_interpolated = None
    if target_hs_reference_path is not None:
        target_hs_reference_interpolated = _read_window(
            target_hs_reference_path,
            low_window,
            out_height=patch_size,
            out_width=patch_size,
        )
    with rasterio.open(inspection.target_ms.path) as target_dataset:
        transform = target_dataset.window_transform(high_window)

    return PreparedTiffPatch(
        auxiliary_ms=np.ascontiguousarray(auxiliary_ms / np.float32(NORM_FACTOR), dtype=np.float32),
        auxiliary_hs_interpolated=np.ascontiguousarray(
            auxiliary_hs / np.float32(NORM_FACTOR), dtype=np.float32
        ),
        target_ms=np.ascontiguousarray(target_ms / np.float32(NORM_FACTOR), dtype=np.float32),
        target_hs_reference_interpolated=(
            np.ascontiguousarray(
                target_hs_reference_interpolated / np.float32(NORM_FACTOR), dtype=np.float32
            )
            if target_hs_reference_interpolated is not None
            else None
        ),
        inspection=inspection,
        spatial_metadata=TiffPatchSpatialMetadata(
            crs=inspection.target_ms.crs or "",
            transform=tuple(float(value) for value in transform[:6]),
            width=patch_size,
            height=patch_size,
            alignment_mode=alignment_mode,
        ),
        row_offset=row_offset,
        col_offset=col_offset,
        crop_manifest_path=crop_manifest_path,
    )


def prepare_tiff_patch(
    auxiliary_ms_path: str | Path,
    auxiliary_hs_path: str | Path,
    target_ms_path: str | Path,
    *,
    target_hs_reference_path: str | Path | None = None,
    patch_size: int = 180,
    row_offset: int = 0,
    col_offset: int = 0,
) -> PreparedTiffPatch:
    """Create the existing runtime NPZ input from one raw TIFF triplet crop.

    The auxiliary HS crop is explicitly bilinearly upsampled by three, matching the
    legacy ``Database.py`` behavior before the isolated runtime downsamples it for
    DC-STSF. No registration or reprojection is attempted.
    """

    inspection = inspect_tiff_triplet(
        auxiliary_ms_path,
        auxiliary_hs_path,
        target_ms_path,
        scale=SCALE,
        expected_ms_bands=MS_BANDS,
        expected_hs_bands=HS_BANDS,
    )
    if not inspection.is_ready_for_preprocessing:
        details = " ".join(inspection.blocking_issues)
        raise ValueError(f"TIFF triplet is not ready for preprocessing: {details}")
    if target_hs_reference_path is not None:
        target_reference = inspect_raster(target_hs_reference_path)
        if target_reference.band_count != HS_BANDS:
            raise ValueError(f"Target HS reference must have {HS_BANDS} bands")
    return _prepare_from_inspection(
        inspection,
        patch_size=patch_size,
        row_offset=row_offset,
        col_offset=col_offset,
        alignment_mode="strict_geospatial",
        target_hs_reference_path=(
            str(Path(target_hs_reference_path).expanduser().resolve())
            if target_hs_reference_path is not None
            else None
        ),
    )


def prepare_tiff_patch_from_manifest(
    crop_manifest_path: str | Path,
    *,
    patch_size: int = 180,
    row_offset: int = 0,
    col_offset: int = 0,
) -> PreparedTiffPatch:
    """Prepare one model patch from a user-authored, validated crop manifest."""

    inspection, crop = inspect_manifest_crop_triplet(crop_manifest_path)
    return _prepare_from_inspection(
        inspection,
        patch_size=patch_size,
        row_offset=row_offset,
        col_offset=col_offset,
        alignment_mode="explicit_crop_manifest",
        crop_manifest_path=str(Path(crop_manifest_path).expanduser().resolve()),
        target_hs_reference_path=crop.target_hs_reference_path,
    )
