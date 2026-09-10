"""Faithfully reproduce the real and simulated YRE TIFF experiment contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio import Affine
from rasterio.windows import Window

from rsfusion_agent.tools.h5_patch import NORM_FACTOR, SCALE
from rsfusion_agent.tools.tiff_crop import (
    REAL_EXPERIMENT,
    SIMULATION_EXPERIMENT,
    ExperimentMode,
    TiffCropResult,
)
from rsfusion_agent.tools.tiff_patch import (
    TiffPatchSpatialMetadata,
    inspect_manifest_crop_triplet,
)
from rsfusion_agent.tools.tiff_triplet import TiffTripletInspection

SIMULATION_PATCH_SIZE = 180
REAL_PATCH_SIZE = 540
GAUSSIAN_KERNEL_SIZE = 7
GAUSSIAN_SIGMA = 5.0


@dataclass(frozen=True)
class TiffPatchSelection:
    """A selectable non-overlapping patch in the output grid of one experiment."""

    patch_index: int | None
    row_offset: int
    col_offset: int
    patch_size: int
    grid_rows: int
    grid_columns: int
    total_patch_count: int


@dataclass(frozen=True)
class PreparedExperimentPatch:
    """Common-grid runtime tensors plus experiment and patch provenance."""

    auxiliary_ms: np.ndarray
    auxiliary_hs_interpolated: np.ndarray
    target_ms: np.ndarray
    target_hs_reference: np.ndarray | None
    inspection: TiffTripletInspection
    spatial_metadata: TiffPatchSpatialMetadata
    experiment_mode: ExperimentMode
    reference_kind: str
    selection: TiffPatchSelection
    crop_manifest_path: str | None

    def save_npz(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            auxiliary_ms=self.auxiliary_ms,
            auxiliary_hs_interpolated=self.auxiliary_hs_interpolated,
            target_ms=self.target_ms,
            experiment_mode=np.asarray(self.experiment_mode),
            **(
                {"target_hs_reference": self.target_hs_reference}
                if self.target_hs_reference is not None
                else {}
            ),
        )
        return output


def default_patch_size(experiment_mode: ExperimentMode) -> int:
    return SIMULATION_PATCH_SIZE if experiment_mode == SIMULATION_EXPERIMENT else REAL_PATCH_SIZE


def output_grid_shape(
    *, ms_height: int, ms_width: int, experiment_mode: ExperimentMode
) -> tuple[int, int]:
    """Resolve model output dimensions after Database.py preprocessing."""

    if ms_height <= 0 or ms_width <= 0 or ms_height % SCALE or ms_width % SCALE:
        raise ValueError(f"MS dimensions must be positive and divisible by {SCALE}")
    if experiment_mode == SIMULATION_EXPERIMENT:
        return ms_height // SCALE, ms_width // SCALE
    return ms_height, ms_width


def select_patch(
    *,
    output_height: int,
    output_width: int,
    experiment_mode: ExperimentMode,
    patch_size: int | None = None,
    patch_index: int | None = None,
    row_offset: int = 0,
    col_offset: int = 0,
) -> TiffPatchSelection:
    """Resolve an indexed patch, or preserve an explicit legacy row/column window."""

    resolved_size = patch_size or default_patch_size(experiment_mode)
    if resolved_size <= 0 or resolved_size % SCALE:
        raise ValueError(f"Patch size must be positive and divisible by {SCALE}")
    grid_rows = output_height // resolved_size
    grid_columns = output_width // resolved_size
    total = grid_rows * grid_columns
    if total < 1:
        raise ValueError(
            f"Patch size {resolved_size} exceeds output extent {output_width}x{output_height}"
        )
    if patch_index is not None:
        if patch_index < 0 or patch_index >= total:
            raise ValueError(f"Patch index {patch_index} is outside [0, {total - 1}]")
        row_offset = (patch_index // grid_columns) * resolved_size
        col_offset = (patch_index % grid_columns) * resolved_size
    if row_offset < 0 or col_offset < 0:
        raise ValueError("Patch row and column offsets must be non-negative")
    if row_offset % SCALE or col_offset % SCALE:
        raise ValueError(f"Patch offsets must be divisible by {SCALE}")
    if row_offset + resolved_size > output_height or col_offset + resolved_size > output_width:
        raise ValueError(
            f"Patch window row={row_offset}, col={col_offset}, size={resolved_size} exceeds "
            f"output extent {output_width}x{output_height}"
        )
    return TiffPatchSelection(
        patch_index=patch_index,
        row_offset=row_offset,
        col_offset=col_offset,
        patch_size=resolved_size,
        grid_rows=grid_rows,
        grid_columns=grid_columns,
        total_patch_count=total,
    )


def _read(path: str, window: Window) -> np.ndarray:
    with rasterio.open(path) as dataset:
        array = dataset.read(window=window, out_dtype="float32")
    if not np.isfinite(array).all():
        raise ValueError(f"Raster data contains NaN or Inf: {path}")
    return np.ascontiguousarray(array, dtype=np.float32)


def _resize(array: np.ndarray, *, height: int, width: int) -> np.ndarray:
    output = np.empty((array.shape[0], height, width), dtype=np.float32)
    for band_index, band in enumerate(array):
        output[band_index] = cv2.resize(band, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(output)


def _blur(array: np.ndarray) -> np.ndarray:
    output = np.empty_like(array, dtype=np.float32)
    for band_index, band in enumerate(array):
        output[band_index] = cv2.GaussianBlur(
            band,
            (GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE),
            sigmaX=GAUSSIAN_SIGMA,
            sigmaY=GAUSSIAN_SIGMA,
            borderType=cv2.BORDER_DEFAULT,
        )
    return np.ascontiguousarray(output)


def _normal(array: np.ndarray | None) -> np.ndarray | None:
    if array is None:
        return None
    return np.ascontiguousarray(array / np.float32(NORM_FACTOR), dtype=np.float32)


def _metadata(
    *, target_ms_path: str, source_window: Window, output_size: int, pixel_scale: int
) -> TiffPatchSpatialMetadata:
    with rasterio.open(target_ms_path) as dataset:
        transform = dataset.window_transform(source_window) * Affine.scale(pixel_scale, pixel_scale)
        crs = dataset.crs.to_string() if dataset.crs is not None else ""
    return TiffPatchSpatialMetadata(
        crs=crs,
        transform=tuple(float(value) for value in transform[:6]),
        width=output_size,
        height=output_size,
        alignment_mode="explicit_crop_manifest",
    )


def _prepare_real(
    inspection: TiffTripletInspection,
    crop: TiffCropResult,
    selection: TiffPatchSelection,
    manifest_path: str,
) -> PreparedExperimentPatch:
    high_window = Window(
        selection.col_offset,
        selection.row_offset,
        selection.patch_size,
        selection.patch_size,
    )
    low_window = Window(
        selection.col_offset // SCALE,
        selection.row_offset // SCALE,
        selection.patch_size // SCALE,
        selection.patch_size // SCALE,
    )
    auxiliary_hs = _resize(
        _read(inspection.auxiliary_hs.path, low_window),
        height=selection.patch_size,
        width=selection.patch_size,
    )
    target_reference = None
    reference_kind = "unavailable_without_target_hs_reference"
    if crop.target_hs_reference_path is not None:
        target_reference = _resize(
            _read(crop.target_hs_reference_path, low_window),
            height=selection.patch_size,
            width=selection.patch_size,
        )
        reference_kind = "interpolated_target_hs_pseudo_reference"
    return PreparedExperimentPatch(
        auxiliary_ms=_normal(_read(inspection.auxiliary_ms.path, high_window)),
        auxiliary_hs_interpolated=_normal(auxiliary_hs),
        target_ms=_normal(_read(inspection.target_ms.path, high_window)),
        target_hs_reference=_normal(target_reference),
        inspection=inspection,
        spatial_metadata=_metadata(
            target_ms_path=inspection.target_ms.path,
            source_window=high_window,
            output_size=selection.patch_size,
            pixel_scale=1,
        ),
        experiment_mode=REAL_EXPERIMENT,
        reference_kind=reference_kind,
        selection=selection,
        crop_manifest_path=manifest_path,
    )


def _prepare_simulation(
    inspection: TiffTripletInspection,
    crop: TiffCropResult,
    selection: TiffPatchSelection,
    manifest_path: str,
) -> PreparedExperimentPatch:
    if crop.target_hs_reference_path is None:
        raise ValueError("Simulation experiment requires target-time HS ground truth")
    output_height, output_width = output_grid_shape(
        ms_height=inspection.target_ms.height,
        ms_width=inspection.target_ms.width,
        experiment_mode=SIMULATION_EXPERIMENT,
    )
    full_ms_window = Window(0, 0, inspection.target_ms.width, inspection.target_ms.height)
    full_hs_window = Window(0, 0, inspection.auxiliary_hs.width, inspection.auxiliary_hs.height)
    auxiliary_ms_full = _resize(
        _blur(_read(inspection.auxiliary_ms.path, full_ms_window)),
        height=output_height,
        width=output_width,
    )
    target_ms_full = _resize(
        _blur(_read(inspection.target_ms.path, full_ms_window)),
        height=output_height,
        width=output_width,
    )
    auxiliary_hs_low = _resize(
        _blur(_read(inspection.auxiliary_hs.path, full_hs_window)),
        height=output_height // SCALE,
        width=output_width // SCALE,
    )
    auxiliary_hs_full = _resize(auxiliary_hs_low, height=output_height, width=output_width)
    target_reference_full = _read(crop.target_hs_reference_path, full_hs_window)
    if target_reference_full.shape[1:] != (output_height, output_width):
        raise ValueError("Simulation target-HS crop must match the reduced output grid")
    row, col, size = selection.row_offset, selection.col_offset, selection.patch_size
    high_window = Window(col * SCALE, row * SCALE, size * SCALE, size * SCALE)
    return PreparedExperimentPatch(
        auxiliary_ms=_normal(auxiliary_ms_full[:, row : row + size, col : col + size]),
        auxiliary_hs_interpolated=_normal(auxiliary_hs_full[:, row : row + size, col : col + size]),
        target_ms=_normal(target_ms_full[:, row : row + size, col : col + size]),
        target_hs_reference=_normal(target_reference_full[:, row : row + size, col : col + size]),
        inspection=inspection,
        spatial_metadata=_metadata(
            target_ms_path=inspection.target_ms.path,
            source_window=high_window,
            output_size=size,
            pixel_scale=SCALE,
        ),
        experiment_mode=SIMULATION_EXPERIMENT,
        reference_kind="native_target_hs_reduced_resolution_ground_truth",
        selection=selection,
        crop_manifest_path=manifest_path,
    )


def prepare_experiment_patch_from_manifest(
    crop_manifest_path: str | Path,
    *,
    experiment_mode: ExperimentMode | None = None,
    patch_size: int | None = None,
    patch_index: int | None = None,
    row_offset: int = 0,
    col_offset: int = 0,
) -> PreparedExperimentPatch:
    """Prepare one selected patch using the mode persisted in the crop manifest."""

    inspection, crop = inspect_manifest_crop_triplet(crop_manifest_path)
    if experiment_mode is not None and experiment_mode != crop.experiment_mode:
        raise ValueError(
            f"Requested experiment mode '{experiment_mode}' does not match manifest mode "
            f"'{crop.experiment_mode}'"
        )
    mode = crop.experiment_mode
    output_height, output_width = output_grid_shape(
        ms_height=inspection.target_ms.height,
        ms_width=inspection.target_ms.width,
        experiment_mode=mode,
    )
    selection = select_patch(
        output_height=output_height,
        output_width=output_width,
        experiment_mode=mode,
        patch_size=patch_size,
        patch_index=patch_index,
        row_offset=row_offset,
        col_offset=col_offset,
    )
    resolved_manifest = str(Path(crop_manifest_path).expanduser().resolve())
    if mode == SIMULATION_EXPERIMENT:
        return _prepare_simulation(inspection, crop, selection, resolved_manifest)
    return _prepare_real(inspection, crop, selection, resolved_manifest)
