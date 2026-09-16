"""Metadata-only validation for the future raw-TIFF fusion input triplet.

The deployed YRE model still consumes a prepared HDF5 patch.  This module is the
safe boundary before a TIFF-to-HDF5 adapter: it inspects the three source rasters
without loading pixels, and never reprojects, registers, or resamples data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from rsfusion_agent.tools.raster_inspector import RasterInspectionResult, inspect_raster

STRICT_METADATA_ALIGNMENT = "strict_metadata"
EXTERNAL_REGISTRATION_ALIGNMENT = "external_registration"
AlignmentMode = Literal["strict_metadata", "external_registration"]


class TiffTripletInspection(BaseModel):
    """Compatibility report for auxiliary-MS, auxiliary-HS, and target-MS rasters."""

    auxiliary_ms: RasterInspectionResult
    auxiliary_hs: RasterInspectionResult
    target_ms: RasterInspectionResult
    scale: int = Field(gt=1)
    expected_ms_bands: int = Field(gt=0)
    expected_hs_bands: int = Field(gt=0)
    grid_tolerance: float = Field(ge=0.0)
    alignment_mode: AlignmentMode = STRICT_METADATA_ALIGNMENT
    is_ready_for_preprocessing: bool
    blocking_issues: list[str]
    warnings: list[str]


def _close(left: float, right: float, tolerance: float) -> bool:
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def _same_bounds(
    left: RasterInspectionResult,
    right: RasterInspectionResult,
    tolerance: float,
) -> bool:
    return all(
        _close(first, second, tolerance)
        for first, second in zip(
            (
                left.bounds.left,
                left.bounds.bottom,
                left.bounds.right,
                left.bounds.top,
            ),
            (
                right.bounds.left,
                right.bounds.bottom,
                right.bounds.right,
                right.bounds.top,
            ),
            strict=True,
        )
    )


def inspect_tiff_triplet(
    auxiliary_ms_path: str | Path,
    auxiliary_hs_path: str | Path,
    target_ms_path: str | Path,
    *,
    scale: int = 3,
    expected_ms_bands: int = 4,
    expected_hs_bands: int = 151,
    grid_tolerance: float = 1e-6,
    alignment_mode: AlignmentMode = STRICT_METADATA_ALIGNMENT,
) -> TiffTripletInspection:
    """Validate a raw TIFF triplet against the planned YRE preprocessing contract.

    ``strict_metadata`` requires matching CRS/bounds/resolution metadata. In
    ``external_registration`` mode, the caller explicitly declares that inputs were
    registered outside this system and pixel crop windows carry the correspondence.
    Metadata discrepancies are then reported as warnings, never silently corrected.
    In both modes, band counts and the model's 3x pixel-dimension contract are hard
    requirements. A ready report never proves registration quality or inference.
    """

    if scale <= 1:
        raise ValueError("Scale must be greater than one")
    if expected_ms_bands <= 0 or expected_hs_bands <= 0:
        raise ValueError("Expected band counts must be positive")
    if grid_tolerance < 0:
        raise ValueError("Grid tolerance cannot be negative")
    if alignment_mode not in (STRICT_METADATA_ALIGNMENT, EXTERNAL_REGISTRATION_ALIGNMENT):
        raise ValueError(f"Unsupported alignment mode: {alignment_mode}")

    auxiliary_ms = inspect_raster(auxiliary_ms_path)
    auxiliary_hs = inspect_raster(auxiliary_hs_path)
    target_ms = inspect_raster(target_ms_path)
    issues: list[str] = []
    warnings: list[str] = []

    def record_metadata_difference(message: str) -> None:
        if alignment_mode == STRICT_METADATA_ALIGNMENT:
            issues.append(f"{message} Strict metadata mode requires this metadata to match.")
        else:
            warnings.append(
                "External-registration declaration: "
                f"{message} It is retained as a warning because no automatic registration or reprojection is performed."
            )

    for role, raster, expected_bands in (
        ("auxiliary MS", auxiliary_ms, expected_ms_bands),
        ("target MS", target_ms, expected_ms_bands),
        ("auxiliary HS", auxiliary_hs, expected_hs_bands),
    ):
        if raster.band_count != expected_bands:
            issues.append(
                f"{role} has {raster.band_count} bands; expected {expected_bands}."
            )
        if not raster.is_georeferenced:
            issues.append(f"{role} has no CRS; raw-TIFF preprocessing requires georeferencing.")

    if (auxiliary_ms.width, auxiliary_ms.height) != (target_ms.width, target_ms.height):
        issues.append(
            "Auxiliary MS and target MS must have matching pixel dimensions; "
            "explicit crop windows require a common MS output grid."
        )
    if (
        auxiliary_ms.crs != target_ms.crs
        or not _same_bounds(auxiliary_ms, target_ms, grid_tolerance)
        or not _close(auxiliary_ms.resolution[0], target_ms.resolution[0], grid_tolerance)
        or not _close(auxiliary_ms.resolution[1], target_ms.resolution[1], grid_tolerance)
    ):
        record_metadata_difference(
            "Auxiliary MS and target MS CRS, bounds or resolution metadata differ."
        )

    if auxiliary_ms.crs != auxiliary_hs.crs:
        record_metadata_difference("Auxiliary MS and auxiliary HS CRS metadata differ.")
    if not _same_bounds(auxiliary_ms, auxiliary_hs, grid_tolerance):
        record_metadata_difference("Auxiliary MS and auxiliary HS spatial bounds differ.")

    expected_hs_width = auxiliary_hs.width * scale
    expected_hs_height = auxiliary_hs.height * scale
    if (auxiliary_ms.width, auxiliary_ms.height) != (expected_hs_width, expected_hs_height):
        issues.append(
            "Auxiliary HS dimensions must be exactly 1/"
            f"{scale} of auxiliary MS dimensions; got MS {auxiliary_ms.width}x"
            f"{auxiliary_ms.height} and HS {auxiliary_hs.width}x{auxiliary_hs.height}."
        )

    for axis, ms_resolution, hs_resolution in (
        ("x", auxiliary_ms.resolution[0], auxiliary_hs.resolution[0]),
        ("y", auxiliary_ms.resolution[1], auxiliary_hs.resolution[1]),
    ):
        if not _close(hs_resolution, ms_resolution * scale, grid_tolerance):
            record_metadata_difference(
                f"Auxiliary HS {axis} resolution differs from the expected {scale}x MS resolution."
            )

    if alignment_mode == EXTERNAL_REGISTRATION_ALIGNMENT:
        warnings.append(
            "External-registration mode accepts the configured source-pixel correspondence. "
            "Confirm that registration was completed before this workflow; this system does not "
            "estimate residual offsets, reproject, or resample source TIFFs."
        )
    if not issues:
        if alignment_mode == STRICT_METADATA_ALIGNMENT:
            warnings.append(
                "Metadata is compatible with the planned TIFF adapter. "
                "Radiometric normalization and pixel-level registration remain the caller's responsibility."
            )

    return TiffTripletInspection(
        auxiliary_ms=auxiliary_ms,
        auxiliary_hs=auxiliary_hs,
        target_ms=target_ms,
        scale=scale,
        expected_ms_bands=expected_ms_bands,
        expected_hs_bands=expected_hs_bands,
        grid_tolerance=grid_tolerance,
        alignment_mode=alignment_mode,
        is_ready_for_preprocessing=not issues,
        blocking_issues=issues,
        warnings=warnings,
    )
