"""Metadata-only validation for the future raw-TIFF fusion input triplet.

The deployed YRE model still consumes a prepared HDF5 patch.  This module is the
safe boundary before a TIFF-to-HDF5 adapter: it inspects the three source rasters
without loading pixels, and never reprojects, registers, or resamples data.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from rsfusion_agent.tools.raster_inspector import RasterInspectionResult, inspect_raster


class TiffTripletInspection(BaseModel):
    """Compatibility report for auxiliary-MS, auxiliary-HS, and target-MS rasters."""

    auxiliary_ms: RasterInspectionResult
    auxiliary_hs: RasterInspectionResult
    target_ms: RasterInspectionResult
    scale: int = Field(gt=1)
    expected_ms_bands: int = Field(gt=0)
    expected_hs_bands: int = Field(gt=0)
    grid_tolerance: float = Field(ge=0.0)
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


def _same_grid(
    left: RasterInspectionResult,
    right: RasterInspectionResult,
    tolerance: float,
) -> bool:
    return (
        left.width == right.width
        and left.height == right.height
        and left.crs == right.crs
        and _same_bounds(left, right, tolerance)
        and _close(left.resolution[0], right.resolution[0], tolerance)
        and _close(left.resolution[1], right.resolution[1], tolerance)
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
) -> TiffTripletInspection:
    """Validate a raw TIFF triplet against the planned YRE preprocessing contract.

    A ready report means only that metadata is compatible with the future adapter;
    it does not mean the rasters are radiometrically registered or that inference has
    been performed.
    """

    if scale <= 1:
        raise ValueError("Scale must be greater than one")
    if expected_ms_bands <= 0 or expected_hs_bands <= 0:
        raise ValueError("Expected band counts must be positive")
    if grid_tolerance < 0:
        raise ValueError("Grid tolerance cannot be negative")

    auxiliary_ms = inspect_raster(auxiliary_ms_path)
    auxiliary_hs = inspect_raster(auxiliary_hs_path)
    target_ms = inspect_raster(target_ms_path)
    issues: list[str] = []
    warnings: list[str] = []

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

    if not _same_grid(auxiliary_ms, target_ms, grid_tolerance):
        issues.append(
            "Auxiliary MS and target MS must share CRS, bounds, dimensions and resolution; "
            "registration/reprojection is not performed automatically."
        )

    if auxiliary_ms.crs != auxiliary_hs.crs:
        issues.append("Auxiliary MS and auxiliary HS must use the same CRS.")
    if not _same_bounds(auxiliary_ms, auxiliary_hs, grid_tolerance):
        issues.append("Auxiliary MS and auxiliary HS must cover the same spatial bounds.")

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
            issues.append(
                f"Auxiliary HS {axis} resolution must equal auxiliary MS resolution times {scale}."
            )

    if not issues:
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
        is_ready_for_preprocessing=not issues,
        blocking_issues=issues,
        warnings=warnings,
    )
