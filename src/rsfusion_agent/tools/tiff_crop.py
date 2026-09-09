"""Explicit, traceable TIFF crop windows for YRE raw-data preparation."""

from __future__ import annotations

from pathlib import Path

import rasterio
from pydantic import BaseModel, Field, ValidationError
from rasterio.windows import Window

from rsfusion_agent.tools.h5_patch import HS_BANDS, MS_BANDS, SCALE
from rsfusion_agent.tools.raster_inspector import RasterInspectionResult, inspect_raster

YRE_LEGACY_TEST_PROFILE = "yre_legacy_test_v1"
CUSTOM_PROFILE = "custom"


class CropWindow(BaseModel):
    """A pixel window in one source raster's own grid."""

    row_offset: int = Field(ge=0)
    col_offset: int = Field(ge=0)
    height: int = Field(gt=0)
    width: int = Field(gt=0)


class TiffCropResult(BaseModel):
    """Traceable result of writing three native-grid cropped TIFFs."""

    profile: str
    auxiliary_ms_source: RasterInspectionResult
    auxiliary_hs_source: RasterInspectionResult
    target_ms_source: RasterInspectionResult
    auxiliary_ms_window: CropWindow
    auxiliary_hs_window: CropWindow
    target_ms_window: CropWindow
    auxiliary_ms_path: str
    auxiliary_hs_path: str
    target_ms_path: str
    manifest_path: str
    warnings: list[str]


def load_crop_manifest(path: str | Path) -> TiffCropResult:
    """Read a crop manifest and reject malformed or missing local artifacts."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Crop manifest does not exist: {manifest_path}")
    try:
        result = TiffCropResult.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        raise ValueError(f"Cannot read crop manifest '{manifest_path}': {exc}") from exc
    for artifact_path, role in (
        (result.auxiliary_ms_path, "Auxiliary MS crop"),
        (result.auxiliary_hs_path, "Auxiliary HS crop"),
        (result.target_ms_path, "Target MS crop"),
    ):
        if not Path(artifact_path).is_file():
            raise FileNotFoundError(f"{role} declared by crop manifest does not exist: {artifact_path}")
    return result


def resolve_crop_windows(
    *,
    profile: str,
    ms_row_offset: int | None = None,
    ms_col_offset: int | None = None,
    window_height: int | None = None,
    window_width: int | None = None,
    hs_row_offset: int | None = None,
    hs_col_offset: int | None = None,
) -> tuple[CropWindow, CropWindow]:
    """Resolve the legacy YRE test window or a caller-supplied aligned window.

    The legacy profile exactly mirrors the active test crop in ``Database.py``:
    high-resolution MS `[0:540, 360:900]` and low-resolution HS `[0:180, 120:300]`.
    """

    if profile == YRE_LEGACY_TEST_PROFILE:
        defaults = (0, 360, 540, 540, 0, 120)
    elif profile == CUSTOM_PROFILE:
        if None in (ms_row_offset, ms_col_offset, window_height, window_width):
            raise ValueError(
                "Custom crop requires --ms-row-offset, --ms-col-offset, "
                "--window-height and --window-width"
            )
        defaults = (
            int(ms_row_offset),
            int(ms_col_offset),
            int(window_height),
            int(window_width),
            int(hs_row_offset) if hs_row_offset is not None else -1,
            int(hs_col_offset) if hs_col_offset is not None else -1,
        )
    else:
        raise ValueError(
            f"Unsupported crop profile '{profile}'. Use {YRE_LEGACY_TEST_PROFILE} or {CUSTOM_PROFILE}."
        )

    ms_row, ms_col, height, width, hs_row, hs_col = defaults
    if height <= 0 or width <= 0 or height % SCALE or width % SCALE:
        raise ValueError(f"MS crop height and width must be positive and divisible by {SCALE}")
    if ms_row < 0 or ms_col < 0:
        raise ValueError("MS crop offsets must be non-negative")
    if hs_row < 0:
        if ms_row % SCALE:
            raise ValueError("MS row offset must be divisible by 3 when HS row offset is omitted")
        hs_row = ms_row // SCALE
    if hs_col < 0:
        if ms_col % SCALE:
            raise ValueError("MS column offset must be divisible by 3 when HS column offset is omitted")
        hs_col = ms_col // SCALE
    return (
        CropWindow(row_offset=ms_row, col_offset=ms_col, height=height, width=width),
        CropWindow(
            row_offset=hs_row,
            col_offset=hs_col,
            height=height // SCALE,
            width=width // SCALE,
        ),
    )


def _validate_window(raster: RasterInspectionResult, window: CropWindow, role: str) -> None:
    if window.row_offset + window.height > raster.height or window.col_offset + window.width > raster.width:
        raise ValueError(
            f"{role} crop row={window.row_offset}, col={window.col_offset}, "
            f"size={window.width}x{window.height} exceeds raster {raster.width}x{raster.height}"
        )


def _write_crop(source_path: str, window: CropWindow, output_path: Path) -> None:
    native_window = Window(window.col_offset, window.row_offset, window.width, window.height)
    with rasterio.open(source_path) as source:
        profile = source.profile.copy()
        profile.update(
            height=window.height,
            width=window.width,
            transform=source.window_transform(native_window),
        )
        data = source.read(window=native_window)
        with rasterio.open(output_path, "w", **profile) as destination:
            destination.write(data)


def crop_tiff_triplet(
    auxiliary_ms_path: str | Path,
    auxiliary_hs_path: str | Path,
    target_ms_path: str | Path,
    output_dir: str | Path,
    *,
    profile: str = YRE_LEGACY_TEST_PROFILE,
    ms_row_offset: int | None = None,
    ms_col_offset: int | None = None,
    window_height: int | None = None,
    window_width: int | None = None,
    hs_row_offset: int | None = None,
    hs_col_offset: int | None = None,
) -> TiffCropResult:
    """Write three cropped TIFFs using an explicit source-pixel correspondence rule."""

    auxiliary_ms = inspect_raster(auxiliary_ms_path)
    auxiliary_hs = inspect_raster(auxiliary_hs_path)
    target_ms = inspect_raster(target_ms_path)
    if auxiliary_ms.band_count != MS_BANDS or target_ms.band_count != MS_BANDS:
        raise ValueError(f"Auxiliary and target MS inputs must each have {MS_BANDS} bands")
    if auxiliary_hs.band_count != HS_BANDS:
        raise ValueError(f"Auxiliary HS input must have {HS_BANDS} bands")
    ms_window, hs_window = resolve_crop_windows(
        profile=profile,
        ms_row_offset=ms_row_offset,
        ms_col_offset=ms_col_offset,
        window_height=window_height,
        window_width=window_width,
        hs_row_offset=hs_row_offset,
        hs_col_offset=hs_col_offset,
    )
    _validate_window(auxiliary_ms, ms_window, "Auxiliary MS")
    _validate_window(target_ms, ms_window, "Target MS")
    _validate_window(auxiliary_hs, hs_window, "Auxiliary HS")

    resolved_output = Path(output_dir).expanduser().resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    auxiliary_ms_output = resolved_output / "auxiliary_ms_crop.tif"
    auxiliary_hs_output = resolved_output / "auxiliary_hs_crop.tif"
    target_ms_output = resolved_output / "target_ms_crop.tif"
    _write_crop(auxiliary_ms.path, ms_window, auxiliary_ms_output)
    _write_crop(auxiliary_hs.path, hs_window, auxiliary_hs_output)
    _write_crop(target_ms.path, ms_window, target_ms_output)

    warnings = [
        "Crops preserve each source TIFF's native georeferencing. The configured pixel windows "
        "are an explicit correspondence assumption, not automatic registration or reprojection."
    ]
    result = TiffCropResult(
        profile=profile,
        auxiliary_ms_source=auxiliary_ms,
        auxiliary_hs_source=auxiliary_hs,
        target_ms_source=target_ms,
        auxiliary_ms_window=ms_window,
        auxiliary_hs_window=hs_window,
        target_ms_window=ms_window,
        auxiliary_ms_path=str(auxiliary_ms_output),
        auxiliary_hs_path=str(auxiliary_hs_output),
        target_ms_path=str(target_ms_output),
        manifest_path=str(resolved_output / "crop_manifest.json"),
        warnings=warnings,
    )
    Path(result.manifest_path).write_text(
        result.model_dump_json(indent=2), encoding="utf-8"
    )
    return result
