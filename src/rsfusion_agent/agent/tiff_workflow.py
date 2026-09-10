"""Traceable raw-TIFF single-patch inference for the YRE-151 checkpoint."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from pydantic import BaseModel, Field
from rasterio import Affine

from rsfusion_agent.agent.state import ToolTrace
from rsfusion_agent.tools.artifacts import (
    hyperspectral_rgb_stretch_bounds,
    save_prediction_tiff,
    save_rgb_preview,
    save_sam_heatmap,
    write_json,
)
from rsfusion_agent.tools.metrics import FusionMetrics, calculate_metrics, calculate_sam_map
from rsfusion_agent.tools.model_runtime import ModelRuntimeResult, run_yre151_runtime
from rsfusion_agent.tools.tiff_patch import (
    TiffPatchSpatialMetadata,
    prepare_tiff_patch,
    prepare_tiff_patch_from_manifest,
)
from rsfusion_agent.tools.tiff_triplet import TiffTripletInspection

T = TypeVar("T")
RuntimeRunner = Callable[..., ModelRuntimeResult]


class TiffFusionRequest(BaseModel):
    """Request for one raw TIFF crop, optionally with a legacy target-HS reference."""

    auxiliary_ms_path: Path | None = None
    auxiliary_hs_path: Path | None = None
    target_ms_path: Path | None = None
    target_hs_reference_path: Path | None = None
    crop_manifest_path: Path | None = None
    checkpoint_path: Path
    model_python: Path
    output_dir: Path
    patch_size: int = Field(default=180, gt=0)
    row_offset: int = Field(default=0, ge=0)
    col_offset: int = Field(default=0, ge=0)
    device: str = "auto"
    timeout_seconds: int = Field(default=600, gt=0)
    runtime_retries: int = Field(default=1, ge=0, le=2)
    rgb_bands: tuple[int, int, int] = (28, 18, 9)


class TiffFusionResult(BaseModel):
    status: str
    profile: str
    inspection: TiffTripletInspection
    spatial_metadata: TiffPatchSpatialMetadata
    runtime: ModelRuntimeResult
    metrics_status: str
    metrics: FusionMetrics | None = None
    predicted_hs_path: str
    rgb_preview_path: str
    reference_rgb_preview_path: str | None = None
    sam_heatmap_path: str | None = None
    metrics_path: str | None = None
    manifest_path: str
    report_path: str
    trace: list[ToolTrace]
    warnings: list[str]


class YRE151TiffPatchAgent:
    """Run one metadata-validated raw TIFF crop with optional legacy reference metrics."""

    def __init__(self, runtime_runner: RuntimeRunner = run_yre151_runtime) -> None:
        self.runtime_runner = runtime_runner
        self.trace: list[ToolTrace] = []

    def _step(self, name: str, summary: str, operation: Callable[[], T]) -> T:
        started_at = time.perf_counter()
        try:
            value = operation()
        except Exception:
            self.trace.append(
                ToolTrace(
                    name=name,
                    status="failed",
                    elapsed_seconds=time.perf_counter() - started_at,
                    summary=summary,
                )
            )
            raise
        self.trace.append(
            ToolTrace(
                name=name,
                status="completed",
                elapsed_seconds=time.perf_counter() - started_at,
                summary=summary,
            )
        )
        return value

    def run(self, request: TiffFusionRequest) -> TiffFusionResult:
        self.trace = []
        output_dir = request.output_dir.expanduser().resolve()
        intermediate_dir = output_dir / "intermediate"
        output_dir.mkdir(parents=True, exist_ok=True)
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        if request.crop_manifest_path is not None:
            prepared = self._step(
                "prepare_manifest_tiff_patch",
                "Validated the configured crop manifest and prepared one authorized TIFF patch.",
                lambda: prepare_tiff_patch_from_manifest(
                    request.crop_manifest_path,
                    patch_size=request.patch_size,
                    row_offset=request.row_offset,
                    col_offset=request.col_offset,
                ),
            )
        else:
            if None in (
                request.auxiliary_ms_path,
                request.auxiliary_hs_path,
                request.target_ms_path,
            ):
                raise ValueError(
                    "Raw TIFF inference requires either a crop manifest or all three TIFF paths"
                )
            prepared = self._step(
                "prepare_tiff_patch",
                "Validated TIFF metadata, read one aligned crop and normalized model inputs.",
                lambda: prepare_tiff_patch(
                    request.auxiliary_ms_path,
                    request.auxiliary_hs_path,
                    request.target_ms_path,
                    target_hs_reference_path=request.target_hs_reference_path,
                    patch_size=request.patch_size,
                    row_offset=request.row_offset,
                    col_offset=request.col_offset,
                ),
            )
        input_npz = self._step(
            "write_model_input",
            "Persisted normalized TIFF-derived inputs for the isolated PyTorch runtime.",
            lambda: prepared.save_npz(intermediate_dir / "model_input.npz"),
        )
        runtime_output = intermediate_dir / "model_output.npz"
        runtime = self._step(
            "run_dc_stsf_patch",
            "Loaded the YRE-151 checkpoint and generated the target-time HS crop.",
            lambda: self.runtime_runner(
                input_npz=input_npz,
                output_npz=runtime_output,
                checkpoint_path=request.checkpoint_path,
                model_python=request.model_python,
                device=request.device,
                timeout_seconds=request.timeout_seconds,
                retry_count=request.runtime_retries,
            ),
        )

        def read_prediction() -> np.ndarray:
            with np.load(runtime.output_npz) as payload:
                prediction = np.asarray(payload["predicted_hs"], dtype=np.float32)
            expected_shape = (151, request.patch_size, request.patch_size)
            if prediction.shape != expected_shape:
                raise ValueError(
                    f"Model prediction shape is {prediction.shape}; expected {expected_shape}"
                )
            if not np.isfinite(prediction).all():
                raise ValueError("Model prediction contains NaN or Inf")
            return prediction

        prediction = self._step(
            "validate_prediction",
            "Validated target-time HS shape and finite values.",
            read_prediction,
        )
        predicted_hs_path = self._step(
            "save_georeferenced_hs_patch",
            "Saved a normalized 151-band float32 TIFF using target-MS georeferencing.",
            lambda: save_prediction_tiff(
                prediction,
                output_dir / "predicted_hs.tif",
                transform=Affine(*prepared.spatial_metadata.transform),
                crs=prepared.spatial_metadata.crs,
            ),
        )
        warnings = list(prepared.inspection.warnings)
        metrics: FusionMetrics | None = None
        reference_rgb_preview_path: Path | None = None
        sam_heatmap_path: Path | None = None
        metrics_path: Path | None = None
        if prepared.target_hs_reference_interpolated is None:
            rgb_preview_path = self._step(
                "render_hs_rgb",
                "Rendered a percentile-stretched RGB preview.",
                lambda: save_rgb_preview(
                    prediction,
                    output_dir / "rgb_preview.png",
                    bands=request.rgb_bands,
                ),
            )
            metrics_status = "unavailable_without_target_hs_reference"
            warnings.append(
                "No target-time HS reference was supplied, so PSNR, RMSE, SAM, ERGAS, SSIM, "
                "CC and the SAM heatmap are unavailable for raw-TIFF inference."
            )
        else:
            comparison_rgb_bounds = hyperspectral_rgb_stretch_bounds(
                prediction,
                prepared.target_hs_reference_interpolated,
                bands=request.rgb_bands,
            )
            rgb_preview_path = self._step(
                "render_hs_rgb",
                "Rendered the prediction with shared prediction/reference RGB stretch bounds.",
                lambda: save_rgb_preview(
                    prediction,
                    output_dir / "rgb_preview.png",
                    bands=request.rgb_bands,
                    stretch_bounds=comparison_rgb_bounds,
                ),
            )
            metrics_status = "available_legacy_interpolated_target_hs_reference"
            warnings.append(
                "Metrics use a target-HS reference cropped at native HS resolution and bilinearly "
                "upsampled by three, reproducing the active Database.py test construction. "
                "It is an interpolated pseudo-reference, not native high-resolution ground truth."
            )
            metrics = self._step(
                "calculate_legacy_reference_metrics",
                "Calculated six full-reference metrics against the interpolated target-HS reference.",
                lambda: calculate_metrics(prepared.target_hs_reference_interpolated, prediction),
            )
            reference_rgb_preview_path = self._step(
                "render_reference_hs_rgb",
                "Rendered the interpolated target-HS pseudo-reference for direct comparison.",
                lambda: save_rgb_preview(
                    prepared.target_hs_reference_interpolated,
                    output_dir / "reference_rgb_preview.png",
                    bands=request.rgb_bands,
                    stretch_bounds=comparison_rgb_bounds,
                ),
            )
            sam_heatmap_path = self._step(
                "render_reference_sam_heatmap",
                "Rendered the SAM heatmap against the interpolated target-HS reference.",
                lambda: save_sam_heatmap(
                    calculate_sam_map(prepared.target_hs_reference_interpolated, prediction),
                    output_dir / "sam_heatmap.png",
                ),
            )
            metrics_path = self._step(
                "save_reference_metrics",
                "Saved legacy pseudo-reference metrics as JSON.",
                lambda: write_json(metrics.model_dump(mode="json"), output_dir / "metrics.json"),
            )
        artifacts = {
            "predicted_hs": str(predicted_hs_path),
            "rgb_preview": str(rgb_preview_path),
            "model_input": str(input_npz),
            "model_output": str(runtime_output),
        }
        if sam_heatmap_path is not None:
            artifacts["sam_heatmap"] = str(sam_heatmap_path)
        if reference_rgb_preview_path is not None:
            artifacts["reference_rgb_preview"] = str(reference_rgb_preview_path)
        if metrics_path is not None:
            artifacts["metrics"] = str(metrics_path)
        manifest_data: dict[str, Any] = {
            "status": "completed",
            "profile": "yre151_tiff_single_patch_v1",
            "request": request.model_dump(mode="json"),
            "inspection": prepared.inspection.model_dump(mode="json"),
            "spatial_metadata": prepared.spatial_metadata.model_dump(mode="json"),
            "crop_manifest_path": prepared.crop_manifest_path,
            "runtime": runtime.model_dump(mode="json"),
            "metrics_status": metrics_status,
            "metrics": metrics.model_dump(mode="json") if metrics is not None else None,
            "artifacts": artifacts,
            "trace": [step.model_dump(mode="json") for step in self.trace],
            "warnings": warnings,
        }
        manifest_path = write_json(manifest_data, output_dir / "run_manifest.json")
        report_path = output_dir / "report.md"
        report_path.write_text(
            self._build_report(runtime, artifacts, warnings, metrics_status, metrics), encoding="utf-8"
        )
        return TiffFusionResult(
            status="completed",
            profile="yre151_tiff_single_patch_v1",
            inspection=prepared.inspection,
            spatial_metadata=prepared.spatial_metadata,
            runtime=runtime,
            metrics_status=metrics_status,
            metrics=metrics,
            predicted_hs_path=str(predicted_hs_path),
            rgb_preview_path=str(rgb_preview_path),
            reference_rgb_preview_path=(
                str(reference_rgb_preview_path) if reference_rgb_preview_path is not None else None
            ),
            sam_heatmap_path=str(sam_heatmap_path) if sam_heatmap_path is not None else None,
            metrics_path=str(metrics_path) if metrics_path is not None else None,
            manifest_path=str(manifest_path),
            report_path=str(report_path),
            trace=self.trace,
            warnings=warnings,
        )

    @staticmethod
    def _build_report(
        runtime: ModelRuntimeResult,
        artifacts: dict[str, str],
        warnings: list[str],
        metrics_status: str,
        metrics: FusionMetrics | None,
    ) -> str:
        artifact_rows = "\n".join(f"- `{name}`: `{path}`" for name, path in artifacts.items())
        warning_rows = "\n".join(f"- {warning}" for warning in warnings)
        return (
            "# RSFusionAgent raw-TIFF inference report\n\n"
            "- Profile: `yre151_tiff_single_patch_v1`\n"
            f"- Device: `{runtime.device}`\n"
            f"- Checkpoint epoch: `{runtime.checkpoint_epoch}`\n"
            f"- Model runtime: `{runtime.runtime_seconds:.6f} s`\n"
            f"- Metrics status: `{metrics_status}`\n"
            + (
                "- Metrics: "
                f"PSNR {metrics.psnr:.4f}, SAM {metrics.sam:.4f}, SSIM {metrics.ssim:.4f}\n\n"
                if metrics is not None
                else "- Metrics: unavailable without target-time HS reference\n\n"
            )
            + f"## Artifacts\n\n{artifact_rows}\n\n## Warnings\n\n{warning_rows}\n"
        )
