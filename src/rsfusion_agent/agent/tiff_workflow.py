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
from rsfusion_agent.tools.tiff_crop import SIMULATION_EXPERIMENT, ExperimentMode
from rsfusion_agent.tools.tiff_experiment import (
    PreparedExperimentPatch,
    prepare_experiment_patch_from_manifest,
)
from rsfusion_agent.tools.tiff_patch import TiffPatchSpatialMetadata
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
    experiment_mode: ExperimentMode | None = None
    patch_size: int | None = Field(default=None, gt=0)
    patch_index: int | None = Field(default=None, ge=0)
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
    experiment_mode: ExperimentMode
    reference_kind: str
    patch_index: int | None = None
    total_patch_count: int = Field(gt=0)
    metrics_status: str
    metrics: FusionMetrics | None = None
    predicted_hs_path: str
    rgb_preview_path: str
    reference_rgb_preview_path: str | None = None
    sam_heatmap_path: str | None = None
    metrics_path: str | None = None
    input_preview_paths: dict[str, str]
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

    @staticmethod
    def _save_input_previews(
        prepared: PreparedExperimentPatch,
        output_dir: Path,
    ) -> dict[str, str]:
        """Persist previews of the exact tensors supplied to the model runner."""

        artifacts = {
            "input_auxiliary_ms_preview": str(
                save_rgb_preview(
                    prepared.auxiliary_ms,
                    output_dir / "input_auxiliary_ms_preview.png",
                    bands=(2, 1, 0),
                )
            ),
            "input_auxiliary_hs_preview": str(
                save_rgb_preview(
                    prepared.auxiliary_hs_interpolated,
                    output_dir / "input_auxiliary_hs_preview.png",
                    bands=(28, 18, 9),
                )
            ),
            "input_target_ms_preview": str(
                save_rgb_preview(
                    prepared.target_ms,
                    output_dir / "input_target_ms_preview.png",
                    bands=(2, 1, 0),
                )
            ),
        }
        if prepared.target_hs_reference is not None:
            artifacts["input_target_hs_reference_preview"] = str(
                save_rgb_preview(
                    prepared.target_hs_reference,
                    output_dir / "input_target_hs_reference_preview.png",
                    bands=(28, 18, 9),
                )
            )
        return artifacts

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
                lambda: prepare_experiment_patch_from_manifest(
                    request.crop_manifest_path,
                    experiment_mode=request.experiment_mode,
                    patch_size=request.patch_size,
                    patch_index=request.patch_index,
                    row_offset=request.row_offset,
                    col_offset=request.col_offset,
                ),
            )
        else:
            raise ValueError(
                "V2 TIFF fusion requires a crop manifest so real/simulation preprocessing and "
                "patch provenance are explicit. Run crop-tiff-triplet first."
            )
        input_preview_paths = self._step(
            "render_input_patch_previews",
            "Rendered the exact common-grid MS/HS tensors selected for this fusion patch.",
            lambda: self._save_input_previews(prepared, output_dir),
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
            expected_shape = (151, prepared.selection.patch_size, prepared.selection.patch_size)
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
        if prepared.target_hs_reference is None:
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
                prepared.target_hs_reference,
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
            metrics_status = (
                "available_reduced_resolution_ground_truth"
                if prepared.experiment_mode == SIMULATION_EXPERIMENT
                else "available_interpolated_target_hs_pseudo_reference"
            )
            warnings.append(
                "Simulation metrics use the native target-HS reduced-resolution ground truth."
                if prepared.experiment_mode == SIMULATION_EXPERIMENT
                else "Real-mode metrics use a 3x interpolated target-HS pseudo-reference, not native high-resolution ground truth."
            )
            metrics = self._step(
                "calculate_reference_metrics",
                "Calculated six metrics against the mode-specific target-HS reference.",
                lambda: calculate_metrics(prepared.target_hs_reference, prediction),
            )
            reference_rgb_preview_path = self._step(
                "render_reference_hs_rgb",
                "Rendered the mode-specific target-HS reference for direct comparison.",
                lambda: save_rgb_preview(
                    prepared.target_hs_reference,
                    output_dir / "reference_rgb_preview.png",
                    bands=request.rgb_bands,
                    stretch_bounds=comparison_rgb_bounds,
                ),
            )
            sam_heatmap_path = self._step(
                "render_reference_sam_heatmap",
                "Rendered the SAM heatmap against the mode-specific target-HS reference.",
                lambda: save_sam_heatmap(
                    calculate_sam_map(prepared.target_hs_reference, prediction),
                    output_dir / "sam_heatmap.png",
                ),
            )
            metrics_path = self._step(
                "save_reference_metrics",
                "Saved mode-specific reference metrics as JSON.",
                lambda: write_json(metrics.model_dump(mode="json"), output_dir / "metrics.json"),
            )
        artifacts = {
            "predicted_hs": str(predicted_hs_path),
            "rgb_preview": str(rgb_preview_path),
            "model_input": str(input_npz),
            "model_output": str(runtime_output),
            **input_preview_paths,
        }
        if sam_heatmap_path is not None:
            artifacts["sam_heatmap"] = str(sam_heatmap_path)
        if reference_rgb_preview_path is not None:
            artifacts["reference_rgb_preview"] = str(reference_rgb_preview_path)
        if metrics_path is not None:
            artifacts["metrics"] = str(metrics_path)
        manifest_data: dict[str, Any] = {
            "status": "completed",
            "profile": "yre151_tiff_experiment_patch_v2",
            "experiment_mode": prepared.experiment_mode,
            "reference_kind": prepared.reference_kind,
            "patch_selection": {
                "patch_index": prepared.selection.patch_index,
                "row_offset": prepared.selection.row_offset,
                "col_offset": prepared.selection.col_offset,
                "patch_size": prepared.selection.patch_size,
                "grid_rows": prepared.selection.grid_rows,
                "grid_columns": prepared.selection.grid_columns,
                "total_patch_count": prepared.selection.total_patch_count,
            },
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
            profile="yre151_tiff_experiment_patch_v2",
            inspection=prepared.inspection,
            spatial_metadata=prepared.spatial_metadata,
            runtime=runtime,
            experiment_mode=prepared.experiment_mode,
            reference_kind=prepared.reference_kind,
            patch_index=prepared.selection.patch_index,
            total_patch_count=prepared.selection.total_patch_count,
            metrics_status=metrics_status,
            metrics=metrics,
            predicted_hs_path=str(predicted_hs_path),
            rgb_preview_path=str(rgb_preview_path),
            reference_rgb_preview_path=(
                str(reference_rgb_preview_path) if reference_rgb_preview_path is not None else None
            ),
            sam_heatmap_path=str(sam_heatmap_path) if sam_heatmap_path is not None else None,
            metrics_path=str(metrics_path) if metrics_path is not None else None,
            input_preview_paths=input_preview_paths,
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
            "# RSFusionAgent raw-TIFF experiment inference report\n\n"
            "- Profile: `yre151_tiff_experiment_patch_v2`\n"
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
