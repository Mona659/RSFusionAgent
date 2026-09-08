"""Deterministic, traceable first-version fusion agent."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

from rsfusion_agent.agent.state import FusionRunRequest, FusionRunResult, ToolTrace
from rsfusion_agent.tools.artifacts import (
    save_prediction_tiff,
    save_rgb_preview,
    save_sam_heatmap,
    write_json,
)
from rsfusion_agent.tools.h5_patch import inspect_h5_pair, prepare_yre151_patch
from rsfusion_agent.tools.metrics import calculate_metrics, calculate_sam_map
from rsfusion_agent.tools.model_runtime import ModelRuntimeResult, run_yre151_runtime

T = TypeVar("T")
RuntimeRunner = Callable[..., ModelRuntimeResult]


class YRE151PatchAgent:
    """Orchestrate typed tools for one YRE-151 HDF5 test patch."""

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

    def run(self, request: FusionRunRequest) -> FusionRunResult:
        self.trace = []
        output_dir = request.output_dir.expanduser().resolve()
        intermediate_dir = output_dir / "intermediate"
        output_dir.mkdir(parents=True, exist_ok=True)
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        inspection = self._step(
            "inspect_h5_dataset",
            "Validated the auxiliary and target legacy YRE-151 HDF5 files.",
            lambda: inspect_h5_pair(
                request.auxiliary_h5_path,
                request.target_h5_path,
                patch_index=request.patch_index,
            ),
        )
        prepared = self._step(
            "prepare_yre151_patch",
            "Loaded one NHWC patch, sliced channels and normalized by 10000.",
            lambda: prepare_yre151_patch(
                request.auxiliary_h5_path,
                request.target_h5_path,
                patch_index=request.patch_index,
            ),
        )
        input_npz = self._step(
            "write_model_input",
            "Persisted path-based model inputs for the isolated PyTorch runtime.",
            lambda: prepared.save_npz(intermediate_dir / "model_input.npz"),
        )
        runtime_output = intermediate_dir / "model_output.npz"
        runtime = self._step(
            "run_dc_stsf_patch",
            "Loaded the YRE-151 checkpoint and generated the target-time HS patch.",
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
            expected_shape = (
                151,
                inspection.target.patch_height,
                inspection.target.patch_width,
            )
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
        metrics = self._step(
            "evaluate_hs_patch",
            "Calculated PSNR, RMSE, SAM, ERGAS, SSIM and CC against H5 ground truth.",
            lambda: calculate_metrics(prepared.target_hs_gt, prediction, scale=inspection.scale),
        )
        predicted_hs_path = self._step(
            "save_hs_patch",
            "Saved a normalized 151-band float32 TIFF using the legacy output convention.",
            lambda: save_prediction_tiff(prediction, output_dir / "predicted_hs.tif"),
        )
        rgb_preview_path = self._step(
            "render_hs_rgb",
            "Rendered a percentile-stretched RGB preview.",
            lambda: save_rgb_preview(
                prediction,
                output_dir / "rgb_preview.png",
                bands=request.rgb_bands,
            ),
        )
        sam_heatmap_path = self._step(
            "render_sam_heatmap",
            "Rendered a per-pixel spectral-angle heatmap.",
            lambda: save_sam_heatmap(
                calculate_sam_map(prepared.target_hs_gt, prediction),
                output_dir / "sam_heatmap.png",
            ),
        )
        metrics_path = self._step(
            "write_metrics",
            "Saved machine-readable full-reference metrics.",
            lambda: write_json(metrics.model_dump(mode="json"), output_dir / "metrics.json"),
        )

        warnings = list(inspection.warnings)
        warnings.append(
            "V1 output follows the legacy test convention: normalized float32 TIFF, "
            "synthetic transform and no CRS."
        )
        artifacts = {
            "predicted_hs": str(predicted_hs_path),
            "rgb_preview": str(rgb_preview_path),
            "sam_heatmap": str(sam_heatmap_path),
            "metrics": str(metrics_path),
            "model_input": str(input_npz),
            "model_output": str(runtime_output),
        }
        manifest_data: dict[str, Any] = {
            "status": "completed",
            "profile": inspection.profile,
            "request": request.model_dump(mode="json"),
            "inspection": inspection.model_dump(mode="json"),
            "runtime": runtime.model_dump(mode="json"),
            "metrics": metrics.model_dump(mode="json"),
            "artifacts": artifacts,
            "trace": [step.model_dump(mode="json") for step in self.trace],
            "warnings": warnings,
        }
        manifest_path = write_json(manifest_data, output_dir / "run_manifest.json")
        report_path = output_dir / "report.md"
        report_path.write_text(
            self._build_report(inspection.profile, runtime, metrics, artifacts, warnings),
            encoding="utf-8",
        )
        return FusionRunResult(
            status="completed",
            profile=inspection.profile,
            inspection=inspection,
            runtime=runtime,
            metrics=metrics,
            predicted_hs_path=str(predicted_hs_path),
            rgb_preview_path=str(rgb_preview_path),
            sam_heatmap_path=str(sam_heatmap_path),
            metrics_path=str(metrics_path),
            manifest_path=str(manifest_path),
            report_path=str(report_path),
            trace=self.trace,
            warnings=warnings,
        )

    @staticmethod
    def _build_report(
        profile: str,
        runtime: ModelRuntimeResult,
        metrics: Any,
        artifacts: dict[str, str],
        warnings: list[str],
    ) -> str:
        metric_rows = "\n".join(
            f"| {name.upper()} | {value:.6f} |"
            for name, value in metrics.model_dump().items()
        )
        warning_rows = "\n".join(f"- {warning}" for warning in warnings)
        artifact_rows = "\n".join(f"- `{name}`: `{path}`" for name, path in artifacts.items())
        return (
            "# RSFusionAgent inference report\n\n"
            f"- Profile: `{profile}`\n"
            f"- Device: `{runtime.device}`\n"
            f"- Checkpoint epoch: `{runtime.checkpoint_epoch}`\n"
            f"- Model runtime: `{runtime.runtime_seconds:.6f} s`\n\n"
            "## Metrics\n\n| Metric | Value |\n|---|---:|\n"
            f"{metric_rows}\n\n## Artifacts\n\n{artifact_rows}\n\n"
            f"## Warnings\n\n{warning_rows}\n"
        )
