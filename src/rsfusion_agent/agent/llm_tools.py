"""Allowlisted local tools exposed to the language-model control plane."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from rsfusion_agent.agent.state import FusionRunRequest, FusionRunResult
from rsfusion_agent.agent.tiff_workflow import (
    TiffFusionRequest,
    TiffFusionResult,
    YRE151TiffPatchAgent,
)
from rsfusion_agent.agent.workflow import YRE151PatchAgent
from rsfusion_agent.tools.error_diagnosis import diagnose_error
from rsfusion_agent.tools.h5_patch import H5PairInspection, inspect_h5_pair
from rsfusion_agent.tools.raster_inspector import inspect_raster
from rsfusion_agent.tools.raster_preview import render_raster_rgb
from rsfusion_agent.tools.tiff_crop import ExperimentMode, crop_tiff_triplet
from rsfusion_agent.tools.tiff_patch import inspect_manifest_crop_triplet
from rsfusion_agent.tools.tiff_triplet import inspect_tiff_triplet


class LLMToolContext(BaseModel):
    auxiliary_h5_path: Path
    target_h5_path: Path
    checkpoint_path: Path
    model_python: Path
    output_dir: Path
    patch_index: int = Field(default=0, ge=0)
    device: str = "auto"
    timeout_seconds: int = Field(default=600, gt=0)
    runtime_retries: int = Field(default=1, ge=0, le=2)


class PatchIndexArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patch_index: int = Field(ge=0)


WorkflowFactory = Callable[[], YRE151PatchAgent]


class AgentToolbox:
    """Execute a small, explicit tool allowlist against configured local paths."""

    def __init__(
        self,
        context: LLMToolContext,
        workflow_factory: WorkflowFactory = YRE151PatchAgent,
    ) -> None:
        self.context = context
        self.workflow_factory = workflow_factory
        self.inspected_patch: int | None = None
        self.latest_result: FusionRunResult | None = None
        self.runtime_preflight: Any | None = None

    @staticmethod
    def definitions() -> list[dict[str, Any]]:
        patch_schema = {
            "type": "object",
            "properties": {
                "patch_index": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Zero-based patch index configured for this request.",
                }
            },
            "required": ["patch_index"],
            "additionalProperties": False,
        }
        empty_schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
        return [
            {
                "type": "function",
                "name": "get_runtime_preflight",
                "description": "Read the already-completed local model environment preflight.",
                "parameters": empty_schema,
                "strict": True,
            },
            {
                "type": "function",
                "name": "inspect_yre151_h5",
                "description": (
                    "Validate the configured auxiliary and target YRE-151 HDF5 files "
                    "before inference."
                ),
                "parameters": patch_schema,
                "strict": True,
            },
            {
                "type": "function",
                "name": "run_yre151_fusion",
                "description": (
                    "Run the configured DC-STSF checkpoint for one patch. Call "
                    "inspect_yre151_h5 successfully first."
                ),
                "parameters": patch_schema,
                "strict": True,
            },
            {
                "type": "function",
                "name": "get_latest_fusion_result",
                "description": "Read the metrics and artifact names from the latest local run.",
                "parameters": empty_schema,
                "strict": True,
            },
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "get_runtime_preflight":
            if arguments:
                raise ValueError("get_runtime_preflight does not accept arguments")
            return self._preflight_summary()
        if name == "inspect_yre151_h5":
            parsed = PatchIndexArguments.model_validate(arguments)
            self._require_configured_patch(parsed.patch_index)
            inspection = inspect_h5_pair(
                self.context.auxiliary_h5_path,
                self.context.target_h5_path,
                patch_index=parsed.patch_index,
            )
            self.inspected_patch = parsed.patch_index
            return self._inspection_summary(inspection)

        if name == "run_yre151_fusion":
            parsed = PatchIndexArguments.model_validate(arguments)
            self._require_configured_patch(parsed.patch_index)
            if self.inspected_patch != parsed.patch_index:
                raise ValueError(
                    "Safety gate: call inspect_yre151_h5 successfully for this patch first"
                )
            result = self.workflow_factory().run(
                FusionRunRequest(
                    auxiliary_h5_path=self.context.auxiliary_h5_path,
                    target_h5_path=self.context.target_h5_path,
                    checkpoint_path=self.context.checkpoint_path,
                    model_python=self.context.model_python,
                    output_dir=self.context.output_dir,
                    patch_index=parsed.patch_index,
                    device=self.context.device,
                    timeout_seconds=self.context.timeout_seconds,
                    runtime_retries=self.context.runtime_retries,
                )
            )
            self.latest_result = result
            return self._result_summary(result)

        if name == "get_latest_fusion_result":
            if arguments:
                raise ValueError("get_latest_fusion_result does not accept arguments")
            if self.latest_result is None:
                raise ValueError("No fusion result is available in this agent session")
            return self._result_summary(self.latest_result)

        raise ValueError(f"Tool is not allowlisted: {name}")

    def _preflight_summary(self) -> dict[str, Any]:
        if self.runtime_preflight is None:
            raise ValueError("No runtime preflight result is available for this agent session")
        return {"ok": True, **self.runtime_preflight.model_dump(mode="json")}

    def sanitize_error(self, error: Exception) -> dict[str, Any]:
        """Remove configured absolute paths before returning an error to the LLM."""
        message = str(error)
        configured_paths = {
            "auxiliary_h5_path": self.context.auxiliary_h5_path,
            "target_h5_path": self.context.target_h5_path,
            "checkpoint_path": self.context.checkpoint_path,
            "model_python": self.context.model_python,
            "output_dir": self.context.output_dir,
        }
        for label, path in configured_paths.items():
            raw_path = str(path)
            resolved_path = str(path.expanduser().resolve())
            for candidate in {raw_path, resolved_path, raw_path.replace("\\", "/")}:
                if candidate:
                    message = message.replace(candidate, f"<{label}>")
        diagnosis = diagnose_error(error)
        return {
            "ok": False,
            "error_type": type(error).__name__,
            "error": message,
            "diagnosis": diagnosis.model_dump(mode="json"),
        }

    def _require_configured_patch(self, patch_index: int) -> None:
        if patch_index != self.context.patch_index:
            raise ValueError(
                f"Patch index {patch_index} is not authorized; "
                f"this session is configured for {self.context.patch_index}"
            )

    @staticmethod
    def _inspection_summary(inspection: H5PairInspection) -> dict[str, Any]:
        return {
            "ok": True,
            "profile": inspection.profile,
            "patch_index": inspection.patch_index,
            "auxiliary": {
                "file_name": Path(inspection.auxiliary.path).name,
                "shape": inspection.auxiliary.shape,
                "dtype": inspection.auxiliary.dtype,
                "sample_min": inspection.auxiliary.sample_min,
                "sample_max": inspection.auxiliary.sample_max,
            },
            "target": {
                "file_name": Path(inspection.target.path).name,
                "shape": inspection.target.shape,
                "dtype": inspection.target.dtype,
                "sample_min": inspection.target.sample_min,
                "sample_max": inspection.target.sample_max,
            },
            "warnings": inspection.warnings,
        }

    @staticmethod
    def _result_summary(result: FusionRunResult) -> dict[str, Any]:
        return {
            "ok": True,
            "status": result.status,
            "profile": result.profile,
            "device": result.runtime.device,
            "checkpoint_epoch": result.runtime.checkpoint_epoch,
            "runtime_seconds": result.runtime.runtime_seconds,
            "metrics": result.metrics.model_dump(mode="json"),
            "artifacts": {
                "predicted_hs": Path(result.predicted_hs_path).name,
                "rgb_preview": Path(result.rgb_preview_path).name,
                "reference_rgb_preview": Path(result.reference_rgb_preview_path).name,
                "sam_heatmap": Path(result.sam_heatmap_path).name,
                "metrics": Path(result.metrics_path).name,
                **{key: Path(path).name for key, path in result.input_preview_paths.items()},
                "manifest": Path(result.manifest_path).name,
                "report": Path(result.report_path).name,
            },
            "warnings": result.warnings,
        }


class TiffLLMToolContext(BaseModel):
    """Local-only configuration for one crop-manifest TIFF Agent session."""

    crop_manifest_path: Path | None = None
    auxiliary_ms_path: Path | None = None
    auxiliary_hs_path: Path | None = None
    target_ms_path: Path | None = None
    target_hs_reference_path: Path | None = None
    crop_profile: str = "yre_legacy_test_v1"
    ms_row_offset: int | None = Field(default=None, ge=0)
    ms_col_offset: int | None = Field(default=None, ge=0)
    window_height: int | None = Field(default=None, gt=0)
    window_width: int | None = Field(default=None, gt=0)
    hs_row_offset: int | None = Field(default=None, ge=0)
    hs_col_offset: int | None = Field(default=None, ge=0)
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


TiffWorkflowFactory = Callable[[], YRE151TiffPatchAgent]


class TiffAgentToolbox:
    """Allowlisted raw-TIFF tools bound to one user-authored crop manifest."""

    def __init__(
        self,
        context: TiffLLMToolContext,
        workflow_factory: TiffWorkflowFactory = YRE151TiffPatchAgent,
    ) -> None:
        self.context = context
        self.workflow_factory = workflow_factory
        self.inspected = False
        self.latest_result: TiffFusionResult | None = None
        self.raw_inputs_inspected = False
        self.active_crop_manifest_path = context.crop_manifest_path
        self.runtime_preflight: Any | None = None

    @staticmethod
    def definitions() -> list[dict[str, Any]]:
        empty_schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
        return [
            {
                "type": "function",
                "name": "get_runtime_preflight",
                "description": "Read the already-completed local model environment preflight.",
                "parameters": empty_schema,
                "strict": True,
            },
            {
                "type": "function",
                "name": "inspect_raw_tiff_inputs",
                "description": (
                    "Inspect only the configured original T1 MS, T1 HS, T2 MS and optional "
                    "T2 HS TIFF inputs. It writes bounded RGB previews. Do not crop or run "
                    "fusion for an inspection-only request."
                ),
                "parameters": empty_schema,
                "strict": True,
            },
            {
                "type": "function",
                "name": "prepare_tiff_crop",
                "description": (
                    "Create the configured traceable crop manifest from the authorized raw TIFF "
                    "paths and crop window. Call inspect_raw_tiff_inputs successfully first."
                ),
                "parameters": empty_schema,
                "strict": True,
            },
            {
                "type": "function",
                "name": "inspect_yre151_tiff_crop",
                "description": (
                    "Validate the configured local crop manifest and its three TIFF crop artifacts. "
                    "The manifest is an explicit pixel-correspondence assumption, not auto-registration."
                ),
                "parameters": empty_schema,
                "strict": True,
            },
            {
                "type": "function",
                "name": "run_yre151_tiff_fusion",
                "description": (
                    "Run the configured DC-STSF checkpoint for the single TIFF patch authorized by "
                    "the crop manifest. Call inspect_yre151_tiff_crop successfully first."
                ),
                "parameters": empty_schema,
                "strict": True,
            },
            {
                "type": "function",
                "name": "get_latest_tiff_fusion_result",
                "description": "Read artifact names and runtime metadata from the latest local TIFF run.",
                "parameters": empty_schema,
                "strict": True,
            },
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if arguments:
            raise ValueError(f"{name} does not accept arguments")
        if name == "get_runtime_preflight":
            return self._preflight_summary()
        if name == "inspect_raw_tiff_inputs":
            return self._inspect_raw_tiff_inputs()
        if name == "prepare_tiff_crop":
            return self._prepare_tiff_crop()
        if name == "inspect_yre151_tiff_crop":
            if self.active_crop_manifest_path is None:
                raise ValueError(
                    "No crop manifest is available. Call inspect_raw_tiff_inputs and "
                    "prepare_tiff_crop first."
                )
            inspection, crop = inspect_manifest_crop_triplet(self.active_crop_manifest_path)
            self.inspected = True
            return {
                "ok": True,
                "profile": crop.profile,
                "alignment_mode": "explicit_crop_manifest",
                "ms_shape": [inspection.target_ms.height, inspection.target_ms.width, inspection.target_ms.band_count],
                "hs_shape": [inspection.auxiliary_hs.height, inspection.auxiliary_hs.width, inspection.auxiliary_hs.band_count],
                "windows": {
                    "auxiliary_ms": crop.auxiliary_ms_window.model_dump(),
                    "auxiliary_hs": crop.auxiliary_hs_window.model_dump(),
                    "target_ms": crop.target_ms_window.model_dump(),
                    "target_hs_reference": (
                        crop.target_hs_reference_window.model_dump()
                        if crop.target_hs_reference_window is not None
                        else None
                    ),
                },
                "experiment_mode": crop.experiment_mode,
                "warnings": inspection.warnings,
            }
        if name == "run_yre151_tiff_fusion":
            if not self.inspected:
                raise ValueError(
                    "Safety gate: call inspect_yre151_tiff_crop successfully before TIFF inference"
                )
            result = self.workflow_factory().run(
                TiffFusionRequest(
                    crop_manifest_path=self.active_crop_manifest_path,
                    checkpoint_path=self.context.checkpoint_path,
                    model_python=self.context.model_python,
                    output_dir=self.context.output_dir,
                    experiment_mode=self.context.experiment_mode,
                    patch_size=self.context.patch_size,
                    patch_index=self.context.patch_index,
                    row_offset=self.context.row_offset,
                    col_offset=self.context.col_offset,
                    device=self.context.device,
                    timeout_seconds=self.context.timeout_seconds,
                    runtime_retries=self.context.runtime_retries,
                )
            )
            self.latest_result = result
            return self._result_summary(result)
        if name == "get_latest_tiff_fusion_result":
            if self.latest_result is None:
                raise ValueError("No TIFF fusion result is available in this agent session")
            return self._result_summary(self.latest_result)
        raise ValueError(f"Tool is not allowlisted: {name}")

    def _preflight_summary(self) -> dict[str, Any]:
        if self.runtime_preflight is None:
            raise ValueError("No runtime preflight result is available for this agent session")
        return {"ok": True, **self.runtime_preflight.model_dump(mode="json")}

    def _require_raw_inputs(self) -> tuple[Path, Path, Path]:
        paths = (
            self.context.auxiliary_ms_path,
            self.context.auxiliary_hs_path,
            self.context.target_ms_path,
        )
        if any(path is None for path in paths):
            raise ValueError(
                "Raw TIFF inputs are not configured for this session. Use a configured raw-TIFF "
                "run or provide a crop manifest."
            )
        return paths[0], paths[1], paths[2]

    def _inspect_raw_tiff_inputs(self) -> dict[str, Any]:
        auxiliary_ms, auxiliary_hs, target_ms = self._require_raw_inputs()
        if self.context.experiment_mode == "simulation" and self.context.target_hs_reference_path is None:
            raise ValueError("Simulation mode requires a configured target-time HS TIFF reference")
        inspection = inspect_tiff_triplet(auxiliary_ms, auxiliary_hs, target_ms)
        output_dir = self.context.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, str] = {}
        previews = (
            ("raw_auxiliary_ms_rgb", auxiliary_ms, (2, 1, 0)),
            ("raw_auxiliary_hs_rgb", auxiliary_hs, (28, 18, 9)),
            ("raw_target_ms_rgb", target_ms, (2, 1, 0)),
        )
        for artifact_key, path, bands in previews:
            artifact_name = f"{artifact_key}.png"
            Image.fromarray(render_raster_rgb(path, bands=bands, max_dimension=1024)).save(
                output_dir / artifact_name
            )
            artifacts[artifact_key] = artifact_name
        target_hs_reference = None
        if self.context.target_hs_reference_path is not None:
            target_hs_reference = inspect_raster(self.context.target_hs_reference_path)
            artifact_name = "raw_target_hs_reference_rgb.png"
            Image.fromarray(
                render_raster_rgb(
                    self.context.target_hs_reference_path,
                    bands=(28, 18, 9),
                    max_dimension=1024,
                )
            ).save(output_dir / artifact_name)
            artifacts["raw_target_hs_reference_rgb"] = artifact_name
        self.raw_inputs_inspected = True
        return {
            "ok": True,
            "is_ready_for_preprocessing": inspection.is_ready_for_preprocessing,
            "auxiliary_ms": inspection.auxiliary_ms.model_dump(mode="json"),
            "auxiliary_hs": inspection.auxiliary_hs.model_dump(mode="json"),
            "target_ms": inspection.target_ms.model_dump(mode="json"),
            "target_hs_reference": (
                target_hs_reference.model_dump(mode="json")
                if target_hs_reference is not None
                else None
            ),
            "artifacts": artifacts,
            "blocking_issues": inspection.blocking_issues,
            "warnings": inspection.warnings,
        }

    def _prepare_tiff_crop(self) -> dict[str, Any]:
        if not self.raw_inputs_inspected:
            raise ValueError("Safety gate: inspect_raw_tiff_inputs before creating a TIFF crop")
        auxiliary_ms, auxiliary_hs, target_ms = self._require_raw_inputs()
        result = crop_tiff_triplet(
            auxiliary_ms,
            auxiliary_hs,
            target_ms,
            self.context.output_dir / "prepared_crop",
            target_hs_reference_path=self.context.target_hs_reference_path,
            experiment_mode=self.context.experiment_mode or "real",
            profile=self.context.crop_profile,
            ms_row_offset=self.context.ms_row_offset,
            ms_col_offset=self.context.ms_col_offset,
            window_height=self.context.window_height,
            window_width=self.context.window_width,
            hs_row_offset=self.context.hs_row_offset,
            hs_col_offset=self.context.hs_col_offset,
        )
        self.active_crop_manifest_path = Path(result.manifest_path)
        return {
            "ok": True,
            "profile": result.profile,
            "experiment_mode": result.experiment_mode,
            "manifest": Path(result.manifest_path).name,
            "artifacts": {"crop_manifest": str(Path(result.manifest_path).relative_to(self.context.output_dir))},
            "warnings": result.warnings,
        }

    def sanitize_error(self, error: Exception) -> dict[str, Any]:
        message = str(error)
        configured_paths = {
            "crop_manifest_path": self.context.crop_manifest_path,
            "checkpoint_path": self.context.checkpoint_path,
            "model_python": self.context.model_python,
            "output_dir": self.context.output_dir,
        }
        for label, path in configured_paths.items():
            raw_path = str(path)
            resolved_path = str(path.expanduser().resolve())
            for candidate in {raw_path, resolved_path, raw_path.replace("\\", "/")}:
                if candidate:
                    message = message.replace(candidate, f"<{label}>")
        diagnosis = diagnose_error(error)
        return {
            "ok": False,
            "error_type": type(error).__name__,
            "error": message,
            "diagnosis": diagnosis.model_dump(mode="json"),
        }

    @staticmethod
    def _result_summary(result: TiffFusionResult) -> dict[str, Any]:
        return {
            "ok": True,
            "status": result.status,
            "profile": result.profile,
            "experiment_mode": result.experiment_mode,
            "reference_kind": result.reference_kind,
            "patch_index": result.patch_index,
            "total_patch_count": result.total_patch_count,
            "alignment_mode": result.spatial_metadata.alignment_mode,
            "device": result.runtime.device,
            "checkpoint_epoch": result.runtime.checkpoint_epoch,
            "runtime_seconds": result.runtime.runtime_seconds,
            "metrics_status": result.metrics_status,
            "metrics": result.metrics.model_dump(mode="json") if result.metrics is not None else None,
            "artifacts": {
                "predicted_hs": Path(result.predicted_hs_path).name,
                "rgb_preview": Path(result.rgb_preview_path).name,
                "reference_rgb_preview": (
                    Path(result.reference_rgb_preview_path).name
                    if result.reference_rgb_preview_path
                    else None
                ),
                "sam_heatmap": (
                    Path(result.sam_heatmap_path).name if result.sam_heatmap_path else None
                ),
                "metrics": Path(result.metrics_path).name if result.metrics_path else None,
                **{key: Path(path).name for key, path in result.input_preview_paths.items()},
                "manifest": Path(result.manifest_path).name,
                "report": Path(result.report_path).name,
            },
            "warnings": result.warnings,
        }
