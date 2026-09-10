"""Allowlisted local tools exposed to the language-model control plane."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

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
from rsfusion_agent.tools.tiff_crop import ExperimentMode
from rsfusion_agent.tools.tiff_patch import inspect_manifest_crop_triplet


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

    crop_manifest_path: Path
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
        if name == "inspect_yre151_tiff_crop":
            inspection, crop = inspect_manifest_crop_triplet(self.context.crop_manifest_path)
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
                    crop_manifest_path=self.context.crop_manifest_path,
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
