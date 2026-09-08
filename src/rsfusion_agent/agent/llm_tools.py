"""Allowlisted local tools exposed to the language-model control plane."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rsfusion_agent.agent.state import FusionRunRequest, FusionRunResult
from rsfusion_agent.agent.workflow import YRE151PatchAgent
from rsfusion_agent.tools.h5_patch import H5PairInspection, inspect_h5_pair


class LLMToolContext(BaseModel):
    auxiliary_h5_path: Path
    target_h5_path: Path
    checkpoint_path: Path
    model_python: Path
    output_dir: Path
    patch_index: int = Field(default=0, ge=0)
    device: str = "auto"
    timeout_seconds: int = Field(default=600, gt=0)


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
        return {
            "ok": False,
            "error_type": type(error).__name__,
            "error": message,
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
                "sam_heatmap": Path(result.sam_heatmap_path).name,
                "metrics": Path(result.metrics_path).name,
                "manifest": Path(result.manifest_path).name,
                "report": Path(result.report_path).name,
            },
            "warnings": result.warnings,
        }
