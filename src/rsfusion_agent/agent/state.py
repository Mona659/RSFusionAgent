"""Typed state models for the first RSFusionAgent workflow."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from rsfusion_agent.tools.h5_patch import H5PairInspection
from rsfusion_agent.tools.metrics import FusionMetrics
from rsfusion_agent.tools.model_runtime import ModelRuntimeResult


class FusionRunRequest(BaseModel):
    auxiliary_h5_path: Path
    target_h5_path: Path
    checkpoint_path: Path
    model_python: Path
    output_dir: Path
    patch_index: int = Field(default=0, ge=0)
    device: str = "auto"
    timeout_seconds: int = Field(default=600, gt=0)
    runtime_retries: int = Field(default=1, ge=0, le=2)
    rgb_bands: tuple[int, int, int] = (28, 18, 9)


class ToolTrace(BaseModel):
    name: str
    status: str
    elapsed_seconds: float = Field(ge=0)
    summary: str


class FusionRunResult(BaseModel):
    status: str
    profile: str
    inspection: H5PairInspection
    runtime: ModelRuntimeResult
    metrics: FusionMetrics
    predicted_hs_path: str
    rgb_preview_path: str
    reference_rgb_preview_path: str
    sam_heatmap_path: str
    metrics_path: str
    input_preview_paths: dict[str, str]
    manifest_path: str
    report_path: str
    trace: list[ToolTrace]
    warnings: list[str]
