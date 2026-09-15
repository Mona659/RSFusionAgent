"""Explicit state machine for a single local fusion task."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class TaskStage(str, Enum):
    CREATED = "created"
    INPUT_CHECKED = "input_checked"
    CROP_PREPARED = "crop_prepared"
    RUNTIME_READY = "runtime_ready"
    INFERENCE_AUTHORIZED = "inference_authorized"
    FUSION_COMPLETED = "fusion_completed"
    RESULT_LOADED = "result_loaded"


_TRANSITIONS: dict[str, TaskStage] = {
    "inspect_yre151_h5": TaskStage.INPUT_CHECKED,
    "inspect_raw_tiff_inputs": TaskStage.INPUT_CHECKED,
    "prepare_tiff_crop": TaskStage.CROP_PREPARED,
    "inspect_yre151_tiff_crop": TaskStage.INPUT_CHECKED,
    "get_runtime_preflight": TaskStage.RUNTIME_READY,
    "run_yre151_fusion": TaskStage.FUSION_COMPLETED,
    "run_yre151_tiff_fusion": TaskStage.FUSION_COMPLETED,
    "get_latest_fusion_result": TaskStage.RESULT_LOADED,
    "get_latest_tiff_fusion_result": TaskStage.RESULT_LOADED,
}

_STAGE_LABELS: dict[TaskStage, str] = {
    TaskStage.CREATED: "任务已创建",
    TaskStage.INPUT_CHECKED: "输入检查已完成",
    TaskStage.CROP_PREPARED: "裁剪已完成（可融合）",
    TaskStage.RUNTIME_READY: "模型环境已就绪",
    TaskStage.INFERENCE_AUTHORIZED: "推理已获授权",
    TaskStage.FUSION_COMPLETED: "融合已完成",
    TaskStage.RESULT_LOADED: "结果已读取",
}

_STAGE_NEXT_ACTIONS: dict[TaskStage, str] = {
    TaskStage.CREATED: "请先检查输入数据。",
    TaskStage.INPUT_CHECKED: "可执行裁剪，或完成模型环境预检。",
    TaskStage.CROP_PREPARED: "裁剪清单和 Patch 已就绪，可直接执行融合。",
    TaskStage.RUNTIME_READY: "模型环境已就绪；完成输入检查和裁剪后可执行融合。",
    TaskStage.INFERENCE_AUTHORIZED: "可执行融合。",
    TaskStage.FUSION_COMPLETED: "可读取结果、指标和可视化产物。",
    TaskStage.RESULT_LOADED: "任务已完成，可查看或导出结果。",
}


class TaskState(BaseModel):
    stage: TaskStage = TaskStage.CREATED
    completed_tools: list[str] = Field(default_factory=list)

    def record(self, tool_name: str) -> None:
        next_stage = _TRANSITIONS.get(tool_name)
        if next_stage is None:
            return
        if tool_name not in self.completed_tools:
            self.completed_tools.append(tool_name)
        order = list(TaskStage)
        if order.index(next_stage) >= order.index(self.stage):
            self.stage = next_stage

    def summary(self) -> dict[str, object]:
        """Return state-machine data plus unambiguous user-facing wording."""

        return {
            "stage": self.stage.value,
            "stage_label": _STAGE_LABELS[self.stage],
            "next_action": _STAGE_NEXT_ACTIONS[self.stage],
            "completed_tools": self.completed_tools,
        }


def load_task_state(path: str | Path | None) -> TaskState:
    """Load a previous task snapshot, or start a new safe state when none exists."""

    if path is None:
        return TaskState()
    state_path = Path(path)
    if not state_path.is_file():
        return TaskState()
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    return TaskState.model_validate(payload)


def save_task_state(path: str | Path | None, state: TaskState) -> None:
    """Atomically persist a small state snapshot without storing image data."""

    if path is None:
        return
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_path.with_suffix(state_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(state_path)


__all__ = ["TaskStage", "TaskState", "load_task_state", "save_task_state"]
