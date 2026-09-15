"""Explicit, recoverable execution plans derived from persisted task state."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from rsfusion_agent.agent.task_state import TaskState
from rsfusion_agent.tools.error_diagnosis import diagnose_error


class PlanStepStatus(str, Enum):
    """Presentation state for one deterministic workflow step."""

    COMPLETED = "completed"
    READY = "ready"
    BLOCKED = "blocked"
    NOT_REQUIRED = "not_required"


class TaskPlanStep(BaseModel):
    key: str
    label: str
    status: PlanStepStatus
    detail: str
    blocked_by: list[str] = Field(default_factory=list)


class RecoveryAction(BaseModel):
    category: str
    summary: str
    action: str


class TaskExecutionPlan(BaseModel):
    """Small serializable plan that never contains image arrays or secrets."""

    input_mode: str
    steps: list[TaskPlanStep]
    next_action: str
    recovery_actions: list[RecoveryAction] = Field(default_factory=list)


def _has_any(completed_tools: set[str], *names: str) -> bool:
    return bool(completed_tools.intersection(names))


def _step(
    key: str,
    label: str,
    *,
    completed: bool,
    ready: bool,
    detail: str,
    blocked_by: list[str] | None = None,
) -> TaskPlanStep:
    if completed:
        status = PlanStepStatus.COMPLETED
    elif ready:
        status = PlanStepStatus.READY
    else:
        status = PlanStepStatus.BLOCKED
    return TaskPlanStep(
        key=key,
        label=label,
        status=status,
        detail=detail,
        blocked_by=blocked_by or [],
    )


def build_recovery_actions(error_text: str | None) -> list[RecoveryAction]:
    """Turn a sanitized error into a short, executable recovery suggestion."""

    if not error_text or not error_text.strip():
        return []
    message = error_text.lower()
    if "crop manifest" in message:
        return [
            RecoveryAction(
                category="workflow_dependency",
                summary="当前运行没有可复用的裁剪清单。",
                action="先检查原始 TIFF 输入并执行裁剪，确认生成 crop_manifest.json 后再融合。",
            )
        ]
    if "target-time hs" in message or "target hs" in message:
        return [
            RecoveryAction(
                category="input_requirement",
                summary="模拟实验缺少目标时相 HS 参考。",
                action="在左侧补充 T2 HS TIFF；真实实验可不填，但指标会标记为伪参考评估。",
            )
        ]
    if "patch size exceeds" in message:
        return [
            RecoveryAction(
                category="patch_configuration",
                summary="模型 Patch 超出当前裁剪输出网格。",
                action="减小模型 Patch 边长，或扩大原始 TIFF 裁剪范围后重新生成裁剪清单。",
            )
        ]
    diagnosis = diagnose_error(RuntimeError(error_text))
    return [
        RecoveryAction(
            category=diagnosis.category,
            summary=diagnosis.summary,
            action=diagnosis.recommended_action,
        )
    ]


def build_task_execution_plan(
    state: TaskState,
    *,
    input_mode: str,
    crop_manifest_available: bool = False,
    last_error: str | None = None,
) -> TaskExecutionPlan:
    """Build the next executable plan from tool history rather than chat history.

    TIFF work requires a crop manifest before fusion. H5 data is already preprocessed,
    so its crop step is intentionally omitted. Input checking and runtime preflight can
    proceed independently after a task is created.
    """

    if input_mode not in {"h5", "tiff"}:
        raise ValueError("input_mode must be 'h5' or 'tiff'")
    completed = set(state.completed_tools)
    input_checked = _has_any(completed, "inspect_yre151_h5", "inspect_raw_tiff_inputs")
    runtime_ready = "get_runtime_preflight" in completed
    crop_completed = crop_manifest_available or "prepare_tiff_crop" in completed
    fusion_completed = _has_any(completed, "run_yre151_fusion", "run_yre151_tiff_fusion")
    result_loaded = _has_any(
        completed, "get_latest_fusion_result", "get_latest_tiff_fusion_result"
    )

    steps = [
        _step(
            "input_check",
            "输入数据检查",
            completed=input_checked,
            ready=not input_checked,
            detail="校验数据契约并生成输入 RGB 预览。",
        ),
        _step(
            "runtime_preflight",
            "模型环境预检",
            completed=runtime_ready,
            ready=not runtime_ready,
            detail="验证 Checkpoint、Conda Python、PyTorch 与 CUDA，不运行模型推理。",
        ),
    ]
    if input_mode == "tiff":
        steps.append(
            _step(
                "crop",
                "生成裁剪清单",
                completed=crop_completed,
                ready=input_checked and not crop_completed,
                detail="生成可追溯 crop_manifest.json，并选择后续融合 Patch。",
                blocked_by=[] if input_checked else ["输入数据检查"],
            )
        )
    fusion_blockers: list[str] = []
    if not input_checked:
        fusion_blockers.append("输入数据检查")
    if not runtime_ready:
        fusion_blockers.append("模型环境预检")
    if input_mode == "tiff" and not crop_completed:
        fusion_blockers.append("裁剪清单")
    steps.extend(
        [
            _step(
                "fusion",
                "融合推理",
                completed=fusion_completed,
                ready=not fusion_completed and not fusion_blockers,
                detail="调用已授权的本地 GPU 推理工具执行单 Patch 融合。",
                blocked_by=fusion_blockers,
            ),
            _step(
                "result",
                "结果评估与导出",
                completed=result_loaded,
                ready=fusion_completed and not result_loaded,
                detail="读取指标、可视化、结构化 Trace 与本地实验报告。",
                blocked_by=[] if fusion_completed else ["融合推理"],
            ),
        ]
    )
    next_step = next((step for step in steps if step.status is PlanStepStatus.READY), None)
    next_action = next_step.detail if next_step is not None else "任务流程已完成，可查看或导出结果。"
    return TaskExecutionPlan(
        input_mode=input_mode,
        steps=steps,
        next_action=next_action,
        recovery_actions=build_recovery_actions(last_error),
    )


__all__ = [
    "PlanStepStatus",
    "RecoveryAction",
    "TaskExecutionPlan",
    "TaskPlanStep",
    "build_recovery_actions",
    "build_task_execution_plan",
]
