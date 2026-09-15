"""Portable execution traces and Markdown reports for local Agent runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from rsfusion_agent.agent.task_plan import build_recovery_actions
from rsfusion_agent.agent.task_state import TaskState, load_task_state


class ToolExecutionEvent(BaseModel):
    round_index: int | None = None
    tool: str
    status: str
    elapsed_seconds: float = 0.0
    argument_keys: list[str] = Field(default_factory=list)
    error_type: str | None = None
    error: str | None = None


class ExecutionObservabilityRecord(BaseModel):
    """Sanitized trace suitable for sharing without data arrays or API keys."""

    request: str
    status: str
    tools: list[ToolExecutionEvent]
    total_tool_seconds: float
    failed_tool_count: int
    llm_usage: dict[str, int] = Field(default_factory=dict)
    estimated_cost: dict[str, Any] = Field(default_factory=dict)
    task_state: dict[str, object]
    recovery_actions: list[dict[str, str]] = Field(default_factory=list)
    artifact_names: list[str] = Field(default_factory=list)


def _artifact_names(value: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key.endswith(("_path", "_file")) and isinstance(item, str):
                names.add(Path(item).name)
            else:
                names.update(_artifact_names(item))
    elif isinstance(value, list):
        for item in value:
            names.update(_artifact_names(item))
    return names


def build_execution_observability_record(
    agent_result: dict[str, Any], *, task_state: TaskState | None = None
) -> ExecutionObservabilityRecord:
    """Reduce a raw Agent result into a compact, path-safe diagnostic trace."""

    events: list[ToolExecutionEvent] = []
    errors: list[str] = []
    for item in agent_result.get("trace") or []:
        if not isinstance(item, dict):
            continue
        output = item.get("output") if isinstance(item.get("output"), dict) else {}
        error = output.get("error") if isinstance(output, dict) else None
        if isinstance(error, str):
            errors.append(error)
        arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        events.append(
            ToolExecutionEvent(
                round_index=item.get("round_index"),
                tool=str(item.get("name", "unknown")),
                status=str(item.get("status", "unknown")),
                elapsed_seconds=float(item.get("elapsed_seconds", 0.0)),
                argument_keys=sorted(str(key) for key in arguments),
                error_type=str(output.get("error_type")) if output.get("error_type") else None,
                error=error if isinstance(error, str) else None,
            )
        )
    state = task_state or TaskState()
    usage = agent_result.get("usage") if isinstance(agent_result.get("usage"), dict) else {}
    artifact_names = sorted(_artifact_names(agent_result.get("fusion_result") or {}))
    recovery = [
        item.model_dump(mode="json") for error in errors for item in build_recovery_actions(error)
    ]
    return ExecutionObservabilityRecord(
        request=str(agent_result.get("request", "")),
        status=str(agent_result.get("status", "unknown")),
        tools=events,
        total_tool_seconds=round(sum(event.elapsed_seconds for event in events), 6),
        failed_tool_count=sum(event.status == "failed" for event in events),
        llm_usage={key: int(value) for key, value in usage.items() if isinstance(value, int)},
        estimated_cost=(
            agent_result.get("estimated_cost")
            if isinstance(agent_result.get("estimated_cost"), dict)
            else {}
        ),
        task_state=state.summary(),
        recovery_actions=recovery,
        artifact_names=artifact_names,
    )


def _report_markdown(record: ExecutionObservabilityRecord) -> str:
    rows = ["| Tool | Status | Seconds | Arguments |", "|---|---:|---:|---|"]
    for event in record.tools:
        rows.append(
            f"| {event.tool} | {event.status} | {event.elapsed_seconds:.3f} | "
            f"{', '.join(event.argument_keys) or '-'} |"
        )
    recovery = "\n".join(
        f"- **{item['category']}**: {item['summary']} Action: {item['action']}"
        for item in record.recovery_actions
    ) or "- No recovery action is needed."
    artifacts = ", ".join(record.artifact_names) or "None"
    return (
        "# RSFusionAgent execution report\n\n"
        f"- Status: `{record.status}`\n"
        f"- Task stage: `{record.task_state['stage']}` ({record.task_state['stage_label']})\n"
        f"- Next action: {record.task_state['next_action']}\n"
        f"- Tool time: {record.total_tool_seconds:.3f} seconds\n"
        f"- Failed tools: {record.failed_tool_count}\n"
        f"- Artifacts: {artifacts}\n\n"
        "## Request\n\n"
        f"{record.request}\n\n"
        "## Tool trace\n\n"
        + "\n".join(rows)
        + "\n\n## Recovery guidance\n\n"
        + recovery
        + "\n"
    )


def write_execution_observability(
    output_dir: str | Path,
    agent_result: dict[str, Any],
    *,
    task_state_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write trace JSON and a human-readable report alongside an Agent result."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    state = load_task_state(task_state_path or directory / "task_state.json")
    record = build_execution_observability_record(agent_result, task_state=state)
    trace_path = directory / "execution_trace.json"
    report_path = directory / "agent_execution_report.md"
    trace_path.write_text(
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_report_markdown(record), encoding="utf-8")
    return trace_path, report_path


__all__ = [
    "ExecutionObservabilityRecord",
    "ToolExecutionEvent",
    "build_execution_observability_record",
    "write_execution_observability",
]
