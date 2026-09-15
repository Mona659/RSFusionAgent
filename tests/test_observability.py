from pathlib import Path

from rsfusion_agent.agent.observability import (
    build_execution_observability_record,
    write_execution_observability,
)
from rsfusion_agent.agent.task_state import TaskState, save_task_state


def _result() -> dict:
    return {
        "status": "completed_with_tool_errors",
        "request": "继续融合",
        "usage": {"input_tokens": 12, "output_tokens": 3},
        "estimated_cost": {"currency": "CNY", "estimated_cost": 0.01},
        "trace": [
            {
                "round_index": 1,
                "name": "run_yre151_tiff_fusion",
                "arguments": {"patch_index": 0, "checkpoint_path": "secret-local-path"},
                "status": "failed",
                "elapsed_seconds": 1.25,
                "output": {"error_type": "ValueError", "error": "No crop manifest is available"},
            }
        ],
        "fusion_result": {"artifacts": {"rgb_preview_path": "D:/private/output/rgb.png"}},
    }


def test_observability_record_keeps_argument_names_but_not_values() -> None:
    record = build_execution_observability_record(_result())

    assert record.tools[0].argument_keys == ["checkpoint_path", "patch_index"]
    assert "secret-local-path" not in record.model_dump_json()
    assert record.recovery_actions[0]["category"] == "workflow_dependency"
    assert record.artifact_names == ["rgb.png"]


def test_observability_writer_creates_trace_and_markdown_report(tmp_path: Path) -> None:
    state = TaskState()
    state.record("inspect_raw_tiff_inputs")
    state_path = tmp_path / "task_state.json"
    save_task_state(state_path, state)

    trace_path, report_path = write_execution_observability(
        tmp_path, _result(), task_state_path=state_path
    )

    assert trace_path.is_file()
    assert report_path.is_file()
    assert "RSFusionAgent execution report" in report_path.read_text(encoding="utf-8")
