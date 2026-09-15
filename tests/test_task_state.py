from rsfusion_agent.agent.llm_tools import QueryLLMToolContext, QueryToolbox
from rsfusion_agent.agent.task_state import (
    TaskStage,
    TaskState,
    load_task_state,
    save_task_state,
)


def test_task_state_advances_with_tools_and_does_not_regress() -> None:
    state = TaskState()
    state.record("inspect_raw_tiff_inputs")
    state.record("prepare_tiff_crop")
    state.record("run_yre151_tiff_fusion")
    state.record("inspect_raw_tiff_inputs")
    assert state.stage == TaskStage.FUSION_COMPLETED
    assert state.completed_tools == [
        "inspect_raw_tiff_inputs",
        "prepare_tiff_crop",
        "run_yre151_tiff_fusion",
    ]


def test_task_state_persists_without_image_or_model_payloads(tmp_path) -> None:
    path = tmp_path / "task_state.json"
    state = TaskState()
    state.record("inspect_raw_tiff_inputs")
    state.record("prepare_tiff_crop")

    save_task_state(path, state)
    restored = load_task_state(path)

    assert restored.stage is TaskStage.CROP_PREPARED
    assert restored.completed_tools == ["inspect_raw_tiff_inputs", "prepare_tiff_crop"]
    assert "image" not in path.read_text(encoding="utf-8").lower()


def test_read_only_query_toolbox_reads_persisted_ui_task_state(tmp_path) -> None:
    path = tmp_path / "task_state.json"
    state = TaskState()
    state.record("inspect_raw_tiff_inputs")
    state.record("prepare_tiff_crop")
    save_task_state(path, state)

    toolbox = QueryToolbox(
        QueryLLMToolContext(output_dir=tmp_path / "query", task_state_path=path)
    )

    summary = toolbox.execute("get_task_state", {})

    assert summary["stage"] == "crop_prepared"
    assert summary["stage_label"] == "裁剪已完成（可融合）"
    assert summary["next_action"] == "裁剪清单和 Patch 已就绪，可直接执行融合。"
    assert summary["completed_tools"] == ["inspect_raw_tiff_inputs", "prepare_tiff_crop"]
