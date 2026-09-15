from rsfusion_agent.agent.task_plan import build_recovery_actions, build_task_execution_plan
from rsfusion_agent.agent.task_state import TaskState


def test_tiff_plan_blocks_fusion_until_crop_and_preflight_are_complete() -> None:
    state = TaskState()
    state.record("inspect_raw_tiff_inputs")

    plan = build_task_execution_plan(state, input_mode="tiff")
    steps = {item.key: item for item in plan.steps}

    assert steps["crop"].status.value == "ready"
    assert steps["runtime_preflight"].status.value == "ready"
    assert steps["fusion"].status.value == "blocked"
    assert set(steps["fusion"].blocked_by) == {"模型环境预检", "裁剪清单"}


def test_plan_marks_fusion_ready_after_independent_dependencies_complete() -> None:
    state = TaskState()
    state.record("inspect_raw_tiff_inputs")
    state.record("prepare_tiff_crop")
    state.record("get_runtime_preflight")

    plan = build_task_execution_plan(state, input_mode="tiff")
    steps = {item.key: item for item in plan.steps}

    assert steps["fusion"].status.value == "ready"
    assert "Patch" in plan.next_action


def test_recovery_actions_explain_crop_manifest_error() -> None:
    actions = build_recovery_actions("No crop manifest is available")

    assert actions[0].category == "workflow_dependency"
    assert "crop_manifest.json" in actions[0].action
