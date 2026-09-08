from pathlib import Path

from rsfusion_agent.ui.streamlit_app import (
    UiRunConfig,
    build_agent_command,
    format_history_label,
    list_run_history,
    load_json_object,
    resolve_result_artifact,
)


def test_build_agent_command_uses_paths_and_never_accepts_api_keys(tmp_path: Path) -> None:
    config = UiRunConfig(
        request="融合 patch 0",
        auxiliary_h5_path=tmp_path / "aux.h5",
        target_h5_path=tmp_path / "target.h5",
        checkpoint_path=tmp_path / "model.pth",
        model_python=tmp_path / "model-python.exe",
        output_dir=tmp_path / "outputs" / "run-1",
        provider="qwen",
        llm_model="qwen3.7-flash",
        base_url="https://example.invalid/v1",
        patch_index=0,
        device="cuda",
    )

    command = build_agent_command(config, python_executable="ui-python.exe")

    assert command[:4] == ["ui-python.exe", "-m", "rsfusion_agent.cli", "agent"]
    assert "--provider" in command
    assert command[command.index("--provider") + 1] == "qwen"
    assert "--skip-preflight" not in command
    assert command[command.index("--runtime-retries") + 1] == "1"
    assert not any("api-key" in item.lower() or "api_key" in item.lower() for item in command)


def test_load_json_object_handles_valid_and_invalid_files(tmp_path: Path) -> None:
    valid = tmp_path / "result.json"
    valid.write_text('{"status": "completed"}\n', encoding="utf-8")
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not json", encoding="utf-8")

    assert load_json_object(valid) == {"status": "completed"}
    assert load_json_object(invalid) is None
    assert load_json_object(tmp_path / "missing.json") is None


def test_resolve_result_artifact_supports_full_workflow_paths_and_stays_in_output_dir(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    preview = output_dir / "rgb_preview.png"
    preview.write_bytes(b"synthetic preview")

    path = resolve_result_artifact(
        {"rgb_preview_path": str(preview)},
        output_dir=output_dir,
        artifact_key="rgb_preview",
    )

    assert path == preview
    assert (
        resolve_result_artifact(
            {"rgb_preview_path": str(tmp_path / "outside.png")},
            output_dir=output_dir,
            artifact_key="rgb_preview",
        )
        is None
    )


def test_list_run_history_loads_results_without_running_the_agent(tmp_path: Path) -> None:
    root = tmp_path / "ui_runs"
    successful = root / "successful"
    failed = root / "failed"
    successful.mkdir(parents=True)
    failed.mkdir()
    (successful / "agent_result.json").write_text(
        '{"status": "completed", "fusion_result": {"metrics": {"psnr": 32.0}}}',
        encoding="utf-8",
    )
    (failed / "agent_result.json").write_text(
        '{"status": "completed_with_tool_errors"}',
        encoding="utf-8",
    )

    history = list_run_history(root)

    assert {item.status for item in history} == {"completed", "completed_with_tool_errors"}
    assert any("PSNR 32.000" in format_history_label(item) for item in history)
