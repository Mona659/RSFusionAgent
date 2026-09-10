from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import rasterio
from rasterio.transform import from_origin

from rsfusion_agent.ui.streamlit_app import (
    UiRunConfig,
    UiStageRecord,
    _load_crop_preview_cards,
    build_agent_command,
    build_crop_command,
    crop_reuse_key,
    crop_source_bounds,
    format_history_label,
    h5_input_check_key,
    inspect_h5_inputs,
    inspect_raw_tiff_inputs,
    list_run_history,
    load_json_object,
    raw_tiff_check_key,
    resolve_result_artifact,
    sorted_stage_records,
)


def _write_raster(path: Path, *, count: int, height: int, width: int, resolution: int) -> None:
    data = np.arange(count * height * width, dtype=np.float32).reshape(count, height, width)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype="float32",
        transform=from_origin(100, 200, resolution, resolution),
        crs="EPSG:32650",
    ) as dataset:
        dataset.write(data)


def _write_h5(path: Path, *, offset: int = 0) -> None:
    data = np.arange(2 * 12 * 15 * 306, dtype=np.int32).reshape(2, 12, 15, 306)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("test", data=((data + offset) % 5000).astype(np.int16))


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


def test_build_tiff_commands_bind_the_manifest_and_configured_crop_paths(tmp_path: Path) -> None:
    config = UiRunConfig(
        request="检查裁剪清单并融合 TIFF patch",
        checkpoint_path=tmp_path / "model.pth",
        model_python=tmp_path / "model-python.exe",
        output_dir=tmp_path / "outputs" / "run-1",
        input_mode="tiff",
        auxiliary_ms_path=tmp_path / "aux_ms.tif",
        auxiliary_hs_path=tmp_path / "aux_hs.tif",
        target_ms_path=tmp_path / "target_ms.tif",
        target_hs_reference_path=tmp_path / "target_hs.tif",
        experiment_mode="simulation",
        crop_profile="custom",
        ms_row_offset=0,
        ms_col_offset=360,
        window_height=540,
        window_width=540,
        hs_row_offset=0,
        hs_col_offset=120,
        crop_manifest_path=tmp_path / "existing_crop" / "crop_manifest.json",
    )

    agent_command = build_agent_command(config, python_executable="ui-python.exe")
    crop_command = build_crop_command(config, python_executable="ui-python.exe")

    assert agent_command[:4] == ["ui-python.exe", "-m", "rsfusion_agent.cli", "agent-tiff"]
    assert "--crop-manifest" in agent_command
    assert agent_command[agent_command.index("--patch-size") + 1] == "180"
    assert agent_command[agent_command.index("--patch-index") + 1] == "0"
    assert agent_command[agent_command.index("--experiment-mode") + 1] == "simulation"
    assert crop_command[:4] == ["ui-python.exe", "-m", "rsfusion_agent.cli", "crop-tiff-triplet"]
    assert crop_command[crop_command.index("--profile") + 1] == "custom"
    assert crop_command[crop_command.index("--experiment-mode") + 1] == "simulation"
    assert crop_command[crop_command.index("--hs-col-offset") + 1] == "120"
    assert crop_command[crop_command.index("--target-hs-reference") + 1] == str(
        tmp_path / "target_hs.tif"
    )


def test_build_tiff_agent_command_can_authorize_raw_tools_without_a_crop_manifest(
    tmp_path: Path,
) -> None:
    config = UiRunConfig(
        request="仅检查原始 TIFF 并显示 RGB",
        checkpoint_path=tmp_path / "model.pth",
        model_python=tmp_path / "model-python.exe",
        output_dir=tmp_path / "outputs" / "run-raw",
        input_mode="tiff",
        auxiliary_ms_path=tmp_path / "aux_ms.tif",
        auxiliary_hs_path=tmp_path / "aux_hs.tif",
        target_ms_path=tmp_path / "target_ms.tif",
        experiment_mode="real",
    )

    command = build_agent_command(config, python_executable="ui-python.exe")

    assert command[:4] == ["ui-python.exe", "-m", "rsfusion_agent.cli", "agent-tiff"]
    assert command[command.index("--aux-ms") + 1] == str(tmp_path / "aux_ms.tif")
    assert "--crop-manifest" not in command


def test_crop_reuse_key_changes_when_crop_authorization_changes(tmp_path: Path) -> None:
    base = UiRunConfig(
        request="融合",
        checkpoint_path=tmp_path / "model.pth",
        model_python=tmp_path / "python.exe",
        output_dir=tmp_path / "output",
        input_mode="tiff",
        auxiliary_ms_path=tmp_path / "aux_ms.tif",
        auxiliary_hs_path=tmp_path / "aux_hs.tif",
        target_ms_path=tmp_path / "target_ms.tif",
        crop_profile="custom",
        ms_row_offset=0,
        ms_col_offset=0,
        window_height=540,
        window_width=540,
    )
    changed = replace(base, ms_col_offset=360)

    assert crop_reuse_key(base) != crop_reuse_key(changed)


def test_stage_records_are_sorted_newest_first_and_keep_each_stage() -> None:
    now = datetime.now()
    session_state = {
        "ui_stage_records": {
            "preflight": UiStageRecord("preflight", "环境预检", now, {}),
            "crop": UiStageRecord("crop", "TIFF 裁剪", now + timedelta(seconds=1), {}),
            "agent": UiStageRecord("agent", "Agent 融合", now + timedelta(seconds=2), {}),
        }
    }

    records = sorted_stage_records(session_state)

    assert [record.stage for record in records] == ["agent", "crop", "preflight"]


def test_inspect_raw_tiff_inputs_returns_metadata_and_rgb_previews(tmp_path: Path) -> None:
    auxiliary_ms = tmp_path / "aux_ms.tif"
    auxiliary_hs = tmp_path / "aux_hs.tif"
    target_ms = tmp_path / "target_ms.tif"
    _write_raster(auxiliary_ms, count=4, height=12, width=12, resolution=3)
    _write_raster(target_ms, count=4, height=12, width=12, resolution=3)
    _write_raster(auxiliary_hs, count=151, height=4, width=4, resolution=9)
    config = UiRunConfig(
        request="检查输入 TIFF",
        checkpoint_path=tmp_path / "model.pth",
        model_python=tmp_path / "model-python.exe",
        output_dir=tmp_path / "outputs" / "run-1",
        input_mode="tiff",
        auxiliary_ms_path=auxiliary_ms,
        auxiliary_hs_path=auxiliary_hs,
        target_ms_path=target_ms,
    )

    result = inspect_raw_tiff_inputs(config, preview_max_dimension=8)

    assert result.is_ready_for_preprocessing
    assert result.auxiliary_ms["width"] == 12
    assert result.auxiliary_hs["band_count"] == 151
    assert result.auxiliary_ms_rgb.shape == (8, 8, 3)
    assert result.auxiliary_hs_rgb.shape == (4, 4, 3)


def test_raw_tiff_check_key_and_crop_bounds_follow_the_smallest_required_input(
    tmp_path: Path,
) -> None:
    auxiliary_ms = tmp_path / "aux_ms.tif"
    target_ms = tmp_path / "target_ms.tif"
    auxiliary_hs = tmp_path / "aux_hs.tif"
    target_hs = tmp_path / "target_hs.tif"
    _write_raster(auxiliary_ms, count=4, height=15, width=18, resolution=3)
    _write_raster(target_ms, count=4, height=12, width=15, resolution=3)
    _write_raster(auxiliary_hs, count=151, height=5, width=6, resolution=9)
    _write_raster(target_hs, count=151, height=4, width=5, resolution=9)
    config = UiRunConfig(
        request="检查输入 TIFF",
        checkpoint_path=tmp_path / "model.pth",
        model_python=tmp_path / "model-python.exe",
        output_dir=tmp_path / "outputs" / "run-1",
        input_mode="tiff",
        auxiliary_ms_path=auxiliary_ms,
        auxiliary_hs_path=auxiliary_hs,
        target_ms_path=target_ms,
        target_hs_reference_path=target_hs,
        experiment_mode="simulation",
    )

    bounds = crop_source_bounds(inspect_raw_tiff_inputs(config, preview_max_dimension=8))

    assert raw_tiff_check_key(config) == (
        "simulation",
        str(auxiliary_ms),
        str(auxiliary_hs),
        str(target_ms),
        str(target_hs),
    )
    assert (bounds.ms_height, bounds.ms_width) == (12, 15)
    assert (bounds.hs_height, bounds.hs_width) == (4, 5)


def test_inspect_h5_inputs_exposes_patch_count_and_model_input_previews(tmp_path: Path) -> None:
    auxiliary = tmp_path / "auxiliary.h5"
    target = tmp_path / "target.h5"
    _write_h5(auxiliary)
    _write_h5(target, offset=10)
    config = UiRunConfig(
        request="检查 H5 输入",
        checkpoint_path=tmp_path / "model.pth",
        model_python=tmp_path / "model-python.exe",
        output_dir=tmp_path / "outputs" / "run-1",
        input_mode="h5",
        auxiliary_h5_path=auxiliary,
        target_h5_path=target,
        patch_index=1,
    )

    result = inspect_h5_inputs(config)

    assert h5_input_check_key(config) == (str(auxiliary), str(target))
    assert result.patch_index == 1
    assert result.patch_count == 2
    assert (result.patch_height, result.patch_width) == (12, 15)
    assert result.auxiliary_ms_rgb.shape == (12, 15, 3)
    assert result.target_hs_gt_rgb.shape == (12, 15, 3)


def test_crop_preview_cards_render_available_crop_artifacts(tmp_path: Path) -> None:
    auxiliary_ms = tmp_path / "aux_ms_crop.tif"
    auxiliary_hs = tmp_path / "aux_hs_crop.tif"
    target_ms = tmp_path / "target_ms_crop.tif"
    _write_raster(auxiliary_ms, count=4, height=12, width=12, resolution=3)
    _write_raster(auxiliary_hs, count=151, height=4, width=4, resolution=9)
    _write_raster(target_ms, count=4, height=12, width=12, resolution=3)

    cards = _load_crop_preview_cards(
        {
            "auxiliary_ms_path": str(auxiliary_ms),
            "auxiliary_hs_path": str(auxiliary_hs),
            "target_ms_path": str(target_ms),
        }
    )

    assert [card[0] for card in cards] == ["T1 MS 裁剪结果", "T1 HS 裁剪结果", "T2 MS 裁剪结果"]
    assert all(card[2].shape[-1] == 3 for card in cards)


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
