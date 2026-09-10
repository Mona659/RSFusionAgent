"""Streamlit demonstration UI that reuses the hardened RSFusionAgent CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")
WaitReporter = Callable[[float, float], None]


@dataclass(frozen=True)
class UiRunConfig:
    """Non-secret configuration collected by the local UI."""

    request: str
    checkpoint_path: Path
    model_python: Path
    output_dir: Path
    input_mode: str = "h5"
    auxiliary_h5_path: Path | None = None
    target_h5_path: Path | None = None
    auxiliary_ms_path: Path | None = None
    auxiliary_hs_path: Path | None = None
    target_ms_path: Path | None = None
    target_hs_reference_path: Path | None = None
    experiment_mode: str = "real"
    crop_manifest_path: Path | None = None
    crop_profile: str = "yre_legacy_test_v1"
    ms_row_offset: int | None = None
    ms_col_offset: int | None = None
    window_height: int | None = None
    window_width: int | None = None
    hs_row_offset: int | None = None
    hs_col_offset: int | None = None
    provider: str = "qwen"
    llm_model: str | None = None
    base_url: str | None = None
    patch_index: int = 0
    patch_size: int = 180
    row_offset: int = 0
    col_offset: int = 0
    device: str = "cuda"
    timeout_seconds: int = 600
    preflight_timeout_seconds: int = 60
    runtime_retries: int = 1


@dataclass(frozen=True)
class UiAgentExecution:
    """Result of invoking the existing CLI from the UI process."""

    return_code: int
    stdout: str
    stderr: str
    result_path: Path
    result: dict[str, Any] | None


@dataclass(frozen=True)
class UiCropExecution:
    """Result of running the explicit crop stage without invoking the LLM."""

    return_code: int
    stdout: str
    stderr: str
    manifest_path: Path
    result: dict[str, Any] | None


@dataclass(frozen=True)
class UiRunHistoryItem:
    """One locally persisted agent result available for display without a rerun."""

    output_dir: Path
    status: str
    created_at: datetime
    psnr: float | None


@dataclass(frozen=True)
class RawTiffInputCheck:
    """Metadata and compact RGB previews collected before TIFF cropping."""

    source_paths: tuple[Path, ...]
    auxiliary_ms: dict[str, Any]
    auxiliary_hs: dict[str, Any]
    target_ms: dict[str, Any]
    target_hs_reference: dict[str, Any] | None
    auxiliary_ms_rgb: Any
    auxiliary_hs_rgb: Any
    target_ms_rgb: Any
    target_hs_reference_rgb: Any | None
    is_ready_for_preprocessing: bool
    blocking_issues: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class H5InputCheck:
    """Metadata and previews for one selected legacy H5 test patch."""

    source_paths: tuple[Path, Path]
    inspection: dict[str, Any]
    patch_index: int
    patch_count: int
    patch_height: int
    patch_width: int
    auxiliary_ms_rgb: Any
    auxiliary_hs_rgb: Any
    target_ms_rgb: Any
    target_hs_gt_rgb: Any


@dataclass(frozen=True)
class TiffCropSourceBounds:
    """Shared source-grid limits derived from a successful TIFF input check."""

    ms_height: int
    ms_width: int
    hs_height: int
    hs_width: int


@dataclass(frozen=True)
class UiStageRecord:
    """Latest persisted result for one stage in the current UI configuration session."""

    stage: str
    title: str
    completed_at: datetime
    payload: Any
    output_dir: Path | None = None


def crop_reuse_key(config: UiRunConfig) -> tuple[Any, ...]:
    """Identify the exact TIFF inputs and windows authorized by one crop manifest."""

    return (
        config.input_mode,
        config.experiment_mode,
        str(config.auxiliary_ms_path),
        str(config.auxiliary_hs_path),
        str(config.target_ms_path),
        str(config.target_hs_reference_path),
        config.crop_profile,
        config.ms_row_offset,
        config.ms_col_offset,
        config.window_height,
        config.window_width,
        config.hs_row_offset,
        config.hs_col_offset,
        str(config.output_dir),
    )


def raw_tiff_check_key(config: UiRunConfig) -> tuple[str, ...]:
    """Identify raw TIFF inputs whose inspected dimensions may be reused safely."""

    if config.input_mode != "tiff":
        return ("h5",)
    paths = (
        config.auxiliary_ms_path,
        config.auxiliary_hs_path,
        config.target_ms_path,
        config.target_hs_reference_path,
    )
    return (config.experiment_mode, *(str(path) if path is not None else "" for path in paths))


def h5_input_check_key(config: UiRunConfig) -> tuple[str, str]:
    """Identify H5 files whose cached patch count remains valid for selection."""

    return (
        str(config.auxiliary_h5_path) if config.auxiliary_h5_path is not None else "",
        str(config.target_h5_path) if config.target_h5_path is not None else "",
    )


def crop_source_bounds(raw_input_check: RawTiffInputCheck) -> TiffCropSourceBounds:
    """Return the largest source-grid extent simultaneously valid for every input role.

    Crop windows are shared by T1/T2 MS rasters and by T1/T2 HS rasters.  Taking the
    minimum dimension prevents the UI from offering a window that later fails for one
    of the required files.
    """

    ms_rasters = (raw_input_check.auxiliary_ms, raw_input_check.target_ms)
    hs_rasters = [raw_input_check.auxiliary_hs]
    if raw_input_check.target_hs_reference is not None:
        hs_rasters.append(raw_input_check.target_hs_reference)
    return TiffCropSourceBounds(
        ms_height=min(int(item["height"]) for item in ms_rasters),
        ms_width=min(int(item["width"]) for item in ms_rasters),
        hs_height=min(int(item["height"]) for item in hs_rasters),
        hs_width=min(int(item["width"]) for item in hs_rasters),
    )


def _largest_multiple_not_above(value: int, divisor: int) -> int:
    """Return the largest positive multiple of ``divisor`` within ``value``."""

    return max(0, value - (value % divisor))


def run_with_elapsed(
    operation: Callable[[], T],
    *,
    timeout_seconds: int,
    on_wait: WaitReporter | None = None,
) -> T:
    """Run a blocking local action while accurately reporting elapsed wall-clock time."""

    from concurrent.futures import ThreadPoolExecutor

    started_at = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(operation)
        while not future.done():
            if on_wait is not None:
                on_wait(time.perf_counter() - started_at, float(timeout_seconds))
            time.sleep(0.2)
        return future.result()


def build_agent_command(config: UiRunConfig, *, python_executable: str | None = None) -> list[str]:
    """Build a shell-free CLI command without exposing API keys in arguments."""

    command = [python_executable or sys.executable, "-m", "rsfusion_agent.cli"]
    if config.input_mode == "h5":
        if config.auxiliary_h5_path is None or config.target_h5_path is None:
            raise ValueError("H5 mode requires auxiliary and target H5 paths")
        command.extend(
            [
                "agent",
                "--request",
                config.request,
                "--aux-h5",
                str(config.auxiliary_h5_path),
                "--target-h5",
                str(config.target_h5_path),
                "--patch-index",
                str(config.patch_index),
            ]
        )
    elif config.input_mode == "tiff":
        if None in (config.auxiliary_ms_path, config.auxiliary_hs_path, config.target_ms_path):
            raise ValueError("TIFF mode requires auxiliary MS/HS and target MS paths")
        command.extend(
            [
                "agent-tiff",
                "--request",
                config.request,
                "--aux-ms",
                str(config.auxiliary_ms_path),
                "--aux-hs",
                str(config.auxiliary_hs_path),
                "--target-ms",
                str(config.target_ms_path),
                "--crop-profile",
                config.crop_profile,
                "--experiment-mode",
                config.experiment_mode,
                "--patch-size",
                str(config.patch_size),
                "--patch-index",
                str(config.patch_index),
                "--row-offset",
                str(config.row_offset),
                "--col-offset",
                str(config.col_offset),
            ]
        )
        if config.crop_manifest_path is not None:
            command.extend(("--crop-manifest", str(config.crop_manifest_path)))
        if config.target_hs_reference_path is not None:
            command.extend(("--target-hs-reference", str(config.target_hs_reference_path)))
        if config.crop_profile == "custom":
            for option, value in (
                ("--ms-row-offset", config.ms_row_offset),
                ("--ms-col-offset", config.ms_col_offset),
                ("--window-height", config.window_height),
                ("--window-width", config.window_width),
                ("--hs-row-offset", config.hs_row_offset),
                ("--hs-col-offset", config.hs_col_offset),
            ):
                if value is not None:
                    command.extend((option, str(value)))
    else:
        raise ValueError(f"Unsupported UI input mode: {config.input_mode}")
    command.extend(
        [
            "--checkpoint",
            str(config.checkpoint_path),
            "--model-python",
            str(config.model_python),
            "--output-dir",
            str(config.output_dir),
            "--device",
            config.device,
            "--provider",
            config.provider,
            "--timeout",
            str(config.timeout_seconds),
            "--preflight-timeout",
            str(config.preflight_timeout_seconds),
            "--runtime-retries",
            str(config.runtime_retries),
            "--pretty",
        ]
    )
    if config.llm_model:
        command.extend(("--llm-model", config.llm_model))
    if config.base_url:
        command.extend(("--base-url", config.base_url))
    return command


def build_crop_command(config: UiRunConfig, *, python_executable: str | None = None) -> list[str]:
    """Build the local TIFF crop command used before a manifest-backed agent run."""

    if config.input_mode != "tiff":
        raise ValueError("Only TIFF mode can create a crop manifest")
    if None in (config.auxiliary_ms_path, config.auxiliary_hs_path, config.target_ms_path):
        raise ValueError("TIFF mode requires auxiliary MS/HS and target MS paths")
    command = [
        python_executable or sys.executable,
        "-m",
        "rsfusion_agent.cli",
        "crop-tiff-triplet",
        "--aux-ms",
        str(config.auxiliary_ms_path),
        "--aux-hs",
        str(config.auxiliary_hs_path),
        "--target-ms",
        str(config.target_ms_path),
        "--output-dir",
        str(config.output_dir / "prepared_crop"),
        "--profile",
        config.crop_profile,
        "--experiment-mode",
        config.experiment_mode,
        "--pretty",
    ]
    if config.experiment_mode == "simulation" and config.target_hs_reference_path is None:
        raise ValueError("模拟实验需要提供目标时相 HS 参考 TIFF")
    if config.target_hs_reference_path is not None:
        command.extend(("--target-hs-reference", str(config.target_hs_reference_path)))
    if config.crop_profile == "custom":
        required = (
            config.ms_row_offset,
            config.ms_col_offset,
            config.window_height,
            config.window_width,
        )
        if any(value is None for value in required):
            raise ValueError("Custom TIFF crop requires MS offset and window values")
        command.extend(
            [
                "--ms-row-offset",
                str(config.ms_row_offset),
                "--ms-col-offset",
                str(config.ms_col_offset),
                "--window-height",
                str(config.window_height),
                "--window-width",
                str(config.window_width),
            ]
        )
        if config.hs_row_offset is not None:
            command.extend(("--hs-row-offset", str(config.hs_row_offset)))
        if config.hs_col_offset is not None:
            command.extend(("--hs-col-offset", str(config.hs_col_offset)))
    return command


def inspect_raw_tiff_inputs(
    config: UiRunConfig, *, preview_max_dimension: int = 1024
) -> RawTiffInputCheck:
    """Inspect all raw TIFF inputs and render compact RGB previews before cropping."""

    if config.input_mode != "tiff":
        raise ValueError("Raw TIFF input checks are only available in TIFF mode")
    if None in (config.auxiliary_ms_path, config.auxiliary_hs_path, config.target_ms_path):
        raise ValueError("TIFF mode requires auxiliary MS/HS and target MS paths")
    if config.experiment_mode == "simulation" and config.target_hs_reference_path is None:
        raise ValueError("模拟实验需要提供目标时相 HS 参考 TIFF")

    from rsfusion_agent.tools.raster_preview import render_raster_rgb
    from rsfusion_agent.tools.tiff_triplet import inspect_tiff_triplet

    auxiliary_ms_path = config.auxiliary_ms_path.resolve()
    auxiliary_hs_path = config.auxiliary_hs_path.resolve()
    target_ms_path = config.target_ms_path.resolve()
    inspection = inspect_tiff_triplet(auxiliary_ms_path, auxiliary_hs_path, target_ms_path)
    target_hs_reference = None
    target_hs_reference_rgb = None
    if config.target_hs_reference_path is not None:
        from rsfusion_agent.tools.raster_inspector import inspect_raster

        target_hs_reference_path = config.target_hs_reference_path.resolve()
        target_hs_reference = inspect_raster(target_hs_reference_path).model_dump(mode="json")
        target_hs_reference_rgb = render_raster_rgb(
            target_hs_reference_path, bands=(28, 18, 9), max_dimension=preview_max_dimension
        )
    source_paths = (auxiliary_ms_path, auxiliary_hs_path, target_ms_path)
    if config.target_hs_reference_path is not None:
        source_paths = (*source_paths, config.target_hs_reference_path.resolve())
    return RawTiffInputCheck(
        source_paths=source_paths,
        auxiliary_ms=inspection.auxiliary_ms.model_dump(mode="json"),
        auxiliary_hs=inspection.auxiliary_hs.model_dump(mode="json"),
        target_ms=inspection.target_ms.model_dump(mode="json"),
        target_hs_reference=target_hs_reference,
        auxiliary_ms_rgb=render_raster_rgb(
            auxiliary_ms_path, bands=(2, 1, 0), max_dimension=preview_max_dimension
        ),
        auxiliary_hs_rgb=render_raster_rgb(
            auxiliary_hs_path, bands=(28, 18, 9), max_dimension=preview_max_dimension
        ),
        target_ms_rgb=render_raster_rgb(
            target_ms_path, bands=(2, 1, 0), max_dimension=preview_max_dimension
        ),
        target_hs_reference_rgb=target_hs_reference_rgb,
        is_ready_for_preprocessing=inspection.is_ready_for_preprocessing,
        blocking_issues=inspection.blocking_issues,
        warnings=inspection.warnings,
    )


def inspect_h5_inputs(config: UiRunConfig) -> H5InputCheck:
    """Inspect one selected H5 patch and build previews of the actual model tensors."""

    if config.input_mode != "h5":
        raise ValueError("H5 input checks are only available in H5 mode")
    if config.auxiliary_h5_path is None or config.target_h5_path is None:
        raise ValueError("H5 mode requires auxiliary and target H5 paths")

    from rsfusion_agent.tools.artifacts import hyperspectral_rgb
    from rsfusion_agent.tools.h5_patch import prepare_yre151_patch

    prepared = prepare_yre151_patch(
        config.auxiliary_h5_path,
        config.target_h5_path,
        split="test",
        patch_index=config.patch_index,
    )
    inspection = prepared.inspection
    return H5InputCheck(
        source_paths=(Path(inspection.auxiliary.path), Path(inspection.target.path)),
        inspection=inspection.model_dump(mode="json"),
        patch_index=config.patch_index,
        patch_count=inspection.auxiliary.patch_count,
        patch_height=inspection.auxiliary.patch_height,
        patch_width=inspection.auxiliary.patch_width,
        auxiliary_ms_rgb=hyperspectral_rgb(prepared.auxiliary_ms, bands=(2, 1, 0)),
        auxiliary_hs_rgb=hyperspectral_rgb(prepared.auxiliary_hs_interpolated),
        target_ms_rgb=hyperspectral_rgb(prepared.target_ms, bands=(2, 1, 0)),
        target_hs_gt_rgb=hyperspectral_rgb(prepared.target_hs_gt),
    )


def load_json_object(path: Path) -> dict[str, Any] | None:
    """Read a result artifact only when it is a valid JSON object."""

    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def list_run_history(output_root: Path, *, limit: int = 20) -> list[UiRunHistoryItem]:
    """List direct child run directories containing persisted agent results."""

    root = output_root.expanduser().resolve()
    if not root.is_dir() or limit < 1:
        return []
    items: list[UiRunHistoryItem] = []
    for result_path in root.glob("*/agent_result.json"):
        result = load_json_object(result_path)
        if result is None:
            continue
        metrics = ((result.get("fusion_result") or {}).get("metrics") or {})
        psnr = metrics.get("psnr")
        items.append(
            UiRunHistoryItem(
                output_dir=result_path.parent,
                status=str(result.get("status", "unknown")),
                created_at=datetime.fromtimestamp(result_path.stat().st_mtime),
                psnr=float(psnr) if isinstance(psnr, (int, float)) else None,
            )
        )
    return sorted(items, key=lambda item: item.created_at, reverse=True)[:limit]


def format_history_label(item: UiRunHistoryItem) -> str:
    metric = f" · PSNR {item.psnr:.3f}" if item.psnr is not None else ""
    return f"{item.created_at:%Y-%m-%d %H:%M:%S} · {item.status}{metric}"


def resolve_result_artifact(
    fusion_result: dict[str, Any], *, output_dir: Path, artifact_key: str
) -> Path | None:
    """Resolve a visual artifact from either agent-tool or full workflow output."""

    artifacts = fusion_result.get("artifacts") or {}
    candidate = artifacts.get(artifact_key)
    if not candidate:
        candidate = fusion_result.get(f"{artifact_key}_path")
    if not isinstance(candidate, str) or not candidate:
        return None

    output_root = output_dir.resolve()
    candidate_path = Path(candidate)
    path = (output_dir / candidate_path).resolve() if not candidate_path.is_absolute() else candidate_path.resolve()
    try:
        path.relative_to(output_root)
    except ValueError:
        return None
    return path if path.is_file() else None


def collect_agent_artifacts(result: dict[str, Any]) -> dict[str, str]:
    """Collect safe, output-directory-local artifact names from all completed tool calls."""

    artifacts: dict[str, str] = {}
    for trace_item in result.get("trace") or []:
        tool_artifacts = (trace_item.get("output") or {}).get("artifacts") or {}
        if isinstance(tool_artifacts, dict):
            artifacts.update(
                {key: value for key, value in tool_artifacts.items() if isinstance(value, str)}
            )
    fusion = result.get("fusion_result") or {}
    fusion_artifacts = fusion.get("artifacts") or {}
    if isinstance(fusion_artifacts, dict):
        artifacts.update(
            {key: value for key, value in fusion_artifacts.items() if isinstance(value, str)}
        )
    input_previews = fusion.get("input_preview_paths") or {}
    if isinstance(input_previews, dict):
        artifacts.update(
            {key: value for key, value in input_previews.items() if isinstance(value, str)}
        )
    return artifacts


def run_agent_from_ui(config: UiRunConfig) -> UiAgentExecution:
    """Execute the CLI so UI and terminal runs share exactly one workflow."""

    completed = subprocess.run(
        build_agent_command(config),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=config.timeout_seconds + config.preflight_timeout_seconds + 30,
        env=os.environ.copy(),
    )
    result_path = config.output_dir / "agent_result.json"
    return UiAgentExecution(
        return_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        result_path=result_path,
        result=load_json_object(result_path),
    )


def run_crop_from_ui(config: UiRunConfig) -> UiCropExecution:
    """Run only the explicit TIFF crop stage, without preflight, model or LLM costs."""

    if config.input_mode != "tiff":
        raise ValueError("仅原始 TIFF 模式支持显式裁剪")
    completed = subprocess.run(
        build_crop_command(config),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=config.timeout_seconds,
        env=os.environ.copy(),
    )
    manifest_path = config.output_dir / "prepared_crop" / "crop_manifest.json"
    return UiCropExecution(
        return_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        manifest_path=manifest_path,
        result=load_json_object(manifest_path),
    )


def _env_default(name: str, fallback: str = "") -> str:
    return os.environ.get(name, fallback)


def _build_config(st: Any) -> UiRunConfig:
    """Render configuration fields and return the current local-only run config."""

    if "ui_run_id" not in st.session_state:
        st.session_state["ui_run_id"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    with st.sidebar:
        st.header("运行配置")
        auxiliary = target = ""
        auxiliary_ms = auxiliary_hs = target_ms = target_hs_reference = ""
        experiment_mode = "real"
        crop_profile = "yre_legacy_test_v1"
        ms_row = ms_col = window_height = window_width = hs_row = hs_col = None
        raw_bounds: TiffCropSourceBounds | None = None
        cached_h5_check: H5InputCheck | None = None
        with st.expander("① 输入数据", expanded=True):
            input_mode = st.radio("输入模式", ("原始 TIFF 模式", "H5 演示模式"), horizontal=True)
            is_tiff = input_mode == "原始 TIFF 模式"
            if not is_tiff:
                auxiliary = st.text_input("辅助时相 H5", value=_env_default("RSFUSION_AUX_H5"))
                target = st.text_input("目标时相 H5", value=_env_default("RSFUSION_TARGET_H5"))
            else:
                experiment = st.radio(
                    "实验类型",
                    ("真实实验（无 T2 HS 真值）", "模拟实验（复现 Database.py）"),
                )
                experiment_mode = "simulation" if experiment.startswith("模拟") else "real"
                auxiliary_ms = st.text_input(
                    "T1 辅助时相 MS TIFF", value=_env_default("RSFUSION_AUX_MS_TIFF")
                )
                auxiliary_hs = st.text_input(
                    "T1 辅助时相 HS TIFF", value=_env_default("RSFUSION_AUX_HS_TIFF")
                )
                target_ms = st.text_input(
                    "T2 目标时相 MS TIFF", value=_env_default("RSFUSION_TARGET_MS_TIFF")
                )
                target_hs_reference = st.text_input(
                    "T2 目标时相 HS TIFF（模拟实验必填；真实实验可选伪标签）",
                    value=_env_default("RSFUSION_TARGET_HS_REFERENCE_TIFF"),
                )
                st.caption(
                    "模拟：T2 HS 原始裁剪作为低分辨率真值；真实：若提供则 3×插值后仅作伪标签。"
                )
        if is_tiff:
            cached_raw_check = st.session_state.get("raw_tiff_input_check")
            current_raw_check_key = (
                experiment_mode,
                auxiliary_ms,
                auxiliary_hs,
                target_ms,
                target_hs_reference,
            )
            if (
                isinstance(cached_raw_check, RawTiffInputCheck)
                and st.session_state.get("raw_tiff_input_check_key") == current_raw_check_key
            ):
                raw_bounds = crop_source_bounds(cached_raw_check)
            with st.expander("② 裁剪参数", expanded=True):
                crop_profile = st.selectbox(
                    "裁剪窗口方案",
                    ("yre_legacy_test_v1", "custom"),
                    format_func=lambda value: "YRE 原始测试窗口"
                    if value == "yre_legacy_test_v1"
                    else "自定义窗口",
                )
                if crop_profile == "yre_legacy_test_v1":
                    st.caption("MS：起始行 0、起始列 0、大小 540 × 540")
                    st.caption("HS（T1 与 T2）：起始行 0、起始列 0、大小 180 × 180")
                else:
                    st.markdown("**原始 TIFF 裁剪范围（降采样前）**")
                    if raw_bounds is None:
                        st.info("请先点击“检查原始 TIFF 输入”。检查完成后，这里的上限会按实际影像尺寸自动锁定。")
                    else:
                        st.success(
                            "已按输入检查结果限制可裁剪范围："
                            f"MS 最大 {raw_bounds.ms_width} × {raw_bounds.ms_height}，"
                            f"HS 最大 {raw_bounds.hs_width} × {raw_bounds.hs_height}。"
                        )
                    st.caption(
                        "这里填写源 TIFF 像素。模拟实验会先对该范围模糊/下采样，"
                        "真实实验则在该范围内把 HS 插值到 MS 网格。"
                    )
                    ms_row_limit = (
                        _largest_multiple_not_above(max(0, raw_bounds.ms_height - 3), 3)
                        if raw_bounds is not None
                        else None
                    )
                    ms_col_limit = (
                        _largest_multiple_not_above(max(0, raw_bounds.ms_width - 3), 3)
                        if raw_bounds is not None
                        else None
                    )
                    ms_left, ms_right = st.columns(2)
                    ms_row = int(
                        ms_left.number_input(
                            "MS 起始行",
                            min_value=0,
                            max_value=ms_row_limit,
                            value=0,
                            step=3,
                        )
                    )
                    ms_col = int(
                        ms_right.number_input(
                            "MS 起始列",
                            min_value=0,
                            max_value=ms_col_limit,
                            value=0,
                            step=3,
                        )
                    )
                    auto_hs_offsets = st.checkbox(
                        "自动按 1/3 映射 HS 起始位置",
                        value=True,
                        help="取消后可手动指定 HS 起点；仍需自行保证与 MS 对应。",
                    )
                    if auto_hs_offsets:
                        hs_row, hs_col = ms_row // 3, ms_col // 3
                        st.caption(f"当前 HS 起始位置：行 {hs_row}、列 {hs_col}（由 MS 起点自动换算）")
                    else:
                        hs_left, hs_right = st.columns(2)
                        hs_row = int(
                            hs_left.number_input(
                                "HS 起始行",
                                min_value=0,
                                max_value=max(0, raw_bounds.hs_height - 1)
                                if raw_bounds is not None
                                else None,
                                value=0,
                            )
                        )
                        hs_col = int(
                            hs_right.number_input(
                                "HS 起始列",
                                min_value=0,
                                max_value=max(0, raw_bounds.hs_width - 1)
                                if raw_bounds is not None
                                else None,
                                value=0,
                            )
                        )

                    max_height = None
                    max_width = None
                    if raw_bounds is not None:
                        max_height = min(
                            raw_bounds.ms_height - ms_row,
                            (raw_bounds.hs_height - hs_row) * 3,
                        )
                        max_width = min(
                            raw_bounds.ms_width - ms_col,
                            (raw_bounds.hs_width - hs_col) * 3,
                        )
                        max_height = _largest_multiple_not_above(max(0, max_height), 3)
                        max_width = _largest_multiple_not_above(max(0, max_width), 3)
                    if max_height is not None and (max_height < 3 or max_width is None or max_width < 3):
                        st.error("当前起始位置无法容纳最小的 3 × 3 MS 源窗口，请调整起始位置。")
                    safe_max_height = max(3, max_height) if max_height is not None else None
                    safe_max_width = max(3, max_width) if max_width is not None else None
                    default_height = min(540, safe_max_height) if safe_max_height is not None else 540
                    default_width = min(540, safe_max_width) if safe_max_width is not None else 540
                    window_height = int(
                        ms_left.number_input(
                            "MS 源裁剪高度（降采样前）",
                            min_value=3,
                            max_value=safe_max_height,
                            value=default_height,
                            step=3,
                        )
                    )
                    window_width = int(
                        ms_right.number_input(
                            "MS 源裁剪宽度（降采样前）",
                            min_value=3,
                            max_value=safe_max_width,
                            value=default_width,
                            step=3,
                        )
                    )
                    st.caption(
                        f"对应 HS 源裁剪尺寸：{window_width // 3} × {window_height // 3}；"
                        "HS 裁剪尺寸由 MS 自动换算。"
                    )
        else:
            cached = st.session_state.get("h5_input_check")
            if (
                isinstance(cached, H5InputCheck)
                and st.session_state.get("h5_input_check_key")
                == (auxiliary, target)
            ):
                cached_h5_check = cached
        with st.expander("③ 模型与执行", expanded=True):
            checkpoint = st.text_input("Checkpoint", value=_env_default("RSFUSION_CHECKPOINT"))
            model_python = st.text_input(
                "模型 Conda Python",
                value=_env_default("RSFUSION_MODEL_PYTHON", sys.executable),
            )
            provider = st.selectbox("LLM Provider", ("qwen", "openai", "deepseek", "custom"))
            llm_model = st.text_input("模型 ID", value=_env_default("RSFUSION_LLM_MODEL"))
            base_url = st.text_input("兼容 API Base URL", value=_env_default("RSFUSION_LLM_BASE_URL"))
            device = st.selectbox("推理设备", ("cuda", "auto", "cpu"))
            if is_tiff:
                from rsfusion_agent.tools.tiff_experiment import (
                    default_patch_size,
                    output_grid_shape,
                )

                crop_height = window_height or 540
                crop_width = window_width or 540
                output_height, output_width = output_grid_shape(
                    ms_height=crop_height,
                    ms_width=crop_width,
                    experiment_mode=experiment_mode,
                )
                maximum_patch_size = _largest_multiple_not_above(
                    min(output_height, output_width), 3
                )
                default_size = min(default_patch_size(experiment_mode), maximum_patch_size)
                if maximum_patch_size < 3:
                    st.error("当前原始裁剪范围在预处理后不足以生成最小 3 × 3 模型 Patch。")
                    patch_size = 3
                else:
                    patch_widget_key = f"ui_tiff_patch_size_{experiment_mode}"
                    stored_patch_size = int(st.session_state.get(patch_widget_key, default_size))
                    if stored_patch_size < 3 or stored_patch_size > maximum_patch_size:
                        st.session_state[patch_widget_key] = default_size
                    st.markdown("**模型 Patch 参数（预处理后的模型网格）**")
                    patch_size = int(
                        st.number_input(
                            "模型 Patch 边长",
                            min_value=3,
                            max_value=maximum_patch_size,
                            value=default_size,
                            step=3,
                            key=patch_widget_key,
                            help=(
                                "模拟实验：该尺寸位于 MS 降采样后的网格；"
                                "真实实验：该尺寸位于 HS 已插值到 MS 的网格。"
                            ),
                        )
                    )
                grid_rows, grid_columns = output_height // patch_size, output_width // patch_size
                total_patches = grid_rows * grid_columns
                if total_patches < 1:
                    st.error(
                        f"当前裁剪输出为 {output_width} × {output_height}，小于该实验的 "
                        f"{patch_size} × {patch_size} 测试块。"
                    )
                    patch_index = 0
                else:
                    patch_index_key = f"ui_tiff_patch_index_{experiment_mode}"
                    stored_patch_index = int(st.session_state.get(patch_index_key, 0))
                    if stored_patch_index < 0 or stored_patch_index >= total_patches:
                        st.session_state[patch_index_key] = 0
                    patch_index = int(
                        st.number_input(
                            "TIFF 测试 Patch 编号",
                            min_value=0,
                            max_value=total_patches - 1,
                            value=0,
                            step=1,
                            key=patch_index_key,
                        )
                    )
                    row_offset = (patch_index // grid_columns) * patch_size
                    col_offset = (patch_index % grid_columns) * patch_size
                    st.caption(
                        f"共 {total_patches} 块（{grid_rows} × {grid_columns}）；当前块位于 "
                        f"输出网格 row={row_offset}, col={col_offset}。当前 V1 按无重叠网格切块。"
                    )
                row_offset = (patch_index // grid_columns) * patch_size if total_patches else 0
                col_offset = (patch_index % grid_columns) * patch_size if total_patches else 0
            else:
                if cached_h5_check is None:
                    st.info("请先点击“检查 H5 输入”，系统会读取 test 集 Patch 总数并限制编号范围。")
                    patch_index = int(
                        st.number_input("H5 测试 Patch 编号", min_value=0, value=0, step=1)
                    )
                    patch_size = 180
                else:
                    h5_patch_key = "ui_h5_patch_index"
                    stored_h5_patch = int(st.session_state.get(h5_patch_key, 0))
                    if stored_h5_patch < 0 or stored_h5_patch >= cached_h5_check.patch_count:
                        st.session_state[h5_patch_key] = 0
                    st.markdown("**H5 测试 Patch 参数**")
                    patch_index = int(
                        st.number_input(
                            "H5 测试 Patch 编号",
                            min_value=0,
                            max_value=cached_h5_check.patch_count - 1,
                            value=0,
                            step=1,
                            key=h5_patch_key,
                        )
                    )
                    patch_size = cached_h5_check.patch_height
                    st.caption(
                        f"test 集共 {cached_h5_check.patch_count} 个 Patch；每个为 "
                        f"{cached_h5_check.patch_width} × {cached_h5_check.patch_height}。"
                        "修改编号后点击“检查 H5 输入”可查看该块的模型输入 RGB。"
                    )
                row_offset = col_offset = 0
            timeout_seconds = st.number_input("融合超时（秒）", min_value=30, value=600, step=30)
            preflight_timeout = st.number_input("预检超时（秒）", min_value=10, value=60, step=10)
            runtime_retries = st.selectbox("原生崩溃额外重试次数", (0, 1, 2), index=1)
        with st.expander("④ 输出目录", expanded=True):
            output_root = st.text_input("输出根目录", value="outputs/ui_runs")
            st.caption(f"当前运行目录：ui_run_{st.session_state['ui_run_id']}")

    default_request = (
        "请先检查数据，再融合第0个patch，并汇报PSNR、SAM、SSIM和输出文件。"
        if not is_tiff
        else (
            "请检查裁剪清单，融合当前 TIFF patch，并汇报输出文件、运行时间和指标。"
            if experiment_mode == "simulation"
            else "请检查裁剪清单，融合当前 TIFF patch，并汇报输出文件、运行时间和指标可用性。"
        )
    )
    with st.form("natural_language_task_form", clear_on_submit=False):
        request = st.text_area(
            "自然语言任务",
            value=default_request,
            height=110,
            help="填写任务后点击提交，或在文本框中按 Ctrl+Enter 提交。",
        )
        submit_task = st.form_submit_button("提交自然语言任务（Ctrl+Enter）")
    if submit_task:
        st.session_state["ui_submit_agent"] = True
    return UiRunConfig(
        request=request,
        checkpoint_path=Path(checkpoint),
        model_python=Path(model_python),
        output_dir=Path(output_root) / f"ui_run_{st.session_state['ui_run_id']}",
        input_mode="tiff" if is_tiff else "h5",
        auxiliary_h5_path=Path(auxiliary) if auxiliary else None,
        target_h5_path=Path(target) if target else None,
        auxiliary_ms_path=Path(auxiliary_ms) if auxiliary_ms else None,
        auxiliary_hs_path=Path(auxiliary_hs) if auxiliary_hs else None,
        target_ms_path=Path(target_ms) if target_ms else None,
        target_hs_reference_path=Path(target_hs_reference) if target_hs_reference else None,
        experiment_mode=experiment_mode,
        crop_profile=crop_profile,
        ms_row_offset=ms_row,
        ms_col_offset=ms_col,
        window_height=window_height,
        window_width=window_width,
        hs_row_offset=hs_row,
        hs_col_offset=hs_col,
        provider=provider,
        llm_model=llm_model or None,
        base_url=base_url or None,
        patch_index=int(patch_index),
        patch_size=int(patch_size),
        row_offset=int(row_offset),
        col_offset=int(col_offset),
        device=device,
        timeout_seconds=int(timeout_seconds),
        preflight_timeout_seconds=int(preflight_timeout),
        runtime_retries=int(runtime_retries),
    )


def _render_preflight(st: Any, payload: dict[str, Any] | None) -> None:
    if not payload:
        return
    st.subheader("本地环境预检")
    columns = st.columns(4)
    columns[0].metric("状态", payload.get("status", "unknown"))
    columns[1].metric("设备", payload.get("resolved_device", "unknown"))
    columns[2].metric("GPU", payload.get("cuda_device_name") or "未使用")
    columns[3].metric("Checkpoint", f"epoch {payload.get('checkpoint_epoch', 'unknown')}")
    with st.expander("查看预检 JSON"):
        st.json(payload)


def _render_raw_tiff_input_check(st: Any, result: RawTiffInputCheck | None) -> None:
    """Render the no-cost raw input stage that precedes crop-manifest creation."""

    if result is None:
        return
    st.subheader("原始 TIFF 输入检查")
    if result.is_ready_for_preprocessing:
        st.success("三景 TIFF 的元数据满足严格预处理契约。")
    else:
        st.warning(
            "检测到源 TIFF 的空间网格或波段存在差异。当前 YRE 数据可继续按明确的"
            "源像素裁剪窗口生成 crop manifest；系统不会自动配准、重投影或重采样。"
        )
        for issue in result.blocking_issues:
            st.caption(f"检查项：{issue}")
    for warning in result.warnings:
        st.caption(f"提示：{warning}")

    cards: list[tuple[str, dict[str, Any], Any, str]] = [
        ("辅助时相 MS", result.auxiliary_ms, result.auxiliary_ms_rgb, "RGB：波段 3 / 2 / 1"),
        ("辅助时相 HS", result.auxiliary_hs, result.auxiliary_hs_rgb, "RGB：波段 29 / 19 / 10"),
        ("目标时相 MS", result.target_ms, result.target_ms_rgb, "RGB：波段 3 / 2 / 1"),
    ]
    if result.target_hs_reference is not None and result.target_hs_reference_rgb is not None:
        cards.append(
            (
                "目标时相 HS 参考",
                result.target_hs_reference,
                result.target_hs_reference_rgb,
                "RGB：波段 29 / 19 / 10",
            )
        )
    columns = st.columns(len(cards))
    for column, (title, metadata, preview, band_label) in zip(columns, cards, strict=True):
        column.markdown(f"#### {title}")
        column.metric("原始尺寸", f"{metadata['width']} × {metadata['height']}")
        column.caption(
            f"{metadata['band_count']} bands · {metadata['dtypes'][0]} · "
            f"分辨率 {metadata['resolution'][0]:g} × {metadata['resolution'][1]:g}"
        )
        column.image(preview, caption=band_label, use_container_width=True)
        with column.expander("查看完整元数据"):
            column.json(metadata)


def _render_h5_input_check(st: Any, result: H5InputCheck | None) -> None:
    """Render a selected H5 patch exactly as it will enter the frozen YRE model."""

    if result is None:
        return
    st.subheader("H5 输入检查")
    inspection = result.inspection
    warnings = inspection.get("warnings", [])
    st.success(
        f"YRE-151 H5 数据通过检查：test 集共 {result.patch_count} 个 Patch，"
        f"当前为 Patch {result.patch_index}。"
    )
    for warning in warnings:
        st.caption(f"提示：{warning}")

    cards = (
        ("T1 辅助 MS（模型输入）", result.auxiliary_ms_rgb, "RGB：波段 3 / 2 / 1"),
        ("T1 辅助 HS（模型输入）", result.auxiliary_hs_rgb, "RGB：波段 29 / 19 / 10"),
        ("T2 目标 MS（模型输入）", result.target_ms_rgb, "RGB：波段 3 / 2 / 1"),
        ("T2 目标 HS 真值", result.target_hs_gt_rgb, "RGB：波段 29 / 19 / 10"),
    )
    columns = st.columns(len(cards))
    for column, (title, preview, band_label) in zip(columns, cards, strict=True):
        column.markdown(f"#### {title}")
        column.metric("Patch 尺寸", f"{result.patch_width} × {result.patch_height}")
        column.image(preview, caption=band_label, use_container_width=True)
    with st.expander("查看 H5 元数据"):
        st.json(inspection)


def _load_crop_preview_cards(crop_result: dict[str, Any]) -> list[tuple[str, dict[str, Any], Any, str]]:
    """Read compact RGB previews for the TIFF crops declared by one manifest."""

    from rsfusion_agent.tools.raster_inspector import inspect_raster
    from rsfusion_agent.tools.raster_preview import render_raster_rgb

    sources = (
        ("T1 MS 裁剪结果", "auxiliary_ms_path", (2, 1, 0), "RGB：波段 3 / 2 / 1"),
        ("T1 HS 裁剪结果", "auxiliary_hs_path", (28, 18, 9), "RGB：波段 29 / 19 / 10"),
        ("T2 MS 裁剪结果", "target_ms_path", (2, 1, 0), "RGB：波段 3 / 2 / 1"),
        (
            "T2 HS 参考裁剪结果",
            "target_hs_reference_path",
            (28, 18, 9),
            "RGB：波段 29 / 19 / 10",
        ),
    )
    cards: list[tuple[str, dict[str, Any], Any, str]] = []
    for title, path_key, bands, band_label in sources:
        raw_path = crop_result.get(path_key)
        if not isinstance(raw_path, str) or not raw_path:
            continue
        path = Path(raw_path)
        if not path.is_file():
            continue
        cards.append(
            (
                title,
                inspect_raster(path).model_dump(mode="json"),
                render_raster_rgb(path, bands=bands, max_dimension=512),
                band_label,
            )
        )
    return cards


def _load_selected_tiff_patch_cards(
    manifest_path: Path, config: UiRunConfig
) -> tuple[Any, list[tuple[str, Any, str]]]:
    """Build previews from the exact mode-specific tensors that inference will receive."""

    from rsfusion_agent.tools.artifacts import hyperspectral_rgb
    from rsfusion_agent.tools.tiff_experiment import prepare_experiment_patch_from_manifest

    prepared = prepare_experiment_patch_from_manifest(
        manifest_path,
        experiment_mode=config.experiment_mode,
        patch_size=config.patch_size,
        patch_index=config.patch_index,
    )
    cards: list[tuple[str, Any, str]] = [
        ("T1 MS 输入块", hyperspectral_rgb(prepared.auxiliary_ms, bands=(2, 1, 0)), "RGB：3 / 2 / 1"),
        (
            "T1 HS 输入块",
            hyperspectral_rgb(prepared.auxiliary_hs_interpolated, bands=(28, 18, 9)),
            "RGB：29 / 19 / 10",
        ),
        ("T2 MS 输入块", hyperspectral_rgb(prepared.target_ms, bands=(2, 1, 0)), "RGB：3 / 2 / 1"),
    ]
    if prepared.target_hs_reference is not None:
        label = "T2 HS 模拟真值块" if config.experiment_mode == "simulation" else "T2 HS 伪标签块"
        cards.append(
            (
                label,
                hyperspectral_rgb(prepared.target_hs_reference, bands=(28, 18, 9)),
                "RGB：29 / 19 / 10",
            )
        )
    return prepared, cards


def _render_crop_result(st: Any, execution: UiCropExecution | None, config: UiRunConfig) -> None:
    """Show only the deterministic preprocessing result, separate from fusion output."""

    if execution is None:
        return
    st.subheader("裁剪结果")
    if execution.return_code != 0 or execution.result is None:
        st.error("裁剪失败。")
        with st.expander("裁剪命令输出"):
            st.code(execution.stderr or execution.stdout or "No output")
        return
    st.success("裁剪完成，已生成可追溯的 crop_manifest.json。")
    windows = {
        "T1 MS": execution.result.get("auxiliary_ms_window"),
        "T1 HS": execution.result.get("auxiliary_hs_window"),
        "T2 MS": execution.result.get("target_ms_window"),
        "T2 HS 参考": execution.result.get("target_hs_reference_window"),
    }
    st.dataframe(
        [
            {"数据": name, **window}
            for name, window in windows.items()
            if isinstance(window, dict)
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"清单：{execution.manifest_path}")
    try:
        prepared, cards = _load_selected_tiff_patch_cards(execution.manifest_path, config)
    except (FileNotFoundError, ValueError) as exc:
        st.warning(f"可融合 patch 预览生成失败：{exc}")
        return
    if cards:
        selection = prepared.selection
        st.subheader("裁剪后可融合 Patch 可视化")
        st.caption(
            f"共 {selection.total_patch_count} 个测试块（{selection.grid_rows} × "
            f"{selection.grid_columns}）；当前为 Patch {selection.patch_index}，"
            f"row={selection.row_offset}, col={selection.col_offset}，"
            f"大小 {selection.patch_size} × {selection.patch_size}。"
        )
        columns = st.columns(len(cards))
        for column, (title, preview, band_label) in zip(columns, cards, strict=True):
            column.markdown(f"#### {title}")
            column.metric("输入尺寸", f"{selection.patch_size} × {selection.patch_size}")
            column.image(preview, caption=band_label, use_container_width=True)


def _render_result(
    st: Any, execution: UiAgentExecution, output_dir: Path, config: UiRunConfig
) -> None:
    result = execution.result
    if result is None:
        st.error("Agent 未生成可解析的 agent_result.json。")
        with st.expander("CLI 输出"):
            st.code(execution.stderr or execution.stdout or "No output")
        return

    status = result.get("status", "unknown")
    if status == "completed":
        st.success("Agent 已完成当前自然语言任务。")
    else:
        st.warning(f"Agent 已结束，状态：{status}")
        for item in reversed(result.get("trace") or []):
            diagnosis = (item.get("output") or {}).get("diagnosis")
            if diagnosis:
                st.error(diagnosis.get("summary", "Agent 工具执行失败。"))
                st.caption(f"建议：{diagnosis.get('recommended_action', '')}")
                break

    called_tools = {str(item.get("name")) for item in result.get("trace") or []}
    if "get_runtime_preflight" in called_tools or any(
        name.startswith("run_yre151") for name in called_tools
    ):
        _render_preflight(st, result.get("runtime_preflight"))
    fusion = result.get("fusion_result") or {}
    visual_result = {"artifacts": collect_agent_artifacts(result)}
    raw_specs = [
        ("T1 辅助时相 MS 原始 RGB", "raw_auxiliary_ms_rgb"),
        ("T1 辅助时相 HS 原始 RGB", "raw_auxiliary_hs_rgb"),
        ("T2 目标时相 MS 原始 RGB", "raw_target_ms_rgb"),
        ("T2 目标时相 HS 原始 RGB", "raw_target_hs_reference_rgb"),
    ]
    raw_specs = [
        item
        for item in raw_specs
        if resolve_result_artifact(visual_result, output_dir=output_dir, artifact_key=item[1])
    ]
    if raw_specs:
        st.subheader("Agent 原始 TIFF 输入检查与 RGB 可视化")
        raw_columns = st.columns(len(raw_specs))
        for column, (title, artifact_key) in zip(raw_columns, raw_specs, strict=True):
            path = resolve_result_artifact(visual_result, output_dir=output_dir, artifact_key=artifact_key)
            if path:
                column.image(str(path), caption=title, use_container_width=True)
    crop_manifest_path = resolve_result_artifact(
        visual_result, output_dir=output_dir, artifact_key="crop_manifest"
    )
    if crop_manifest_path is not None:
        try:
            prepared_crop, crop_cards = _load_selected_tiff_patch_cards(crop_manifest_path, config)
        except (FileNotFoundError, ValueError) as exc:
            st.warning(f"Agent 裁剪 Patch 预览生成失败：{exc}")
        else:
            selection = prepared_crop.selection
            st.subheader("Agent 裁剪后的可融合 Patch")
            st.caption(
                f"共 {selection.total_patch_count} 个测试块；当前为 Patch "
                f"{selection.patch_index}，row={selection.row_offset}, "
                f"col={selection.col_offset}。"
            )
            crop_columns = st.columns(len(crop_cards))
            for column, (title, preview, band_label) in zip(
                crop_columns, crop_cards, strict=True
            ):
                column.image(preview, caption=f"{title} · {band_label}", use_container_width=True)
    input_specs = [
        ("T1 MS 输入测试块", "input_auxiliary_ms_preview"),
        ("T1 HS 输入测试块", "input_auxiliary_hs_preview"),
        ("T2 MS 输入测试块", "input_target_ms_preview"),
    ]
    if resolve_result_artifact(
        visual_result, output_dir=output_dir, artifact_key="input_target_hs_reference_preview"
    ):
        reference_name = (
            "T2 HS 模拟真值输入块"
            if fusion.get("metrics_status") == "available_reduced_resolution_ground_truth"
            else "T2 HS 伪标签输入块"
        )
        input_specs.append((reference_name, "input_target_hs_reference_preview"))
    available_input_specs = [
        item
        for item in input_specs
        if resolve_result_artifact(visual_result, output_dir=output_dir, artifact_key=item[1])
    ]
    if available_input_specs:
        st.subheader("输入测试块可视化")
        input_columns = st.columns(len(available_input_specs))
        for column, (title, artifact_key) in zip(input_columns, available_input_specs, strict=True):
            path = resolve_result_artifact(visual_result, output_dir=output_dir, artifact_key=artifact_key)
            if path:
                column.image(str(path), caption=title, use_container_width=True)
    metrics = fusion.get("metrics") or {}
    if metrics:
        if fusion.get("metrics_status") == "available_interpolated_target_hs_pseudo_reference":
            st.warning("以下指标基于真实实验的插值伪标签，不等同于原生高分辨率真值。")
        st.subheader("融合指标")
        columns = st.columns(6)
        columns[0].metric("PSNR", f"{metrics.get('psnr', 0):.4f}")
        columns[1].metric("SAM", f"{metrics.get('sam', 0):.4f}")
        columns[2].metric("SSIM", f"{metrics.get('ssim', 0):.4f}")
        columns[3].metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
        columns[4].metric("ERGAS", f"{metrics.get('ergas', 0):.4f}")
        columns[5].metric("CC", f"{metrics.get('cc', 0):.4f}")
    elif fusion.get("metrics_status"):
        st.info("TIFF 模式：" + str(fusion["metrics_status"]))

    runtime = fusion.get("runtime") or {}
    if runtime.get("attempt_count", 1) > 1:
        st.info(
            "模型进程已在第 "
            f"{runtime['attempt_count']} 次尝试成功；重试退出码：{runtime.get('retried_exit_codes', [])}"
        )

    st.subheader("Agent 调用记录")
    trace = result.get("trace") or []
    if trace:
        st.dataframe(
            [
                {
                    "轮次": item.get("round_index"),
                    "工具": item.get("name"),
                    "状态": item.get("status"),
                    "耗时（秒）": round(float(item.get("elapsed_seconds", 0)), 3),
                }
                for item in trace
            ],
            use_container_width=True,
            hide_index=True,
        )

    estimated_cost = result.get("estimated_cost") or {}
    if estimated_cost:
        st.caption(
            f"本次 LLM 估算成本：{estimated_cost.get('estimated_cost')} "
            f"{estimated_cost.get('currency', '')}（以服务商账单为准）"
        )

    image_specs = [("融合结果 RGB", "rgb_preview")]
    if resolve_result_artifact(
        visual_result, output_dir=output_dir, artifact_key="reference_rgb_preview"
    ):
        reference_title = (
            "插值伪标签 RGB（真实实验，同尺度）"
            if fusion.get("metrics_status")
            == "available_interpolated_target_hs_pseudo_reference"
            else "标签 RGB（模拟实验 / H5 真值，同尺度）"
        )
        image_specs.append((reference_title, "reference_rgb_preview"))
    if resolve_result_artifact(visual_result, output_dir=output_dir, artifact_key="sam_heatmap"):
        image_specs.append(("SAM 热力图", "sam_heatmap"))
    image_columns = st.columns(len(image_specs))
    for column, (title, artifact_key) in zip(image_columns, image_specs, strict=True):
        path = resolve_result_artifact(visual_result, output_dir=output_dir, artifact_key=artifact_key)
        if path:
            column.image(str(path), caption=title, use_container_width=True)

    st.subheader("本地结果文件")
    for name in (
        "report.md",
        "preflight.json",
        "agent_result.json",
        "metrics.json",
        "run_manifest.json",
        "predicted_hs.tif",
        "rgb_preview.png",
        "reference_rgb_preview.png",
        "sam_heatmap.png",
        "input_auxiliary_ms_preview.png",
        "input_auxiliary_hs_preview.png",
        "input_target_ms_preview.png",
        "input_target_hs_reference_preview.png",
        "raw_auxiliary_ms_rgb.png",
        "raw_auxiliary_hs_rgb.png",
        "raw_target_ms_rgb.png",
        "raw_target_hs_reference_rgb.png",
    ):
        path = output_dir / name
        if path.is_file():
            st.download_button(
                label=f"下载 {name}",
                data=path.read_bytes(),
                file_name=name,
                mime=(
                    "application/json"
                    if name.endswith(".json")
                    else "image/tiff"
                    if name.endswith(".tif")
                    else "image/png"
                    if name.endswith(".png")
                    else "text/markdown"
                ),
            )
    with st.expander("查看 agent_result.json"):
        st.json(result)


def _store_stage_record(
    st: Any,
    *,
    stage: str,
    title: str,
    payload: Any,
    output_dir: Path | None = None,
) -> None:
    """Keep the latest result for each stage in the current configured run."""

    records = dict(st.session_state.get("ui_stage_records", {}))
    records[stage] = UiStageRecord(
        stage=stage,
        title=title,
        completed_at=datetime.now(),
        payload=payload,
        output_dir=output_dir,
    )
    st.session_state["ui_stage_records"] = records


def sorted_stage_records(session_state: Any) -> list[UiStageRecord]:
    """Return retained stage results in newest-first completion order."""

    records = list(dict(session_state.get("ui_stage_records", {})).values())
    return sorted(records, key=lambda item: item.completed_at, reverse=True)


def _render_stage_records(st: Any, config: UiRunConfig) -> Path | None:
    """Render current-session stage results newest first and preserve older stages."""

    records = sorted_stage_records(st.session_state)
    st.subheader("本次配置执行记录")
    if not records:
        st.caption("尚未执行输入检查、环境预检、裁剪或融合。")
        return None

    current_output_dir: Path | None = None
    for index, record in enumerate(records):
        label = f"{record.completed_at:%H:%M:%S} · {record.title}"
        with st.expander(label, expanded=index == 0):
            if record.stage == "agent":
                _render_result(st, record.payload, record.output_dir or Path("."), config)
                current_output_dir = record.output_dir
            elif record.stage == "crop":
                _render_crop_result(st, record.payload, config)
            elif record.stage == "input_check":
                _render_raw_tiff_input_check(st, record.payload)
            elif record.stage == "h5_input_check":
                _render_h5_input_check(st, record.payload)
            elif record.stage == "preflight":
                _render_preflight(st, record.payload)
    return current_output_dir


def _render_history_selector(
    st: Any,
    output_root: Path,
    *,
    current_output_dir: Path | None = None,
) -> None:
    """Render previous runs after the current result, newest first."""

    history = list_run_history(output_root)
    if current_output_dir is not None:
        current_resolved = current_output_dir.resolve()
        history = [item for item in history if item.output_dir.resolve() != current_resolved]
    st.divider()
    st.subheader("历史记录")
    if not history:
        st.caption("暂无其他已完成的 agent_result.json。")
        return
    selected = st.selectbox("选择历史运行", history, format_func=format_history_label)
    if st.button("加载历史结果", use_container_width=True):
        result_path = selected.output_dir / "agent_result.json"
        result = load_json_object(result_path)
        if result is not None:
            st.session_state["ui_execution"] = UiAgentExecution(
                return_code=0,
                stdout="",
                stderr="",
                result_path=result_path,
                result=result,
            )
            st.session_state["ui_output_dir"] = selected.output_dir
            _store_stage_record(
                st,
                stage="agent",
                title="加载历史融合结果",
                payload=st.session_state["ui_execution"],
                output_dir=selected.output_dir,
            )


def _task_wait_reporter(st: Any, task_name: str, timeout_seconds: int) -> tuple[Any, WaitReporter]:
    """Create one honest per-task progress indicator based on elapsed timeout budget."""

    progress = st.progress(0, text=f"{task_name}：准备开始")

    def report(elapsed_seconds: float, _: float) -> None:
        percentage = min(95, max(1, int(elapsed_seconds / max(timeout_seconds, 1) * 100)))
        progress.progress(
            percentage,
            text=(
                f"{task_name}：已等待 {elapsed_seconds:.1f} 秒"
                f"（超时上限 {timeout_seconds} 秒）"
            ),
        )

    return progress, report


def run_app() -> None:
    """Render the Streamlit page without importing Streamlit during unit tests."""

    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit('Install the UI dependency first: pip install -e ".[ui]"') from exc

    st.set_page_config(page_title="RSFusionAgent", page_icon="🛰️", layout="wide")
    st.title("RSFusionAgent · 遥感图像融合 Agent")
    st.caption("本地运行：H5 数据与模型权重不会上传；API Key 仅从终端环境变量读取。")
    config = _build_config(st)

    previous_crop = st.session_state.get("ui_crop_execution")
    current_crop_key = crop_reuse_key(config)
    reusing_crop = (
        config.input_mode == "tiff"
        and isinstance(previous_crop, UiCropExecution)
        and previous_crop.return_code == 0
        and previous_crop.manifest_path.is_file()
        and st.session_state.get("ui_crop_reuse_key") == current_crop_key
    )
    if reusing_crop:
        config = replace(config, crop_manifest_path=previous_crop.manifest_path)

    st.subheader("功能操作")
    if reusing_crop:
        st.caption(f"已复用本次配置的裁剪清单，不会重复裁剪：{previous_crop.manifest_path}")
    if config.input_mode == "tiff":
        check_col, preflight_col, crop_col, run_col = st.columns(4)
        if check_col.button("检查原始 TIFF 输入", use_container_width=True):
            progress, on_wait = _task_wait_reporter(st, "原始 TIFF 输入检查", 60)
            try:
                raw_input_check = run_with_elapsed(
                    lambda: inspect_raw_tiff_inputs(config),
                    timeout_seconds=60,
                    on_wait=on_wait,
                )
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                st.error(str(exc))
            else:
                st.session_state["raw_tiff_input_check"] = raw_input_check
                st.session_state["raw_tiff_input_check_key"] = raw_tiff_check_key(config)
                progress.progress(100, text="原始 TIFF 输入检查完成")
                _store_stage_record(
                    st,
                    stage="input_check",
                    title="原始 TIFF 输入检查",
                    payload=raw_input_check,
                )
        if crop_col.button("执行裁剪", use_container_width=True):
            progress, on_wait = _task_wait_reporter(st, "裁剪", config.timeout_seconds)
            try:
                crop_execution = run_with_elapsed(
                    lambda: run_crop_from_ui(replace(config, crop_manifest_path=None)),
                    timeout_seconds=config.timeout_seconds,
                    on_wait=on_wait,
                )
            except subprocess.TimeoutExpired:
                st.error("裁剪超时。请检查输入文件和裁剪窗口。")
            else:
                st.session_state["ui_crop_execution"] = crop_execution
                if crop_execution.return_code == 0:
                    st.session_state["ui_crop_reuse_key"] = current_crop_key
                    progress.progress(100, text="裁剪完成，可执行 Agent 融合")
                else:
                    st.session_state.pop("ui_crop_reuse_key", None)
                _store_stage_record(
                    st,
                    stage="crop",
                    title="TIFF 裁剪",
                    payload=crop_execution,
                )
    else:
        check_col, preflight_col, run_col = st.columns(3)
        if check_col.button("检查 H5 输入", use_container_width=True):
            progress, on_wait = _task_wait_reporter(st, "H5 输入检查", 60)
            try:
                h5_input_check = run_with_elapsed(
                    lambda: inspect_h5_inputs(config),
                    timeout_seconds=60,
                    on_wait=on_wait,
                )
            except (FileNotFoundError, RuntimeError, ValueError, IndexError) as exc:
                st.error(str(exc))
            else:
                st.session_state["h5_input_check"] = h5_input_check
                st.session_state["h5_input_check_key"] = h5_input_check_key(config)
                progress.progress(100, text="H5 输入检查完成")
                _store_stage_record(
                    st,
                    stage="h5_input_check",
                    title=f"H5 输入检查 · Patch {config.patch_index}",
                    payload=h5_input_check,
                )
    if preflight_col.button("预检模型环境", use_container_width=True):
        from rsfusion_agent.tools.model_runtime import preflight_yre151_runtime

        progress, on_wait = _task_wait_reporter(st, "模型环境预检", config.preflight_timeout_seconds)
        try:
            result = run_with_elapsed(
                lambda: preflight_yre151_runtime(
                    checkpoint_path=config.checkpoint_path,
                    model_python=config.model_python,
                    device=config.device,
                    timeout_seconds=config.preflight_timeout_seconds,
                ),
                timeout_seconds=config.preflight_timeout_seconds,
                on_wait=on_wait,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            st.error(str(exc))
        else:
            preflight_payload = result.model_dump(mode="json")
            st.session_state["ui_preflight"] = preflight_payload
            progress.progress(100, text="模型环境预检完成")
            _store_stage_record(
                st,
                stage="preflight",
                title="模型环境预检",
                payload=preflight_payload,
            )

    run_requested = run_col.button("执行 Agent 融合", type="primary", use_container_width=True)
    run_requested = run_requested or st.session_state.pop("ui_submit_agent", False)
    if run_requested:
        if not config.request.strip():
            st.error("请输入自然语言任务。")
        else:
            agent_timeout = config.timeout_seconds + config.preflight_timeout_seconds + 30
            progress, on_wait = _task_wait_reporter(st, "Agent 任务", agent_timeout)
            try:
                execution = run_with_elapsed(
                    lambda: run_agent_from_ui(config),
                    timeout_seconds=agent_timeout,
                    on_wait=on_wait,
                )
            except subprocess.TimeoutExpired:
                st.error("Agent 进程超时。请检查模型环境或适当增大融合超时。")
            else:
                st.session_state["ui_execution"] = execution
                st.session_state["ui_output_dir"] = config.output_dir
                progress.progress(100, text="Agent 任务完成")
                _store_stage_record(
                    st,
                    stage="agent",
                    title="Agent 自然语言任务",
                    payload=execution,
                    output_dir=config.output_dir,
                )

    with st.container(border=True):
        current_output_dir = _render_stage_records(st, config)
        _render_history_selector(
            st,
            config.output_dir.parent,
            current_output_dir=current_output_dir,
        )


def main() -> None:
    """Launch the Streamlit runner for the ``rsfusion-ui`` console command."""

    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError as exc:
        raise SystemExit('Install the UI dependency first: pip install -e ".[ui]"') from exc
    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]
    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    run_app()
