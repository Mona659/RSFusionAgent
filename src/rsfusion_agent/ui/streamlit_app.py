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
        if config.crop_manifest_path is None:
            raise ValueError("TIFF mode requires a generated crop manifest")
        command.extend(
            [
                "agent-tiff",
                "--request",
                config.request,
                "--crop-manifest",
                str(config.crop_manifest_path),
                "--patch-size",
                str(config.patch_size),
                "--row-offset",
                str(config.row_offset),
                "--col-offset",
                str(config.col_offset),
            ]
        )
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
        "--pretty",
    ]
    if config.experiment_mode == "simulation":
        if config.target_hs_reference_path is None:
            raise ValueError("模拟实验需要提供目标时相 HS 参考 TIFF")
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
    path = (output_dir / candidate).resolve() if Path(candidate).name == candidate else Path(candidate).resolve()
    try:
        path.relative_to(output_root)
    except ValueError:
        return None
    return path if path.is_file() else None


def run_agent_from_ui(config: UiRunConfig) -> UiAgentExecution:
    """Execute the CLI so UI and terminal runs share exactly one workflow."""

    active_config = config
    if config.input_mode == "tiff" and config.crop_manifest_path is None:
        crop = subprocess.run(
            build_crop_command(config),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=config.timeout_seconds,
            env=os.environ.copy(),
        )
        if crop.returncode != 0:
            return UiAgentExecution(
                return_code=crop.returncode,
                stdout=crop.stdout,
                stderr=crop.stderr,
                result_path=config.output_dir / "agent_result.json",
                result=None,
            )
        active_config = replace(
            config, crop_manifest_path=config.output_dir / "prepared_crop" / "crop_manifest.json"
        )
    completed = subprocess.run(
        build_agent_command(active_config),
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
                if experiment_mode == "simulation":
                    target_hs_reference = st.text_input(
                        "T2 目标时相 HS 参考 TIFF（如 ZY2）",
                        value=_env_default("RSFUSION_TARGET_HS_REFERENCE_TIFF"),
                    )
                    st.caption("按原 Database.py：T2 HS 裁剪后线性插值 3 倍，作为伪真值评测。")
        if is_tiff:
            with st.expander("② 裁剪参数", expanded=True):
                crop_profile = st.selectbox(
                    "裁剪窗口方案",
                    ("yre_legacy_test_v1", "custom"),
                    format_func=lambda value: "YRE 原始测试窗口"
                    if value == "yre_legacy_test_v1"
                    else "自定义窗口",
                )
                if crop_profile == "yre_legacy_test_v1":
                    st.caption("MS：起始行 0、起始列 360、大小 540 × 540")
                    st.caption("HS（T1 与 T2）：起始行 0、起始列 120、大小 180 × 180")
                else:
                    st.caption("MS 与 HS 分别使用各自原始像素网格；HS 默认对应 MS 的 1/3。")
                    ms_left, ms_right = st.columns(2)
                    ms_row = int(ms_left.number_input("MS 起始行", min_value=0, value=0, step=3))
                    ms_col = int(ms_right.number_input("MS 起始列", min_value=0, value=360, step=3))
                    window_height = int(ms_left.number_input("MS 裁剪高度", min_value=3, value=540, step=3))
                    window_width = int(ms_right.number_input("MS 裁剪宽度", min_value=3, value=540, step=3))
                    hs_left, hs_right = st.columns(2)
                    hs_row = int(hs_left.number_input("HS 起始行", min_value=0, value=0))
                    hs_col = int(hs_right.number_input("HS 起始列", min_value=0, value=120))
                    st.caption("HS 裁剪大小自动为 MS 高度/宽度的 1/3。")
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
            patch_index = (
                st.number_input("Patch Index", min_value=0, value=0, step=1) if not is_tiff else 0
            )
            patch_size = (
                st.number_input("TIFF 推理 Patch 尺寸", min_value=3, value=180, step=3)
                if is_tiff
                else 180
            )
            row_offset = (
                st.number_input("Patch 起始行（相对裁剪结果）", min_value=0, value=0, step=3)
                if is_tiff
                else 0
            )
            col_offset = (
                st.number_input("Patch 起始列（相对裁剪结果）", min_value=0, value=0, step=3)
                if is_tiff
                else 0
            )
            timeout_seconds = st.number_input("融合超时（秒）", min_value=30, value=600, step=30)
            preflight_timeout = st.number_input("预检超时（秒）", min_value=10, value=60, step=10)
            runtime_retries = st.selectbox("原生崩溃额外重试次数", (0, 1, 2), index=1)
        with st.expander("④ 输出目录", expanded=True):
            output_root = st.text_input("输出根目录", value="outputs/ui_runs")
            st.caption(f"当前运行目录：ui_run_{st.session_state['ui_run_id']}")

    request = st.text_area(
        "自然语言任务",
        value=(
            "请先检查数据，再融合第0个patch，并汇报PSNR、SAM、SSIM和输出文件。"
            if not is_tiff
            else (
                "请检查裁剪清单，融合当前 TIFF patch，并汇报输出文件、运行时间和指标。"
                if experiment_mode == "simulation"
                else "请检查裁剪清单，融合当前 TIFF patch，并汇报输出文件、运行时间和指标可用性。"
            )
        ),
        height=110,
    )
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
    st.subheader("第一步：原始 TIFF 输入检查")
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
                "目标时相 HS 参考（伪真值）",
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


def _render_crop_result(st: Any, execution: UiCropExecution | None) -> None:
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
        cards = _load_crop_preview_cards(execution.result)
    except (FileNotFoundError, ValueError) as exc:
        st.warning(f"裁剪预览生成失败：{exc}")
        return
    if cards:
        st.subheader("裁剪结果可视化")
        columns = st.columns(len(cards))
        for column, (title, metadata, preview, band_label) in zip(columns, cards, strict=True):
            column.markdown(f"#### {title}")
            column.metric("裁剪尺寸", f"{metadata['width']} × {metadata['height']}")
            column.image(preview, caption=band_label, use_container_width=True)


def _render_result(st: Any, execution: UiAgentExecution, output_dir: Path) -> None:
    result = execution.result
    if result is None:
        st.error("Agent 未生成可解析的 agent_result.json。")
        with st.expander("CLI 输出"):
            st.code(execution.stderr or execution.stdout or "No output")
        return

    status = result.get("status", "unknown")
    if status == "completed":
        st.success("Agent 已完成融合任务。")
    else:
        st.warning(f"Agent 已结束，状态：{status}")
        for item in reversed(result.get("trace") or []):
            diagnosis = (item.get("output") or {}).get("diagnosis")
            if diagnosis:
                st.error(diagnosis.get("summary", "Agent 工具执行失败。"))
                st.caption(f"建议：{diagnosis.get('recommended_action', '')}")
                break

    _render_preflight(st, result.get("runtime_preflight"))
    fusion = result.get("fusion_result") or {}
    metrics = fusion.get("metrics") or {}
    if metrics:
        if fusion.get("metrics_status") == "available_legacy_interpolated_target_hs_reference":
            st.warning("以下指标基于 Database.py 复现的插值伪真值，不等同于原生高分辨率真值。")
        st.subheader("融合指标")
        columns = st.columns(3)
        columns[0].metric("PSNR", f"{metrics.get('psnr', 0):.4f}")
        columns[1].metric("SAM", f"{metrics.get('sam', 0):.4f}")
        columns[2].metric("SSIM", f"{metrics.get('ssim', 0):.4f}")
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
        fusion, output_dir=output_dir, artifact_key="reference_rgb_preview"
    ):
        reference_title = (
            "插值伪标签 RGB（同尺度）"
            if fusion.get("metrics_status")
            == "available_legacy_interpolated_target_hs_reference"
            else "标签 RGB（H5 真值，同尺度）"
        )
        image_specs.append((reference_title, "reference_rgb_preview"))
    if resolve_result_artifact(fusion, output_dir=output_dir, artifact_key="sam_heatmap"):
        image_specs.append(("SAM 热力图", "sam_heatmap"))
    image_columns = st.columns(len(image_specs))
    for column, (title, artifact_key) in zip(image_columns, image_specs, strict=True):
        path = resolve_result_artifact(fusion, output_dir=output_dir, artifact_key=artifact_key)
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


def _render_stage_records(st: Any) -> Path | None:
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
                _render_result(st, record.payload, record.output_dir or Path("."))
                current_output_dir = record.output_dir
            elif record.stage == "crop":
                _render_crop_result(st, record.payload)
            elif record.stage == "input_check":
                _render_raw_tiff_input_check(st, record.payload)
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
        preflight_col, run_col = st.columns(2)
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

    if run_col.button("执行 Agent 融合", type="primary", use_container_width=True):
        if not config.request.strip():
            st.error("请输入自然语言任务。")
        else:
            active_config = config
            if active_config.input_mode == "tiff" and active_config.crop_manifest_path is None:
                crop_progress, crop_on_wait = _task_wait_reporter(
                    st, "Agent 前置自动裁剪", active_config.timeout_seconds
                )
                try:
                    crop_execution = run_with_elapsed(
                        lambda: run_crop_from_ui(active_config),
                        timeout_seconds=active_config.timeout_seconds,
                        on_wait=crop_on_wait,
                    )
                except subprocess.TimeoutExpired:
                    st.error("裁剪超时。请检查输入文件和裁剪窗口。")
                    crop_execution = None
                if crop_execution is not None:
                    st.session_state["ui_crop_execution"] = crop_execution
                    if crop_execution.return_code == 0:
                        st.session_state["ui_crop_reuse_key"] = current_crop_key
                        active_config = replace(
                            active_config, crop_manifest_path=crop_execution.manifest_path
                        )
                        crop_progress.progress(100, text="Agent 前置自动裁剪完成")
                        _store_stage_record(
                            st,
                            stage="crop",
                            title="TIFF 裁剪（Agent 自动执行）",
                            payload=crop_execution,
                        )
                    else:
                        st.error("自动裁剪失败，未执行 Agent 融合。")
            if active_config.input_mode != "tiff" or active_config.crop_manifest_path is not None:
                agent_timeout = (
                    active_config.timeout_seconds + active_config.preflight_timeout_seconds + 30
                )
                progress, on_wait = _task_wait_reporter(st, "Agent 融合", agent_timeout)
                try:
                    execution = run_with_elapsed(
                        lambda: run_agent_from_ui(active_config),
                        timeout_seconds=agent_timeout,
                        on_wait=on_wait,
                    )
                except subprocess.TimeoutExpired:
                    st.error("Agent 进程超时。请检查模型环境或适当增大融合超时。")
                else:
                    st.session_state["ui_execution"] = execution
                    st.session_state["ui_output_dir"] = active_config.output_dir
                    progress.progress(100, text="Agent 融合完成")
                    _store_stage_record(
                        st,
                        stage="agent",
                        title="Agent 融合",
                        payload=execution,
                        output_dir=active_config.output_dir,
                    )

    with st.container(border=True):
        current_output_dir = _render_stage_records(st)
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
