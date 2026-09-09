"""Streamlit demonstration UI that reuses the hardened RSFusionAgent CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any


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
class UiRunHistoryItem:
    """One locally persisted agent result available for display without a rerun."""

    output_dir: Path
    status: str
    created_at: datetime
    psnr: float | None


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


def _env_default(name: str, fallback: str = "") -> str:
    return os.environ.get(name, fallback)


def _build_config(st: Any) -> UiRunConfig:
    """Render configuration fields and return the current local-only run config."""

    with st.sidebar:
        st.header("运行配置")
        input_mode = st.radio("输入模式", ("H5 演示模式", "原始 TIFF 模式"), horizontal=True)
        is_tiff = input_mode == "原始 TIFF 模式"
        auxiliary = target = ""
        auxiliary_ms = auxiliary_hs = target_ms = ""
        crop_profile = "yre_legacy_test_v1"
        ms_row = ms_col = window_height = window_width = hs_row = hs_col = None
        if not is_tiff:
            auxiliary = st.text_input("辅助时相 H5", value=_env_default("RSFUSION_AUX_H5"))
            target = st.text_input("目标时相 H5", value=_env_default("RSFUSION_TARGET_H5"))
        else:
            auxiliary_ms = st.text_input("辅助时相 MS TIFF", value=_env_default("RSFUSION_AUX_MS_TIFF"))
            auxiliary_hs = st.text_input("辅助时相 HS TIFF", value=_env_default("RSFUSION_AUX_HS_TIFF"))
            target_ms = st.text_input("目标时相 MS TIFF", value=_env_default("RSFUSION_TARGET_MS_TIFF"))
            crop_profile = st.selectbox(
                "裁剪窗口",
                ("yre_legacy_test_v1", "custom"),
                format_func=lambda value: "YRE 原始测试窗口"
                if value == "yre_legacy_test_v1"
                else "自定义窗口",
            )
            if crop_profile == "custom":
                ms_row = int(st.number_input("MS 起始行", min_value=0, value=0, step=3))
                ms_col = int(st.number_input("MS 起始列", min_value=0, value=360, step=3))
                window_height = int(st.number_input("MS 窗口高", min_value=3, value=540, step=3))
                window_width = int(st.number_input("MS 窗口宽", min_value=3, value=540, step=3))
                hs_row = int(st.number_input("HS 起始行", min_value=0, value=0))
                hs_col = int(st.number_input("HS 起始列", min_value=0, value=120))
        checkpoint = st.text_input("Checkpoint", value=_env_default("RSFUSION_CHECKPOINT"))
        model_python = st.text_input(
            "模型 Conda Python",
            value=_env_default("RSFUSION_MODEL_PYTHON", sys.executable),
        )
        output_root = st.text_input("输出根目录", value="outputs/ui_runs")
        provider = st.selectbox("LLM Provider", ("qwen", "openai", "deepseek", "custom"))
        llm_model = st.text_input("模型 ID", value=_env_default("RSFUSION_LLM_MODEL"))
        base_url = st.text_input("兼容 API Base URL", value=_env_default("RSFUSION_LLM_BASE_URL"))
        device = st.selectbox("推理设备", ("cuda", "auto", "cpu"))
        patch_index = st.number_input("Patch Index", min_value=0, value=0, step=1) if not is_tiff else 0
        patch_size = (
            st.number_input("TIFF 推理 Patch 尺寸", min_value=3, value=180, step=3)
            if is_tiff
            else 180
        )
        row_offset = (
            st.number_input("TIFF Patch 起始行（相对裁剪窗口）", min_value=0, value=0, step=3)
            if is_tiff
            else 0
        )
        col_offset = (
            st.number_input("TIFF Patch 起始列（相对裁剪窗口）", min_value=0, value=0, step=3)
            if is_tiff
            else 0
        )
        timeout_seconds = st.number_input("融合超时（秒）", min_value=30, value=600, step=30)
        preflight_timeout = st.number_input("预检超时（秒）", min_value=10, value=60, step=10)
        runtime_retries = st.selectbox("原生崩溃额外重试次数", (0, 1, 2), index=1)

    request = st.text_area(
        "自然语言任务",
        value=(
            "请先检查数据，再融合第0个patch，并汇报PSNR、SAM、SSIM和输出文件。"
            if not is_tiff
            else "请检查裁剪清单，融合当前 TIFF patch，并汇报输出文件、运行时间和指标可用性。"
        ),
        height=110,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return UiRunConfig(
        request=request,
        checkpoint_path=Path(checkpoint),
        model_python=Path(model_python),
        output_dir=Path(output_root) / f"ui_run_{timestamp}",
        input_mode="tiff" if is_tiff else "h5",
        auxiliary_h5_path=Path(auxiliary) if auxiliary else None,
        target_h5_path=Path(target) if target else None,
        auxiliary_ms_path=Path(auxiliary_ms) if auxiliary_ms else None,
        auxiliary_hs_path=Path(auxiliary_hs) if auxiliary_hs else None,
        target_ms_path=Path(target_ms) if target_ms else None,
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

    image_columns = st.columns(2)
    for column, title, artifact_key in (
        (image_columns[0], "RGB 预览", "rgb_preview"),
        (image_columns[1], "SAM 热力图", "sam_heatmap"),
    ):
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


def _render_history_selector(st: Any, output_root: Path) -> None:
    history = list_run_history(output_root)
    with st.sidebar:
        st.divider()
        st.subheader("本地运行历史")
        if not history:
            st.caption("当前输出根目录下尚无 agent_result.json。")
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
    _render_history_selector(st, config.output_dir.parent)

    preflight_col, run_col = st.columns(2)
    if preflight_col.button("预检模型环境", use_container_width=True):
        from rsfusion_agent.tools.model_runtime import preflight_yre151_runtime

        try:
            result = preflight_yre151_runtime(
                checkpoint_path=config.checkpoint_path,
                model_python=config.model_python,
                device=config.device,
                timeout_seconds=config.preflight_timeout_seconds,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.session_state["ui_preflight"] = result.model_dump(mode="json")

    if run_col.button("执行 Agent 融合", type="primary", use_container_width=True):
        if not config.request.strip():
            st.error("请输入自然语言任务。")
        else:
            with st.spinner("正在执行预检、Agent 工具调用和本地融合，请勿关闭页面..."):
                try:
                    execution = run_agent_from_ui(config)
                except subprocess.TimeoutExpired:
                    st.error("Agent 进程超时。请检查模型环境或适当增大融合超时。")
                else:
                    st.session_state["ui_execution"] = execution
                    st.session_state["ui_output_dir"] = config.output_dir

    _render_preflight(st, st.session_state.get("ui_preflight"))
    execution = st.session_state.get("ui_execution")
    output_dir = st.session_state.get("ui_output_dir")
    if execution and output_dir:
        _render_result(st, execution, output_dir)


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
