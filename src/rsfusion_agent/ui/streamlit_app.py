"""Streamlit demonstration UI that reuses the hardened RSFusionAgent CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UiRunConfig:
    """Non-secret configuration collected by the local UI."""

    request: str
    auxiliary_h5_path: Path
    target_h5_path: Path
    checkpoint_path: Path
    model_python: Path
    output_dir: Path
    provider: str = "qwen"
    llm_model: str | None = None
    base_url: str | None = None
    patch_index: int = 0
    device: str = "cuda"
    timeout_seconds: int = 600
    preflight_timeout_seconds: int = 60


@dataclass(frozen=True)
class UiAgentExecution:
    """Result of invoking the existing CLI from the UI process."""

    return_code: int
    stdout: str
    stderr: str
    result_path: Path
    result: dict[str, Any] | None


def build_agent_command(config: UiRunConfig, *, python_executable: str | None = None) -> list[str]:
    """Build a shell-free CLI command without exposing API keys in arguments."""

    command = [
        python_executable or sys.executable,
        "-m",
        "rsfusion_agent.cli",
        "agent",
        "--request",
        config.request,
        "--aux-h5",
        str(config.auxiliary_h5_path),
        "--target-h5",
        str(config.target_h5_path),
        "--checkpoint",
        str(config.checkpoint_path),
        "--model-python",
        str(config.model_python),
        "--output-dir",
        str(config.output_dir),
        "--patch-index",
        str(config.patch_index),
        "--device",
        config.device,
        "--provider",
        config.provider,
        "--timeout",
        str(config.timeout_seconds),
        "--preflight-timeout",
        str(config.preflight_timeout_seconds),
        "--pretty",
    ]
    if config.llm_model:
        command.extend(("--llm-model", config.llm_model))
    if config.base_url:
        command.extend(("--base-url", config.base_url))
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


def _env_default(name: str, fallback: str = "") -> str:
    return os.environ.get(name, fallback)


def _build_config(st: Any) -> UiRunConfig:
    """Render configuration fields and return the current local-only run config."""

    with st.sidebar:
        st.header("运行配置")
        auxiliary = st.text_input("辅助时相 H5", value=_env_default("RSFUSION_AUX_H5"))
        target = st.text_input("目标时相 H5", value=_env_default("RSFUSION_TARGET_H5"))
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
        patch_index = st.number_input("Patch Index", min_value=0, value=0, step=1)
        timeout_seconds = st.number_input("融合超时（秒）", min_value=30, value=600, step=30)
        preflight_timeout = st.number_input("预检超时（秒）", min_value=10, value=60, step=10)

    request = st.text_area(
        "自然语言任务",
        value="请先检查数据，再融合第0个patch，并汇报PSNR、SAM、SSIM和输出文件。",
        height=110,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return UiRunConfig(
        request=request,
        auxiliary_h5_path=Path(auxiliary),
        target_h5_path=Path(target),
        checkpoint_path=Path(checkpoint),
        model_python=Path(model_python),
        output_dir=Path(output_root) / f"ui_run_{timestamp}",
        provider=provider,
        llm_model=llm_model or None,
        base_url=base_url or None,
        patch_index=int(patch_index),
        device=device,
        timeout_seconds=int(timeout_seconds),
        preflight_timeout_seconds=int(preflight_timeout),
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

    _render_preflight(st, result.get("runtime_preflight"))
    fusion = result.get("fusion_result") or {}
    metrics = fusion.get("metrics") or {}
    if metrics:
        st.subheader("融合指标")
        columns = st.columns(3)
        columns[0].metric("PSNR", f"{metrics.get('psnr', 0):.4f}")
        columns[1].metric("SAM", f"{metrics.get('sam', 0):.4f}")
        columns[2].metric("SSIM", f"{metrics.get('ssim', 0):.4f}")

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

    artifacts = fusion.get("artifacts") or {}
    image_columns = st.columns(2)
    for column, title, artifact_key in (
        (image_columns[0], "RGB 预览", "rgb_preview"),
        (image_columns[1], "SAM 热力图", "sam_heatmap"),
    ):
        file_name = artifacts.get(artifact_key)
        path = output_dir / file_name if file_name else None
        if path and path.is_file():
            column.image(str(path), caption=title, use_container_width=True)

    st.subheader("本地结果文件")
    for name in ("report.md", "preflight.json", "agent_result.json", "metrics.json"):
        path = output_dir / name
        if path.is_file():
            st.download_button(
                label=f"下载 {name}",
                data=path.read_bytes(),
                file_name=name,
                mime="application/json" if name.endswith(".json") else "text/markdown",
            )
    with st.expander("查看 agent_result.json"):
        st.json(result)


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
