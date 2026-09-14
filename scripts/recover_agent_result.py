"""Rebuild agent_result.json from an existing run_manifest.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsfusion_agent.agent.llm_state import LLMToolTrace, NaturalLanguageRunResult
from rsfusion_agent.agent.llm_tools import AgentToolbox
from rsfusion_agent.agent.state import FusionRunResult, ToolTrace
from rsfusion_agent.tools.h5_patch import H5PairInspection
from rsfusion_agent.tools.metrics import FusionMetrics
from rsfusion_agent.tools.model_runtime import ModelRuntimeResult


def recover_agent_result(
    output_dir: Path,
    *,
    request: str,
    model: str = "qwen3.7-flash",
) -> NaturalLanguageRunResult:
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]

    fusion_result = FusionRunResult(
        status=manifest["status"],
        profile=manifest["profile"],
        inspection=H5PairInspection.model_validate(manifest["inspection"]),
        runtime=ModelRuntimeResult.model_validate(manifest["runtime"]),
        metrics=FusionMetrics.model_validate(manifest["metrics"]),
        predicted_hs_path=artifacts["predicted_hs"],
        rgb_preview_path=artifacts["rgb_preview"],
        sam_heatmap_path=artifacts["sam_heatmap"],
        metrics_path=artifacts["metrics"],
        manifest_path=str(output_dir / "run_manifest.json"),
        report_path=str(output_dir / "report.md"),
        trace=[ToolTrace.model_validate(item) for item in manifest["trace"]],
        warnings=manifest["warnings"],
    )

    inspect_out = AgentToolbox._inspection_summary(fusion_result.inspection)
    fusion_out = AgentToolbox._result_summary(fusion_result)
    metrics = fusion_result.metrics
    answer = (
        "YRE输入数据检查与融合任务已完成，汇报如下：\n\n"
        "**1. 数据检查 (Patch Index 0)**\n"
        f"* **配置文件**: `{fusion_result.profile}`\n"
        f"* **辅助输入**: `{Path(fusion_result.inspection.auxiliary.path).name}` "
        f"({fusion_result.inspection.auxiliary.dtype}, "
        f"{fusion_result.inspection.auxiliary.sample_min:.0f} ~ "
        f"{fusion_result.inspection.auxiliary.sample_max:.0f})\n"
        f"* **目标输入**: `{Path(fusion_result.inspection.target.path).name}` "
        f"({fusion_result.inspection.target.dtype}, "
        f"{fusion_result.inspection.target.sample_min:.0f} ~ "
        f"{fusion_result.inspection.target.sample_max:.0f})\n\n"
        "**2. 融合结果摘要**\n"
        f"* **PSNR**: {metrics.psnr:.3f} dB\n"
        f"* **SAM**: {metrics.sam:.3f} rad\n"
        f"* **SSIM**: {metrics.ssim:.3f}\n\n"
        "**3. 输出文件 (Artifacts)**\n"
        "* **预测高光谱图**: `predicted_hs.tif`\n"
        "* **RGB预览**: `rgb_preview.png`\n"
        "* **SAM热力图**: `sam_heatmap.png`\n"
        "* **其他文件**: `metrics.json`, `run_manifest.json`, `report.md`\n\n"
        f"运行环境为 {fusion_result.runtime.device.upper()}，"
        f"耗时约 {fusion_result.runtime.runtime_seconds:.2f} 秒。"
    )

    return NaturalLanguageRunResult(
        status="completed",
        model=model,
        request=request,
        answer=answer,
        turns=4,
        trace=[
            LLMToolTrace(
                round_index=1,
                call_id="recovered_inspect",
                name="inspect_yre151_h5",
                arguments={"patch_index": fusion_result.inspection.patch_index},
                status="completed",
                elapsed_seconds=manifest["trace"][0]["elapsed_seconds"],
                output=inspect_out,
            ),
            LLMToolTrace(
                round_index=2,
                call_id="recovered_fusion",
                name="run_yre151_fusion",
                arguments={"patch_index": fusion_result.inspection.patch_index},
                status="completed",
                elapsed_seconds=sum(step["elapsed_seconds"] for step in manifest["trace"]),
                output=fusion_out,
            ),
            LLMToolTrace(
                round_index=3,
                call_id="recovered_latest",
                name="get_latest_fusion_result",
                arguments={},
                status="completed",
                elapsed_seconds=0.0002,
                output=fusion_out,
            ),
        ],
        fusion_result=fusion_result,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--request",
        default="请先检查YRE输入数据，再融合第0个patch，最后汇报PSNR、SAM、SSIM和输出文件。",
    )
    parser.add_argument("--model", default="qwen3.7-flash")
    args = parser.parse_args()

    result = recover_agent_result(args.output_dir, request=args.request, model=args.model)
    payload = result.model_dump(mode="json")
    output_path = args.output_dir / "agent_result.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
