"""Command-line interface for RSFusionAgent."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from rsfusion_agent.agent.llm_client import (
    CompatibleResponsesClient,
    resolve_provider_settings,
)
from rsfusion_agent.agent.llm_tools import AgentToolbox, LLMToolContext
from rsfusion_agent.agent.llm_workflow import LLMFusionAgent
from rsfusion_agent.agent.state import FusionRunRequest
from rsfusion_agent.agent.tiff_workflow import TiffFusionRequest, YRE151TiffPatchAgent
from rsfusion_agent.agent.workflow import YRE151PatchAgent
from rsfusion_agent.tools.h5_patch import inspect_h5_pair
from rsfusion_agent.tools.model_runtime import preflight_yre151_runtime
from rsfusion_agent.tools.raster_inspector import inspect_raster
from rsfusion_agent.tools.tiff_crop import (
    CUSTOM_PROFILE,
    YRE_LEGACY_TEST_PROFILE,
    crop_tiff_triplet,
)
from rsfusion_agent.tools.tiff_triplet import inspect_tiff_triplet


def _emit_json(payload: dict, *, indent: int | None) -> None:
    """Write JSON to stdout as UTF-8 so Windows GBK consoles do not crash on emoji."""
    text = json.dumps(payload, ensure_ascii=False, indent=indent)
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is not None:
        buffer.write(text.encode("utf-8"))
        buffer.write(b"\n")
        buffer.flush()
        return
    print(text)


def _write_json_file(path: Path, payload: dict, *, indent: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=indent)
    path.write_text(f"{text}\n", encoding="utf-8")


def _preflight_from_args(args: argparse.Namespace):
    return preflight_yre151_runtime(
        checkpoint_path=args.checkpoint,
        model_python=args.model_python,
        device=args.device,
        timeout_seconds=args.preflight_timeout,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rsfusion",
        description="Executable tools for the RSFusionAgent project.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect-raster",
        help="Inspect the metadata of a raster image without loading all pixels.",
    )
    inspect_parser.add_argument("image_path", help="Path to a GeoTIFF or GDAL-readable raster.")
    inspect_parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output.",
    )

    tiff_triplet_parser = subparsers.add_parser(
        "inspect-tiff-triplet",
        help="Validate raw auxiliary-MS, auxiliary-HS and target-MS TIFF metadata.",
    )
    tiff_triplet_parser.add_argument("--aux-ms", required=True, help="Auxiliary-time MS TIFF.")
    tiff_triplet_parser.add_argument("--aux-hs", required=True, help="Auxiliary-time HS TIFF.")
    tiff_triplet_parser.add_argument("--target-ms", required=True, help="Target-time MS TIFF.")
    tiff_triplet_parser.add_argument("--scale", type=int, default=3)
    tiff_triplet_parser.add_argument("--expected-ms-bands", type=int, default=4)
    tiff_triplet_parser.add_argument("--expected-hs-bands", type=int, default=151)
    tiff_triplet_parser.add_argument("--grid-tolerance", type=float, default=1e-6)
    tiff_triplet_parser.add_argument("--pretty", action="store_true")

    crop_tiff_parser = subparsers.add_parser(
        "crop-tiff-triplet",
        help="Crop a TIFF triplet with the legacy YRE window or an explicit custom window.",
    )
    crop_tiff_parser.add_argument("--aux-ms", required=True)
    crop_tiff_parser.add_argument("--aux-hs", required=True)
    crop_tiff_parser.add_argument("--target-ms", required=True)
    crop_tiff_parser.add_argument("--output-dir", required=True)
    crop_tiff_parser.add_argument(
        "--profile",
        choices=(YRE_LEGACY_TEST_PROFILE, CUSTOM_PROFILE),
        default=YRE_LEGACY_TEST_PROFILE,
    )
    crop_tiff_parser.add_argument("--ms-row-offset", type=int)
    crop_tiff_parser.add_argument("--ms-col-offset", type=int)
    crop_tiff_parser.add_argument("--window-height", type=int)
    crop_tiff_parser.add_argument("--window-width", type=int)
    crop_tiff_parser.add_argument("--hs-row-offset", type=int)
    crop_tiff_parser.add_argument("--hs-col-offset", type=int)
    crop_tiff_parser.add_argument("--pretty", action="store_true")

    h5_parser = subparsers.add_parser(
        "inspect-h5",
        help="Validate a legacy YRE-151 auxiliary/target HDF5 pair.",
    )
    h5_parser.add_argument("--aux-h5", required=True, help="Auxiliary-time DownT1 HDF5.")
    h5_parser.add_argument("--target-h5", required=True, help="Target-time DownT2 HDF5.")
    h5_parser.add_argument("--patch-index", type=int, default=0)
    h5_parser.add_argument("--pretty", action="store_true")

    preflight_parser = subparsers.add_parser(
        "preflight-runtime",
        help="Check the configured PyTorch/CUDA environment and checkpoint without inference.",
    )
    preflight_parser.add_argument("--checkpoint", required=True)
    preflight_parser.add_argument(
        "--model-python",
        default=os.environ.get("RSFUSION_MODEL_PYTHON", sys.executable),
        help="Python executable from an environment containing compatible PyTorch.",
    )
    preflight_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    preflight_parser.add_argument("--preflight-timeout", type=int, default=60)
    preflight_parser.add_argument("--pretty", action="store_true")

    infer_parser = subparsers.add_parser(
        "infer-h5",
        help="Run the first YRE-151 single-patch agent workflow.",
    )
    infer_parser.add_argument("--aux-h5", required=True)
    infer_parser.add_argument("--target-h5", required=True)
    infer_parser.add_argument("--checkpoint", required=True)
    infer_parser.add_argument("--output-dir", required=True)
    infer_parser.add_argument("--patch-index", type=int, default=0)
    infer_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    infer_parser.add_argument(
        "--model-python",
        default=os.environ.get("RSFUSION_MODEL_PYTHON", sys.executable),
        help="Python executable from an environment containing compatible PyTorch.",
    )
    infer_parser.add_argument("--timeout", type=int, default=600)
    infer_parser.add_argument(
        "--runtime-retries",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="Extra retries for known native model-process crashes.",
    )
    infer_parser.add_argument("--pretty", action="store_true")

    infer_tiff_parser = subparsers.add_parser(
        "infer-tiff",
        help="Run one metadata-validated raw-TIFF YRE-151 crop without reference metrics.",
    )
    infer_tiff_parser.add_argument("--aux-ms", required=True)
    infer_tiff_parser.add_argument("--aux-hs", required=True)
    infer_tiff_parser.add_argument("--target-ms", required=True)
    infer_tiff_parser.add_argument("--checkpoint", required=True)
    infer_tiff_parser.add_argument("--output-dir", required=True)
    infer_tiff_parser.add_argument("--patch-size", type=int, default=180)
    infer_tiff_parser.add_argument("--row-offset", type=int, default=0)
    infer_tiff_parser.add_argument("--col-offset", type=int, default=0)
    infer_tiff_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    infer_tiff_parser.add_argument(
        "--model-python",
        default=os.environ.get("RSFUSION_MODEL_PYTHON", sys.executable),
    )
    infer_tiff_parser.add_argument("--timeout", type=int, default=600)
    infer_tiff_parser.add_argument("--runtime-retries", type=int, choices=(0, 1, 2), default=1)
    infer_tiff_parser.add_argument("--pretty", action="store_true")

    agent_parser = subparsers.add_parser(
        "agent",
        help="Use natural language and OpenAI function calling to orchestrate local tools.",
    )
    agent_parser.add_argument("--request", required=True, help="Natural-language task request.")
    agent_parser.add_argument("--aux-h5", required=True)
    agent_parser.add_argument("--target-h5", required=True)
    agent_parser.add_argument("--checkpoint", required=True)
    agent_parser.add_argument("--output-dir", required=True)
    agent_parser.add_argument("--patch-index", type=int, default=0)
    agent_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    agent_parser.add_argument(
        "--model-python",
        default=os.environ.get("RSFUSION_MODEL_PYTHON", sys.executable),
        help="Python executable from an environment containing compatible PyTorch.",
    )
    agent_parser.add_argument(
        "--provider",
        choices=("openai", "qwen", "deepseek", "custom"),
        default=os.environ.get("RSFUSION_LLM_PROVIDER", "openai"),
        help="LLM provider used through an OpenAI-compatible Responses API.",
    )
    agent_parser.add_argument(
        "--llm-model",
        default=None,
        help="Provider model ID. Defaults to RSFUSION_LLM_MODEL, OPENAI_MODEL, or provider default.",
    )
    agent_parser.add_argument(
        "--base-url",
        default=None,
        help="Provider base URL. Defaults to RSFUSION_LLM_BASE_URL or OPENAI_BASE_URL.",
    )
    agent_parser.add_argument("--max-turns", type=int, default=6)
    agent_parser.add_argument("--timeout", type=int, default=600)
    agent_parser.add_argument(
        "--runtime-retries",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="Extra retries for known native model-process crashes.",
    )
    agent_parser.add_argument(
        "--preflight-timeout",
        type=int,
        default=60,
        help="Maximum seconds for the free local runtime health check before calling the LLM.",
    )
    agent_parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the local runtime health check. Intended only for troubleshooting.",
    )
    agent_parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect-raster":
        try:
            result = inspect_raster(args.image_path)
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))

        indent = 2 if args.pretty else None
        _emit_json(result.model_dump(mode="json"), indent=indent)
        return 0

    if args.command == "inspect-tiff-triplet":
        try:
            result = inspect_tiff_triplet(
                args.aux_ms,
                args.aux_hs,
                args.target_ms,
                scale=args.scale,
                expected_ms_bands=args.expected_ms_bands,
                expected_hs_bands=args.expected_hs_bands,
                grid_tolerance=args.grid_tolerance,
            )
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        indent = 2 if args.pretty else None
        _emit_json(result.model_dump(mode="json"), indent=indent)
        return 0

    if args.command == "crop-tiff-triplet":
        try:
            result = crop_tiff_triplet(
                args.aux_ms,
                args.aux_hs,
                args.target_ms,
                args.output_dir,
                profile=args.profile,
                ms_row_offset=args.ms_row_offset,
                ms_col_offset=args.ms_col_offset,
                window_height=args.window_height,
                window_width=args.window_width,
                hs_row_offset=args.hs_row_offset,
                hs_col_offset=args.hs_col_offset,
            )
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        indent = 2 if args.pretty else None
        _emit_json(result.model_dump(mode="json"), indent=indent)
        return 0

    if args.command == "inspect-h5":
        try:
            result = inspect_h5_pair(
                args.aux_h5,
                args.target_h5,
                patch_index=args.patch_index,
            )
        except (FileNotFoundError, IndexError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        indent = 2 if args.pretty else None
        _emit_json(result.model_dump(mode="json"), indent=indent)
        return 0

    if args.command == "preflight-runtime":
        try:
            result = _preflight_from_args(args)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        indent = 2 if args.pretty else None
        _emit_json(result.model_dump(mode="json"), indent=indent)
        return 0

    if args.command == "infer-h5":
        try:
            request = FusionRunRequest(
                auxiliary_h5_path=Path(args.aux_h5),
                target_h5_path=Path(args.target_h5),
                checkpoint_path=Path(args.checkpoint),
                model_python=Path(args.model_python),
                output_dir=Path(args.output_dir),
                patch_index=args.patch_index,
                device=args.device,
                timeout_seconds=args.timeout,
                runtime_retries=args.runtime_retries,
            )
            result = YRE151PatchAgent().run(request)
        except (FileNotFoundError, IndexError, RuntimeError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        indent = 2 if args.pretty else None
        _emit_json(result.model_dump(mode="json"), indent=indent)
        return 0

    if args.command == "infer-tiff":
        try:
            result = YRE151TiffPatchAgent().run(
                TiffFusionRequest(
                    auxiliary_ms_path=Path(args.aux_ms),
                    auxiliary_hs_path=Path(args.aux_hs),
                    target_ms_path=Path(args.target_ms),
                    checkpoint_path=Path(args.checkpoint),
                    model_python=Path(args.model_python),
                    output_dir=Path(args.output_dir),
                    patch_size=args.patch_size,
                    row_offset=args.row_offset,
                    col_offset=args.col_offset,
                    device=args.device,
                    timeout_seconds=args.timeout,
                    runtime_retries=args.runtime_retries,
                )
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        indent = 2 if args.pretty else None
        _emit_json(result.model_dump(mode="json"), indent=indent)
        return 0

    if args.command == "agent":
        try:
            context = LLMToolContext(
                auxiliary_h5_path=Path(args.aux_h5),
                target_h5_path=Path(args.target_h5),
                checkpoint_path=Path(args.checkpoint),
                model_python=Path(args.model_python),
                output_dir=Path(args.output_dir),
                patch_index=args.patch_index,
                device=args.device,
                timeout_seconds=args.timeout,
                runtime_retries=args.runtime_retries,
            )
            runtime_preflight = None
            if not args.skip_preflight:
                runtime_preflight = _preflight_from_args(args)
                _write_json_file(
                    Path(args.output_dir) / "preflight.json",
                    runtime_preflight.model_dump(mode="json"),
                    indent=2,
                )
            settings = resolve_provider_settings(
                provider=args.provider,
                model=args.llm_model,
                base_url=args.base_url,
            )
            client = CompatibleResponsesClient(
                provider=settings.provider,
                model=settings.model,
                base_url=settings.base_url,
            )
            result = LLMFusionAgent(
                client=client,
                toolbox=AgentToolbox(context),
                max_turns=args.max_turns,
                runtime_preflight=runtime_preflight,
            ).run(args.request)
        except Exception as exc:
            parser.error(str(exc))
        indent = 2 if args.pretty else None
        payload = result.model_dump(mode="json")
        _write_json_file(Path(args.output_dir) / "agent_result.json", payload, indent=indent)
        _emit_json(payload, indent=indent)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
