"""Command-line interface for RSFusionAgent."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from rsfusion_agent.agent.state import FusionRunRequest
from rsfusion_agent.agent.workflow import YRE151PatchAgent
from rsfusion_agent.tools.h5_patch import inspect_h5_pair
from rsfusion_agent.tools.raster_inspector import inspect_raster


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

    h5_parser = subparsers.add_parser(
        "inspect-h5",
        help="Validate a legacy YRE-151 auxiliary/target HDF5 pair.",
    )
    h5_parser.add_argument("--aux-h5", required=True, help="Auxiliary-time DownT1 HDF5.")
    h5_parser.add_argument("--target-h5", required=True, help="Target-time DownT2 HDF5.")
    h5_parser.add_argument("--patch-index", type=int, default=0)
    h5_parser.add_argument("--pretty", action="store_true")

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
    infer_parser.add_argument("--pretty", action="store_true")
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
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=indent))
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
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=indent))
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
            )
            result = YRE151PatchAgent().run(request)
        except (FileNotFoundError, IndexError, RuntimeError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        indent = 2 if args.pretty else None
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=indent))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
