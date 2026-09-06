"""Command-line interface for RSFusionAgent."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

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

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

