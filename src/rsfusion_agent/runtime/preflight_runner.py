"""Report health of the isolated YRE-151 PyTorch environment as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from rsfusion_agent.runtime.yre151_runner import load_checkpoint, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the YRE-151 model environment")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    state_dict, checkpoint_epoch = load_checkpoint(args.checkpoint.resolve(), torch.device("cpu"))
    warnings: list[str] = []
    cuda_device_name: str | None = None
    cuda_total_memory_bytes: int | None = None
    cuda_available = torch.cuda.is_available()

    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        cuda_device_name = properties.name
        cuda_total_memory_bytes = properties.total_memory
    elif args.device == "auto" and not cuda_available:
        warnings.append("CUDA is unavailable; auto selected CPU inference.")

    print(
        json.dumps(
            {
                "status": "ready",
                "python_executable": sys.executable,
                "python_version": sys.version.split()[0],
                "torch_version": torch.__version__,
                "requested_device": args.device,
                "resolved_device": str(device),
                "cuda_available": cuda_available,
                "cuda_device_name": cuda_device_name,
                "cuda_total_memory_bytes": cuda_total_memory_bytes,
                "checkpoint_epoch": checkpoint_epoch,
                "checkpoint_tensor_count": len(state_dict),
                "warnings": warnings,
            }
        )
    )


if __name__ == "__main__":
    main()
