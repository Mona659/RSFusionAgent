"""Run one YRE-151 patch in a Python environment that provides PyTorch."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from rsfusion_agent.models.dc_stsf import DC_STSF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolated DC-STSF YRE-151 model runner")
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in the model environment")
    return torch.device(requested)


def load_checkpoint(path: Path, device: torch.device) -> tuple[dict[str, torch.Tensor], int | None]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Checkpoint must be a dictionary containing the 'model' state dict")
    epoch = checkpoint.get("epoch")
    return checkpoint["model"], int(epoch) if epoch is not None else None


def validate_array(name: str, array: np.ndarray, channels: int) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 3 or array.shape[0] != channels:
        raise ValueError(f"{name} must have shape ({channels}, H, W); got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return np.ascontiguousarray(array)


def main() -> None:
    args = parse_args()
    input_path = args.input_npz.resolve()
    output_path = args.output_npz.resolve()
    device = resolve_device(args.device)

    with np.load(input_path) as payload:
        auxiliary_ms = validate_array("auxiliary_ms", payload["auxiliary_ms"], 4)
        auxiliary_hs = validate_array(
            "auxiliary_hs_interpolated", payload["auxiliary_hs_interpolated"], 151
        )
        target_ms = validate_array("target_ms", payload["target_ms"], 4)

    if auxiliary_ms.shape[1:] != auxiliary_hs.shape[1:] or auxiliary_ms.shape[1:] != target_ms.shape[1:]:
        raise ValueError("All prepared high-resolution arrays must have matching spatial dimensions")

    model = DC_STSF(ms_bands=4, hs_bands=151, out_channels=151).to(device)
    state_dict, checkpoint_epoch = load_checkpoint(args.checkpoint.resolve(), device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    ms1 = torch.from_numpy(auxiliary_ms).unsqueeze(0).to(device)
    hs1_interpolated = torch.from_numpy(auxiliary_hs).unsqueeze(0).to(device)
    ms2 = torch.from_numpy(target_ms).unsqueeze(0).to(device)
    hs1_low_resolution = functional.interpolate(
        hs1_interpolated,
        scale_factor=1 / 3,
        mode="bilinear",
        align_corners=False,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    started_at = time.perf_counter()
    with torch.inference_mode():
        predicted_hs, reconstructed_auxiliary_hs, _, _ = model(ms1, hs1_low_resolution, ms2)
    if device.type == "cuda":
        torch.cuda.synchronize()
    runtime_seconds = time.perf_counter() - started_at

    prediction = predicted_hs[0].detach().cpu().numpy().astype(np.float32, copy=False)
    reconstruction = (
        reconstructed_auxiliary_hs[0].detach().cpu().numpy().astype(np.float32, copy=False)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        predicted_hs=prediction,
        reconstructed_auxiliary_hs=reconstruction,
        runtime_seconds=np.asarray(runtime_seconds, dtype=np.float64),
        checkpoint_epoch=np.asarray(-1 if checkpoint_epoch is None else checkpoint_epoch),
        device=np.asarray(str(device)),
    )
    print(
        json.dumps(
            {
                "output_npz": str(output_path),
                "prediction_shape": list(prediction.shape),
                "runtime_seconds": runtime_seconds,
                "checkpoint_epoch": checkpoint_epoch,
                "device": str(device),
            }
        )
    )


if __name__ == "__main__":
    main()
