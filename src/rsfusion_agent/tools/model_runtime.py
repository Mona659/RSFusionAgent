"""Subprocess adapter for the isolated YRE-151 PyTorch runtime."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field


class ModelRuntimeResult(BaseModel):
    output_npz: str
    prediction_shape: list[int]
    runtime_seconds: float = Field(ge=0)
    checkpoint_epoch: int | None
    device: str


def run_yre151_runtime(
    *,
    input_npz: str | Path,
    output_npz: str | Path,
    checkpoint_path: str | Path,
    model_python: str | Path,
    device: str = "auto",
    timeout_seconds: int = 600,
) -> ModelRuntimeResult:
    """Execute PyTorch outside the lightweight agent environment."""

    python_path = Path(model_python).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    input_path = Path(input_npz).expanduser().resolve()
    output_path = Path(output_npz).expanduser().resolve()
    for path, role in (
        (python_path, "Model Python executable"),
        (checkpoint, "Checkpoint"),
        (input_path, "Prepared model input"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{role} does not exist: {path}")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device}")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    source_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(source_root)
        if not existing_pythonpath
        else os.pathsep.join((str(source_root), existing_pythonpath))
    )
    command = [
        str(python_path),
        "-m",
        "rsfusion_agent.runtime.yre151_runner",
        "--input-npz",
        str(input_path),
        "--output-npz",
        str(output_path),
        "--checkpoint",
        str(checkpoint),
        "--device",
        device,
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Model inference exceeded {timeout_seconds} seconds") from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "No runtime output")[-5000:]
        raise RuntimeError(f"Model runtime failed with exit code {exc.returncode}:\n{details}") from exc

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Model runtime completed without returning metadata")
    try:
        result = ModelRuntimeResult.model_validate(json.loads(lines[-1]))
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"Cannot parse model runtime output:\n{completed.stdout[-5000:]}") from exc
    if not Path(result.output_npz).is_file():
        raise RuntimeError(f"Model runtime did not create its declared output: {result.output_npz}")
    return result

