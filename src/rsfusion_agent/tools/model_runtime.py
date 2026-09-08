"""Subprocess adapter for the isolated YRE-151 PyTorch runtime."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field


class ModelRuntimeResult(BaseModel):
    output_npz: str
    prediction_shape: list[int]
    runtime_seconds: float = Field(ge=0)
    checkpoint_epoch: int | None
    device: str
    attempt_count: int = Field(default=1, ge=1)
    retried_exit_codes: list[int] = Field(default_factory=list)


class RuntimePreflightResult(BaseModel):
    """Non-secret health information from the isolated model environment."""

    status: str = "ready"
    python_executable: str
    python_version: str
    torch_version: str
    requested_device: str
    resolved_device: str
    cuda_available: bool
    cuda_device_name: str | None = None
    cuda_total_memory_bytes: int | None = Field(default=None, ge=0)
    checkpoint_epoch: int | None = None
    checkpoint_tensor_count: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class RuntimeWorkerResponse:
    payload: dict[str, object]
    attempt_count: int
    retried_exit_codes: list[int]


_RETRYABLE_NATIVE_EXIT_CODES = frozenset({3221226505})  # Windows STATUS_STACK_BUFFER_OVERRUN


def _runtime_environment(source_root: Path) -> dict[str, str]:
    """Prepare the minimum import path needed by the isolated model interpreter."""

    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(source_root)
        if not existing_pythonpath
        else os.pathsep.join((str(source_root), existing_pythonpath))
    )
    return environment


def _run_runtime_module(
    *,
    python_path: Path,
    module: str,
    arguments: list[str],
    timeout_seconds: int,
    operation: str,
    retry_count: int = 0,
    retryable_exit_codes: frozenset[int] = frozenset(),
) -> RuntimeWorkerResponse:
    """Run a JSON-producing worker in the configured PyTorch environment."""

    source_root = Path(__file__).resolve().parents[2]
    command = [str(python_path), "-m", module, *arguments]
    if retry_count < 0:
        raise ValueError("retry_count must be non-negative")
    retried_exit_codes: list[int] = []
    for attempt_count in range(1, retry_count + 2):
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=_runtime_environment(source_root),
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Model {operation} exceeded {timeout_seconds} seconds") from exc
        except subprocess.CalledProcessError as exc:
            if exc.returncode in retryable_exit_codes and attempt_count <= retry_count:
                retried_exit_codes.append(exc.returncode)
                time.sleep(1.0)
                continue
            details = (exc.stderr or exc.stdout or "No runtime output")[-5000:]
            attempts_note = f" after {attempt_count} attempts" if attempt_count > 1 else ""
            raise RuntimeError(
                f"Model runtime failed with exit code {exc.returncode}{attempts_note}:\n{details}"
            ) from exc
        break
    else:  # pragma: no cover - the loop returns or raises on every path.
        raise RuntimeError("Model runtime did not start")

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Model runtime completed without returning metadata")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Cannot parse model runtime output:\n{completed.stdout[-5000:]}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Model runtime metadata must be a JSON object")
    return RuntimeWorkerResponse(
        payload=payload,
        attempt_count=attempt_count,
        retried_exit_codes=retried_exit_codes,
    )


def preflight_yre151_runtime(
    *,
    checkpoint_path: str | Path,
    model_python: str | Path,
    device: str = "auto",
    timeout_seconds: int = 60,
) -> RuntimePreflightResult:
    """Check the configured PyTorch environment before a paid agent request.

    This verifies imports, device visibility and checkpoint readability. It does not
    run a model forward pass or read HDF5 pixels.
    """

    python_path = Path(model_python).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    for path, role in ((python_path, "Model Python executable"), (checkpoint, "Checkpoint")):
        if not path.is_file():
            raise FileNotFoundError(f"{role} does not exist: {path}")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device}")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    response = _run_runtime_module(
        python_path=python_path,
        module="rsfusion_agent.runtime.preflight_runner",
        arguments=["--checkpoint", str(checkpoint), "--device", device],
        timeout_seconds=timeout_seconds,
        operation="preflight",
    )
    try:
        return RuntimePreflightResult.model_validate(response.payload)
    except ValueError as exc:
        raise RuntimeError(f"Cannot parse model preflight output: {response.payload!r}") from exc


def run_yre151_runtime(
    *,
    input_npz: str | Path,
    output_npz: str | Path,
    checkpoint_path: str | Path,
    model_python: str | Path,
    device: str = "auto",
    timeout_seconds: int = 600,
    retry_count: int = 1,
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
    if retry_count < 0 or retry_count > 2:
        raise ValueError("retry_count must be between 0 and 2")

    response = _run_runtime_module(
        python_path=python_path,
        module="rsfusion_agent.runtime.yre151_runner",
        arguments=[
            "--input-npz",
            str(input_path),
            "--output-npz",
            str(output_path),
            "--checkpoint",
            str(checkpoint),
            "--device",
            device,
        ],
        timeout_seconds=timeout_seconds,
        operation="inference",
        retry_count=retry_count,
        retryable_exit_codes=_RETRYABLE_NATIVE_EXIT_CODES,
    )
    try:
        result = ModelRuntimeResult.model_validate(
            {
                **response.payload,
                "attempt_count": response.attempt_count,
                "retried_exit_codes": response.retried_exit_codes,
            }
        )
    except ValueError as exc:
        raise RuntimeError(f"Cannot parse model runtime output: {response.payload!r}") from exc
    if not Path(result.output_npz).is_file():
        raise RuntimeError(f"Model runtime did not create its declared output: {result.output_npz}")
    return result
