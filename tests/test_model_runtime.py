import json
import subprocess
from pathlib import Path

import pytest

from rsfusion_agent.tools import model_runtime


def test_preflight_runs_worker_and_normalizes_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_python = tmp_path / "python.exe"
    checkpoint = tmp_path / "checkpoint.pth"
    model_python.touch()
    checkpoint.touch()
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["kwargs"] = kwargs
        payload = {
            "status": "ready",
            "python_executable": str(model_python),
            "python_version": "3.10.18",
            "torch_version": "2.5.1+cu121",
            "requested_device": "cuda",
            "resolved_device": "cuda",
            "cuda_available": True,
            "cuda_device_name": "Synthetic GPU",
            "cuda_total_memory_bytes": 8_000_000_000,
            "checkpoint_epoch": 200,
            "checkpoint_tensor_count": 42,
            "warnings": [],
        }
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(model_runtime.subprocess, "run", fake_run)

    result = model_runtime.preflight_yre151_runtime(
        checkpoint_path=checkpoint,
        model_python=model_python,
        device="cuda",
    )

    assert result.status == "ready"
    assert result.cuda_device_name == "Synthetic GPU"
    assert result.checkpoint_epoch == 200
    assert captured["command"][2] == "rsfusion_agent.runtime.preflight_runner"


def test_preflight_rejects_missing_checkpoint(tmp_path: Path) -> None:
    model_python = tmp_path / "python.exe"
    model_python.touch()

    with pytest.raises(FileNotFoundError, match="Checkpoint does not exist"):
        model_runtime.preflight_yre151_runtime(
            checkpoint_path=tmp_path / "missing.pth",
            model_python=model_python,
        )


def test_runtime_retries_known_native_exit_code_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_python = tmp_path / "python.exe"
    checkpoint = tmp_path / "checkpoint.pth"
    input_npz = tmp_path / "input.npz"
    output_npz = tmp_path / "output.npz"
    for path in (model_python, checkpoint, input_npz, output_npz):
        path.touch()
    calls = 0

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise subprocess.CalledProcessError(
                3221226505,
                command,
                output="",
                stderr="native process ended",
            )
        payload = {
            "output_npz": str(output_npz),
            "prediction_shape": [151, 180, 180],
            "runtime_seconds": 0.4,
            "checkpoint_epoch": 200,
            "device": "cuda",
        }
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(model_runtime.subprocess, "run", fake_run)
    monkeypatch.setattr(model_runtime.time, "sleep", lambda _: None)

    result = model_runtime.run_yre151_runtime(
        input_npz=input_npz,
        output_npz=output_npz,
        checkpoint_path=checkpoint,
        model_python=model_python,
        device="cuda",
        retry_count=1,
    )

    assert calls == 2
    assert result.attempt_count == 2
    assert result.retried_exit_codes == [3221226505]
