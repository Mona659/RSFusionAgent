from pathlib import Path

import h5py
import numpy as np
import rasterio

from rsfusion_agent.agent.state import FusionRunRequest
from rsfusion_agent.agent.workflow import YRE151PatchAgent
from rsfusion_agent.tools.model_runtime import ModelRuntimeResult


def _create_h5(path: Path, target: np.ndarray) -> None:
    test = np.zeros((1, target.shape[1], target.shape[2], 306), dtype=np.float32)
    test[0, ..., :4] = 1000.0
    test[0, ..., 4:155] = np.moveaxis(target * 10_000.0, 0, -1)
    test[0, ..., 155:306] = np.moveaxis(target * 10_000.0, 0, -1)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("test", data=test)


def test_agent_workflow_writes_traceable_artifacts(tmp_path: Path) -> None:
    target = np.linspace(0.1, 0.9, 151 * 12 * 12, dtype=np.float32).reshape(151, 12, 12)
    auxiliary_h5 = tmp_path / "aux.h5"
    target_h5 = tmp_path / "target.h5"
    checkpoint = tmp_path / "checkpoint.pth"
    model_python = tmp_path / "python.exe"
    checkpoint.touch()
    model_python.touch()
    _create_h5(auxiliary_h5, target)
    _create_h5(target_h5, target)

    def fake_runtime(**kwargs: object) -> ModelRuntimeResult:
        input_npz = Path(str(kwargs["input_npz"]))
        output_npz = Path(str(kwargs["output_npz"]))
        with np.load(input_npz) as payload:
            prediction = payload["target_hs_gt"]
        np.savez_compressed(output_npz, predicted_hs=prediction)
        return ModelRuntimeResult(
            output_npz=str(output_npz),
            prediction_shape=list(prediction.shape),
            runtime_seconds=0.01,
            checkpoint_epoch=200,
            device="cpu",
        )

    result = YRE151PatchAgent(runtime_runner=fake_runtime).run(
        FusionRunRequest(
            auxiliary_h5_path=auxiliary_h5,
            target_h5_path=target_h5,
            checkpoint_path=checkpoint,
            model_python=model_python,
            output_dir=tmp_path / "outputs",
        )
    )

    assert result.status == "completed"
    assert result.metrics.rmse == 0.0
    assert all(step.status == "completed" for step in result.trace)
    assert Path(result.manifest_path).is_file()
    assert Path(result.report_path).is_file()
    with rasterio.open(result.predicted_hs_path) as dataset:
        assert (dataset.count, dataset.height, dataset.width) == (151, 12, 12)
        assert dataset.dtypes == ("float32",) * 151

