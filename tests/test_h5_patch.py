from pathlib import Path

import h5py
import numpy as np
import pytest

from rsfusion_agent.tools.h5_patch import inspect_h5_pair, prepare_yre151_patch


def _create_h5(path: Path, offset: int = 0) -> None:
    data = np.arange(2 * 12 * 15 * 306, dtype=np.int32).reshape(2, 12, 15, 306)
    data = ((data + offset) % 5000).astype(np.int16)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("test", data=data)
        handle.create_dataset("train", data=data[..., :155])
        handle.create_dataset("label", data=data[..., 155:306])


def test_prepare_yre151_patch_slices_and_normalizes(tmp_path: Path) -> None:
    auxiliary_path = tmp_path / "aux.h5"
    target_path = tmp_path / "target.h5"
    _create_h5(auxiliary_path)
    _create_h5(target_path, offset=10)

    prepared = prepare_yre151_patch(auxiliary_path, target_path, patch_index=1)

    assert prepared.auxiliary_ms.shape == (4, 12, 15)
    assert prepared.auxiliary_hs_interpolated.shape == (151, 12, 15)
    assert prepared.target_ms.shape == (4, 12, 15)
    assert prepared.target_hs_gt.shape == (151, 12, 15)

    with h5py.File(auxiliary_path, "r") as handle:
        expected = handle["test"][1, 0, 0, 0] / 10_000.0
    assert prepared.auxiliary_ms[0, 0, 0] == pytest.approx(expected)


def test_inspect_h5_pair_rejects_wrong_channel_count(tmp_path: Path) -> None:
    auxiliary_path = tmp_path / "aux.h5"
    target_path = tmp_path / "target.h5"
    wrong = np.zeros((1, 12, 12, 305), dtype=np.int16)
    for path in (auxiliary_path, target_path):
        with h5py.File(path, "w") as handle:
            handle.create_dataset("test", data=wrong)

    with pytest.raises(ValueError, match="must have 306 channels"):
        inspect_h5_pair(auxiliary_path, target_path)


def test_inspect_h5_pair_rejects_invalid_patch_index(tmp_path: Path) -> None:
    auxiliary_path = tmp_path / "aux.h5"
    target_path = tmp_path / "target.h5"
    _create_h5(auxiliary_path)
    _create_h5(target_path)

    with pytest.raises(IndexError, match="out of range"):
        inspect_h5_pair(auxiliary_path, target_path, patch_index=2)

