"""YRE-151 HDF5 inspection and single-patch preparation tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from pydantic import BaseModel, Field

MS_BANDS = 4
HS_BANDS = 151
TEST_CHANNELS = MS_BANDS + HS_BANDS + HS_BANDS
NORM_FACTOR = 10_000.0
SCALE = 3


class H5FileInspection(BaseModel):
    """Metadata for one legacy YRE HDF5 file."""

    path: str
    split: str
    shape: tuple[int, int, int, int]
    dtype: str
    patch_count: int = Field(gt=0)
    patch_height: int = Field(gt=0)
    patch_width: int = Field(gt=0)
    channels: int = Field(gt=0)
    sample_min: float
    sample_max: float


class H5PairInspection(BaseModel):
    """Validated auxiliary/target HDF5 pair."""

    profile: str = "yre151_reduce_v1"
    auxiliary: H5FileInspection
    target: H5FileInspection
    patch_index: int = Field(ge=0)
    ms_bands: int = MS_BANDS
    hs_bands: int = HS_BANDS
    scale: int = SCALE
    norm_factor: float = NORM_FACTOR
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class PreparedPatch:
    """Framework-neutral CHW arrays consumed by the PyTorch runtime."""

    auxiliary_ms: np.ndarray
    auxiliary_hs_interpolated: np.ndarray
    target_ms: np.ndarray
    auxiliary_hs_gt: np.ndarray
    target_hs_gt: np.ndarray
    inspection: H5PairInspection

    def save_npz(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            auxiliary_ms=self.auxiliary_ms,
            auxiliary_hs_interpolated=self.auxiliary_hs_interpolated,
            target_ms=self.target_ms,
            auxiliary_hs_gt=self.auxiliary_hs_gt,
            target_hs_gt=self.target_hs_gt,
        )
        return output


def _validate_path(path: str | Path, role: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{role} HDF5 file does not exist: {resolved}")
    if resolved.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError(f"{role} input must be an .h5 or .hdf5 file: {resolved}")
    return resolved


def _inspect_one(path: Path, split: str, patch_index: int, role: str) -> H5FileInspection:
    try:
        with h5py.File(path, "r") as handle:
            if split not in handle:
                raise ValueError(
                    f"{role} HDF5 is missing split '{split}'. Available keys: {list(handle.keys())}"
                )
            dataset = handle[split]
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError(f"{role} key '{split}' is not an HDF5 dataset")
            if dataset.ndim != 4:
                raise ValueError(
                    f"{role} '{split}' must be NHWC with 4 dimensions; got {dataset.shape}"
                )

            patch_count, height, width, channels = (int(value) for value in dataset.shape)
            if channels != TEST_CHANNELS:
                raise ValueError(
                    f"{role} '{split}' must have {TEST_CHANNELS} channels "
                    f"(4 MS + 151 HS + 151 GT); got {channels}"
                )
            if patch_index >= patch_count:
                raise IndexError(
                    f"Patch index {patch_index} is out of range for {role}; "
                    f"valid range is 0..{patch_count - 1}"
                )
            if height % SCALE or width % SCALE:
                raise ValueError(
                    f"{role} patch size {height}x{width} must be divisible by scale {SCALE}"
                )
            if not np.issubdtype(dataset.dtype, np.number):
                raise TypeError(f"{role} dataset must be numeric; got {dataset.dtype}")

            sample = np.asarray(dataset[patch_index])
            if not np.isfinite(sample).all():
                raise ValueError(f"{role} patch {patch_index} contains NaN or Inf")
            return H5FileInspection(
                path=str(path),
                split=split,
                shape=(patch_count, height, width, channels),
                dtype=str(dataset.dtype),
                patch_count=patch_count,
                patch_height=height,
                patch_width=width,
                channels=channels,
                sample_min=float(sample.min()),
                sample_max=float(sample.max()),
            )
    except OSError as exc:
        raise ValueError(f"Cannot open {role} HDF5 '{path}': {exc}") from exc


def inspect_h5_pair(
    auxiliary_h5_path: str | Path,
    target_h5_path: str | Path,
    *,
    split: str = "test",
    patch_index: int = 0,
) -> H5PairInspection:
    """Validate a legacy YRE-151 auxiliary/target HDF5 pair."""

    if patch_index < 0:
        raise ValueError("patch_index must be non-negative")
    auxiliary_path = _validate_path(auxiliary_h5_path, "Auxiliary")
    target_path = _validate_path(target_h5_path, "Target")
    auxiliary = _inspect_one(auxiliary_path, split, patch_index, "Auxiliary")
    target = _inspect_one(target_path, split, patch_index, "Target")

    if auxiliary.shape[:3] != target.shape[:3]:
        raise ValueError(
            "Auxiliary and target datasets must have matching patch count and spatial size; "
            f"got {auxiliary.shape} and {target.shape}"
        )

    warnings: list[str] = []
    if auxiliary.dtype != "int16" or target.dtype != "int16":
        warnings.append(
            "The frozen YRE-151 profile was validated with int16 HDF5 data; "
            f"received auxiliary={auxiliary.dtype}, target={target.dtype}."
        )
    return H5PairInspection(
        auxiliary=auxiliary,
        target=target,
        patch_index=patch_index,
        warnings=warnings,
    )


def prepare_yre151_patch(
    auxiliary_h5_path: str | Path,
    target_h5_path: str | Path,
    *,
    split: str = "test",
    patch_index: int = 0,
) -> PreparedPatch:
    """Load one patch and reproduce the legacy channel slicing and normalization."""

    inspection = inspect_h5_pair(
        auxiliary_h5_path,
        target_h5_path,
        split=split,
        patch_index=patch_index,
    )
    with h5py.File(inspection.auxiliary.path, "r") as auxiliary_handle:
        auxiliary_nhwc = np.asarray(
            auxiliary_handle[split][patch_index], dtype=np.float32
        )
    with h5py.File(inspection.target.path, "r") as target_handle:
        target_nhwc = np.asarray(target_handle[split][patch_index], dtype=np.float32)

    auxiliary_chw = np.ascontiguousarray(np.moveaxis(auxiliary_nhwc, -1, 0))
    target_chw = np.ascontiguousarray(np.moveaxis(target_nhwc, -1, 0))

    def normalized(array: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(array / np.float32(NORM_FACTOR), dtype=np.float32)

    return PreparedPatch(
        auxiliary_ms=normalized(auxiliary_chw[:MS_BANDS]),
        auxiliary_hs_interpolated=normalized(auxiliary_chw[MS_BANDS : MS_BANDS + HS_BANDS]),
        target_ms=normalized(target_chw[:MS_BANDS]),
        auxiliary_hs_gt=normalized(auxiliary_chw[MS_BANDS + HS_BANDS : TEST_CHANNELS]),
        target_hs_gt=normalized(target_chw[MS_BANDS + HS_BANDS : TEST_CHANNELS]),
        inspection=inspection,
    )

