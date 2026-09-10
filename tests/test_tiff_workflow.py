from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from rsfusion_agent.agent.tiff_workflow import TiffFusionRequest, YRE151TiffPatchAgent
from rsfusion_agent.tools.model_runtime import ModelRuntimeResult
from rsfusion_agent.tools.tiff_crop import CUSTOM_PROFILE, crop_tiff_triplet
from rsfusion_agent.tools.tiff_patch import prepare_tiff_patch


def _write_raster(
    path: Path,
    *,
    bands: int,
    width: int,
    height: int,
    resolution: float,
    value: float,
) -> None:
    pixels = np.full((bands, height, width), value, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=bands,
        dtype=pixels.dtype,
        crs="EPSG:4326",
        transform=from_origin(120.0, 30.0, resolution, resolution),
    ) as dataset:
        dataset.write(pixels)


def _create_triplet(tmp_path: Path) -> tuple[Path, Path, Path]:
    auxiliary_ms = tmp_path / "aux_ms.tif"
    auxiliary_hs = tmp_path / "aux_hs.tif"
    target_ms = tmp_path / "target_ms.tif"
    _write_raster(auxiliary_ms, bands=4, width=12, height=12, resolution=0.5, value=2000.0)
    _write_raster(auxiliary_hs, bands=151, width=4, height=4, resolution=1.5, value=3000.0)
    _write_raster(target_ms, bands=4, width=12, height=12, resolution=0.5, value=4000.0)
    return auxiliary_ms, auxiliary_hs, target_ms


def test_prepare_tiff_patch_normalizes_and_preserves_window_georeferencing(tmp_path: Path) -> None:
    auxiliary_ms, auxiliary_hs, target_ms = _create_triplet(tmp_path)

    prepared = prepare_tiff_patch(
        auxiliary_ms,
        auxiliary_hs,
        target_ms,
        patch_size=6,
        row_offset=3,
        col_offset=6,
    )

    assert prepared.auxiliary_ms.shape == (4, 6, 6)
    assert prepared.auxiliary_hs_interpolated.shape == (151, 6, 6)
    assert prepared.target_ms.shape == (4, 6, 6)
    assert np.allclose(prepared.auxiliary_ms, 0.2)
    assert np.allclose(prepared.auxiliary_hs_interpolated, 0.3)
    assert np.allclose(prepared.target_ms, 0.4)
    assert prepared.spatial_metadata.transform == (0.5, 0.0, 123.0, 0.0, -0.5, 28.5)


def test_tiff_workflow_writes_georeferenced_prediction_without_metrics(tmp_path: Path) -> None:
    auxiliary_ms, auxiliary_hs, target_ms = _create_triplet(tmp_path)
    checkpoint = tmp_path / "checkpoint.pth"
    model_python = tmp_path / "python.exe"
    checkpoint.touch()
    model_python.touch()

    def fake_runtime(**kwargs: object) -> ModelRuntimeResult:
        output_npz = Path(str(kwargs["output_npz"]))
        prediction = np.full((151, 6, 6), 0.5, dtype=np.float32)
        np.savez_compressed(output_npz, predicted_hs=prediction)
        return ModelRuntimeResult(
            output_npz=str(output_npz),
            prediction_shape=list(prediction.shape),
            runtime_seconds=0.01,
            checkpoint_epoch=200,
            device="cpu",
        )

    result = YRE151TiffPatchAgent(runtime_runner=fake_runtime).run(
        TiffFusionRequest(
            auxiliary_ms_path=auxiliary_ms,
            auxiliary_hs_path=auxiliary_hs,
            target_ms_path=target_ms,
            checkpoint_path=checkpoint,
            model_python=model_python,
            output_dir=tmp_path / "outputs",
            patch_size=6,
            row_offset=3,
            col_offset=6,
        )
    )

    assert result.metrics_status == "unavailable_without_target_hs_reference"
    assert result.reference_rgb_preview_path is None
    assert all(step.status == "completed" for step in result.trace)
    with rasterio.open(result.predicted_hs_path) as dataset:
        assert (dataset.count, dataset.height, dataset.width) == (151, 6, 6)
        assert dataset.crs.to_string() == "EPSG:4326"
        assert tuple(dataset.transform)[:6] == (0.5, 0.0, 123.0, 0.0, -0.5, 28.5)


def test_tiff_workflow_accepts_explicit_crop_manifest(tmp_path: Path) -> None:
    auxiliary_ms, auxiliary_hs, target_ms = _create_triplet(tmp_path)
    crop = crop_tiff_triplet(
        auxiliary_ms,
        auxiliary_hs,
        target_ms,
        tmp_path / "crops",
        profile=CUSTOM_PROFILE,
        ms_row_offset=0,
        ms_col_offset=0,
        window_height=6,
        window_width=6,
        hs_row_offset=0,
        hs_col_offset=0,
    )

    def fake_runtime(**kwargs: object) -> ModelRuntimeResult:
        output_npz = Path(str(kwargs["output_npz"]))
        prediction = np.full((151, 6, 6), 0.5, dtype=np.float32)
        np.savez_compressed(output_npz, predicted_hs=prediction)
        return ModelRuntimeResult(
            output_npz=str(output_npz),
            prediction_shape=list(prediction.shape),
            runtime_seconds=0.01,
            checkpoint_epoch=200,
            device="cpu",
        )

    result = YRE151TiffPatchAgent(runtime_runner=fake_runtime).run(
        TiffFusionRequest(
            crop_manifest_path=Path(crop.manifest_path),
            checkpoint_path=tmp_path / "checkpoint.pth",
            model_python=tmp_path / "python.exe",
            output_dir=tmp_path / "outputs",
            patch_size=6,
        )
    )

    assert result.spatial_metadata.alignment_mode == "explicit_crop_manifest"
    assert "crop manifest authorizes" in " ".join(result.warnings)


def test_tiff_workflow_calculates_legacy_pseudo_reference_metrics(tmp_path: Path) -> None:
    auxiliary_ms, auxiliary_hs, target_ms = _create_triplet(tmp_path)
    target_hs_reference = tmp_path / "target_hs_reference.tif"
    _write_raster(target_hs_reference, bands=151, width=4, height=4, resolution=1.5, value=6000.0)

    def fake_runtime(**kwargs: object) -> ModelRuntimeResult:
        output_npz = Path(str(kwargs["output_npz"]))
        prediction = np.full((151, 9, 9), 0.5, dtype=np.float32)
        np.savez_compressed(output_npz, predicted_hs=prediction)
        return ModelRuntimeResult(
            output_npz=str(output_npz),
            prediction_shape=list(prediction.shape),
            runtime_seconds=0.01,
            checkpoint_epoch=200,
            device="cpu",
        )

    result = YRE151TiffPatchAgent(runtime_runner=fake_runtime).run(
        TiffFusionRequest(
            auxiliary_ms_path=auxiliary_ms,
            auxiliary_hs_path=auxiliary_hs,
            target_ms_path=target_ms,
            target_hs_reference_path=target_hs_reference,
            checkpoint_path=tmp_path / "checkpoint.pth",
            model_python=tmp_path / "python.exe",
            output_dir=tmp_path / "outputs",
            patch_size=9,
            row_offset=0,
            col_offset=0,
        )
    )

    assert result.metrics_status == "available_legacy_interpolated_target_hs_reference"
    assert result.metrics is not None
    assert result.metrics_path is not None
    assert result.sam_heatmap_path is not None
    assert result.reference_rgb_preview_path is not None
    assert Path(result.metrics_path).is_file()
    assert Path(result.sam_heatmap_path).is_file()
    assert Path(result.reference_rgb_preview_path).is_file()
