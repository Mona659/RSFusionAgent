import numpy as np
import pytest

from rsfusion_agent.tools.metrics import calculate_metrics, calculate_sam_map


def test_metrics_are_perfect_for_identical_images() -> None:
    target = np.linspace(0.1, 0.9, 3 * 12 * 12, dtype=np.float32).reshape(3, 12, 12)
    metrics = calculate_metrics(target, target)

    assert np.isinf(metrics.psnr)
    assert metrics.rmse == pytest.approx(0.0)
    assert metrics.sam == pytest.approx(0.0, abs=1e-6)
    assert metrics.ergas == pytest.approx(0.0)
    assert metrics.ssim == pytest.approx(1.0)
    assert metrics.cc == pytest.approx(1.0)
    assert calculate_sam_map(target, target).max() == pytest.approx(0.0, abs=1e-6)


def test_metrics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        calculate_metrics(np.zeros((3, 8, 8)), np.zeros((3, 7, 8)))

