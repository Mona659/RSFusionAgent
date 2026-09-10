import numpy as np

from rsfusion_agent.tools.artifacts import (
    hyperspectral_rgb,
    hyperspectral_rgb_stretch_bounds,
)


def test_shared_rgb_stretch_keeps_prediction_and_reference_comparable() -> None:
    prediction = np.full((3, 4, 4), 0.25, dtype=np.float32)
    reference = np.full((3, 4, 4), 0.75, dtype=np.float32)
    bounds = hyperspectral_rgb_stretch_bounds(prediction, reference, bands=(0, 1, 2))

    prediction_rgb = hyperspectral_rgb(
        prediction,
        bands=(0, 1, 2),
        stretch_bounds=bounds,
    )
    reference_rgb = hyperspectral_rgb(
        reference,
        bands=(0, 1, 2),
        stretch_bounds=bounds,
    )

    assert np.all(prediction_rgb == 0)
    assert np.all(reference_rgb == 255)
