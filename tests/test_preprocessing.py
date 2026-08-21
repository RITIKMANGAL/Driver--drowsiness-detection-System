import numpy as np
import pytest

from driver_drowsiness.exceptions import DetectionError
from driver_drowsiness.preprocessing import preprocess_frame, validate_frame


def test_preprocessing_preserves_aspect_ratio() -> None:
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    processed = preprocess_frame(
        frame,
        max_width=512,
        max_height=512,
        processing_scale=1.0,
    )

    assert processed.shape == (384, 512, 3)


def test_preprocessing_does_not_upscale_small_frames() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    processed = preprocess_frame(
        frame,
        max_width=512,
        max_height=512,
        processing_scale=1.0,
    )

    assert processed is frame
    assert processed.shape == (120, 160, 3)


def test_preprocessing_uses_configured_processing_scale() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    processed = preprocess_frame(
        frame,
        max_width=512,
        max_height=512,
        processing_scale=0.5,
    )

    assert processed.shape == (240, 320, 3)


@pytest.mark.parametrize(
    "frame",
    [
        None,
        np.array([], dtype=np.uint8),
        np.zeros((10,), dtype=np.uint8),
        np.zeros((10, 10, 2), dtype=np.uint8),
        np.array([[object()]], dtype=object),
    ],
)
def test_validate_frame_rejects_invalid_frames(frame) -> None:
    with pytest.raises(DetectionError):
        validate_frame(frame)


def test_validate_frame_rejects_non_finite_values() -> None:
    frame = np.zeros((10, 10), dtype=float)
    frame[0, 0] = np.inf

    with pytest.raises(DetectionError, match="finite"):
        validate_frame(frame)
