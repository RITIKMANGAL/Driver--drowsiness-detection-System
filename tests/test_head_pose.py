import numpy as np
import pytest

from driver_drowsiness.head_pose import (
    estimate_head_pose_metrics,
    is_head_tilted,
)


def test_head_pose_normal_is_not_tilted() -> None:
    metrics = estimate_head_pose_metrics(_landmarks())

    assert metrics.roll_angle_degrees == pytest.approx(0.0)
    assert metrics.downward_ratio == pytest.approx(0.4)
    assert not is_head_tilted(
        metrics,
        roll_threshold_degrees=20.0,
        downward_ratio_threshold=0.65,
    )


def test_head_pose_roll_tilt_is_detected() -> None:
    metrics = estimate_head_pose_metrics(_landmarks(right_eye_y=4.0))

    assert is_head_tilted(
        metrics,
        roll_threshold_degrees=20.0,
        downward_ratio_threshold=0.65,
    )


def test_head_pose_downward_orientation_is_detected() -> None:
    metrics = estimate_head_pose_metrics(_landmarks(nose_y=14.0))

    assert is_head_tilted(
        metrics,
        roll_threshold_degrees=20.0,
        downward_ratio_threshold=0.65,
    )


def test_head_pose_rejects_invalid_landmarks() -> None:
    with pytest.raises(ValueError, match="68x2"):
        estimate_head_pose_metrics(np.zeros((4, 2), dtype=float))


def test_head_pose_rejects_insufficient_geometry() -> None:
    landmarks = _landmarks()
    landmarks[42:48] = landmarks[36:42]

    with pytest.raises(ValueError, match="non-zero"):
        estimate_head_pose_metrics(landmarks)


def _landmarks(*, right_eye_y: float = 0.0, nose_y: float = 8.0) -> np.ndarray:
    landmarks = np.zeros((68, 2), dtype=float)
    landmarks[36:42] = _eye(x_offset=0.0, y_offset=0.0)
    landmarks[42:48] = _eye(x_offset=10.0, y_offset=right_eye_y)
    landmarks[33] = [6.6666666667, nose_y]
    landmarks[8] = [6.6666666667, 20.0]
    return landmarks


def _eye(*, x_offset: float, y_offset: float) -> np.ndarray:
    return np.array(
        [
            [0, 0],
            [1, 2],
            [2, 2],
            [4, 0],
            [2, -2],
            [1, -2],
        ],
        dtype=float,
    ) + np.array([x_offset, y_offset], dtype=float)
