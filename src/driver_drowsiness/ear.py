"""Eye Aspect Ratio calculations."""

from __future__ import annotations

import numpy as np


def eye_aspect_ratio(eye: np.ndarray) -> float:
    """Calculate EAR for six eye landmark points."""

    points = np.asarray(eye, dtype=float)
    if points.shape != (6, 2):
        msg = "EAR calculation requires a 6x2 array of eye landmarks."
        raise ValueError(msg)
    if not np.isfinite(points).all():
        msg = "Eye landmarks must contain only finite numeric coordinates."
        raise ValueError(msg)

    vertical_1 = _euclidean_distance(points[1], points[5])
    vertical_2 = _euclidean_distance(points[2], points[4])
    horizontal = _euclidean_distance(points[0], points[3])
    if horizontal == 0:
        msg = "Horizontal eye landmark distance must be non-zero."
        raise ValueError(msg)

    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def average_eye_aspect_ratio(left_eye: np.ndarray, right_eye: np.ndarray) -> float:
    """Calculate the average EAR for left and right eyes."""

    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    return (left_ear + right_ear) / 2.0


def _euclidean_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(first - second))
