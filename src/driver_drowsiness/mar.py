"""Mouth Aspect Ratio calculations."""

from __future__ import annotations

import numpy as np


def mouth_aspect_ratio(mouth: np.ndarray) -> float:
    """Calculate MAR for the 20 mouth landmark points, indices 48-67."""

    points = np.asarray(mouth, dtype=float)
    if points.shape != (20, 2):
        msg = "MAR calculation requires a 20x2 array of mouth landmarks."
        raise ValueError(msg)
    if not np.isfinite(points).all():
        msg = "Mouth landmarks must contain only finite numeric coordinates."
        raise ValueError(msg)

    vertical_1 = _euclidean_distance(points[2], points[10])
    vertical_2 = _euclidean_distance(points[3], points[9])
    vertical_3 = _euclidean_distance(points[4], points[8])
    horizontal = _euclidean_distance(points[0], points[6])
    if horizontal == 0:
        msg = "Horizontal mouth landmark distance must be non-zero."
        raise ValueError(msg)

    return float((vertical_1 + vertical_2 + vertical_3) / (3.0 * horizontal))


def _euclidean_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(first - second))
