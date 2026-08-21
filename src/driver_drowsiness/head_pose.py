"""Simple 2D head-pose heuristics from 68-point facial landmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HeadPoseMetrics:
    roll_angle_degrees: float
    downward_ratio: float


LEFT_EYE = tuple(range(36, 42))
RIGHT_EYE = tuple(range(42, 48))
NOSE_TIP = 33
CHIN = 8


def estimate_head_pose_metrics(landmarks: np.ndarray) -> HeadPoseMetrics:
    """Estimate simple roll and downward metrics from 68 facial landmarks."""

    points = np.asarray(landmarks, dtype=float)
    if points.shape != (68, 2):
        msg = "Head-pose heuristic requires a 68x2 array of facial landmarks."
        raise ValueError(msg)
    if not np.isfinite(points).all():
        msg = "Facial landmarks must contain only finite numeric coordinates."
        raise ValueError(msg)

    left_eye_center = points[list(LEFT_EYE)].mean(axis=0)
    right_eye_center = points[list(RIGHT_EYE)].mean(axis=0)
    eye_delta = right_eye_center - left_eye_center
    eye_distance = float(np.linalg.norm(eye_delta))
    if eye_distance == 0:
        msg = "Eye-center distance must be non-zero."
        raise ValueError(msg)

    roll_angle = float(np.degrees(np.arctan2(eye_delta[1], eye_delta[0])))
    eye_midpoint = (left_eye_center + right_eye_center) / 2.0
    face_height = float(np.linalg.norm(points[CHIN] - eye_midpoint))
    if face_height == 0:
        msg = "Face height proxy must be non-zero."
        raise ValueError(msg)

    downward_ratio = float((points[NOSE_TIP][1] - eye_midpoint[1]) / face_height)
    return HeadPoseMetrics(
        roll_angle_degrees=roll_angle,
        downward_ratio=downward_ratio,
    )


def is_head_tilted(
    metrics: HeadPoseMetrics,
    *,
    roll_threshold_degrees: float,
    downward_ratio_threshold: float,
) -> bool:
    return (
        abs(metrics.roll_angle_degrees) >= roll_threshold_degrees
        or metrics.downward_ratio >= downward_ratio_threshold
    )
