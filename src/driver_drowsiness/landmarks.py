"""dlib face and facial landmark helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dlib
import numpy as np

from driver_drowsiness.exceptions import ModelLoadError

LOGGER = logging.getLogger(__name__)

LEFT_EYE_INDICES = tuple(range(36, 42))
RIGHT_EYE_INDICES = tuple(range(42, 48))
MOUTH_INDICES = tuple(range(48, 68))


@dataclass(frozen=True)
class EyeLandmarks:
    left_eye: np.ndarray
    right_eye: np.ndarray


@dataclass(frozen=True)
class FacialLandmarks:
    points: np.ndarray

    @property
    def left_eye(self) -> np.ndarray:
        return self.points[list(LEFT_EYE_INDICES)]

    @property
    def right_eye(self) -> np.ndarray:
        return self.points[list(RIGHT_EYE_INDICES)]

    @property
    def mouth(self) -> np.ndarray:
        return self.points[list(MOUTH_INDICES)]

    @property
    def eye_landmarks(self) -> EyeLandmarks:
        return EyeLandmarks(left_eye=self.left_eye, right_eye=self.right_eye)


class FacialLandmarkDetector:
    """Wrapper around dlib's face detector and 68-point shape predictor."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise ModelLoadError(f"Landmark model not found: {self.model_path}")

        try:
            self.face_detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(str(self.model_path))
        except RuntimeError as exc:
            raise ModelLoadError(
                f"Unable to load landmark model: {self.model_path}"
            ) from exc
        LOGGER.info("Facial landmark model loaded.")

    def detect_faces(self, gray_frame: np.ndarray) -> list[Any]:
        return list(self.face_detector(gray_frame, 0))

    def eye_landmarks(self, gray_frame: np.ndarray, face: Any) -> EyeLandmarks:
        return self.facial_landmarks(gray_frame, face).eye_landmarks

    def facial_landmarks(self, gray_frame: np.ndarray, face: Any) -> FacialLandmarks:
        shape = self.predictor(gray_frame, face)
        return FacialLandmarks(points=_points_to_array(shape, tuple(range(68))))


def _points_to_array(shape: Any, indices: tuple[int, ...]) -> np.ndarray:
    return np.array(
        [(shape.part(index).x, shape.part(index).y) for index in indices],
        dtype=int,
    )
