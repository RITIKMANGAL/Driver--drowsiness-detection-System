"""Frame-level drowsiness detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from driver_drowsiness.ear import eye_aspect_ratio
from driver_drowsiness.exceptions import DetectionError
from driver_drowsiness.head_pose import (
    HeadPoseMetrics,
    estimate_head_pose_metrics,
    is_head_tilted,
)
from driver_drowsiness.landmarks import EyeLandmarks, FacialLandmarkDetector
from driver_drowsiness.mar import mouth_aspect_ratio
from driver_drowsiness.preprocessing import validate_frame
from driver_drowsiness.state import (
    ConsecutiveSignalTracker,
    DrowsinessState,
    DrowsinessTracker,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectionResult:
    face_detected: bool
    face_count: int
    primary_face_selected: bool
    left_ear: float | None
    right_ear: float | None
    average_ear: float | None
    eyes_closed: bool
    consecutive_closed_frames: int
    drowsiness_state: DrowsinessState
    alert_required: bool
    mar: float | None = None
    yawning_detected: bool = False
    consecutive_yawning_frames: int = 0
    head_tilt_detected: bool = False
    consecutive_head_tilt_frames: int = 0
    head_tilt_angle: float | None = None
    head_downward_ratio: float | None = None
    eye_landmarks: EyeLandmarks | None = None
    mouth_landmarks: np.ndarray | None = None
    primary_face_bounds: tuple[int, int, int, int] | None = None


class DrowsinessDetector:
    """Combines face detection, landmarks, EAR, and temporal state."""

    def __init__(
        self,
        landmark_detector: FacialLandmarkDetector,
        tracker: DrowsinessTracker,
        *,
        yawning_tracker: ConsecutiveSignalTracker | None = None,
        head_tilt_tracker: ConsecutiveSignalTracker | None = None,
        yawning_enabled: bool = True,
        mar_threshold: float = 0.6,
        head_pose_enabled: bool = True,
        head_roll_threshold_degrees: float = 20.0,
        head_downward_ratio_threshold: float = 0.65,
    ) -> None:
        self.landmark_detector = landmark_detector
        self.tracker = tracker
        self.yawning_tracker = yawning_tracker or ConsecutiveSignalTracker(15)
        self.head_tilt_tracker = head_tilt_tracker or ConsecutiveSignalTracker(15)
        self.yawning_enabled = yawning_enabled
        self.mar_threshold = mar_threshold
        self.head_pose_enabled = head_pose_enabled
        self.head_roll_threshold_degrees = head_roll_threshold_degrees
        self.head_downward_ratio_threshold = head_downward_ratio_threshold
        self._last_drowsiness_state = DrowsinessState.ATTENTIVE

    def process(self, frame: np.ndarray) -> DetectionResult:
        frame = validate_frame(frame)
        gray_frame = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _valid_faces(self.landmark_detector.detect_faces(gray_frame))

        if not faces:
            yawning = self.yawning_tracker.update(None)
            head_tilt = self.head_tilt_tracker.update(None)
            state = self.tracker.update(None)
            result = DetectionResult(
                face_detected=False,
                face_count=0,
                primary_face_selected=False,
                left_ear=None,
                right_ear=None,
                average_ear=None,
                eyes_closed=state.eyes_closed,
                consecutive_closed_frames=state.consecutive_closed_frames,
                drowsiness_state=state.drowsiness_state,
                alert_required=state.alert_required,
                yawning_detected=yawning.detected,
                consecutive_yawning_frames=yawning.consecutive_frames,
                head_tilt_detected=head_tilt.detected,
                consecutive_head_tilt_frames=head_tilt.consecutive_frames,
            )
            self._log_transitions(result, yawning, head_tilt)
            return result

        primary_face = _select_primary_face(faces)
        facial_landmarks = self.landmark_detector.facial_landmarks(
            gray_frame,
            primary_face,
        )
        eye_landmarks = facial_landmarks.eye_landmarks
        mouth_landmarks = facial_landmarks.mouth
        try:
            left_ear = eye_aspect_ratio(eye_landmarks.left_eye)
            right_ear = eye_aspect_ratio(eye_landmarks.right_eye)
        except ValueError:
            yawning = self.yawning_tracker.update(None)
            head_tilt = self.head_tilt_tracker.update(None)
            state = self.tracker.update(None)
            result = DetectionResult(
                face_detected=True,
                face_count=len(faces),
                primary_face_selected=True,
                left_ear=None,
                right_ear=None,
                average_ear=None,
                eyes_closed=False,
                consecutive_closed_frames=state.consecutive_closed_frames,
                drowsiness_state=DrowsinessState.INVALID_EAR,
                alert_required=False,
                yawning_detected=yawning.detected,
                consecutive_yawning_frames=yawning.consecutive_frames,
                head_tilt_detected=head_tilt.detected,
                consecutive_head_tilt_frames=head_tilt.consecutive_frames,
                eye_landmarks=eye_landmarks,
                mouth_landmarks=mouth_landmarks,
                primary_face_bounds=_face_bounds(primary_face),
            )
            self._log_transitions(result, yawning, head_tilt)
            return result

        average_ear = (left_ear + right_ear) / 2.0
        mar = _calculate_mar(mouth_landmarks) if self.yawning_enabled else None
        yawning = self.yawning_tracker.update(
            None if mar is None else mar >= self.mar_threshold
        )
        head_metrics = (
            _calculate_head_metrics(facial_landmarks.points)
            if self.head_pose_enabled
            else None
        )
        head_tilt = self.head_tilt_tracker.update(
            None
            if head_metrics is None
            else is_head_tilted(
                head_metrics,
                roll_threshold_degrees=self.head_roll_threshold_degrees,
                downward_ratio_threshold=self.head_downward_ratio_threshold,
            )
        )
        state = self.tracker.update(
            average_ear,
            yawning_detected=yawning.detected,
            head_tilt_detected=head_tilt.detected,
        )

        result = DetectionResult(
            face_detected=True,
            face_count=len(faces),
            primary_face_selected=True,
            left_ear=left_ear,
            right_ear=right_ear,
            average_ear=average_ear,
            eyes_closed=state.eyes_closed,
            consecutive_closed_frames=state.consecutive_closed_frames,
            drowsiness_state=state.drowsiness_state,
            alert_required=state.alert_required,
            mar=mar,
            yawning_detected=yawning.detected,
            consecutive_yawning_frames=yawning.consecutive_frames,
            head_tilt_detected=head_tilt.detected,
            consecutive_head_tilt_frames=head_tilt.consecutive_frames,
            head_tilt_angle=None
            if head_metrics is None
            else head_metrics.roll_angle_degrees,
            head_downward_ratio=None
            if head_metrics is None
            else head_metrics.downward_ratio,
            eye_landmarks=eye_landmarks,
            mouth_landmarks=mouth_landmarks,
            primary_face_bounds=_face_bounds(primary_face),
        )
        self._log_transitions(result, yawning, head_tilt)
        return result

    def _log_transitions(self, result, yawning, head_tilt) -> None:
        if yawning.transitioned_to_detected:
            LOGGER.info("Yawning detected.")
        if yawning.transitioned_to_clear:
            LOGGER.info("Yawning recovered.")
        if head_tilt.transitioned_to_detected:
            LOGGER.info("Head tilt detected.")
        if head_tilt.transitioned_to_clear:
            LOGGER.info("Head tilt recovered.")
        if result.drowsiness_state != self._last_drowsiness_state:
            LOGGER.info("Drowsiness state changed to %s.", result.drowsiness_state)
            self._last_drowsiness_state = result.drowsiness_state


def select_primary_face(faces: list[object]) -> object | None:
    valid_faces = _valid_faces(faces)
    if not valid_faces:
        return None
    return max(valid_faces, key=_face_area)


def _select_primary_face(faces: list[object]) -> object:
    primary_face = select_primary_face(faces)
    if primary_face is None:
        raise DetectionError("No valid face rectangle was available.")
    return primary_face


def _valid_faces(faces: list[object]) -> list[object]:
    return [face for face in faces if _face_area(face) > 0]


def _face_area(face: object) -> int:
    width = int(face.right() - face.left())
    height = int(face.bottom() - face.top())
    if width <= 0 or height <= 0:
        return 0
    return width * height


def _face_bounds(face: object) -> tuple[int, int, int, int]:
    return (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))


def _calculate_mar(mouth_landmarks: np.ndarray) -> float | None:
    try:
        return mouth_aspect_ratio(mouth_landmarks)
    except ValueError:
        return None


def _calculate_head_metrics(landmarks: np.ndarray) -> HeadPoseMetrics | None:
    try:
        return estimate_head_pose_metrics(landmarks)
    except ValueError:
        return None
