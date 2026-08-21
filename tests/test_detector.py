from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from driver_drowsiness.detector import DrowsinessDetector, select_primary_face
from driver_drowsiness.exceptions import DetectionError
from driver_drowsiness.landmarks import FacialLandmarks
from driver_drowsiness.state import (
    ConsecutiveSignalTracker,
    DrowsinessState,
    DrowsinessTracker,
)


@dataclass(frozen=True)
class FakeFace:
    left_value: int
    top_value: int
    right_value: int
    bottom_value: int

    def left(self) -> int:
        return self.left_value

    def top(self) -> int:
        return self.top_value

    def right(self) -> int:
        return self.right_value

    def bottom(self) -> int:
        return self.bottom_value


class FakeLandmarkDetector:
    def __init__(
        self,
        faces: list[FakeFace],
        landmarks: dict[FakeFace, FacialLandmarks],
    ):
        self.faces = faces
        self.landmarks = landmarks
        self.detect_calls = 0
        self.landmark_calls: list[FakeFace] = []

    def detect_faces(self, _gray_frame: np.ndarray) -> list[FakeFace]:
        self.detect_calls += 1
        return self.faces

    def facial_landmarks(
        self,
        _gray_frame: np.ndarray,
        face: FakeFace,
    ) -> FacialLandmarks:
        self.landmark_calls.append(face)
        return self.landmarks[face]


def test_select_primary_face_handles_zero_faces() -> None:
    assert select_primary_face([]) is None


def test_select_primary_face_handles_one_face() -> None:
    face = FakeFace(0, 0, 10, 10)

    assert select_primary_face([face]) == face


def test_select_primary_face_selects_largest_face() -> None:
    small = FakeFace(0, 0, 10, 10)
    large = FakeFace(0, 0, 20, 20)
    medium = FakeFace(0, 0, 15, 15)

    assert select_primary_face([small, large, medium]) == large


def test_select_primary_face_ignores_invalid_rectangles() -> None:
    invalid = FakeFace(10, 10, 10, 20)
    valid = FakeFace(0, 0, 5, 5)

    assert select_primary_face([invalid, valid]) == valid


def test_detector_rejects_invalid_frame() -> None:
    detector = _detector([])

    with pytest.raises(DetectionError):
        detector.process(None)


def test_detector_handles_valid_frame_with_no_face() -> None:
    detector = _detector([])

    result = detector.process(_frame())

    assert result.face_detected is False
    assert result.face_count == 0
    assert result.primary_face_selected is False
    assert result.left_ear is None
    assert result.right_ear is None
    assert result.average_ear is None
    assert result.drowsiness_state == DrowsinessState.NO_FACE
    assert not result.alert_required


def test_detector_processes_one_face_and_returns_ear_fields() -> None:
    face = FakeFace(0, 0, 20, 20)
    detector = _detector([face], {face: _landmarks(2)})

    result = detector.process(_frame())

    assert result.face_detected is True
    assert result.face_count == 1
    assert result.primary_face_selected is True
    assert result.left_ear == pytest.approx(1.0)
    assert result.right_ear == pytest.approx(1.0)
    assert result.average_ear == pytest.approx(1.0)
    assert result.mar == pytest.approx(0.2)
    assert not result.yawning_detected
    assert not result.head_tilt_detected
    assert result.drowsiness_state == DrowsinessState.ATTENTIVE
    assert result.primary_face_bounds == (0, 0, 20, 20)


def test_detector_processes_only_largest_face_when_multiple_faces_exist() -> None:
    small = FakeFace(0, 0, 10, 10)
    large = FakeFace(0, 0, 30, 30)
    landmarks = {
        small: _landmarks(0.2),
        large: _landmarks(2),
    }
    landmark_detector = FakeLandmarkDetector([small, large], landmarks)
    detector = DrowsinessDetector(
        landmark_detector,
        DrowsinessTracker(ear_threshold=0.3, consecutive_frame_limit=30),
    )

    result = detector.process(_frame())

    assert result.face_count == 2
    assert result.average_ear == pytest.approx(1.0)
    assert landmark_detector.landmark_calls == [large]
    assert result.drowsiness_state == DrowsinessState.ATTENTIVE


def test_secondary_face_cannot_corrupt_state_counter() -> None:
    secondary_closed = FakeFace(0, 0, 10, 10)
    primary_open = FakeFace(0, 0, 30, 30)
    detector = _detector(
        [secondary_closed, primary_open],
        {
            secondary_closed: _landmarks(0.2),
            primary_open: _landmarks(2),
        },
        threshold=0.3,
        frame_limit=3,
    )

    for _ in range(3):
        result = detector.process(_frame())

    assert result.consecutive_closed_frames == 0
    assert result.drowsiness_state == DrowsinessState.ATTENTIVE
    assert not result.alert_required


def test_detector_valid_landmarks_can_trigger_drowsy_transition() -> None:
    face = FakeFace(0, 0, 20, 20)
    detector = _detector(
        [face],
        {face: _landmarks(0.2)},
        threshold=0.3,
        frame_limit=2,
    )

    first = detector.process(_frame())
    second = detector.process(_frame())
    third = detector.process(_frame())

    assert first.drowsiness_state == DrowsinessState.EYES_CLOSED
    assert not first.alert_required
    assert second.drowsiness_state == DrowsinessState.DROWSY
    assert second.alert_required
    assert third.drowsiness_state == DrowsinessState.DROWSY
    assert not third.alert_required


def test_detector_yawning_alone_is_supporting_signal_only() -> None:
    face = FakeFace(0, 0, 20, 20)
    detector = _detector(
        [face],
        {face: _landmarks(2, mouth_vertical=4)},
        yawning_frame_limit=2,
        head_tilt_frame_limit=2,
    )

    first = detector.process(_frame())
    second = detector.process(_frame())

    assert not first.yawning_detected
    assert second.yawning_detected
    assert second.mar == pytest.approx(0.8)
    assert second.drowsiness_state == DrowsinessState.ATTENTIVE
    assert not second.alert_required


def test_detector_yawning_recovers_when_mouth_closes() -> None:
    face = FakeFace(0, 0, 20, 20)
    landmark_detector = FakeLandmarkDetector(
        [face],
        {face: _landmarks(2, mouth_vertical=4)},
    )
    detector = DrowsinessDetector(
        landmark_detector,
        DrowsinessTracker(ear_threshold=0.3, consecutive_frame_limit=30),
        yawning_tracker=ConsecutiveSignalTracker(2),
    )

    detector.process(_frame())
    detector.process(_frame())
    landmark_detector.landmarks = {face: _landmarks(2, mouth_vertical=1)}
    recovered = detector.process(_frame())

    assert not recovered.yawning_detected
    assert recovered.consecutive_yawning_frames == 0


def test_detector_head_tilt_alone_is_supporting_signal_only() -> None:
    face = FakeFace(0, 0, 20, 20)
    detector = _detector(
        [face],
        {face: _landmarks(2, right_eye_y=4)},
        yawning_frame_limit=2,
        head_tilt_frame_limit=2,
    )

    first = detector.process(_frame())
    second = detector.process(_frame())

    assert not first.head_tilt_detected
    assert second.head_tilt_detected
    assert second.head_tilt_angle is not None
    assert second.drowsiness_state == DrowsinessState.ATTENTIVE
    assert not second.alert_required


def test_detector_head_tilt_recovers() -> None:
    face = FakeFace(0, 0, 20, 20)
    landmark_detector = FakeLandmarkDetector(
        [face],
        {face: _landmarks(2, right_eye_y=4)},
    )
    detector = DrowsinessDetector(
        landmark_detector,
        DrowsinessTracker(ear_threshold=0.3, consecutive_frame_limit=30),
        head_tilt_tracker=ConsecutiveSignalTracker(2),
    )

    detector.process(_frame())
    detector.process(_frame())
    landmark_detector.landmarks = {face: _landmarks(2)}
    recovered = detector.process(_frame())

    assert not recovered.head_tilt_detected
    assert recovered.consecutive_head_tilt_frames == 0


def test_detector_combined_supporting_signals_can_trigger_drowsiness() -> None:
    face = FakeFace(0, 0, 20, 20)
    detector = _detector(
        [face],
        {face: _landmarks(2, mouth_vertical=4, right_eye_y=4)},
        yawning_frame_limit=2,
        head_tilt_frame_limit=2,
    )

    first = detector.process(_frame())
    second = detector.process(_frame())
    third = detector.process(_frame())

    assert first.drowsiness_state == DrowsinessState.ATTENTIVE
    assert second.drowsiness_state == DrowsinessState.DROWSY
    assert second.alert_required
    assert third.drowsiness_state == DrowsinessState.DROWSY
    assert not third.alert_required


def test_detector_invalid_landmark_geometry_returns_invalid_ear_result() -> None:
    face = FakeFace(0, 0, 20, 20)
    detector = _detector([face], {face: _invalid_landmarks()})

    result = detector.process(_frame())

    assert result.face_detected is True
    assert result.left_ear is None
    assert result.right_ear is None
    assert result.average_ear is None
    assert result.mar is None
    assert not result.yawning_detected
    assert not result.head_tilt_detected
    assert result.drowsiness_state == DrowsinessState.INVALID_EAR
    assert not result.alert_required


def test_detector_no_face_resets_stale_state() -> None:
    face = FakeFace(0, 0, 20, 20)
    landmark_detector = FakeLandmarkDetector([face], {face: _landmarks(0.2)})
    detector = DrowsinessDetector(
        landmark_detector,
        DrowsinessTracker(ear_threshold=0.3, consecutive_frame_limit=3),
    )

    detector.process(_frame())
    landmark_detector.faces = []
    result = detector.process(_frame())

    assert result.drowsiness_state == DrowsinessState.NO_FACE
    assert result.consecutive_closed_frames == 0
    assert not result.alert_required
    assert result.average_ear is None


def _detector(
    faces: list[FakeFace],
    landmarks: dict[FakeFace, FacialLandmarks] | None = None,
    *,
    threshold: float = 0.3,
    frame_limit: int = 30,
    yawning_frame_limit: int = 15,
    head_tilt_frame_limit: int = 15,
) -> DrowsinessDetector:
    return DrowsinessDetector(
        FakeLandmarkDetector(faces, landmarks or {}),
        DrowsinessTracker(ear_threshold=threshold, consecutive_frame_limit=frame_limit),
        yawning_tracker=ConsecutiveSignalTracker(yawning_frame_limit),
        head_tilt_tracker=ConsecutiveSignalTracker(head_tilt_frame_limit),
    )


def _frame() -> np.ndarray:
    return np.zeros((32, 32, 3), dtype=np.uint8)


def _landmarks(
    vertical: float,
    *,
    mouth_vertical: float = 1.0,
    right_eye_y: float = 0.0,
) -> FacialLandmarks:
    points = np.zeros((68, 2), dtype=float)
    points[36:42] = _eye(vertical, x_offset=0, y_offset=0)
    points[42:48] = _eye(vertical, x_offset=10, y_offset=right_eye_y)
    points[48:68] = _mouth(mouth_vertical)
    points[33] = [6.0, 8.0]
    points[8] = [6.0, 20.0]
    return FacialLandmarks(points=points)


def _eye(vertical: float, *, x_offset: float, y_offset: float) -> np.ndarray:
    return np.array(
        [
            [0, 0],
            [1, vertical],
            [2, vertical],
            [4, 0],
            [2, -vertical],
            [1, -vertical],
        ],
        dtype=float,
    ) + np.array([x_offset, y_offset], dtype=float)


def _mouth(vertical: float) -> np.ndarray:
    mouth = np.zeros((20, 2), dtype=float)
    mouth[0] = [0, 0]
    mouth[6] = [10, 0]
    mouth[2] = [2, vertical]
    mouth[10] = [2, -vertical]
    mouth[3] = [5, vertical]
    mouth[9] = [5, -vertical]
    mouth[4] = [8, vertical]
    mouth[8] = [8, -vertical]
    return mouth


def _invalid_landmarks() -> FacialLandmarks:
    landmarks = _landmarks(2)
    landmarks.points[39] = landmarks.points[36]
    landmarks.points[45] = landmarks.points[42]
    return landmarks
