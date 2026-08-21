from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from driver_drowsiness.app import Application
from driver_drowsiness.config import (
    AlertConfig,
    ApplicationConfig,
    AssetConfig,
    CameraConfig,
    DetectionConfig,
    HeadPoseConfig,
    LoggingConfig,
    YawningConfig,
)
from driver_drowsiness.detector import DetectionResult
from driver_drowsiness.exceptions import ApplicationError, AudioError, CameraError
from driver_drowsiness.state import DrowsinessState


class FakeCamera:
    def __init__(self, frames: list[np.ndarray], *, fail_read: bool = False) -> None:
        self.frames = frames
        self.fail_read = fail_read
        self.release_calls = 0
        self.read_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.release()

    def read(self) -> np.ndarray:
        self.read_calls += 1
        if self.fail_read:
            raise CameraError("camera failed")
        return self.frames.pop(0)

    def release(self) -> None:
        self.release_calls += 1


class FakeDetector:
    def __init__(
        self,
        results: list[DetectionResult],
        *,
        fail_process: bool = False,
    ) -> None:
        self.results = results
        self.fail_process = fail_process
        self.process_calls = 0

    def process(self, _frame: np.ndarray) -> DetectionResult:
        self.process_calls += 1
        if self.fail_process:
            raise RuntimeError("detector failed")
        return self.results.pop(0)


class FakeAlarm:
    def __init__(self, *, fail_close: bool = False) -> None:
        self.fail_close = fail_close
        self.start_calls = 0
        self.stop_calls = 0
        self.close_calls = 0
        self.playback_calls = 0

    def start(self) -> bool:
        self.start_calls += 1
        self.playback_calls += 1
        return True

    def stop(self) -> bool:
        self.stop_calls += 1
        return True

    def close(self) -> None:
        self.close_calls += 1
        if self.fail_close:
            raise RuntimeError("cleanup failed")


class DisabledFakeAlarm(FakeAlarm):
    def start(self) -> bool:
        self.start_calls += 1
        return False


class FakeCV2:
    FONT_HERSHEY_SIMPLEX = 0

    def __init__(self, wait_keys: list[int] | None = None, *, fail_imshow: bool = False):
        self.wait_keys = wait_keys or [27]
        self.fail_imshow = fail_imshow
        self.imshow_calls = 0
        self.destroy_calls = 0
        self.put_text_calls = 0

    def imshow(self, _window_name: str, _frame: np.ndarray) -> None:
        self.imshow_calls += 1
        if self.fail_imshow:
            raise RuntimeError("headless")

    def waitKey(self, _delay: int) -> int:
        return self.wait_keys.pop(0) if self.wait_keys else 27

    def destroyAllWindows(self) -> None:
        self.destroy_calls += 1

    def putText(self, *_args, **_kwargs) -> None:
        self.put_text_calls += 1

    def rectangle(self, *_args, **_kwargs) -> None:
        pass

    def convexHull(self, points):
        return points

    def drawContours(self, *_args, **_kwargs) -> None:
        pass


def test_normal_startup_and_esc_shutdown() -> None:
    camera = FakeCamera([_frame()])
    detector = FakeDetector([_result(DrowsinessState.ATTENTIVE)])
    alarm = FakeAlarm()
    cv2 = FakeCV2([27])
    app = _app(camera, detector, alarm, cv2)

    assert app.run() == 0

    assert detector.process_calls == 1
    assert camera.release_calls == 1
    assert alarm.close_calls == 1
    assert cv2.destroy_calls == 1


def test_camera_failure_cleans_up_resources() -> None:
    camera = FakeCamera([_frame()], fail_read=True)
    detector = FakeDetector([_result(DrowsinessState.ATTENTIVE)])
    alarm = FakeAlarm()
    cv2 = FakeCV2([27])
    app = _app(camera, detector, alarm, cv2)

    with pytest.raises(CameraError, match="camera failed"):
        app.run()

    assert camera.release_calls == 1
    assert alarm.close_calls == 1
    assert cv2.destroy_calls == 1


def test_detector_failure_cleans_up_resources_and_preserves_error() -> None:
    camera = FakeCamera([_frame()])
    detector = FakeDetector([_result(DrowsinessState.ATTENTIVE)], fail_process=True)
    alarm = FakeAlarm()
    cv2 = FakeCV2([27])
    app = _app(camera, detector, alarm, cv2)

    with pytest.raises(RuntimeError, match="detector failed"):
        app.run()

    assert camera.release_calls == 1
    assert alarm.close_calls == 1
    assert cv2.destroy_calls == 1


def test_alarm_cleanup_failure_does_not_mask_original_exception() -> None:
    camera = FakeCamera([_frame()])
    detector = FakeDetector([_result(DrowsinessState.ATTENTIVE)], fail_process=True)
    alarm = FakeAlarm(fail_close=True)
    cv2 = FakeCV2([27])
    app = _app(camera, detector, alarm, cv2)

    with pytest.raises(RuntimeError, match="detector failed"):
        app.run()

    assert alarm.close_calls == 1
    assert cv2.destroy_calls == 1


def test_gui_failure_is_reported_as_application_error() -> None:
    camera = FakeCamera([_frame()])
    detector = FakeDetector([_result(DrowsinessState.ATTENTIVE)])
    alarm = FakeAlarm()
    cv2 = FakeCV2([27], fail_imshow=True)
    app = _app(camera, detector, alarm, cv2)

    with pytest.raises(ApplicationError, match="OpenCV GUI operation failed"):
        app.run()

    assert camera.release_calls == 1
    assert alarm.close_calls == 1
    assert cv2.destroy_calls == 1


def test_alert_transition_starts_once_and_stops_once() -> None:
    states = [
        DrowsinessState.ATTENTIVE,
        DrowsinessState.ATTENTIVE,
        DrowsinessState.ATTENTIVE,
        DrowsinessState.DROWSY,
        DrowsinessState.DROWSY,
        DrowsinessState.DROWSY,
        DrowsinessState.DROWSY,
        DrowsinessState.ATTENTIVE,
    ]
    results = [
        _result(state, alert_required=(index == 3))
        for index, state in enumerate(states)
    ]
    camera = FakeCamera([_frame() for _ in results])
    detector = FakeDetector(results)
    alarm = FakeAlarm()
    cv2 = FakeCV2([0, 0, 0, 0, 0, 0, 0, 27])
    app = _app(camera, detector, alarm, cv2)

    app.run()

    assert alarm.start_calls == 1
    assert alarm.stop_calls == 1


def test_alert_disabled_continues_detection_without_playback() -> None:
    camera = FakeCamera([_frame()])
    detector = FakeDetector([_result(DrowsinessState.DROWSY, alert_required=True)])
    alarm = DisabledFakeAlarm()
    cv2 = FakeCV2([27])
    app = _app(camera, detector, alarm, cv2, alert_enabled=False)

    assert app.run() == 0

    assert detector.process_calls == 1
    assert alarm.playback_calls == 0


def test_audio_initialization_failure_continues_detection() -> None:
    camera = FakeCamera([_frame()])
    detector = FakeDetector([_result(DrowsinessState.ATTENTIVE)])
    cv2 = FakeCV2([27])
    app = Application(
        _config(),
        camera_factory=lambda _index: camera,
        detector_factory=lambda _config: detector,
        alarm_factory=lambda _config: (_ for _ in ()).throw(AudioError("bad audio")),
        cv2_module=cv2,
    )

    assert app.run() == 0

    assert detector.process_calls == 1
    assert camera.release_calls == 1
    assert cv2.destroy_calls == 1


def _app(
    camera: FakeCamera,
    detector: FakeDetector,
    alarm: FakeAlarm,
    cv2: FakeCV2,
    *,
    alert_enabled: bool = True,
) -> Application:
    return Application(
        _config(alert_enabled=alert_enabled),
        camera_factory=lambda _index: camera,
        detector_factory=lambda _config: detector,
        alarm_factory=lambda _config: alarm,
        cv2_module=cv2,
    )


def _config(*, alert_enabled: bool = True) -> ApplicationConfig:
    return ApplicationConfig(
        camera=CameraConfig(index=0, width=512, height=512, processing_scale=1.0),
        detection=DetectionConfig(ear_threshold=0.3, closed_frames_threshold=30),
        yawning=YawningConfig(),
        head_pose=HeadPoseConfig(),
        assets=AssetConfig(
            model_path=Path("shape_predictor_68_face_landmarks.dat"),
            alarm_path=Path("loud_alarm.wav"),
        ),
        alert=AlertConfig(enabled=alert_enabled),
        logging=LoggingConfig(level="INFO"),
    )


def _result(
    state: DrowsinessState,
    *,
    alert_required: bool = False,
) -> DetectionResult:
    return DetectionResult(
        face_detected=state not in {DrowsinessState.NO_FACE, DrowsinessState.INVALID_EAR},
        face_count=1 if state not in {DrowsinessState.NO_FACE} else 0,
        primary_face_selected=state not in {DrowsinessState.NO_FACE},
        left_ear=0.2 if state == DrowsinessState.DROWSY else 0.4,
        right_ear=0.2 if state == DrowsinessState.DROWSY else 0.4,
        average_ear=0.2 if state == DrowsinessState.DROWSY else 0.4,
        eyes_closed=state == DrowsinessState.DROWSY,
        consecutive_closed_frames=30 if state == DrowsinessState.DROWSY else 0,
        drowsiness_state=state,
        alert_required=alert_required,
    )


def _frame() -> np.ndarray:
    return np.zeros((32, 32, 3), dtype=np.uint8)
