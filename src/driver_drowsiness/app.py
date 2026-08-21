"""Application orchestration for the driver drowsiness detector."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2

from driver_drowsiness.alerts import AudioAlarm
from driver_drowsiness.camera import Camera
from driver_drowsiness.config import (
    ApplicationConfig,
    ConfigOverrides,
    load_config,
)
from driver_drowsiness.diagnostics import run_diagnostics_command
from driver_drowsiness.detector import DetectionResult, DrowsinessDetector
from driver_drowsiness.exceptions import (
    ApplicationError,
    AudioError,
    DriverDrowsinessError,
)
from driver_drowsiness.landmarks import FacialLandmarkDetector
from driver_drowsiness.logging_config import configure_logging
from driver_drowsiness.preprocessing import preprocess_frame
from driver_drowsiness.state import (
    ConsecutiveSignalTracker,
    DrowsinessState,
    DrowsinessTracker,
)

LOGGER = logging.getLogger(__name__)


def build_detector(config: ApplicationConfig) -> DrowsinessDetector:
    landmark_detector = FacialLandmarkDetector(config.assets.model_path)
    tracker = DrowsinessTracker(
        ear_threshold=config.detection.ear_threshold,
        consecutive_frame_limit=config.detection.closed_frames_threshold,
    )
    detector = DrowsinessDetector(
        landmark_detector,
        tracker,
        yawning_tracker=ConsecutiveSignalTracker(config.yawning.consecutive_frames),
        head_tilt_tracker=ConsecutiveSignalTracker(
            config.head_pose.consecutive_frames,
        ),
        yawning_enabled=config.yawning.enabled,
        mar_threshold=config.yawning.mar_threshold,
        head_pose_enabled=config.head_pose.enabled,
        head_roll_threshold_degrees=config.head_pose.roll_threshold_degrees,
        head_downward_ratio_threshold=config.head_pose.downward_ratio_threshold,
    )
    LOGGER.info("Drowsiness detector initialized.")
    return detector


def build_alarm(config: ApplicationConfig) -> AudioAlarm:
    return AudioAlarm(config.assets.alarm_path, enabled=config.alert.enabled)


@dataclass
class Application:
    config: ApplicationConfig
    camera_factory: Callable[[int], Any] = Camera
    detector_factory: Callable[[ApplicationConfig], Any] = build_detector
    alarm_factory: Callable[[ApplicationConfig], Any] = build_alarm
    cv2_module: Any = cv2
    logger: logging.Logger = LOGGER

    def run(self) -> int:
        self.logger.info("Application startup.")
        alarm = None

        try:
            alarm = self._initialize_alarm()
            detector = self.detector_factory(self.config)
            alarm_active = False
            with self.camera_factory(self.config.camera.index) as camera:
                while True:
                    frame = camera.read()
                    frame = preprocess_frame(
                        frame,
                        max_width=self.config.camera.width,
                        max_height=self.config.camera.height,
                        processing_scale=self.config.camera.processing_scale,
                    )
                    result = detector.process(frame)
                    _draw_overlay(frame, result, self.cv2_module)
                    alarm_active = _handle_alert(result, alarm, alarm_active)

                    if result.alert_required:
                        _draw_alert(frame, self.cv2_module)

                    if self._should_shutdown(frame):
                        self.logger.info("Shutdown requested from GUI.")
                        break
        finally:
            self._cleanup(alarm)

        self.logger.info("Application shutdown complete.")
        return 0

    def _initialize_alarm(self) -> Any:
        try:
            return self.alarm_factory(self.config)
        except AudioError as exc:
            self.logger.error("Alert audio unavailable; continuing without audio: %s", exc)
            return NoopAlarm()

    def _should_shutdown(self, frame) -> bool:
        try:
            self.cv2_module.imshow("Drowsiness Detection", frame)
            return bool(self.cv2_module.waitKey(1) & 0xFF == 27)
        except Exception as exc:
            raise ApplicationError("OpenCV GUI operation failed.") from exc

    def _cleanup(self, alarm: Any | None) -> None:
        if alarm is not None:
            try:
                alarm.close()
            except Exception as exc:
                self.logger.warning("Alarm cleanup failed: %s", exc)

        try:
            self.cv2_module.destroyAllWindows()
        except Exception as exc:
            self.logger.warning("OpenCV window cleanup failed: %s", exc)

        _flush_logs()


class NoopAlarm:
    def start(self) -> bool:
        return False

    def stop(self) -> bool:
        return False

    def close(self) -> None:
        return None


def run(config: ApplicationConfig) -> int:
    return Application(config).run()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    overrides = _overrides_from_args(args)

    if args.command == "diagnose":
        return run_diagnostics_command(overrides=overrides)

    try:
        config = load_config(overrides=overrides)
        configure_logging(config.logging.level_value)
        return run(config)
    except DriverDrowsinessError as exc:
        configure_logging()
        LOGGER.error("%s", exc)
        return 1
    except Exception:
        configure_logging()
        LOGGER.exception("Unexpected application failure.")
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the driver drowsiness detection application."
    )
    parser.set_defaults(command="run")
    _add_runtime_arguments(parser)
    subparsers = parser.add_subparsers(dest="command")
    diagnose_parser = subparsers.add_parser(
        "diagnose",
        help="Run non-invasive environment and hardware diagnostics.",
    )
    _add_runtime_arguments(diagnose_parser)
    return parser


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="Path to an application config.yml.")
    parser.add_argument("--camera", type=int, help="Camera index to open.")
    parser.add_argument("--width", type=int, help="Frame width before detection.")
    parser.add_argument("--height", type=int, help="Frame height before detection.")
    parser.add_argument(
        "--processing-scale",
        type=float,
        help="Frame processing scale, greater than 0 and up to 1.",
    )
    parser.add_argument("--ear-threshold", type=float, help="EAR threshold.")
    parser.add_argument(
        "--closed-frames",
        type=int,
        help="Closed-eye frames required before alerting.",
    )
    parser.add_argument("--model", type=Path, help="Path to dlib landmark model.")
    parser.add_argument("--alarm", type=Path, help="Path to WAV alarm file.")
    parser.add_argument(
        "--disable-alarm",
        action="store_true",
        help="Disable audio alarm playback.",
    )
    parser.add_argument("--log-level", help="Logging level, such as INFO or DEBUG.")


def _overrides_from_args(args: argparse.Namespace) -> ConfigOverrides:
    return ConfigOverrides(
        config_path=getattr(args, "config", None),
        camera_index=getattr(args, "camera", None),
        width=getattr(args, "width", None),
        height=getattr(args, "height", None),
        processing_scale=getattr(args, "processing_scale", None),
        ear_threshold=getattr(args, "ear_threshold", None),
        closed_frames_threshold=getattr(args, "closed_frames", None),
        model_path=getattr(args, "model", None),
        alarm_path=getattr(args, "alarm", None),
        alert_enabled=False if getattr(args, "disable_alarm", False) else None,
        log_level=getattr(args, "log_level", None),
    )


def _handle_alert(result: DetectionResult, alarm: Any, alarm_active: bool) -> bool:
    if result.alert_required and not alarm_active:
        return bool(alarm.start())
    if result.drowsiness_state == DrowsinessState.DROWSY:
        return alarm_active
    if alarm_active:
        alarm.stop()
    return False


def _draw_alert(frame, cv2_module: Any = cv2) -> None:
    cv2_module.putText(
        frame,
        "ALERT",
        (10, 30),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )


def _draw_overlay(frame, result: DetectionResult, cv2_module: Any = cv2) -> None:
    if result.primary_face_bounds is not None:
        left, top, right, bottom = result.primary_face_bounds
        cv2_module.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 1)

    if result.eye_landmarks is not None:
        left_eye_hull = cv2_module.convexHull(result.eye_landmarks.left_eye)
        right_eye_hull = cv2_module.convexHull(result.eye_landmarks.right_eye)
        cv2_module.drawContours(frame, [left_eye_hull], -1, (0, 0, 255), 1)
        cv2_module.drawContours(frame, [right_eye_hull], -1, (0, 0, 255), 1)

    if result.average_ear is not None:
        cv2_module.putText(
            frame,
            "EAR: {:.2f}".format(result.average_ear),
            (400, 40),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    if result.mar is not None:
        cv2_module.putText(
            frame,
            "MAR: {:.2f}".format(result.mar),
            (400, 70),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    cv2_module.putText(
        frame,
        f"Drowsiness: {result.drowsiness_state.upper()}",
        (10, 60),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2_module.putText(
        frame,
        f"Yawning: {'YES' if result.yawning_detected else 'NO'}",
        (10, 90),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2_module.putText(
        frame,
        f"Head Tilt: {'YES' if result.head_tilt_detected else 'NO'}",
        (10, 120),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )


def _flush_logs() -> None:
    for handler in logging.getLogger().handlers:
        handler.flush()
