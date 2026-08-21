"""Camera lifecycle and frame capture."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

from driver_drowsiness.exceptions import CameraError

LOGGER = logging.getLogger(__name__)


@dataclass
class Camera:
    index: int = 0
    capture_factory: Callable[[int], Any] = cv2.VideoCapture
    logger: logging.Logger = LOGGER

    def __post_init__(self) -> None:
        self._capture: Any | None = None

    def open(self) -> None:
        if self._capture is not None:
            return

        try:
            capture = self.capture_factory(self.index)
        except Exception as exc:
            raise CameraError(f"Unable to initialize camera index {self.index}.") from exc

        if not capture.isOpened():
            _safe_release(capture, self.logger)
            raise CameraError(f"Unable to open camera index {self.index}.")

        self._capture = capture
        self.logger.info("Camera initialized: index %s", self.index)

    def read(self) -> np.ndarray:
        if self._capture is None:
            raise CameraError("Camera has not been opened.")

        try:
            ok, frame = self._capture.read()
        except Exception as exc:
            raise CameraError("Unable to read frame from camera.") from exc
        if not ok or frame is None:
            raise CameraError("Unable to read frame from camera.")
        return frame

    def release(self) -> None:
        if self._capture is not None:
            _safe_release(self._capture, self.logger)
            self._capture = None

    def __enter__(self) -> Camera:
        self.open()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.release()


def _safe_release(capture: Any, logger: logging.Logger) -> None:
    try:
        capture.release()
    except Exception as exc:
        logger.warning("Camera release failed: %s", exc)
