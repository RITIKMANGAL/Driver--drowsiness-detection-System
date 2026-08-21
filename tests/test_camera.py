from __future__ import annotations

import numpy as np
import pytest

from driver_drowsiness.camera import Camera
from driver_drowsiness.exceptions import CameraError


class FakeCapture:
    def __init__(
        self,
        *,
        opened: bool = True,
        read_ok: bool = True,
        frame=None,
        fail_read: bool = False,
        fail_release: bool = False,
    ) -> None:
        self.opened = opened
        self.read_ok = read_ok
        self.frame = frame if frame is not None else np.zeros((2, 2, 3), dtype=np.uint8)
        self.fail_read = fail_read
        self.fail_release = fail_release
        self.release_calls = 0

    def isOpened(self) -> bool:
        return self.opened

    def read(self):
        if self.fail_read:
            raise RuntimeError("camera disconnected")
        return self.read_ok, self.frame

    def release(self) -> None:
        self.release_calls += 1
        if self.fail_release:
            raise RuntimeError("release failed")


def test_successful_initialization() -> None:
    capture = FakeCapture()
    camera = Camera(0, capture_factory=lambda _index: capture)

    camera.open()

    assert camera._capture is capture


def test_failed_initialization() -> None:
    capture = FakeCapture(opened=False)
    camera = Camera(0, capture_factory=lambda _index: capture)

    with pytest.raises(CameraError, match="Unable to open camera index 0"):
        camera.open()

    assert capture.release_calls == 1


def test_successful_frame_read() -> None:
    frame = np.ones((3, 4, 3), dtype=np.uint8)
    camera = Camera(0, capture_factory=lambda _index: FakeCapture(frame=frame))
    camera.open()

    assert camera.read() is frame


def test_failed_frame_read_false_status() -> None:
    camera = Camera(0, capture_factory=lambda _index: FakeCapture(read_ok=False))
    camera.open()

    with pytest.raises(CameraError, match="Unable to read frame"):
        camera.read()


def test_failed_frame_read_exception() -> None:
    camera = Camera(0, capture_factory=lambda _index: FakeCapture(fail_read=True))
    camera.open()

    with pytest.raises(CameraError, match="Unable to read frame"):
        camera.read()


def test_release_is_safe_and_repeatable() -> None:
    capture = FakeCapture()
    camera = Camera(0, capture_factory=lambda _index: capture)
    camera.open()

    camera.release()
    camera.release()

    assert capture.release_calls == 1


def test_release_failure_does_not_raise() -> None:
    capture = FakeCapture(fail_release=True)
    camera = Camera(0, capture_factory=lambda _index: capture)
    camera.open()

    camera.release()

    assert camera._capture is None


def test_context_manager_releases_camera() -> None:
    capture = FakeCapture()

    with Camera(0, capture_factory=lambda _index: capture):
        pass

    assert capture.release_calls == 1
