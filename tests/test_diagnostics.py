from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np

from driver_drowsiness.diagnostics import run_diagnostics, run_diagnostics_command


class FakeCamera:
    def __init__(self, _index: int, *, fail_open: bool = False) -> None:
        self.fail_open = fail_open
        self.release_calls = 0

    def open(self) -> None:
        if self.fail_open:
            raise RuntimeError("camera unavailable")

    def read(self) -> np.ndarray:
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def release(self) -> None:
        self.release_calls += 1


class FakeSoundFile:
    def __init__(self, *, fail_read: bool = False) -> None:
        self.fail_read = fail_read

    def read(self, _path: Path, dtype: str):
        if self.fail_read:
            raise RuntimeError("bad wav")
        return np.zeros((2,), dtype=dtype), 44100


class FakeSoundDevice:
    def __init__(self, *, fail_query: bool = False) -> None:
        self.fail_query = fail_query

    def query_devices(self):
        if self.fail_query:
            raise RuntimeError("audio unavailable")
        return [{"name": "speaker", "max_output_channels": 2}]


class FakeModelLoader:
    def __init__(self, _path: Path) -> None:
        pass


def test_successful_diagnostic(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    report = run_diagnostics(
        config_path=config_path,
        camera_factory=lambda index: FakeCamera(index),
        landmark_detector_factory=FakeModelLoader,
        soundfile_module=FakeSoundFile(),
        sounddevice_module=FakeSoundDevice(),
    )

    assert report.ready is True


def test_invalid_configuration_returns_failed_report(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("camera: [", encoding="utf-8")

    report = run_diagnostics(
        config_path=config_path,
        camera_factory=lambda index: FakeCamera(index),
        landmark_detector_factory=FakeModelLoader,
        soundfile_module=FakeSoundFile(),
        sounddevice_module=FakeSoundDevice(),
    )

    assert report.ready is False
    assert any("Invalid YAML" in reason for reason in report.reasons)


def test_missing_model_returns_failed_report(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, model_path="missing.dat")

    report = run_diagnostics(
        config_path=config_path,
        camera_factory=lambda index: FakeCamera(index),
        landmark_detector_factory=FakeModelLoader,
        soundfile_module=FakeSoundFile(),
        sounddevice_module=FakeSoundDevice(),
    )

    assert report.ready is False
    assert any("assets.model_path" in reason for reason in report.reasons)


def test_missing_alarm_returns_failed_report(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, alarm_path="missing.wav")

    report = run_diagnostics(
        config_path=config_path,
        camera_factory=lambda index: FakeCamera(index),
        landmark_detector_factory=FakeModelLoader,
        soundfile_module=FakeSoundFile(),
        sounddevice_module=FakeSoundDevice(),
    )

    assert report.ready is False
    assert any("assets.alarm_path" in reason for reason in report.reasons)


def test_camera_unavailable_returns_failed_report(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    report = run_diagnostics(
        config_path=config_path,
        camera_factory=lambda index: FakeCamera(index, fail_open=True),
        landmark_detector_factory=FakeModelLoader,
        soundfile_module=FakeSoundFile(),
        sounddevice_module=FakeSoundDevice(),
    )

    assert report.ready is False
    assert any("camera unavailable" in reason for reason in report.reasons)


def test_camera_is_released_after_diagnostic(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    camera = FakeCamera(0)

    run_diagnostics(
        config_path=config_path,
        camera_factory=lambda _index: camera,
        landmark_detector_factory=FakeModelLoader,
        soundfile_module=FakeSoundFile(),
        sounddevice_module=FakeSoundDevice(),
    )

    assert camera.release_calls == 1


def test_diagnostic_exit_status_success(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    output = StringIO()

    exit_code = run_diagnostics_command(
        config_path=config_path,
        output=output,
        camera_factory=lambda index: FakeCamera(index),
        landmark_detector_factory=FakeModelLoader,
        soundfile_module=FakeSoundFile(),
        sounddevice_module=FakeSoundDevice(),
    )

    assert exit_code == 0
    assert "Overall" in output.getvalue()
    assert "READY" in output.getvalue()


def test_diagnostic_exit_status_failure(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    output = StringIO()

    exit_code = run_diagnostics_command(
        config_path=config_path,
        output=output,
        camera_factory=lambda index: FakeCamera(index, fail_open=True),
        landmark_detector_factory=FakeModelLoader,
        soundfile_module=FakeSoundFile(),
        sounddevice_module=FakeSoundDevice(),
    )

    assert exit_code == 1
    assert "NOT READY" in output.getvalue()


def _write_config(
    tmp_path: Path,
    *,
    model_path: str = "model.dat",
    alarm_path: str = "alarm.wav",
) -> Path:
    if model_path != "missing.dat":
        (tmp_path / model_path).write_text("model", encoding="utf-8")
    if alarm_path != "missing.wav":
        (tmp_path / alarm_path).write_text("alarm", encoding="utf-8")
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        f"""
camera:
  index: 0
  width: 512
  height: 512
  processing_scale: 1.0
detection:
  ear_threshold: 0.3
  closed_frames_threshold: 30
assets:
  model_path: {model_path}
  alarm_path: {alarm_path}
alert:
  enabled: true
logging:
  level: INFO
""",
        encoding="utf-8",
    )
    return config_path
