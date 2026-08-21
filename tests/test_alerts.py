from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from driver_drowsiness.alerts import AlarmState, AudioAlarm
from driver_drowsiness.exceptions import AudioError


class FakeSoundFile:
    def __init__(self, *, fail_read: bool = False) -> None:
        self.fail_read = fail_read

    def read(self, _path: Path, dtype: str):
        if self.fail_read:
            raise RuntimeError("bad wav")
        return np.zeros((4,), dtype=dtype), 44100


class FakeSoundDevice:
    def __init__(self, *, fail_play: bool = False, fail_stop: bool = False) -> None:
        self.fail_play = fail_play
        self.fail_stop = fail_stop
        self.play_calls = 0
        self.stop_calls = 0

    def play(self, _data, _sample_rate: int) -> None:
        self.play_calls += 1
        if self.fail_play:
            raise RuntimeError("no output device")

    def stop(self) -> None:
        self.stop_calls += 1
        if self.fail_stop:
            raise RuntimeError("device disappeared")


def test_disabled_alert_does_not_load_or_play_missing_file(tmp_path: Path) -> None:
    sounddevice = FakeSoundDevice()

    alarm = AudioAlarm(
        tmp_path / "missing.wav",
        enabled=False,
        sounddevice_module=sounddevice,
        soundfile_module=FakeSoundFile(),
    )

    assert alarm.state == AlarmState.DISABLED
    assert alarm.start() is False
    assert sounddevice.play_calls == 0


def test_start_alarm(tmp_path: Path) -> None:
    alarm_path = _alarm_file(tmp_path)
    sounddevice = FakeSoundDevice()
    alarm = AudioAlarm(
        alarm_path,
        sounddevice_module=sounddevice,
        soundfile_module=FakeSoundFile(),
    )

    assert alarm.start() is True
    assert alarm.state == AlarmState.PLAYING
    assert sounddevice.play_calls == 1


def test_repeated_start_does_not_duplicate_playback(tmp_path: Path) -> None:
    alarm_path = _alarm_file(tmp_path)
    sounddevice = FakeSoundDevice()
    alarm = AudioAlarm(
        alarm_path,
        sounddevice_module=sounddevice,
        soundfile_module=FakeSoundFile(),
    )

    assert alarm.start() is True
    assert alarm.start() is False

    assert sounddevice.play_calls == 1


def test_stop_alarm(tmp_path: Path) -> None:
    alarm_path = _alarm_file(tmp_path)
    sounddevice = FakeSoundDevice()
    alarm = AudioAlarm(
        alarm_path,
        sounddevice_module=sounddevice,
        soundfile_module=FakeSoundFile(),
    )

    alarm.start()

    assert alarm.stop() is True
    assert alarm.state == AlarmState.IDLE
    assert sounddevice.stop_calls == 1


def test_repeated_stop_is_safe(tmp_path: Path) -> None:
    alarm_path = _alarm_file(tmp_path)
    sounddevice = FakeSoundDevice()
    alarm = AudioAlarm(
        alarm_path,
        sounddevice_module=sounddevice,
        soundfile_module=FakeSoundFile(),
    )

    assert alarm.stop() is False
    alarm.start()
    assert alarm.stop() is True
    assert alarm.stop() is False

    assert sounddevice.stop_calls == 1


def test_playback_failure_marks_audio_unavailable(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    alarm_path = _alarm_file(tmp_path)
    sounddevice = FakeSoundDevice(fail_play=True)
    alarm = AudioAlarm(
        alarm_path,
        sounddevice_module=sounddevice,
        soundfile_module=FakeSoundFile(),
    )

    with caplog.at_level(logging.ERROR):
        assert alarm.start() is False

    assert alarm.state == AlarmState.ERROR
    assert alarm.audio_available is False
    assert sounddevice.play_calls == 1
    assert "Alarm playback failed" in caplog.text

    assert alarm.start() is False
    assert sounddevice.play_calls == 1


def test_missing_audio_file_produces_clear_error(tmp_path: Path) -> None:
    with pytest.raises(AudioError, match="Alarm file not found"):
        AudioAlarm(
            tmp_path / "missing.wav",
            sounddevice_module=FakeSoundDevice(),
            soundfile_module=FakeSoundFile(),
        )


def test_malformed_wav_produces_clear_error(tmp_path: Path) -> None:
    with pytest.raises(AudioError, match="Unable to load alarm audio"):
        AudioAlarm(
            _alarm_file(tmp_path),
            sounddevice_module=FakeSoundDevice(),
            soundfile_module=FakeSoundFile(fail_read=True),
        )


def test_audio_stop_failure_is_recoverable(tmp_path: Path) -> None:
    alarm_path = _alarm_file(tmp_path)
    sounddevice = FakeSoundDevice(fail_stop=True)
    alarm = AudioAlarm(
        alarm_path,
        sounddevice_module=sounddevice,
        soundfile_module=FakeSoundFile(),
    )

    alarm.start()

    assert alarm.stop() is False
    assert alarm.state == AlarmState.ERROR
    assert alarm.audio_available is False


def test_cleanup_stops_and_releases_loaded_audio(tmp_path: Path) -> None:
    alarm_path = _alarm_file(tmp_path)
    sounddevice = FakeSoundDevice()
    alarm = AudioAlarm(
        alarm_path,
        sounddevice_module=sounddevice,
        soundfile_module=FakeSoundFile(),
    )

    alarm.start()
    alarm.close()

    assert alarm.state == AlarmState.IDLE
    assert alarm.data is None
    assert alarm.sample_rate is None
    assert sounddevice.stop_calls == 1


def _alarm_file(tmp_path: Path) -> Path:
    alarm_path = tmp_path / "alarm.wav"
    alarm_path.write_text("fake wav", encoding="utf-8")
    return alarm_path
