"""Alarm audio lifecycle management."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
import soundfile as sf

from driver_drowsiness.exceptions import AudioError

LOGGER = logging.getLogger(__name__)


class AlarmState(StrEnum):
    DISABLED = "disabled"
    IDLE = "idle"
    PLAYING = "playing"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class AudioAlarm:
    """Load and control local WAV alarm playback."""

    path: Path
    enabled: bool = True
    sounddevice_module: Any = sd
    soundfile_module: Any = sf
    logger: logging.Logger = LOGGER
    data: np.ndarray | None = field(default=None, init=False)
    sample_rate: int | None = field(default=None, init=False)
    state: AlarmState = field(default=AlarmState.IDLE, init=False)
    audio_available: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if not self.enabled:
            self.state = AlarmState.DISABLED
            self.audio_available = False
            self.logger.info("Audio alarm disabled by configuration.")
            return

        alarm_path = Path(self.path)
        if not alarm_path.is_file():
            self.state = AlarmState.ERROR
            self.audio_available = False
            raise AudioError(f"Alarm file not found: {alarm_path}")

        try:
            self.data, self.sample_rate = self.soundfile_module.read(
                alarm_path,
                dtype="float32",
            )
        except Exception as exc:
            self.state = AlarmState.ERROR
            self.audio_available = False
            raise AudioError(f"Unable to load alarm audio: {alarm_path}") from exc

        self.state = AlarmState.IDLE
        self.audio_available = True
        self.logger.info("Alarm audio initialized.")

    def start(self) -> bool:
        """Start alarm playback once; repeated starts while playing are ignored."""

        if not self.enabled:
            return False
        if self.state == AlarmState.PLAYING:
            return False
        if not self.audio_available or self.state == AlarmState.ERROR:
            self.logger.warning("Alarm audio unavailable; alert audio skipped.")
            return False
        if self.data is None or self.sample_rate is None:
            self.state = AlarmState.ERROR
            self.audio_available = False
            self.logger.error("Alarm audio is not loaded; alert audio disabled.")
            return False

        try:
            self.sounddevice_module.play(self.data, self.sample_rate)
        except Exception as exc:
            self.state = AlarmState.ERROR
            self.audio_available = False
            self.logger.error("Alarm playback failed; alert audio disabled: %s", exc)
            return False

        self.state = AlarmState.PLAYING
        self.logger.info("Alarm playback started.")
        return True

    def play(self) -> bool:
        """Backward-compatible alias for start()."""

        return self.start()

    def stop(self) -> bool:
        """Stop alarm playback. Safe to call repeatedly."""

        if not self.enabled or self.state == AlarmState.DISABLED:
            return False
        if self.state != AlarmState.PLAYING:
            return False

        self.state = AlarmState.STOPPING
        try:
            self.sounddevice_module.stop()
        except Exception as exc:
            self.state = AlarmState.ERROR
            self.audio_available = False
            self.logger.warning("Alarm stop failed; audio marked unavailable: %s", exc)
            return False

        self.state = AlarmState.IDLE
        self.logger.info("Alarm playback stopped.")
        return True

    def close(self) -> None:
        """Stop playback and release in-memory audio references."""

        self.stop()
        self.data = None
        self.sample_rate = None
        if self.enabled and self.state != AlarmState.ERROR:
            self.state = AlarmState.IDLE

    def cleanup(self) -> None:
        """Backward-compatible cleanup alias."""

        self.close()
