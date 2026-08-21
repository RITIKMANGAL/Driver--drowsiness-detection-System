"""Drowsiness state tracking."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DrowsinessState(StrEnum):
    NO_FACE = "no_face"
    INVALID_EAR = "invalid_ear"
    ATTENTIVE = "attentive"
    EYES_CLOSED = "eyes_closed"
    DROWSY = "drowsy"


@dataclass(frozen=True)
class StateUpdate:
    eyes_closed: bool
    consecutive_closed_frames: int
    drowsiness_state: DrowsinessState
    alert_required: bool


@dataclass(frozen=True)
class SignalUpdate:
    detected: bool
    consecutive_frames: int
    transitioned_to_detected: bool
    transitioned_to_clear: bool


@dataclass
class ConsecutiveSignalTracker:
    consecutive_frame_limit: int
    consecutive_frames: int = 0
    detected: bool = False

    def update(self, active: bool | None) -> SignalUpdate:
        previous = self.detected
        if active is None or not active:
            self.consecutive_frames = 0
            self.detected = False
        else:
            self.consecutive_frames += 1
            self.detected = self.consecutive_frames >= self.consecutive_frame_limit

        return SignalUpdate(
            detected=self.detected,
            consecutive_frames=self.consecutive_frames,
            transitioned_to_detected=self.detected and not previous,
            transitioned_to_clear=previous and not self.detected,
        )


@dataclass
class DrowsinessTracker:
    ear_threshold: float = 0.3
    consecutive_frame_limit: int = 30
    consecutive_closed_frames: int = 0
    is_drowsy: bool = False

    def update(
        self,
        average_ear: float | None,
        *,
        yawning_detected: bool = False,
        head_tilt_detected: bool = False,
    ) -> StateUpdate:
        if average_ear is None:
            self.consecutive_closed_frames = 0
            self.is_drowsy = False
            return StateUpdate(
                eyes_closed=False,
                consecutive_closed_frames=self.consecutive_closed_frames,
                drowsiness_state=DrowsinessState.NO_FACE,
                alert_required=False,
            )

        eyes_closed = average_ear <= self.ear_threshold
        supporting_signals_drowsy = yawning_detected and head_tilt_detected
        if not eyes_closed:
            self.consecutive_closed_frames = 0
            if supporting_signals_drowsy:
                alert_required = not self.is_drowsy
                self.is_drowsy = True
                return StateUpdate(
                    eyes_closed=False,
                    consecutive_closed_frames=self.consecutive_closed_frames,
                    drowsiness_state=DrowsinessState.DROWSY,
                    alert_required=alert_required,
                )

            self.is_drowsy = False
            return StateUpdate(
                eyes_closed=False,
                consecutive_closed_frames=self.consecutive_closed_frames,
                drowsiness_state=DrowsinessState.ATTENTIVE,
                alert_required=False,
            )

        self.consecutive_closed_frames += 1
        if (
            self.consecutive_closed_frames >= self.consecutive_frame_limit
            or supporting_signals_drowsy
        ):
            alert_required = not self.is_drowsy
            self.is_drowsy = True
            return StateUpdate(
                eyes_closed=True,
                consecutive_closed_frames=self.consecutive_closed_frames,
                drowsiness_state=DrowsinessState.DROWSY,
                alert_required=alert_required,
            )

        self.is_drowsy = False
        return StateUpdate(
            eyes_closed=True,
            consecutive_closed_frames=self.consecutive_closed_frames,
            drowsiness_state=DrowsinessState.EYES_CLOSED,
            alert_required=False,
        )
