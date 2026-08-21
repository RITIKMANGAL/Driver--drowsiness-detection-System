from driver_drowsiness.state import (
    ConsecutiveSignalTracker,
    DrowsinessState,
    DrowsinessTracker,
)


def test_tracker_initial_state() -> None:
    tracker = DrowsinessTracker(ear_threshold=0.3, consecutive_frame_limit=30)

    assert tracker.consecutive_closed_frames == 0
    assert tracker.is_drowsy is False


def test_tracker_open_eyes_are_attentive() -> None:
    tracker = DrowsinessTracker(ear_threshold=0.3, consecutive_frame_limit=30)

    update = tracker.update(0.4)

    assert update.drowsiness_state == DrowsinessState.ATTENTIVE
    assert update.consecutive_closed_frames == 0
    assert not update.alert_required


def test_tracker_counts_closed_eye_frames_and_alerts_at_limit() -> None:
    tracker = DrowsinessTracker(ear_threshold=0.3, consecutive_frame_limit=3)

    first = tracker.update(0.2)
    second = tracker.update(0.2)
    third = tracker.update(0.2)

    assert first.drowsiness_state == DrowsinessState.EYES_CLOSED
    assert first.consecutive_closed_frames == 1
    assert not first.alert_required
    assert second.consecutive_closed_frames == 2
    assert not second.alert_required
    assert third.drowsiness_state == DrowsinessState.DROWSY
    assert third.alert_required
    assert third.consecutive_closed_frames == 3


def test_tracker_29_closed_frames_is_not_drowsy() -> None:
    tracker = DrowsinessTracker(ear_threshold=0.3, consecutive_frame_limit=30)

    update = None
    for _ in range(29):
        update = tracker.update(0.2)

    assert update is not None
    assert update.drowsiness_state == DrowsinessState.EYES_CLOSED
    assert update.consecutive_closed_frames == 29
    assert not update.alert_required


def test_tracker_30_closed_frames_transitions_to_drowsy() -> None:
    tracker = DrowsinessTracker(ear_threshold=0.3, consecutive_frame_limit=30)

    update = None
    for _ in range(30):
        update = tracker.update(0.2)

    assert update is not None
    assert update.drowsiness_state == DrowsinessState.DROWSY
    assert update.consecutive_closed_frames == 30
    assert update.alert_required


def test_tracker_more_than_30_closed_frames_remains_drowsy_without_realert() -> None:
    tracker = DrowsinessTracker(ear_threshold=0.3, consecutive_frame_limit=30)

    for _ in range(30):
        tracker.update(0.2)
    update = tracker.update(0.2)

    assert update.drowsiness_state == DrowsinessState.DROWSY
    assert update.consecutive_closed_frames == 31
    assert not update.alert_required


def test_tracker_resets_when_eyes_reopen() -> None:
    tracker = DrowsinessTracker(ear_threshold=0.3, consecutive_frame_limit=3)

    for _ in range(3):
        tracker.update(0.2)
    reopened = tracker.update(0.4)

    assert reopened.drowsiness_state == DrowsinessState.ATTENTIVE
    assert reopened.consecutive_closed_frames == 0
    assert not reopened.eyes_closed
    assert tracker.is_drowsy is False


def test_tracker_resets_count_when_no_face_is_detected() -> None:
    tracker = DrowsinessTracker(ear_threshold=0.3, consecutive_frame_limit=3)

    tracker.update(0.2)
    no_face = tracker.update(None)

    assert no_face.drowsiness_state == DrowsinessState.NO_FACE
    assert no_face.consecutive_closed_frames == 0
    assert not no_face.alert_required


def test_tracker_no_face_recovers_from_drowsy_state() -> None:
    tracker = DrowsinessTracker(ear_threshold=0.3, consecutive_frame_limit=3)

    for _ in range(3):
        tracker.update(0.2)
    no_face = tracker.update(None)

    assert no_face.drowsiness_state == DrowsinessState.NO_FACE
    assert no_face.consecutive_closed_frames == 0
    assert tracker.is_drowsy is False


def test_tracker_supporting_signals_alone_can_transition_to_drowsy() -> None:
    tracker = DrowsinessTracker(ear_threshold=0.3, consecutive_frame_limit=30)

    update = tracker.update(0.4, yawning_detected=True, head_tilt_detected=True)

    assert update.drowsiness_state == DrowsinessState.DROWSY
    assert update.alert_required
    assert not update.eyes_closed


def test_tracker_one_supporting_signal_alone_is_not_drowsy() -> None:
    tracker = DrowsinessTracker(ear_threshold=0.3, consecutive_frame_limit=30)

    update = tracker.update(0.4, yawning_detected=True, head_tilt_detected=False)

    assert update.drowsiness_state == DrowsinessState.ATTENTIVE
    assert not update.alert_required


def test_consecutive_signal_tracker_requires_persistence() -> None:
    tracker = ConsecutiveSignalTracker(consecutive_frame_limit=2)

    first = tracker.update(True)
    second = tracker.update(True)

    assert not first.detected
    assert second.detected
    assert second.transitioned_to_detected


def test_consecutive_signal_tracker_recovers() -> None:
    tracker = ConsecutiveSignalTracker(consecutive_frame_limit=2)
    tracker.update(True)
    tracker.update(True)

    recovered = tracker.update(False)

    assert not recovered.detected
    assert recovered.consecutive_frames == 0
    assert recovered.transitioned_to_clear
