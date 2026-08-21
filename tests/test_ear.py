import numpy as np
import pytest

from driver_drowsiness.ear import average_eye_aspect_ratio, eye_aspect_ratio


def test_eye_aspect_ratio_matches_legacy_formula() -> None:
    assert eye_aspect_ratio(_eye(vertical=2)) == pytest.approx(1.0)


def test_low_eye_aspect_ratio() -> None:
    assert eye_aspect_ratio(_eye(vertical=0.4)) == pytest.approx(0.2)


def test_high_eye_aspect_ratio() -> None:
    assert eye_aspect_ratio(_eye(vertical=3)) == pytest.approx(1.5)


def test_average_eye_aspect_ratio_averages_both_eyes() -> None:
    open_eye = _eye(vertical=2)
    narrow_eye = _eye(vertical=1)

    assert average_eye_aspect_ratio(open_eye, narrow_eye) == pytest.approx(0.75)


def test_left_and_right_eye_aspect_ratios_are_independent() -> None:
    left_eye = _eye(vertical=0.4)
    right_eye = _eye(vertical=2)

    assert eye_aspect_ratio(left_eye) == pytest.approx(0.2)
    assert eye_aspect_ratio(right_eye) == pytest.approx(1.0)


def test_eye_aspect_ratio_rejects_zero_horizontal_distance() -> None:
    eye = _eye(vertical=2)
    eye[3] = eye[0]

    with pytest.raises(ValueError, match="non-zero"):
        eye_aspect_ratio(eye)


def test_eye_aspect_ratio_rejects_non_finite_values() -> None:
    eye = _eye(vertical=2)
    eye[1, 1] = np.nan

    with pytest.raises(ValueError, match="finite"):
        eye_aspect_ratio(eye)


def test_eye_aspect_ratio_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="6x2"):
        eye_aspect_ratio(np.array([[0, 0], [1, 1]]))


def _eye(vertical: float) -> np.ndarray:
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
    )
