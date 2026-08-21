import numpy as np
import pytest

from driver_drowsiness.mar import mouth_aspect_ratio


def test_mouth_aspect_ratio_normal_mouth() -> None:
    assert mouth_aspect_ratio(_mouth(vertical=1.0)) == pytest.approx(0.2)


def test_mouth_aspect_ratio_open_mouth() -> None:
    assert mouth_aspect_ratio(_mouth(vertical=4.0)) == pytest.approx(0.8)


def test_mouth_aspect_ratio_rejects_zero_width() -> None:
    mouth = _mouth(vertical=1.0)
    mouth[6] = mouth[0]

    with pytest.raises(ValueError, match="non-zero"):
        mouth_aspect_ratio(mouth)


def test_mouth_aspect_ratio_rejects_invalid_landmarks() -> None:
    with pytest.raises(ValueError, match="20x2"):
        mouth_aspect_ratio(np.zeros((2, 2), dtype=float))


def _mouth(vertical: float) -> np.ndarray:
    mouth = np.zeros((20, 2), dtype=float)
    mouth[0] = [0, 0]
    mouth[6] = [10, 0]
    mouth[2] = [2, vertical]
    mouth[10] = [2, -vertical]
    mouth[3] = [5, vertical]
    mouth[9] = [5, -vertical]
    mouth[4] = [8, vertical]
    mouth[8] = [8, -vertical]
    return mouth
