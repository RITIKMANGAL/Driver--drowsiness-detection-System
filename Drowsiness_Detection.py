"""Compatibility entry point for the package application."""

from driver_drowsiness.app import main
from driver_drowsiness.ear import eye_aspect_ratio as eyeaspectratio

__all__ = ["eyeaspectratio", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
