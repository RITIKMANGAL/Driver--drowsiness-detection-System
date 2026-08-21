"""Frame validation and aspect-ratio-safe preprocessing."""

from __future__ import annotations

import cv2
import numpy as np

from driver_drowsiness.exceptions import DetectionError


def validate_frame(frame: np.ndarray | None) -> np.ndarray:
    if frame is None:
        raise DetectionError("Frame must not be None.")
    if not isinstance(frame, np.ndarray):
        raise DetectionError("Frame must be a numpy array.")
    if frame.size == 0:
        raise DetectionError("Frame must not be empty.")
    if frame.ndim not in {2, 3}:
        raise DetectionError("Frame must be a grayscale or color image.")
    if frame.ndim == 3 and frame.shape[2] not in {1, 3, 4}:
        raise DetectionError("Color frame must have 1, 3, or 4 channels.")
    if not np.issubdtype(frame.dtype, np.integer) and not np.issubdtype(
        frame.dtype,
        np.floating,
    ):
        raise DetectionError("Frame dtype must be numeric.")
    if not np.isfinite(frame).all():
        raise DetectionError("Frame must contain only finite values.")
    return frame


def preprocess_frame(
    frame: np.ndarray | None,
    *,
    max_width: int,
    max_height: int,
    processing_scale: float,
) -> np.ndarray:
    """Resize without distorting aspect ratio and without upscaling."""

    frame = validate_frame(frame)
    original_height, original_width = frame.shape[:2]
    if max_width <= 0 or max_height <= 0:
        raise DetectionError("Processing dimensions must be positive.")
    if processing_scale <= 0 or processing_scale > 1:
        raise DetectionError("Processing scale must be > 0 and <= 1.")

    scale = min(
        processing_scale,
        max_width / original_width,
        max_height / original_height,
        1.0,
    )
    target_width = max(1, int(round(original_width * scale)))
    target_height = max(1, int(round(original_height * scale)))

    if target_width == original_width and target_height == original_height:
        return frame

    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
