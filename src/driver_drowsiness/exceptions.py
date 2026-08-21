"""Application-specific exceptions."""


class DriverDrowsinessError(Exception):
    """Base exception for driver drowsiness detection errors."""


class ConfigurationError(DriverDrowsinessError):
    """Raised when application configuration is invalid."""


class ModelLoadError(DriverDrowsinessError):
    """Raised when the facial landmark model cannot be loaded."""


class CameraError(DriverDrowsinessError):
    """Raised when camera initialization or frame capture fails."""


class AudioError(DriverDrowsinessError):
    """Raised when alarm audio cannot be loaded or played."""


class DetectionError(DriverDrowsinessError):
    """Raised when a frame cannot be processed for detection."""


class ApplicationError(DriverDrowsinessError):
    """Raised when application orchestration fails."""
