from driver_drowsiness.exceptions import (
    ApplicationError,
    AudioError,
    CameraError,
    ConfigurationError,
    DetectionError,
    DriverDrowsinessError,
    ModelLoadError,
)


def test_application_exceptions_share_base_type() -> None:
    for exception_type in [
        ApplicationError,
        AudioError,
        CameraError,
        ConfigurationError,
        DetectionError,
        ModelLoadError,
    ]:
        assert issubclass(exception_type, DriverDrowsinessError)


def test_exception_message_is_preserved() -> None:
    error = ApplicationError("OpenCV GUI operation failed.")

    assert str(error) == "OpenCV GUI operation failed."
