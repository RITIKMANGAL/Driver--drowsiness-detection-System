from __future__ import annotations

from pathlib import Path

import pytest

from driver_drowsiness.app import _overrides_from_args, build_parser
from driver_drowsiness.config import ConfigOverrides, load_config
from driver_drowsiness.exceptions import ConfigurationError


def test_default_configuration_preserves_detection_defaults() -> None:
    config = load_config()

    assert config.camera.index == 0
    assert config.camera.width == 512
    assert config.camera.height == 512
    assert config.camera.processing_scale == 1.0
    assert config.detection.ear_threshold == 0.3
    assert config.detection.closed_frames_threshold == 30
    assert config.yawning.enabled is True
    assert config.yawning.mar_threshold == 0.6
    assert config.yawning.consecutive_frames == 15
    assert config.head_pose.enabled is True
    assert config.head_pose.roll_threshold_degrees == 20.0
    assert config.head_pose.downward_ratio_threshold == 0.65
    assert config.head_pose.consecutive_frames == 15
    assert config.alert.enabled is True
    assert config.logging.level == "INFO"


def test_valid_yaml_configuration(tmp_path: Path) -> None:
    config_path, model_path, alarm_path = _write_config(
        tmp_path,
        """
camera:
  index: 1
  width: 320
  height: 240
  processing_scale: 0.5
detection:
  ear_threshold: 0.25
  closed_frames_threshold: 20
yawning:
  enabled: true
  mar_threshold: 0.7
  consecutive_frames: 8
head_pose:
  enabled: true
  roll_threshold_degrees: 25
  downward_ratio_threshold: 0.7
  consecutive_frames: 9
assets:
  model_path: model.dat
  alarm_path: alarm.wav
alert:
  enabled: true
logging:
  level: DEBUG
""",
    )

    config = load_config(config_path)

    assert config.camera.index == 1
    assert config.camera.processing_size == (160, 120)
    assert config.detection.ear_threshold == 0.25
    assert config.detection.closed_frames_threshold == 20
    assert config.yawning.mar_threshold == 0.7
    assert config.yawning.consecutive_frames == 8
    assert config.head_pose.roll_threshold_degrees == 25
    assert config.head_pose.downward_ratio_threshold == 0.7
    assert config.head_pose.consecutive_frames == 9
    assert config.assets.model_path == model_path
    assert config.assets.alarm_path == alarm_path
    assert config.logging.level == "DEBUG"


def test_missing_configuration_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="Configuration file not found"):
        load_config(tmp_path / "missing.yml")


def test_invalid_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("camera: [", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Invalid YAML"):
        load_config(config_path)


def test_missing_sections_are_filled_from_defaults(tmp_path: Path) -> None:
    config_path, _, _ = _write_config(
        tmp_path,
        """
assets:
  model_path: model.dat
  alarm_path: alarm.wav
""",
    )

    config = load_config(config_path)

    assert config.camera.index == 0
    assert config.detection.ear_threshold == 0.3
    assert config.alert.enabled is True


def test_invalid_camera_index(tmp_path: Path) -> None:
    config_path, _, _ = _write_config(
        tmp_path,
        """
camera:
  index: -1
assets:
  model_path: model.dat
  alarm_path: alarm.wav
""",
    )

    with pytest.raises(ConfigurationError, match="camera.index must be >= 0"):
        load_config(config_path)


def test_invalid_frame_width(tmp_path: Path) -> None:
    config_path, _, _ = _write_config(
        tmp_path,
        """
camera:
  width: 0
assets:
  model_path: model.dat
  alarm_path: alarm.wav
""",
    )

    with pytest.raises(ConfigurationError, match="camera.width must be >= 1"):
        load_config(config_path)


def test_invalid_frame_height(tmp_path: Path) -> None:
    config_path, _, _ = _write_config(
        tmp_path,
        """
camera:
  height: 0
assets:
  model_path: model.dat
  alarm_path: alarm.wav
""",
    )

    with pytest.raises(ConfigurationError, match="camera.height must be >= 1"):
        load_config(config_path)


def test_invalid_processing_scale(tmp_path: Path) -> None:
    config_path, _, _ = _write_config(
        tmp_path,
        """
camera:
  processing_scale: 1.5
assets:
  model_path: model.dat
  alarm_path: alarm.wav
""",
    )

    with pytest.raises(ConfigurationError, match="camera.processing_scale must be <= 1"):
        load_config(config_path)


def test_invalid_ear_threshold(tmp_path: Path) -> None:
    config_path, _, _ = _write_config(
        tmp_path,
        """
detection:
  ear_threshold: 1
assets:
  model_path: model.dat
  alarm_path: alarm.wav
""",
    )

    with pytest.raises(ConfigurationError, match="detection.ear_threshold must be < 1"):
        load_config(config_path)


def test_invalid_closed_frame_threshold(tmp_path: Path) -> None:
    config_path, _, _ = _write_config(
        tmp_path,
        """
detection:
  closed_frames_threshold: 0
assets:
  model_path: model.dat
  alarm_path: alarm.wav
""",
    )

    with pytest.raises(
        ConfigurationError,
        match="detection.closed_frames_threshold must be >= 1",
    ):
        load_config(config_path)


def test_invalid_yawning_enabled(tmp_path: Path) -> None:
    config_path, _, _ = _write_config(
        tmp_path,
        """
assets:
  model_path: model.dat
  alarm_path: alarm.wav
yawning:
  enabled: yes please
""",
    )

    with pytest.raises(ConfigurationError, match="yawning.enabled must be a boolean"):
        load_config(config_path)


def test_invalid_mar_threshold(tmp_path: Path) -> None:
    config_path, _, _ = _write_config(
        tmp_path,
        """
assets:
  model_path: model.dat
  alarm_path: alarm.wav
yawning:
  mar_threshold: 0
""",
    )

    with pytest.raises(ConfigurationError, match="yawning.mar_threshold must be > 0"):
        load_config(config_path)


def test_invalid_yawning_consecutive_frames(tmp_path: Path) -> None:
    config_path, _, _ = _write_config(
        tmp_path,
        """
assets:
  model_path: model.dat
  alarm_path: alarm.wav
yawning:
  consecutive_frames: 0
""",
    )

    with pytest.raises(ConfigurationError, match="yawning.consecutive_frames"):
        load_config(config_path)


def test_invalid_head_pose_enabled(tmp_path: Path) -> None:
    config_path, _, _ = _write_config(
        tmp_path,
        """
assets:
  model_path: model.dat
  alarm_path: alarm.wav
head_pose:
  enabled: yes please
""",
    )

    with pytest.raises(ConfigurationError, match="head_pose.enabled must be a boolean"):
        load_config(config_path)


def test_invalid_head_pose_roll_threshold(tmp_path: Path) -> None:
    config_path, _, _ = _write_config(
        tmp_path,
        """
assets:
  model_path: model.dat
  alarm_path: alarm.wav
head_pose:
  roll_threshold_degrees: 0
""",
    )

    with pytest.raises(
        ConfigurationError,
        match="head_pose.roll_threshold_degrees must be > 0",
    ):
        load_config(config_path)


def test_invalid_head_pose_downward_threshold(tmp_path: Path) -> None:
    config_path, _, _ = _write_config(
        tmp_path,
        """
assets:
  model_path: model.dat
  alarm_path: alarm.wav
head_pose:
  downward_ratio_threshold: 0
""",
    )

    with pytest.raises(
        ConfigurationError,
        match="head_pose.downward_ratio_threshold must be > 0",
    ):
        load_config(config_path)


def test_invalid_head_pose_consecutive_frames(tmp_path: Path) -> None:
    config_path, _, _ = _write_config(
        tmp_path,
        """
assets:
  model_path: model.dat
  alarm_path: alarm.wav
head_pose:
  consecutive_frames: 0
""",
    )

    with pytest.raises(ConfigurationError, match="head_pose.consecutive_frames"):
        load_config(config_path)


def test_invalid_alarm_boolean(tmp_path: Path) -> None:
    config_path, _, _ = _write_config(
        tmp_path,
        """
assets:
  model_path: model.dat
  alarm_path: alarm.wav
alert:
  enabled: yes please
""",
    )

    with pytest.raises(ConfigurationError, match="alert.enabled must be a boolean"):
        load_config(config_path)


def test_invalid_logging_level(tmp_path: Path) -> None:
    config_path, _, _ = _write_config(
        tmp_path,
        """
assets:
  model_path: model.dat
  alarm_path: alarm.wav
logging:
  level: VERBOSE
""",
    )

    with pytest.raises(ConfigurationError, match="logging.level is invalid"):
        load_config(config_path)


def test_missing_model_asset(tmp_path: Path) -> None:
    config_path, _, _ = _write_config(
        tmp_path,
        """
assets:
  model_path: missing.dat
  alarm_path: alarm.wav
""",
    )

    with pytest.raises(ConfigurationError, match="assets.model_path does not exist"):
        load_config(config_path)


def test_missing_alarm_asset(tmp_path: Path) -> None:
    config_path, _, _ = _write_config(
        tmp_path,
        """
assets:
  model_path: model.dat
  alarm_path: missing.wav
""",
    )

    with pytest.raises(ConfigurationError, match="assets.alarm_path does not exist"):
        load_config(config_path)


def test_relative_path_resolution(tmp_path: Path) -> None:
    config_path, model_path, alarm_path = _write_config(
        tmp_path,
        """
assets:
  model_path: model.dat
  alarm_path: alarm.wav
""",
    )

    config = load_config(config_path)

    assert config.assets.model_path == model_path
    assert config.assets.alarm_path == alarm_path


def test_absolute_path_handling(tmp_path: Path) -> None:
    model_path = tmp_path / "absolute-model.dat"
    alarm_path = tmp_path / "absolute-alarm.wav"
    model_path.write_text("model", encoding="utf-8")
    alarm_path.write_text("alarm", encoding="utf-8")
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        f"""
assets:
  model_path: {model_path.as_posix()}
  alarm_path: {alarm_path.as_posix()}
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.assets.model_path == model_path
    assert config.assets.alarm_path == alarm_path


def test_cli_overrides_are_parsed() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--camera",
            "2",
            "--width",
            "640",
            "--height",
            "480",
            "--processing-scale",
            "0.5",
            "--ear-threshold",
            "0.25",
            "--closed-frames",
            "15",
            "--model",
            "model.dat",
            "--alarm",
            "alarm.wav",
            "--log-level",
            "debug",
        ]
    )

    overrides = _overrides_from_args(args)

    assert overrides.camera_index == 2
    assert overrides.width == 640
    assert overrides.height == 480
    assert overrides.processing_scale == 0.5
    assert overrides.ear_threshold == 0.25
    assert overrides.closed_frames_threshold == 15
    assert overrides.model_path == Path("model.dat")
    assert overrides.alarm_path == Path("alarm.wav")
    assert overrides.log_level == "debug"


def test_cli_yaml_default_precedence(tmp_path: Path) -> None:
    config_path, _, _ = _write_config(
        tmp_path,
        """
detection:
  ear_threshold: 0.4
  closed_frames_threshold: 22
assets:
  model_path: model.dat
  alarm_path: alarm.wav
""",
    )

    config = load_config(
        config_path,
        ConfigOverrides(
            ear_threshold=0.25,
            closed_frames_threshold=10,
        ),
    )

    assert config.detection.ear_threshold == 0.25
    assert config.detection.closed_frames_threshold == 10
    assert config.camera.index == 0


def test_disable_alarm_behavior_skips_alarm_path_validation(tmp_path: Path) -> None:
    config_path, _, _ = _write_config(
        tmp_path,
        """
assets:
  model_path: model.dat
  alarm_path: missing.wav
alert:
  enabled: true
""",
    )

    config = load_config(config_path, ConfigOverrides(alert_enabled=False))

    assert config.alert.enabled is False
    assert config.assets.alarm_path == tmp_path / "missing.wav"


def _write_config(tmp_path: Path, content: str) -> tuple[Path, Path, Path]:
    model_path = tmp_path / "model.dat"
    alarm_path = tmp_path / "alarm.wav"
    model_path.write_text("model", encoding="utf-8")
    alarm_path.write_text("alarm", encoding="utf-8")
    config_path = tmp_path / "config.yml"
    config_path.write_text(content, encoding="utf-8")
    return config_path, model_path, alarm_path
