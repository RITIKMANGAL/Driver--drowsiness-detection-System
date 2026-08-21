"""Typed configuration loading, validation, and path resolution."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from driver_drowsiness.exceptions import ConfigurationError

PROJECT_NAME = "driver-drowsiness-detection-system"
DEFAULT_CONFIG_FILENAME = "config.yml"
DEFAULT_MODEL_FILENAME = "shape_predictor_68_face_landmarks.dat"
DEFAULT_ALARM_FILENAME = "loud_alarm.wav"


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 512
    height: int = 512
    processing_scale: float = 1.0

    @property
    def frame_size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def processing_size(self) -> tuple[int, int]:
        return (
            max(1, int(self.width * self.processing_scale)),
            max(1, int(self.height * self.processing_scale)),
        )


@dataclass(frozen=True)
class DetectionConfig:
    ear_threshold: float = 0.3
    closed_frames_threshold: int = 30


@dataclass(frozen=True)
class YawningConfig:
    enabled: bool = True
    mar_threshold: float = 0.6
    consecutive_frames: int = 15


@dataclass(frozen=True)
class HeadPoseConfig:
    enabled: bool = True
    roll_threshold_degrees: float = 20.0
    downward_ratio_threshold: float = 0.65
    consecutive_frames: int = 15


@dataclass(frozen=True)
class AssetConfig:
    model_path: Path
    alarm_path: Path


@dataclass(frozen=True)
class AlertConfig:
    enabled: bool = True


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"

    @property
    def level_value(self) -> int:
        return logging.getLevelName(self.level)


@dataclass(frozen=True)
class ApplicationConfig:
    camera: CameraConfig
    detection: DetectionConfig
    yawning: YawningConfig
    head_pose: HeadPoseConfig
    assets: AssetConfig
    alert: AlertConfig
    logging: LoggingConfig
    config_path: Path | None = None


@dataclass(frozen=True)
class ConfigOverrides:
    config_path: Path | None = None
    camera_index: int | None = None
    width: int | None = None
    height: int | None = None
    processing_scale: float | None = None
    ear_threshold: float | None = None
    closed_frames_threshold: int | None = None
    yawning_enabled: bool | None = None
    mar_threshold: float | None = None
    yawning_consecutive_frames: int | None = None
    head_pose_enabled: bool | None = None
    head_roll_threshold_degrees: float | None = None
    head_downward_ratio_threshold: float | None = None
    head_pose_consecutive_frames: int | None = None
    model_path: Path | None = None
    alarm_path: Path | None = None
    alert_enabled: bool | None = None
    log_level: str | None = None


AppConfig = ApplicationConfig


def load_config(
    config_path: str | Path | None = None,
    overrides: ConfigOverrides | None = None,
) -> ApplicationConfig:
    """Load defaults, optional YAML, and CLI overrides into typed config.

    Precedence is CLI overrides > configuration file > application defaults.
    Relative paths in YAML are resolved relative to the YAML file. Relative paths
    from CLI overrides are resolved relative to the current shell path because
    they are explicit user input.
    """

    overrides = overrides or ConfigOverrides()
    selected_config_path = Path(config_path) if config_path is not None else None
    if overrides.config_path is not None:
        selected_config_path = overrides.config_path

    discovered_config_path = (
        _resolve_config_path(selected_config_path)
        if selected_config_path is not None
        else discover_default_config_path()
    )
    config_base = discovered_config_path.parent if discovered_config_path else None

    raw_config = _load_yaml_file(discovered_config_path) if discovered_config_path else {}
    config = _config_from_mapping(raw_config, config_base, discovered_config_path)
    config = apply_overrides(config, overrides)
    validate_config(config)
    return config


def create_default_config() -> ApplicationConfig:
    """Create validated built-in defaults without requiring config.yml."""

    config = _config_from_mapping({}, None, None)
    validate_config(config)
    return config


def discover_default_config_path() -> Path | None:
    """Find a default config file without depending on the working directory."""

    for candidate in _default_search_roots(DEFAULT_CONFIG_FILENAME):
        if candidate.is_file():
            return candidate
    return None


def apply_overrides(
    config: ApplicationConfig,
    overrides: ConfigOverrides,
) -> ApplicationConfig:
    camera = config.camera
    detection = config.detection
    yawning = config.yawning
    head_pose = config.head_pose
    assets = config.assets
    alert = config.alert
    logging_config = config.logging

    if overrides.camera_index is not None:
        camera = replace(camera, index=overrides.camera_index)
    if overrides.width is not None:
        camera = replace(camera, width=overrides.width)
    if overrides.height is not None:
        camera = replace(camera, height=overrides.height)
    if overrides.processing_scale is not None:
        camera = replace(camera, processing_scale=overrides.processing_scale)
    if overrides.ear_threshold is not None:
        detection = replace(detection, ear_threshold=overrides.ear_threshold)
    if overrides.closed_frames_threshold is not None:
        detection = replace(
            detection,
            closed_frames_threshold=overrides.closed_frames_threshold,
        )
    if overrides.yawning_enabled is not None:
        yawning = replace(yawning, enabled=overrides.yawning_enabled)
    if overrides.mar_threshold is not None:
        yawning = replace(yawning, mar_threshold=overrides.mar_threshold)
    if overrides.yawning_consecutive_frames is not None:
        yawning = replace(
            yawning,
            consecutive_frames=overrides.yawning_consecutive_frames,
        )
    if overrides.head_pose_enabled is not None:
        head_pose = replace(head_pose, enabled=overrides.head_pose_enabled)
    if overrides.head_roll_threshold_degrees is not None:
        head_pose = replace(
            head_pose,
            roll_threshold_degrees=overrides.head_roll_threshold_degrees,
        )
    if overrides.head_downward_ratio_threshold is not None:
        head_pose = replace(
            head_pose,
            downward_ratio_threshold=overrides.head_downward_ratio_threshold,
        )
    if overrides.head_pose_consecutive_frames is not None:
        head_pose = replace(
            head_pose,
            consecutive_frames=overrides.head_pose_consecutive_frames,
        )
    if overrides.model_path is not None:
        assets = replace(assets, model_path=_resolve_user_path(overrides.model_path))
    if overrides.alarm_path is not None:
        assets = replace(assets, alarm_path=_resolve_user_path(overrides.alarm_path))
    if overrides.alert_enabled is not None:
        alert = replace(alert, enabled=overrides.alert_enabled)
    if overrides.log_level is not None:
        logging_config = replace(logging_config, level=overrides.log_level.upper())

    return ApplicationConfig(
        camera=camera,
        detection=detection,
        yawning=yawning,
        head_pose=head_pose,
        assets=assets,
        alert=alert,
        logging=logging_config,
        config_path=config.config_path,
    )


def validate_config(config: ApplicationConfig) -> None:
    _require_int("camera.index", config.camera.index, minimum=0)
    _require_int("camera.width", config.camera.width, minimum=1)
    _require_int("camera.height", config.camera.height, minimum=1)
    _require_number(
        "camera.processing_scale",
        config.camera.processing_scale,
        minimum=0,
        maximum=1,
        minimum_inclusive=False,
    )
    _require_number(
        "detection.ear_threshold",
        config.detection.ear_threshold,
        minimum=0,
        maximum=1,
        minimum_inclusive=False,
        maximum_inclusive=False,
    )
    _require_int(
        "detection.closed_frames_threshold",
        config.detection.closed_frames_threshold,
        minimum=1,
    )
    if not isinstance(config.yawning.enabled, bool):
        raise ConfigurationError("yawning.enabled must be a boolean.")
    _require_number(
        "yawning.mar_threshold",
        config.yawning.mar_threshold,
        minimum=0,
        maximum=2,
        minimum_inclusive=False,
    )
    _require_int(
        "yawning.consecutive_frames",
        config.yawning.consecutive_frames,
        minimum=1,
    )
    if not isinstance(config.head_pose.enabled, bool):
        raise ConfigurationError("head_pose.enabled must be a boolean.")
    _require_number(
        "head_pose.roll_threshold_degrees",
        config.head_pose.roll_threshold_degrees,
        minimum=0,
        maximum=90,
        minimum_inclusive=False,
    )
    _require_number(
        "head_pose.downward_ratio_threshold",
        config.head_pose.downward_ratio_threshold,
        minimum=0,
        maximum=1,
        minimum_inclusive=False,
    )
    _require_int(
        "head_pose.consecutive_frames",
        config.head_pose.consecutive_frames,
        minimum=1,
    )
    if not isinstance(config.alert.enabled, bool):
        raise ConfigurationError("alert.enabled must be a boolean.")
    _require_logging_level(config.logging.level)
    _require_file("assets.model_path", config.assets.model_path)
    if config.alert.enabled:
        _require_file("assets.alarm_path", config.assets.alarm_path)


def _config_from_mapping(
    raw_config: dict[str, Any],
    config_base: Path | None,
    config_path: Path | None,
) -> ApplicationConfig:
    if not isinstance(raw_config, dict):
        raise ConfigurationError("Configuration root must be a mapping.")

    allowed_sections = {
        "camera",
        "detection",
        "yawning",
        "head_pose",
        "assets",
        "alert",
        "logging",
    }
    unknown_sections = sorted(set(raw_config) - allowed_sections)
    if unknown_sections:
        raise ConfigurationError(
            "Unknown configuration section(s): " + ", ".join(unknown_sections)
        )

    camera_raw = _section(raw_config, "camera")
    detection_raw = _section(raw_config, "detection")
    yawning_raw = _section(raw_config, "yawning")
    head_pose_raw = _section(raw_config, "head_pose")
    assets_raw = _section(raw_config, "assets")
    alert_raw = _section(raw_config, "alert")
    logging_raw = _section(raw_config, "logging")

    camera = CameraConfig(
        index=_value(camera_raw, "index", CameraConfig.index),
        width=_value(camera_raw, "width", CameraConfig.width),
        height=_value(camera_raw, "height", CameraConfig.height),
        processing_scale=_value(
            camera_raw,
            "processing_scale",
            CameraConfig.processing_scale,
        ),
    )
    detection = DetectionConfig(
        ear_threshold=_value(
            detection_raw,
            "ear_threshold",
            DetectionConfig.ear_threshold,
        ),
        closed_frames_threshold=_value(
            detection_raw,
            "closed_frames_threshold",
            DetectionConfig.closed_frames_threshold,
        ),
    )
    yawning = YawningConfig(
        enabled=_value(yawning_raw, "enabled", YawningConfig.enabled),
        mar_threshold=_value(
            yawning_raw,
            "mar_threshold",
            YawningConfig.mar_threshold,
        ),
        consecutive_frames=_value(
            yawning_raw,
            "consecutive_frames",
            YawningConfig.consecutive_frames,
        ),
    )
    head_pose = HeadPoseConfig(
        enabled=_value(head_pose_raw, "enabled", HeadPoseConfig.enabled),
        roll_threshold_degrees=_value(
            head_pose_raw,
            "roll_threshold_degrees",
            HeadPoseConfig.roll_threshold_degrees,
        ),
        downward_ratio_threshold=_value(
            head_pose_raw,
            "downward_ratio_threshold",
            HeadPoseConfig.downward_ratio_threshold,
        ),
        consecutive_frames=_value(
            head_pose_raw,
            "consecutive_frames",
            HeadPoseConfig.consecutive_frames,
        ),
    )
    alert = AlertConfig(enabled=_value(alert_raw, "enabled", AlertConfig.enabled))
    logging_config = LoggingConfig(
        level=str(_value(logging_raw, "level", LoggingConfig.level)).upper()
    )
    assets = AssetConfig(
        model_path=_resolve_asset_config_path(
            _value(assets_raw, "model_path", DEFAULT_MODEL_FILENAME),
            config_base,
        ),
        alarm_path=_resolve_asset_config_path(
            _value(assets_raw, "alarm_path", DEFAULT_ALARM_FILENAME),
            config_base,
        ),
    )

    return ApplicationConfig(
        camera=camera,
        detection=detection,
        yawning=yawning,
        head_pose=head_pose,
        assets=assets,
        alert=alert,
        logging=logging_config,
        config_path=config_path,
    )


def _load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")
    if not path.is_file():
        raise ConfigurationError(f"Configuration path is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8") as config_file:
            loaded = yaml.safe_load(config_file)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in configuration file: {path}") from exc
    except OSError as exc:
        raise ConfigurationError(f"Unable to read configuration file: {path}") from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigurationError("Configuration root must be a mapping.")
    return loaded


def _resolve_config_path(path: Path) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved.resolve()


def _resolve_asset_config_path(value: object, config_base: Path | None) -> Path:
    if not isinstance(value, str | Path):
        raise ConfigurationError("Asset paths must be strings.")

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if config_base is not None:
        return (config_base / path).resolve()

    return resolve_default_asset_path(str(value))


def resolve_default_asset_path(filename: str) -> Path:
    for candidate in _default_search_roots(filename):
        if candidate.is_file():
            return candidate
    return _default_search_roots(filename)[0]


def _resolve_user_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return (Path.cwd() / expanded).resolve()


def _default_search_roots(filename: str) -> list[Path]:
    package_file = Path(__file__).resolve()
    source_root = package_file.parents[2]
    return [
        source_root / filename,
        Path(sys.prefix) / "share" / PROJECT_NAME / filename,
    ]


def _section(raw_config: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw_config.get(name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ConfigurationError(f"{name} must be a mapping.")
    return section


def _value(section: dict[str, Any], key: str, default: Any) -> Any:
    return section[key] if key in section else default


def _require_int(name: str, value: object, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigurationError(f"{name} must be an integer.")
    if value < minimum:
        raise ConfigurationError(f"{name} must be >= {minimum}.")


def _require_number(
    name: str,
    value: object,
    minimum: float,
    maximum: float,
    *,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigurationError(f"{name} must be numeric.")

    if minimum_inclusive:
        invalid_minimum = value < minimum
        minimum_message = f"{name} must be >= {minimum}."
    else:
        invalid_minimum = value <= minimum
        minimum_message = f"{name} must be > {minimum}."

    if maximum_inclusive:
        invalid_maximum = value > maximum
        maximum_message = f"{name} must be <= {maximum}."
    else:
        invalid_maximum = value >= maximum
        maximum_message = f"{name} must be < {maximum}."

    if invalid_minimum:
        raise ConfigurationError(minimum_message)
    if invalid_maximum:
        raise ConfigurationError(maximum_message)


def _require_logging_level(level: object) -> None:
    if not isinstance(level, str):
        raise ConfigurationError("logging.level must be a string.")
    if logging.getLevelName(level.upper()) == f"Level {level.upper()}":
        raise ConfigurationError(f"logging.level is invalid: {level}")


def _require_file(name: str, path: Path) -> None:
    if not path.exists():
        raise ConfigurationError(f"{name} does not exist: {path}")
    if not path.is_file():
        raise ConfigurationError(f"{name} must be a file: {path}")
