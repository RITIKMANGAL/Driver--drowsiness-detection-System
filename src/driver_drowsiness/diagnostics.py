"""Non-invasive runtime diagnostics."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO

import sounddevice as sd
import soundfile as sf

from driver_drowsiness.camera import Camera
from driver_drowsiness.config import ApplicationConfig, ConfigOverrides, load_config
from driver_drowsiness.exceptions import DriverDrowsinessError
from driver_drowsiness.landmarks import FacialLandmarkDetector

REQUIRED_IMPORTS = ("cv2", "dlib", "numpy", "sounddevice", "soundfile", "yaml")


@dataclass(frozen=True)
class DiagnosticCheck:
    section: str
    name: str
    passed: bool
    message: str = ""


@dataclass(frozen=True)
class DiagnosticReport:
    checks: list[DiagnosticCheck]

    @property
    def ready(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def reasons(self) -> list[str]:
        return [check.message for check in self.checks if not check.passed and check.message]


def run_diagnostics(
    *,
    config_path: str | Path | None = None,
    overrides: ConfigOverrides | None = None,
    camera_factory: Callable[[int], Any] = Camera,
    landmark_detector_factory: Callable[[Path], Any] = FacialLandmarkDetector,
    soundfile_module: Any = sf,
    sounddevice_module: Any = sd,
    required_imports: tuple[str, ...] = REQUIRED_IMPORTS,
) -> DiagnosticReport:
    checks: list[DiagnosticCheck] = []
    checks.extend(_environment_checks(required_imports))

    config = _load_config_check(checks, config_path=config_path, overrides=overrides)
    if config is None:
        return DiagnosticReport(checks)

    checks.extend(
        _asset_checks(
            config,
            landmark_detector_factory=landmark_detector_factory,
            soundfile_module=soundfile_module,
        )
    )
    checks.extend(
        _hardware_checks(
            config,
            camera_factory=camera_factory,
            sounddevice_module=sounddevice_module,
        )
    )
    return DiagnosticReport(checks)


def run_diagnostics_command(
    *,
    config_path: str | Path | None = None,
    overrides: ConfigOverrides | None = None,
    output: TextIO = sys.stdout,
    camera_factory: Callable[[int], Any] = Camera,
    landmark_detector_factory: Callable[[Path], Any] = FacialLandmarkDetector,
    soundfile_module: Any = sf,
    sounddevice_module: Any = sd,
) -> int:
    report = run_diagnostics(
        config_path=config_path,
        overrides=overrides,
        camera_factory=camera_factory,
        landmark_detector_factory=landmark_detector_factory,
        soundfile_module=soundfile_module,
        sounddevice_module=sounddevice_module,
    )
    output.write(format_report(report))
    return 0 if report.ready else 1


def format_report(report: DiagnosticReport) -> str:
    lines = ["Driver Drowsiness Detection - Diagnostics", ""]
    sections = ["Environment", "Configuration", "Assets", "Hardware"]
    for section in sections:
        section_checks = [check for check in report.checks if check.section == section]
        if not section_checks:
            continue
        lines.append(section)
        for check in section_checks:
            status = "PASS" if check.passed else "FAIL"
            detail = f" - {check.message}" if check.message else ""
            lines.append(f"  {check.name:<15} {status}{detail}")
        lines.append("")

    lines.append("Overall")
    lines.append(f"  {'READY' if report.ready else 'NOT READY'}")
    if report.reasons:
        lines.append("")
        lines.append("Reason:")
        for reason in report.reasons:
            lines.append(f"  - {reason}")
    lines.append("")
    return "\n".join(lines)


def _environment_checks(required_imports: tuple[str, ...]) -> list[DiagnosticCheck]:
    checks = [
        DiagnosticCheck(
            "Environment",
            "Python",
            sys.version_info >= (3, 12) and sys.version_info < (3, 13),
            sys.version.split()[0],
        )
    ]
    missing: list[str] = []
    for module_name in required_imports:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(module_name)

    checks.append(
        DiagnosticCheck(
            "Environment",
            "Dependencies",
            not missing,
            "" if not missing else "missing: " + ", ".join(missing),
        )
    )
    return checks


def _load_config_check(
    checks: list[DiagnosticCheck],
    *,
    config_path: str | Path | None,
    overrides: ConfigOverrides | None,
) -> ApplicationConfig | None:
    try:
        config = load_config(config_path=config_path, overrides=overrides)
    except DriverDrowsinessError as exc:
        checks.append(DiagnosticCheck("Configuration", "Config", False, str(exc)))
        return None

    message = str(config.config_path) if config.config_path else "built-in defaults"
    checks.append(DiagnosticCheck("Configuration", "Config", True, message))
    return config


def _asset_checks(
    config: ApplicationConfig,
    *,
    landmark_detector_factory: Callable[[Path], Any],
    soundfile_module: Any,
) -> list[DiagnosticCheck]:
    checks: list[DiagnosticCheck] = []
    model_path = config.assets.model_path
    alarm_path = config.assets.alarm_path

    checks.append(
        DiagnosticCheck(
            "Assets",
            "Model file",
            model_path.is_file(),
            str(model_path) if model_path.is_file() else f"missing: {model_path}",
        )
    )
    try:
        landmark_detector_factory(model_path)
    except Exception as exc:
        checks.append(DiagnosticCheck("Assets", "Model load", False, str(exc)))
    else:
        checks.append(DiagnosticCheck("Assets", "Model load", True))

    checks.append(
        DiagnosticCheck(
            "Assets",
            "Alarm WAV",
            alarm_path.is_file(),
            str(alarm_path) if alarm_path.is_file() else f"missing: {alarm_path}",
        )
    )
    try:
        soundfile_module.read(alarm_path, dtype="float32")
    except Exception as exc:
        checks.append(DiagnosticCheck("Assets", "WAV read", False, str(exc)))
    else:
        checks.append(DiagnosticCheck("Assets", "WAV read", True))

    return checks


def _hardware_checks(
    config: ApplicationConfig,
    *,
    camera_factory: Callable[[int], Any],
    sounddevice_module: Any,
) -> list[DiagnosticCheck]:
    checks: list[DiagnosticCheck] = []

    camera = None
    try:
        camera = camera_factory(config.camera.index)
        camera.open()
        frame = camera.read()
    except Exception as exc:
        checks.append(DiagnosticCheck("Hardware", "Camera", False, str(exc)))
    else:
        shape = getattr(frame, "shape", None)
        checks.append(DiagnosticCheck("Hardware", "Camera", True, f"frame={shape}"))
    finally:
        if camera is not None:
            try:
                camera.release()
            except Exception:
                pass

    try:
        devices = sounddevice_module.query_devices()
        output_devices = [
            device
            for device in devices
            if int(device.get("max_output_channels", 0)) > 0
        ]
    except Exception as exc:
        checks.append(DiagnosticCheck("Hardware", "Audio", False, str(exc)))
    else:
        checks.append(
            DiagnosticCheck(
                "Hardware",
                "Audio",
                bool(output_devices),
                f"output devices={len(output_devices)}",
            )
        )

    return checks
