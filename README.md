# Driver Drowsiness Detection System

## Overview

Driver Drowsiness Detection System is a local Python application for real-time webcam-based drowsiness detection. It uses OpenCV video capture, dlib 68-point facial landmarks, Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), a simple landmark-based head-tilt heuristic, and temporal persistence to decide when to trigger a local WAV alarm.

This project is a computer-vision prototype/research-oriented application. It does not claim medical accuracy, guaranteed accident prevention, driver identification, or automotive safety certification.

## Features

- Real-time webcam frame processing.
- dlib 68-point facial landmark detection.
- Eye Aspect Ratio calculation for both eyes.
- Mouth Aspect Ratio calculation for yawning.
- Basic 2D head-tilt detection from facial landmarks.
- Configurable EAR threshold.
- Configurable consecutive closed-frame threshold.
- Configurable yawning and head-tilt persistence thresholds.
- Multi-signal heuristic: sustained eye closure is primary, while sustained yawning and head tilt are supporting signals.
- Largest-face primary-face heuristic when multiple faces are detected.
- Local WAV alarm with graceful audio failure handling.
- Controlled camera, GUI, and cleanup behavior.
- Structured detection pipeline with automated tests.

## Detection Approach

The detector uses one dlib face detection pass and one 68-point landmark extraction for the selected primary face. The largest valid detected face is used as the primary face.

- EAR is calculated from eye landmarks and remains the primary drowsiness signal.
- MAR is calculated from mouth landmarks and becomes a yawning signal only after sustained high-MAR frames.
- Head tilt uses a simple 2D landmark heuristic based on eye-line roll and nose/chin geometry.
- The final decision is heuristic: sustained eye closure can trigger drowsiness on its own, while yawning and head tilt are supporting signals that must persist before contributing.

## Architecture

```text
Webcam
  |
  v
Frame preprocessing
  |
  v
Face detection
  |
  v
Primary face selection
  |
  v
68-point landmarks
  |
  v
Eye landmarks
  |
  v
EAR
  |
  v
Mouth landmarks
  |
  v
MAR/yawning
  |
  v
Head landmarks
  |
  v
Head tilt heuristic
  |
  v
Temporal state
  |
  v
Multi-signal drowsiness decision
  |
  v
Alert
```

Important modules:

- `driver_drowsiness.app`: CLI and application orchestration.
- `driver_drowsiness.camera`: webcam ownership and frame reads.
- `driver_drowsiness.landmarks`: dlib face detector and landmark predictor.
- `driver_drowsiness.ear`: pure EAR calculations.
- `driver_drowsiness.mar`: pure MAR calculations.
- `driver_drowsiness.head_pose`: simple 2D head-tilt metrics.
- `driver_drowsiness.detector`: frame-to-detection-result pipeline.
- `driver_drowsiness.state`: consecutive-frame drowsiness state.
- `driver_drowsiness.alerts`: local WAV alarm lifecycle.
- `driver_drowsiness.config`: typed configuration loading and validation.
- `driver_drowsiness.diagnostics`: non-invasive environment, asset, and hardware checks.

## Project Structure

```text
.
|-- AGENTS.md
|-- Drowsiness_Detection.py
|-- README.md
|-- config.yml
|-- loud_alarm.wav
|-- pyproject.toml
|-- requirements-dev.txt
|-- requirements.txt
|-- shape_predictor_68_face_landmarks.dat
|-- src/
|   `-- driver_drowsiness/
|       |-- alerts.py
|       |-- app.py
|       |-- camera.py
|       |-- config.py
|       |-- detector.py
|       |-- diagnostics.py
|       |-- ear.py
|       |-- exceptions.py
|       |-- head_pose.py
|       |-- landmarks.py
|       |-- logging_config.py
|       |-- mar.py
|       |-- preprocessing.py
|       `-- state.py
`-- tests/
```

## Requirements

- Python 3.12.x.
- Local webcam.
- Desktop environment with OpenCV GUI support.
- Audio output device for audible alerts.
- `shape_predictor_68_face_landmarks.dat`.
- `loud_alarm.wav`.

## Installation

For development:

```bash
python -m pip install -e ".[dev]"
```

For normal runtime installation:

```bash
python -m pip install .
```

## Usage

Run the application:

```bash
python -m driver_drowsiness
```

or:

```bash
driver-drowsiness
```

Show CLI help:

```bash
driver-drowsiness --help
```

Run diagnostics without starting the full detection loop or playing the alarm:

```bash
driver-drowsiness diagnose
```

Press `ESC` in the OpenCV window to shut down cleanly.

## Configuration

Runtime settings live in `config.yml`. CLI overrides take precedence over `config.yml`, and `config.yml` takes precedence over built-in defaults.

Current configuration fields:

- `camera.index`: webcam index, default `0`.
- `camera.width`: maximum frame width before detection.
- `camera.height`: maximum frame height before detection.
- `camera.processing_scale`: processing downscale factor.
- `detection.ear_threshold`: EAR threshold, default `0.3`.
- `detection.closed_frames_threshold`: consecutive closed-eye frames before alerting, default `30`.
- `yawning.enabled`: enable or disable MAR/yawning as a supporting signal.
- `yawning.mar_threshold`: MAR threshold for open-mouth frames.
- `yawning.consecutive_frames`: high-MAR frames required before yawning is detected.
- `head_pose.enabled`: enable or disable head tilt as a supporting signal.
- `head_pose.roll_threshold_degrees`: 2D eye-line roll threshold.
- `head_pose.downward_ratio_threshold`: simple downward-orientation threshold.
- `head_pose.consecutive_frames`: head-tilt frames required before detection.
- `assets.model_path`: path to the dlib landmark predictor.
- `assets.alarm_path`: path to the WAV alarm.
- `alert.enabled`: enable or disable audio playback.
- `logging.level`: logging level such as `INFO` or `DEBUG`.

Useful overrides:

```bash
driver-drowsiness --camera 1
driver-drowsiness --ear-threshold 0.25
driver-drowsiness --closed-frames 20
driver-drowsiness --disable-alarm
driver-drowsiness --config path/to/config.yml
```

## Diagnostics

Diagnostics check Python/imports, configuration loading, asset availability, dlib model loading, WAV readability, camera open/read/release, and audio device discovery.

```bash
python -m driver_drowsiness diagnose
driver-drowsiness diagnose
```

The diagnostic command does not play the alarm, show a continuous GUI, save frames, upload data, or modify configuration.

## Troubleshooting

Camera unavailable:

- Verify the webcam is connected.
- Verify OS camera permissions.
- Verify `camera.index` in `config.yml`.
- Run `driver-drowsiness diagnose`.

Alarm unavailable:

- Verify an audio output device is available.
- Verify `loud_alarm.wav` exists and is readable.
- Run `driver-drowsiness diagnose`.

Model unavailable:

- Verify `shape_predictor_68_face_landmarks.dat` exists.
- Verify `assets.model_path` in `config.yml`.
- Run `driver-drowsiness diagnose`.

GUI unavailable:

- Run the application in a desktop environment with OpenCV GUI support.

## Limitations

- The largest detected face is treated as the primary face.
- Primary-face selection does not guarantee driver identification.
- EAR, MAR, and head-tilt heuristics can be affected by lighting, camera angle, glasses, occlusion, facial differences, landmark quality, and frame rate.
- Head tilt is a simple 2D landmark heuristic, not full 3D head pose estimation.
- A local webcam and local audio output are required for the full experience.
- The system is not a certified automotive safety product.

## Privacy

- Webcam frames are processed locally.
- The current application does not intentionally store video.
- The current application does not upload webcam frames.
- No cloud service is required.

## Testing

```bash
python -m pip check
python -m ruff check .
python -m pytest
```

## License

No license file is currently included. Add an explicit license before publishing or accepting external contributions.
