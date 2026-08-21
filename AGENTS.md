# AGENTS.md

## Project Purpose

Driver Drowsiness Detection System using OpenCV, dlib facial landmarks, and Eye Aspect Ratio based temporal drowsiness detection.

## Current Architecture

- Python application.
- OpenCV webcam processing.
- dlib frontal face detector.
- dlib 68-point facial landmark predictor.
- Eye Aspect Ratio (EAR) calculation.
- Consecutive-frame drowsiness detection.
- Local audio alarm playback.

## Important Assets

- `shape_predictor_68_face_landmarks.dat`: required dlib landmark model.
- `loud_alarm.wav`: required local alert sound.

Do not delete, replace, or move these assets without explicit approval.

## Rules For Future Agents

- Do not invent ML models or claim SVM, KNN, XGBoost, LSTM, or similar model usage unless it is actually implemented in the codebase.
- Preserve working drowsiness-detection behavior unless a change is explicitly justified.
- Do not hard-code machine-specific paths.
- Do not commit secrets, credentials, tokens, private keys, or personal data.
- Do not add unnecessary dependencies.
- Do not modify unrelated files.
- Prefer small, testable modules over large rewrites.
- Separate hardware-dependent functionality, such as webcam, GUI, and audio playback, from testable core logic.

## Testing Expectations

- Run tests after changes when tests exist or are added.
- Run static/lint checks if configured.
- Validate the application as far as possible, especially importability and core EAR/detection logic.
- Report test, lint, hardware, dependency, camera, display, or audio failures clearly instead of hiding them.

## Git Expectations

- Do not push to GitHub without explicit approval.
- Review `git diff` before considering work complete.
- Do not commit generated files, virtual environments, caches, build artifacts, logs, or secrets.

## Production Principles

Prioritize correctness, reliability, maintainability, testability, reproducibility, security, and observability.
