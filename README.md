# Possible-Fall Detector – Phase 1

> **Disclaimer:** This is an academic prototype only. It is not a medical device
> and must not be used as the only patient safety system.

---

## Overview

A local, CPU-only video-based possible-fall detector built with Python,
OpenCV, and MediaPipe Pose. No model training, no cloud upload, no GPU required.

---

## Requirements

| Item | Version |
|------|---------|
| Python | 3.10 + |
| opencv-python | ≥ 4.8 |
| mediapipe | ≥ 0.10 |
| numpy | ≥ 1.24 |

Target hardware: Intel i5-1135G7, 8 GB RAM, no discrete GPU (Windows).

---

## Installation

```powershell
# Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Run

```powershell
python main.py --source "path\to\video.mp4"
```

### Keyboard controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `p` | Pause / Resume |
| `r` | Reset detector |

---

## Display

| Colour / style | Meaning |
|----------------|---------|
| GREEN text `NORMAL` | No fall concern |
| YELLOW text `POSE NOT DETECTED` | Landmarks not found |
| RED border + large text `POSSIBLE FALL DETECTED` | Alert condition met |

The overlay also shows live FPS, torso angle, aspect ratio, hip-Y position,
persistence counter, and cooldown counter.

---

## Project structure

```
main.py              – entry point
config.py            – all thresholds and settings
requirements.txt
README.md
.gitignore
src/
  __init__.py
  pose_estimator.py  – MediaPipe Pose wrapper
  feature_extractor.py – geometric feature calculation
  fall_detector.py   – stateful alert logic
  visualization.py   – OpenCV drawing helpers
```

---

## Detection logic (summary)

1. MediaPipe Pose extracts 8 key landmarks (shoulders, hips, knees, ankles).
2. Per frame: torso angle, bounding-box aspect ratio, and normalised hip-Y
   are computed.
3. A `FallDetector` maintains a rolling history and fires
   `POSSIBLE FALL DETECTED` only when **all three** conditions are true
   for `FALL_PERSISTENCE_FRAMES` consecutive frames.
4. A cooldown suppresses repeated alerts.

All thresholds are in `config.py`.

---

## What this is NOT

- Not a confirmed medical fall detector.
- Not a real-time camera feed (Phase 1 is file-based only).
- Not connected to any external service, database, or cloud.
