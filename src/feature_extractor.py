# src/feature_extractor.py
"""Computes geometric features from landmark dictionaries."""

import math
import numpy as np


def _midpoint(a, b):
    """Return pixel midpoint of two (x, y, vis) tuples."""
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _angle_from_vertical(p1, p2):
    """
    Angle (degrees) between the vector p1→p2 and the vertical axis.
    0° = perfectly upright, 90° = fully horizontal.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    # Angle from vertical: atan2(|dx|, |dy|)
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))


def compute_features(landmarks: dict, frame_w: int, frame_h: int) -> dict:
    """
    Parameters
    ----------
    landmarks : dict returned by PoseEstimator.extract_landmarks()
    frame_w, frame_h : frame dimensions for normalisation

    Returns
    -------
    dict with keys:
        shoulder_center     (x, y) pixels
        hip_center          (x, y) pixels
        torso_angle_deg     float  – angle of torso from vertical (0=upright, 90=horizontal)
        shoulder_angle_deg  float  – angle of shoulder line from horizontal (0=level)
        aspect_ratio        float  – bounding-box width / height
        hip_y_norm          float  – hip_center.y / frame_h  (0=top, 1=bottom)
        torso_len_norm      float  – torso length / frame_h
        visibility_ok       bool   – key landmarks visible enough
    """
    required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    if not all(k in landmarks for k in required):
        return {}

    ls = landmarks["left_shoulder"]
    rs = landmarks["right_shoulder"]
    lh = landmarks["left_hip"]
    rh = landmarks["right_hip"]

    # Visibility gate – skip if major landmarks are unreliable
    vis_ok = all(landmarks[k][2] > 0.4 for k in required)

    shoulder_center = _midpoint(ls, rs)
    hip_center      = _midpoint(lh, rh)

    # ── Torso angle from vertical ─────────────────────────────────
    # Measures how much the torso vector (shoulder_center → hip_center)
    # deviates from vertical.
    # Upright: shoulders directly above hips → angle ≈ 0°
    # Lying flat: shoulders beside hips   → angle ≈ 90°
    torso_angle_deg = _angle_from_vertical(shoulder_center, hip_center)

    # ── Shoulder line angle from horizontal ───────────────────────
    # When a person falls sideways, one shoulder drops sharply.
    # Measures tilt of the shoulder line.
    shoulder_dx = rs[0] - ls[0]
    shoulder_dy = rs[1] - ls[1]
    shoulder_angle_deg = abs(math.degrees(math.atan2(shoulder_dy,
                                                     abs(shoulder_dx) + 1e-6)))

    # ── Torso length (normalised) ─────────────────────────────────
    torso_len = math.hypot(
        shoulder_center[0] - hip_center[0],
        shoulder_center[1] - hip_center[1],
    )
    torso_len_norm = torso_len / (frame_h + 1e-6)

    # ── Bounding box of all 8 key points ─────────────────────────
    xs, ys = [], []
    for name in ["left_shoulder", "right_shoulder",
                 "left_hip",      "right_hip",
                 "left_knee",     "right_knee",
                 "left_ankle",    "right_ankle"]:
        if name in landmarks:
            xs.append(landmarks[name][0])
            ys.append(landmarks[name][1])

    if len(xs) >= 4:
        bbox_w = max(xs) - min(xs)
        bbox_h = max(ys) - min(ys)
        aspect_ratio = bbox_w / (bbox_h + 1e-6)
    else:
        aspect_ratio = 0.0

    hip_y_norm = hip_center[1] / (frame_h + 1e-6)

    return {
        "shoulder_center":    shoulder_center,
        "hip_center":         hip_center,
        "torso_angle_deg":    torso_angle_deg,
        "shoulder_angle_deg": shoulder_angle_deg,
        "aspect_ratio":       aspect_ratio,
        "hip_y_norm":         hip_y_norm,
        "torso_len_norm":     torso_len_norm,
        "visibility_ok":      vis_ok,
    }
