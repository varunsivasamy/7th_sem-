# src/visualization.py
"""All OpenCV drawing helpers."""

import cv2
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.fall_detector import (
    STATUS_NORMAL, STATUS_NO_POSE, STATUS_FALL_WARN, STATUS_NO_RESPONSE
)


def draw_status_overlay(
    frame: np.ndarray,
    status: str,
    fps: float,
    features: dict,
    persist: int,
    countdown: float,       # seconds remaining before red alert (0 if N/A)
) -> np.ndarray:
    """
    Draw the appropriate alert box, status text, FPS, and detection values.
    All drawing is in-place; the frame is also returned.

    Visual rules
    ────────────
    NORMAL              – no box, green status text
    POSE NOT DETECTED   – no box, yellow status text
    FALL DETECTED       – GREEN border + green box with message + countdown
    NO RESPONSE         – RED border + red box with large alert message
    """
    h, w = frame.shape[:2]
    font_main   = cv2.FONT_HERSHEY_DUPLEX
    font_small  = cv2.FONT_HERSHEY_SIMPLEX

    # ── Phase 1: FALL DETECTED – green border + green info box ───────
    if status == STATUS_FALL_WARN:
        # Green border
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1),
                      config.COLOR_FALL_WARN, config.BORDER_THICKNESS)

        # Centred message box
        line1 = "FALL DETECTED"
        line2 = f"Monitoring... {countdown:.1f}s"

        _draw_centred_box(frame, [line1, line2],
                          box_color=config.COLOR_FALL_WARN,
                          text_color=(0, 0, 0),          # black text on green
                          y_frac=0.45)
        status_color = config.COLOR_FALL_WARN

    # ── Phase 2: NO RESPONSE – red border + red alert box ────────────
    elif status == STATUS_NO_RESPONSE:
        # Red border
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1),
                      config.COLOR_NO_RESPONSE, config.BORDER_THICKNESS)

        line1 = "NO RESPONSE"
        line2 = "POSSIBLE FALL – SEEK HELP!"

        _draw_centred_box(frame, [line1, line2],
                          box_color=config.COLOR_NO_RESPONSE,
                          text_color=(255, 255, 255),     # white text on red
                          y_frac=0.45)
        status_color = config.COLOR_NO_RESPONSE

    elif status == STATUS_NO_POSE:
        status_color = config.COLOR_NO_POSE
    else:
        status_color = config.COLOR_NORMAL

    # ── Top-left info panel ───────────────────────────────────────────
    lh = 24
    lines = [
        (f"Status : {status}", status_color),
        (f"FPS    : {fps:.1f}", (220, 220, 220)),
    ]
    if features:
        lines += [
            (f"Angle  : {features.get('torso_angle_deg',    0):.1f} deg", (200, 200, 200)),
            (f"ShldAng: {features.get('shoulder_angle_deg', 0):.1f} deg", (200, 200, 200)),
            (f"Ratio  : {features.get('aspect_ratio',       0):.2f}",     (200, 200, 200)),
            (f"Hip-Y  : {features.get('hip_y_norm',         0):.2f}",     (200, 200, 200)),
            (f"Persist: {persist}",                                         (170, 170, 170)),
        ]
        if countdown > 0:
            lines.append((f"Countdown: {countdown:.1f}s", config.COLOR_FALL_WARN))

    for i, (txt, color) in enumerate(lines):
        y = 20 + i * lh
        cv2.putText(frame, txt, (11, y + 1),
                    font_small, config.FONT_SCALE_SMALL,
                    (0, 0, 0), config.FONT_THICKNESS + 1, cv2.LINE_AA)
        cv2.putText(frame, txt, (10, y),
                    font_small, config.FONT_SCALE_SMALL,
                    color, config.FONT_THICKNESS, cv2.LINE_AA)

    return frame


# ── Helper ────────────────────────────────────────────────────────
def _draw_centred_box(frame, lines: list, box_color, text_color, y_frac=0.45):
    """Draw a filled rounded rectangle centred horizontally at y_frac * height."""
    h, w = frame.shape[:2]
    font  = cv2.FONT_HERSHEY_DUPLEX
    scale = config.FONT_SCALE_LARGE
    thick = config.FONT_THICKNESS + 1
    pad   = 18
    lh    = 44   # line height inside box

    # Measure widest line
    max_tw = 0
    sizes  = []
    for txt in lines:
        (tw, th), _ = cv2.getTextSize(txt, font, scale, thick)
        sizes.append((tw, th))
        max_tw = max(max_tw, tw)

    box_w = max_tw + pad * 2
    box_h = lh * len(lines) + pad * 2
    bx    = (w - box_w) // 2
    by    = int(h * y_frac) - box_h // 2

    # Filled rectangle (no rounded-rect in older OpenCV → plain rect)
    cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), box_color, -1)
    # Thin white/dark outline
    cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), (0, 0, 0), 2)

    for i, (txt, (tw, th)) in enumerate(zip(lines, sizes)):
        tx = (w - tw) // 2
        ty = by + pad + th + i * lh
        cv2.putText(frame, txt, (tx, ty),
                    font, scale, text_color, thick, cv2.LINE_AA)
