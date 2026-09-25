# src/visualization.py
"""
High-aesthetic PIL + OpenCV drawing helpers for Fall Detector.
Provides crisp anti-aliased typography, glassmorphism cards, and smooth human box tracking.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.fall_detector import (
    STATUS_NORMAL, STATUS_NO_POSE, STATUS_FALL_WARN, STATUS_NO_RESPONSE
)

# ── Font Loader Helper ───────────────────────────────────────────────
_font_cache = {}

def get_font(size: int, bold: bool = False):
    key = (size, bold)
    if key in _font_cache:
        return _font_cache[key]

    font_candidates = [
        "C:\\Windows\\Fonts\\segoeui.ttf" if not bold else "C:\\Windows\\Fonts\\segoeuib.ttf",
        "C:\\Windows\\Fonts\\arial.ttf" if not bold else "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\tahoma.ttf" if not bold else "C:\\Windows\\Fonts\\tahomabd.ttf",
    ]

    font = None
    for path in font_candidates:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:
                pass

    if font is None:
        font = ImageFont.load_default()

    _font_cache[key] = font
    return font


# ── Bounding Box Smoother ─────────────────────────────────────────────
class BBoxSmoother:
    """Exponential Moving Average (EMA) smoother for human bounding box."""
    def __init__(self, alpha: float = config.BBOX_SMOOTH_ALPHA):
        self.alpha = alpha
        self._curr = None

    def update(self, rect):
        if rect is None:
            return self._curr
        x, y, w, h = rect
        if self._curr is None:
            self._curr = (float(x), float(y), float(w), float(h))
        else:
            cx, cy, cw, ch = self._curr
            nx = cx * (1.0 - self.alpha) + x * self.alpha
            ny = cy * (1.0 - self.alpha) + y * self.alpha
            nw = cw * (1.0 - self.alpha) + w * self.alpha
            nh = ch * (1.0 - self.alpha) + h * self.alpha
            self._curr = (nx, ny, nw, nh)

        return (int(self._curr[0]), int(self._curr[1]),
                int(self._curr[2]), int(self._curr[3]))

    def reset(self):
        self._curr = None


_global_bbox_smoother = BBoxSmoother()


# ── Human Bounding Box Drawer ─────────────────────────────────────────
def draw_human_bounding_box(
    frame: np.ndarray,
    rect,
    status: str,
    source: str = "MP"
) -> np.ndarray:
    """Draw a sleek, smooth human tracking bounding box with corner brackets and badge."""
    if not rect:
        return frame

    smoothed = _global_bbox_smoother.update(rect)
    if not smoothed:
        return frame

    bx, by, bw, bh = smoothed
    h_f, w_f = frame.shape[:2]

    # Clip coordinates to frame boundaries
    bx = max(0, min(bx, w_f - 10))
    by = max(0, min(by, h_f - 10))
    bw = max(20, min(bw, w_f - bx))
    bh = max(20, min(bh, h_f - by))

    if status == STATUS_FALL_WARN:
        color_bgr = config.COLOR_FALL_WARN
        label_text = f"PERSON - FALL DETECTED [{source}]"
    elif status == STATUS_NO_RESPONSE:
        color_bgr = config.COLOR_NO_RESPONSE
        label_text = f"PERSON - NO RESPONSE [{source}]"
    elif status == STATUS_NO_POSE:
        color_bgr = config.COLOR_NO_POSE
        label_text = f"PERSON - UNCERTAIN [{source}]"
    else:
        color_bgr = config.COLOR_NORMAL
        label_text = f"PERSON - TRACKED [{source}]"

    # Draw semi-transparent inner glow overlay inside bounding box
    sub_box = frame[by:by+bh, bx:bx+bw]
    if sub_box.size > 0:
        overlay = sub_box.copy()
        cv2.rectangle(overlay, (0, 0), (bw, bh), color_bgr, -1)
        cv2.addWeighted(overlay, 0.08, sub_box, 0.92, 0, sub_box)

    # Draw corner brackets for sleek modern look
    corner_len = max(12, min(bw // 4, bh // 4, 28))
    thick = 3

    # Top-Left
    cv2.line(frame, (bx, by), (bx + corner_len, by), color_bgr, thick, cv2.LINE_AA)
    cv2.line(frame, (bx, by), (bx, by + corner_len), color_bgr, thick, cv2.LINE_AA)
    # Top-Right
    cv2.line(frame, (bx + bw, by), (bx + bw - corner_len, by), color_bgr, thick, cv2.LINE_AA)
    cv2.line(frame, (bx + bw, by), (bx + bw, by + corner_len), color_bgr, thick, cv2.LINE_AA)
    # Bottom-Left
    cv2.line(frame, (bx, by + bh), (bx + corner_len, by + bh), color_bgr, thick, cv2.LINE_AA)
    cv2.line(frame, (bx, by + bh), (bx, by + bh - corner_len), color_bgr, thick, cv2.LINE_AA)
    # Bottom-Right
    cv2.line(frame, (bx + bw, by + bh), (bx + bw - corner_len, by + bh), color_bgr, thick, cv2.LINE_AA)
    cv2.line(frame, (bx + bw, by + bh), (bx + bw, by + bh - corner_len), color_bgr, thick, cv2.LINE_AA)

    # Thin full outline
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color_bgr, 1, cv2.LINE_AA)

    # Badge overlay above box using PIL for crisp text
    badge_y = max(10, by - 24)
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_frame, "RGBA")
    font_badge = get_font(12, bold=True)

    bbox = font_badge.getbbox(label_text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    r, g, b = color_bgr[2], color_bgr[1], color_bgr[0]
    badge_rect = [bx, badge_y, bx + tw + 14, badge_y + th + 8]
    draw.rounded_rectangle(badge_rect, radius=4, fill=(r, g, b, 230))
    draw.text((bx + 7, badge_y + 3), label_text, font=font_badge, fill=(255, 255, 255, 255))

    return cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)


# ── Status Overlay & UI Cards ─────────────────────────────────────────
def draw_status_overlay(
    frame: np.ndarray,
    status: str,
    fps: float,
    features: dict,
    persist: int,
    countdown: float,
) -> np.ndarray:
    """Draw high-aesthetic glassmorphism status overlay, telemetry HUD, and crisp alert cards."""
    h, w = frame.shape[:2]

    # Draw full-frame border alert during warning / emergency states
    if status == STATUS_FALL_WARN:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), config.COLOR_FALL_WARN, config.BORDER_THICKNESS)
    elif status == STATUS_NO_RESPONSE:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), config.COLOR_NO_RESPONSE, config.BORDER_THICKNESS)

    # Convert BGR frame to PIL Image (RGBA)
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(pil_img, "RGBA")

    # ── Top-Left HUD Card ─────────────────────────────────────────────
    hud_x, hud_y = 12, 12
    hud_w, hud_h = 175, 145

    # Glass background
    draw.rounded_rectangle(
        [hud_x, hud_y, hud_x + hud_w, hud_y + hud_h],
        radius=10,
        fill=(15, 20, 30, 205),
        outline=(255, 255, 255, 45),
        width=1
    )

    font_title = get_font(13, bold=True)
    font_body  = get_font(11, bold=False)

    # Status pill color mapping
    if status == STATUS_FALL_WARN:
        st_color = (40, 220, 110)
    elif status == STATUS_NO_RESPONSE:
        st_color = (240, 60, 60)
    elif status == STATUS_NO_POSE:
        st_color = (240, 200, 50)
    else:
        st_color = (40, 220, 110)

    # Header indicator pill
    draw.rounded_rectangle(
        [hud_x + 8, hud_y + 8, hud_x + 16, hud_y + 24],
        radius=4,
        fill=st_color
    )
    draw.text((hud_x + 22, hud_y + 8), status, font=font_title, fill=(255, 255, 255, 255))

    # Metrics
    metrics = [
        f"FPS       : {fps:.1f}",
        f"Torso Ang : {features.get('torso_angle_deg', 0):.1f}°",
        f"Shld Ang  : {features.get('shoulder_angle_deg', 0):.1f}°",
        f"Ratio (W/H): {features.get('aspect_ratio', 0):.2f}",
        f"Hip Height: {features.get('hip_y_norm', 0):.2f}",
    ]

    for i, text in enumerate(metrics):
        draw.text(
            (hud_x + 10, hud_y + 32 + i * 20),
            text,
            font=font_body,
            fill=(210, 220, 235, 255)
        )

    # ── Centered Glass Alert Cards ────────────────────────────────────
    if status == STATUS_FALL_WARN:
        _draw_glass_alert_card(
            draw, w, h,
            title="FALL DETECTED",
            subtitle=f"Monitoring posture... {countdown:.1f}s",
            bg_color=(20, 90, 45, 225),
            border_color=(40, 220, 110, 255),
            text_color=(255, 255, 255, 255),
            countdown=countdown,
            max_countdown=config.NO_RESPONSE_SECONDS
        )

    elif status == STATUS_NO_RESPONSE:
        _draw_glass_alert_card(
            draw, w, h,
            title="NO RESPONSE - EMERGENCY",
            subtitle="POSSIBLE FALL DETECTED! SEEK HELP IMMEDIATELY",
            bg_color=(120, 15, 20, 235),
            border_color=(240, 60, 60, 255),
            text_color=(255, 255, 255, 255),
            countdown=0.0,
            max_countdown=1.0
        )

    out_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    return out_bgr


def _draw_glass_alert_card(
    draw: ImageDraw.ImageDraw,
    frame_w: int,
    frame_h: int,
    title: str,
    subtitle: str,
    bg_color: tuple,
    border_color: tuple,
    text_color: tuple,
    countdown: float = 0.0,
    max_countdown: float = 5.0
):
    """Draw a centered glassmorphism alert banner card with crisp typography."""
    card_w = min(frame_w - 30, 320)
    card_h = 100
    cx = (frame_w - card_w) // 2
    cy = int(frame_h * 0.44) - card_h // 2

    # Drop shadow
    draw.rounded_rectangle(
        [cx + 4, cy + 4, cx + card_w + 4, cy + card_h + 4],
        radius=14,
        fill=(0, 0, 0, 120)
    )

    # Main Card
    draw.rounded_rectangle(
        [cx, cy, cx + card_w, cy + card_h],
        radius=14,
        fill=bg_color,
        outline=border_color,
        width=2
    )

    font_head = get_font(18, bold=True)
    font_sub  = get_font(12, bold=False)

    # Title
    t_box = font_head.getbbox(title)
    tw = t_box[2] - t_box[0]
    draw.text((cx + (card_w - tw) // 2, cy + 18), title, font=font_head, fill=text_color)

    # Subtitle
    s_box = font_sub.getbbox(subtitle)
    sw = s_box[2] - s_box[0]
    draw.text((cx + (card_w - sw) // 2, cy + 48), subtitle, font=font_sub, fill=(230, 240, 255, 255))

    # Countdown Progress Bar
    if countdown > 0 and max_countdown > 0:
        bar_x1 = cx + 24
        bar_y1 = cy + 76
        bar_w  = card_w - 48
        bar_h  = 6

        pct = max(0.0, min(1.0, countdown / max_countdown))
        fill_w = int(bar_w * pct)

        # Bar track
        draw.rounded_rectangle(
            [bar_x1, bar_y1, bar_x1 + bar_w, bar_y1 + bar_h],
            radius=3,
            fill=(0, 0, 0, 100)
        )
        # Bar fill
        if fill_w > 0:
            draw.rounded_rectangle(
                [bar_x1, bar_y1, bar_x1 + fill_w, bar_y1 + bar_h],
                radius=3,
                fill=border_color
            )

