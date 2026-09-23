# src/blob_detector.py
"""
OpenCV-based foreground blob detector with static-scene fallback.

Strategy
────────
1. MOG2 (motion-based)  – works well for moving person
2. Frame-diff from a "reference" frame taken just before a fall
   – works when the person is still on the floor (MOG2 absorbs them)
3. First-frame diff       – broad fallback using the very first frame

Returns per-frame features compatible with fall_detector.py:
  torso_angle_deg, aspect_ratio, hip_y_norm, visibility_ok, …
"""

import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


class BlobDetector:

    # Min blob area as fraction of frame area
    MIN_AREA_FRAC = 0.008

    def __init__(self):
        # MOG2 with long history so a still person stays as foreground
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=1000,           # long history
            varThreshold=25,        # more sensitive
            detectShadows=False,
        )
        self._kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))

        self._frame_count = 0
        self._first_frame = None        # very first frame (static background ref)
        self._ref_frame   = None        # rolling reference taken ~2s ago
        self._ref_buf     = []          # circular buffer to pick ref frame
        self._ref_buf_size = 60         # ~2s at 30fps

    # ------------------------------------------------------------------ #
    def update(self, frame_bgr: np.ndarray):
        self._frame_count += 1
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Store first frame
        if self._first_frame is None:
            self._first_frame = gray.copy()

        # Rolling reference buffer
        self._ref_buf.append(gray.copy())
        if len(self._ref_buf) > self._ref_buf_size:
            self._ref_buf.pop(0)
        # Use the oldest frame in the buffer as reference (2s ago)
        self._ref_frame = self._ref_buf[0]

        min_area = w * h * self.MIN_AREA_FRAC

        # ── Method 1: MOG2 ────────────────────────────────────────────
        fg_mog = self._bg.apply(frame_bgr)
        fg_mog = cv2.morphologyEx(fg_mog, cv2.MORPH_OPEN,  self._kernel_open)
        fg_mog = cv2.morphologyEx(fg_mog, cv2.MORPH_CLOSE, self._kernel_close)
        feat_mog, area_mog = self._contour_features(fg_mog, w, h, min_area)

        # ── Method 2: diff from rolling reference (2s ago) ────────────
        diff_ref = cv2.absdiff(self._ref_frame, gray)
        _, fg_ref = cv2.threshold(diff_ref, 20, 255, cv2.THRESH_BINARY)
        fg_ref = cv2.morphologyEx(fg_ref, cv2.MORPH_OPEN,  self._kernel_open)
        fg_ref = cv2.morphologyEx(fg_ref, cv2.MORPH_CLOSE, self._kernel_close)
        feat_ref, area_ref = self._contour_features(fg_ref, w, h, min_area)

        # ── Method 3: diff from first frame ───────────────────────────
        diff_first = cv2.absdiff(self._first_frame, gray)
        _, fg_first = cv2.threshold(diff_first, 25, 255, cv2.THRESH_BINARY)
        fg_first = cv2.morphologyEx(fg_first, cv2.MORPH_OPEN,  self._kernel_open)
        fg_first = cv2.morphologyEx(fg_first, cv2.MORPH_CLOSE, self._kernel_close)
        feat_first, area_first = self._contour_features(fg_first, w, h, min_area)

        # ── Pick best result (largest blob wins) ──────────────────────
        candidates = [
            (area_mog,   feat_mog,   fg_mog),
            (area_ref,   feat_ref,   fg_ref),
            (area_first, feat_first, fg_first),
        ]
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_area, best_feat, best_mask = candidates[0]

        if best_area < min_area or not best_feat:
            # Combine all masks and retry
            combined = cv2.bitwise_or(fg_mog, fg_ref)
            combined = cv2.bitwise_or(combined, fg_first)
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self._kernel_close)
            best_feat, best_area = self._contour_features(combined, w, h, min_area * 0.5)
            best_mask = combined

        return best_feat if best_feat else {}, best_mask

    # ------------------------------------------------------------------ #
    def _contour_features(self, mask, w, h, min_area):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return {}, 0

        largest = max(contours, key=cv2.contourArea)
        area    = cv2.contourArea(largest)
        if area < min_area:
            return {}, 0

        bx, by, bw, bh = cv2.boundingRect(largest)
        aspect_ratio = bw / (bh + 1e-6)
        hip_y_norm   = (by + bh * 0.6) / (h + 1e-6)
        center_x     = bx + bw / 2

        # Torso angle from minAreaRect
        rect      = cv2.minAreaRect(largest)
        angle_raw = rect[2]
        torso_angle_deg = (90 + angle_raw) if bw > bh else abs(angle_raw)

        feat = {
            "shoulder_center":    (center_x, by + bh * 0.2),
            "hip_center":         (center_x, by + bh * 0.6),
            "torso_angle_deg":    float(torso_angle_deg),
            "shoulder_angle_deg": float(torso_angle_deg),
            "aspect_ratio":       float(aspect_ratio),
            "hip_y_norm":         float(hip_y_norm),
            "torso_len_norm":     float(bh / (h + 1e-6)),
            "visibility_ok":      True,
            "_blob_rect":         (bx, by, bw, bh),
            "_blob_area":         float(area),
        }
        return feat, area

    # ------------------------------------------------------------------ #
    def reset(self):
        self.__init__()
