# src/pose_estimator.py
"""
Wraps the MediaPipe Tasks PoseLandmarker API (mediapipe >= 0.10 / 1.x).
Uses VIDEO running mode so timestamps drive tracking between frames.
"""

import os
import sys
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

# Convenience aliases for the new tasks API
_vision      = mp.tasks.vision
_BaseOptions = mp.tasks.BaseOptions
_PoseLandmarker        = _vision.PoseLandmarker
_PoseLandmarkerOptions = _vision.PoseLandmarkerOptions
_RunningMode           = _vision.RunningMode
_drawing_utils  = _vision.drawing_utils
_drawing_styles = _vision.drawing_styles
_PoseLandmarksConnections = _vision.PoseLandmarksConnections


# ── Landmark index constants (same numbering as old solutions API) ─
class _LM:
    LEFT_SHOULDER  = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP       = 23
    RIGHT_HIP      = 24
    LEFT_KNEE      = 25
    RIGHT_KNEE     = 26
    LEFT_ANKLE     = 27
    RIGHT_ANKLE    = 28


LANDMARK_NAMES = {
    "left_shoulder":  _LM.LEFT_SHOULDER,
    "right_shoulder": _LM.RIGHT_SHOULDER,
    "left_hip":       _LM.LEFT_HIP,
    "right_hip":      _LM.RIGHT_HIP,
    "left_knee":      _LM.LEFT_KNEE,
    "right_knee":     _LM.RIGHT_KNEE,
    "left_ankle":     _LM.LEFT_ANKLE,
    "right_ankle":    _LM.RIGHT_ANKLE,
}


# ──────────────────────────────────────────────────────────────────
def ensure_model(model_path: str, url: str) -> None:
    """Download the .task model file if not already present."""
    if os.path.exists(model_path):
        print(f"[PoseEstimator] Model found: {model_path}")
        return
    print(f"[PoseEstimator] Downloading model from:\n  {url}")
    print("[PoseEstimator] This only happens once (~5 MB) ...")
    try:
        urllib.request.urlretrieve(url, model_path)
        print(f"[PoseEstimator] Saved to: {model_path}")
    except Exception as exc:
        raise RuntimeError(
            f"[PoseEstimator] Could not download model: {exc}\n"
            "Download manually and place as: " + model_path
        ) from exc


# ──────────────────────────────────────────────────────────────────
class PoseEstimator:
    """Thin wrapper around MediaPipe Tasks PoseLandmarker."""

    def __init__(self):
        ensure_model(config.POSE_MODEL_PATH, config.POSE_MODEL_URL)

        options = _PoseLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=config.POSE_MODEL_PATH),
            running_mode=_RunningMode.VIDEO,
            num_poses=config.POSE_NUM_POSES,
            min_pose_detection_confidence=config.POSE_MIN_DETECTION_CONF,
            min_pose_presence_confidence=config.POSE_MIN_PRESENCE_CONF,
            min_tracking_confidence=config.POSE_MIN_TRACKING_CONF,
            output_segmentation_masks=False,
        )
        self._landmarker = _PoseLandmarker.create_from_options(options)
        self._frame_ms   = 0   # monotonically increasing timestamp

    # ------------------------------------------------------------------ #
    def process(self, rgb_frame: np.ndarray):
        """
        Run pose on an RGB frame.
        Returns a PoseLandmarkerResult (or None on error).
        """
        try:
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )
            self._frame_ms += 33          # ~30 fps; must be strictly increasing
            result = self._landmarker.detect_for_video(mp_image, self._frame_ms)
            return result
        except Exception as exc:
            print(f"[PoseEstimator] inference error: {exc}")
            return None

    # ------------------------------------------------------------------ #
    def extract_landmarks(self, result, frame_w: int, frame_h: int):
        """
        Extract the 8 key landmarks as pixel coordinates from the first pose.

        Returns
        -------
        dict  {name: (px_x, px_y, visibility)}  or  None if no pose found.
        """
        if result is None or not result.pose_landmarks:
            return None

        lms = result.pose_landmarks[0]   # first (and only) pose
        out = {}
        for name, idx in LANDMARK_NAMES.items():
            if idx >= len(lms):
                continue
            lm = lms[idx]
            out[name] = (
                int(lm.x * frame_w),
                int(lm.y * frame_h),
                float(lm.visibility) if lm.visibility is not None else 1.0,
            )
        return out if out else None

    # ------------------------------------------------------------------ #
    def draw_skeleton(self, bgr_frame: np.ndarray, result) -> np.ndarray:
        """Draw the full pose skeleton onto bgr_frame (in-place)."""
        if result is None or not result.pose_landmarks:
            return bgr_frame

        h, w = bgr_frame.shape[:2]

        for pose_landmarks in result.pose_landmarks:
            # draw_landmarks expects NormalizedLandmark list
            _drawing_utils.draw_landmarks(
                bgr_frame,
                pose_landmarks,
                _PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=_drawing_utils.DrawingSpec(
                    color=config.COLOR_LANDMARK,
                    thickness=2,
                    circle_radius=3,
                ),
                connection_drawing_spec=_drawing_utils.DrawingSpec(
                    color=config.COLOR_SKELETON,
                    thickness=2,
                ),
            )
        return bgr_frame

    # ------------------------------------------------------------------ #
    def close(self):
        self._landmarker.close()
