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

_vision      = mp.tasks.vision
_BaseOptions = mp.tasks.BaseOptions
_PoseLandmarker        = _vision.PoseLandmarker
_PoseLandmarkerOptions = _vision.PoseLandmarkerOptions
_RunningMode           = _vision.RunningMode


class _LM:
    NOSE           = 0
    LEFT_EYE       = 2
    RIGHT_EYE      = 5
    LEFT_EAR       = 7
    RIGHT_EAR      = 8
    LEFT_SHOULDER  = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW     = 13
    RIGHT_ELBOW    = 14
    LEFT_WRIST     = 15
    RIGHT_WRIST    = 16
    LEFT_HIP       = 23
    RIGHT_HIP      = 24
    LEFT_KNEE      = 25
    RIGHT_KNEE     = 26
    LEFT_ANKLE     = 27
    RIGHT_ANKLE    = 28
    LEFT_HEEL      = 29
    RIGHT_HEEL     = 30
    LEFT_FOOT_IDX  = 31
    RIGHT_FOOT_IDX = 32


LANDMARK_NAMES = {
    "nose":           _LM.NOSE,
    "left_shoulder":  _LM.LEFT_SHOULDER,
    "right_shoulder": _LM.RIGHT_SHOULDER,
    "left_elbow":     _LM.LEFT_ELBOW,
    "right_elbow":    _LM.RIGHT_ELBOW,
    "left_wrist":     _LM.LEFT_WRIST,
    "right_wrist":    _LM.RIGHT_WRIST,
    "left_hip":       _LM.LEFT_HIP,
    "right_hip":      _LM.RIGHT_HIP,
    "left_knee":      _LM.LEFT_KNEE,
    "right_knee":     _LM.RIGHT_KNEE,
    "left_ankle":     _LM.LEFT_ANKLE,
    "right_ankle":    _LM.RIGHT_ANKLE,
}

# Major clean body skeleton pairs (joint1, joint2)
CLEAN_POSE_CONNECTIONS = [
    (_LM.LEFT_SHOULDER,  _LM.RIGHT_SHOULDER),
    (_LM.LEFT_SHOULDER,  _LM.LEFT_HIP),
    (_LM.RIGHT_SHOULDER, _LM.RIGHT_HIP),
    (_LM.LEFT_HIP,       _LM.RIGHT_HIP),
    (_LM.LEFT_SHOULDER,  _LM.LEFT_ELBOW),
    (_LM.LEFT_ELBOW,     _LM.LEFT_WRIST),
    (_LM.RIGHT_SHOULDER, _LM.RIGHT_ELBOW),
    (_LM.RIGHT_ELBOW,    _LM.RIGHT_WRIST),
    (_LM.LEFT_HIP,       _LM.LEFT_KNEE),
    (_LM.LEFT_KNEE,      _LM.LEFT_ANKLE),
    (_LM.RIGHT_HIP,      _LM.RIGHT_KNEE),
    (_LM.RIGHT_KNEE,     _LM.RIGHT_ANKLE),
    (_LM.NOSE,           _LM.LEFT_SHOULDER),
    (_LM.NOSE,           _LM.RIGHT_SHOULDER),
]


def ensure_model(model_path: str, url: str) -> None:
    """Download the .task model file if not already present."""
    if os.path.exists(model_path):
        return
    print(f"[PoseEstimator] Downloading model from:\n  {url}")
    try:
        urllib.request.urlretrieve(url, model_path)
        print(f"[PoseEstimator] Saved to: {model_path}")
    except Exception as exc:
        raise RuntimeError(
            f"[PoseEstimator] Could not download model: {exc}\n"
            "Download manually and place as: " + model_path
        ) from exc


class PoseEstimator:
    """Wrapper around MediaPipe Tasks PoseLandmarker."""

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
        self._frame_ms   = 0

    def process(self, rgb_frame: np.ndarray):
        """Run pose on an RGB frame."""
        try:
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )
            self._frame_ms += 33
            result = self._landmarker.detect_for_video(mp_image, self._frame_ms)
            return result
        except Exception as exc:
            print(f"[PoseEstimator] inference error: {exc}")
            return None

    def extract_landmarks(self, result, frame_w: int, frame_h: int):
        """
        Extract named key landmarks as pixel coordinates AND full landmark array.

        Returns
        -------
        dict {name: (px_x, px_y, visibility)} + "_all_points": [(px_x, px_y, vis), ...]
        """
        if result is None or not result.pose_landmarks:
            return None

        lms = result.pose_landmarks[0]
        out = {}
        all_pts = []

        for lm in lms:
            px = int(lm.x * frame_w)
            py = int(lm.y * frame_h)
            vis = float(lm.visibility) if lm.visibility is not None else 1.0
            all_pts.append((px, py, vis))

        out["_all_points"] = all_pts

        for name, idx in LANDMARK_NAMES.items():
            if idx < len(lms):
                out[name] = all_pts[idx]

        return out if len(all_pts) > 0 else None

    def draw_skeleton(self, bgr_frame: np.ndarray, result, color_skeleton=None, color_joint=None) -> np.ndarray:
        """Draw a clean, sleek pose skeleton onto bgr_frame without face/hand clutter."""
        if result is None or not result.pose_landmarks:
            return bgr_frame

        h, w = bgr_frame.shape[:2]
        lms  = result.pose_landmarks[0]

        skel_color  = color_skeleton or config.COLOR_SKELETON
        joint_color = color_joint or config.COLOR_LANDMARK

        # Convert to pixel coords
        pts = {}
        for idx, lm in enumerate(lms):
            vis = float(lm.visibility) if lm.visibility is not None else 1.0
            pts[idx] = (int(lm.x * w), int(lm.y * h), vis)

        # Draw clean limbs
        for idx1, idx2 in CLEAN_POSE_CONNECTIONS:
            if idx1 in pts and idx2 in pts:
                p1, p2 = pts[idx1], pts[idx2]
                if p1[2] > 0.35 and p2[2] > 0.35:
                    cv2.line(bgr_frame, (p1[0], p1[1]), (p2[0], p2[1]), (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.line(bgr_frame, (p1[0], p1[1]), (p2[0], p2[1]), skel_color, 2, cv2.LINE_AA)

        # Draw joint nodes
        for idx, (px, py, vis) in pts.items():
            if vis > 0.4 and idx in [
                _LM.NOSE, _LM.LEFT_SHOULDER, _LM.RIGHT_SHOULDER,
                _LM.LEFT_ELBOW, _LM.RIGHT_ELBOW, _LM.LEFT_WRIST, _LM.RIGHT_WRIST,
                _LM.LEFT_HIP, _LM.RIGHT_HIP, _LM.LEFT_KNEE, _LM.RIGHT_KNEE,
                _LM.LEFT_ANKLE, _LM.RIGHT_ANKLE
            ]:
                cv2.circle(bgr_frame, (px, py), 5, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(bgr_frame, (px, py), 3, joint_color, -1, cv2.LINE_AA)

        return bgr_frame

    def close(self):
        self._landmarker.close()

