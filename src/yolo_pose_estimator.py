# src/yolo_pose_estimator.py
"""
YOLOv8-Pose Estimator Wrapper.
Provides robust person detection & 17 COCO keypoints pose estimation.
Eliminates false detections on air, cars, or background clutter.
"""

import os
import sys
import numpy as np
import cv2
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

# COCO Keypoints Mapping (0..16)
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

YOLO_LANDMARK_NAMES = {
    "nose": 0,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# Major clean body skeleton pairs (COCO indices)
YOLO_POSE_CONNECTIONS = [
    (5, 6),   # shoulders
    (5, 11),  # left torso
    (6, 12),  # right torso
    (11, 12), # hips
    (5, 7),   # left upper arm
    (7, 9),   # left forearm
    (6, 8),   # right upper arm
    (8, 10),  # right forearm
    (11, 13), # left thigh
    (13, 15), # left leg
    (12, 14), # right thigh
    (14, 16), # right leg
    (0, 5),   # nose to left shoulder
    (0, 6),   # nose to right shoulder
]

class YoloPoseResultWrapper:
    """Wrapper to maintain compatibility with MediaPipe Pose result format."""
    def __init__(self, keypoints_xy, confs, bbox_xyxy):
        self.keypoints_xy = keypoints_xy # (17, 2)
        self.confs = confs               # (17,)
        self.bbox_xyxy = bbox_xyxy       # (4,) [x1, y1, x2, y2]
        self.pose_landmarks = [self] if keypoints_xy is not None else []

class YoloPoseEstimator:
    """Ultra-accurate pose estimator powered by YOLOv8-Pose."""

    def __init__(self, model_name: str = "yolov8n-pose.pt", conf_thresh: float = 0.45):
        print(f"[YoloPoseEstimator] Initializing {model_name} (conf_thresh={conf_thresh})...")
        self.model = YOLO(model_name)
        self.conf_thresh = conf_thresh

    def process(self, rgb_frame: np.ndarray):
        """Run YOLO pose inference on an RGB frame (OpenCV BGR converted to RGB)."""
        # YOLO accepts RGB or BGR numpy array directly
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        results = self.model(bgr_frame, verbose=False, conf=self.conf_thresh)
        
        if not results or len(results) == 0:
            return YoloPoseResultWrapper(None, None, None)

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None or len(res.keypoints) == 0:
            return YoloPoseResultWrapper(None, None, None)

        # Select the person detection with the highest confidence
        boxes = res.boxes
        best_idx = int(boxes.conf.argmax())
        best_conf = float(boxes.conf[best_idx])
        
        if best_conf < self.conf_thresh:
            return YoloPoseResultWrapper(None, None, None)

        bbox = boxes.xyxy[best_idx].cpu().numpy() # [x1, y1, x2, y2]
        kpts = res.keypoints[best_idx]
        
        xy = kpts.xy[0].cpu().numpy() # shape (17, 2)
        if kpts.conf is not None:
            confs = kpts.conf[0].cpu().numpy() # shape (17,)
        else:
            confs = np.ones(17)

        return YoloPoseResultWrapper(xy, confs, bbox)

    def extract_landmarks(self, result: YoloPoseResultWrapper, frame_w: int, frame_h: int):
        """
        Extract named key landmarks as pixel coordinates AND full landmark array.

        Returns
        -------
        dict {name: (px_x, px_y, visibility)} + "_all_points": [(px_x, px_y, vis), ...]
        """
        if result is None or not result.pose_landmarks or result.keypoints_xy is None:
            return None

        xy = result.keypoints_xy
        confs = result.confs
        
        out = {}
        all_pts = []

        for idx in range(17):
            px = int(xy[idx][0])
            py = int(xy[idx][1])
            vis = float(confs[idx]) if idx < len(confs) else 1.0
            all_pts.append((px, py, vis))

        out["_all_points"] = all_pts

        for name, idx in YOLO_LANDMARK_NAMES.items():
            if idx < len(all_pts):
                out[name] = all_pts[idx]

        # Add explicit bounding box from YOLO detection if available
        if result.bbox_xyxy is not None:
            b = result.bbox_xyxy
            out["yolo_bbox"] = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))

        return out if len(all_pts) > 0 else None

    def draw_skeleton(self, bgr_frame: np.ndarray, result: YoloPoseResultWrapper, color_skeleton=None, color_joint=None) -> np.ndarray:
        """Draw a clean body skeleton using YOLO keypoint results."""
        if result is None or not result.pose_landmarks or result.keypoints_xy is None:
            return bgr_frame

        xy = result.keypoints_xy
        confs = result.confs

        skel_color  = color_skeleton or config.COLOR_SKELETON
        joint_color = color_joint or config.COLOR_LANDMARK

        pts = {}
        for idx in range(17):
            px, py = int(xy[idx][0]), int(xy[idx][1])
            vis = float(confs[idx]) if idx < len(confs) else 1.0
            pts[idx] = (px, py, vis)

        # Draw clean limbs
        for idx1, idx2 in YOLO_POSE_CONNECTIONS:
            if idx1 in pts and idx2 in pts:
                p1, p2 = pts[idx1], pts[idx2]
                if p1[2] > 0.35 and p2[2] > 0.35 and p1[0] > 0 and p2[0] > 0:
                    cv2.line(bgr_frame, (p1[0], p1[1]), (p2[0], p2[1]), (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.line(bgr_frame, (p1[0], p1[1]), (p2[0], p2[1]), skel_color, 2, cv2.LINE_AA)

        # Draw major joint nodes
        for idx in [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
            if idx in pts:
                px, py, vis = pts[idx]
                if vis > 0.35 and px > 0 and py > 0:
                    cv2.circle(bgr_frame, (px, py), 5, (0, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(bgr_frame, (px, py), 3, joint_color, -1, cv2.LINE_AA)

        return bgr_frame

    def close(self):
        pass
