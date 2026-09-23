# main.py
"""
Phase 1 – Local video-based possible-fall detector.

Detection strategy (dual)
─────────────────────────
Primary  : MediaPipe PoseLandmarker  (skeleton + accurate features)
Fallback : OpenCV MOG2 blob detector (works when person is on the floor
           and MediaPipe cannot find a skeleton)

Two-phase alert
───────────────
1. FALL DETECTED  (green box + countdown)   – fall confirmed, monitoring…
2. NO RESPONSE    (red border + large text) – no recovery after 5 s

Usage:
    python main.py --source "path\\to\\video.mp4"
    python main.py --source "path\\to\\video.mp4" --output out.mp4

Keys:  q=quit   p=pause/resume   r=reset detector
"""

import argparse
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import numpy as np

import config
from src.pose_estimator    import PoseEstimator
from src.blob_detector     import BlobDetector
from src.feature_extractor import compute_features
from src.fall_detector     import FallDetector
from src.visualization     import draw_status_overlay


# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Fall detector – dual mode")
    p.add_argument("--source", required=True, help="Input MP4 path.")
    p.add_argument("--output", default=None,  help="Save annotated MP4 here.")
    p.add_argument("--show-mask", action="store_true",
                   help="Show the foreground mask in a second window.")
    return p.parse_args()


def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {path}")
        sys.exit(1)
    return cap


def draw_blob(frame, features: dict):
    """Draw the bounding rect of the blob when MediaPipe didn't fire."""
    rect = features.get("_blob_rect")
    if rect:
        bx, by, bw, bh = rect
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 165, 0), 2)


# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    cap  = open_video(args.source)

    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Source : {args.source}  ({src_w}x{src_h})")
    print(f"[INFO] FPS    : {video_fps:.2f}  |  Frames: {total_frames}  "
          f"({total_frames/video_fps:.1f}s)")

    pose     = PoseEstimator()
    blob     = BlobDetector()
    detector = FallDetector(video_fps=video_fps)

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            args.output, fourcc, video_fps,
            (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT),
        )
        print(f"[INFO] Output : {args.output}")

    paused      = False
    frame_idx   = 0
    fps_display = 0.0
    t_prev      = time.perf_counter()

    print("[INFO] Keys: q=quit  p=pause/resume  r=reset")
    print("[INFO] Processing…")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            paused = not paused
            print(f"[INFO] {'Paused' if paused else 'Resumed'}")
        if key == ord('r'):
            detector.reset()
            blob.reset()
            print("[INFO] Detector reset.")

        if paused:
            time.sleep(0.05)
            continue

        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video.")
            break

        frame_idx += 1
        if frame_idx % config.FRAME_SKIP != 0:
            continue

        video_time_sec = frame_idx / video_fps

        # ── Resize keeping original aspect ratio ──────────────────────
        # PROCESS_WIDTH and PROCESS_HEIGHT set to match video orientation
        proc = cv2.resize(frame, (config.PROCESS_WIDTH, config.PROCESS_HEIGHT))
        h, w = proc.shape[:2]

        # ── MediaPipe pose ────────────────────────────────────────────
        rgb       = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        mp_result = pose.process(rgb)
        landmarks = pose.extract_landmarks(mp_result, w, h)
        mp_feat   = compute_features(landmarks, w, h) if landmarks else {}

        # ── OpenCV blob fallback ──────────────────────────────────────
        blob_feat, fg_mask = blob.update(proc)

        # ── Choose features: prefer MediaPipe, fall back to blob ──────
        using_mp = bool(mp_feat.get("visibility_ok"))
        features = mp_feat if using_mp else blob_feat
        source   = "MP" if using_mp else "BL"

        # ── Fall detection ────────────────────────────────────────────
        status    = detector.update(features, video_time_sec)
        countdown = detector.countdown_remaining(video_time_sec)

        # ── Draw skeleton (MP) or blob rect (fallback) ────────────────
        if using_mp:
            pose.draw_skeleton(proc, mp_result)
        else:
            draw_blob(proc, blob_feat)

        # ── Source indicator (top-right) ──────────────────────────────
        src_label = f"Detector: {source}"
        src_color = (0, 220, 0) if using_mp else (0, 165, 255)
        cv2.putText(proc, src_label, (w - 130, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(proc, src_label, (w - 130, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, src_color, 1, cv2.LINE_AA)

        # ── Overlay ───────────────────────────────────────────────────
        draw_status_overlay(proc, status, fps_display, features,
                            detector.persist_count, countdown)

        # ── Resize for display / output ───────────────────────────────
        display = cv2.resize(proc, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
        cv2.imshow("Fall Detector", display)

        if args.show_mask:
            mask_disp = cv2.resize(fg_mask, (config.DISPLAY_WIDTH // 2,
                                             config.DISPLAY_HEIGHT // 2))
            cv2.imshow("Foreground mask", mask_disp)

        if writer:
            writer.write(display)

        # ── FPS ───────────────────────────────────────────────────────
        t_now       = time.perf_counter()
        fps_display = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev      = t_now

        if frame_idx % 30 == 0:
            pct = frame_idx / max(total_frames, 1) * 100
            print(f"[{pct:3.0f}%] f={frame_idx}  t={video_time_sec:.1f}s  "
                  f"src={source}  status={status}  "
                  f"ratio={features.get('aspect_ratio',0):.2f}  "
                  f"angle={features.get('torso_angle_deg',0):.1f}")

    # ── Cleanup ───────────────────────────────────────────────────────
    cap.release()
    pose.close()
    if writer:
        writer.release()
        print(f"[INFO] Saved → {args.output}")
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
