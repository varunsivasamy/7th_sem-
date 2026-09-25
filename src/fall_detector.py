# src/fall_detector.py
"""
Two-phase fall detector.  Uses VIDEO time (frame number × fps), not
wall-clock time, so it works correctly when processing a saved file.

Phase 1 – FALL DETECTED  (green warning box)
    Fires when pose criteria are met for FALL_PERSISTENCE_FRAMES frames.
    Starts a countdown of NO_RESPONSE_SECONDS (measured in video time).

Phase 2 – NO RESPONSE  (red border + large text)
    Fires if the person has NOT returned to an upright posture within
    NO_RESPONSE_SECONDS of video time after the fall was first detected.

Recovery:
    If the person stands back up before the timer expires the state
    returns to NORMAL with no red alert.
"""

from collections import deque

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


# ── Public status strings ─────────────────────────────────────────
STATUS_NORMAL      = "NORMAL"
STATUS_NO_POSE     = "POSE NOT DETECTED"
STATUS_FALL_WARN   = "FALL DETECTED"
STATUS_NO_RESPONSE = "NO RESPONSE – RED ALERT"


class FallDetector:
    """
    Feed one feature-dict per frame via update(features, video_time_sec).
    video_time_sec = frame_index / fps  (caller computes this).

    State machine
    ─────────────
    NORMAL ──(criteria met N frames)──► FALL DETECTED
                                             │
               ┌── person stands up ─────────┤
               │                             │ (NO_RESPONSE_SECONDS elapsed)
               ▼                             ▼
            NORMAL                     NO RESPONSE
                                            │
                        (RED_ALERT_HOLD_SECONDS) ──► NORMAL
    """

    def __init__(self, video_fps: float = 30.0):
        self._fps = max(video_fps, 1.0)
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self):
        self._hip_y_history     = deque(maxlen=config.HISTORY_FRAMES)
        self._angle_history     = deque(maxlen=config.HISTORY_FRAMES)
        self._shoulder_ang_hist = deque(maxlen=config.HISTORY_FRAMES)
        self._ratio_history     = deque(maxlen=config.HISTORY_FRAMES)

        self._persist_count     = 0
        self._recovery_count    = 0

        # Phase-1
        self._fall_detected     = False
        self._fall_start_t      = None    # video-time when fall was confirmed

        # Phase-2
        self._no_response       = False
        self._red_alert_end_t   = None    # video-time when red alert expires
        self._last_recovery_t   = None


    # ------------------------------------------------------------------ #
    def update(self, features: dict, video_time_sec: float) -> str:
        """
        Parameters
        ----------
        features       : output of compute_features()  (may be empty dict)
        video_time_sec : frame_index / video_fps

        Returns
        -------
        One of STATUS_* strings.
        """
        t = video_time_sec

        # ── Red-alert hold ────────────────────────────────────────────
        if self._no_response:
            if t >= self._red_alert_end_t:
                self._clear_all()
            else:
                return STATUS_NO_RESPONSE

        # ── No valid human pose ───────────────────────────────────────
        if not features or not features.get("is_pose_valid", False):
            if self._fall_detected:
                elapsed = t - self._fall_start_t
                if elapsed >= config.NO_RESPONSE_SECONDS:
                    self._trigger_red(t)
                    return STATUS_NO_RESPONSE
                return STATUS_FALL_WARN
            return STATUS_NO_POSE


        # ── Update rolling histories ──────────────────────────────────
        self._hip_y_history.append(features["hip_y_norm"])
        self._angle_history.append(features["torso_angle_deg"])
        self._shoulder_ang_hist.append(features.get("shoulder_angle_deg", 0.0))
        self._ratio_history.append(features["aspect_ratio"])

        # ── Already in warning phase ──────────────────────────────────
        if self._fall_detected:
            if self._is_recovering(features):
                # Instantly cancel countdown on the very first frame of wake up / standing
                self._fall_detected   = False
                self._fall_start_t    = None
                self._persist_count   = 0
                self._recovery_count  = 0
                self._last_recovery_t = t
                return STATUS_NORMAL


            elapsed = t - self._fall_start_t
            if elapsed >= config.NO_RESPONSE_SECONDS:
                self._trigger_red(t)
                return STATUS_NO_RESPONSE

            return STATUS_FALL_WARN   # green phase, countdown running

        # ── Normal evaluation ─────────────────────────────────────────
        if t < config.WARMUP_SECONDS:
            return STATUS_NORMAL

        # Post-recovery window: allow 3.0 seconds for person to fully stand up without re-triggering
        if hasattr(self, "_last_recovery_t") and self._last_recovery_t is not None:
            if t - self._last_recovery_t < 3.0:
                return STATUS_NORMAL

        if self._evaluate():
            self._persist_count += 1
        else:
            self._persist_count = max(0, self._persist_count - 1)

        if self._persist_count >= config.FALL_PERSISTENCE_FRAMES:
            self._fall_detected = True
            self._fall_start_t  = t
            self._persist_count = 0
            return STATUS_FALL_WARN

        return STATUS_NORMAL


    # ------------------------------------------------------------------ #
    def _trigger_red(self, t: float):
        self._no_response     = True
        self._fall_detected   = False
        self._fall_start_t    = None
        self._red_alert_end_t = t + config.RED_ALERT_HOLD_SECONDS

    def _clear_all(self):
        self._no_response     = False
        self._fall_detected   = False
        self._fall_start_t    = None
        self._persist_count   = 0
        self._recovery_count  = 0
        self._hip_y_history.clear()
        self._angle_history.clear()
        self._shoulder_ang_hist.clear()
        self._ratio_history.clear()

    # ------------------------------------------------------------------ #
    def _is_recovering(self, features: dict) -> bool:
        """
        Check if a fallen person is getting back up / standing up.
        Cancels countdown immediately when posture returns upright or hip moves upward.
        """
        angle = features.get("torso_angle_deg", 90.0)
        ratio = features.get("aspect_ratio", 1.0)
        shld  = features.get("shoulder_angle_deg", 90.0)
        hip_y = features.get("hip_y_norm", 1.0)

        # Posture returning upright (person lifting head/torso or sitting/standing up)
        upright_posture = (angle < 50.0) and (shld < 45.0) and (ratio < 0.75)


        # Hip Y moving UPWARD back toward standing position
        hip_upward = False
        if len(self._hip_y_history) >= 3:
            hips = list(self._hip_y_history)
            lowest_hip = max(hips)
            if (lowest_hip - hip_y) > 0.04 and angle < 55.0:
                hip_upward = True

        return upright_posture or hip_upward

    # ------------------------------------------------------------------ #
    def _evaluate(self) -> bool:
        """
        Evaluate if a human fall onto the floor has occurred.
        Distinguishes bending over (standing, hips elevated) vs falling (hips dropped low + horizontal posture).
        """
        if not self._angle_history or not self._hip_y_history:
            return False

        angle = self._angle_history[-1]
        shld  = self._shoulder_ang_hist[-1]
        ratio = self._ratio_history[-1]
        hip_y = self._hip_y_history[-1]

        # 1. Standing posture gate: if torso is upright (< 35°) AND ratio is narrow (< 0.70), NOT a fall
        if angle < 35.0 and ratio < 0.70:
            return False

        # 2. Bending over while standing (hips elevated in upper half of frame, hip_y < 0.52 and narrow ratio)
        if hip_y < 0.52 and ratio < 0.85:
            return False

        # 3. Real Fall on floor: Torso horizontal (angle > 38°), shoulder tilt (shld > 20°), or wide aspect ratio (> 0.80)
        posture_fallen = (angle > 38.0) or (shld > 22.0)
        ratio_fallen   = (ratio > 0.80)
        ground_level   = (hip_y > 0.55)  # hips low in frame near floor

        return (posture_fallen or ratio_fallen) and ground_level






    # ------------------------------------------------------------------ #
    def countdown_remaining(self, video_time_sec: float) -> float:
        """Seconds left in the green-phase countdown (0.0 if not active)."""
        if self._fall_detected and self._fall_start_t is not None:
            return max(0.0, config.NO_RESPONSE_SECONDS - (video_time_sec - self._fall_start_t))
        return 0.0

    @property
    def persist_count(self) -> int:
        return self._persist_count

    @property
    def fall_detected(self) -> bool:
        return self._fall_detected

    @property
    def no_response(self) -> bool:
        return self._no_response
