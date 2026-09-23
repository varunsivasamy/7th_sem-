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

        # ── No usable pose ────────────────────────────────────────────
        if not features or not features.get("visibility_ok", False):
            if self._fall_detected:
                if t - self._fall_start_t >= config.NO_RESPONSE_SECONDS:
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
            if self._is_upright(features):
                self._recovery_count += 1
                # Require several consecutive upright frames to confirm recovery
                if self._recovery_count >= config.RECOVERY_PERSISTENCE_FRAMES:
                    self._fall_detected  = False
                    self._fall_start_t   = None
                    self._persist_count  = 0
                    self._recovery_count = 0
                    return STATUS_NORMAL
            else:
                self._recovery_count = 0   # reset if not upright

            elapsed = t - self._fall_start_t
            if elapsed >= config.NO_RESPONSE_SECONDS:
                self._trigger_red(t)
                return STATUS_NO_RESPONSE

            return STATUS_FALL_WARN   # green phase, countdown running

        # ── Normal evaluation ─────────────────────────────────────────
        # Skip detection during warm-up period
        if t < config.WARMUP_SECONDS:
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
    def _is_upright(self, features: dict) -> bool:
        ratio_ok = features.get("aspect_ratio",     1.0) < config.RECOVERY_ASPECT_RATIO
        angle_ok = features.get("torso_angle_deg",   90) < config.TORSO_ANGLE_FALL_THRESHOLD
        shld_ok  = features.get("shoulder_angle_deg", 90) < config.SHOULDER_ANGLE_FALL_THRESHOLD
        return ratio_ok and angle_ok and shld_ok

    # ------------------------------------------------------------------ #
    def _evaluate(self) -> bool:
        if len(self._hip_y_history) < 2:
            return False

        posture_ok = (
            self._angle_history[-1]     > config.TORSO_ANGLE_FALL_THRESHOLD or
            self._shoulder_ang_hist[-1] > config.SHOULDER_ANGLE_FALL_THRESHOLD
        )
        ratio_ok    = self._ratio_history[-1] > config.ASPECT_RATIO_FALL_THRESHOLD
        hip_arr     = list(self._hip_y_history)
        movement_ok = (max(hip_arr) - min(hip_arr)) > config.HIP_MOVEMENT_THRESHOLD

        return posture_ok and ratio_ok and movement_ok

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
