# config.py
# All tunable thresholds and settings live here.

# ── MediaPipe Pose (tasks API – mediapipe >= 0.10 / 1.x) ─────────
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)
POSE_MODEL_PATH          = "pose_landmarker_heavy.task"
POSE_MIN_DETECTION_CONF  = 0.25
POSE_MIN_PRESENCE_CONF   = 0.25
POSE_MIN_TRACKING_CONF   = 0.25
POSE_NUM_POSES           = 1


# ── Tracking & Smoothing ──────────────────────────────────────────
BBOX_SMOOTH_ALPHA        = 0.40        # EMA factor for bounding box tracking (0.0 - 1.0)
BBOX_PADDING_FRAC        = 0.08        # Padding ratio around detected body box

# ── Video processing ──────────────────────────────────────────────
# Portrait phone video (1080×1920) → keep aspect ratio
# Use 360 wide × 640 tall so MediaPipe sees the correct proportions.
PROCESS_WIDTH  = 360
PROCESS_HEIGHT = 640
FRAME_SKIP     = 1          # process every Nth frame (1 = all)

# ── Fall detection thresholds ─────────────────────────────────────
# Torso angle: 0° = upright, 90° = fully horizontal
TORSO_ANGLE_FALL_THRESHOLD    = 50.0

# Shoulder line tilt (catches sideways falls)
SHOULDER_ANGLE_FALL_THRESHOLD = 40.0

# Bounding-box width/height ratio: upright ≈ 0.3; fallen > 0.85
ASPECT_RATIO_FALL_THRESHOLD   = 0.85

# Hip Y change over history window (fraction of frame height)
HIP_MOVEMENT_THRESHOLD        = 0.04

# Rolling history window (frames)
HISTORY_FRAMES = 20

# Frames the fall condition must persist before "FALL DETECTED" fires
FALL_PERSISTENCE_FRAMES = 4


# Consecutive upright/recovering frames needed to confirm recovery (1 = immediate on wake up)
RECOVERY_PERSISTENCE_FRAMES = 1


# Warm-up: ignore detections for first N seconds (background model settling)
WARMUP_SECONDS = 0.2

# "FALL DETECTED" grace period before escalating to "NO RESPONSE"
# Set as real-world seconds — converted to frames at runtime using video FPS
NO_RESPONSE_SECONDS = 10.0

# After NO RESPONSE fires, hold the red alert for this many seconds
# before allowing the detector to return to NORMAL
RED_ALERT_HOLD_SECONDS = 3.0

# Upright-recovery: aspect ratio must drop below this to consider
# the person "standing again" after a fall
RECOVERY_ASPECT_RATIO = 0.70


# ── Display ───────────────────────────────────────────────────────
DISPLAY_WIDTH  = 360
DISPLAY_HEIGHT = 640

FONT_SCALE_SMALL  = 0.6
FONT_SCALE_LARGE  = 1.6
FONT_THICKNESS    = 2

COLOR_NORMAL      = (0,   220,  100)    # BGR - sleek emerald green
COLOR_NO_POSE     = (0,   200,  230)    # BGR - warm gold/yellow
COLOR_FALL_WARN   = (0,   220,  100)    # BGR - green "FALL DETECTED"
COLOR_NO_RESPONSE = (40,   40,  235)    # BGR - rich red "NO RESPONSE"
COLOR_SKELETON    = (255, 230,  50)     # BGR - bright cyan / sky blue
COLOR_LANDMARK    = (255,  80, 200)     # BGR - vibrant magenta / violet
BORDER_THICKNESS  = 12

