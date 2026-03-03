import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import time
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque

# Mouse control
try:
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE    = 0
    MOUSE_AVAILABLE = True
except ImportError:
    MOUSE_AVAILABLE = False
    print("[WARN] pyautogui not found. Run:  pip install pyautogui")

# Optional: sound alert on Windows
try:
    import winsound
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False


# =============================================================================
#  AttentionTracker  v2
# =============================================================================
class AttentionTracker:

    def __init__(self):

        # ── MODEL ──────────────────────────────────────────────────────────── #
        self.model_path = "face_landmarker.task"
        if not os.path.exists(self.model_path):
            print("Downloading Face Landmarker model...")
            url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )
            urllib.request.urlretrieve(url, self.model_path)

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # ── CAMERA ─────────────────────────────────────────────────────────── #
        self.cap = cv2.VideoCapture(0)
        # self.cap = cv2.VideoCapture(1)  # use for external webcam
        if not self.cap.isOpened():
            raise Exception("Webcam not detected")
        self.fps_cam = self.cap.get(cv2.CAP_PROP_FPS) or 30

        # ── SCREEN SIZE (for mouse mapping) ────────────────────────────────── #
        if MOUSE_AVAILABLE:
            self.screen_w, self.screen_h = pyautogui.size()
        else:
            self.screen_w, self.screen_h = 1920, 1080

        # ── LANDMARK INDICES ───────────────────────────────────────────────── #
        self.LEFT_IRIS   = [474, 475, 476, 477]
        self.RIGHT_IRIS  = [469, 470, 471, 472]
        self.LEFT_EYE    = [33,  160, 158, 133, 153, 144]
        self.RIGHT_EYE   = [362, 385, 387, 263, 373, 380]
        self.L_EYE_LEFT  = 33
        self.L_EYE_RIGHT = 133
        self.R_EYE_LEFT  = 362
        self.R_EYE_RIGHT = 263

        # ── BLINK THRESHOLDS (personalised during calibration) ─────────────── #
        self.EAR_THRESHOLD_L = 0.23
        self.EAR_THRESHOLD_R = 0.23
        self.CONSEC_FRAMES   = 3        # min consecutive frames eye must be closed

        # ── SEPARATE EYE BLINK STATE ───────────────────────────────────────── #
        self.left_blink_counter  = 0
        self.right_blink_counter = 0
        self.left_blink_total    = 0
        self.right_blink_total   = 0

        # ── CLICK COOLDOWN (prevent accidental rapid-fire clicks) ──────────── #
        self.CLICK_COOLDOWN     = 0.6   # seconds
        self.last_left_click_t  = 0
        self.last_right_click_t = 0

        # ── GAZE MOUSE CONTROL ─────────────────────────────────────────────── #
        self.mouse_mode  = True         # press M to toggle
        self.mouse_alpha = 0.20         # EMA smoothing (lower = smoother/slower)
        self.mouse_x     = self.screen_w  // 2
        self.mouse_y     = self.screen_h  // 2

        # Gaze-to-screen calibration extents [gx_min, gx_max, gy_min, gy_max]
        self.gaze_calib  = [0.35, 0.65, 0.30, 0.70]   # updated in calibrate()

        # ── DROWSINESS ─────────────────────────────────────────────────────── #
        self.DROWSY_FRAMES   = int(self.fps_cam * 2)   # ~2 seconds
        self.drowsy_counter  = 0
        self.last_alert_time = 0

        # ── SMOOTHING (iris display dot) ───────────────────────────────────── #
        self.prev_x = None
        self.prev_y = None
        self.alpha  = 0.7

        # ── HEAD POSE ─────────────────────────────────────────────────────── #
        self.YAW_THRESHOLD   = 25
        self.PITCH_THRESHOLD = 20

        # ── SESSION ────────────────────────────────────────────────────────── #
        self.session_start = time.time()
        self.prev_time     = 0

        self.rolling_blinks = deque()
        self.rolling_gaze   = deque()
        self.log_rows       = []

        # ── DISPLAY HELPERS ────────────────────────────────────────────────── #
        self.live_attention     = "CALIBRATING"
        self.live_attention_pct = 0
        self.gaze_direction     = "CENTER"
        self.head_status        = "FORWARD"
        self.attention_history  = deque(maxlen=150)

        print("\n╔══════════════════════════════════════════╗")
        print("║    Attention Tracker v2 — Starting Up    ║")
        print("╠══════════════════════════════════════════╣")
        print("║  Gaze moves cursor                       ║")
        print("║  Left  eye blink → Left  click           ║")
        print("║  Right eye blink → Right click           ║")
        print("║  M key  → Toggle mouse control ON/OFF    ║")
        print("║  ESC    → End session & show report      ║")
        print("╚══════════════════════════════════════════╝\n")

    # =========================================================================
    #  CALIBRATION
    # =========================================================================
    def calibrate(self, duration=3):
        """
        Phase 1 – EAR calibration : collect per-eye EAR with eyes open.
        Phase 2 – Gaze calibration: look at 4 corners to map gaze -> screen.
        """
        # ── Phase 1: EAR ──────────────────────────────────────────────────── #
        print(f"[CALIBRATION] Phase 1: Keep both eyes open naturally for {duration}s ...")
        left_ears, right_ears = [], []
        deadline = time.time() + duration

        while time.time() < deadline:
            success, frame = self.cap.read()
            if not success:
                continue
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
            cd = int(deadline - time.time()) + 1
            cv2.putText(frame, "CALIBRATION  --  keep eyes open",
                        (w // 2 - 230, h // 2 - 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 220, 255), 2)
            cv2.putText(frame, f"{cd}s remaining",
                        (w // 2 - 80, h // 2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 100), 2)
            cv2.imshow("Attention Tracking System", frame)
            cv2.waitKey(1)

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.detector.detect_for_video(mp_img, int(time.time() * 1000))

            if result.face_landmarks:
                lm = result.face_landmarks[0]
                le = [(int(lm[i].x * w), int(lm[i].y * h)) for i in self.LEFT_EYE]
                re = [(int(lm[i].x * w), int(lm[i].y * h)) for i in self.RIGHT_EYE]
                left_ears.append(self.calculate_EAR(le))
                right_ears.append(self.calculate_EAR(re))

        if left_ears:
            self.EAR_THRESHOLD_L = round(np.mean(left_ears)  - 0.05, 4)
            self.EAR_THRESHOLD_R = round(np.mean(right_ears) - 0.05, 4)
            print(f"[CALIBRATION] Left  EAR threshold : {self.EAR_THRESHOLD_L}")
            print(f"[CALIBRATION] Right EAR threshold : {self.EAR_THRESHOLD_R}")
        else:
            print("[CALIBRATION] No face detected — using default threshold 0.23")

        # ── Phase 2: Gaze corners ─────────────────────────────────────────── #
        if not MOUSE_AVAILABLE:
            return

        print("[CALIBRATION] Phase 2: Gaze corner mapping ...")
        corners = [
            ("TOP-LEFT",     (0.05, 0.08)),
            ("TOP-RIGHT",    (0.95, 0.08)),
            ("BOTTOM-LEFT",  (0.05, 0.92)),
            ("BOTTOM-RIGHT", (0.95, 0.92)),
        ]
        all_gx, all_gy = [], []

        for label, (tx, ty) in corners:
            deadline = time.time() + 1.8
            while time.time() < deadline:
                success, frame = self.cap.read()
                if not success:
                    continue
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                cv2.putText(frame, f"Look at  {label}",
                            (w // 2 - 180, h // 2),
                            cv2.FONT_HERSHEY_DUPLEX, 0.95, (0, 220, 255), 2)
                dx, dy = int(tx * w), int(ty * h)
                cv2.circle(frame, (dx, dy), 16, (0, 255, 0), -1)
                cv2.circle(frame, (dx, dy), 20, (255, 255, 255), 2)
                cv2.imshow("Attention Tracking System", frame)
                cv2.waitKey(1)

                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self.detector.detect_for_video(mp_img, int(time.time() * 1000))
                if result.face_landmarks:
                    lm = result.face_landmarks[0]
                    gx = (lm[self.LEFT_IRIS[0]].x + lm[self.LEFT_IRIS[2]].x) / 2
                    gy = (lm[self.LEFT_IRIS[0]].y + lm[self.LEFT_IRIS[2]].y) / 2
                    all_gx.append(gx)
                    all_gy.append(gy)

        if len(all_gx) > 20:
            self.gaze_calib = [
                np.percentile(all_gx, 10) - 0.02,
                np.percentile(all_gx, 90) + 0.02,
                np.percentile(all_gy, 10) - 0.02,
                np.percentile(all_gy, 90) + 0.02,
            ]
            print(f"[CALIBRATION] Gaze mapping  X : {self.gaze_calib[:2]}")
            print(f"[CALIBRATION] Gaze mapping  Y : {self.gaze_calib[2:]}")
        else:
            print("[CALIBRATION] Not enough gaze data — using default mapping")

    # =========================================================================
    #  EAR
    # =========================================================================
    def calculate_EAR(self, pts):
        A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        return (A + B) / (2.0 * C + 1e-6)

    # =========================================================================
    #  GAZE DIRECTION LABEL
    # =========================================================================
    def get_gaze_direction(self, landmarks, w, h):
        left_corner  = landmarks[self.L_EYE_LEFT].x  * w
        right_corner = landmarks[self.L_EYE_RIGHT].x * w
        iris_x = (landmarks[self.LEFT_IRIS[0]].x * w +
                  landmarks[self.LEFT_IRIS[2]].x * w) / 2
        ratio = (iris_x - left_corner) / (right_corner - left_corner + 1e-6)
        if   ratio < 0.37: return "RIGHT ->"
        elif ratio > 0.63: return "<- LEFT"
        return "CENTER"

    # =========================================================================
    #  GAZE -> SCREEN COORDINATES
    # =========================================================================
    def gaze_to_screen(self, gx, gy):
        gx_min, gx_max, gy_min, gy_max = self.gaze_calib
        sx = (gx - gx_min) / (gx_max - gx_min + 1e-6)
        sy = (gy - gy_min) / (gy_max - gy_min + 1e-6)
        sx = max(0.0, min(1.0, sx))
        sy = max(0.0, min(1.0, sy))
        return int(sx * self.screen_w), int(sy * self.screen_h)

    # =========================================================================
    #  HEAD POSE
    # =========================================================================
    def get_head_pose(self, matrix):
        m = np.array(matrix).reshape(4, 4)
        R = m[:3, :3]
        yaw   = np.degrees(np.arctan2(R[1][0], R[0][0]))
        pitch = np.degrees(np.arctan2(-R[2][0],
                                      np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)))
        status = "FORWARD"
        if   abs(yaw) > self.YAW_THRESHOLD:   status = f"TURNED {'RIGHT' if yaw > 0 else 'LEFT'}"
        elif pitch >  self.PITCH_THRESHOLD:    status = "LOOKING DOWN"
        elif pitch < -self.PITCH_THRESHOLD:    status = "LOOKING UP"
        return yaw, pitch, status

    # =========================================================================
    #  LIVE ATTENTION SCORE
    # =========================================================================
    def update_live_attention(self, timestamp):
        cutoff = timestamp - 5.0
        while self.rolling_blinks and self.rolling_blinks[0] < cutoff:
            self.rolling_blinks.popleft()
        while self.rolling_gaze and self.rolling_gaze[0][0] < cutoff:
            self.rolling_gaze.popleft()

        if len(self.rolling_gaze) < 10:
            self.live_attention     = "CALIBRATING"
            self.live_attention_pct = 0
            return

        blink_rate     = len(self.rolling_blinks) * 12
        gaze_data      = np.array([[r[1], r[2]] for r in self.rolling_gaze])
        variance       = gaze_data[:, 0].var() + gaze_data[:, 1].var()
        focus_mask     = (gaze_data[:, 1] > 0.25) & (gaze_data[:, 1] < 0.75)
        focus_ratio    = focus_mask.sum() / len(gaze_data)

        blink_score    = 1.0 if 8 <= blink_rate <= 22 else max(0, 1 - abs(blink_rate - 15) / 15)
        variance_score = 1 / (1 + variance * 10)
        raw = focus_ratio * 0.45 + variance_score * 0.30 + blink_score * 0.25
        pct = min(100, max(0, int(raw * 100)))

        self.live_attention_pct = pct
        self.live_attention     = ("HIGH"   if pct >= 70 else
                                   "MEDIUM" if pct >= 40 else "LOW")
        self.attention_history.append(pct)

    # =========================================================================
    #  HUD HELPERS
    # =========================================================================
    @staticmethod
    def draw_rounded_rect(img, pt1, pt2, color, alpha=0.55, radius=12, thickness=-1):
        overlay = img.copy()
        x1, y1 = pt1;  x2, y2 = pt2
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                       (x1+radius, y2-radius), (x2-radius, y2-radius)]:
            cv2.circle(overlay, (cx, cy), radius, color, thickness)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    def draw_hud(self, frame, fps, timestamp, left_ear, right_ear):
        h, w = frame.shape[:2]
        PAD  = 14

        panel_w, panel_h = 285, 258
        self.draw_rounded_rect(frame, (10, 10),
                               (10 + panel_w, 10 + panel_h),
                               (15, 15, 15), alpha=0.65)

        attn_color = {
            "HIGH": (50, 220, 80), "MEDIUM": (40, 180, 255),
            "LOW": (30, 30, 220),  "CALIBRATING": (160, 160, 160),
        }.get(self.live_attention, (255, 255, 255))

        def put(text, y, scale=0.60, color=(220, 220, 220), bold=1):
            cv2.putText(frame, text, (PAD + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, bold, cv2.LINE_AA)

        elapsed = int(timestamp)
        put(f"Time      : {elapsed // 60:02d}:{elapsed % 60:02d}", 40)
        put(f"FPS       : {int(fps)}", 64)
        put(f"L-Blinks  : {self.left_blink_total}  (left click)", 88)
        put(f"R-Blinks  : {self.right_blink_total}  (right click)", 112)
        put(f"EAR  L:{left_ear:.3f}   R:{right_ear:.3f}", 136)
        put(f"Gaze      : {self.gaze_direction}", 160)
        put(f"Head      : {self.head_status}", 184)
        mouse_col = (50, 220, 80) if self.mouse_mode else (120, 120, 120)
        put(f"Mouse ctrl: {'ON' if self.mouse_mode else 'OFF'}   (press M)", 208, color=mouse_col)
        put(f"Attn      : {self.live_attention} ({self.live_attention_pct}%)",
            238, scale=0.68, color=attn_color, bold=2)

        # Progress bar
        bar_x, bar_y, bar_w, bar_h = 10, 262, panel_w, 13
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (50,50,50), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + int(bar_w * self.live_attention_pct / 100), bar_y+bar_h),
                      attn_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (100,100,100), 1)

        # Sparkline
        if len(self.attention_history) > 2:
            spark_x, spark_y, spark_w, spark_h = 10, 282, panel_w, 42
            self.draw_rounded_rect(frame, (spark_x, spark_y),
                                   (spark_x+spark_w, spark_y+spark_h),
                                   (15,15,15), alpha=0.5)
            hist = list(self.attention_history);  n = len(hist)
            for i in range(1, n):
                x1 = spark_x + int((i-1)/(n-1)*spark_w)
                x2 = spark_x + int(i/(n-1)*spark_w)
                y1 = spark_y + spark_h - int(hist[i-1]/100*spark_h)
                y2 = spark_y + spark_h - int(hist[i]/100*spark_h)
                cv2.line(frame, (x1,y1), (x2,y2), attn_color, 1, cv2.LINE_AA)

        # Drowsiness alert
        if self.drowsy_counter >= self.DROWSY_FRAMES:
            ax, ay = w//2 - 190, h//2 - 30
            self.draw_rounded_rect(frame, (ax-10, ay-42), (ax+400, ay+22), (0,0,180), alpha=0.80)
            cv2.putText(frame, "!! DROWSY - WAKE UP !!",
                        (ax, ay), cv2.FONT_HERSHEY_DUPLEX, 1.1, (30,80,255), 2, cv2.LINE_AA)
            if SOUND_AVAILABLE and time.time() - self.last_alert_time > 3:
                winsound.Beep(900, 400);  self.last_alert_time = time.time()

        # Head pose warning
        if self.head_status != "FORWARD":
            cv2.putText(frame, f"HEAD: {self.head_status}", (w-290, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30,165,255), 2, cv2.LINE_AA)

        # Click flash
        now = time.time()
        if now - self.last_left_click_t < 0.35:
            cv2.putText(frame, "<  LEFT CLICK", (w//2-130, h-55),
                        cv2.FONT_HERSHEY_DUPLEX, 0.95, (0,220,255), 2, cv2.LINE_AA)
        elif now - self.last_right_click_t < 0.35:
            cv2.putText(frame, "RIGHT CLICK  >", (w//2-130, h-55),
                        cv2.FONT_HERSHEY_DUPLEX, 0.95, (255,140,0), 2, cv2.LINE_AA)

        cv2.putText(frame, "ESC = end session  |  M = toggle mouse",
                    (w//2-200, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (110,110,110), 1, cv2.LINE_AA)

    # =========================================================================
    #  MAIN LOOP
    # =========================================================================
    def run(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                break

            frame     = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _   = frame.shape
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            ts_ms     = int(time.time() * 1000)
            result    = self.detector.detect_for_video(mp_image, ts_ms)
            timestamp = time.time() - self.session_start

            left_ear = right_ear = 0.0
            blink_flag = 0

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]

                # Iris circles
                lpts = [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in self.LEFT_IRIS]
                rpts = [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in self.RIGHT_IRIS]
                (lcx,lcy), l_r = cv2.minEnclosingCircle(np.array(lpts))
                (rcx,rcy), r_r = cv2.minEnclosingCircle(np.array(rpts))
                cv2.circle(frame, (int(lcx),int(lcy)), int(l_r), (0,230,100), 2)
                cv2.circle(frame, (int(rcx),int(rcy)), int(r_r), (0,230,100), 2)

                avg_x = (lcx + rcx) / 2;  avg_y = (lcy + rcy) / 2
                if self.prev_x is not None:
                    avg_x = self.alpha*avg_x + (1-self.alpha)*self.prev_x
                    avg_y = self.alpha*avg_y + (1-self.alpha)*self.prev_y
                self.prev_x, self.prev_y = avg_x, avg_y
                cv2.circle(frame, (int(avg_x),int(avg_y)), 5, (0,0,255), -1)

                # Per-eye EAR
                le = [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in self.LEFT_EYE]
                re = [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in self.RIGHT_EYE]
                left_ear  = self.calculate_EAR(le)
                right_ear = self.calculate_EAR(re)
                now = time.time()

                # Left eye blink -> left click
                if left_ear < self.EAR_THRESHOLD_L:
                    self.left_blink_counter += 1;  self.drowsy_counter += 1
                else:
                    if self.left_blink_counter >= self.CONSEC_FRAMES:
                        self.left_blink_total += 1;  blink_flag = 1
                        self.rolling_blinks.append(timestamp)
                        if (self.mouse_mode and MOUSE_AVAILABLE
                                and now - self.last_left_click_t > self.CLICK_COOLDOWN):
                            pyautogui.click(button='left')
                            self.last_left_click_t = now
                    self.left_blink_counter = 0

                # Right eye blink -> right click
                if right_ear < self.EAR_THRESHOLD_R:
                    self.right_blink_counter += 1;  self.drowsy_counter += 1
                else:
                    if self.right_blink_counter >= self.CONSEC_FRAMES:
                        self.right_blink_total += 1;  blink_flag = 1
                        self.rolling_blinks.append(timestamp)
                        if (self.mouse_mode and MOUSE_AVAILABLE
                                and now - self.last_right_click_t > self.CLICK_COOLDOWN):
                            pyautogui.click(button='right')
                            self.last_right_click_t = now
                    self.right_blink_counter = 0

                # Reset drowsy only when both eyes open
                if left_ear >= self.EAR_THRESHOLD_L and right_ear >= self.EAR_THRESHOLD_R:
                    self.drowsy_counter = 0

                self.gaze_direction = self.get_gaze_direction(landmarks, w, h)

                if result.facial_transformation_matrixes:
                    _, _, self.head_status = self.get_head_pose(
                        result.facial_transformation_matrixes[0])

                # Gaze -> mouse
                gx_raw = (landmarks[self.LEFT_IRIS[0]].x + landmarks[self.LEFT_IRIS[2]].x) / 2
                gy_raw = (landmarks[self.LEFT_IRIS[0]].y + landmarks[self.LEFT_IRIS[2]].y) / 2
                if self.mouse_mode and MOUSE_AVAILABLE:
                    sx, sy = self.gaze_to_screen(gx_raw, gy_raw)
                    self.mouse_x = int(self.mouse_alpha*sx + (1-self.mouse_alpha)*self.mouse_x)
                    self.mouse_y = int(self.mouse_alpha*sy + (1-self.mouse_alpha)*self.mouse_y)
                    pyautogui.moveTo(self.mouse_x, self.mouse_y)

                gx_norm = avg_x / w;  gy_norm = avg_y / h
                self.rolling_gaze.append((timestamp, gx_norm, gy_norm))
                self.update_live_attention(timestamp)

                self.log_rows.append([
                    round(timestamp,4), round(gx_norm,5), round(gy_norm,5),
                    blink_flag, self.gaze_direction, self.head_status,
                    round(left_ear,4), round(right_ear,4), self.live_attention_pct,
                ])

            curr_time = time.time()
            fps = 1 / (curr_time - self.prev_time) if self.prev_time else 0
            self.prev_time = curr_time

            self.draw_hud(frame, fps, timestamp, left_ear, right_ear)
            cv2.imshow("Attention Tracking System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key in (ord('m'), ord('M')):
                self.mouse_mode = not self.mouse_mode
                print(f"[INFO] Mouse control {'ENABLED' if self.mouse_mode else 'DISABLED'}")

        self.cap.release()
        cv2.destroyAllWindows()
        self._save_csv()
        self.session_summary()
        self.generate_dashboard()

    # =========================================================================
    #  SAVE CSV
    # =========================================================================
    def _save_csv(self):
        with open("gaze_log.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp","gaze_x","gaze_y","blink",
                             "gaze_dir","head_status","ear_left","ear_right","attention_pct"])
            writer.writerows(self.log_rows)
        print("[INFO] gaze_log.csv saved.")

    # =========================================================================
    #  SESSION SUMMARY
    # =========================================================================
    def session_summary(self):
        data = pd.read_csv("gaze_log.csv")
        if data.empty:
            print("No data recorded.");  return

        total_time   = data["timestamp"].iloc[-1]
        total_blinks = int(data["blink"].sum())
        blink_rate   = total_blinks / (total_time / 60) if total_time > 0 else 0
        gaze_var     = data["gaze_x"].var() + data["gaze_y"].var()
        avg_attn     = data["attention_pct"].mean()
        distract_pct = len(data[data["head_status"] != "FORWARD"]) / len(data) * 100

        print("\n╔══════════════════════════════════════╗")
        print("║         SESSION SUMMARY              ║")
        print("╠══════════════════════════════════════╣")
        print(f"║  Duration      : {int(total_time//60):02d}m {int(total_time%60):02d}s          ║")
        print(f"║  L-Eye Blinks  : {self.left_blink_total:<22}║")
        print(f"║  R-Eye Blinks  : {self.right_blink_total:<22}║")
        print(f"║  Total Blinks  : {total_blinks:<22}║")
        print(f"║  Blink Rate    : {blink_rate:.2f} blinks/min       ║")
        print(f"║  Gaze Variance : {gaze_var:.4f}               ║")
        print(f"║  Avg Attention : {avg_attn:.1f}%                  ║")
        print(f"║  Distracted    : {distract_pct:.1f}% of session       ║")
        print("╚══════════════════════════════════════╝\n")

    # =========================================================================
    #  DASHBOARD
    # =========================================================================
    def generate_dashboard(self):
        data = pd.read_csv("gaze_log.csv")
        if len(data) < 5:
            print("Not enough data to generate dashboard.");  return

        plt.style.use("dark_background")
        fig = plt.figure(figsize=(20, 10), facecolor="#0d0d0d")
        fig.suptitle("Attention Tracker v2 — Session Report",
                     fontsize=18, color="#e0e0e0", y=0.97)
        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35,
                               left=0.05, right=0.97, top=0.91, bottom=0.08)

        def styled_ax(ax, title):
            ax.set_facecolor("#111111");  ax.set_title(title, color="#aaaaaa", fontsize=11)
            for sp in ax.spines.values(): sp.set_edgecolor("#333333")
            ax.tick_params(colors="#777777", labelsize=8)

        ax1 = fig.add_subplot(gs[0,0]);  styled_ax(ax1, "Gaze Heatmap")
        sns.kdeplot(x=data["gaze_x"], y=data["gaze_y"], fill=True, cmap="inferno",
                    bw_adjust=0.6, ax=ax1)
        ax1.set_xlim(0,1);  ax1.set_ylim(0,1);  ax1.invert_yaxis()
        ax1.set_xlabel("X (norm)", color="#888", fontsize=9)
        ax1.set_ylabel("Y (norm)", color="#888", fontsize=9)

        ax2 = fig.add_subplot(gs[0,1]);  styled_ax(ax2, "Blink Timeline")
        blinks = data[data["blink"]==1]
        ax2.vlines(blinks["timestamp"], 0, 1, colors="#00e5ff", linewidth=1.2, alpha=0.85)
        ax2.set_xlabel("Time (s)", color="#888", fontsize=9);  ax2.set_yticks([])
        ax2.set_xlim(data["timestamp"].min(), data["timestamp"].max())

        ax3 = fig.add_subplot(gs[0,2]);  styled_ax(ax3, "Attention % Over Time")
        ax3.plot(data["timestamp"], data["attention_pct"], color="#39ff14", linewidth=1.2)
        ax3.axhline(70, color="#ffcc00", linewidth=0.8, linestyle="--", label="High (70%)")
        ax3.axhline(40, color="#ff4500", linewidth=0.8, linestyle="--", label="Low (40%)")
        ax3.set_ylim(0,105);  ax3.legend(fontsize=7, loc="lower right")
        ax3.set_xlabel("Time (s)", color="#888", fontsize=9)
        ax3.set_ylabel("Score (%)", color="#888", fontsize=9)

        ax4 = fig.add_subplot(gs[0,3]);  styled_ax(ax4, "Gaze Direction Distribution")
        dc = data["gaze_dir"].value_counts()
        ax4.pie(dc.values, labels=dc.index,
                colors=["#00e5ff","#ff6ec7","#39ff14","#ffa500"][:len(dc)],
                autopct="%1.1f%%", textprops={"color":"#cccccc","fontsize":9}, startangle=90)

        ax5 = fig.add_subplot(gs[1,0]);  styled_ax(ax5, "Left Eye EAR")
        ax5.plot(data["timestamp"], data["ear_left"], color="#00e5ff", linewidth=0.9)
        ax5.axhline(self.EAR_THRESHOLD_L, color="#ff4444", linewidth=1.0, linestyle="--",
                    label=f"Threshold ({self.EAR_THRESHOLD_L})")
        ax5.set_xlabel("Time (s)", color="#888", fontsize=9)
        ax5.set_ylabel("EAR", color="#888", fontsize=9);  ax5.legend(fontsize=7)

        ax6 = fig.add_subplot(gs[1,1]);  styled_ax(ax6, "Right Eye EAR")
        ax6.plot(data["timestamp"], data["ear_right"], color="#ffa500", linewidth=0.9)
        ax6.axhline(self.EAR_THRESHOLD_R, color="#ff4444", linewidth=1.0, linestyle="--",
                    label=f"Threshold ({self.EAR_THRESHOLD_R})")
        ax6.set_xlabel("Time (s)", color="#888", fontsize=9)
        ax6.set_ylabel("EAR", color="#888", fontsize=9);  ax6.legend(fontsize=7)

        ax7 = fig.add_subplot(gs[1,2]);  styled_ax(ax7, "Gaze X Drift")
        ax7.plot(data["timestamp"], data["gaze_x"], color="#ff6ec7", linewidth=0.8)
        ax7.axhline(0.5, color="#ffffff", linewidth=0.6, linestyle=":", label="Centre")
        ax7.set_xlabel("Time (s)", color="#888", fontsize=9)
        ax7.set_ylabel("Gaze X", color="#888", fontsize=9);  ax7.legend(fontsize=7)

        ax8 = fig.add_subplot(gs[1,3]);  styled_ax(ax8, "Head Status Distribution")
        hc = data["head_status"].value_counts()
        ax8.bar(hc.index, hc.values,
                color=["#39ff14" if s=="FORWARD" else "#ff4500" for s in hc.index])
        ax8.set_xlabel("Status", color="#888", fontsize=9)
        ax8.set_ylabel("Frames", color="#888", fontsize=9)
        ax8.tick_params(axis='x', rotation=15)

        plt.savefig("session_report.png", dpi=140, bbox_inches="tight", facecolor="#0d0d0d")
        print("[INFO] session_report.png saved.")
        plt.show()


# =============================================================================
#  ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    tracker = AttentionTracker()
    tracker.calibrate(duration=3)
    tracker.run()