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

# Optional: sound alert on Windows
try:
    import winsound
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
#  AttentionTracker
# ─────────────────────────────────────────────────────────────────────────────
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
            output_facial_transformation_matrixes=True,   # head pose
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # ── CAMERA ─────────────────────────────────────────────────────────── #
        self.cap = cv2.VideoCapture(0)
        #self.cap = cv2.VideoCapture(1)# this one is for web cam depending on the situation
        if not self.cap.isOpened():
            raise Exception("Webcam not detected")

        self.fps_cam = self.cap.get(cv2.CAP_PROP_FPS) or 30

        # ── LANDMARK INDICES ───────────────────────────────────────────────── #
        self.LEFT_IRIS  = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        self.LEFT_EYE   = [33,  160, 158, 133, 153, 144]
        self.RIGHT_EYE  = [362, 385, 387, 263, 373, 380]
        # Eye corners for gaze direction
        self.L_EYE_LEFT  = 33
        self.L_EYE_RIGHT = 133
        self.R_EYE_LEFT  = 362
        self.R_EYE_RIGHT = 263

        # ── BLINK ──────────────────────────────────────────────────────────── #
        self.EAR_THRESHOLD  = 0.23          # updated after calibration
        self.CONSEC_FRAMES  = 3
        self.blink_counter  = 0
        self.blink_total    = 0

        # ── DROWSINESS ────────────────────────────────────────────────────── #
        self.DROWSY_FRAMES   = int(self.fps_cam * 2)   # ~2 seconds
        self.drowsy_counter  = 0
        self.last_alert_time = 0

        # ── SMOOTHING ─────────────────────────────────────────────────────── #
        self.prev_x = None
        self.prev_y = None
        self.alpha  = 0.7                  # EMA weight

        # ── HEAD POSE ─────────────────────────────────────────────────────── #
        self.YAW_THRESHOLD   = 25          # degrees
        self.PITCH_THRESHOLD = 20

        # ── SESSION ────────────────────────────────────────────────────────── #
        self.session_start = time.time()
        self.prev_time     = 0

        # rolling 5-second window for live attention score
        self.rolling_blinks = deque()   # (timestamp, 1)
        self.rolling_gaze   = deque()   # (timestamp, gaze_x, gaze_y)

        self.log_rows = []              # in-memory log (written to CSV at end)

        # ── DISPLAY HELPERS ────────────────────────────────────────────────── #
        self.live_attention    = "CALIBRATING"
        self.live_attention_pct = 0
        self.gaze_direction    = "CENTER"
        self.head_status       = "FORWARD"
        self.attention_history = deque(maxlen=150)   # for mini sparkline

        print("\n╔══════════════════════════════════════╗")
        print("║   Attention Tracker — Starting Up    ║")
        print("╚══════════════════════════════════════╝\n")

    # ─────────────────────────────────────────────────────────────────────────
    #  CALIBRATION
    # ─────────────────────────────────────────────────────────────────────────
    def calibrate(self, duration=3):
        """
        Collect EAR values while user keeps eyes open to set a personalized
        threshold.  Useful for glasses wearers.
        """
        print(f"[CALIBRATION] Keep your eyes open naturally for {duration}s...")
        ears = []
        deadline = time.time() + duration

        while time.time() < deadline:
            success, frame = self.cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            countdown = int(deadline - time.time()) + 1
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

            cv2.putText(frame, "CALIBRATION", (w // 2 - 140, h // 2 - 60),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 220, 255), 2)
            cv2.putText(frame, "Keep eyes open normally",
                        (w // 2 - 165, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            cv2.putText(frame, f"Starting in {countdown}...",
                        (w // 2 - 110, h // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 255, 100), 2)

            cv2.imshow("Attention Tracking System", frame)
            cv2.waitKey(1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(time.time() * 1000)
            result = self.detector.detect_for_video(mp_img, ts)

            if result.face_landmarks:
                lm = result.face_landmarks[0]
                le = [(int(lm[i].x * w), int(lm[i].y * h)) for i in self.LEFT_EYE]
                re = [(int(lm[i].x * w), int(lm[i].y * h)) for i in self.RIGHT_EYE]
                ear = (self.calculate_EAR(le) + self.calculate_EAR(re)) / 2.0
                ears.append(ear)

        if ears:
            mean_ear = np.mean(ears)
            self.EAR_THRESHOLD = round(mean_ear - 0.05, 4)
            print(f"[CALIBRATION] Mean EAR={mean_ear:.4f} → threshold={self.EAR_THRESHOLD}")
        else:
            print("[CALIBRATION] No face detected — using default threshold 0.23")

    # ─────────────────────────────────────────────────────────────────────────
    #  EAR
    # ─────────────────────────────────────────────────────────────────────────
    def calculate_EAR(self, pts):
        A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        return (A + B) / (2.0 * C + 1e-6)

    # ─────────────────────────────────────────────────────────────────────────
    #  GAZE DIRECTION
    # ─────────────────────────────────────────────────────────────────────────
    def get_gaze_direction(self, landmarks, w, h):
        # Use left eye only (more stable with glasses)
        left_corner  = landmarks[self.L_EYE_LEFT].x  * w
        right_corner = landmarks[self.L_EYE_RIGHT].x * w
        iris_x = (landmarks[self.LEFT_IRIS[0]].x * w +
                  landmarks[self.LEFT_IRIS[2]].x * w) / 2

        eye_width = right_corner - left_corner + 1e-6
        ratio = (iris_x - left_corner) / eye_width

        if ratio < 0.37:
            return "RIGHT →"
        elif ratio > 0.63:
            return "← LEFT"
        return "CENTER"

    # ─────────────────────────────────────────────────────────────────────────
    #  HEAD POSE  (yaw & pitch from transformation matrix)
    # ─────────────────────────────────────────────────────────────────────────
    def get_head_pose(self, matrix):
        """
        Returns (yaw_deg, pitch_deg, status_string).
        Positive yaw  = face turned right.
        Positive pitch = face tilted down.
        """
        m = np.array(matrix).reshape(4, 4)
        R = m[:3, :3]

        # yaw (Y-axis rotation)
        yaw   = np.degrees(np.arctan2(R[1][0], R[0][0]))
        # pitch (X-axis rotation)
        pitch = np.degrees(np.arctan2(-R[2][0],
                                      np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)))

        status = "FORWARD"
        if abs(yaw) > self.YAW_THRESHOLD:
            status = f"TURNED {'RIGHT' if yaw > 0 else 'LEFT'}"
        elif pitch > self.PITCH_THRESHOLD:
            status = "LOOKING DOWN"
        elif pitch < -self.PITCH_THRESHOLD:
            status = "LOOKING UP"

        return yaw, pitch, status

    # ─────────────────────────────────────────────────────────────────────────
    #  LIVE ATTENTION SCORE  (rolling 5-second window)
    # ─────────────────────────────────────────────────────────────────────────
    def update_live_attention(self, timestamp):
        cutoff = timestamp - 5.0

        # Purge old entries
        while self.rolling_blinks and self.rolling_blinks[0] < cutoff:
            self.rolling_blinks.popleft()
        while self.rolling_gaze and self.rolling_gaze[0][0] < cutoff:
            self.rolling_gaze.popleft()

        if len(self.rolling_gaze) < 10:
            self.live_attention = "CALIBRATING"
            self.live_attention_pct = 0
            return

        blink_rate = len(self.rolling_blinks) * 12   # per minute (5s → ×12)
        gaze_data  = np.array([[r[1], r[2]] for r in self.rolling_gaze])
        variance   = gaze_data[:, 0].var() + gaze_data[:, 1].var()

        focus_mask = (gaze_data[:, 1] > 0.25) & (gaze_data[:, 1] < 0.75)
        focus_ratio = focus_mask.sum() / len(gaze_data)

        blink_score   = 1.0 if 8 <= blink_rate <= 22 else max(0, 1 - abs(blink_rate - 15) / 15)
        focus_score   = focus_ratio
        variance_score = 1 / (1 + variance * 10)

        raw = (focus_score * 0.45 + variance_score * 0.30 + blink_score * 0.25)
        pct = min(100, max(0, int(raw * 100)))

        self.live_attention_pct = pct
        if pct >= 70:
            self.live_attention = "HIGH"
        elif pct >= 40:
            self.live_attention = "MEDIUM"
        else:
            self.live_attention = "LOW"

        self.attention_history.append(pct)

    # ─────────────────────────────────────────────────────────────────────────
    #  HUD DRAWING HELPERS
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def draw_rounded_rect(img, pt1, pt2, color, alpha=0.55, radius=12, thickness=-1):
        overlay = img.copy()
        x1, y1 = pt1
        x2, y2 = pt2
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, thickness)
        cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, thickness)
        cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, thickness)
        cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, thickness)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    def draw_hud(self, frame, fps, timestamp, ear):
        h, w = frame.shape[:2]
        PAD = 14

        # ── TOP-LEFT PANEL (stats) ────────────────────────────────────────── #
        panel_w, panel_h = 240, 200
        self.draw_rounded_rect(frame, (10, 10), (10 + panel_w, 10 + panel_h),
                               (15, 15, 15), alpha=0.6)

        attn_color = {
            "HIGH":        (50,  220,  80),
            "MEDIUM":      (40,  180, 255),
            "LOW":         (30,   30, 220),
            "CALIBRATING": (160, 160, 160),
        }.get(self.live_attention, (255, 255, 255))

        def put(text, y, scale=0.65, color=(220, 220, 220), bold=1):
            cv2.putText(frame, text, (PAD + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, bold, cv2.LINE_AA)

        put(f"Blinks : {self.blink_total}", 42)
        put(f"EAR    : {ear:.3f}", 68)
        put(f"FPS    : {int(fps)}", 94)
        elapsed = int(timestamp)
        put(f"Time   : {elapsed // 60:02d}:{elapsed % 60:02d}", 120)
        put(f"Gaze   : {self.gaze_direction}", 146)
        put(f"Head   : {self.head_status}", 172)
        put(f"Attn   : {self.live_attention} ({self.live_attention_pct}%)",
            198, scale=0.68, color=attn_color, bold=2)

        # ── ATTENTION PROGRESS BAR ────────────────────────────────────────── #
        bar_x, bar_y, bar_w, bar_h = 10, 218, panel_w, 14
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (50, 50, 50), -1)
        filled = int(bar_w * self.live_attention_pct / 100)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                      attn_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (100, 100, 100), 1)

        # ── SPARKLINE (attention history) ─────────────────────────────────── #
        if len(self.attention_history) > 2:
            spark_x, spark_y = 10, 240
            spark_w, spark_h = panel_w, 40
            self.draw_rounded_rect(frame, (spark_x, spark_y),
                                   (spark_x + spark_w, spark_y + spark_h),
                                   (15, 15, 15), alpha=0.5)
            hist = list(self.attention_history)
            n = len(hist)
            for i in range(1, n):
                x1 = spark_x + int((i - 1) / (n - 1) * spark_w)
                x2 = spark_x + int(i / (n - 1) * spark_w)
                y1 = spark_y + spark_h - int(hist[i - 1] / 100 * spark_h)
                y2 = spark_y + spark_h - int(hist[i] / 100 * spark_h)
                cv2.line(frame, (x1, y1), (x2, y2), attn_color, 1, cv2.LINE_AA)

        # ── DROWSINESS ALERT ──────────────────────────────────────────────── #
        if self.drowsy_counter >= self.DROWSY_FRAMES:
            alert_text = "⚠  DROWSY — WAKE UP!"
            tw, _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)[0], 0
            ax = w // 2 - 180
            ay = h // 2 - 30
            self.draw_rounded_rect(frame, (ax - 10, ay - 40),
                                   (ax + 380, ay + 20),
                                   (0, 0, 180), alpha=0.75)
            cv2.putText(frame, alert_text, (ax, ay),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 80, 255), 2, cv2.LINE_AA)

            # Audible alert (Windows only, once every 3s)
            if SOUND_AVAILABLE and time.time() - self.last_alert_time > 3:
                winsound.Beep(900, 400)
                self.last_alert_time = time.time()

        # ── HEAD POSE WARNING ─────────────────────────────────────────────── #
        if self.head_status != "FORWARD":
            cv2.putText(frame, f"HEAD: {self.head_status}",
                        (w - 280, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (30, 165, 255), 2, cv2.LINE_AA)

        # ── PRESS ESC ─────────────────────────────────────────────────────── #
        cv2.putText(frame, "ESC to end session",
                    (w - 200, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)

    # ─────────────────────────────────────────────────────────────────────────
    #  MAIN LOOP
    # ─────────────────────────────────────────────────────────────────────────
    def run(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape

            mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            ts_ms      = int(time.time() * 1000)
            result     = self.detector.detect_for_video(mp_image, ts_ms)
            timestamp  = time.time() - self.session_start

            ear = 0.0

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]

                # ── IRIS ──────────────────────────────────────────────────── #
                left_pts  = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
                             for i in self.LEFT_IRIS]
                right_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
                             for i in self.RIGHT_IRIS]

                (lcx, lcy), l_r = cv2.minEnclosingCircle(np.array(left_pts))
                (rcx, rcy), r_r = cv2.minEnclosingCircle(np.array(right_pts))

                # Draw iris circles
                cv2.circle(frame, (int(lcx), int(lcy)), int(l_r), (0, 230, 100), 2)
                cv2.circle(frame, (int(rcx), int(rcy)), int(r_r), (0, 230, 100), 2)

                avg_x = (lcx + rcx) / 2
                avg_y = (lcy + rcy) / 2

                # Smooth
                if self.prev_x is not None:
                    avg_x = self.alpha * avg_x + (1 - self.alpha) * self.prev_x
                    avg_y = self.alpha * avg_y + (1 - self.alpha) * self.prev_y
                self.prev_x, self.prev_y = avg_x, avg_y

                cv2.circle(frame, (int(avg_x), int(avg_y)), 5, (0, 0, 255), -1)

                # ── BLINK ─────────────────────────────────────────────────── #
                le  = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
                       for i in self.LEFT_EYE]
                re  = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
                       for i in self.RIGHT_EYE]
                ear = (self.calculate_EAR(le) + self.calculate_EAR(re)) / 2.0

                blink_flag = 0
                if ear < self.EAR_THRESHOLD:
                    self.blink_counter  += 1
                    self.drowsy_counter += 1
                else:
                    if self.blink_counter >= self.CONSEC_FRAMES:
                        self.blink_total += 1
                        blink_flag = 1
                        self.rolling_blinks.append(timestamp)
                    self.blink_counter  = 0
                    self.drowsy_counter = 0

                # ── GAZE DIRECTION ────────────────────────────────────────── #
                self.gaze_direction = self.get_gaze_direction(landmarks, w, h)

                # ── HEAD POSE ─────────────────────────────────────────────── #
                if result.facial_transformation_matrixes:
                    yaw, pitch, self.head_status = self.get_head_pose(
                        result.facial_transformation_matrixes[0]
                    )

                # ── ROLLING GAZE LOG ──────────────────────────────────────── #
                gx, gy = avg_x / w, avg_y / h
                self.rolling_gaze.append((timestamp, gx, gy))

                # ── LIVE ATTENTION ────────────────────────────────────────── #
                self.update_live_attention(timestamp)

                # ── CSV LOG ───────────────────────────────────────────────── #
                self.log_rows.append([
                    round(timestamp, 4),
                    round(gx, 5),
                    round(gy, 5),
                    blink_flag,
                    self.gaze_direction,
                    self.head_status,
                    round(ear, 4),
                    self.live_attention_pct,
                ])

            # ── FPS ───────────────────────────────────────────────────────── #
            curr_time      = time.time()
            fps            = 1 / (curr_time - self.prev_time) if self.prev_time else 0
            self.prev_time = curr_time

            # ── HUD ───────────────────────────────────────────────────────── #
            self.draw_hud(frame, fps, timestamp, ear)

            cv2.imshow("Attention Tracking System", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        # ── CLEANUP ───────────────────────────────────────────────────────── #
        self.cap.release()
        cv2.destroyAllWindows()
        self._save_csv()
        self.session_summary()
        self.generate_dashboard()

    # ─────────────────────────────────────────────────────────────────────────
    #  SAVE CSV
    # ─────────────────────────────────────────────────────────────────────────
    def _save_csv(self):
        with open("gaze_log.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "gaze_x", "gaze_y", "blink",
                        "gaze_dir", "head_status", "ear", "attention_pct"])
            w.writerows(self.log_rows)
        print("[INFO] gaze_log.csv saved.")

    # ─────────────────────────────────────────────────────────────────────────
    #  SESSION SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    def session_summary(self):
        data = pd.read_csv("gaze_log.csv")
        if data.empty:
            print("No data recorded.")
            return

        total_time   = data["timestamp"].iloc[-1]
        total_blinks = int(data["blink"].sum())
        blink_rate   = total_blinks / (total_time / 60) if total_time > 0 else 0
        gaze_var     = data["gaze_x"].var() + data["gaze_y"].var()
        avg_attn     = data["attention_pct"].mean()

        distracted = data[data["head_status"] != "FORWARD"]
        distract_pct = len(distracted) / len(data) * 100

        print("\n╔══════════════════════════════════════╗")
        print("║         SESSION SUMMARY              ║")
        print("╠══════════════════════════════════════╣")
        print(f"║  Duration      : {int(total_time // 60):02d}m {int(total_time % 60):02d}s          ║")
        print(f"║  Total Blinks  : {total_blinks:<22}║")
        print(f"║  Blink Rate    : {blink_rate:.2f} blinks/min       ║")
        print(f"║  Gaze Variance : {gaze_var:.4f}               ║")
        print(f"║  Avg Attention : {avg_attn:.1f}%                  ║")
        print(f"║  Distracted    : {distract_pct:.1f}% of session       ║")
        print("╚══════════════════════════════════════╝\n")

    # ─────────────────────────────────────────────────────────────────────────
    #  DASHBOARD
    # ─────────────────────────────────────────────────────────────────────────
    def generate_dashboard(self):
        data = pd.read_csv("gaze_log.csv")
        if len(data) < 5:
            print("Not enough data to generate dashboard.")
            return

        plt.style.use("dark_background")
        fig = plt.figure(figsize=(16, 10), facecolor="#0d0d0d")
        fig.suptitle("Attention Tracking — Session Report",
                     fontsize=18, color="#e0e0e0", y=0.97,
                     fontfamily="DejaVu Sans")

        gs = gridspec.GridSpec(2, 3, figure=fig,
                               hspace=0.45, wspace=0.35,
                               left=0.06, right=0.97,
                               top=0.91, bottom=0.08)

        # ── 1. GAZE HEATMAP ───────────────────────────────────────────────── #
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor("#111111")
        ax1.set_title("Gaze Heatmap", color="#aaaaaa", fontsize=11)
        sns.kdeplot(x=data["gaze_x"], y=data["gaze_y"],
                    fill=True, cmap="inferno", bw_adjust=0.6, ax=ax1)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.invert_yaxis()
        ax1.set_xlabel("X (normalized)", color="#888888", fontsize=9)
        ax1.set_ylabel("Y (normalized)", color="#888888", fontsize=9)

        # ── 2. BLINK TIMELINE ─────────────────────────────────────────────── #
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor("#111111")
        ax2.set_title("Blink Timeline", color="#aaaaaa", fontsize=11)
        blinks = data[data["blink"] == 1]
        ax2.vlines(blinks["timestamp"], 0, 1,
                   colors="#00e5ff", linewidth=1.2, alpha=0.85)
        ax2.set_xlabel("Time (s)", color="#888888", fontsize=9)
        ax2.set_yticks([])
        ax2.set_xlim(data["timestamp"].min(), data["timestamp"].max())

        # ── 3. ATTENTION SCORE OVER TIME ──────────────────────────────────── #
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_facecolor("#111111")
        ax3.set_title("Attention % Over Time", color="#aaaaaa", fontsize=11)
        ax3.plot(data["timestamp"], data["attention_pct"],
                 color="#39ff14", linewidth=1.2, alpha=0.8)
        ax3.axhline(70, color="#ffcc00", linewidth=0.8,
                    linestyle="--", alpha=0.6, label="High (70%)")
        ax3.axhline(40, color="#ff4500", linewidth=0.8,
                    linestyle="--", alpha=0.6, label="Low (40%)")
        ax3.set_ylim(0, 105)
        ax3.set_xlabel("Time (s)", color="#888888", fontsize=9)
        ax3.set_ylabel("Score (%)", color="#888888", fontsize=9)
        ax3.legend(fontsize=7, loc="lower right")

        # ── 4. GAZE X DRIFT ───────────────────────────────────────────────── #
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.set_facecolor("#111111")
        ax4.set_title("Gaze X Drift", color="#aaaaaa", fontsize=11)
        ax4.plot(data["timestamp"], data["gaze_x"],
                 color="#ff6ec7", linewidth=0.8, alpha=0.8)
        ax4.axhline(0.5, color="#ffffff", linewidth=0.6,
                    linestyle=":", alpha=0.4)
        ax4.set_xlabel("Time (s)", color="#888888", fontsize=9)
        ax4.set_ylabel("Gaze X", color="#888888", fontsize=9)

        # ── 5. EAR OVER TIME ──────────────────────────────────────────────── #
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.set_facecolor("#111111")
        ax5.set_title("Eye Aspect Ratio (EAR)", color="#aaaaaa", fontsize=11)
        ax5.plot(data["timestamp"], data["ear"],
                 color="#ffa500", linewidth=0.9, alpha=0.8)
        ax5.axhline(self.EAR_THRESHOLD, color="#ff4444", linewidth=1.0,
                    linestyle="--", alpha=0.7, label=f"Threshold ({self.EAR_THRESHOLD})")
        ax5.set_xlabel("Time (s)", color="#888888", fontsize=9)
        ax5.set_ylabel("EAR", color="#888888", fontsize=9)
        ax5.legend(fontsize=7)

        # ── 6. GAZE DIRECTION PIE ─────────────────────────────────────────── #
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.set_facecolor("#111111")
        ax6.set_title("Gaze Direction Distribution", color="#aaaaaa", fontsize=11)
        dir_counts = data["gaze_dir"].value_counts()
        colors_pie = ["#00e5ff", "#ff6ec7", "#39ff14", "#ffa500"]
        ax6.pie(dir_counts.values,
                labels=dir_counts.index,
                colors=colors_pie[:len(dir_counts)],
                autopct="%1.1f%%",
                textprops={"color": "#cccccc", "fontsize": 9},
                startangle=90)

        plt.savefig("session_report.png", dpi=140, bbox_inches="tight",
                    facecolor="#0d0d0d")
        print("[INFO] session_report.png saved.")
        plt.show()
print("hello")

# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tracker = AttentionTracker()
    tracker.calibrate(duration=3)   # personalized EAR threshold
    tracker.run()
