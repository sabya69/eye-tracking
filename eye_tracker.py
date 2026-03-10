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
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque
import threading
import queue

# Mouse control
try:
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE    = 0
    MOUSE_AVAILABLE = True
except ImportError:
    MOUSE_AVAILABLE = False
    print("[WARN] pyautogui not found.")

# Optional: UI snapping (Windows only)
try:
    import uiautomation as auto
    auto.SetGlobalSearchTimeout(0)
    UI_SNAP_AVAILABLE = True
except ImportError:
    UI_SNAP_AVAILABLE = False
    print("[WARN] uiautomation not found. UI snapping disabled. Run: pip install uiautomation")

# Optional: sound alert on Windows
try:
    import winsound
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False


# =============================================================================
#  SMART UI SNAPPER  (background thread, ported from EyeOS)
# =============================================================================
class SmartUISnapper:
    """
    Runs a background thread that continuously checks which UI element the
    gaze is near and stores its bounding rect for magnetic cursor snapping.
    Anti-jitter sticky margin prevents the box jumping between adjacent icons.
    """

    STICKY_MARGIN = 80   # px — gaze must move this far outside the box to re-scan

    def __init__(self, screen_w, screen_h):
        self.screen_w      = screen_w
        self.screen_h      = screen_h
        self.gaze_x        = screen_w / 2
        self.gaze_y        = screen_h / 2
        self.active_rect   = None          # (x, y, w, h) or None
        self._lock         = threading.Lock()

        if UI_SNAP_AVAILABLE:
            self._thread = threading.Thread(target=self._scan_loop, daemon=True)
            self._thread.start()

    def update_gaze(self, gx, gy):
        self.gaze_x, self.gaze_y = gx, gy

    def get_rect(self):
        with self._lock:
            return self.active_rect

    def _check_point(self, px, py):
        try:
            ctrl = auto.ControlFromPoint(px, py)
            r    = ctrl.BoundingRectangle
            w, h = r.right - r.left, r.bottom - r.top
            if 5 < w < self.screen_w * 0.5 and 5 < h < self.screen_h * 0.5:
                return (r.left, r.top, w, h)
        except Exception:
            pass
        return None

    def _scan_loop(self):
        while True:
            time.sleep(0.05)   # ~20 Hz scan
            gx, gy = int(self.gaze_x), int(self.gaze_y)

            # --- STICKY MARGIN: hold onto the current box if gaze hasn't moved far ---
            current = self.active_rect
            if current:
                rx, ry, rw, rh = current
                m = self.STICKY_MARGIN
                if (rx-m) < gx < (rx+rw+m) and (ry-m) < gy < (ry+rh+m):
                    continue  # still looking at same element, skip re-scan

            # Try exact gaze point, then spiral outward
            rect = self._check_point(gx, gy)
            if not rect:
                for offset in (25, 50):
                    for px, py in [(gx, gy-offset),(gx-offset, gy),(gx+offset, gy),(gx, gy+offset)]:
                        rect = self._check_point(px, py)
                        if rect:
                            break
                    if rect:
                        break

            if rect:
                with self._lock:
                    self.active_rect = rect
            # If gaze lands on empty space, keep the last box (reduces flicker)


# =============================================================================
#  FACE IDENTITY STORE  (ported from EyeOS)
# =============================================================================
class FaceIdentityStore:
    """
    Saves and loads per-face calibration profiles to a CSV.
    Each row: sig0, sig1, sig2, gaze_x_min, gaze_x_max, gaze_y_min, gaze_y_max, ear_l, ear_r
    """

    MATCH_TOLERANCE = 0.08
    CSV_PATH        = "calibration_profiles.csv"

    @staticmethod
    def compute_signature(landmarks):
        def dist(i, j):
            return math.hypot(landmarks[i].x - landmarks[j].x,
                              landmarks[i].y - landmarks[j].y)
        d_eyes = dist(33, 263)
        if d_eyes == 0:
            return None
        return [
            dist(168,   1) / d_eyes,   # nose length / eye dist
            dist(168, 152) / d_eyes,   # face height / eye dist
            dist(234, 454) / d_eyes,   # face width  / eye dist
        ]

    def find_match(self, avg_sig):
        if not os.path.exists(self.CSV_PATH):
            return None
        try:
            with open(self.CSV_PATH, newline='') as f:
                for row in csv.reader(f):
                    if len(row) == 9:
                        s0,s1,s2, xmn,xmx,ymn,ymx, el,er = map(float, row)
                        d = math.sqrt((avg_sig[0]-s0)**2+(avg_sig[1]-s1)**2+(avg_sig[2]-s2)**2)
                        if d < self.MATCH_TOLERANCE:
                            return {
                                "gaze_calib": [xmn,xmx,ymn,ymx],
                                "ear_l": el, "ear_r": er,
                            }
        except Exception as e:
            print(f"[WARN] Profile read error: {e}")
        return None

    def save(self, avg_sig, gaze_calib, ear_l, ear_r):
        try:
            with open(self.CSV_PATH, 'a', newline='') as f:
                csv.writer(f).writerow([
                    *avg_sig, *gaze_calib, round(ear_l,4), round(ear_r,4)
                ])
            print("[INFO] Calibration profile saved.")
        except Exception as e:
            print(f"[WARN] Profile save error: {e}")


# =============================================================================
#  MULTI-BLINK GESTURE DETECTOR  (ported from EyeOS)
# =============================================================================
class MultiBlinkDetector:
    """
    Detects double / triple / quad both-eyes blink sequences.
    Also detects LEFT_WINK and RIGHT_WINK (single-eye hold).

    Returns gesture strings via  .update()  or None.
    """

    BOTH_HOLD_TIME  = 0.6    # hold both eyes closed this long → BOTH gesture
    WINK_HOLD_TIME  = 0.10   # hold one eye closed this long → wink gesture
    SEQUENCE_WINDOW = 0.50   # time window to collect multi-blink sequence

    def __init__(self):
        self._both_count      = 0
        self._last_both_time  = 0
        self._was_both_closed = False
        self._action_type     = None
        self._action_start    = 0

    def update(self, left_closed, right_closed):
        """
        left_closed / right_closed: bool — True if that eye is below EAR threshold.
        Returns a gesture string or None.
        """
        is_both = left_closed and right_closed
        gesture = None

        # Track both-eye blink sequences
        if is_both:
            if not self._was_both_closed:
                self._was_both_closed = True
                self._both_count     += 1
                self._last_both_time  = time.time()
        else:
            self._was_both_closed = False
            # Check if we've timed out the sequence window → fire multi-blink
            if self._both_count > 0 and (time.time() - self._last_both_time) > self.SEQUENCE_WINDOW:
                if   self._both_count == 2: gesture = "DOUBLE_BLINK"
                elif self._both_count == 3: gesture = "TRIPLE_BLINK"
                elif self._both_count >= 4: gesture = "QUAD_BLINK"
                self._both_count = 0
                return gesture

        # Determine current single action type
        new_type = None
        if is_both:
            new_type = "BOTH"
        elif right_closed:   # mediapipe flips: right_closed = physical LEFT wink
            new_type = "LEFT_WINK"
        elif left_closed:
            new_type = "RIGHT_WINK"

        if new_type != self._action_type:
            self._action_type  = new_type
            self._action_start = time.time()

        # Fire held gestures
        if self._action_type:
            hold = time.time() - self._action_start
            if self._action_type == "BOTH" and hold > self.BOTH_HOLD_TIME:
                self._both_count = 0
                gesture = "BOTH"
            elif self._action_type in ("LEFT_WINK","RIGHT_WINK") and hold >= self.WINK_HOLD_TIME:
                gesture = self._action_type

        return gesture


# =============================================================================
#  SMOOTH CURSOR ENGINE
# =============================================================================
class SmoothCursor:
    """
    Combines:
    - Large deadzone (sticky targeting) to kill micro-jitter
    - Magnetic snapping to the active UI rect centre
    - Smooth glide (capped max speed) so the cursor never teleports
    """

    DEADZONE   = 55.0   # px — ignore movements smaller than this
    GAZE_ALPHA = 0.08   # how fast the invisible gaze anchor follows raw input
    SNAP_SPEED = 10.0   # max px per frame the physical cursor can move

    def __init__(self, screen_w, screen_h):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.gaze_x   = screen_w / 2
        self.gaze_y   = screen_h / 2
        self.smooth_x = screen_w / 2
        self.smooth_y = screen_h / 2

    def update(self, target_x, target_y, active_rect):
        """
        target_x/y  : raw screen-mapped gaze coords
        active_rect : (rx, ry, rw, rh) from SmartUISnapper or None
        Returns (cursor_x, cursor_y)
        """
        # --- DEADZONE: only move gaze anchor when eyes really shift ---
        dx = target_x - self.gaze_x
        dy = target_y - self.gaze_y
        if (dx**2 + dy**2)**0.5 > self.DEADZONE:
            self.gaze_x = self.gaze_x*(1-self.GAZE_ALPHA) + target_x*self.GAZE_ALPHA
            self.gaze_y = self.gaze_y*(1-self.GAZE_ALPHA) + target_y*self.GAZE_ALPHA

        # --- MAGNETIC SNAP: aim at centre of hovered UI element ---
        if active_rect:
            rx, ry, rw, rh = active_rect
            ideal_x = rx + rw / 2
            ideal_y = ry + rh / 2
        else:
            ideal_x = self.gaze_x
            ideal_y = self.gaze_y

        # --- GLIDE: smooth approach, capped speed ---
        ddx = ideal_x - self.smooth_x
        ddy = ideal_y - self.smooth_y
        dist = (ddx**2 + ddy**2)**0.5
        if dist > self.SNAP_SPEED:
            ddx = (ddx / dist) * self.SNAP_SPEED
            ddy = (ddy / dist) * self.SNAP_SPEED

        self.smooth_x += ddx
        self.smooth_y += ddy

        return self.smooth_x, self.smooth_y


# =============================================================================
#  VIRTUAL KEYBOARD  (unchanged from v3)
# =============================================================================
class VirtualKeyboard:
    DWELL_TIME = 1.2
    KEY_W = 72;  KEY_H = 60;  KEY_GAP = 6;  MARGIN = 18;  TEXT_H = 52

    ROWS = [
        ["Q","W","E","R","T","Y","U","I","O","P"],
        ["A","S","D","F","G","H","J","K","L"],
        ["Z","X","C","V","B","N","M","←"],
        ["SPACE","ENTER","CLEAR","CLOSE"],
    ]

    def __init__(self):
        self.visible       = False
        self.typed_text    = ""
        self.hovered_key   = None
        self.dwell_start   = None
        self.dwell_progress= 0.0
        self.last_pressed  = None
        self.last_press_t  = 0
        self._build_key_rects()
        self._canvas_h = (self.TEXT_H + self.MARGIN
                          + len(self.ROWS)*(self.KEY_H+self.KEY_GAP) + self.MARGIN)
        max_row_w = max(sum(self._key_w(k)+self.KEY_GAP for k in row)-self.KEY_GAP for row in self.ROWS)
        self._canvas_w = max_row_w + 2*self.MARGIN

    def _key_w(self, label):
        if label == "SPACE":                    return self.KEY_W*4
        if label in ("ENTER","CLEAR","CLOSE"):  return self.KEY_W*2
        return self.KEY_W

    def _build_key_rects(self):
        self.key_rects = {};  self.key_list = []
        y = self.TEXT_H + self.MARGIN
        for row in self.ROWS:
            total_w   = sum(self._key_w(k)+self.KEY_GAP for k in row)-self.KEY_GAP
            max_row_w = max(sum(self._key_w(k)+self.KEY_GAP for k in r)-self.KEY_GAP for r in self.ROWS)
            x = self.MARGIN + (max_row_w-total_w)//2
            for label in row:
                kw   = self._key_w(label)
                rect = (x, y, x+kw, y+self.KEY_H)
                self.key_rects[label] = rect
                self.key_list.append((label,*rect))
                x += kw+self.KEY_GAP
            y += self.KEY_H+self.KEY_GAP

    def toggle(self):
        self.visible = not self.visible
        if not self.visible: cv2.destroyWindow("Virtual Keyboard")

    def open(self):  self.visible = True
    def close(self):
        self.visible = False
        cv2.destroyWindow("Virtual Keyboard")

    def update_gaze(self, gx, gy):
        if not self.visible: return None
        hit = next((l for l,x1,y1,x2,y2 in self.key_list if x1<=gx<x2 and y1<=gy<y2), None)
        fired = None
        if hit != self.hovered_key:
            self.hovered_key  = hit
            self.dwell_start  = time.time() if hit else None
            self.dwell_progress = 0.0
        elif hit and self.dwell_start:
            elapsed = time.time()-self.dwell_start
            self.dwell_progress = min(1.0, elapsed/self.DWELL_TIME)
            if self.dwell_progress >= 1.0:
                if time.time()-self.last_press_t > self.DWELL_TIME*0.8:
                    fired = hit;  self._press(hit)
                    self.last_press_t = time.time()
                    self.dwell_start  = time.time()
                    self.dwell_progress = 0.0
        return fired

    def blink_press(self):
        if self.hovered_key and self.visible and time.time()-self.last_press_t > 0.4:
            self._press(self.hovered_key)
            self.last_press_t   = time.time()
            self.dwell_start    = time.time()
            self.dwell_progress = 0.0
            return self.hovered_key
        return None

    def _press(self, label):
        self.last_pressed = label
        if   label == "←":     self.typed_text = self.typed_text[:-1]
        elif label == "SPACE": self.typed_text += " "
        elif label == "CLEAR": self.typed_text = ""
        elif label == "ENTER":
            if MOUSE_AVAILABLE and self.typed_text:
                pyautogui.write(self.typed_text, interval=0.03)
            self.typed_text = ""
        elif label == "CLOSE": self.close()
        else: self.typed_text += label

    def render(self):
        if not self.visible: return None
        img = np.zeros((self._canvas_h, self._canvas_w, 3), dtype=np.uint8)
        img[:] = (20,20,30)
        cv2.rectangle(img,(self.MARGIN,6),(self._canvas_w-self.MARGIN,self.TEXT_H-6),(40,40,60),-1)
        cv2.rectangle(img,(self.MARGIN,6),(self._canvas_w-self.MARGIN,self.TEXT_H-6),(80,80,120),1)
        display = self.typed_text if self.typed_text else "▮  start typing…"
        col = (220,220,255) if self.typed_text else (80,80,100)
        cv2.putText(img,display[-48:],(self.MARGIN+8,self.TEXT_H-16),cv2.FONT_HERSHEY_SIMPLEX,0.7,col,1,cv2.LINE_AA)
        for (label,x1,y1,x2,y2) in self.key_list:
            is_hov = label==self.hovered_key
            is_prs = label==self.last_pressed and time.time()-self.last_press_t<0.3
            is_spc = label in ("←","SPACE","ENTER","CLEAR","CLOSE")
            bg = (0,200,100) if is_prs else (60,80,140) if is_hov else (40,40,70) if is_spc else (35,35,55)
            cv2.rectangle(img,(x1+2,y1+2),(x2-2,y2-2),bg,-1)
            if is_hov and self.dwell_progress>0:
                cx,cy=(x1+x2)//2,(y1+y2)//2; r=min(x2-x1,y2-y1)//2-4
                cv2.ellipse(img,(cx,cy),(r,r),-90,0,int(360*self.dwell_progress),(0,220,255),3)
            cv2.rectangle(img,(x1+2,y1+2),(x2-2,y2-2),(0,220,255) if is_hov else (55,55,80),1)
            fs = 0.55 if len(label)>1 else 0.68
            ts = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,fs,1)[0]
            cv2.putText(img,label,(x1+(x2-x1-ts[0])//2,y1+(y2-y1+ts[1])//2),
                        cv2.FONT_HERSHEY_SIMPLEX,fs,(220,230,255) if not is_prs else (10,10,10),1,cv2.LINE_AA)
        cv2.putText(img,"Gaze-dwell or LEFT blink to type  |  K = close keyboard",
                    (self.MARGIN,self._canvas_h-6),cv2.FONT_HERSHEY_SIMPLEX,0.38,(70,70,90),1,cv2.LINE_AA)
        cv2.imshow("Virtual Keyboard",img)
        return img

    @property
    def window_size(self): return self._canvas_w, self._canvas_h


# =============================================================================
#  ATTENTION TRACKER  v4
# =============================================================================
class AttentionTracker:

    # ── landmark indices ──────────────────────────────────────────────────── #
    LEFT_IRIS   = [474,475,476,477]
    RIGHT_IRIS  = [469,470,471,472]
    LEFT_EYE    = [33, 160,158,133,153,144]
    RIGHT_EYE   = [362,385,387,263,373,380]
    L_EYE_LEFT  = 33;   L_EYE_RIGHT = 133
    R_EYE_LEFT  = 362;  R_EYE_RIGHT = 263

    def __init__(self):

        # ── model ─────────────────────────────────────────────────────────── #
        self.model_path = "face_landmarker.task"
        if not os.path.exists(self.model_path):
            print("Downloading Face Landmarker model...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                self.model_path)

        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # ── camera ────────────────────────────────────────────────────────── #
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened(): raise Exception("Webcam not detected")
        self.fps_cam = self.cap.get(cv2.CAP_PROP_FPS) or 30

        # ── screen ────────────────────────────────────────────────────────── #
        if MOUSE_AVAILABLE:
            self.screen_w, self.screen_h = pyautogui.size()
        else:
            self.screen_w, self.screen_h = 1920,1080

        # ── sub-systems (NEW) ─────────────────────────────────────────────── #
        self.ui_snapper   = SmartUISnapper(self.screen_w, self.screen_h)
        self.face_store   = FaceIdentityStore()
        self.multi_blink  = MultiBlinkDetector()
        self.smooth_cursor= SmoothCursor(self.screen_w, self.screen_h)

        # ── face identity state ───────────────────────────────────────────── #
        self.app_state        = "IDENTIFYING"   # IDENTIFYING → CALIBRATING → TRACKING
        self.face_signatures  = []
        self.final_signature  = None
        self.identity_timer   = time.time()

        # ── blink thresholds (adjusted after calibration) ─────────────────── #
        self.EAR_THRESHOLD_L = 0.23
        self.EAR_THRESHOLD_R = 0.23
        self.CONSEC_FRAMES   = 3
        self.left_blink_counter  = 0
        self.right_blink_counter = 0
        self.left_blink_total    = 0
        self.right_blink_total   = 0

        # ── click cooldown ────────────────────────────────────────────────── #
        self.CLICK_COOLDOWN    = 0.6
        self.last_left_click_t = 0
        self.last_right_click_t= 0
        self.last_gesture_t    = 0   # for multi-blink cooldown
        self.GESTURE_COOLDOWN  = 1.5

        # ── gaze mouse ────────────────────────────────────────────────────── #
        self.mouse_mode  = True
        self.mouse_alpha = 0.20
        self.mouse_x     = self.screen_w  // 2
        self.mouse_y     = self.screen_h  // 2
        self.gaze_calib  = [0.35, 0.65, 0.30, 0.70]

        # ── drowsiness ────────────────────────────────────────────────────── #
        self.DROWSY_FRAMES   = int(self.fps_cam * 2)
        self.drowsy_counter  = 0
        self.last_alert_time = 0

        # ── smoothing (legacy iris smoothing for attention score) ─────────── #
        self.prev_x = None;  self.prev_y = None;  self.alpha = 0.7

        # ── head pose ─────────────────────────────────────────────────────── #
        self.YAW_THRESHOLD   = 25
        self.PITCH_THRESHOLD = 20

        # ── session data ──────────────────────────────────────────────────── #
        self.session_start      = time.time()
        self.prev_time          = 0
        self.rolling_blinks     = deque()
        self.rolling_gaze       = deque()
        self.log_rows           = []
        self.live_attention     = "CALIBRATING"
        self.live_attention_pct = 0
        self.gaze_direction     = "CENTER"
        self.head_status        = "FORWARD"
        self.attention_history  = deque(maxlen=150)

        # ── smart tv box animation ────────────────────────────────────────── #
        self.anim_rx = 0.0;  self.anim_ry = 0.0
        self.anim_rw = 0.0;  self.anim_rh = 0.0
        self.first_box = True
        self.click_flash = None   # ("LEFT"|"RIGHT"|gesture, timestamp)

        # ── virtual keyboard ──────────────────────────────────────────────── #
        self.vkb = VirtualKeyboard()

        print("\n╔══════════════════════════════════════════════════════════╗")
        print("║   Attention Tracker v4  —  Starting Up                  ║")
        print("╠══════════════════════════════════════════════════════════╣")
        print("║  Scanning face identity first (2 s)…                    ║")
        print("║  Left  eye blink → Left  click / KB type                ║")
        print("║  Right eye blink → Right click                          ║")
        print("║  Double blink    → Double click                         ║")
        print("║  Triple blink    → Minimize all  (Win+D)                ║")
        print("║  Quad   blink    → Close window  (Alt+F4)               ║")
        print("║  Both eyes hold  → Open on-screen keyboard              ║")
        print("║  K key  → Toggle Virtual Keyboard                       ║")
        print("║  M key  → Toggle mouse control ON/OFF                   ║")
        print("║  ESC    → End session & show report                     ║")
        print("╚══════════════════════════════════════════════════════════╝\n")

    # =========================================================================
    #  CALIBRATION  (EAR phase only — gaze phase runs if no profile matched)
    # =========================================================================
    def calibrate_ear(self, duration=3):
        print(f"[CALIBRATION] Keep both eyes open naturally for {duration}s …")
        left_ears, right_ears = [], []
        deadline = time.time() + duration
        while time.time() < deadline:
            success, frame = self.cap.read()
            if not success: continue
            frame = cv2.flip(frame, 1);  h,w,_ = frame.shape
            overlay = frame.copy()
            cv2.rectangle(overlay,(0,0),(w,h),(20,20,20),-1)
            cv2.addWeighted(overlay,0.45,frame,0.55,0,frame)
            cd = int(deadline-time.time())+1
            cv2.putText(frame,"CALIBRATION  --  keep eyes open",
                        (w//2-230,h//2-20),cv2.FONT_HERSHEY_DUPLEX,0.9,(0,220,255),2)
            cv2.putText(frame,f"{cd}s remaining",
                        (w//2-80,h//2+40),cv2.FONT_HERSHEY_SIMPLEX,1.0,(100,255,100),2)
            cv2.imshow("Attention Tracking System",frame); cv2.waitKey(1)
            rgb    = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
            result = self.detector.detect_for_video(mp_img,int(time.time()*1000))
            if result.face_landmarks:
                lm = result.face_landmarks[0]
                le = [(int(lm[i].x*w),int(lm[i].y*h)) for i in self.LEFT_EYE]
                re = [(int(lm[i].x*w),int(lm[i].y*h)) for i in self.RIGHT_EYE]
                left_ears.append(self.calculate_EAR(le))
                right_ears.append(self.calculate_EAR(re))
        if left_ears:
            self.EAR_THRESHOLD_L = round(np.mean(left_ears) -0.05,4)
            self.EAR_THRESHOLD_R = round(np.mean(right_ears)-0.05,4)
            print(f"[CALIBRATION] EAR thresholds  L:{self.EAR_THRESHOLD_L}  R:{self.EAR_THRESHOLD_R}")
        else:
            print("[CALIBRATION] No face detected — using defaults (0.23)")

    def calibrate_gaze(self):
        if not MOUSE_AVAILABLE: return
        print("[CALIBRATION] Phase 2: Gaze corner mapping …")
        corners = [
            ("TOP-LEFT",    (0.05,0.08)),("TOP-RIGHT",   (0.95,0.08)),
            ("BOTTOM-LEFT", (0.05,0.92)),("BOTTOM-RIGHT",(0.95,0.92)),
        ]
        all_gx, all_gy = [], []
        for label,(tx,ty) in corners:
            deadline = time.time()+1.8
            while time.time()<deadline:
                success,frame = self.cap.read()
                if not success: continue
                frame = cv2.flip(frame,1);  h,w,_ = frame.shape
                overlay = frame.copy()
                cv2.rectangle(overlay,(0,0),(w,h),(20,20,20),-1)
                cv2.addWeighted(overlay,0.5,frame,0.5,0,frame)
                cv2.putText(frame,f"Look at  {label}",(w//2-180,h//2),cv2.FONT_HERSHEY_DUPLEX,0.95,(0,220,255),2)
                dx,dy = int(tx*w),int(ty*h)
                cv2.circle(frame,(dx,dy),16,(0,255,0),-1); cv2.circle(frame,(dx,dy),20,(255,255,255),2)
                cv2.imshow("Attention Tracking System",frame); cv2.waitKey(1)
                rgb    = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                result = self.detector.detect_for_video(mp_img,int(time.time()*1000))
                if result.face_landmarks:
                    lm = result.face_landmarks[0]
                    gx = (lm[self.LEFT_IRIS[0]].x+lm[self.LEFT_IRIS[2]].x)/2
                    gy = (lm[self.LEFT_IRIS[0]].y+lm[self.LEFT_IRIS[2]].y)/2
                    all_gx.append(gx); all_gy.append(gy)
        if len(all_gx)>20:
            self.gaze_calib = [
                np.percentile(all_gx,10)-0.02, np.percentile(all_gx,90)+0.02,
                np.percentile(all_gy,10)-0.02, np.percentile(all_gy,90)+0.02,
            ]
            print(f"[CALIBRATION] Gaze X: {self.gaze_calib[:2]}  Y: {self.gaze_calib[2:]}")
        else:
            print("[CALIBRATION] Not enough gaze data — using defaults")

    # =========================================================================
    #  FACE IDENTITY PHASE  (runs before calibration in the main loop)
    # =========================================================================
    def _process_identity(self, landmarks):
        sig = FaceIdentityStore.compute_signature(landmarks)
        if sig: self.face_signatures.append(sig)

        elapsed = time.time()-self.identity_timer
        if elapsed < 2.0: return   # keep collecting for 2 s

        if len(self.face_signatures) > 10:
            avg_sig = [sum(col)/len(col) for col in zip(*self.face_signatures)]
            self.final_signature = avg_sig
            match = self.face_store.find_match(avg_sig)
            if match:
                self.gaze_calib      = match["gaze_calib"]
                self.EAR_THRESHOLD_L = match["ear_l"]
                self.EAR_THRESHOLD_R = match["ear_r"]
                self.app_state       = "TRACKING"
                print("\n✅ Face recognised! Loaded saved calibration profile.")
                print("⚠️  Press ESC to end session.\n")
            else:
                print("\n🆕 New face — starting calibration …")
                self.calibrate_ear(duration=3)
                self.calibrate_gaze()
                # Save the new profile
                if self.final_signature:
                    self.face_store.save(self.final_signature, self.gaze_calib,
                                         self.EAR_THRESHOLD_L, self.EAR_THRESHOLD_R)
                self.app_state = "TRACKING"
                print("✅ Calibration complete — OS Mouse Control Active.")
                print("⚠️  Press ESC to end session.\n")
        else:
            # Reset if face wasn't visible enough
            self.face_signatures = []
            self.identity_timer  = time.time()

    # =========================================================================
    #  HELPERS
    # =========================================================================
    def calculate_EAR(self, pts):
        A = np.linalg.norm(np.array(pts[1])-np.array(pts[5]))
        B = np.linalg.norm(np.array(pts[2])-np.array(pts[4]))
        C = np.linalg.norm(np.array(pts[0])-np.array(pts[3]))
        return (A+B)/(2.0*C+1e-6)

    def get_gaze_direction(self, landmarks, w, h):
        lc  = landmarks[self.L_EYE_LEFT].x *w
        rc  = landmarks[self.L_EYE_RIGHT].x*w
        ix  = (landmarks[self.LEFT_IRIS[0]].x*w + landmarks[self.LEFT_IRIS[2]].x*w)/2
        r   = (ix-lc)/(rc-lc+1e-6)
        if   r<0.37: return "RIGHT ->"
        elif r>0.63: return "<- LEFT"
        return "CENTER"

    def gaze_to_screen(self, gx, gy):
        xmn,xmx,ymn,ymx = self.gaze_calib
        sx = max(0.,min(1.,(gx-xmn)/(xmx-xmn+1e-6)))
        sy = max(0.,min(1.,(gy-ymn)/(ymx-ymn+1e-6)))
        return int(sx*self.screen_w), int(sy*self.screen_h)

    def screen_to_keyboard(self, sx, sy):
        kw,kh = self.vkb.window_size
        return int((sx/self.screen_w)*kw), int((sy/self.screen_h)*kh)

    def get_head_pose(self, matrix):
        m = np.array(matrix).reshape(4,4);  R = m[:3,:3]
        yaw   = np.degrees(np.arctan2(R[1][0],R[0][0]))
        pitch = np.degrees(np.arctan2(-R[2][0],np.sqrt(R[2][1]**2+R[2][2]**2)))
        status = "FORWARD"
        if   abs(yaw) > self.YAW_THRESHOLD:   status = f"TURNED {'RIGHT' if yaw>0 else 'LEFT'}"
        elif pitch >  self.PITCH_THRESHOLD:    status = "LOOKING DOWN"
        elif pitch < -self.PITCH_THRESHOLD:    status = "LOOKING UP"
        return yaw, pitch, status

    def update_live_attention(self, timestamp):
        cutoff = timestamp-5.0
        while self.rolling_blinks and self.rolling_blinks[0]<cutoff: self.rolling_blinks.popleft()
        while self.rolling_gaze   and self.rolling_gaze[0][0]<cutoff: self.rolling_gaze.popleft()
        if len(self.rolling_gaze)<10:
            self.live_attention="CALIBRATING"; self.live_attention_pct=0; return
        blink_rate  = len(self.rolling_blinks)*12
        gaze_data   = np.array([[r[1],r[2]] for r in self.rolling_gaze])
        variance    = gaze_data[:,0].var()+gaze_data[:,1].var()
        focus_mask  = (gaze_data[:,1]>0.25)&(gaze_data[:,1]<0.75)
        focus_ratio = focus_mask.sum()/len(gaze_data)
        blink_score    = 1.0 if 8<=blink_rate<=22 else max(0,1-abs(blink_rate-15)/15)
        variance_score = 1/(1+variance*10)
        raw = focus_ratio*0.45+variance_score*0.30+blink_score*0.25
        pct = min(100,max(0,int(raw*100)))
        self.live_attention_pct = pct
        self.live_attention     = ("HIGH" if pct>=70 else "MEDIUM" if pct>=40 else "LOW")
        self.attention_history.append(pct)

    # =========================================================================
    #  HUD
    # =========================================================================
    @staticmethod
    def draw_rounded_rect(img, pt1, pt2, color, alpha=0.55, radius=12, thickness=-1):
        overlay = img.copy()
        x1,y1=pt1; x2,y2=pt2
        cv2.rectangle(overlay,(x1+radius,y1),(x2-radius,y2),color,thickness)
        cv2.rectangle(overlay,(x1,y1+radius),(x2,y2-radius),color,thickness)
        for cx,cy in [(x1+radius,y1+radius),(x2-radius,y1+radius),(x1+radius,y2-radius),(x2-radius,y2-radius)]:
            cv2.circle(overlay,(cx,cy),radius,color,thickness)
        cv2.addWeighted(overlay,alpha,img,1-alpha,0,img)

    def _draw_smart_tv_box(self, frame, active_rect):
        """Draw animated glowing bounding box around hovered UI element."""
        if not active_rect: 
            self.first_box = True
            return
        rx,ry,rw,rh = active_rect
        if self.first_box:
            self.anim_rx,self.anim_ry = float(rx),float(ry)
            self.anim_rw,self.anim_rh = float(rw),float(rh)
            self.first_box = False
        else:
            spd = 0.08   # slightly faster than EyeOS for responsiveness
            self.anim_rx += (rx-self.anim_rx)*spd
            self.anim_ry += (ry-self.anim_ry)*spd
            self.anim_rw += (rw-self.anim_rw)*spd
            self.anim_rh += (rh-self.anim_rh)*spd

        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (int(self.anim_rx),int(self.anim_ry)),
                      (int(self.anim_rx+self.anim_rw),int(self.anim_ry+self.anim_rh)),
                      (0,255,255),-1)
        cv2.addWeighted(overlay,0.08,frame,0.92,0,frame)
        cv2.rectangle(frame,
                      (int(self.anim_rx)-1,int(self.anim_ry)-1),
                      (int(self.anim_rx+self.anim_rw)+1,int(self.anim_ry+self.anim_rh)+1),
                      (0,200,200),3)
        # Rounded corners drawn on top
        r = 10
        for (cx,cy) in [
            (int(self.anim_rx)+r,        int(self.anim_ry)+r),
            (int(self.anim_rx+self.anim_rw)-r, int(self.anim_ry)+r),
            (int(self.anim_rx)+r,        int(self.anim_ry+self.anim_rh)-r),
            (int(self.anim_rx+self.anim_rw)-r, int(self.anim_ry+self.anim_rh)-r),
        ]:
            cv2.circle(frame,(cx,cy),r,(0,220,255),2)

    def draw_hud(self, frame, fps, timestamp, left_ear, right_ear, active_rect):
        h,w = frame.shape[:2]
        PAD  = 14
        panel_w, panel_h = 310, 315
        self.draw_rounded_rect(frame,(10,10),(10+panel_w,10+panel_h),(15,15,15),alpha=0.65)

        attn_color = {"HIGH":(50,220,80),"MEDIUM":(40,180,255),
                      "LOW":(30,30,220),"CALIBRATING":(160,160,160)}.get(self.live_attention,(255,255,255))

        def put(text,y,scale=0.60,color=(220,220,220),bold=1):
            cv2.putText(frame,text,(PAD+10,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,bold,cv2.LINE_AA)

        elapsed = int(timestamp)
        put(f"Time      : {elapsed//60:02d}:{elapsed%60:02d}",40)
        put(f"FPS       : {int(fps)}",64)
        put(f"L-Blinks  : {self.left_blink_total}  (left click)",88)
        put(f"R-Blinks  : {self.right_blink_total}  (right click)",112)
        put(f"EAR  L:{left_ear:.3f}   R:{right_ear:.3f}",136)
        put(f"Gaze      : {self.gaze_direction}",160)
        put(f"Head      : {self.head_status}",184)
        mouse_col = (50,220,80) if self.mouse_mode else (120,120,120)
        put(f"Mouse ctrl: {'ON' if self.mouse_mode else 'OFF'}   (M)",208,color=mouse_col)
        snap_col = (0,220,255) if (UI_SNAP_AVAILABLE and active_rect) else (80,80,80)
        put(f"UI Snap   : {'LOCKED' if active_rect else ('ON' if UI_SNAP_AVAILABLE else 'OFF')}",232,color=snap_col)
        kb_col = (0,220,255) if self.vkb.visible else (100,100,100)
        put(f"Keyboard  : {'ON  (K)' if self.vkb.visible else 'OFF (K)'}",256,color=kb_col)
        put(f"Attn      : {self.live_attention} ({self.live_attention_pct}%)",
            282,scale=0.68,color=attn_color,bold=2)

        # Progress bar
        bx,by,bw,bh = 10,282+16,panel_w,13
        cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(50,50,50),-1)
        cv2.rectangle(frame,(bx,by),(bx+int(bw*self.live_attention_pct/100),by+bh),attn_color,-1)
        cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(100,100,100),1)

        # Sparkline
        if len(self.attention_history)>2:
            sx,sy2,sw2,sh2 = 10,311+16,panel_w,42
            self.draw_rounded_rect(frame,(sx,sy2),(sx+sw2,sy2+sh2),(15,15,15),alpha=0.5)
            hist=list(self.attention_history); n=len(hist)
            for i in range(1,n):
                x1=sx+int((i-1)/(n-1)*sw2); x2=sx+int(i/(n-1)*sw2)
                y1=sy2+sh2-int(hist[i-1]/100*sh2); y2=sy2+sh2-int(hist[i]/100*sh2)
                cv2.line(frame,(x1,y1),(x2,y2),attn_color,1,cv2.LINE_AA)

        # Smart TV box
        self._draw_smart_tv_box(frame, active_rect)

        # Drowsy alert
        if self.drowsy_counter >= self.DROWSY_FRAMES:
            ax,ay = w//2-190,h//2-30
            self.draw_rounded_rect(frame,(ax-10,ay-42),(ax+400,ay+22),(0,0,180),alpha=0.80)
            cv2.putText(frame,"!! DROWSY - WAKE UP !!",(ax,ay),cv2.FONT_HERSHEY_DUPLEX,1.1,(30,80,255),2,cv2.LINE_AA)
            if SOUND_AVAILABLE and time.time()-self.last_alert_time>3:
                winsound.Beep(900,400); self.last_alert_time=time.time()

        if self.head_status!="FORWARD":
            cv2.putText(frame,f"HEAD: {self.head_status}",(w-290,40),cv2.FONT_HERSHEY_SIMPLEX,0.75,(30,165,255),2,cv2.LINE_AA)

        # Gesture flash
        now = time.time()
        if self.click_flash and now-self.click_flash[1]<0.45:
            label,_=self.click_flash
            colors = {"LEFT":(0,220,255),"RIGHT":(255,140,0),
                      "DOUBLE_BLINK":(0,255,255),"TRIPLE_BLINK":(0,0,255),
                      "QUAD_BLINK":(255,0,0),"BOTH":(255,0,255)}
            texts  = {"LEFT":"< LEFT CLICK","RIGHT":"RIGHT CLICK >",
                      "DOUBLE_BLINK":"◆ DOUBLE CLICK","TRIPLE_BLINK":"⊞ MINIMISED ALL",
                      "QUAD_BLINK":"✕ CLOSED WINDOW","BOTH":"⌨ KEYBOARD"}
            col = colors.get(label,(200,200,200))
            txt = texts.get(label, label)
            cv2.putText(frame,txt,(w//2-160,h-55),cv2.FONT_HERSHEY_DUPLEX,0.95,col,2,cv2.LINE_AA)

        if self.vkb.visible and self.vkb.typed_text:
            cv2.putText(frame,f"Typing: {self.vkb.typed_text[-30:]}",(w//2-220,h-30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,220,255),1,cv2.LINE_AA)

        # Identity status (shown briefly at top-right)
        id_text = "✅ Face ID: Recognised" if self.final_signature else "🔍 Scanning face…"
        id_col  = (50,220,80) if self.final_signature else (160,160,160)
        cv2.putText(frame,id_text,(w-330,30),cv2.FONT_HERSHEY_SIMPLEX,0.55,id_col,1,cv2.LINE_AA)

        cv2.putText(frame,"ESC=end  |  M=mouse  |  K=keyboard",
                    (w//2-180,h-12),cv2.FONT_HERSHEY_SIMPLEX,0.44,(110,110,110),1,cv2.LINE_AA)

    # =========================================================================
    #  MAIN LOOP
    # =========================================================================
    def run(self):
        while True:
            success, frame = self.cap.read()
            if not success: break

            frame     = cv2.flip(frame,1)
            rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            h,w,_     = frame.shape
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb_frame)
            ts_ms     = int(time.time()*1000)
            result    = self.detector.detect_for_video(mp_image,ts_ms)
            timestamp = time.time()-self.session_start

            left_ear = right_ear = 0.0
            blink_flag = 0
            active_rect = self.ui_snapper.get_rect()

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]

                # ── FACE IDENTITY PHASE ───────────────────────────────────── #
                if self.app_state == "IDENTIFYING":
                    # Draw scanning overlay
                    overlay = frame.copy()
                    cv2.rectangle(overlay,(0,0),(w,h),(10,10,20),-1)
                    cv2.addWeighted(overlay,0.4,frame,0.6,0,frame)
                    cv2.putText(frame,"Scanning Face Identity…",
                                (w//2-200,h//2-10),cv2.FONT_HERSHEY_DUPLEX,0.9,(0,220,255),2)
                    cv2.putText(frame,"Please look at the screen.",
                                (w//2-170,h//2+40),cv2.FONT_HERSHEY_SIMPLEX,0.75,(150,150,150),1)
                    cv2.imshow("Attention Tracking System",frame); cv2.waitKey(1)
                    self._process_identity(landmarks)
                    continue   # skip rest of loop until tracking starts

                # ── IRIS CIRCLES ─────────────────────────────────────────── #
                lpts = [(int(landmarks[i].x*w),int(landmarks[i].y*h)) for i in self.LEFT_IRIS]
                rpts = [(int(landmarks[i].x*w),int(landmarks[i].y*h)) for i in self.RIGHT_IRIS]
                (lcx,lcy),l_r = cv2.minEnclosingCircle(np.array(lpts))
                (rcx,rcy),r_r = cv2.minEnclosingCircle(np.array(rpts))
                cv2.circle(frame,(int(lcx),int(lcy)),int(l_r),(0,230,100),2)
                cv2.circle(frame,(int(rcx),int(rcy)),int(r_r),(0,230,100),2)

                avg_x = (lcx+rcx)/2;  avg_y = (lcy+rcy)/2
                if self.prev_x is not None:
                    avg_x = self.alpha*avg_x+(1-self.alpha)*self.prev_x
                    avg_y = self.alpha*avg_y+(1-self.alpha)*self.prev_y
                self.prev_x,self.prev_y = avg_x,avg_y
                cv2.circle(frame,(int(avg_x),int(avg_y)),5,(0,0,255),-1)

                # ── EAR ──────────────────────────────────────────────────── #
                le = [(int(landmarks[i].x*w),int(landmarks[i].y*h)) for i in self.LEFT_EYE]
                re = [(int(landmarks[i].x*w),int(landmarks[i].y*h)) for i in self.RIGHT_EYE]
                left_ear  = self.calculate_EAR(le)
                right_ear = self.calculate_EAR(re)
                now = time.time()

                left_closed  = left_ear  < self.EAR_THRESHOLD_L
                right_closed = right_ear < self.EAR_THRESHOLD_R

                # ── MULTI-BLINK GESTURE DETECTION ────────────────────────── #
                gesture = self.multi_blink.update(left_closed, right_closed)
                if gesture and now-self.last_gesture_t > self.GESTURE_COOLDOWN:
                    self.last_gesture_t = now
                    self.click_flash    = (gesture, now)
                    if   gesture == "DOUBLE_BLINK":
                        if MOUSE_AVAILABLE: pyautogui.doubleClick()
                    elif gesture == "TRIPLE_BLINK":
                        if MOUSE_AVAILABLE: pyautogui.hotkey('win','d')
                    elif gesture == "QUAD_BLINK":
                        if MOUSE_AVAILABLE: pyautogui.hotkey('alt','f4')
                    elif gesture == "BOTH":
                        self.vkb.toggle()

                # ── SINGLE BLINK — LEFT click ─────────────────────────────── #
                if left_closed:
                    self.left_blink_counter += 1;  self.drowsy_counter += 1
                else:
                    if self.left_blink_counter >= self.CONSEC_FRAMES:
                        self.left_blink_total += 1;  blink_flag = 1
                        self.rolling_blinks.append(timestamp)
                        if self.vkb.visible:
                            self.vkb.blink_press()
                        elif (self.mouse_mode and MOUSE_AVAILABLE
                              and now-self.last_left_click_t > self.CLICK_COOLDOWN
                              and not gesture):
                            pyautogui.click(button='left')
                            self.last_left_click_t = now
                            self.click_flash = ("LEFT", now)
                    self.left_blink_counter = 0

                # ── SINGLE BLINK — RIGHT click ────────────────────────────── #
                if right_closed:
                    self.right_blink_counter += 1;  self.drowsy_counter += 1
                else:
                    if self.right_blink_counter >= self.CONSEC_FRAMES:
                        self.right_blink_total += 1;  blink_flag = 1
                        self.rolling_blinks.append(timestamp)
                        if (self.mouse_mode and MOUSE_AVAILABLE and not self.vkb.visible
                              and now-self.last_right_click_t > self.CLICK_COOLDOWN
                              and not gesture):
                            pyautogui.click(button='right')
                            self.last_right_click_t = now
                            self.click_flash = ("RIGHT", now)
                    self.right_blink_counter = 0

                # Reset drowsy
                if not left_closed and not right_closed:
                    self.drowsy_counter = 0

                self.gaze_direction = self.get_gaze_direction(landmarks,w,h)
                if result.facial_transformation_matrixes:
                    _,_,self.head_status = self.get_head_pose(result.facial_transformation_matrixes[0])

                # ── GAZE → SCREEN → SMOOTH CURSOR ────────────────────────── #
                gx_raw = (landmarks[self.LEFT_IRIS[0]].x+landmarks[self.LEFT_IRIS[2]].x)/2
                gy_raw = (landmarks[self.LEFT_IRIS[0]].y+landmarks[self.LEFT_IRIS[2]].y)/2

                if MOUSE_AVAILABLE:
                    raw_sx, raw_sy = self.gaze_to_screen(gx_raw, gy_raw)
                    # Feed raw gaze into UI snapper for element detection
                    self.ui_snapper.update_gaze(raw_sx, raw_sy)
                    # Smooth cursor engine handles deadzone + snapping
                    cx, cy = self.smooth_cursor.update(raw_sx, raw_sy, active_rect)
                    self.mouse_x, self.mouse_y = int(cx), int(cy)

                    if self.mouse_mode and not self.vkb.visible:
                        pyautogui.moveTo(self.mouse_x, self.mouse_y)
                    if self.vkb.visible:
                        kx,ky = self.screen_to_keyboard(self.mouse_x,self.mouse_y)
                        self.vkb.update_gaze(kx,ky)

                gx_norm = avg_x/w;  gy_norm = avg_y/h
                self.rolling_gaze.append((timestamp,gx_norm,gy_norm))
                self.update_live_attention(timestamp)

                self.log_rows.append([
                    round(timestamp,4),round(gx_norm,5),round(gy_norm,5),
                    blink_flag,self.gaze_direction,self.head_status,
                    round(left_ear,4),round(right_ear,4),self.live_attention_pct,
                ])

            curr_time = time.time()
            fps = 1/(curr_time-self.prev_time) if self.prev_time else 0
            self.prev_time = curr_time

            if self.app_state == "TRACKING":
                self.draw_hud(frame,fps,timestamp,left_ear,right_ear,active_rect)
            cv2.imshow("Attention Tracking System",frame)
            self.vkb.render()

            key = cv2.waitKey(1)&0xFF
            if key==27: break
            elif key in (ord('m'),ord('M')):
                self.mouse_mode = not self.mouse_mode
                print(f"[INFO] Mouse control {'ENABLED' if self.mouse_mode else 'DISABLED'}")
            elif key in (ord('k'),ord('K')):
                self.vkb.toggle()
                print(f"[INFO] Virtual keyboard {'OPENED' if self.vkb.visible else 'CLOSED'}")

        self.cap.release()
        cv2.destroyAllWindows()
        self._save_csv()
        self.session_summary()
        self.generate_dashboard()

    # =========================================================================
    #  CSV / SUMMARY / DASHBOARD  (unchanged from v3)
    # =========================================================================
    def _save_csv(self):
        with open("gaze_log.csv","w",newline="") as f:
            csv.writer(f).writerow(["timestamp","gaze_x","gaze_y","blink",
                                     "gaze_dir","head_status","ear_left","ear_right","attention_pct"])
            csv.writer(f).writerows(self.log_rows)
        print("[INFO] gaze_log.csv saved.")

    def session_summary(self):
        data = pd.read_csv("gaze_log.csv")
        if data.empty: print("No data recorded."); return
        total_time   = data["timestamp"].iloc[-1]
        total_blinks = int(data["blink"].sum())
        blink_rate   = total_blinks/(total_time/60) if total_time>0 else 0
        gaze_var     = data["gaze_x"].var()+data["gaze_y"].var()
        avg_attn     = data["attention_pct"].mean()
        distract_pct = len(data[data["head_status"]!="FORWARD"])/len(data)*100
        print("\n╔══════════════════════════════════════════╗")
        print("║         SESSION SUMMARY                  ║")
        print("╠══════════════════════════════════════════╣")
        print(f"║  Duration      : {int(total_time//60):02d}m {int(total_time%60):02d}s          ║")
        print(f"║  L-Eye Blinks  : {self.left_blink_total:<24}║")
        print(f"║  R-Eye Blinks  : {self.right_blink_total:<24}║")
        print(f"║  Total Blinks  : {total_blinks:<24}║")
        print(f"║  Blink Rate    : {blink_rate:.2f} blinks/min       ║")
        print(f"║  Gaze Variance : {gaze_var:.4f}               ║")
        print(f"║  Avg Attention : {avg_attn:.1f}%                  ║")
        print(f"║  Distracted    : {distract_pct:.1f}% of session       ║")
        print("╚══════════════════════════════════════════╝\n")

    def generate_dashboard(self):
        data = pd.read_csv("gaze_log.csv")
        if len(data)<5: print("Not enough data for dashboard."); return
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(20,10),facecolor="#0d0d0d")
        fig.suptitle("Attention Tracker v4 — Session Report",
                     fontsize=18,color="#e0e0e0",y=0.97)
        gs = gridspec.GridSpec(2,4,figure=fig,hspace=0.45,wspace=0.35,
                               left=0.05,right=0.97,top=0.91,bottom=0.08)

        def styled_ax(ax,title):
            ax.set_facecolor("#111111"); ax.set_title(title,color="#aaaaaa",fontsize=11)
            for sp in ax.spines.values(): sp.set_edgecolor("#333333")
            ax.tick_params(colors="#777777",labelsize=8)

        ax1=fig.add_subplot(gs[0,0]); styled_ax(ax1,"Gaze Heatmap")
        sns.kdeplot(x=data["gaze_x"],y=data["gaze_y"],fill=True,cmap="inferno",bw_adjust=0.6,ax=ax1)
        ax1.set_xlim(0,1); ax1.set_ylim(0,1); ax1.invert_yaxis()
        ax1.set_xlabel("X (norm)",color="#888",fontsize=9); ax1.set_ylabel("Y (norm)",color="#888",fontsize=9)

        ax2=fig.add_subplot(gs[0,1]); styled_ax(ax2,"Blink Timeline")
        blinks=data[data["blink"]==1]
        ax2.vlines(blinks["timestamp"],0,1,colors="#00e5ff",linewidth=1.2,alpha=0.85)
        ax2.set_xlabel("Time (s)",color="#888",fontsize=9); ax2.set_yticks([])
        ax2.set_xlim(data["timestamp"].min(),data["timestamp"].max())

        ax3=fig.add_subplot(gs[0,2]); styled_ax(ax3,"Attention % Over Time")
        ax3.plot(data["timestamp"],data["attention_pct"],color="#39ff14",linewidth=1.2)
        ax3.axhline(70,color="#ffcc00",linewidth=0.8,linestyle="--",label="High (70%)")
        ax3.axhline(40,color="#ff4500",linewidth=0.8,linestyle="--",label="Low (40%)")
        ax3.set_ylim(0,105); ax3.legend(fontsize=7,loc="lower right")
        ax3.set_xlabel("Time (s)",color="#888",fontsize=9); ax3.set_ylabel("Score (%)",color="#888",fontsize=9)

        ax4=fig.add_subplot(gs[0,3]); styled_ax(ax4,"Gaze Direction Distribution")
        dc=data["gaze_dir"].value_counts()
        ax4.pie(dc.values,labels=dc.index,
                colors=["#00e5ff","#ff6ec7","#39ff14","#ffa500"][:len(dc)],
                autopct="%1.1f%%",textprops={"color":"#cccccc","fontsize":9},startangle=90)

        ax5=fig.add_subplot(gs[1,0]); styled_ax(ax5,"Left Eye EAR")
        ax5.plot(data["timestamp"],data["ear_left"],color="#00e5ff",linewidth=0.9)
        ax5.axhline(self.EAR_THRESHOLD_L,color="#ff4444",linewidth=1.0,linestyle="--",
                    label=f"Threshold ({self.EAR_THRESHOLD_L})")
        ax5.set_xlabel("Time (s)",color="#888",fontsize=9); ax5.set_ylabel("EAR",color="#888",fontsize=9)
        ax5.legend(fontsize=7)

        ax6=fig.add_subplot(gs[1,1]); styled_ax(ax6,"Right Eye EAR")
        ax6.plot(data["timestamp"],data["ear_right"],color="#ffa500",linewidth=0.9)
        ax6.axhline(self.EAR_THRESHOLD_R,color="#ff4444",linewidth=1.0,linestyle="--",
                    label=f"Threshold ({self.EAR_THRESHOLD_R})")
        ax6.set_xlabel("Time (s)",color="#888",fontsize=9); ax6.set_ylabel("EAR",color="#888",fontsize=9)
        ax6.legend(fontsize=7)

        ax7=fig.add_subplot(gs[1,2]); styled_ax(ax7,"Gaze X Drift")
        ax7.plot(data["timestamp"],data["gaze_x"],color="#ff6ec7",linewidth=0.8)
        ax7.axhline(0.5,color="#ffffff",linewidth=0.6,linestyle=":",label="Centre")
        ax7.set_xlabel("Time (s)",color="#888",fontsize=9); ax7.set_ylabel("Gaze X",color="#888",fontsize=9)
        ax7.legend(fontsize=7)

        ax8=fig.add_subplot(gs[1,3]); styled_ax(ax8,"Head Status Distribution")
        hc=data["head_status"].value_counts()
        ax8.bar(hc.index,hc.values,
                color=["#39ff14" if s=="FORWARD" else "#ff4500" for s in hc.index])
        ax8.set_xlabel("Status",color="#888",fontsize=9); ax8.set_ylabel("Frames",color="#888",fontsize=9)
        ax8.tick_params(axis='x',rotation=15)

        plt.savefig("session_report.png",dpi=140,bbox_inches="tight",facecolor="#0d0d0d")
        print("[INFO] session_report.png saved.")
        plt.show()


# =============================================================================
#  ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    tracker = AttentionTracker()
    tracker.run()   # identity scan + calibration run automatically inside the loop