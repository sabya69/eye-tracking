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
import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque

# ── Mouse control ─────────────────────────────────────────────────────────── #
try:
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE    = 0
    MOUSE_AVAILABLE = True
except ImportError:
    MOUSE_AVAILABLE = False
    print("[WARN] pyautogui not found.  pip install pyautogui")

# ── Windows sound ─────────────────────────────────────────────────────────── #
try:
    import winsound
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False


# =============================================================================
#  GAZE CURSOR  — animated crosshair + reticle drawn on the webcam feed
# =============================================================================
class GazeCursor:
    def __init__(self):
        self._sx    = 0.5   # smoothed normalised x in camera space
        self._sy    = 0.5
        self._alpha = 0.18  # EMA factor
        self._pulse = 0.0
        self.active = True

    def update(self, nx: float, ny: float):
        self._sx += self._alpha * (nx - self._sx)
        self._sy += self._alpha * (ny - self._sy)
        self._pulse = (self._pulse + 0.12) % (2 * np.pi)

    def draw(self, frame):
        if not self.active:
            return
        h, w = frame.shape[:2]
        cx = int(np.clip(self._sx, 0.01, 0.99) * w)
        cy = int(np.clip(self._sy, 0.01, 0.99) * h)

        CROSS_LEN  = 18;  CROSS_GAP = 6
        BOX_HALF   = 28;  BRACKET   = 10
        RING_MIN   = 14;  RING_MAX  = 20

        # pulsing ring
        pr = int(RING_MIN + (RING_MAX - RING_MIN) * (0.5 + 0.5 * np.sin(self._pulse)))
        pa = 0.4 + 0.3 * np.sin(self._pulse)
        ov = frame.copy()
        cv2.circle(ov, (cx, cy), pr, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.addWeighted(ov, pa, frame, 1 - pa, 0, frame)

        # centre dot
        cv2.circle(frame, (cx, cy), 4, (0, 255, 180), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 0),     1,  cv2.LINE_AA)

        # crosshair
        lc = (0, 230, 255)
        cv2.line(frame, (cx-CROSS_GAP-CROSS_LEN, cy), (cx-CROSS_GAP, cy),        lc, 1, cv2.LINE_AA)
        cv2.line(frame, (cx+CROSS_GAP, cy),            (cx+CROSS_GAP+CROSS_LEN, cy), lc, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy-CROSS_GAP-CROSS_LEN),  (cx, cy-CROSS_GAP),        lc, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy+CROSS_GAP),            (cx, cy+CROSS_GAP+CROSS_LEN), lc, 1, cv2.LINE_AA)

        # corner brackets
        bx1, by1 = cx-BOX_HALF, cy-BOX_HALF
        bx2, by2 = cx+BOX_HALF, cy+BOX_HALF
        bc = (0, 200, 255)
        for ox, oy, dx, dy in [(bx1,by1,1,1),(bx2,by1,-1,1),(bx1,by2,1,-1),(bx2,by2,-1,-1)]:
            cv2.line(frame, (ox, oy), (ox+dx*BRACKET, oy), bc, 2, cv2.LINE_AA)
            cv2.line(frame, (ox, oy), (ox, oy+dy*BRACKET), bc, 2, cv2.LINE_AA)

        cv2.putText(frame, "GAZE", (cx+BOX_HALF+4, cy-BOX_HALF+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 200), 1, cv2.LINE_AA)


# =============================================================================
#  VIRTUAL KEYBOARD  — gaze-dwell / blink operated on-screen keyboard
# =============================================================================
class VirtualKeyboard:
    DWELL_TIME = 1.2
    KEY_W = 72;  KEY_H = 60;  KEY_GAP = 6;  MARGIN = 18;  TEXT_H = 52

    ROWS = [
        list("QWERTYUIOP"),
        list("ASDFGHJKL"),
        list("ZXCVBNM") + ["←"],
        ["SPACE", "ENTER", "CLEAR", "CLOSE"],
    ]

    def __init__(self):
        self.visible        = False
        self.typed_text     = ""
        self.hovered_key    = None
        self.dwell_start    = None
        self.dwell_progress = 0.0
        self.last_pressed   = None
        self.last_press_t   = 0.0
        self._textpad       = None   # set via link_textpad()
        self._build_layout()

    # ── geometry ─────────────────────────────────────────────────────────────
    def _kw(self, label):
        if label == "SPACE":                      return self.KEY_W * 4
        if label in ("ENTER", "CLEAR", "CLOSE"):  return self.KEY_W * 2
        return self.KEY_W

    def _build_layout(self):
        self.key_list = []
        max_row_w = max(sum(self._kw(k)+self.KEY_GAP for k in r)-self.KEY_GAP for r in self.ROWS)
        self._canvas_w = max_row_w + 2*self.MARGIN
        self._canvas_h = self.TEXT_H + self.MARGIN + len(self.ROWS)*(self.KEY_H+self.KEY_GAP) + self.MARGIN + 20
        y = self.TEXT_H + self.MARGIN
        for row in self.ROWS:
            rw = sum(self._kw(k)+self.KEY_GAP for k in row)-self.KEY_GAP
            x  = self.MARGIN + (max_row_w - rw)//2
            for label in row:
                kw = self._kw(label)
                self.key_list.append((label, x, y, x+kw, y+self.KEY_H))
                x += kw + self.KEY_GAP
            y += self.KEY_H + self.KEY_GAP

    @property
    def window_size(self):
        return self._canvas_w, self._canvas_h

    # ── API ───────────────────────────────────────────────────────────────────
    def link_textpad(self, pad):
        """Connect TextPad so ENTER sends text there."""
        self._textpad = pad

    def toggle(self):
        self.visible = not self.visible
        if not self.visible:
            try: cv2.destroyWindow("Virtual Keyboard")
            except: pass

    def close(self):
        self.visible = False
        try: cv2.destroyWindow("Virtual Keyboard")
        except: pass

    def update_gaze(self, kx, ky):
        """Feed gaze in keyboard-window pixels. Returns activated key or None."""
        if not self.visible:
            return None
        hit = next((lbl for lbl,x1,y1,x2,y2 in self.key_list
                    if x1<=kx<x2 and y1<=ky<y2), None)
        if hit != self.hovered_key:
            self.hovered_key    = hit
            self.dwell_start    = time.time() if hit else None
            self.dwell_progress = 0.0
            return None
        if hit and self.dwell_start:
            self.dwell_progress = min(1.0, (time.time()-self.dwell_start)/self.DWELL_TIME)
            if self.dwell_progress >= 1.0 and time.time()-self.last_press_t > self.DWELL_TIME*0.8:
                self._press(hit)
                self.last_press_t   = time.time()
                self.dwell_start    = time.time()
                self.dwell_progress = 0.0
                return hit
        return None

    def blink_press(self):
        """Activate hovered key on left-blink."""
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
        elif label == "CLOSE": self.close()
        elif label == "ENTER":
            text = self.typed_text.strip()
            if text:
                if self._textpad is not None:
                    self._textpad.append(text)   # ← send to TextPad
                elif MOUSE_AVAILABLE:
                    pyautogui.write(text, interval=0.03)
            self.typed_text = ""
        else:
            self.typed_text += label

    # ── render ────────────────────────────────────────────────────────────────
    def render(self):
        if not self.visible:
            return
        img = np.full((self._canvas_h, self._canvas_w, 3), (20, 20, 30), dtype=np.uint8)

        # typed-text bar
        cv2.rectangle(img, (self.MARGIN, 6), (self._canvas_w-self.MARGIN, self.TEXT_H-6), (40,40,60), -1)
        cv2.rectangle(img, (self.MARGIN, 6), (self._canvas_w-self.MARGIN, self.TEXT_H-6), (80,80,120), 1)
        disp = self.typed_text[-48:] if self.typed_text else "▮  start typing…"
        col  = (220,220,255) if self.typed_text else (80,80,100)
        cv2.putText(img, disp, (self.MARGIN+8, self.TEXT_H-16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 1, cv2.LINE_AA)

        now = time.time()
        for label, x1, y1, x2, y2 in self.key_list:
            hov  = label == self.hovered_key
            prs  = label == self.last_pressed and now-self.last_press_t < 0.3
            spec = label in ("←","SPACE","ENTER","CLEAR","CLOSE")
            bg   = (0,200,100) if prs else (60,80,140) if hov else (40,40,70) if spec else (35,35,55)
            cv2.rectangle(img, (x1+2,y1+2), (x2-2,y2-2), bg, -1)
            if hov and self.dwell_progress > 0:
                cx2,cy2 = (x1+x2)//2, (y1+y2)//2
                r = min(x2-x1, y2-y1)//2-4
                cv2.ellipse(img, (cx2,cy2), (r,r), -90, 0, int(360*self.dwell_progress), (0,220,255), 3)
            cv2.rectangle(img, (x1+2,y1+2), (x2-2,y2-2), (0,220,255) if hov else (55,55,80), 1)
            fs   = 0.55 if len(label)>1 else 0.68
            tsz  = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0]
            tx   = x1+(x2-x1-tsz[0])//2;  ty = y1+(y2-y1+tsz[1])//2
            cv2.putText(img, label, (tx,ty), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (10,10,10) if prs else (220,230,255), 1, cv2.LINE_AA)

        cv2.putText(img, "Gaze-dwell or LEFT blink  |  ENTER sends to Text Pad  |  K=close",
                    (self.MARGIN, self._canvas_h-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (70,70,90), 1, cv2.LINE_AA)
        cv2.imshow("Virtual Keyboard", img)


# =============================================================================
#  TEXT PAD  — dedicated notepad window; receives text from keyboard ENTER
# =============================================================================
class TextPad:
    DWELL_TIME = 1.0
    PAD_W = 700;  PAD_H = 500
    MARGIN = 18;  BTN_H = 52;  BTN_GAP = 10
    LINE_H = 28;  FONT_SCALE = 0.60
    FONT   = cv2.FONT_HERSHEY_SIMPLEX

    # Fixed window position on screen — gaze mapping uses these offsets
    PAD_WIN_X = 50    # pixels from left edge of screen
    PAD_WIN_Y = 80    # pixels from top edge of screen

    # (display label, action key, button color BGR)
    BUTTONS = [
        ("SAVE TO .TXT", "save",  (0, 110, 180)),
        ("CLEAR ALL",    "clear", (120, 55, 0)),
        ("CLOSE PAD",    "close", (60, 0, 100)),
    ]

    def __init__(self):
        self.visible      = False
        self.buffer       = ""
        self._hovered     = None   # currently hovered button action key
        self._dwell_start = None
        self._dwell_prog  = 0.0
        self._last_act_t  = 0.0
        self._flash_msg   = ""
        self._flash_t     = 0.0
        self._cooldown    = 1.5
        self._gaze_dot    = None   # (x, y) in pad-window pixels, drawn as indicator

        # pre-compute button rects
        total_w = self.PAD_W - 2*self.MARGIN
        n  = len(self.BUTTONS)
        bw = (total_w - (n-1)*self.BTN_GAP) // n
        by = self.PAD_H - self.MARGIN - self.BTN_H
        self._btn_rects = []   # (action, x1, y1, x2, y2)
        bx = self.MARGIN
        for _, action, _ in self.BUTTONS:
            self._btn_rects.append((action, bx, by, bx+bw, by+self.BTN_H))
            bx += bw + self.BTN_GAP

    # ── text area ─────────────────────────────────────────────────────────────
    @property
    def _text_area(self):
        x1 = self.MARGIN
        y1 = self.MARGIN + 36
        x2 = self.PAD_W - self.MARGIN
        y2 = self.PAD_H - self.MARGIN - self.BTN_H - self.BTN_GAP - 12
        return x1, y1, x2, y2

    # ── public API ────────────────────────────────────────────────────────────
    def toggle(self):
        self.visible = not self.visible
        if not self.visible:
            try: cv2.destroyWindow("Text Pad")
            except: pass

    def close(self):
        self.visible = False
        try: cv2.destroyWindow("Text Pad")
        except: pass

    def append(self, text: str):
        """Add a sentence/word from the virtual keyboard."""
        if self.buffer and not self.buffer.endswith(" "):
            self.buffer += " "
        self.buffer += text
        self.visible = True          # auto-open pad when text arrives
        self._flash("Added to pad!")
        print(f"[TextPad] Received: {text!r}  (total {len(self.buffer)} chars)")

    def save_now(self):
        """Called directly by keyboard shortcut S — always works."""
        self._save()

    def update_gaze(self, screen_x: float, screen_y: float):
        """
        Feed smoothed SCREEN pixel coords.
        Converts to pad-window coords using fixed window position,
        then checks button hit and runs dwell logic.
        Returns triggered action or None.
        """
        if not self.visible:
            return None
        # Convert screen coords → pad-window local coords
        kx = int(screen_x - self.PAD_WIN_X)
        ky = int(screen_y - self.PAD_WIN_Y)
        self._gaze_dot = (kx, ky)   # store for rendering

        hit = next((act for act,x1,y1,x2,y2 in self._btn_rects
                    if x1<=kx<x2 and y1<=ky<y2), None)
        if hit != self._hovered:
            self._hovered     = hit
            self._dwell_start = time.time() if hit else None
            self._dwell_prog  = 0.0
            return None
        if hit and self._dwell_start:
            self._dwell_prog = min(1.0, (time.time()-self._dwell_start)/self.DWELL_TIME)
            if self._dwell_prog >= 1.0 and time.time()-self._last_act_t > self._cooldown:
                self._last_act_t  = time.time()
                self._dwell_start = time.time()
                self._dwell_prog  = 0.0
                self._do(hit)
                return hit
        return None

    def blink_press(self):
        """Trigger hovered button on left blink."""
        if self._hovered and self.visible and time.time()-self._last_act_t > 0.5:
            self._do(self._hovered)
            self._last_act_t = time.time()
            return self._hovered
        return None

    def _do(self, action: str):
        if   action == "save":  self._save()
        elif action == "clear":
            self.buffer = ""
            self._flash("Cleared!")
        elif action == "close": self.close()

    def _save(self):
        if not self.buffer.strip():
            self._flash("Nothing to save!")
            return
        fname = "typed_text_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
        try:
            with open(fname, "w", encoding="utf-8") as f:
                f.write(self.buffer.strip())
            self._flash(f"Saved: {fname}")
            print(f"[TextPad] Saved → {fname}")
        except Exception as e:
            self._flash("Save failed!")
            print(f"[TextPad] Save error: {e}")

    def _flash(self, msg: str):
        self._flash_msg = msg
        self._flash_t   = time.time()

    # ── word-wrap ─────────────────────────────────────────────────────────────
    def _wrap(self, text: str, max_w: int):
        lines, cur = [], ""
        for word in text.split(" "):
            test = (cur + " " + word).strip()
            tw   = cv2.getTextSize(test, self.FONT, self.FONT_SCALE, 1)[0][0]
            if tw <= max_w:
                cur = test
            else:
                if cur: lines.append(cur)
                cur = word
        if cur: lines.append(cur)
        return lines

    # ── render ────────────────────────────────────────────────────────────────
    def render(self):
        if not self.visible:
            return
        img = np.full((self.PAD_H, self.PAD_W, 3), (14, 14, 22), dtype=np.uint8)

        # title bar
        cv2.rectangle(img, (0,0), (self.PAD_W, 34), (25,25,42), -1)
        cv2.putText(img, "Text Pad  —  Gaze-to-Type Buffer   |   S = save now",
                    (self.MARGIN, 24), self.FONT, 0.50, (160,200,255), 1, cv2.LINE_AA)
        cv2.line(img, (0,34), (self.PAD_W,34), (50,50,80), 1)

        # text area
        tx1, ty1, tx2, ty2 = self._text_area
        cv2.rectangle(img, (tx1-4,ty1-4), (tx2+4,ty2+4), (22,22,35), -1)
        cv2.rectangle(img, (tx1-4,ty1-4), (tx2+4,ty2+4), (50,50,80), 1)

        placeholder = "[ Start typing with the virtual keyboard, then press ENTER ]"
        raw = self.buffer if self.buffer else placeholder
        lines = self._wrap(raw, tx2-tx1)
        max_lines = (ty2-ty1)//self.LINE_H
        vis_lines = lines[-max_lines:]
        col = (210,220,255) if self.buffer else (65,65,85)
        for i, line in enumerate(vis_lines):
            cv2.putText(img, line, (tx1, ty1+(i+1)*self.LINE_H),
                        self.FONT, self.FONT_SCALE, col, 1, cv2.LINE_AA)

        # blinking cursor
        if self.buffer and int(time.time()*2)%2 == 0 and vis_lines:
            lw = cv2.getTextSize(vis_lines[-1], self.FONT, self.FONT_SCALE, 1)[0][0]
            cx2 = tx1+lw+4;  cy2 = ty1+len(vis_lines)*self.LINE_H
            lh  = cv2.getTextSize("A", self.FONT, self.FONT_SCALE, 1)[0][1]
            cv2.line(img, (cx2, cy2-lh-2), (cx2, cy2+4), (0,220,255), 2)

        # char / word count
        wc  = len(self.buffer.split()) if self.buffer.strip() else 0
        cv2.putText(img, f"{len(self.buffer)} chars  |  {wc} words",
                    (tx1, ty2+18), self.FONT, 0.40, (80,80,110), 1, cv2.LINE_AA)

        # buttons
        for (disp, action, base_col), (act2, x1, y1, x2, y2) in zip(self.BUTTONS, self._btn_rects):
            hov = self._hovered == action
            bg  = tuple(min(255, int(c*1.6)) for c in base_col) if hov else base_col
            cv2.rectangle(img, (x1,y1), (x2,y2), bg, -1)
            if hov and self._dwell_prog > 0:
                ccx, ccy = (x1+x2)//2, (y1+y2)//2
                r = min(x2-x1, y2-y1)//2-4
                cv2.ellipse(img, (ccx,ccy), (r,r), -90, 0, int(360*self._dwell_prog), (0,255,200), 3)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,220,255) if hov else (60,60,90), 1)
            fs  = 0.48
            tsz = cv2.getTextSize(disp, self.FONT, fs, 1)[0]
            cv2.putText(img, disp, (x1+(x2-x1-tsz[0])//2, y1+(y2-y1+tsz[1])//2),
                        self.FONT, fs, (230,240,255), 1, cv2.LINE_AA)

        # flash message
        now = time.time()
        if now-self._flash_t < 2.5 and self._flash_msg:
            fade = min(1.0, (2.5-(now-self._flash_t))/0.5)
            fc   = tuple(int(c*fade) for c in (0, 255, 160))
            cv2.putText(img, self._flash_msg,
                        (self.MARGIN, self.PAD_H-self.MARGIN-self.BTN_H-24),
                        self.FONT, 0.62, fc, 2, cv2.LINE_AA)

        # ── gaze dot indicator on the pad itself ──────────────────────────────
        if self._gaze_dot:
            gx, gy = self._gaze_dot
            if 0 <= gx < self.PAD_W and 0 <= gy < self.PAD_H:
                cv2.circle(img, (gx, gy), 7, (0, 220, 255), -1, cv2.LINE_AA)
                cv2.circle(img, (gx, gy), 7, (0, 0, 0),     1,  cv2.LINE_AA)
                cv2.circle(img, (gx, gy), 14, (0, 180, 200), 1, cv2.LINE_AA)

        # hint
        cv2.putText(img, "Gaze-dwell or left-blink buttons  |  S=save  T=toggle  ESC=end",
                    (self.MARGIN, self.PAD_H-5), self.FONT, 0.33, (55,55,75), 1, cv2.LINE_AA)

        # pin window to fixed screen position so gaze mapping is accurate
        cv2.imshow("Text Pad", img)
        cv2.moveWindow("Text Pad", self.PAD_WIN_X, self.PAD_WIN_Y)


# =============================================================================
#  ATTENTION TRACKER  v5
# =============================================================================
class AttentionTracker:

    def __init__(self):
        # ── MediaPipe model ───────────────────────────────────────────────── #
        self.model_path = "face_landmarker.task"
        if not os.path.exists(self.model_path):
            print("Downloading Face Landmarker model…")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                self.model_path)
        opts = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        self.detector = vision.FaceLandmarker.create_from_options(opts)

        # ── Camera ────────────────────────────────────────────────────────── #
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Webcam not detected")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps_cam = fps if 5 < fps < 120 else 30

        # ── Screen ────────────────────────────────────────────────────────── #
        self.screen_w, self.screen_h = (pyautogui.size() if MOUSE_AVAILABLE else (1920, 1080))

        # ── Landmark indices ──────────────────────────────────────────────── #
        self.LEFT_IRIS   = [474, 475, 476, 477]
        self.RIGHT_IRIS  = [469, 470, 471, 472]
        self.LEFT_EYE    = [33,  160, 158, 133, 153, 144]
        self.RIGHT_EYE   = [362, 385, 387, 263, 373, 380]
        self.L_EYE_L     = 33;   self.L_EYE_R = 133

        # ── Blink ─────────────────────────────────────────────────────────── #
        self.EAR_TH_L        = 0.23;  self.EAR_TH_R = 0.23
        self.CONSEC_FRAMES   = 3
        self.l_blink_ctr     = 0;  self.r_blink_ctr = 0
        self.l_blink_total   = 0;  self.r_blink_total = 0
        self.CLICK_COOL      = 0.65
        self.last_l_click    = 0.0;  self.last_r_click = 0.0

        # ── Gaze smoothing ────────────────────────────────────────────────── #
        self.GAZE_ALPHA  = 0.15
        self.gaze_sx_sm  = None   # smoothed screen x
        self.gaze_sy_sm  = None   # smoothed screen y
        self.mouse_mode  = True
        self.gaze_calib  = [0.35, 0.65, 0.30, 0.70]  # [xmin,xmax,ymin,ymax]

        # ── Drowsiness ────────────────────────────────────────────────────── #
        self.DROWSY_FRAMES   = int(self.fps_cam * 2.5)
        self.drowsy_ctr      = 0
        self.last_alert_t    = 0.0

        # ── Head pose ─────────────────────────────────────────────────────── #
        self.YAW_TH   = 25;  self.PITCH_TH = 20

        # ── Session data ──────────────────────────────────────────────────── #
        self.session_start   = time.time()
        self.prev_time       = 0.0
        self.rolling_blinks  = deque()
        self.rolling_gaze    = deque()
        self.log_rows        = []

        # ── Display state ─────────────────────────────────────────────────── #
        self.live_attn       = "CALIBRATING"
        self.live_attn_pct   = 0
        self.gaze_dir        = "CENTER"
        self.head_status     = "FORWARD"
        self.attn_history    = deque(maxlen=150)

        # ── Sub-systems ───────────────────────────────────────────────────── #
        self.gaze_cursor = GazeCursor()
        self.vkb         = VirtualKeyboard()
        self.pad         = TextPad()
        self.vkb.link_textpad(self.pad)   # ← ENTER on keyboard → TextPad

        print("\n╔══════════════════════════════════════════════╗")
        print("║   Attention Tracker v5                      ║")
        print("╠══════════════════════════════════════════════╣")
        print("║  G   → Toggle gaze cursor                  ║")
        print("║  K   → Toggle virtual keyboard             ║")
        print("║  T   → Toggle text pad                     ║")
        print("║  S   → Save text pad to .txt NOW           ║")
        print("║  M   → Toggle mouse control                ║")
        print("║  ESC → End session + report                ║")
        print("╚══════════════════════════════════════════════╝\n")

    # =========================================================================
    #  CALIBRATION
    # =========================================================================
    def calibrate(self, duration=3):
        # ── Phase 1: EAR baseline ─────────────────────────────────────────── #
        print(f"[CAL] Phase 1: Eyes open for {duration}s …")
        l_ears, r_ears = [], []
        t_end = time.time() + duration
        while time.time() < t_end:
            ok, frame = self.cap.read()
            if not ok: continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            ov = frame.copy()
            cv2.rectangle(ov, (0,0), (w,h), (20,20,20), -1)
            cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)
            cd = max(1, int(t_end-time.time())+1)
            cv2.putText(frame, "CALIBRATION  —  keep eyes OPEN",
                        (w//2-245, h//2-20), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,220,255), 2)
            cv2.putText(frame, f"{cd}s", (w//2-20, h//2+44),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100,255,100), 2)
            cv2.imshow("Attention Tracking System", frame)
            if cv2.waitKey(1) & 0xFF == 27: return

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res    = self.detector.detect_for_video(mp_img, int(time.time()*1000))
            if res.face_landmarks:
                lm = res.face_landmarks[0]
                le = [(int(lm[i].x*w), int(lm[i].y*h)) for i in self.LEFT_EYE]
                re = [(int(lm[i].x*w), int(lm[i].y*h)) for i in self.RIGHT_EYE]
                l_ears.append(self._ear(le))
                r_ears.append(self._ear(re))

        if l_ears:
            self.EAR_TH_L = round(max(0.10, np.mean(l_ears)-1.5*np.std(l_ears)-0.02), 4)
            self.EAR_TH_R = round(max(0.10, np.mean(r_ears)-1.5*np.std(r_ears)-0.02), 4)
            print(f"[CAL] EAR thresholds  L:{self.EAR_TH_L}  R:{self.EAR_TH_R}")
        else:
            print("[CAL] No face — using default EAR 0.23")

        if not MOUSE_AVAILABLE:
            return

        # ── Phase 2: Gaze corners ─────────────────────────────────────────── #
        print("[CAL] Phase 2: Gaze corner mapping …")
        corners = [("TOP-LEFT",(0.05,0.08)),("TOP-RIGHT",(0.95,0.08)),
                   ("BOTTOM-LEFT",(0.05,0.92)),("BOTTOM-RIGHT",(0.95,0.92))]
        all_gx, all_gy = [], []
        for label, (tx,ty) in corners:
            t_end = time.time() + 2.0
            while time.time() < t_end:
                ok, frame = self.cap.read()
                if not ok: continue
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                ov = frame.copy()
                cv2.rectangle(ov, (0,0), (w,h), (20,20,20), -1)
                cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
                cv2.putText(frame, f"Look at  {label}", (w//2-180,h//2),
                            cv2.FONT_HERSHEY_DUPLEX, 0.95, (0,220,255), 2)
                dx, dy = int(tx*w), int(ty*h)
                cv2.circle(frame, (dx,dy), 18, (0,255,0), -1)
                cv2.circle(frame, (dx,dy), 22, (255,255,255), 2)
                frac = (t_end-time.time())/2.0
                cv2.ellipse(frame, (dx,dy), (30,30), -90, 0, int(360*(1-frac)), (0,220,100), 2)
                cv2.imshow("Attention Tracking System", frame)
                if cv2.waitKey(1) & 0xFF == 27: return

                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res    = self.detector.detect_for_video(mp_img, int(time.time()*1000))
                if res.face_landmarks:
                    lm = res.face_landmarks[0]
                    lgx = (lm[self.LEFT_IRIS[0]].x  + lm[self.LEFT_IRIS[2]].x)  / 2
                    rgx = (lm[self.RIGHT_IRIS[0]].x + lm[self.RIGHT_IRIS[2]].x) / 2
                    lgy = (lm[self.LEFT_IRIS[0]].y  + lm[self.LEFT_IRIS[2]].y)  / 2
                    rgy = (lm[self.RIGHT_IRIS[0]].y + lm[self.RIGHT_IRIS[2]].y) / 2
                    all_gx.append((lgx+rgx)/2)
                    all_gy.append((lgy+rgy)/2)

        if len(all_gx) > 20:
            self.gaze_calib = [
                np.percentile(all_gx,10)-0.02, np.percentile(all_gx,90)+0.02,
                np.percentile(all_gy,10)-0.02, np.percentile(all_gy,90)+0.02,
            ]
            print(f"[CAL] Gaze calib X:{self.gaze_calib[:2]}  Y:{self.gaze_calib[2:]}")
        else:
            print("[CAL] Not enough gaze data — using defaults")

    # =========================================================================
    #  HELPERS
    # =========================================================================
    def _ear(self, pts):
        A = np.linalg.norm(np.array(pts[1])-np.array(pts[5]))
        B = np.linalg.norm(np.array(pts[2])-np.array(pts[4]))
        C = np.linalg.norm(np.array(pts[0])-np.array(pts[3]))
        return (A+B) / (2.0*C+1e-6)

    def _gaze_dir_label(self, lm, w, h):
        lc = lm[self.L_EYE_L].x * w
        rc = lm[self.L_EYE_R].x * w
        ix = (lm[self.LEFT_IRIS[0]].x + lm[self.LEFT_IRIS[2]].x) / 2 * w
        r  = (ix-lc) / (rc-lc+1e-6)
        if   r < 0.37: return "RIGHT ->"
        elif r > 0.63: return "<- LEFT"
        return "CENTER"

    def _gaze_to_screen(self, gx, gy):
        xmin,xmax,ymin,ymax = self.gaze_calib
        sx = np.clip((gx-xmin)/(xmax-xmin+1e-6), 0, 1)
        sy = np.clip((gy-ymin)/(ymax-ymin+1e-6), 0, 1)
        return sx*self.screen_w, sy*self.screen_h

    def _head_pose(self, matrix):
        R = np.array(matrix).reshape(4,4)[:3,:3]
        yaw   = np.degrees(np.arctan2(R[1][0], R[0][0]))
        pitch = np.degrees(np.arctan2(-R[2][0], np.sqrt(R[2][1]**2+R[2][2]**2)))
        if   abs(yaw) > self.YAW_TH:   return f"TURNED {'RIGHT' if yaw>0 else 'LEFT'}"
        elif pitch >  self.PITCH_TH:   return "LOOKING DOWN"
        elif pitch < -self.PITCH_TH:   return "LOOKING UP"
        return "FORWARD"

    def _update_attention(self, ts):
        cutoff = ts - 5.0
        while self.rolling_blinks and self.rolling_blinks[0] < cutoff:
            self.rolling_blinks.popleft()
        while self.rolling_gaze and self.rolling_gaze[0][0] < cutoff:
            self.rolling_gaze.popleft()
        if len(self.rolling_gaze) < 10:
            self.live_attn = "CALIBRATING";  self.live_attn_pct = 0;  return
        blink_rate     = len(self.rolling_blinks) * 12
        gd             = np.array([[r[1],r[2]] for r in self.rolling_gaze])
        variance       = gd[:,0].var() + gd[:,1].var()
        focus_ratio    = ((gd[:,1]>0.25)&(gd[:,1]<0.75)).mean()
        blink_score    = 1.0 if 10<=blink_rate<=22 else max(0, 1-abs(blink_rate-16)/16)
        var_score      = 1.0 / (1+variance*10)
        pct            = min(100, max(0, int((focus_ratio*0.45+var_score*0.30+blink_score*0.25)*100)))
        self.live_attn_pct = pct
        self.live_attn     = "HIGH" if pct>=70 else "MEDIUM" if pct>=40 else "LOW"
        self.attn_history.append(pct)

    # =========================================================================
    #  HUD
    # =========================================================================
    @staticmethod
    def _rrect(img, pt1, pt2, color, alpha=0.55, r=12):
        ov = img.copy()
        x1,y1 = pt1;  x2,y2 = pt2
        cv2.rectangle(ov,(x1+r,y1),(x2-r,y2),color,-1)
        cv2.rectangle(ov,(x1,y1+r),(x2,y2-r),color,-1)
        for cx,cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
            cv2.circle(ov,(cx,cy),r,color,-1)
        cv2.addWeighted(ov,alpha,img,1-alpha,0,img)

    def _draw_hud(self, frame, fps, ts, lear, rear):
        h, w = frame.shape[:2]
        PW = 315;  PH = 360
        self._rrect(frame, (10,10), (10+PW,10+PH), (15,15,15), alpha=0.68)

        ac = {"HIGH":(50,220,80),"MEDIUM":(40,180,255),
              "LOW":(30,30,220),"CALIBRATING":(160,160,160)}.get(self.live_attn,(200,200,200))

        def put(txt, y, sc=0.58, col=(220,220,220), bld=1):
            cv2.putText(frame, txt, (24,y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, bld, cv2.LINE_AA)

        el = int(ts)
        put(f"Time     : {el//60:02d}:{el%60:02d}", 40)
        put(f"FPS      : {int(fps)}", 62)
        put(f"L-Blinks : {self.l_blink_total}", 84)
        put(f"R-Blinks : {self.r_blink_total}", 106)
        put(f"EAR  L:{lear:.3f}  R:{rear:.3f}", 128)
        put(f"Gaze     : {self.gaze_dir}", 150)
        put(f"Head     : {self.head_status}", 172)
        mc = (50,220,80) if self.mouse_mode else (120,120,120)
        put(f"Mouse    : {'ON (M)' if self.mouse_mode else 'OFF (M)'}", 194, col=mc)
        kc = (0,220,255) if self.vkb.visible else (100,100,100)
        put(f"Keyboard : {'ON (K)' if self.vkb.visible else 'OFF (K)'}", 216, col=kc)
        pc = (80,200,120) if self.pad.visible else (100,100,100)
        n_chars = len(self.pad.buffer)
        put(f"Text Pad : {'ON (T)' if self.pad.visible else 'OFF (T)'}"
            + (f"  [{n_chars}ch]" if n_chars else ""), 238, col=pc)
        put(f"Attn     : {self.live_attn} ({self.live_attn_pct}%)", 266, sc=0.66, col=ac, bld=2)

        # progress bar
        bx,by,bw,bh = 10,278,PW,11
        cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(50,50,50),-1)
        cv2.rectangle(frame,(bx,by),(bx+int(bw*self.live_attn_pct/100),by+bh),ac,-1)
        cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(100,100,100),1)

        # sparkline
        if len(self.attn_history) > 2:
            sx,sy,sw,sh = 10,293,PW,46
            self._rrect(frame,(sx,sy),(sx+sw,sy+sh),(15,15,15),alpha=0.5)
            hist = list(self.attn_history);  n = len(hist)
            pts  = [(sx+int(i/(n-1)*sw), sy+sh-int(hist[i]/100*sh)) for i in range(n)]
            for i in range(1,len(pts)):
                cv2.line(frame, pts[i-1], pts[i], ac, 1, cv2.LINE_AA)

        # drowsiness alert
        if self.drowsy_ctr >= self.DROWSY_FRAMES:
            ax,ay = w//2-200,h//2-34
            self._rrect(frame,(ax-10,ay-44),(ax+420,ay+24),(0,0,160),alpha=0.85)
            cv2.putText(frame,"!! DROWSY — WAKE UP !!",(ax,ay),
                        cv2.FONT_HERSHEY_DUPLEX,1.1,(30,80,255),2,cv2.LINE_AA)
            if SOUND_AVAILABLE and time.time()-self.last_alert_t>3:
                winsound.Beep(900,400);  self.last_alert_t=time.time()

        # head warning
        if self.head_status != "FORWARD":
            cv2.putText(frame,f"HEAD: {self.head_status}",(w-295,40),
                        cv2.FONT_HERSHEY_SIMPLEX,0.72,(30,165,255),2,cv2.LINE_AA)

        # click flash
        now = time.time()
        if now-self.last_l_click < 0.35:
            cv2.putText(frame,"< LEFT CLICK",(w//2-120,h-55),
                        cv2.FONT_HERSHEY_DUPLEX,0.95,(0,220,255),2,cv2.LINE_AA)
        elif now-self.last_r_click < 0.35:
            cv2.putText(frame,"RIGHT CLICK >",(w//2-120,h-55),
                        cv2.FONT_HERSHEY_DUPLEX,0.95,(255,140,0),2,cv2.LINE_AA)

        # current typed text preview
        if self.vkb.visible and self.vkb.typed_text:
            cv2.putText(frame,f"Typing: {self.vkb.typed_text[-32:]}",(w//2-230,h-30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.58,(0,220,255),1,cv2.LINE_AA)

        cv2.putText(frame,"G=cursor  K=keyboard  T=textpad  S=save  M=mouse  ESC=end",
                    (w//2-230,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.38,(100,100,100),1,cv2.LINE_AA)

    # =========================================================================
    #  MAIN LOOP
    # =========================================================================
    def run(self):
        while True:
            ok, frame = self.cap.read()
            if not ok: break

            frame     = cv2.flip(frame, 1)
            h, w      = frame.shape[:2]
            rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms     = int(time.time()*1000)
            res       = self.detector.detect_for_video(mp_img, ts_ms)
            ts        = time.time() - self.session_start

            lear = rear = 0.0
            blink_flag  = 0
            face_ok     = bool(res.face_landmarks)

            if face_ok:
                lm = res.face_landmarks[0]

                # iris circles
                for iris, col in [(self.LEFT_IRIS,(0,230,100)),(self.RIGHT_IRIS,(0,230,100))]:
                    pts = np.array([(int(lm[i].x*w),int(lm[i].y*h)) for i in iris])
                    (cx,cy),r = cv2.minEnclosingCircle(pts)
                    cv2.circle(frame,(int(cx),int(cy)),int(r),col,2)

                # EAR
                le   = [(int(lm[i].x*w),int(lm[i].y*h)) for i in self.LEFT_EYE]
                re   = [(int(lm[i].x*w),int(lm[i].y*h)) for i in self.RIGHT_EYE]
                lear = self._ear(le)
                rear = self._ear(re)
                now  = time.time()

                # ── left blink ────────────────────────────────────────────── #
                if lear < self.EAR_TH_L:
                    self.l_blink_ctr += 1;  self.drowsy_ctr += 1
                else:
                    if self.l_blink_ctr >= self.CONSEC_FRAMES:
                        self.l_blink_total += 1;  blink_flag = 1
                        self.rolling_blinks.append(ts)
                        if self.vkb.visible:
                            self.vkb.blink_press()
                        elif self.pad.visible:
                            self.pad.blink_press()
                        elif self.mouse_mode and MOUSE_AVAILABLE and now-self.last_l_click>self.CLICK_COOL:
                            pyautogui.click(button='left')
                            self.last_l_click = now
                    self.l_blink_ctr = 0

                # ── right blink ───────────────────────────────────────────── #
                if rear < self.EAR_TH_R:
                    self.r_blink_ctr += 1;  self.drowsy_ctr += 1
                else:
                    if self.r_blink_ctr >= self.CONSEC_FRAMES:
                        self.r_blink_total += 1;  blink_flag = 1
                        self.rolling_blinks.append(ts)
                        if not self.vkb.visible and not self.pad.visible:
                            if self.mouse_mode and MOUSE_AVAILABLE and now-self.last_r_click>self.CLICK_COOL:
                                pyautogui.click(button='right')
                                self.last_r_click = now
                    self.r_blink_ctr = 0

                # drowsy decay when both eyes open
                if lear >= self.EAR_TH_L and rear >= self.EAR_TH_R:
                    self.drowsy_ctr = max(0, self.drowsy_ctr-1)

                # gaze direction label
                self.gaze_dir = self._gaze_dir_label(lm, w, h)

                # head pose
                if res.facial_transformation_matrixes:
                    self.head_status = self._head_pose(res.facial_transformation_matrixes[0])

                # ── raw gaze (average both irises) ────────────────────────── #
                gx = ((lm[self.LEFT_IRIS[0]].x+lm[self.LEFT_IRIS[2]].x)/2
                    + (lm[self.RIGHT_IRIS[0]].x+lm[self.RIGHT_IRIS[2]].x)/2) / 2
                gy = ((lm[self.LEFT_IRIS[0]].y+lm[self.LEFT_IRIS[2]].y)/2
                    + (lm[self.RIGHT_IRIS[0]].y+lm[self.RIGHT_IRIS[2]].y)/2) / 2

                # ── unified smoothing ─────────────────────────────────────── #
                sx_r, sy_r = self._gaze_to_screen(gx, gy)
                if self.gaze_sx_sm is None:
                    self.gaze_sx_sm, self.gaze_sy_sm = sx_r, sy_r
                else:
                    self.gaze_sx_sm += self.GAZE_ALPHA*(sx_r-self.gaze_sx_sm)
                    self.gaze_sy_sm += self.GAZE_ALPHA*(sy_r-self.gaze_sy_sm)

                # ── mouse movement ────────────────────────────────────────── #
                if MOUSE_AVAILABLE and self.mouse_mode and not self.vkb.visible and not self.pad.visible:
                    pyautogui.moveTo(int(self.gaze_sx_sm), int(self.gaze_sy_sm))

                # ── gaze → virtual keyboard ───────────────────────────────── #
                if self.vkb.visible:
                    kw, kh = self.vkb.window_size
                    kx = int((self.gaze_sx_sm/self.screen_w)*kw)
                    ky = int((self.gaze_sy_sm/self.screen_h)*kh)
                    self.vkb.update_gaze(kx, ky)

                # ── gaze → text pad buttons ───────────────────────────────── #
                if self.pad.visible and not self.vkb.visible:
                    # Pass raw smoothed screen coords — pad converts using PAD_WIN_X/Y
                    self.pad.update_gaze(self.gaze_sx_sm, self.gaze_sy_sm)

                # ── gaze cursor update ────────────────────────────────────── #
                self.gaze_cursor.update(gx, gy)

                # ── attention rolling window ──────────────────────────────── #
                self.rolling_gaze.append((ts, gx, gy))
                self._update_attention(ts)

                self.log_rows.append([
                    round(ts,4), round(gx,5), round(gy,5),
                    blink_flag, self.gaze_dir, self.head_status,
                    round(lear,4), round(rear,4), self.live_attn_pct,
                ])

            else:
                self.drowsy_ctr = max(0, self.drowsy_ctr-1)

            # ── FPS ───────────────────────────────────────────────────────── #
            now2 = time.time()
            fps  = 1.0/(now2-self.prev_time+1e-9)
            self.prev_time = now2

            # ── render ────────────────────────────────────────────────────── #
            if face_ok:
                self.gaze_cursor.draw(frame)
            self._draw_hud(frame, fps, ts, lear, rear)
            cv2.imshow("Attention Tracking System", frame)
            self.vkb.render()
            self.pad.render()

            # ── key handling ──────────────────────────────────────────────── #
            key = cv2.waitKey(1) & 0xFF
            if   key == 27:                  break                         # ESC
            elif key in (ord('m'),ord('M')): self.mouse_mode = not self.mouse_mode
            elif key in (ord('k'),ord('K')): self.vkb.toggle()
            elif key in (ord('t'),ord('T')): self.pad.toggle()
            elif key in (ord('s'),ord('S')): self.pad.save_now()          # S = save
            elif key in (ord('g'),ord('G')): self.gaze_cursor.active = not self.gaze_cursor.active

        self.cap.release()
        cv2.destroyAllWindows()
        self._save_csv()
        self._session_summary()
        self._dashboard()

    # =========================================================================
    #  CSV
    # =========================================================================
    def _save_csv(self):
        with open("gaze_log.csv","w",newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp","gaze_x","gaze_y","blink","gaze_dir",
                        "head_status","ear_left","ear_right","attention_pct"])
            w.writerows(self.log_rows)
        print("[INFO] gaze_log.csv saved.")

    # =========================================================================
    #  SESSION SUMMARY
    # =========================================================================
    def _session_summary(self):
        try:
            data = pd.read_csv("gaze_log.csv")
        except Exception:
            print("No session data."); return
        if data.empty: return
        tt  = data["timestamp"].iloc[-1]
        tb  = int(data["blink"].sum())
        br  = tb/(tt/60) if tt>0 else 0
        gv  = data["gaze_x"].var()+data["gaze_y"].var()
        aa  = data["attention_pct"].mean()
        dp  = len(data[data["head_status"]!="FORWARD"])/len(data)*100
        print("\n╔══════════════════════════════════════════╗")
        print("║         SESSION SUMMARY                  ║")
        print("╠══════════════════════════════════════════╣")
        print(f"║  Duration    : {int(tt//60):02d}m {int(tt%60):02d}s              ║")
        print(f"║  L-Blinks    : {self.l_blink_total:<26}║")
        print(f"║  R-Blinks    : {self.r_blink_total:<26}║")
        print(f"║  Total Blinks: {tb:<26}║")
        print(f"║  Blink Rate  : {br:.2f} /min                ║")
        print(f"║  Gaze Var    : {gv:.4f}                  ║")
        print(f"║  Avg Attn    : {aa:.1f}%                    ║")
        print(f"║  Distracted  : {dp:.1f}%                    ║")
        print("╚══════════════════════════════════════════╝\n")

    # =========================================================================
    #  DASHBOARD
    # =========================================================================
    def _dashboard(self):
        try:
            data = pd.read_csv("gaze_log.csv")
        except Exception:
            return
        if len(data) < 5:
            print("Not enough data for dashboard."); return

        plt.style.use("dark_background")
        fig = plt.figure(figsize=(20,10), facecolor="#0d0d0d")
        fig.suptitle("Attention Tracker v5 — Session Report",
                     fontsize=18, color="#e0e0e0", y=0.97)
        gs  = gridspec.GridSpec(2,4, hspace=0.45, wspace=0.35,
                                left=0.05, right=0.97, top=0.91, bottom=0.08)

        def sax(ax, title):
            ax.set_facecolor("#111111")
            ax.set_title(title, color="#aaaaaa", fontsize=11)
            for sp in ax.spines.values(): sp.set_edgecolor("#333333")
            ax.tick_params(colors="#777777", labelsize=8)

        ax = fig.add_subplot(gs[0,0]); sax(ax,"Gaze Heatmap")
        sns.kdeplot(x=data["gaze_x"],y=data["gaze_y"],fill=True,cmap="inferno",bw_adjust=0.6,ax=ax)
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.invert_yaxis()

        ax = fig.add_subplot(gs[0,1]); sax(ax,"Blink Timeline")
        bk = data[data["blink"]==1]
        ax.vlines(bk["timestamp"],0,1,colors="#00e5ff",linewidth=1.2,alpha=0.85)
        ax.set_yticks([]); ax.set_xlim(data["timestamp"].min(),data["timestamp"].max())

        ax = fig.add_subplot(gs[0,2]); sax(ax,"Attention % Over Time")
        ax.plot(data["timestamp"],data["attention_pct"],color="#39ff14",linewidth=1.2)
        ax.axhline(70,color="#ffcc00",lw=0.8,ls="--",label="High 70%")
        ax.axhline(40,color="#ff4500",lw=0.8,ls="--",label="Low 40%")
        ax.set_ylim(0,105); ax.legend(fontsize=7)

        ax = fig.add_subplot(gs[0,3]); sax(ax,"Gaze Direction")
        dc = data["gaze_dir"].value_counts()
        ax.pie(dc.values,labels=dc.index,autopct="%1.1f%%",startangle=90,
               colors=["#00e5ff","#ff6ec7","#39ff14"][:len(dc)],
               textprops={"color":"#ccc","fontsize":9})

        ax = fig.add_subplot(gs[1,0]); sax(ax,"Left Eye EAR")
        ax.plot(data["timestamp"],data["ear_left"],color="#00e5ff",lw=0.9)
        ax.axhline(self.EAR_TH_L,color="#ff4444",lw=1,ls="--",label=f"TH={self.EAR_TH_L}")
        ax.legend(fontsize=7)

        ax = fig.add_subplot(gs[1,1]); sax(ax,"Right Eye EAR")
        ax.plot(data["timestamp"],data["ear_right"],color="#ffa500",lw=0.9)
        ax.axhline(self.EAR_TH_R,color="#ff4444",lw=1,ls="--",label=f"TH={self.EAR_TH_R}")
        ax.legend(fontsize=7)

        ax = fig.add_subplot(gs[1,2]); sax(ax,"Gaze X Drift")
        ax.plot(data["timestamp"],data["gaze_x"],color="#ff6ec7",lw=0.8)
        ax.axhline(0.5,color="#fff",lw=0.6,ls=":",label="Centre")
        ax.legend(fontsize=7)

        ax = fig.add_subplot(gs[1,3]); sax(ax,"Head Status")
        hc = data["head_status"].value_counts()
        ax.bar(hc.index, hc.values,
               color=["#39ff14" if s=="FORWARD" else "#ff4500" for s in hc.index])
        ax.tick_params(axis='x', rotation=15)

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