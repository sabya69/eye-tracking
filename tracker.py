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

# ── Import the 3 helper modules from this same folder ─────────────────────── #
from gaze_cursor      import GazeCursor
from virtual_keyboard import VirtualKeyboard
from text_pad         import TextPad


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
        self.GAZE_ALPHA      = 0.06   # lower = smoother cursor (was 0.15)
        self.gaze_sx_sm      = None
        self.gaze_sy_sm      = None
        self.mouse_mode      = True
        self.gaze_calib      = [0.35, 0.65, 0.30, 0.70]
        self.MOUSE_SPEED_CAP = 40     # max pixels to move per frame
        self.MOUSE_DEADZONE  = 4      # ignore jitter under this many pixels

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

        # ── Sub-systems (imported from other files) ────────────────────────── #
        self.gaze_cursor = GazeCursor()
        self.vkb         = VirtualKeyboard()
        self.pad         = TextPad()
        self.vkb.link_textpad(self.pad)   # ENTER on keyboard → TextPad

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

                # left blink
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

                # right blink
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

                self.gaze_dir = self._gaze_dir_label(lm, w, h)

                if res.facial_transformation_matrixes:
                    self.head_status = self._head_pose(res.facial_transformation_matrixes[0])

                # raw gaze — average both irises
                gx = ((lm[self.LEFT_IRIS[0]].x+lm[self.LEFT_IRIS[2]].x)/2
                    + (lm[self.RIGHT_IRIS[0]].x+lm[self.RIGHT_IRIS[2]].x)/2) / 2
                gy = ((lm[self.LEFT_IRIS[0]].y+lm[self.LEFT_IRIS[2]].y)/2
                    + (lm[self.RIGHT_IRIS[0]].y+lm[self.RIGHT_IRIS[2]].y)/2) / 2

                # unified smoothing
                sx_r, sy_r = self._gaze_to_screen(gx, gy)
                if self.gaze_sx_sm is None:
                    self.gaze_sx_sm, self.gaze_sy_sm = sx_r, sy_r
                else:
                    self.gaze_sx_sm += self.GAZE_ALPHA*(sx_r-self.gaze_sx_sm)
                    self.gaze_sy_sm += self.GAZE_ALPHA*(sy_r-self.gaze_sy_sm)

                # ── mouse movement (speed-capped + deadzone) ──────────────── #
                if MOUSE_AVAILABLE and self.mouse_mode and not self.vkb.visible and not self.pad.visible:
                    tx = int(self.gaze_sx_sm)
                    ty = int(self.gaze_sy_sm)
                    cx, cy = pyautogui.position()
                    dx = tx - cx
                    dy = ty - cy
                    dist = (dx**2 + dy**2) ** 0.5
                    if dist > self.MOUSE_DEADZONE:
                        if dist > self.MOUSE_SPEED_CAP:
                            scale = self.MOUSE_SPEED_CAP / dist
                            dx = int(dx * scale)
                            dy = int(dy * scale)
                        pyautogui.moveRel(dx, dy)

                # gaze → virtual keyboard
                if self.vkb.visible:
                    kw, kh = self.vkb.window_size
                    kx = int((self.gaze_sx_sm/self.screen_w)*kw)
                    ky = int((self.gaze_sy_sm/self.screen_h)*kh)
                    self.vkb.update_gaze(kx, ky)

                # gaze → text pad buttons
                if self.pad.visible and not self.vkb.visible:
                    self.pad.update_gaze(self.gaze_sx_sm, self.gaze_sy_sm)

                # gaze cursor overlay
                self.gaze_cursor.update(gx, gy)

                # attention rolling window
                self.rolling_gaze.append((ts, gx, gy))
                self._update_attention(ts)

                self.log_rows.append([
                    round(ts,4), round(gx,5), round(gy,5),
                    blink_flag, self.gaze_dir, self.head_status,
                    round(lear,4), round(rear,4), self.live_attn_pct,
                ])

            else:
                self.drowsy_ctr = max(0, self.drowsy_ctr-1)

            # FPS
            now2 = time.time()
            fps  = 1.0/(now2-self.prev_time+1e-9)
            self.prev_time = now2

            # render
            if face_ok:
                self.gaze_cursor.draw(frame)
            self._draw_hud(frame, fps, ts, lear, rear)
            cv2.imshow("Attention Tracking System", frame)
            self.vkb.render()
            self.pad.render()

            # key handling
            key = cv2.waitKey(1) & 0xFF
            if   key == 27:                  break
            elif key in (ord('m'),ord('M')): self.mouse_mode = not self.mouse_mode
            elif key in (ord('k'),ord('K')): self.vkb.toggle()
            elif key in (ord('t'),ord('T')): self.pad.toggle()
            elif key in (ord('s'),ord('S')): self.pad.save_now()
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
