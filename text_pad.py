
import cv2
import numpy as np
import time
import datetime


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
        self._hovered     = None
        self._dwell_start = None
        self._dwell_prog  = 0.0
        self._last_act_t  = 0.0
        self._flash_msg   = ""
        self._flash_t     = 0.0
        self._cooldown    = 1.5
        self._gaze_dot    = None   # (x, y) in pad-window pixels, drawn as indicator

        # pre-compute button rects  (action, x1, y1, x2, y2)
        total_w = self.PAD_W - 2*self.MARGIN
        n  = len(self.BUTTONS)
        bw = (total_w - (n-1)*self.BTN_GAP) // n
        by = self.PAD_H - self.MARGIN - self.BTN_H
        self._btn_rects = []
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
        kx = int(screen_x - self.PAD_WIN_X)
        ky = int(screen_y - self.PAD_WIN_Y)
        self._gaze_dot = (kx, ky)

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

        # gaze dot indicator
        if self._gaze_dot:
            gx, gy = self._gaze_dot
            if 0 <= gx < self.PAD_W and 0 <= gy < self.PAD_H:
                cv2.circle(img, (gx, gy), 7,  (0, 220, 255), -1, cv2.LINE_AA)
                cv2.circle(img, (gx, gy), 7,  (0, 0, 0),      1, cv2.LINE_AA)
                cv2.circle(img, (gx, gy), 14, (0, 180, 200),   1, cv2.LINE_AA)

        # hint
        cv2.putText(img, "Gaze-dwell or left-blink buttons  |  S=save  T=toggle  ESC=end",
                    (self.MARGIN, self.PAD_H-5), self.FONT, 0.33, (55,55,75), 1, cv2.LINE_AA)

        # pin window to fixed screen position so gaze mapping is accurate
        cv2.imshow("Text Pad", img)
        cv2.moveWindow("Text Pad", self.PAD_WIN_X, self.PAD_WIN_Y)
