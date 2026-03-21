# ============================================================
#  FILE: virtual_keyboard.py
#  Paste this file as-is into your eye_tracker/ folder.
#  No changes needed.
# ============================================================

import cv2
import numpy as np
import time

try:
    import pyautogui
    MOUSE_AVAILABLE = True
except ImportError:
    MOUSE_AVAILABLE = False


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
