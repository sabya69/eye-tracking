import cv2
import numpy as np


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
        cv2.line(frame, (cx-CROSS_GAP-CROSS_LEN, cy), (cx-CROSS_GAP, cy),            lc, 1, cv2.LINE_AA)
        cv2.line(frame, (cx+CROSS_GAP, cy),            (cx+CROSS_GAP+CROSS_LEN, cy), lc, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy-CROSS_GAP-CROSS_LEN),  (cx, cy-CROSS_GAP),           lc, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy+CROSS_GAP),            (cx, cy+CROSS_GAP+CROSS_LEN), lc, 1, cv2.LINE_AA)

        # corner brackets
        bx1, by1 = cx - BOX_HALF, cy - BOX_HALF
        bx2, by2 = cx + BOX_HALF, cy + BOX_HALF
        bc = (0, 200, 255)
        for ox, oy, dx, dy in [(bx1,by1,1,1),(bx2,by1,-1,1),(bx1,by2,1,-1),(bx2,by2,-1,-1)]:
            cv2.line(frame, (ox, oy), (ox+dx*BRACKET, oy), bc, 2, cv2.LINE_AA)
            cv2.line(frame, (ox, oy), (ox, oy+dy*BRACKET), bc, 2, cv2.LINE_AA)

        cv2.putText(frame, "GAZE", (cx+BOX_HALF+4, cy-BOX_HALF+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 200), 1, cv2.LINE_AA)
