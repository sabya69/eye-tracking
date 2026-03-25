"""
quiz_module.py  —  Gaze-Controlled Quiz with UX Behaviour Analysis
===================================================================
Drop this file into the attention-tracker folder.
The AttentionTracker will import and use it when the user presses Q.

Each question has 4 answer zones.  The user selects an answer by:
  • Dwell gaze on a zone for DWELL_TIME seconds  (progress arc shown)
  • OR left-blink while hovering a zone

After the quiz, a detailed UX-behaviour report is saved:
  session_quiz_report.png  +  quiz_results.csv
"""

import cv2
import numpy as np
import time
import csv
import datetime

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False


# ─────────────────────────────────────────────────────────────────────────────
#  QUESTION BANK  (extend freely)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_QUESTIONS = [
    {
        "q": "Which part of the brain is primarily responsible for vision?",
        "opts": ["Frontal Lobe", "Temporal Lobe", "Occipital Lobe", "Cerebellum"],
        "ans": 2,
        "category": "Neuroscience"
    },
    {
        "q": "What does EAR stand for in eye-tracking research?",
        "opts": ["Eye Action Rate", "Eye Aspect Ratio", "Event Action Record", "Eye Area Radius"],
        "ans": 1,
        "category": "Eye Tracking"
    },
    {
        "q": "Which technology does this tracker use for landmark detection?",
        "opts": ["OpenPose", "Dlib", "MediaPipe", "TensorFlow Vision"],
        "ans": 2,
        "category": "Computer Vision"
    },
    {
        "q": "A blink rate of how many per minute is considered normal?",
        "opts": ["5–8 / min", "15–20 / min", "30–35 / min", "40–50 / min"],
        "ans": 1,
        "category": "Physiology"
    },
    {
        "q": "What does a HIGH attention score (≥70%) indicate?",
        "opts": ["Drowsiness", "Distraction", "Focused engagement", "Random gaze"],
        "ans": 2,
        "category": "Attention"
    },
    {
        "q": "Which metric measures how much the gaze position varies?",
        "opts": ["Blink rate", "EAR threshold", "Gaze variance", "Head pitch"],
        "ans": 2,
        "category": "Eye Tracking"
    },
    {
        "q": "What is the primary output file format for session logs?",
        "opts": ["JSON", "XML", "CSV", "TXT"],
        "ans": 2,
        "category": "Software"
    },
    {
        "q": "Which colour channel is used by OpenCV by default?",
        "opts": ["RGB", "HSV", "BGR", "YUV"],
        "ans": 2,
        "category": "Computer Vision"
    },
]


# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR PALETTE  (matches tracker dark theme)
# ─────────────────────────────────────────────────────────────────────────────
C_BG        = (14, 14, 22)
C_PANEL     = (22, 22, 35)
C_BORDER    = (50, 50, 80)
C_ACCENT    = (0, 220, 255)
C_GREEN     = (50, 220, 80)
C_RED       = (30, 30, 220)
C_ORANGE    = (0, 140, 255)
C_WHITE     = (220, 230, 255)
C_DIM       = (80, 80, 110)
C_HOVER     = (60, 80, 140)
C_CORRECT   = (40, 200, 60)
C_WRONG     = (30, 30, 200)
C_GOLD      = (0, 200, 255)
FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_D      = cv2.FONT_HERSHEY_DUPLEX


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _rrect(img, pt1, pt2, color, alpha=0.72, r=10):
    """Semi-transparent rounded rectangle."""
    ov = img.copy()
    x1, y1 = pt1;  x2, y2 = pt2
    cv2.rectangle(ov, (x1+r, y1), (x2-r, y2), color, -1)
    cv2.rectangle(ov, (x1, y1+r), (x2, y2-r), color, -1)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.circle(ov, (cx, cy), r, color, -1)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)


def _put(img, text, pos, scale=0.6, color=C_WHITE, bold=1):
    cv2.putText(img, text, pos, FONT, scale, color, bold, cv2.LINE_AA)


def _centre_text(img, text, cy, scale=0.65, color=C_WHITE, bold=1):
    h_img, w_img = img.shape[:2]
    tw = cv2.getTextSize(text, FONT, scale, bold)[0][0]
    _put(img, text, (w_img//2 - tw//2, cy), scale, color, bold)


# ─────────────────────────────────────────────────────────────────────────────
#  QUIZ ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class QuizModule:
    """
    Overlay quiz driven entirely by gaze + blink.

    Usage (inside AttentionTracker.run()):
        self.quiz = QuizModule()
        # on keypress Q:
        self.quiz.start()
        # each frame:
        result = self.quiz.update(frame, gaze_nx, gaze_ny, blink_occurred)
        # result == "done"  when quiz ends
        # After done:
        self.quiz.save_report()
    """

    DWELL_TIME  = 1.5    # seconds to confirm a selection
    RESULT_HOLD = 2.2    # seconds to show correct/wrong before next question
    ZONES       = 4      # always 4 answer options

    def __init__(self, questions=None):
        self.questions   = questions or DEFAULT_QUESTIONS
        self.active      = False
        self._state      = "idle"   # idle | question | result | summary
        self._q_idx      = 0

        # per-question behaviour data
        self._q_start_t  = 0.0
        self._hovered    = None
        self._dwell_st   = None
        self._dwell_prog = 0.0
        self._selected   = None
        self._result_t   = 0.0

        # gaze fixation tracking per option
        self._fixation   = [0.0] * self.ZONES   # cumulative dwell per option
        self._fix_last   = None
        self._fix_last_t = 0.0

        # recorded results
        self.results     = []   # list of dicts per question

        # layout cache (computed on first frame)
        self._layout     = None
        self._frame_sz   = (0, 0)

        # summary stats (filled after quiz ends)
        self.summary     = {}

    # ── public ───────────────────────────────────────────────────────────────
    def start(self, questions=None):
        if questions:
            self.questions = questions
        self._q_idx      = 0
        self.results     = []
        self._state      = "question"
        self.active      = True
        self._begin_question()
        print(f"\n[QUIZ] Started — {len(self.questions)} questions")

    def stop(self):
        self.active = False
        self._state = "idle"
        print("[QUIZ] Stopped by user.")

    def update(self, frame, gaze_nx: float, gaze_ny: float, blink: bool) -> str:
        """
        Call every frame while quiz is active.
        gaze_nx / gaze_ny: normalised [0,1] gaze coords on the camera frame.
        blink: True if a blink was detected this frame.
        Returns: "active" | "done"
        """
        if not self.active:
            return "done"

        h, w = frame.shape[:2]
        if (w, h) != self._frame_sz:
            self._layout  = self._compute_layout(w, h)
            self._frame_sz = (w, h)

        if   self._state == "question": self._frame_question(frame, gaze_nx, gaze_ny, blink)
        elif self._state == "result":   self._frame_result(frame)
        elif self._state == "summary":  self._frame_summary(frame)
        else:
            self.active = False
            return "done"

        return "active" if self.active else "done"

    def save_report(self, csv_path="quiz_results.csv", png_path="session_quiz_report.png"):
        self._save_csv(csv_path)
        if MATPLOTLIB_OK:
            self._save_png(png_path)
        print(f"[QUIZ] Report saved → {csv_path}  {png_path if MATPLOTLIB_OK else ''}")

    # ── internal state machine ────────────────────────────────────────────────
    def _begin_question(self):
        self._q_start_t  = time.time()
        self._hovered    = None
        self._dwell_st   = None
        self._dwell_prog = 0.0
        self._selected   = None
        self._fixation   = [0.0] * self.ZONES
        self._fix_last   = None
        self._fix_last_t = 0.0

    def _frame_question(self, frame, gnx, gny, blink):
        h, w = frame.shape[:2]
        elapsed = time.time() - self._q_start_t
        q       = self.questions[self._q_idx]

        # dark overlay on webcam
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (8, 8, 14), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        L = self._layout

        # ── header ───────────────────────────────────────────────────────────
        _rrect(frame, (L["hx1"], L["hy1"]), (L["hx2"], L["hy2"]), C_PANEL, alpha=0.85)
        # progress bar
        prog_w = int((L["hx2"]-L["hx1"]) * (self._q_idx / len(self.questions)))
        cv2.rectangle(frame, (L["hx1"], L["hy2"]-4), (L["hx1"]+prog_w, L["hy2"]), C_ACCENT, -1)

        tag  = f"Q {self._q_idx+1} / {len(self.questions)}   [{q['category']}]"
        _put(frame, tag, (L["hx1"]+16, L["hy1"]+24), 0.52, C_ACCENT)
        timer_col = C_GREEN if elapsed < 20 else C_ORANGE if elapsed < 30 else C_RED
        _put(frame, f"{int(elapsed)}s", (L["hx2"]-60, L["hy1"]+24), 0.52, timer_col)

        # question text (word-wrapped)
        self._draw_wrapped(frame, q["q"], L["hx1"]+16, L["hy1"]+52, L["hx2"]-L["hx1"]-32,
                           scale=0.68, color=C_WHITE, line_h=28)

        # ── answer zones ─────────────────────────────────────────────────────
        gaze_px = int(gnx * w)
        gaze_py = int(gny * h)

        hit_zone = None
        for i, zone in enumerate(L["zones"]):
            x1, y1, x2, y2 = zone
            inside = x1 <= gaze_px < x2 and y1 <= gaze_py < y2
            if inside:
                hit_zone = i

        # update fixation accumulator
        now = time.time()
        if self._fix_last == hit_zone and hit_zone is not None:
            self._fixation[hit_zone] += now - self._fix_last_t
        self._fix_last   = hit_zone
        self._fix_last_t = now

        # dwell logic
        if hit_zone != self._hovered:
            self._hovered    = hit_zone
            self._dwell_st   = now if hit_zone is not None else None
            self._dwell_prog = 0.0
        if self._hovered is not None and self._dwell_st:
            self._dwell_prog = min(1.0, (now - self._dwell_st) / self.DWELL_TIME)
            if self._dwell_prog >= 1.0:
                self._confirm_selection(self._hovered)
                return

        # blink-select
        if blink and self._hovered is not None:
            self._confirm_selection(self._hovered)
            return

        # draw answer boxes
        for i, zone in enumerate(L["zones"]):
            x1, y1, x2, y2 = zone
            hov  = (i == self._hovered)
            bg   = C_HOVER if hov else C_PANEL
            bord = C_ACCENT if hov else C_BORDER
            _rrect(frame, (x1, y1), (x2, y2), bg, alpha=0.85)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bord, 1)

            # option letter badge
            letter = ["A", "B", "C", "D"][i]
            badge_col = C_ACCENT if hov else C_DIM
            cv2.circle(frame, (x1+22, (y1+y2)//2), 14, badge_col, -1 if hov else 1)
            _put(frame, letter, (x1+15, (y1+y2)//2+5), 0.50,
                 (10, 10, 10) if hov else C_WHITE, 1)

            # option text
            self._draw_wrapped(frame, q["opts"][i], x1+44, y1+20,
                               x2-x1-54, scale=0.56, color=C_WHITE, line_h=22)

            # dwell progress arc
            if hov and self._dwell_prog > 0:
                cx2, cy2 = (x1+x2)//2, (y1+y2)//2
                r = min(x2-x1, y2-y1)//2 - 6
                cv2.ellipse(frame, (cx2, cy2), (r, r), -90, 0,
                            int(360 * self._dwell_prog), C_ACCENT, 3, cv2.LINE_AA)

            # fixation bar (small, bottom of zone)
            if self._fixation[i] > 0:
                max_fix = max(self._fixation) + 0.01
                fw = int((x2-x1-4) * min(1.0, self._fixation[i]/max_fix))
                cv2.rectangle(frame, (x1+2, y2-5), (x1+2+fw, y2-2), C_ACCENT, -1)

        # ── gaze dot on quiz overlay ──────────────────────────────────────────
        cv2.circle(frame, (gaze_px, gaze_py), 8,  C_ACCENT, -1, cv2.LINE_AA)
        cv2.circle(frame, (gaze_px, gaze_py), 8,  (0, 0, 0), 1,  cv2.LINE_AA)
        cv2.circle(frame, (gaze_px, gaze_py), 16, C_ACCENT,  1,  cv2.LINE_AA)

        # ── hint bar ─────────────────────────────────────────────────────────
        _put(frame, "Gaze-dwell (1.5s) or LEFT-BLINK to select   |   ESC = exit quiz",
             (L["hx1"], h-10), 0.38, C_DIM)

    def _confirm_selection(self, idx):
        q       = self.questions[self._q_idx]
        correct = (idx == q["ans"])
        react_t = time.time() - self._q_start_t

        self.results.append({
            "q_idx":       self._q_idx,
            "question":    q["q"],
            "category":    q["category"],
            "selected":    idx,
            "correct_ans": q["ans"],
            "correct":     correct,
            "react_time":  round(react_t, 2),
            "fixation_A":  round(self._fixation[0], 2),
            "fixation_B":  round(self._fixation[1], 2),
            "fixation_C":  round(self._fixation[2], 2),
            "fixation_D":  round(self._fixation[3], 2),
            "hesitation":  round(sum(self._fixation) - self._fixation[idx], 2),
        })

        print(f"[QUIZ] Q{self._q_idx+1}: {'✓ CORRECT' if correct else '✗ WRONG'} "
              f"(sel={['A','B','C','D'][idx]}, react={react_t:.1f}s)")

        self._selected = idx
        self._result_t = time.time()
        self._state    = "result"

    def _frame_result(self, frame):
        h, w = frame.shape[:2]
        L    = self._layout
        q    = self.questions[self._q_idx]
        sel  = self._selected
        correct = self.results[-1]["correct"]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (8, 8, 14), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        # result panel
        col  = C_CORRECT if correct else C_WRONG
        msg  = "CORRECT!" if correct else "WRONG!"
        _rrect(frame, (L["hx1"], L["hy1"]), (L["hx2"], L["hy1"]+90), col, alpha=0.7)
        _centre_text(frame, msg, L["hy1"]+52, scale=1.4, color=C_WHITE, bold=2)

        # show all options colour-coded
        for i, zone in enumerate(L["zones"]):
            x1, y1, x2, y2 = zone
            if i == q["ans"]:
                bg = (20, 100, 30)
                bd = C_CORRECT
            elif i == sel and not correct:
                bg = (40, 20, 120)
                bd = C_WRONG
            else:
                bg = C_PANEL
                bd = C_BORDER
            _rrect(frame, (x1, y1), (x2, y2), bg, alpha=0.85)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bd, 2)

            letter = ["A", "B", "C", "D"][i]
            cv2.circle(frame, (x1+22, (y1+y2)//2), 14, bd, -1)
            _put(frame, letter, (x1+15, (y1+y2)//2+5), 0.50, (10, 10, 10), 1)
            self._draw_wrapped(frame, q["opts"][i], x1+44, y1+20,
                               x2-x1-54, scale=0.56, color=C_WHITE, line_h=22)

        # behaviour insight strip
        r    = self.results[-1]
        hes  = r["hesitation"]
        rt   = r["react_time"]
        istr = (f"React: {rt:.1f}s   |   "
                f"Hesitation: {hes:.1f}s   |   "
                f"Fixations: A={r['fixation_A']:.1f}s  B={r['fixation_B']:.1f}s  "
                f"C={r['fixation_C']:.1f}s  D={r['fixation_D']:.1f}s")
        _centre_text(frame, istr, h-22, scale=0.42, color=C_ACCENT)

        # countdown ring
        elapsed = time.time() - self._result_t
        frac    = min(1.0, elapsed / self.RESULT_HOLD)
        cx2, cy2 = w - 60, 60
        cv2.ellipse(frame, (cx2, cy2), (28, 28), -90, 0,
                    int(360 * frac), col, 3, cv2.LINE_AA)

        if elapsed >= self.RESULT_HOLD:
            self._q_idx += 1
            if self._q_idx >= len(self.questions):
                self._build_summary()
                self._state = "summary"
                self._result_t = time.time()
            else:
                self._state = "question"
                self._begin_question()

    def _frame_summary(self, frame):
        h, w = frame.shape[:2]
        L    = self._layout
        s    = self.summary

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (8, 8, 14), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

        _rrect(frame, (L["hx1"], 20), (L["hx2"], h-20), C_PANEL, alpha=0.90)

        # title
        _centre_text(frame, "QUIZ COMPLETE", 70, scale=1.2, color=C_ACCENT, bold=2)
        _centre_text(frame, "UX Behaviour Analysis", 100, scale=0.65, color=C_DIM)

        # score arc
        cx2, cy2, R = w//2, 185, 60
        angle = int(360 * s["score_pct"] / 100)
        col   = C_GREEN if s["score_pct"] >= 70 else C_ORANGE if s["score_pct"] >= 40 else C_RED
        cv2.ellipse(frame, (cx2, cy2), (R, R), -90, 0, 360, C_BORDER, 3)
        cv2.ellipse(frame, (cx2, cy2), (R, R), -90, 0, angle, col, 6, cv2.LINE_AA)
        _centre_text(frame, f"{s['score_pct']:.0f}%", cy2+8, scale=1.0, color=col, bold=2)
        _centre_text(frame, f"{s['correct']} / {s['total']} correct", cy2+30, scale=0.52, color=C_WHITE)

        # stats grid
        stats = [
            ("Avg Reaction Time",  f"{s['avg_react']:.1f} s"),
            ("Avg Hesitation",     f"{s['avg_hes']:.1f} s"),
            ("Fastest Answer",     f"{s['min_react']:.1f} s"),
            ("Slowest Answer",     f"{s['max_react']:.1f} s"),
            ("Correct Streak",     str(s['best_streak'])),
            ("Accuracy",           f"{s['score_pct']:.1f} %"),
        ]
        gx1 = L["hx1"] + 20
        gy  = 270
        col_w = (L["hx2"] - L["hx1"] - 40) // 2
        for k, (label, val) in enumerate(stats):
            col_x = gx1 + (k % 2) * col_w
            row_y = gy + (k // 2) * 52
            _rrect(frame, (col_x, row_y), (col_x+col_w-12, row_y+44), (30, 30, 50), alpha=0.75)
            _put(frame, label, (col_x+10, row_y+18), 0.44, C_DIM)
            _put(frame, val,   (col_x+10, row_y+38), 0.62, C_ACCENT, 2)

        # per-question mini results
        bar_y = gy + 3*52 + 20
        bw_each = (L["hx2"] - L["hx1"] - 20) // len(self.questions)
        for i, r in enumerate(self.results):
            bx = L["hx1"] + 10 + i * bw_each
            bc = C_CORRECT if r["correct"] else C_WRONG
            _rrect(frame, (bx, bar_y), (bx+bw_each-6, bar_y+32), bc, alpha=0.70)
            _put(frame, f"Q{i+1}", (bx+4, bar_y+22), 0.45, C_WHITE)

        _centre_text(frame, "Press Q to close  |  Report auto-saved",
                     h-30, scale=0.44, color=C_DIM)

        # auto-close after 8 seconds
        if time.time() - self._result_t > 12.0:
            self.active = False

    # ── layout ───────────────────────────────────────────────────────────────
    def _compute_layout(self, w, h):
        """Compute pixel rects for header + 4 answer zones."""
        margin = 30
        hx1, hx2 = margin, w - margin
        hy1 = 20
        hy2 = hy1 + max(100, h // 5)   # header box, taller if question is long

        zone_top = hy2 + 16
        zone_h   = (h - zone_top - margin - 40) // 2
        zone_w   = (hx2 - hx1 - 12) // 2
        gap      = 12

        zones = []
        for row in range(2):
            for col in range(2):
                x1 = hx1 + col * (zone_w + gap)
                y1 = zone_top + row * (zone_h + gap)
                zones.append((x1, y1, x1+zone_w, y1+zone_h))

        return {"hx1": hx1, "hy1": hy1, "hx2": hx2, "hy2": hy2, "zones": zones}

    # ── text wrap ────────────────────────────────────────────────────────────
    @staticmethod
    def _draw_wrapped(img, text, x, y, max_w, scale=0.6, color=C_WHITE, line_h=24):
        words = text.split()
        line  = ""
        dy    = 0
        for word in words:
            test = (line + " " + word).strip()
            tw   = cv2.getTextSize(test, FONT, scale, 1)[0][0]
            if tw <= max_w:
                line = test
            else:
                if line:
                    _put(img, line, (x, y+dy), scale, color)
                    dy += line_h
                line = word
        if line:
            _put(img, line, (x, y+dy), scale, color)

    # ── summary builder ───────────────────────────────────────────────────────
    def _build_summary(self):
        n       = len(self.results)
        correct = sum(r["correct"] for r in self.results)
        reacts  = [r["react_time"] for r in self.results]
        hes     = [r["hesitation"] for r in self.results]

        streak = best = cur = 0
        for r in self.results:
            if r["correct"]: cur += 1;  best = max(best, cur)
            else:            cur  = 0

        self.summary = {
            "total":       n,
            "correct":     correct,
            "score_pct":   correct / n * 100 if n else 0,
            "avg_react":   sum(reacts) / n if n else 0,
            "min_react":   min(reacts) if reacts else 0,
            "max_react":   max(reacts) if reacts else 0,
            "avg_hes":     sum(hes) / n if n else 0,
            "best_streak": best,
        }

    # ── CSV export ───────────────────────────────────────────────────────────
    def _save_csv(self, path):
        fields = ["q_idx","question","category","selected","correct_ans",
                  "correct","react_time","fixation_A","fixation_B",
                  "fixation_C","fixation_D","hesitation"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(self.results)

    # ── PNG report ───────────────────────────────────────────────────────────
    def _save_png(self, path):
        if not self.results:
            return
        s = self.summary
        n = len(self.results)

        plt.style.use("dark_background")
        fig = plt.figure(figsize=(18, 10), facecolor="#0d0d0d")
        fig.suptitle("Gaze Quiz — UX Behaviour Report",
                     fontsize=16, color="#e0e0e0", y=0.97)
        gs = gridspec.GridSpec(2, 4, hspace=0.50, wspace=0.35,
                               left=0.05, right=0.97, top=0.91, bottom=0.08)

        def sax(ax, title):
            ax.set_facecolor("#111111")
            ax.set_title(title, color="#aaaaaa", fontsize=10)
            for sp in ax.spines.values(): sp.set_edgecolor("#333333")
            ax.tick_params(colors="#777777", labelsize=8)

        labels = [f"Q{r['q_idx']+1}" for r in self.results]
        reacts = [r["react_time"]  for r in self.results]
        hes    = [r["hesitation"]  for r in self.results]
        colors = ["#39ff14" if r["correct"] else "#ff3333" for r in self.results]

        # 1. Accuracy pie
        ax = fig.add_subplot(gs[0, 0]);  sax(ax, "Overall Accuracy")
        ax.pie([s["correct"], s["total"]-s["correct"]],
               labels=["Correct", "Wrong"],
               colors=["#39ff14", "#ff3333"],
               autopct="%1.0f%%", startangle=90,
               textprops={"color": "#ccc", "fontsize": 9})

        # 2. Reaction time bar
        ax = fig.add_subplot(gs[0, 1]);  sax(ax, "Reaction Time per Question (s)")
        ax.bar(labels, reacts, color=colors)
        ax.axhline(s["avg_react"], color="#00e5ff", lw=1.2, ls="--", label=f"avg {s['avg_react']:.1f}s")
        ax.legend(fontsize=7)

        # 3. Hesitation bar
        ax = fig.add_subplot(gs[0, 2]);  sax(ax, "Hesitation (gaze on wrong options) (s)")
        ax.bar(labels, hes, color="#ffa500")

        # 4. Cumulative fixation per option (stacked)
        ax = fig.add_subplot(gs[0, 3]);  sax(ax, "Total Gaze Fixation by Option")
        tots = [sum(r[f"fixation_{o}"] for r in self.results) for o in ["A","B","C","D"]]
        ax.bar(["A","B","C","D"], tots, color=["#00e5ff","#ff6ec7","#39ff14","#ffa500"])

        # 5. Category accuracy
        ax = fig.add_subplot(gs[1, 0]);  sax(ax, "Accuracy by Category")
        cats  = {}
        for r in self.results:
            cats.setdefault(r["category"], []).append(int(r["correct"]))
        cat_names = list(cats.keys())
        cat_acc   = [sum(v)/len(v)*100 for v in cats.values()]
        ax.barh(cat_names, cat_acc, color="#00e5ff")
        ax.set_xlim(0, 100)
        ax.axvline(50, color="#ff4500", lw=0.8, ls="--")

        # 6. React vs Hesitation scatter
        ax = fig.add_subplot(gs[1, 1]);  sax(ax, "React Time vs Hesitation")
        for r in self.results:
            c = "#39ff14" if r["correct"] else "#ff3333"
            ax.scatter(r["react_time"], r["hesitation"], color=c, s=60, alpha=0.9)
        ax.set_xlabel("Reaction Time (s)", color="#777")
        ax.set_ylabel("Hesitation (s)",    color="#777")

        # 7. Per-question correct strip
        ax = fig.add_subplot(gs[1, 2]);  sax(ax, "Per-Question Result")
        for i, r in enumerate(self.results):
            col2 = "#39ff14" if r["correct"] else "#ff3333"
            ax.bar(i, 1, color=col2)
            ax.text(i, 0.5, "✓" if r["correct"] else "✗",
                    ha="center", va="center", fontsize=14, color="white")
        ax.set_xticks(range(n));  ax.set_xticklabels(labels)
        ax.set_yticks([])

        # 8. Summary text panel
        ax = fig.add_subplot(gs[1, 3]);  sax(ax, "Session Summary")
        ax.axis("off")
        lines = [
            f"Score       : {s['correct']} / {s['total']}  ({s['score_pct']:.1f}%)",
            f"Avg React   : {s['avg_react']:.2f} s",
            f"Avg Hesit.  : {s['avg_hes']:.2f} s",
            f"Fastest     : {s['min_react']:.2f} s",
            f"Slowest     : {s['max_react']:.2f} s",
            f"Best Streak : {s['best_streak']}",
        ]
        for k, line in enumerate(lines):
            ax.text(0.05, 0.88 - k*0.14, line, transform=ax.transAxes,
                    color="#c0c8ff", fontsize=10, family="monospace")

        plt.savefig(path, dpi=130, bbox_inches="tight", facecolor="#0d0d0d")
        plt.close(fig)
