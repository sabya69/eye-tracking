"""
GazeOS Launcher
===============
Run this instead of main.py:
    python launcher.py

A simple window appears with 4 buttons:
  1. Start Eye Tracker  → runs calibration + opens the CV2 tracker window
                          (which already has the keyboard K and text pad T built in)
  2. Notepad            → standalone tkinter text editor
  3. Snake Game         → playable with keyboard arrows
  4. Word Shortcuts     → searchable cheatsheet

All files (tracker.py, gaze_cursor.py, virtual_keyboard.py, text_pad.py,
face_landmarker.task) must be in the SAME folder as this file.
"""

import tkinter as tk
from tkinter import font as tkfont, filedialog
import threading, subprocess, sys, os, datetime, random

# ── colours ──────────────────────────────────────────────────────────────────
BG     = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
CYAN   = "#00e5ff"
GREEN  = "#3fb950"
AMBER  = "#e3b341"
PINK   = "#f778ba"
WHITE  = "#e6edf3"
MUTED  = "#8b949e"
DARK   = "#010409"

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LAUNCHER WINDOW
# ─────────────────────────────────────────────────────────────────────────────
class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GazeOS  —  Attention Tracker")
        self.configure(bg=BG)
        self.resizable(False, False)
        self._tracker_proc = None   # subprocess for the tracker
        self._build()
        self._center()

    # ── layout ───────────────────────────────────────────────────────────────
    def _build(self):
        # ── top bar ──────────────────────────────────────────────────────────
        top = tk.Frame(self, bg=PANEL, pady=14, padx=20)
        top.pack(fill="x")

        # eye icon (drawn with canvas)
        eye_c = tk.Canvas(top, width=36, height=36, bg=PANEL, highlightthickness=0)
        eye_c.pack(side="left")
        eye_c.create_oval(3, 10, 33, 26, outline=CYAN, width=2)
        eye_c.create_oval(14, 13, 22, 23, fill=CYAN, outline="")
        eye_c.create_oval(16, 15, 20, 21, fill=DARK, outline="")

        tk.Label(top, text="  GazeOS", fg=CYAN, bg=PANEL,
                 font=tkfont.Font(family="Consolas", size=20, weight="bold")
                 ).pack(side="left")
        tk.Label(top, text="  v5", fg=MUTED, bg=PANEL,
                 font=tkfont.Font(family="Consolas", size=11)
                 ).pack(side="left", pady=6)

        # clock
        self._clock_v = tk.StringVar()
        tk.Label(top, textvariable=self._clock_v, fg=MUTED, bg=PANEL,
                 font=tkfont.Font(family="Consolas", size=10)
                 ).pack(side="right")
        self._tick()

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # ── subtitle ─────────────────────────────────────────────────────────
        tk.Label(self, text="Click a button to open a module",
                 fg=MUTED, bg=BG,
                 font=tkfont.Font(family="Consolas", size=9)
                 ).pack(pady=(14, 4))

        # ── 4 big buttons ────────────────────────────────────────────────────
        grid = tk.Frame(self, bg=BG, padx=28, pady=8)
        grid.pack()

        btn_data = [
            ("👁  Start Eye Tracker",
             "Calibrate + open webcam\nKeyboard (K) · TextPad (T) inside",
             CYAN,  self._open_tracker),

            ("📝  Notepad",
             "Text editor · save / open\nLeft-blink saves (when tracker on)",
             GREEN, self._open_notepad),

            ("🎮  Snake Game",
             "Arrow keys to play\nLeft-blink = up  (when tracker on)",
             AMBER, self._open_game),

            ("📋  Word Shortcuts",
             "50+ MS Word hotkeys\nSearchable cheatsheet",
             PINK,  self._open_shortcuts),
        ]

        for i, (title, sub, color, cmd) in enumerate(btn_data):
            r, c = divmod(i, 2)
            tile = _Tile(grid, title, sub, color, cmd)
            tile.grid(row=r, column=c, padx=10, pady=10)

        # ── tracker status bar ───────────────────────────────────────────────
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", pady=(8, 0))
        status_row = tk.Frame(self, bg=PANEL, pady=6, padx=20)
        status_row.pack(fill="x")

        self._status_dot = tk.Label(status_row, text="●", fg=MUTED, bg=PANEL,
                                    font=tkfont.Font(family="Consolas", size=10))
        self._status_dot.pack(side="left")
        self._status_v = tk.StringVar(value="  Tracker not running")
        tk.Label(status_row, textvariable=self._status_v,
                 fg=MUTED, bg=PANEL,
                 font=tkfont.Font(family="Consolas", size=9)
                 ).pack(side="left")

        tk.Label(status_row,
                 text="Sabyasachi Das Biswas · CV Mini Project 2026",
                 fg=MUTED, bg=PANEL,
                 font=tkfont.Font(family="Consolas", size=8)
                 ).pack(side="right")

    # ── clock ─────────────────────────────────────────────────────────────────
    def _tick(self):
        self._clock_v.set(datetime.datetime.now().strftime("%H:%M:%S"))
        self.after(1000, self._tick)

    # ── centre on screen ──────────────────────────────────────────────────────
    def _center(self):
        self.update_idletasks()
        w = self.winfo_width();  h = self.winfo_height()
        sw = self.winfo_screenwidth();  sh = self.winfo_screenheight()
        self.geometry(f"+{(sw - w)//2}+{(sh - h)//2}")

    # ── button actions ────────────────────────────────────────────────────────
    def _open_tracker(self):
        """Launch tracker.py in a subprocess so it gets its own CV2 windows."""
        if self._tracker_proc and self._tracker_proc.poll() is None:
            # already running
            self._flash("Tracker already running! Use K / T in CV2 window.")
            return
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "tracker_run.py")
        # write a tiny runner so we don't depend on main.py name
        runner = (
            "from tracker import AttentionTracker\n"
            "t = AttentionTracker()\n"
            "t.calibrate(duration=3)\n"
            "t.run()\n"
        )
        with open(script, "w") as f:
            f.write(runner)

        self._tracker_proc = subprocess.Popen(
            [sys.executable, script],
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        self._status_dot.config(fg=GREEN)
        self._status_v.set("  Tracker running  —  K=keyboard  T=textpad  ESC=stop")
        self._poll_tracker()

    def _poll_tracker(self):
        """Check every second if the tracker subprocess is still alive."""
        if self._tracker_proc and self._tracker_proc.poll() is not None:
            self._status_dot.config(fg=MUTED)
            self._status_v.set("  Tracker stopped")
        else:
            self.after(1000, self._poll_tracker)

    def _open_notepad(self):
        NotepadWindow(self)

    def _open_game(self):
        GameWindow(self)

    def _open_shortcuts(self):
        ShortcutsWindow(self)

    # ── flash message in status bar ──────────────────────────────────────────
    def _flash(self, msg):
        self._status_v.set(f"  {msg}")
        self.after(3000, lambda: self._status_v.set(
            "  Tracker running" if (self._tracker_proc and
                                    self._tracker_proc.poll() is None)
            else "  Tracker not running"))


# ─────────────────────────────────────────────────────────────────────────────
#  TILE WIDGET  (the big buttons)
# ─────────────────────────────────────────────────────────────────────────────
class _Tile(tk.Frame):
    W = 240;  H = 120

    def __init__(self, parent, title, subtitle, color, command):
        super().__init__(parent, bg=PANEL, width=self.W, height=self.H,
                         highlightbackground=BORDER, highlightthickness=1,
                         cursor="hand2")
        self.pack_propagate(False)
        self._color   = color
        self._command = command

        # coloured top stripe
        tk.Frame(self, bg=color, height=3).pack(fill="x")

        # title
        self._title_lbl = tk.Label(
            self, text=title, fg=WHITE, bg=PANEL,
            font=tkfont.Font(family="Consolas", size=12, weight="bold"),
            anchor="w", padx=14)
        self._title_lbl.pack(fill="x", pady=(8, 2))

        # subtitle
        self._sub_lbl = tk.Label(
            self, text=subtitle, fg=MUTED, bg=PANEL,
            font=tkfont.Font(family="Consolas", size=8),
            anchor="w", padx=14, justify="left")
        self._sub_lbl.pack(fill="x")

        # bind hover + click on every child too
        for w in (self, self._title_lbl, self._sub_lbl):
            w.bind("<Enter>",    self._hover_on)
            w.bind("<Leave>",    self._hover_off)
            w.bind("<Button-1>", lambda e: self._command())

    def _hover_on(self, _=None):
        self.configure(bg=DARK, highlightbackground=self._color)
        self._title_lbl.configure(bg=DARK, fg=self._color)
        self._sub_lbl.configure(bg=DARK)

    def _hover_off(self, _=None):
        self.configure(bg=PANEL, highlightbackground=BORDER)
        self._title_lbl.configure(bg=PANEL, fg=WHITE)
        self._sub_lbl.configure(bg=PANEL)


# ─────────────────────────────────────────────────────────────────────────────
#  NOTEPAD  (standalone tkinter text editor)
# ─────────────────────────────────────────────────────────────────────────────
class NotepadWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Notepad")
        self.configure(bg=BG)
        self.geometry("700x500")
        self._filepath = None
        self._build()
        self._center()

    def _build(self):
        mf = tkfont.Font(family="Consolas", size=9)
        bf = tkfont.Font(family="Consolas", size=9, weight="bold")

        # ── menu bar ─────────────────────────────────────────────────────────
        menu = tk.Menu(self, bg=PANEL, fg=WHITE,
                       activebackground=CYAN, activeforeground=BG, tearoff=0)
        self.config(menu=menu)

        fm = tk.Menu(menu, bg=PANEL, fg=WHITE,
                     activebackground=CYAN, activeforeground=BG, tearoff=0)
        menu.add_cascade(label="File", menu=fm)
        fm.add_command(label="New          Ctrl+N", command=self._new)
        fm.add_command(label="Open...      Ctrl+O", command=self._open_file)
        fm.add_command(label="Save         Ctrl+S", command=self._save)
        fm.add_command(label="Save As...   Ctrl+Shift+S", command=self._save_as)
        fm.add_separator()
        fm.add_command(label="Exit", command=self.destroy)

        em = tk.Menu(menu, bg=PANEL, fg=WHITE,
                     activebackground=CYAN, activeforeground=BG, tearoff=0)
        menu.add_cascade(label="Edit", menu=em)
        em.add_command(label="Cut     Ctrl+X",
                       command=lambda: self._txt.event_generate("<<Cut>>"))
        em.add_command(label="Copy    Ctrl+C",
                       command=lambda: self._txt.event_generate("<<Copy>>"))
        em.add_command(label="Paste   Ctrl+V",
                       command=lambda: self._txt.event_generate("<<Paste>>"))
        em.add_command(label="Select All  Ctrl+A",
                       command=lambda: self._txt.tag_add("sel", "1.0", "end"))
        em.add_separator()
        em.add_command(label="Undo  Ctrl+Z",
                       command=lambda: self._txt.edit_undo())
        em.add_command(label="Redo  Ctrl+Y",
                       command=lambda: self._txt.edit_redo())

        # ── toolbar ──────────────────────────────────────────────────────────
        tb = tk.Frame(self, bg=PANEL, pady=5)
        tb.pack(fill="x")

        for label, cmd, color in [
            ("NEW",     self._new,       WHITE),
            ("OPEN",    self._open_file, WHITE),
            ("SAVE",    self._save,      GREEN),
            ("SAVE AS", self._save_as,   CYAN),
        ]:
            b = tk.Button(tb, text=label, command=cmd,
                          bg=BG, fg=color,
                          activebackground=color, activeforeground=BG,
                          relief="flat", bd=0, font=bf,
                          padx=10, pady=4, cursor="hand2")
            b.pack(side="left", padx=3)

        # font size spinner
        tk.Label(tb, text="│", fg=BORDER, bg=PANEL,
                 font=mf).pack(side="left", padx=4)
        tk.Label(tb, text="Size:", fg=MUTED, bg=PANEL,
                 font=mf).pack(side="left")
        self._fsize = tk.IntVar(value=12)
        tk.Spinbox(tb, from_=8, to=36, textvariable=self._fsize,
                   width=3, bg=BG, fg=WHITE, buttonbackground=BG,
                   font=mf, command=self._change_font,
                   relief="flat").pack(side="left", padx=4)

        # word wrap toggle
        self._wrap_v = tk.BooleanVar(value=True)
        tk.Checkbutton(tb, text="Wrap", variable=self._wrap_v,
                       command=self._toggle_wrap,
                       bg=PANEL, fg=MUTED, selectcolor=BG,
                       activebackground=PANEL, font=mf).pack(side="left", padx=4)

        # status
        self._status_v = tk.StringVar(value="ready")
        tk.Label(tb, textvariable=self._status_v, fg=MUTED, bg=PANEL,
                 font=mf).pack(side="right", padx=10)

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # ── text area ────────────────────────────────────────────────────────
        frame = tk.Frame(self, bg=BG)
        frame.pack(fill="both", expand=True)

        self._tfont = tkfont.Font(family="Consolas", size=12)
        self._txt = tk.Text(
            frame,
            bg=PANEL, fg=WHITE,
            insertbackground=CYAN,
            selectbackground=BORDER, selectforeground=CYAN,
            font=self._tfont, wrap="word", undo=True,
            padx=14, pady=12, relief="flat", bd=0,
        )
        vsb = tk.Scrollbar(frame, command=self._txt.yview,
                           bg=PANEL, troughcolor=BG,
                           activebackground=CYAN, relief="flat")
        self._txt.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._txt.pack(fill="both", expand=True)
        self._txt.bind("<KeyRelease>", self._update_status)

        # keyboard shortcuts
        self.bind("<Control-s>",       lambda e: self._save())
        self.bind("<Control-n>",       lambda e: self._new())
        self.bind("<Control-o>",       lambda e: self._open_file())
        self.bind("<Control-S>",       lambda e: self._save_as())

    # ── actions ──────────────────────────────────────────────────────────────
    def insert_text(self, text):
        self._txt.insert("end", text)
        self._update_status()
        self.lift()

    def _new(self):
        self._txt.delete("1.0", "end")
        self._filepath = None
        self.title("Notepad — untitled")
        self._status_v.set("new file")

    def _open_file(self):
        p = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not p:
            return
        with open(p, encoding="utf-8") as f:
            self._txt.delete("1.0", "end")
            self._txt.insert("1.0", f.read())
        self._filepath = p
        self.title(f"Notepad — {os.path.basename(p)}")
        self._update_status()

    def _save(self):
        if not self._filepath:
            self._save_as()
            return
        content = self._txt.get("1.0", "end").rstrip()
        with open(self._filepath, "w", encoding="utf-8") as f:
            f.write(content)
        self._status_v.set(f"saved  ✓  {os.path.basename(self._filepath)}")

    def _save_as(self):
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        p   = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=f"typed_text_{ts}.txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not p:
            return
        self._filepath = p
        self._save()
        self.title(f"Notepad — {os.path.basename(p)}")

    def _toggle_wrap(self):
        self._txt.configure(wrap="word" if self._wrap_v.get() else "none")

    def _change_font(self):
        self._tfont.configure(size=self._fsize.get())

    def _update_status(self, _=None):
        content = self._txt.get("1.0", "end").strip()
        words   = len(content.split()) if content else 0
        self._status_v.set(f"{len(content)} chars  ·  {words} words")

    def _center(self):
        self.update_idletasks()
        w = self.winfo_width();  h = self.winfo_height()
        sw = self.winfo_screenwidth();  sh = self.winfo_screenheight()
        self.geometry(f"+{(sw - w)//2}+{(sh - h)//2}")


# ─────────────────────────────────────────────────────────────────────────────
#  SNAKE GAME
# ─────────────────────────────────────────────────────────────────────────────
class GameWindow(tk.Toplevel):
    CELL = 22;  COLS = 24;  ROWS = 18

    def __init__(self, master):
        super().__init__(master)
        self.title("Snake")
        self.configure(bg=BG)
        self.resizable(False, False)
        self._hi    = 0
        self._build()
        self._center()
        self._reset()

    def _build(self):
        W = self.COLS * self.CELL
        H = self.ROWS * self.CELL

        # top bar
        top = tk.Frame(self, bg=PANEL, pady=6)
        top.pack(fill="x")
        self._score_v = tk.StringVar(value="SCORE: 0")
        self._hi_v    = tk.StringVar(value="BEST: 0")
        tk.Label(top, textvariable=self._score_v, fg=AMBER, bg=PANEL,
                 font=tkfont.Font(family="Consolas", size=12, weight="bold")
                 ).pack(side="left", padx=14)
        tk.Label(top, textvariable=self._hi_v, fg=MUTED, bg=PANEL,
                 font=tkfont.Font(family="Consolas", size=12)
                 ).pack(side="right", padx=14)

        # canvas
        self._cv = tk.Canvas(self, width=W, height=H, bg=DARK,
                             highlightthickness=1,
                             highlightbackground=BORDER)
        self._cv.pack(padx=12, pady=6)

        # hint
        tk.Label(self,
                 text="Arrow keys / WASD to move  ·  Click canvas to start / restart",
                 fg=MUTED, bg=BG,
                 font=tkfont.Font(family="Consolas", size=8)
                 ).pack(pady=(0, 8))

        # bindings
        for key, d in [("<Up>",(0,-1)),("<Down>",(0,1)),
                        ("<Left>",(-1,0)),("<Right>",(1,0)),
                        ("<w>",(0,-1)),("<s>",(0,1)),
                        ("<a>",(-1,0)),("<d>",(1,0))]:
            self.bind(key, lambda e, dv=d: self._steer(*dv))
        self._cv.bind("<Button-1>", lambda e: self._start())
        self.focus_set()

    def _reset(self):
        cx, cy = self.COLS // 2, self.ROWS // 2
        self._snake   = [(cx, cy), (cx-1, cy), (cx-2, cy)]
        self._dx      = 1;  self._dy = 0
        self._score   = 0
        self._alive   = False
        self._aid     = None
        self._food    = self._spawn_food()
        self._score_v.set("SCORE: 0")
        self._draw_screen("SNAKE", "Click to start")

    def _start(self):
        if self._aid:
            self.after_cancel(self._aid)
        self._reset()
        self._alive = True
        self._loop()

    def _steer(self, dx, dy):
        if (dx, dy) != (-self._dx, -self._dy):
            self._dx, self._dy = dx, dy

    def _spawn_food(self):
        occupied = set(self._snake)
        while True:
            p = (random.randint(0, self.COLS-1),
                 random.randint(0, self.ROWS-1))
            if p not in occupied:
                return p

    def _loop(self):
        if not self._alive:
            return
        hx, hy = self._snake[0]
        nx, ny  = hx + self._dx, hy + self._dy

        # collision
        if not (0 <= nx < self.COLS and 0 <= ny < self.ROWS) \
                or (nx, ny) in self._snake:
            self._alive = False
            if self._score > self._hi:
                self._hi = self._score
                self._hi_v.set(f"BEST: {self._hi}")
            self._draw_screen("GAME OVER",
                              f"Score: {self._score}  ·  Click to restart",
                              color=AMBER)
            return

        self._snake.insert(0, (nx, ny))
        if (nx, ny) == self._food:
            self._score += 10
            self._score_v.set(f"SCORE: {self._score}")
            self._food = self._spawn_food()
        else:
            self._snake.pop()

        self._draw()
        delay = max(60, 130 - self._score // 5)
        self._aid = self.after(delay, self._loop)

    # ── drawing ──────────────────────────────────────────────────────────────
    def _draw(self):
        C = self.CELL
        self._cv.delete("all")
        W = self.COLS * C;  H = self.ROWS * C

        # grid lines
        for x in range(0, W, C):
            self._cv.create_line(x, 0, x, H, fill="#0d1520", width=1)
        for y in range(0, H, C):
            self._cv.create_line(0, y, W, y, fill="#0d1520", width=1)

        # food
        fx, fy = self._food
        pad = 4
        self._cv.create_oval(
            fx*C+pad, fy*C+pad,
            fx*C+C-pad, fy*C+C-pad,
            fill=PINK, outline="")

        # snake
        for i, (x, y) in enumerate(self._snake):
            if   i == 0: fill = CYAN
            elif i < 5:  fill = GREEN
            else:        fill = "#1a6b35"
            self._cv.create_rectangle(
                x*C+2, y*C+2, x*C+C-2, y*C+C-2,
                fill=fill, outline=DARK, width=1)

    def _draw_screen(self, title, sub, color=GREEN):
        W = self.COLS * self.CELL;  H = self.ROWS * self.CELL
        self._cv.delete("all")
        self._cv.create_rectangle(
            W//2-160, H//2-50, W//2+160, H//2+56,
            fill=PANEL, outline=color, width=2)
        self._cv.create_text(W//2, H//2-18,
                             text=title, fill=color,
                             font=("Consolas", 22, "bold"))
        self._cv.create_text(W//2, H//2+18,
                             text=sub, fill=MUTED,
                             font=("Consolas", 10))

    def _center(self):
        self.update_idletasks()
        w = self.winfo_width();  h = self.winfo_height()
        sw = self.winfo_screenwidth();  sh = self.winfo_screenheight()
        self.geometry(f"+{(sw - w)//2 + 240}+{(sh - h)//2}")


# ─────────────────────────────────────────────────────────────────────────────
#  WORD SHORTCUTS  (searchable cheatsheet)
# ─────────────────────────────────────────────────────────────────────────────
SHORTCUTS = {
    "FILE": [
        ("Ctrl + N",         "New document"),
        ("Ctrl + O",         "Open document"),
        ("Ctrl + S",         "Save"),
        ("Ctrl + Shift + S", "Save As"),
        ("Ctrl + W",         "Close document"),
        ("Ctrl + P",         "Print"),
        ("Ctrl + Z",         "Undo"),
        ("Ctrl + Y",         "Redo"),
        ("F12",              "Save As dialog"),
        ("Alt + F4",         "Close Word"),
    ],
    "EDITING": [
        ("Ctrl + C",         "Copy"),
        ("Ctrl + X",         "Cut"),
        ("Ctrl + V",         "Paste"),
        ("Ctrl + A",         "Select All"),
        ("Ctrl + F",         "Find"),
        ("Ctrl + H",         "Find & Replace"),
        ("Ctrl + G",         "Go To page"),
        ("Delete",           "Delete next character"),
        ("Backspace",        "Delete previous character"),
        ("Ctrl + Delete",    "Delete next word"),
        ("Ctrl + Backspace", "Delete previous word"),
    ],
    "FORMATTING": [
        ("Ctrl + B",          "Bold"),
        ("Ctrl + I",          "Italic"),
        ("Ctrl + U",          "Underline"),
        ("Ctrl + E",          "Centre align"),
        ("Ctrl + L",          "Left align"),
        ("Ctrl + R",          "Right align"),
        ("Ctrl + J",          "Justify"),
        ("Ctrl + ]",          "Increase font size"),
        ("Ctrl + [",          "Decrease font size"),
        ("Ctrl + D",          "Font dialog"),
        ("Ctrl + Shift + >",  "Larger font"),
        ("Ctrl + Shift + <",  "Smaller font"),
    ],
    "NAVIGATION": [
        ("Ctrl + Home",       "Go to beginning"),
        ("Ctrl + End",        "Go to end"),
        ("Ctrl + →",          "Next word"),
        ("Ctrl + ←",          "Previous word"),
        ("Ctrl + ↑",          "Previous paragraph"),
        ("Ctrl + ↓",          "Next paragraph"),
        ("Page Up",           "Scroll up one page"),
        ("Page Down",         "Scroll down one page"),
    ],
    "SELECTION": [
        ("Shift + → / ←",      "Select character by character"),
        ("Ctrl + Shift + →",   "Select next word"),
        ("Shift + Home",       "Select to line start"),
        ("Shift + End",        "Select to line end"),
        ("Ctrl + Shift + End", "Select to document end"),
        ("Ctrl + Shift + Home","Select to document start"),
    ],
    "REVIEW & MISC": [
        ("Ctrl + K",          "Insert hyperlink"),
        ("Alt + Shift + D",   "Insert current date"),
        ("Alt + Shift + T",   "Insert current time"),
        ("F7",                "Spell check"),
        ("Ctrl + Shift + E",  "Track changes"),
        ("Ctrl + Alt + M",    "Insert comment"),
        ("Ctrl + 1",          "Single line spacing"),
        ("Ctrl + 2",          "Double line spacing"),
        ("Ctrl + 5",          "1.5 line spacing"),
    ],
}

SEC_COLORS = {
    "FILE":          CYAN,
    "EDITING":       GREEN,
    "FORMATTING":    AMBER,
    "NAVIGATION":    PINK,
    "SELECTION":     "#a371f7",
    "REVIEW & MISC": "#f0883e",
}


class ShortcutsWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Microsoft Word Shortcuts")
        self.configure(bg=BG)
        self.geometry("620x580")
        self._all_rows = []   # (section, key, desc, row_frame)
        self._build()
        self._center()

    def _build(self):
        mf = tkfont.Font(family="Consolas", size=9)
        bf = tkfont.Font(family="Consolas", size=9, weight="bold")

        # ── search bar ───────────────────────────────────────────────────────
        search_row = tk.Frame(self, bg=PANEL, pady=8, padx=12)
        search_row.pack(fill="x")

        tk.Label(search_row, text="🔍  Filter:", fg=MUTED, bg=PANEL,
                 font=mf).pack(side="left")

        self._q = tk.StringVar()
        self._q.trace_add("write", lambda *_: self._filter())
        entry = tk.Entry(search_row, textvariable=self._q,
                         bg=BG, fg=WHITE, insertbackground=CYAN,
                         relief="flat", font=mf, width=30)
        entry.pack(side="left", padx=8, ipady=4)

        tk.Button(search_row, text="✕ Clear",
                  command=lambda: self._q.set(""),
                  bg=BG, fg=MUTED, activebackground=BG,
                  activeforeground=WHITE, relief="flat",
                  font=mf, cursor="hand2").pack(side="left")

        tk.Label(search_row, text=f"{sum(len(v) for v in SHORTCUTS.values())} shortcuts",
                 fg=MUTED, bg=PANEL, font=mf).pack(side="right")

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # ── scrollable content ───────────────────────────────────────────────
        wrapper = tk.Frame(self, bg=BG)
        wrapper.pack(fill="both", expand=True)

        self._canvas = tk.Canvas(wrapper, bg=BG, highlightthickness=0)
        vsb = tk.Scrollbar(wrapper, orient="vertical",
                           command=self._canvas.yview,
                           bg=PANEL, troughcolor=BG,
                           activebackground=CYAN, relief="flat")
        self._canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        self._inner = tk.Frame(self._canvas, bg=BG)
        self._cwin  = self._canvas.create_window(
            (0, 0), window=self._inner, anchor="nw")

        self._inner.bind("<Configure>", lambda e: self._canvas.configure(
            scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>", lambda e: self._canvas.itemconfig(
            self._cwin, width=e.width))

        # mousewheel scroll
        self._canvas.bind_all(
            "<MouseWheel>",
            lambda e: self._canvas.yview_scroll(-1*(e.delta//120), "units"))

        # ── populate rows ────────────────────────────────────────────────────
        for section, items in SHORTCUTS.items():
            color = SEC_COLORS.get(section, WHITE)

            # section header
            hdr = tk.Frame(self._inner, bg=BG)
            hdr.pack(fill="x", padx=14, pady=(10, 3))
            tk.Frame(hdr, bg=color, width=3, height=18).pack(side="left")
            tk.Label(hdr, text=f"  {section}", fg=color, bg=BG,
                     font=bf).pack(side="left")

            for key_str, desc in items:
                row = tk.Frame(self._inner, bg=PANEL, pady=5)
                row.pack(fill="x", padx=14, pady=1)

                # key badge
                k_lbl = tk.Label(row, text=key_str,
                                 fg=color, bg=BG,
                                 font=tkfont.Font(family="Consolas", size=9),
                                 width=22, anchor="w", padx=8)
                k_lbl.pack(side="left")

                # description
                d_lbl = tk.Label(row, text=desc,
                                 fg=WHITE, bg=PANEL,
                                 font=mf, anchor="w", padx=6)
                d_lbl.pack(side="left", fill="x", expand=True)

                self._all_rows.append((section, key_str, desc, row))

        # bottom padding
        tk.Label(self._inner, text="", bg=BG, height=1).pack()

    def _filter(self):
        q = self._q.get().lower().strip()
        for section, key_str, desc, row in self._all_rows:
            match = (not q or
                     q in key_str.lower() or
                     q in desc.lower() or
                     q in section.lower())
            if match:
                row.pack(fill="x", padx=14, pady=1)
            else:
                row.pack_forget()

    def _center(self):
        self.update_idletasks()
        w = self.winfo_width();  h = self.winfo_height()
        sw = self.winfo_screenwidth();  sh = self.winfo_screenheight()
        self.geometry(f"+{(sw - w)//2}+{(sh - h)//2}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = Launcher()
    app.mainloop()
