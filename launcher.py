"""
GazeOS  —  Launcher  (Fullscreen Edition)
==========================================
python launcher.py

Requires in the same folder:
  tracker.py · gaze_cursor.py · virtual_keyboard.py
  text_pad.py · face_landmarker.task
"""

import tkinter as tk
from tkinter import font as tkfont, filedialog
import subprocess, sys, os, datetime, random

# ── palette ───────────────────────────────────────────────────────────────────
BG      = "#F5F6F8"
SURFACE = "#FFFFFF"
BORDER  = "#DDE1E7"
SEP     = "#E5E7EB"
TEXT    = "#1A1D23"
MUTED   = "#6B7280"
ACCENT  = "#2563EB"
GREEN   = "#16A34A"
AMBER   = "#D97706"
PURPLE  = "#7C3AED"
DANGER  = "#DC2626"

# ── font stack (larger for fullscreen) ───────────────────────────────────────
FH  = ("Segoe UI", 15, "bold")   # card heading
FB  = ("Segoe UI", 12)           # body
FS  = ("Segoe UI", 10)           # small / muted
FT  = ("Segoe UI", 28, "bold")   # app title
FM  = ("Consolas", 12)           # mono

# ── auto-start cooldown (milliseconds) ───────────────────────────────────────
AUTO_START_DELAY_MS = 0   # 1.5 seconds — change as needed


def _sep(parent):
    tk.Frame(parent, bg=SEP, height=1).pack(fill="x")


# ─────────────────────────────────────────────────────────────────────────────
#  LAUNCHER
# ─────────────────────────────────────────────────────────────────────────────
class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GazeOS")
        self.configure(bg=BG)

        # ── Fullscreen ────────────────────────────────────────────────────────
        self.state("zoomed")          # Windows: maximized (keeps title bar)
        # Uncomment the next line for true borderless fullscreen instead:
        # self.attributes("-fullscreen", True)

        self.resizable(True, True)
        self._proc = None
        self._build()
        self._tick()

        # ── Auto-start eye tracker after cooldown ─────────────────────────────
        self._countdown_val = AUTO_START_DELAY_MS // 1000
        self._schedule_autostart()

    def _schedule_autostart(self):
        """Show a countdown in the status bar, then auto-launch the tracker."""
        if self._countdown_val > 0:
            self._sv.set(f"  Eye Tracker starting in {self._countdown_val}s …")
            self._countdown_val -= 1
            self.after(1000, self._schedule_autostart)
        else:
            self._sv.set("  Auto-starting Eye Tracker …")
            self.after(200, self._start_tracker)   # tiny pause so msg is visible

    def _build(self):
        # ── header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=SURFACE, padx=48, pady=30)
        hdr.pack(fill="x")

        lf = tk.Frame(hdr, bg=SURFACE)
        lf.pack(side="left")
        tk.Label(lf, text="GazeOS", bg=SURFACE, fg=TEXT,
                 font=FT).pack(anchor="w")
        tk.Label(lf, text="Eye-Tracker  ·  Assistive Technology Platform",
                 bg=SURFACE, fg=MUTED, font=FS).pack(anchor="w")

        rf = tk.Frame(hdr, bg=SURFACE)
        rf.pack(side="right")
        self._clock_v = tk.StringVar()
        tk.Label(rf, textvariable=self._clock_v,
                 bg=SURFACE, fg=MUTED, font=FM).pack(anchor="e")
        tk.Label(rf, text="Group 10  ·  Mini Project",
                 bg=SURFACE, fg=MUTED, font=FS).pack(anchor="e")

        _sep(self)

        # ── card grid ─────────────────────────────────────────────────────────
        # Use a centred container that expands with the window
        wrapper = tk.Frame(self, bg=BG)
        wrapper.pack(fill="both", expand=True)

        body = tk.Frame(wrapper, bg=BG)
        body.place(relx=0.5, rely=0.5, anchor="center")

        modules = [
            ("👁  Eye Tracker",    "Calibrate & start gaze tracking",       ACCENT,  self._start_tracker),
            ("📝  Notepad",        "Text editor  ·  save / open files",      GREEN,   lambda: NotepadWindow(self)),
            ("🐍  Snake Game",     "Arrow keys or WASD to play",             AMBER,   lambda: GameWindow(self)),
            ("⌨  Word Shortcuts", "Searchable MS Word keyboard shortcuts",  PURPLE,  lambda: ShortcutsWindow(self)),
        ]

        for i, (name, desc, color, cmd) in enumerate(modules):
            r, c = divmod(i, 2)
            _Card(body, name, desc, color, cmd).grid(
                row=r, column=c, padx=20, pady=20, sticky="nsew")

        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)
        body.grid_rowconfigure(1, weight=1)

        _sep(self)

        # ── status bar ────────────────────────────────────────────────────────
        sb = tk.Frame(self, bg=SURFACE, padx=48, pady=14)
        sb.pack(fill="x")

        self._dot = tk.Label(sb, text="●", fg=BORDER, bg=SURFACE, font=FB)
        self._dot.pack(side="left")

        self._sv = tk.StringVar(value="  Initialising …")
        tk.Label(sb, textvariable=self._sv, bg=SURFACE, fg=MUTED,
                 font=FB).pack(side="left")

        self._stop_btn = tk.Button(
            sb, text="Stop Tracker", command=self._stop,
            bg=SURFACE, fg=DANGER, activebackground=BG,
            activeforeground=DANGER, relief="flat", bd=0,
            font=FB, cursor="hand2")

        # ESC to exit fullscreen / close
        self.bind("<Escape>", lambda e: self.destroy())

    # ── tracker control ───────────────────────────────────────────────────────
    def _start_tracker(self):
        if self._proc and self._proc.poll() is None:
            self._flash("Tracker is already running.")
            return
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "_run.py")
        with open(script, "w") as f:
            f.write("from tracker import AttentionTracker\n"
                    "t = AttentionTracker()\n"
                    "t.calibrate(duration=3)\n"
                    "t.run()\n")
        self._proc = subprocess.Popen(
            [sys.executable, script],
            cwd=os.path.dirname(os.path.abspath(__file__)))
        self._dot.config(fg=GREEN)
        self._sv.set("  Tracker running  —  K = keyboard   T = text pad   ESC = stop")
        self._stop_btn.pack(side="right")
        self._poll()

    def _stop(self):
        if self._proc: self._proc.terminate()
        self._dot.config(fg=BORDER)
        self._sv.set("  Tracker stopped")
        self._stop_btn.pack_forget()

    def _poll(self):
        if self._proc and self._proc.poll() is not None:
            self._dot.config(fg=BORDER)
            self._sv.set("  Tracker finished")
            self._stop_btn.pack_forget()
        else:
            self.after(1000, self._poll)

    def _flash(self, msg):
        old = self._sv.get()
        self._sv.set(f"  {msg}")
        self.after(3000, lambda: self._sv.set(old))

    def _tick(self):
        self._clock_v.set(datetime.datetime.now().strftime("%H:%M:%S"))
        self.after(1000, self._tick)

    def destroy(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        super().destroy()


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE CARD  (button tile)
# ─────────────────────────────────────────────────────────────────────────────
class _Card(tk.Frame):
    W = 380;  H = 150

    def __init__(self, parent, name, desc, color, cmd):
        super().__init__(parent, bg=SURFACE, width=self.W, height=self.H,
                         highlightbackground=BORDER, highlightthickness=1,
                         cursor="hand2")
        self.pack_propagate(False)
        self._color = color
        self._cmd   = cmd

        # left accent bar
        bar = tk.Frame(self, bg=color, width=7)
        bar.pack(side="left", fill="y")

        inner = tk.Frame(self, bg=SURFACE, padx=24, pady=22)
        inner.pack(fill="both", expand=True)

        self._nl = tk.Label(inner, text=name, bg=SURFACE, fg=TEXT, font=FH, anchor="w")
        self._nl.pack(fill="x")

        tk.Frame(inner, bg=SEP, height=1).pack(fill="x", pady=8)

        self._dl = tk.Label(inner, text=desc, bg=SURFACE, fg=MUTED, font=FB, anchor="w",
                            wraplength=320, justify="left")
        self._dl.pack(fill="x")

        for w in (self, bar, inner, self._nl, self._dl):
            w.bind("<Enter>",    self._on)
            w.bind("<Leave>",    self._off)
            w.bind("<Button-1>", lambda e: self._cmd())

    def _on(self, _=None):
        self.configure(highlightbackground=self._color, bg=self._color + "11")
        self._nl.configure(fg=self._color)

    def _off(self, _=None):
        self.configure(highlightbackground=BORDER, bg=SURFACE)
        self._nl.configure(fg=TEXT)


# ─────────────────────────────────────────────────────────────────────────────
#  NOTEPAD
# ─────────────────────────────────────────────────────────────────────────────
class NotepadWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Notepad")
        self.configure(bg=SURFACE)
        self.state("zoomed")
        self._path = None
        self._build()

    def _build(self):
        menu = tk.Menu(self, bg=SURFACE, fg=TEXT,
                       activebackground=ACCENT, activeforeground=SURFACE,
                       relief="flat", tearoff=0)
        self.config(menu=menu)

        fm = tk.Menu(menu, bg=SURFACE, fg=TEXT,
                     activebackground=ACCENT, activeforeground=SURFACE, tearoff=0)
        menu.add_cascade(label="File", menu=fm)
        fm.add_command(label="New",     command=self._new,     accelerator="Ctrl+N")
        fm.add_command(label="Open",    command=self._open,    accelerator="Ctrl+O")
        fm.add_command(label="Save",    command=self._save,    accelerator="Ctrl+S")
        fm.add_command(label="Save As", command=self._save_as, accelerator="Ctrl+Shift+S")
        fm.add_separator()
        fm.add_command(label="Exit", command=self.destroy)

        em = tk.Menu(menu, bg=SURFACE, fg=TEXT,
                     activebackground=ACCENT, activeforeground=SURFACE, tearoff=0)
        menu.add_cascade(label="Edit", menu=em)
        em.add_command(label="Undo",       command=lambda: self._txt.edit_undo(),              accelerator="Ctrl+Z")
        em.add_command(label="Redo",       command=lambda: self._txt.edit_redo(),              accelerator="Ctrl+Y")
        em.add_separator()
        em.add_command(label="Cut",        command=lambda: self._txt.event_generate("<<Cut>>"),   accelerator="Ctrl+X")
        em.add_command(label="Copy",       command=lambda: self._txt.event_generate("<<Copy>>"),  accelerator="Ctrl+C")
        em.add_command(label="Paste",      command=lambda: self._txt.event_generate("<<Paste>>"), accelerator="Ctrl+V")
        em.add_command(label="Select All", command=lambda: self._txt.tag_add("sel","1.0","end"),  accelerator="Ctrl+A")

        tb = tk.Frame(self, bg=SURFACE, pady=8, padx=16)
        tb.pack(fill="x")

        def tbtn(text, cmd, fg=TEXT, bold=False):
            f = ("Segoe UI", 11, "bold") if bold else ("Segoe UI", 11)
            b = tk.Button(tb, text=text, command=cmd, bg=SURFACE, fg=fg,
                          activebackground=BG, activeforeground=fg,
                          relief="flat", bd=0, font=f, padx=12, pady=4,
                          cursor="hand2")
            b.pack(side="left", padx=2)

        tbtn("New",     self._new)
        tbtn("Open",    self._open)
        tbtn("Save",    self._save,    fg=ACCENT, bold=True)
        tbtn("Save As", self._save_as, fg=ACCENT)

        tk.Label(tb, text="|", bg=SURFACE, fg=BORDER, font=FB).pack(side="left", padx=8)
        tk.Label(tb, text="Size", bg=SURFACE, fg=MUTED, font=FS).pack(side="left")
        self._fsize = tk.IntVar(value=13)
        tk.Spinbox(tb, from_=8, to=32, textvariable=self._fsize, width=3,
                   relief="flat", bg=BG, fg=TEXT, font=FM,
                   command=self._resize).pack(side="left", padx=8)

        self._wrap_v = tk.BooleanVar(value=True)
        tk.Checkbutton(tb, text="Wrap", variable=self._wrap_v,
                       command=self._toggle_wrap, bg=SURFACE, fg=MUTED,
                       activebackground=SURFACE, selectcolor=SURFACE,
                       font=FS).pack(side="left")

        self._sv = tk.StringVar(value="Ready")
        tk.Label(tb, textvariable=self._sv, bg=SURFACE, fg=MUTED,
                 font=FS).pack(side="right", padx=12)

        _sep(self)

        frame = tk.Frame(self, bg=SURFACE)
        frame.pack(fill="both", expand=True)

        self._tfont = tkfont.Font(family="Consolas", size=13)
        self._txt = tk.Text(
            frame, bg=SURFACE, fg=TEXT,
            insertbackground=ACCENT,
            selectbackground="#BFDBFE", selectforeground=TEXT,
            font=self._tfont, wrap="word", undo=True,
            padx=32, pady=20, relief="flat", bd=0,
            spacing1=3, spacing3=3)

        vsb = tk.Scrollbar(frame, command=self._txt.yview,
                           relief="flat", bg=BG, troughcolor=BG)
        self._txt.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._txt.pack(fill="both", expand=True)
        self._txt.bind("<KeyRelease>", self._update_sv)

        self.bind("<Control-s>", lambda e: self._save())
        self.bind("<Control-S>", lambda e: self._save_as())
        self.bind("<Control-n>", lambda e: self._new())
        self.bind("<Control-o>", lambda e: self._open())

    def insert_text(self, text):
        self._txt.insert("end", text)
        self._update_sv()
        self.lift(); self.focus_force()

    def _new(self):
        self._txt.delete("1.0","end")
        self._path = None
        self.title("Notepad  —  untitled")
        self._sv.set("New file")

    def _open(self):
        p = filedialog.askopenfilename(
            filetypes=[("Text","*.txt"),("All","*.*")])
        if not p: return
        with open(p, encoding="utf-8") as f:
            self._txt.delete("1.0","end")
            self._txt.insert("1.0", f.read())
        self._path = p
        self.title(f"Notepad  —  {os.path.basename(p)}")
        self._update_sv()

    def _save(self):
        if not self._path: self._save_as(); return
        with open(self._path,"w",encoding="utf-8") as f:
            f.write(self._txt.get("1.0","end").rstrip())
        self._sv.set(f"Saved  —  {os.path.basename(self._path)}")

    def _save_as(self):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        p  = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=f"typed_text_{ts}.txt",
            filetypes=[("Text","*.txt"),("All","*.*")])
        if not p: return
        self._path = p; self._save()
        self.title(f"Notepad  —  {os.path.basename(p)}")

    def _toggle_wrap(self):
        self._txt.configure(wrap="word" if self._wrap_v.get() else "none")

    def _resize(self):
        self._tfont.configure(size=self._fsize.get())

    def _update_sv(self, _=None):
        c = self._txt.get("1.0","end").strip()
        self._sv.set(f"{len(c)} chars  ·  {len(c.split()) if c else 0} words")


# ─────────────────────────────────────────────────────────────────────────────
#  SNAKE GAME
# ─────────────────────────────────────────────────────────────────────────────
class GameWindow(tk.Toplevel):
    CELL=26; COLS=28; ROWS=22

    def __init__(self, master):
        super().__init__(master)
        self.title("Snake")
        self.configure(bg=BG)
        self.resizable(False, False)
        self._hi=0; self._aid=None
        self._build(); self._center(); self._reset()

    def _build(self):
        W=self.COLS*self.CELL; H=self.ROWS*self.CELL

        top = tk.Frame(self, bg=SURFACE, pady=12, padx=24)
        top.pack(fill="x")
        self._sv  = tk.StringVar(value="Score  0")
        self._hiv = tk.StringVar(value="Best  0")
        tk.Label(top, textvariable=self._sv,  bg=SURFACE, fg=TEXT,
                 font=("Segoe UI",14,"bold")).pack(side="left")
        tk.Label(top, textvariable=self._hiv, bg=SURFACE, fg=MUTED,
                 font=("Segoe UI",14)).pack(side="right")

        _sep(self)

        self._cv = tk.Canvas(self, width=W, height=H, bg=SURFACE,
                             highlightthickness=1, highlightbackground=BORDER)
        self._cv.pack(padx=16, pady=12)

        _sep(self)
        tk.Label(self, text="Arrow keys / WASD  ·  Click canvas to start or restart",
                 bg=BG, fg=MUTED, font=FB).pack(pady=10)

        for key,d in [("<Up>",(0,-1)),("<Down>",(0,1)),
                       ("<Left>",(-1,0)),("<Right>",(1,0)),
                       ("<w>",(0,-1)),("<s>",(0,1)),
                       ("<a>",(-1,0)),("<d>",(1,0))]:
            self.bind(key, lambda e,dv=d: self._steer(*dv))
        self._cv.bind("<Button-1>", lambda e: self._start())
        self.focus_set()

    def _reset(self):
        cx,cy=self.COLS//2,self.ROWS//2
        self._snake=[(cx,cy),(cx-1,cy),(cx-2,cy)]
        self._dx=1; self._dy=0
        self._score=0; self._alive=False
        self._food=self._spawn()
        self._sv.set("Score  0")
        self._screen("Snake","Click to start")

    def _start(self):
        if self._aid: self.after_cancel(self._aid)
        self._reset(); self._alive=True; self._loop()

    def _steer(self,dx,dy):
        if (dx,dy)!=(-self._dx,-self._dy):
            self._dx,self._dy=dx,dy

    def _spawn(self):
        occ=set(self._snake)
        while True:
            p=(random.randint(0,self.COLS-1),random.randint(0,self.ROWS-1))
            if p not in occ: return p

    def _loop(self):
        if not self._alive: return
        hx,hy=self._snake[0]
        nx,ny=hx+self._dx,hy+self._dy
        if not(0<=nx<self.COLS and 0<=ny<self.ROWS) or (nx,ny) in self._snake:
            self._alive=False
            if self._score>self._hi:
                self._hi=self._score
                self._hiv.set(f"Best  {self._hi}")
            self._screen("Game Over",f"Score: {self._score}  ·  Click to restart")
            return
        self._snake.insert(0,(nx,ny))
        if (nx,ny)==self._food:
            self._score+=10; self._sv.set(f"Score  {self._score}")
            self._food=self._spawn()
        else:
            self._snake.pop()
        self._draw()
        self._aid=self.after(max(60,130-self._score//5),self._loop)

    def _draw(self):
        C=self.CELL
        self._cv.delete("all")
        W=self.COLS*C; H=self.ROWS*C
        for x in range(0,W+1,C):
            self._cv.create_line(x,0,x,H,fill=SEP)
        for y in range(0,H+1,C):
            self._cv.create_line(0,y,W,y,fill=SEP)
        fx,fy=self._food; p=6
        self._cv.create_oval(fx*C+p,fy*C+p,fx*C+C-p,fy*C+C-p,
                             fill=DANGER,outline="")
        for i,(x,y) in enumerate(self._snake):
            fill=ACCENT if i==0 else "#3B82F6" if i<5 else "#93C5FD"
            self._cv.create_rectangle(x*C+2,y*C+2,x*C+C-2,y*C+C-2,
                                      fill=fill,outline=SURFACE,width=1)

    def _screen(self,title,sub):
        W=self.COLS*self.CELL; H=self.ROWS*self.CELL
        self._cv.delete("all")
        self._cv.create_rectangle(W//2-170,H//2-60,W//2+170,H//2+64,
                                  fill=SURFACE,outline=BORDER,width=1)
        self._cv.create_text(W//2,H//2-18,text=title,fill=TEXT,
                             font=("Segoe UI",22,"bold"))
        self._cv.create_text(W//2,H//2+20,text=sub,fill=MUTED,
                             font=("Segoe UI",12))

    def _center(self):
        self.update_idletasks()
        w=self.winfo_width(); h=self.winfo_height()
        sw=self.winfo_screenwidth(); sh=self.winfo_screenheight()
        self.geometry(f"+{(sw-w)//2}+{(sh-h)//2}")


# ─────────────────────────────────────────────────────────────────────────────
#  WORD SHORTCUTS
# ─────────────────────────────────────────────────────────────────────────────
SHORTCUTS = {
    "File": [
        ("Ctrl + N","New document"),("Ctrl + O","Open document"),
        ("Ctrl + S","Save"),("Ctrl + Shift + S","Save As"),
        ("Ctrl + W","Close document"),("Ctrl + P","Print"),
        ("Ctrl + Z","Undo"),("Ctrl + Y","Redo"),
        ("F12","Save As dialog"),("Alt + F4","Close Word"),
    ],
    "Editing": [
        ("Ctrl + C","Copy"),("Ctrl + X","Cut"),("Ctrl + V","Paste"),
        ("Ctrl + A","Select All"),("Ctrl + F","Find"),
        ("Ctrl + H","Find & Replace"),("Ctrl + G","Go To page"),
        ("Delete","Delete next character"),("Backspace","Delete previous character"),
        ("Ctrl + Delete","Delete next word"),("Ctrl + Backspace","Delete previous word"),
    ],
    "Formatting": [
        ("Ctrl + B","Bold"),("Ctrl + I","Italic"),("Ctrl + U","Underline"),
        ("Ctrl + E","Centre align"),("Ctrl + L","Left align"),
        ("Ctrl + R","Right align"),("Ctrl + J","Justify"),
        ("Ctrl + ]","Increase font size"),("Ctrl + [","Decrease font size"),
        ("Ctrl + D","Font dialog"),("Ctrl + Shift + >","Larger font"),
        ("Ctrl + Shift + <","Smaller font"),
    ],
    "Navigation": [
        ("Ctrl + Home","Go to beginning"),("Ctrl + End","Go to end"),
        ("Ctrl + →","Next word"),("Ctrl + ←","Previous word"),
        ("Ctrl + ↑","Previous paragraph"),("Ctrl + ↓","Next paragraph"),
        ("Page Up","Scroll up one page"),("Page Down","Scroll down one page"),
    ],
    "Selection": [
        ("Shift + →/←","Select character by character"),
        ("Ctrl + Shift + →","Select next word"),
        ("Shift + Home","Select to line start"),
        ("Shift + End","Select to line end"),
        ("Ctrl + Shift + End","Select to document end"),
        ("Ctrl + Shift + Home","Select to document start"),
    ],
    "Review & Misc": [
        ("Ctrl + K","Insert hyperlink"),("Alt + Shift + D","Insert current date"),
        ("Alt + Shift + T","Insert current time"),("F7","Spell check"),
        ("Ctrl + Shift + E","Track changes"),("Ctrl + Alt + M","Insert comment"),
        ("Ctrl + 1","Single line spacing"),("Ctrl + 2","Double line spacing"),
        ("Ctrl + 5","1.5 line spacing"),
    ],
}

SEC_COLOR = {
    "File":ACCENT,"Editing":GREEN,"Formatting":AMBER,
    "Navigation":PURPLE,"Selection":"#0891B2","Review & Misc":"#BE185D",
}


class ShortcutsWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Word Shortcuts")
        self.configure(bg=BG)
        self.state("zoomed")
        self._rows=[]
        self._build()

    def _build(self):
        top = tk.Frame(self, bg=SURFACE, pady=14, padx=24)
        top.pack(fill="x")
        tk.Label(top, text="Search", bg=SURFACE, fg=MUTED, font=FB).pack(side="left")
        self._q = tk.StringVar()
        self._q.trace_add("write", lambda *_: self._filter())
        e = tk.Entry(top, textvariable=self._q, bg=BG, fg=TEXT,
                     insertbackground=ACCENT, relief="flat", font=FB, width=30)
        e.pack(side="left", padx=12, ipady=5)
        e.focus_set()
        tk.Button(top, text="Clear", command=lambda: self._q.set(""),
                  bg=SURFACE, fg=MUTED, activebackground=BG,
                  relief="flat", font=FS, cursor="hand2").pack(side="left")
        total = sum(len(v) for v in SHORTCUTS.values())
        tk.Label(top, text=f"{total} shortcuts", bg=SURFACE, fg=MUTED,
                 font=FS).pack(side="right")

        _sep(self)

        outer = tk.Frame(self, bg=BG)
        outer.pack(fill="both", expand=True)
        cv = tk.Canvas(outer, bg=BG, highlightthickness=0)
        vsb = tk.Scrollbar(outer, orient="vertical", command=cv.yview,
                           relief="flat", bg=BG, troughcolor=BG)
        cv.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        cv.pack(side="left", fill="both", expand=True)
        self._inner = tk.Frame(cv, bg=BG)
        win = cv.create_window((0,0), window=self._inner, anchor="nw")
        self._inner.bind("<Configure>",
            lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>",
            lambda e: cv.itemconfig(win, width=e.width))
        cv.bind_all("<MouseWheel>",
            lambda e: cv.yview_scroll(-1*(e.delta//120),"units"))

        for section, items in SHORTCUTS.items():
            color = SEC_COLOR.get(section, ACCENT)
            hdr = tk.Frame(self._inner, bg=BG)
            hdr.pack(fill="x", padx=24, pady=(16,4))
            tk.Frame(hdr, bg=color, width=4, height=18).pack(side="left")
            tk.Label(hdr, text=f"  {section}", bg=BG, fg=color,
                     font=("Segoe UI",11,"bold")).pack(side="left")

            for key_str, desc in items:
                row = tk.Frame(self._inner, bg=SURFACE)
                row.pack(fill="x", padx=24, pady=2)
                tk.Label(row, text=key_str, bg=SURFACE, fg=color,
                         font=FM, width=22, anchor="w",
                         padx=14, pady=8).pack(side="left")
                tk.Frame(row, bg=SEP, width=1).pack(side="left",
                                                     fill="y", pady=4)
                tk.Label(row, text=desc, bg=SURFACE, fg=TEXT,
                         font=FB, anchor="w", padx=16).pack(
                             side="left", fill="x", expand=True)
                self._rows.append((section, key_str, desc, row))

        tk.Label(self._inner, bg=BG, height=1).pack()

    def _filter(self):
        q = self._q.get().lower().strip()
        for section, key_str, desc, row in self._rows:
            show = (not q or q in key_str.lower()
                    or q in desc.lower() or q in section.lower())
            if show: row.pack(fill="x", padx=24, pady=2)
            else:    row.pack_forget()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = Launcher()
    app.mainloop()