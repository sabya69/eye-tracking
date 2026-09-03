
import tkinter as tk
from tkinter import font as tkfont, filedialog
import subprocess, sys, os, datetime, random, json, csv

# ── Heatmap generation (imported lazily to avoid blocking startup) ─────────────
def _try_generate_heatmap(gaze_pts, out_path, screen_w, screen_h, label):
    """Fire-and-forget heatmap generation in a background thread."""
    import threading
    def _worker():
        try:
            from heatmap_generator import generate_heatmap
            generate_heatmap(gaze_pts, out_path,
                             screen_w=screen_w, screen_h=screen_h,
                             session_label=label)
        except Exception as e:
            print(f"[HeatMap] Experiment heatmap error: {e}")
    threading.Thread(target=_worker, daemon=True).start()

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

# ── auto-start disabled ───────────────────────────────────────────────────────
AUTO_START_DELAY_MS = 1500


def _sep(parent):
    tk.Frame(parent, bg=SEP, height=1).pack(fill="x")


# ─────────────────────────────────────────────────────────────────────────────
#  LAUNCHER
# ─────────────────────────────────────────────────────────────────────────────
class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.attributes('-alpha', 0.0)  # Make completely transparent to prevent flashing
        self.withdraw()  # Hide UI initially
        
        self.configure(bg=BG)

        # ── Fullscreen ────────────────────────────────────────────────────────
        self.resizable(True, True)
        self._proc = None
        self._build()
        self._tick()

        # ── Status bar initial state ──────────────────────────────────────────
        self._sv.set("Tracker starting...")
        
        # ── Auto-start tracker immediately ─────────────────────────────────────
        self.after(100, self._start_tracker)


    def _build(self):
        # ── header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=BG, padx=48, pady=30)
        hdr.pack(fill="x")

        lf = tk.Frame(hdr, bg=BG)
        lf.pack(side="left")
        tk.Label(lf, text="", bg=BG, fg=TEXT,
                 font=FT).pack(anchor="w")
        

        rf = tk.Frame(hdr, bg=BG)
        rf.pack(side="left")
        self._usage_v = tk.StringVar(value="Total Usage: 0h 0m")
        tk.Label(rf, textvariable=self._usage_v,
                 bg=BG, fg=ACCENT, font=FM).pack(anchor="e")
        self._clock_v = tk.StringVar()
        tk.Label(rf, textvariable=self._clock_v,
                 bg=BG, fg=MUTED, font=FM).pack(anchor="e")
        tk.Label(rf, text="Group 10 ---  Mini Project",
                 bg=BG, fg=MUTED, font=FS).pack(anchor="e")

        _sep(self)

        # ── card grid ─────────────────────────────────────────────────────────
        wrapper = tk.Frame(self, bg=BG)
        wrapper.pack(fill="both", expand=True)

        body = tk.Frame(wrapper, bg=BG)
        body.place(relx=0.18, rely=0.5, anchor="center")

        self.modules = [
            ("Eye Tracker", "Calibrate & start gaze tracking",  GREEN, self._start_tracker),
            ("Notepad","Text editor  ·  save / open files", AMBER,  lambda: NotepadWindow(self)),
            ("Text-Entry Experiment", "Gaze-based text entry  ·  measure typing speed & accuracy", PURPLE, self._launch_text_experiment),
        ]
        
 
        self.cards = []
        for i, (name, desc, color, cmd) in enumerate(self.modules):
            card = _Card(body, name, desc, color, cmd)
            card.grid(row=i, column=0, padx=20, pady=20, sticky="nsew")
            self.cards.append(card)

        for i in range(len(self.modules)):
            body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)

             

        

        _sep(self)

        # ── status bar ────────────────────────────────────────────────────────
        sb = tk.Frame(self, bg=BG, padx=48, pady=14)
        sb.pack(fill="x")

        self._dot = tk.Label(sb, text="●", fg=BORDER, bg=BG, font=FB)
        self._dot.pack(side="left")

        self._sv = tk.StringVar(value="  Initialising …")
        tk.Label(sb, textvariable=self._sv, bg=BG, fg=MUTED,
                 font=FB).pack(side="left")

        self._stop_btn = tk.Button(
            sb, text="Stop Tracker", command=self._stop,
            bg=BG   , fg=DANGER, activebackground=BG,
            activeforeground=DANGER, relief="flat", bd=0,
            font=("Segoe UI", 16, "bold"), padx=16, pady=8, cursor="hand2")

        self.bind("<Escape>", lambda e: self.destroy())
        self._refresh_stats()

    # ── stats & reports ───────────────────────────────────────────────────────
    def _refresh_stats(self):
        stats_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "usage_stats.json")
        try:
            if os.path.exists(stats_file):
                with open(stats_file, "r") as f:
                    stats = json.load(f)
                sec = stats.get("total_seconds", 0)
                self._usage_v.set(f"Total Usage: {int(sec//3600)}h {int((sec%3600)//60)}m")
            else:
                self._usage_v.set("Total Usage: 0h 0m")
        except Exception:
            pass

    def _show_report(self):
        report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_report.png")
        if os.path.exists(report_path):
            try:
                if sys.platform == "win32":
                    os.startfile(report_path)
                else:
                    opener = "open" if sys.platform == "darwin" else "xdg-open"
                    subprocess.call([opener, report_path])
                self._sv.set("  Opening session report …")
            except Exception as e:
                self._sv.set(f"  Error opening report: {e}")
        else:
            self._sv.set("  No session report found. Run tracker first.")

    def _launch_external(self, app_name):
        try:
            if sys.platform == "win32":
                os.startfile(app_name)
            else:
                subprocess.Popen([app_name])
            self._flash(f"Launching {app_name}...")
        except Exception as e:
            self._flash(f"Failed to launch {app_name}")
            print(f"[WARN] Could not launch {app_name}: {e}")

    def _launch_text_experiment(self):
        self._flash("Opening Text-Entry Experiment…")
        TextEntryExperiment(self)

    # ── tracker control ───────────────────────────────────────────────────────
    def _start_tracker(self):
        if self._proc and self._proc.poll() is None:
            self._flash("Tracker is already running...")
            return
            
        self.withdraw()  # Hide UI during calibration
        
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "_run.py")
        flag_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_done.flag")
        
        if os.path.exists(flag_file):
            try: os.remove(flag_file)
            except: pass

        with open(script, "w") as f:
            f.write("from tracker import AttentionTracker\n"
                    "import os\n"
                    "t = AttentionTracker()\n"
                    "t.calibrate(duration=3)\n"
                    "with open('calib_done.flag', 'w') as flag: flag.write('1')\n"
                    "t.run()\n")
                    
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        self._proc = subprocess.Popen(
            [sys.executable, script],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            startupinfo=startupinfo)
        self._dot.config(fg=GREEN)
        self._sv.set("Tracker running  properly ESC = stop")
        self._stop_btn.pack(side="right")
        self._poll()
        self._wait_for_calibration()

    def _restore_ui(self):
        self.attributes('-alpha', 1.0)
        self.deiconify()
        self.state("zoomed")
        self.lift()
        self.attributes('-topmost', True)
        self.after(100, lambda: self.attributes('-topmost', False))
        self.focus_force()

    def _wait_for_calibration(self):
        flag_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_done.flag")
        # Check if tracker process ended prematurely
        if self._proc and self._proc.poll() is not None:
            self._restore_ui()
            return

        if os.path.exists(flag_file):
            try:
                os.remove(flag_file)
            except:
                pass
            self._restore_ui()
        else:
            self.after(500, self._wait_for_calibration)

    def _stop(self):
        if self._proc: self._proc.terminate()
        self._dot.config(fg=BORDER)
        self._sv.set("  Tracker is stopped")
        self._refresh_stats()   # Update stats immediately after stopping
        self._stop_btn.pack_forget()

    def _poll(self):
        if self._proc and self._proc.poll() is not None:
            self._dot.config(fg=BORDER)
            self._sv.set("  Tracker finished")
            self._stop_btn.pack_forget()
            self._refresh_stats()   # Update stats after session ends
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
        self.configure(highlightbackground=self._color, bg=SURFACE)
        self._nl.configure(fg=self._color)

    def _off(self, _=None):
        self.configure(highlightbackground=BORDER, bg=SURFACE)
        self._nl.configure(fg=TEXT)

# ─────────────────────────────────────────────────────────────────────────────
#  APP ICON CARD  (smaller button tile for corners)
# ─────────────────────────────────────────────────────────────────────────────
class _AppIcon(tk.Frame):
    W = 180;  H = 60

    def __init__(self, parent, name, color, cmd):
        super().__init__(parent, bg=SURFACE, width=self.W, height=self.H,
                         highlightbackground=BORDER, highlightthickness=1,
                         cursor="hand2")
        self.pack_propagate(False)
        self._color = color
        self._cmd   = cmd

        bar = tk.Frame(self, bg=color, width=5)
        bar.pack(side="left", fill="y")

        inner = tk.Frame(self, bg=SURFACE, padx=12, pady=12)
        inner.pack(fill="both", expand=True)

        self._nl = tk.Label(inner, text=name, bg=SURFACE, fg=TEXT, font=("Segoe UI", 11, "bold"), anchor="center")
        self._nl.pack(fill="both", expand=True)

        for w in (self, bar, inner, self._nl):
            w.bind("<Enter>",    self._on)
            w.bind("<Leave>",    self._off)
            w.bind("<Button-1>", lambda e: self._cmd())

    def _on(self, _=None):
        self.configure(highlightbackground=self._color, bg=SURFACE)
        self._nl.configure(fg=self._color)

    def _off(self, _=None):
        self.configure(highlightbackground=BORDER, bg=SURFACE)
        self._nl.configure(fg=TEXT)


# ─────────────────────────────────────────────────────────────────────────────
#  NOTEPAD  — with integrated on-screen virtual keyboard
# ─────────────────────────────────────────────────────────────────────────────
class NotepadWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Notepad")
        self.configure(bg=SURFACE)
        self.state("zoomed")
        self._path = None
        self._kb_win = None          # keep reference to keyboard window
        self._build()
        # ── Auto-open virtual keyboard on launch ──────────────────────────────
        self.after(300, self._open_keyboard)

    def _build(self):
        menu = tk.Menu(self, bg=SURFACE, fg=TEXT,
                       activebackground=ACCENT, activeforeground=SURFACE,
                       relief="flat", tearoff=0)
        self.config(menu=menu)

        fm = tk.Menu(menu, bg=SURFACE, fg=TEXT,
                     activebackground=ACCENT, activeforeground=SURFACE, tearoff=0)
        menu.add_cascade(label="File", menu=fm)
        
        
        fm.add_command(label="Open Keyboard", command=self._open_keyboard, accelerator="Ctrl+Shift+K")
        fm.add_separator()
        fm.add_command(label="Exit", command=self.destroy)



        tb = tk.Frame(self, bg=SURFACE, pady=8, padx=16)
        tb.pack(fill="x")

        def tbtn(text, cmd, fg=TEXT, bold=False):
            f = ("Segoe UI", 11, "bold") if bold else ("Segoe UI", 11)
            b = tk.Button(tb, text=text, command=cmd, bg=SURFACE, fg=fg,
                          activebackground=BG, activeforeground=fg,
                          relief="flat", bd=0, font=f, padx=12, pady=4,
                          cursor="hand2")
            b.pack(side="left", padx=2)

        tbtn("Alphabetical Keyboard", lambda: self._open_keyboard("alpha"), fg=PURPLE, bold=True)
        tbtn("Normal Keyboard", lambda: self._open_keyboard("normal"), fg=PURPLE, bold=True)
        tbtn("Cluster Keyboard", lambda: self._open_keyboard("cluster"), fg=PURPLE, bold=True)

        tk.Label(tb, text="|", bg=SURFACE, fg=BORDER, font=FB).pack(side="left", padx=8)
        tk.Label(tb, text="Size", bg=SURFACE, fg=MUTED, font=FS).pack(side="left")
        self._fsize = tk.IntVar(value=25)
        tk.Spinbox(tb, from_=8, to=32, textvariable=self._fsize, width=3,
                   relief="flat", bg=BG, fg=TEXT, font=FM,
                   command=self._resize).pack(side="left", padx=8)

        self._wrap_v = tk.BooleanVar(value=True)
        tk.Checkbutton(tb, text="Wrap", variable=self._wrap_v,
                       command=self._toggle_wrap, bg=SURFACE, fg=MUTED,
                       activebackground=SURFACE, selectcolor=SURFACE,
                       font=FS).pack(side="left")

        tk.Button(tb, text="Stop Tracker", command=self.master._stop,
                  bg=SURFACE, fg=DANGER, activebackground=BG,
                  activeforeground=DANGER, relief="flat", bd=0,
                  font=("Segoe UI", 11, "bold"), padx=12, pady=4,
                  cursor="hand2").pack(side="right", padx=8)

        self._sv = tk.StringVar(value="Ready")
        tk.Label(tb, textvariable=self._sv, bg=SURFACE, fg=MUTED,
                 font=FS).pack(side="right", padx=12)

        _sep(self)

        frame = tk.Frame(self, bg=SURFACE)
        frame.pack(fill="both", expand=True)

        text_wrapper = tk.Frame(frame, bg=SURFACE)
        text_wrapper.place(relx=0, rely=0, relwidth=1, relheight=0.33)

        self.kb_wrapper = tk.Frame(frame, bg=BG)
        self.kb_wrapper.place(relx=0, rely=0.33, relwidth=1, relheight=0.67)

        self._tfont = tkfont.Font(family="Consolas", size=16)
        self._txt = tk.Text(
            text_wrapper, bg=SURFACE, fg=TEXT,
            insertbackground=ACCENT,
            selectbackground="#BFDBFE", selectforeground=TEXT,
            font=self._tfont, wrap="word", undo=True,
            padx=32, pady=20, relief="flat", bd=0,
            spacing1=3, spacing3=3)

        vsb = tk.Scrollbar(text_wrapper, command=self._txt.yview,
                           relief="flat", bg=BG, troughcolor=BG)
        self._txt.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._txt.pack(fill="both", expand=True)
        self._txt.bind("<KeyRelease>", self._update_sv)

        self.bind("<Control-s>",       lambda e: self._save())
        self.bind("<Control-S>",       lambda e: self._save_as())
        self.bind("<Control-n>",       lambda e: self._new())
        self.bind("<Control-o>",       lambda e: self._open())
        self.bind("<Control-K>",       lambda e: self._open_keyboard())

    # ── keyboard integration ──────────────────────────────────────────────────
    def _open_keyboard(self, layout="normal"):
        if hasattr(self, '_kb_win') and self._kb_win is not None:
            try:
                if self._kb_win.winfo_exists():
                    self._kb_win.set_layout(layout)
                    return
            except Exception:
                pass
            self._kb_win = None

        self._kb_win = OnScreenKeyboard(self.kb_wrapper, self._txt, layout=layout, notepad_app=self)
        self._kb_win.pack(fill="both", expand=True)

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
        if not self._path:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"typed_text_{ts}.txt")
        with open(self._path,"w",encoding="utf-8") as f:
            f.write(self._txt.get("1.0","end").rstrip())
        self._sv.set(f"Saved  —  {os.path.basename(self._path)}")
        self.title(f"Notepad  —  {os.path.basename(self._path)}")

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
#  BUILT-IN ON-SCREEN KEYBOARD  (fallback when virtual_keyboard.py not found)
# ─────────────────────────────────────────────────────────────────────────────
class OnScreenKeyboard(tk.Frame):
    LAYOUTS = {
        "normal": [
            ["1","2","3","4","5","6","7","8","9","0","-","⌫"],
            ["Q","W","E","R","T","Y","U","I","O","P","[","]"],
            ["A","S","D","F","G","H","J","K","L",";","'"],
            ["Z","X","C","V","B","N","M",",","."],
            ["Enter", "\\", "Space"],
        ],
        "alpha": [
            ["1","2","3","4","5","6","7","8","9","0","-","⌫"],
            ["A","B","C","D","E","F","G","H","I","J","K","L","M"],
            ["N","O","P","Q","R","S","T","U","V","W","X"],
            ["Y","Z",",",".",";","'","[","]"],
            ["Enter", "\\", "Space"],
        ],
        "cluster": [
            ["1","2","3","4","5","6","7","8","9","0","-","⌫"],
            ["A","E","T","N","S","H","F","G","Y","P"],
            ["I","O","R","D","L","C","B","V","K","J"],
            ["U","M","W","X","Q","Z",",",".",";","'","[","]"],
            ["Enter", "\\", "Space"],
        ]
    }

    SHIFT_MAP = {
        "`":"~","1":"!","2":"@","3":"#","4":"$","5":"%","6":"^",
        "7":"&","8":"*","9":"(","0":")","-":"_","=":"+","[":"{",
        "]":"}","\\":"|",";":":","'":'"',",":"<",".":">","/":"?",
    }

    WIDE = {"⌫":2,"Tab":1.5,"Caps":1.8,"Enter":2,"Shift":2.3,"Space":6,"Ctrl":1.5,"Alt":1.5}


    def __init__(self, master, target: tk.Text, layout="normal", notepad_app=None):
        super().__init__(master, bg=BG)
        self._target = target
        self.notepad_app = notepad_app
        self._shift  = False
        self._caps   = False
        self._layout = layout
        self.ROWS = self.LAYOUTS.get(self._layout, self.LAYOUTS["normal"])
        self._build()

    def set_layout(self, layout):
        if layout in self.LAYOUTS and layout != self._layout:
            self._layout = layout
            self.ROWS = self.LAYOUTS[layout]
            for widget in self.winfo_children():
                widget.destroy()
            self._build()

    def _build(self):
        if self._layout == "cluster":
            self._build_cluster()
            return

        pad = tk.Frame(self, bg=BG, padx=8, pady=8)
        pad.pack(fill="both", expand=True)

        for r, row_keys in enumerate(self.ROWS):
            pad.rowconfigure(r, weight=1)
            row_frame = tk.Frame(pad, bg=BG)
            row_frame.grid(row=r, column=0, sticky="nsew", pady=4)
            
            for c, key in enumerate(row_keys):
                w = self.WIDE.get(key, 1)
                row_frame.columnconfigure(c, weight=int(w * 10))
                
                btn = tk.Button(
                    row_frame,
                    text=key,
                    font=("Segoe UI", 16, "bold"),
                    bg=SURFACE, fg=TEXT,
                    activebackground=ACCENT,
                    activeforeground=SURFACE,
                    relief="flat",
                    bd=0,
                    highlightbackground=BORDER,
                    highlightthickness=1,
                    cursor="hand2"
                )
                btn.configure(command=lambda k=key, b=btn: self._press(k, b))
                btn.grid(row=0, column=c, sticky="nsew", padx=4)
            row_frame.rowconfigure(0, weight=1)

        # Info label
        pad.rowconfigure(len(self.ROWS), weight=0)
        pad.columnconfigure(0, weight=1)
        tk.Label(pad, text="After Clicking the keys using tracker you can type in notepad",
                 bg=BG, fg=MUTED, font=FS).grid(row=len(self.ROWS), column=0, pady=(10,0))

    def _build_cluster(self):
        pad = tk.Frame(self, bg=SURFACE, padx=8, pady=8)
        pad.pack(fill="both", expand=True)

        def make_btn(parent, text):
            btn = tk.Button(
                parent, text=text, font=("Segoe UI", 16, "bold"),
                bg=SURFACE, fg=TEXT, activebackground=ACCENT,
                activeforeground=SURFACE, relief="flat", bd=0,
                highlightbackground=BORDER, highlightthickness=1,
                cursor="hand2"
            )
            btn.configure(command=lambda k=text, b=btn: self._press(k, b))
            return btn

        # ROW 0: NUMBERS
        num_frame = tk.Frame(pad, bg=SURFACE)
        num_frame.pack(fill="x", pady=(0, 8))
        nums = ["1","2","3","4","5","6","7","8","9","0","-","⌫"]
        for c, k in enumerate(nums):
            num_frame.columnconfigure(c, weight=int(self.WIDE.get(k, 1)*10))
            btn = make_btn(num_frame, k)
            btn.grid(row=0, column=c, sticky="nsew", padx=4, pady=4)
        num_frame.rowconfigure(0, weight=1, minsize=50)

        # ROW 1: CLUSTERS
        mid_frame = tk.Frame(pad, bg=SURFACE)
        mid_frame.pack(fill="both", expand=True, pady=4)
        
        # Proportional weights so all buttons have exact same width
        mid_frame.columnconfigure(0, weight=5)
        mid_frame.columnconfigure(1, weight=6)
        mid_frame.rowconfigure(0, weight=1)

        # 1. TOP USAGE
        t_frame = tk.Frame(mid_frame, bg=BG, highlightbackground=BORDER, highlightthickness=1, padx=8, pady=4)
        t_frame.grid(row=0, column=0, sticky="nsew", padx=(0,4))
        tk.Label(t_frame, text="TOP USAGE", bg=BG, fg=MUTED, font=("Segoe UI", 9, "bold")).pack(pady=(0,4))
        t_grid = tk.Frame(t_frame, bg=BG)
        t_grid.pack(expand=True, fill="both")
        t_keys = [
            ["E","T","A","O","I"],
            ["N","S","H","R","D"],
            ["L","C","U","M","W"]
        ]
        for r, row in enumerate(t_keys):
            t_grid.rowconfigure(r, weight=1, minsize=65)
            for c, k in enumerate(row):
                t_grid.columnconfigure(c, weight=1)
                btn = make_btn(t_grid, k)
                btn.grid(row=r, column=c, sticky="nsew", padx=4, pady=4)

        # 2. REMAINING KEYS
        r_frame = tk.Frame(mid_frame, bg=BG, highlightbackground=BORDER, highlightthickness=1, padx=8, pady=4)
        r_frame.grid(row=0, column=1, sticky="nsew", padx=(4,0))
        tk.Label(r_frame, text="REMAINING KEYS", bg=BG, fg=MUTED, font=("Segoe UI", 9, "bold")).pack(pady=(0,4))
        r_grid = tk.Frame(r_frame, bg=BG)
        r_grid.pack(expand=True, fill="both")
        r_keys = [
            ["F","G","Y","P","B","V"],
            ["K","J","X","Q","Z","\\"],
            [",",".",";","'","[","]"]
        ]
        for r, row in enumerate(r_keys):
            r_grid.rowconfigure(r, weight=1, minsize=65)
            for c, k in enumerate(row):
                r_grid.columnconfigure(c, weight=1)
                btn = make_btn(r_grid, k)
                btn.grid(row=r, column=c, sticky="nsew", padx=4, pady=4)

        # ROW 2: ACTION
        act_frame = tk.Frame(pad, bg=SURFACE)
        act_frame.pack(fill="x", pady=(8, 0))
        act_frame.rowconfigure(0, weight=1, minsize=50)
        
        act_frame.columnconfigure(0, weight=2)
        btn_enter = make_btn(act_frame, "Enter")
        btn_enter.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        
        act_frame.columnconfigure(1, weight=8)
        btn_space = make_btn(act_frame, "Space")
        btn_space.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)

        tk.Label(pad, text="After Clicking the keys using tracker you can type in notepad",
                 bg=SURFACE, fg=MUTED, font=FS).pack(pady=(10,0))

    def _press(self, key, btn_widget=None):
        if btn_widget:
            orig_bg = btn_widget.cget("bg")
            orig_fg = btn_widget.cget("fg")
            btn_widget.configure(bg="#16A34A", fg="#FFFFFF", relief="sunken")
            self.after(150, lambda: btn_widget.configure(bg=orig_bg, fg=orig_fg, relief="flat"))

        t = self._target
        if key == "⌫":
            # Delete last character
            pos = t.index("insert")
            if pos != "1.0":
                t.delete(f"insert-1c", "insert")
        elif key in ("Shift",):
            self._shift = not self._shift
        elif key == "Caps":
            self._caps = not self._caps
        elif key == "Enter":
            if getattr(self, 'notepad_app', None):
                self.notepad_app._save()
            else:
                t.insert("insert", "\n")
        elif key == "Tab":
            t.insert("insert", "\t")
        elif key == "Space":
            t.insert("insert", " ")
        elif key in ("Ctrl", "Alt"):
            pass  # modifier stubs
        else:
            char = key
            # Apply shift map
            if self._shift and char in self.SHIFT_MAP:
                char = self.SHIFT_MAP[char]
            elif char.isalpha():
                # Caps XOR Shift
                if self._caps ^ self._shift:
                    char = char.upper()
                else:
                    char = char.lower()
            t.insert("insert", char)
            # Auto-release shift after one key
            if self._shift:
                self._shift = False
        t.see("insert")
        t.focus_set()

# ─────────────────────────────────────────────────────────────────────────────
#  CUSTOM WEBVIEW2 WIDGET — to bypass E_ACCESSDENIED by using a custom user data folder
# ─────────────────────────────────────────────────────────────────────────────
try:
    import clr
    import ctypes
    from uuid import uuid4
    from webview.window import Window
    from webview.platforms.edgechromium import EdgeChrome
    
    _webview_windows = []
    
    class CustomWebView2(tk.Frame):
        def __init__(self, parent, width: int, height: int, url: str = '', **kw):
            tk.Frame.__init__(self, parent, width=width, height=height, **kw)
            
            # Use a writable directory in AppData to avoid E_ACCESSDENIED
            cache_dir = os.path.join(os.getenv('APPDATA', os.path.expanduser('~')), 'SSDB_ChatBox_WebView2')
            os.makedirs(cache_dir, exist_ok=True)
            
            clr.AddReference('System.Windows.Forms')
            from System.Windows.Forms import Control
            
            control = Control()
            uid = 'master' if len(_webview_windows) == 0 else 'child_' + uuid4().hex[:8]
            window = Window(uid, str(id(self)), url=None, html=None, js_api=None, width=width, height=height, x=None, y=None,
                          resizable=True, fullscreen=False, min_size=(200, 100), hidden=False,
                          frameless=False, easy_drag=True,
                          minimized=False, on_top=False, confirm_close=False, background_color='#FFFFFF',
                          transparent=False, text_select=True, localization=None,
                          zoomable=True, draggable=True, vibrancy=False)
            self.window = window
            
            self.web_view = EdgeChrome(control, window, cache_dir)
            self.control = control
            if hasattr(self.web_view, 'web_view'):
                self.web = self.web_view.web_view
            else:
                self.web = self.web_view.webview
            _webview_windows.append(window)
            self.width = width
            self.height = height
            self.parent = parent
            self.chwnd = int(str(self.control.Handle))
            
            user32 = ctypes.windll.user32
            user32.SetParent(self.chwnd, self.winfo_id())
            user32.MoveWindow(self.chwnd, 0, 0, width, height, True)
            self.loaded = window.events.loaded
            
            self.bind('<Destroy>', lambda event: self.web.Dispose())
            self.bind('<Configure>', self.__resize_webview)
            self.newwindow = None
            
            if url != '':
                self.load_url(url)
            self.core = None
            self.web.CoreWebView2InitializationCompleted += self.__load_core

        def __resize_webview(self, event):
            ctypes.windll.user32.MoveWindow(self.chwnd, 0, 0, self.winfo_width(), self.winfo_height(), True)

        def __load_core(self, sender, _):
            self.core = sender.CoreWebView2
            self.core.NewWindowRequested -= self.web_view.on_new_window_request
            if self.newwindow != None:
                self.core.NewWindowRequested += self.newwindow
            settings = sender.CoreWebView2.Settings
            settings.AreDefaultContextMenusEnabled = True
            settings.AreDevToolsEnabled = True

        def load_url(self, url):
            self.web_view.load_url(url)

        def reload(self):
            if self.core:
                self.core.Reload()

        def event_new_window(self, command=None):
            self.newwindow = command
except Exception as e:
    import traceback
    traceback.print_exc()
    CustomWebView2 = None


# ─────────────────────────────────────────────────────────────────────────────
#  CHATTING WINDOW — with integrated WebView2 browser
# ─────────────────────────────────────────────────────────────────────────────
class ChatWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Chatting - SSDB Real-Time Chat")
        self.geometry("1100x850")
        self.configure(bg=SURFACE)
        self.state("zoomed")
        
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Header
        self.hdr = tk.Frame(self, bg=SURFACE, padx=24, pady=12)
        self.hdr.pack(fill="x")
        
        tk.Label(self.hdr, text="SSDB ChatBox", bg=SURFACE, fg=TEXT, font=("Segoe UI", 14, "bold")).pack(side="left")
        
        self.status_lbl = tk.Label(self.hdr, text="Loading secure chat session...", bg=SURFACE, fg=MUTED, font=FS)
        self.status_lbl.pack(side="right", padx=10)
        
        _sep(self)
        
        # Frame for webview
        self.web_frame = tk.Frame(self, bg=BG)
        self.web_frame.pack(fill="both", expand=True)
        
        self.after(100, self._load_webview)
        
    def _load_webview(self):
        try:
            if CustomWebView2 is None:
                raise ImportError("Required webview libraries or runtime not available.")
            
            # Create CustomWebView2 widget inside the web_frame
            self.webview = CustomWebView2(self.web_frame, 1100, 800, url="https://test-real-mk6w.onrender.com")
            self.webview.pack(fill="both", expand=True)
            self.status_lbl.config(text="Connected", fg=GREEN)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_lbl.config(text="Connection Error", fg=DANGER)
            self._show_fallback_error(e)
            
    def _show_fallback_error(self, error):
        # Clear the web_frame and show a friendly error
        for widget in self.web_frame.winfo_children():
            widget.destroy()
            
        err_container = tk.Frame(self.web_frame, bg=BG, padx=40, pady=40)
        err_container.place(relx=0.5, rely=0.5, anchor="center")
        
        tk.Label(err_container, text="Could not load Chatting UI", bg=BG, fg=DANGER, font=("Segoe UI", 16, "bold")).pack(pady=(0, 10))
        
        msg = (
            "The 'tkwebview2' and 'pywebview' libraries are required to embed the chat interface.\n\n"
            "If you see this error, please make sure they are installed and you have the WebView2 runtime.\n"
            "Otherwise, you can open the chatting website in your external browser instead."
        )
        tk.Label(err_container, text=msg, bg=BG, fg=TEXT, font=FB, justify="center", wraplength=500).pack(pady=10)
        
        btn_frame = tk.Frame(err_container, bg=BG)
        btn_frame.pack(pady=20)
        
        # Button to open in external browser
        tk.Button(
            btn_frame, text="Open in Browser", command=self._open_external,
            bg=ACCENT, fg=SURFACE, activebackground=ACCENT, activeforeground=SURFACE,
            relief="flat", font=("Segoe UI", 11, "bold"), padx=15, pady=8, cursor="hand2"
        ).pack(side="left", padx=10)
        
        # Button to retry loading
        tk.Button(
            btn_frame, text="Retry", command=self._retry_load,
            bg=SURFACE, fg=TEXT, activebackground=BG, activeforeground=TEXT,
            relief="flat", highlightbackground=BORDER, highlightthickness=1,
            font=("Segoe UI", 11), padx=15, pady=8, cursor="hand2"
        ).pack(side="left", padx=10)
        
    def _open_external(self):
        import webbrowser
        webbrowser.open("https://test-real-mk6w.onrender.com")
        self.destroy()
        
    def _retry_load(self):
        for widget in self.web_frame.winfo_children():
            widget.destroy()
        self.status_lbl.config(text="Retrying connection...", fg=MUTED)
        self.after(500, self._load_webview)
        
    def _on_close(self):
        # Explicitly destroy the webview to free resources
        if hasattr(self, 'webview') and self.webview is not None:
            try:
                self.webview.web.Dispose()
            except:
                pass
        self.destroy()


# ─────────────────────────────────────────────────────────────────────────────
#  EMBEDDED BROWSER WINDOW — with back, forward, reload, and search/address bar
# ─────────────────────────────────────────────────────────────────────────────
class BrowserWindow(tk.Toplevel):
    def __init__(self, master, title, url):
        super().__init__(master)
        self.title(title)
        self.geometry("1200x850")
        self.configure(bg=SURFACE)
        self.state("zoomed")
        
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Header
        self.hdr = tk.Frame(self, bg=SURFACE, padx=24, pady=8)
        self.hdr.pack(fill="x")
        
        # Left side: Title + Navigation buttons
        tk.Label(self.hdr, text=title, bg=SURFACE, fg=TEXT, font=("Segoe UI", 14, "bold")).pack(side="left", padx=(0, 15))
        
        btn_font = ("Segoe UI", 11, "bold")
        self.back_btn = tk.Button(
            self.hdr, text="◀ Back", command=self._go_back,
            bg=SURFACE, fg=ACCENT, activebackground=BG, activeforeground=ACCENT,
            relief="flat", font=btn_font, padx=8, pady=4, cursor="hand2"
        )
        self.back_btn.pack(side="left", padx=2)
        
        self.fwd_btn = tk.Button(
            self.hdr, text="Forward ▶", command=self._go_forward,
            bg=SURFACE, fg=ACCENT, activebackground=BG, activeforeground=ACCENT,
            relief="flat", font=btn_font, padx=8, pady=4, cursor="hand2"
        )
        self.fwd_btn.pack(side="left", padx=2)
        
        self.reload_btn = tk.Button(
            self.hdr, text="Reload ⟳", command=self._reload,
            bg=SURFACE, fg=MUTED, activebackground=BG, activeforeground=TEXT,
            relief="flat", font=btn_font, padx=8, pady=4, cursor="hand2"
        )
        self.reload_btn.pack(side="left", padx=2)
        
        # Address bar (URL entry)
        self.url_var = tk.StringVar(value=url)
        self.url_entry = tk.Entry(
            self.hdr, textvariable=self.url_var, font=("Segoe UI", 11),
            bg=BG, fg=TEXT, relief="flat", highlightbackground=BORDER, highlightthickness=1
        )
        self.url_entry.pack(side="left", fill="x", expand=True, padx=15, pady=4)
        self.url_entry.bind("<Return>", self._navigate_to_entry)
        
        # Right side: Status label
        self.status_lbl = tk.Label(self.hdr, text="Loading secure browser session...", bg=SURFACE, fg=MUTED, font=FS)
        self.status_lbl.pack(side="right", padx=10)
        
        _sep(self)
        
        # Frame for webview
        self.web_frame = tk.Frame(self, bg=BG)
        self.web_frame.pack(fill="both", expand=True)
        
        self.initial_url = url
        self.after(100, self._load_webview)
        
    def _load_webview(self):
        try:
            if CustomWebView2 is None:
                raise ImportError("Required webview libraries or runtime not available.")
            
            # Create CustomWebView2 widget inside the web_frame
            self.webview = CustomWebView2(self.web_frame, 1200, 800, url=self.initial_url)
            self.webview.pack(fill="both", expand=True)
            self.status_lbl.config(text="Connected", fg=GREEN)
            
            # Start loop to update address bar URL as navigation occurs
            self.after(1000, self._check_url_loop)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_lbl.config(text="Connection Error", fg=DANGER)
            self._show_fallback_error(e)
            
    def _check_url_loop(self):
        if hasattr(self, 'webview') and self.webview is not None:
            try:
                if hasattr(self.webview, 'web') and self.webview.web is not None:
                    current_uri = self.webview.web.Source
                    if current_uri is not None:
                        current_url = str(current_uri)
                        # Avoid updating if user is actively editing (has focus)
                        if self.focus_get() != self.url_entry and current_url != self.url_var.get():
                            self.url_var.set(current_url)
            except:
                pass
        try:
            if self.winfo_exists():
                self.after(1000, self._check_url_loop)
        except:
            pass
            
    def _navigate_to_entry(self, event=None):
        url = self.url_var.get().strip()
        if not url:
            return
        
        # Handle search vs URL
        if not (url.startswith("http://") or url.startswith("https://")):
            if "." in url and " " not in url:
                url = "https://" + url
            else:
                import urllib.parse
                query = urllib.parse.quote(url)
                url = f"https://www.google.com/search?q={query}"
        
        self.url_var.set(url)
        if hasattr(self, 'webview') and self.webview is not None:
            try:
                self.webview.load_url(url)
            except Exception as e:
                print("[ERROR] Navigation failed:", e)
                
    def _go_back(self):
        if hasattr(self, 'webview') and self.webview is not None:
            try:
                if hasattr(self.webview, 'web') and self.webview.web is not None:
                    self.webview.web.GoBack()
            except Exception as e:
                print("[ERROR] GoBack failed:", e)
                
    def _go_forward(self):
        if hasattr(self, 'webview') and self.webview is not None:
            try:
                if hasattr(self.webview, 'web') and self.webview.web is not None:
                    self.webview.web.GoForward()
            except Exception as e:
                print("[ERROR] GoForward failed:", e)
                
    def _reload(self):
        if hasattr(self, 'webview') and self.webview is not None:
            try:
                self.webview.reload()
            except Exception as e:
                print("[ERROR] Reload failed:", e)
                
    def _show_fallback_error(self, error):
        for widget in self.web_frame.winfo_children():
            widget.destroy()
            
        err_container = tk.Frame(self.web_frame, bg=BG, padx=40, pady=40)
        err_container.place(relx=0.5, rely=0.5, anchor="center")
        
        tk.Label(err_container, text=f"Could not load {self.title()}", bg=BG, fg=DANGER, font=("Segoe UI", 16, "bold")).pack(pady=(0, 10))
        
        msg = (
            "The 'tkwebview2' and 'pywebview' libraries are required to embed the browser interface.\n\n"
            "If you see this error, please make sure they are installed and you have the WebView2 runtime.\n"
            "Otherwise, you can open the website in your external browser instead."
        )
        tk.Label(err_container, text=msg, bg=BG, fg=TEXT, font=FB, justify="center", wraplength=500).pack(pady=10)
        
        btn_frame = tk.Frame(err_container, bg=BG)
        btn_frame.pack(pady=20)
        
        tk.Button(
            btn_frame, text="Open in Browser", command=self._open_external,
            bg=ACCENT, fg=SURFACE, activebackground=ACCENT, activeforeground=SURFACE,
            relief="flat", font=("Segoe UI", 11, "bold"), padx=15, pady=8, cursor="hand2"
        ).pack(side="left", padx=10)
        
        tk.Button(
            btn_frame, text="Retry", command=self._retry_load,
            bg=SURFACE, fg=TEXT, activebackground=BG, activeforeground=TEXT,
            relief="flat", highlightbackground=BORDER, highlightthickness=1,
            font=("Segoe UI", 11), padx=15, pady=8, cursor="hand2"
        ).pack(side="left", padx=10)
        
    def _open_external(self):
        import webbrowser
        webbrowser.open(self.url_var.get())
        self.destroy()
        
    def _retry_load(self):
        for widget in self.web_frame.winfo_children():
            widget.destroy()
        self.status_lbl.config(text="Retrying connection...", fg=MUTED)
        self.after(500, self._load_webview)
        
    def _on_close(self):
        if hasattr(self, 'webview') and self.webview is not None:
            try:
                self.webview.web.Dispose()
            except:
                pass
        self.destroy()





#Testing part pore changes hoibo eliga  seperate koira rakhsi #


# ─────────────────────────────────────────────────────────────────────────────
#  TEXT-ENTRY EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────
class TextEntryExperiment(tk.Toplevel):
    """
    Gaze-based text-entry experiment — Overt & Covert methods.

    Overt:  Stimulus shown for MEMORIZE_SECS → hidden → subject types from memory
    Covert: Stimulus permanently visible → subject types while reading

    Flow (both methods):
        INTRO  (method selection)
        INSTRUCTIONS  (5 s auto-advance)
        FIXATION  +  (5 s auto-advance)     ← repeated before each trial
        STIMULUS phase  (overt memorize or covert typing)
        RESULTS
    """

    # ── Test stimuli — 20 research phrases ────────────────────────────────
    ALL_PHRASES = [
        "The quick brown fox jumps over the lazy dog.",
        "Pack my box with five dozen liquor jugs.",
        "Sphinx of black quartz, judge my vow",
        "How vexingly quick daft zebras jump!",
        "The five boxing wizards jump quickly.",
        "Waltz, bad nymph, for quick jigs vex.",
    ]
    ALL_WORDS = [
        "water", "help", "food", "apple", "house", "smile", "table", "chair",
        "light", "clock", "sugar", "bread", "paper", "music", "river", "beach",
        "grass", "plant", "green", "happy", "cloud", "storm", "stone", "night"
    ]
    TRIALS_PER_SESSION = 2          # phrases randomly selected per session

    MEMORIZE_SECS = 15                     # overt: memorization window (15 s for testing)

    # ── Experiment colours (plain white theme) ──────────────────────────────
    _C_BG  = "#FFFFFF"   # white background
    _C_FG  = "#1A1D23"   # dark text
    _C_ACC = "#2563EB"   # blue accent  (overt)
    _C_BLU = "#2563EB"   # blue accent  (covert)
    _C_DIM = "#6B7280"   # muted gray
    _C_FIX = "#000000"   # black fixation cross
    _C_GRN = "#16A34A"

    # ── Fonts ─────────────────────────────────────────────────────────────────
    _F_TITLE = ("Segoe UI", 32, "bold")
    _F_HEAD  = ("Segoe UI", 22, "bold")
    _F_STIM  = ("Segoe UI", 36, "bold")   # word/sentence display
    _F_BODY  = ("Segoe UI", 15)
    _F_SMALL = ("Segoe UI", 12)
    _F_FIX   = ("Segoe UI", 120, "bold")

    # ── Per-method instruction text ───────────────────────────────────────────
    _OVERT_INSTR = (
        "Overt Method — Instructions",
        (
            "A word or sentence will appear on screen.\n\n"
            "• You have 2 minutes to memorize it carefully.\n"
            "• After the timer, the word will disappear.\n"
            "• Type what you remember using the gaze keyboard.\n"
            "• Press Save & Next when done.\n\n"
            "Look at the fixation cross  +  before each trial.\n"
            "Press  Esc  at any time to abort."
        )
    )
    _COVERT_INSTR = (
        "Covert Method — Instructions",
        (
            "A word or sentence will remain visible on screen.\n\n"
            "• The word / sentence stays visible the whole time.\n"
            "• Type exactly what you see using the gaze keyboard.\n"
            "• Press Save & Next when done.\n\n"
            "Look at the fixation cross  +  before each trial.\n"
            "Press  Esc  at any time to abort."
        )
    )

    # ─────────────────────────────────────────────────────────────────────────
    def __init__(self, master):
        super().__init__(master)
        self.title("Text-Entry Experiment")
        self.configure(bg=self._C_BG)
        self.state("zoomed")
        self.bind("<Escape>", lambda e: self.destroy())

        self._method            = None   # "overt" or "covert"
        self._idx               = 0
        self._responses         = []
        self._stimuli           = []     # 2 phrases chosen randomly each session
        self._participant_name  = ""     # entered on the name screen
        self._after_id          = None
        self._typing_frame      = None
        # ── Per-trial keystroke tracking ───────────────────────────────────────
        self._trial_start_time  = None
        self._spacebar_count    = 0
        self._backspace_count   = 0
        self._first_key_pressed = False
        self._prev_typed_text   = ""

        # Canvas used by display-only phases
        self._canvas = tk.Canvas(self, bg=self._C_BG, highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)

        # ── Heatmap tracking ───────────────────────────────────────────────────
        self._exp_start_time  = None   # datetime when typing phase begins
        self._gaze_csv_path   = None   # path to the live gaze log CSV

        self._show_name_entry()

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _cancel_after(self):
        if self._after_id:
            try: self.after_cancel(self._after_id)
            except Exception: pass
            self._after_id = None

    def _use_canvas(self):
        """Show canvas, destroy any typing frame."""
        if self._typing_frame:
            try: self._typing_frame.destroy()
            except Exception: pass
            self._typing_frame = None
        self._canvas.pack(fill="both", expand=True)

    def _use_frame(self):
        """Hide canvas, return a fresh typing frame."""
        self._cancel_after()
        self._canvas.pack_forget()
        self._canvas.delete("all")
        if self._typing_frame:
            try: self._typing_frame.destroy()
            except Exception: pass
        f = tk.Frame(self, bg=self._C_BG)
        f.pack(fill="both", expand=True)
        self._typing_frame = f
        return f

    def _clear(self):
        self._cancel_after()
        self._canvas.delete("all")

    def _reset_trial_tracking(self):
        """Reset per-trial keystroke counters before each typing phase."""
        self._trial_start_time  = None
        self._spacebar_count    = 0
        self._backspace_count   = 0
        self._first_key_pressed = False
        self._prev_typed_text   = ""
        self._keystroke_log     = []   # [(char, rel_sec), ...]

    def _cx(self): return self._canvas.winfo_width()  // 2 or self.winfo_screenwidth()  // 2
    def _cy(self): return self._canvas.winfo_height() // 2 or self.winfo_screenheight() // 2
    def _cw(self): return self._canvas.winfo_width()  or self.winfo_screenwidth()
    def _ch(self): return self._canvas.winfo_height() or self.winfo_screenheight()

    def _countdown_text(self, secs, tag="countdown"):
        self._canvas.delete(tag)
        self._canvas.create_text(
            self._cx(), self._cy() + int(self._ch() * 0.38),
            text=f"continuing in  {secs}s …",
            fill=self._C_DIM, font=self._F_SMALL, tags=tag)

    def _current_stim(self):
        return self._stimuli[self._idx] if self._idx < len(self._stimuli) else None

    # ─────────────────────────────────────────────────────────────────────────
    #  PHASE –1 — NAME ENTRY
    # ─────────────────────────────────────────────────────────────────────────
    def _show_name_entry(self):
        frame = self._use_frame()

        # Title block
        title_row = tk.Frame(frame, bg=self._C_BG, pady=28)
        title_row.pack(fill="x")
        tk.Label(title_row, text="Text-Entry Experiment",
                 bg=self._C_BG, fg=self._C_FG,
                 font=self._F_TITLE).pack()
        tk.Label(title_row,
                 text="Enter your name using the eye-gaze keyboard below, then press  Continue",
                 bg=self._C_BG, fg=self._C_DIM,
                 font=self._F_SMALL).pack(pady=(8, 0))

        # Name input
        inp_row = tk.Frame(frame, bg=self._C_BG, padx=40, pady=10)
        inp_row.pack(fill="x")
        tk.Label(inp_row, text="Your Name:",
                 bg=self._C_BG, fg=self._C_DIM, font=self._F_SMALL).pack(anchor="w")

        txt = tk.Text(inp_row, bg=self._C_BG, fg=self._C_FG,
                      insertbackground=self._C_FG,
                      font=("Segoe UI", 22), height=2, wrap="word",
                      relief="flat", padx=14, pady=12,
                      highlightbackground=self._C_DIM, highlightthickness=1)
        txt.pack(fill="x", pady=(6, 0))
        txt.focus_set()

        warn_lbl = tk.Label(inp_row, text="",
                            bg=self._C_BG, fg="#DC2626", font=self._F_SMALL)
        warn_lbl.pack(anchor="w", pady=(4, 0))

        def _continue():
            name = txt.get("1.0", "end").strip()
            if not name:
                warn_lbl.config(text="⚠  Please enter your name before continuing.")
                txt.config(highlightbackground="#DC2626", highlightthickness=2)
                self.after(2000, lambda: (
                    warn_lbl.config(text=""),
                    txt.config(highlightbackground=self._C_DIM, highlightthickness=1)
                ))
                return
            self._participant_name = name
            self._show_intro()

        tk.Button(inp_row, text="  Continue  →  ",
                  command=_continue,
                  bg=self._C_ACC, fg="#FFFFFF",
                  activebackground="#1d4ed8", activeforeground="#FFFFFF",
                  relief="flat",
                  font=("Segoe UI", 13, "bold"),
                  padx=18, pady=8, cursor="hand2").pack(anchor="e", pady=(12, 0))

        # On-screen keyboard
        kb = tk.Frame(frame, bg=self._C_BG)
        kb.pack(fill="both", expand=True, padx=8, pady=4)
        OnScreenKeyboard(kb, txt, layout="normal", notepad_app=None).pack(fill="both", expand=True)

    # ─────────────────────────────────────────────────────────────────────────
    #  PHASE 0 — INTRO  (method selection)
    # ─────────────────────────────────────────────────────────────────────────
    def _show_intro(self):
        self._use_canvas()
        self._clear()
        self.update_idletasks()

        c  = self._canvas
        cx, cy = self._cx(), self._cy()
        w, h   = self._cw(), self._ch()

        c.create_text(cx, cy-155, text="Text-Entry Experiment",
                      fill=self._C_FG, font=self._F_TITLE, anchor="center")
        c.create_line(cx-250, cy-115, cx+250, cy-115, fill=self._C_DIM, width=1)
        c.create_text(cx, cy-85,
                      text="Eye-gaze typing  ·  Choose your method below",
                      fill=self._C_DIM, font=self._F_SMALL, anchor="center")

        c.create_text(cx, cy-40,
                      text="Select Overt or Covert to begin the experiment:",
                      fill=self._C_FG, font=self._F_BODY, anchor="center")

        # ── Method buttons ────────────────────────────────────────────────────
        def method_btn(parent, title, sub, color, cmd):
            outer = tk.Frame(parent, bg=self._C_DIM, padx=1, pady=1, cursor="hand2")
            inner = tk.Frame(outer, bg=self._C_BG, padx=28, pady=18)
            inner.pack()
            lbl_t = tk.Label(inner, text=title, bg=self._C_BG, fg=color,
                             font=("Segoe UI", 18, "bold"))
            lbl_t.pack()
            lbl_s = tk.Label(inner, text=sub, bg=self._C_BG, fg=self._C_DIM,
                             font=self._F_SMALL, wraplength=190, justify="center")
            lbl_s.pack(pady=(6, 0))
            for w in (outer, inner, lbl_t, lbl_s):
                w.bind("<Button-1>", lambda e: cmd())
            return outer

        overt_f = method_btn(c,
            "OVERT",
            "Word shown for 2 min\nthen type from memory",
            self._C_ACC,
            lambda: self._begin("overt"))

        covert_f = method_btn(c,
            "COVERT",
            "Word stays visible;\ntype while reading",
            self._C_BLU,
            lambda: self._begin("covert"))

        c.create_window(cx - 170, cy + 130, window=overt_f,  anchor="center")
        c.create_window(cx + 170, cy + 130, window=covert_f, anchor="center")

        c.create_text(cx, h - 32,
                      text="Press  Esc  to close",
                      fill=self._C_DIM, font=self._F_SMALL, anchor="center")

    # ─────────────────────────────────────────────────────────────────────────
    def _begin(self, method):
        self._method    = method
        self._idx       = 0
        self._responses = []
        if method == "overt":
            words = random.sample(self.ALL_WORDS, 2)
            phrases = random.sample(self.ALL_PHRASES, 2)
            self._stimuli = words + phrases
        else:
            self._stimuli   = random.sample(self.ALL_PHRASES, self.TRIALS_PER_SESSION)
        # ── Mark experiment start time for heatmap ─────────────────────────────
        self._exp_start_time = datetime.datetime.now()
        # ── Find the active gaze log CSV from the running tracker ───────────────
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Tracker names the CSV after the user; fall back to gaze_log.csv
        candidate_csvs = sorted(
            [f for f in os.listdir(base_dir)
             if f.endswith("_gaze_log.csv") or f == "gaze_log.csv"],
            key=lambda f: os.path.getmtime(os.path.join(base_dir, f)),
            reverse=True
        )
        self._gaze_csv_path = os.path.join(base_dir, candidate_csvs[0]) \
            if candidate_csvs else None
        print(f"[HeatMap] Experiment started. Gaze CSV: {self._gaze_csv_path}")
        self._show_instructions()

    # ─────────────────────────────────────────────────────────────────────────
    #  PHASE 1 — INSTRUCTIONS  (5 s auto-advance)
    # ─────────────────────────────────────────────────────────────────────────
    def _show_instructions(self, secs=5):
        if secs == 5:
            self._use_canvas(); self._clear(); self.update_idletasks()

        c = self._canvas
        cx, cy = self._cx(), self._cy()
        instr = self._OVERT_INSTR if self._method == "overt" else self._COVERT_INSTR
        acc   = self._C_ACC if self._method == "overt" else self._C_BLU

        if secs == 5:
            c.create_text(cx, cy - 210, text=instr[0],
                          fill=self._C_FG, font=self._F_HEAD, anchor="center")
            c.create_line(cx-250, cy-175, cx+250, cy-175, fill=self._C_DIM, width=1)
            c.create_text(cx, cy - 20, text=instr[1],
                          fill=self._C_FG, font=self._F_BODY,
                          anchor="center", justify="center")

        self._countdown_text(secs)

        if secs > 0:
            self._after_id = self.after(1000, lambda: self._show_instructions(secs - 1))
        else:
            self._show_fixation()

    # ─────────────────────────────────────────────────────────────────────────
    #  PHASE 2 — FIXATION CROSS  (5 s auto-advance, shown before each trial)
    # ─────────────────────────────────────────────────────────────────────────
    def _show_fixation(self, secs=5):
        if secs == 5:
            self._use_canvas(); self._clear(); self.update_idletasks()

        c = self._canvas
        cx, cy = self._cx(), self._cy()
        acc    = self._C_ACC if self._method == "overt" else self._C_BLU

        if secs == 5:
            c.create_text(cx, cy - 190,
                          text=f"Trial  {self._idx + 1}  of  {len(self._stimuli)}",
                          fill=self._C_DIM, font=self._F_SMALL, anchor="center")
            c.create_text(cx, cy - 145,
                          text="Focus on the cross below",
                          fill=self._C_DIM, font=self._F_SMALL, anchor="center")
            c.create_text(cx, cy, text="+",
                          fill=self._C_FIX, font=self._F_FIX, anchor="center")

        self._countdown_text(secs)

        if secs > 0:
            self._after_id = self.after(1000, lambda: self._show_fixation(secs - 1))
        else:
            if self._method == "overt":
                self._show_overt_memorize()
            else:
                self._show_covert_typing()

    # ─────────────────────────────────────────────────────────────────────────
    #  PHASE 3-A — OVERT: memorize  (2-min countdown)
    # ─────────────────────────────────────────────────────────────────────────
    def _show_overt_memorize(self, secs=None):
        if secs is None:
            secs = self.MEMORIZE_SECS
            self._use_canvas(); self._clear(); self.update_idletasks()

        c  = self._canvas
        cx, cy = self._cx(), self._cy()
        ch     = self._ch()
        stim   = self._current_stim()

        if secs == self.MEMORIZE_SECS:
            c.create_text(cx, cy - 265, text="MEMORIZE  —  OVERT",
                          fill=self._C_DIM, font=("Segoe UI", 11, "bold"),
                          anchor="center")
            c.create_line(cx-250, cy-240, cx+250, cy-240, fill=self._C_DIM, width=1)
            # Word / sentence (large)
            c.create_text(cx, cy - 130, text=stim,
                          fill=self._C_FG, font=self._F_STIM,
                          anchor="center", justify="center",
                          width=int(self._cw() * 0.85 or 1000))
            # Hint
            c.create_text(cx, cy + 70,
                          text="Memorize this carefully.\nThe keyboard will appear after the timer ends.",
                          fill=self._C_DIM, font=self._F_SMALL,
                          anchor="center", justify="center")

        # Live countdown timer
        mins = secs // 60
        secs_r = secs % 60
        self._canvas.delete("timer")
        self._canvas.create_text(
            cx, ch - 55,
            text=f"⏱  {mins}:{secs_r:02d}  remaining",
            fill=self._C_DIM, font=("Segoe UI", 18, "bold"),
            tags="timer", anchor="center")

        if secs > 0:
            self._after_id = self.after(1000, lambda: self._show_overt_memorize(secs - 1))
        else:
            self._show_overt_typing()

    # ─────────────────────────────────────────────────────────────────────────
    #  PHASE 3-B — OVERT: type from memory
    # ─────────────────────────────────────────────────────────────────────────
    def _show_overt_typing(self):
        frame = self._use_frame()
        self._reset_trial_tracking()

        # Header
        hdr = tk.Frame(frame, bg=self._C_BG, padx=24, pady=14,
                       highlightbackground=self._C_DIM, highlightthickness=1)
        hdr.pack(fill="x")
        tk.Label(hdr, text="OVERT  ·  Type from memory",
                 bg=self._C_BG, fg=self._C_FG,
                 font=("Segoe UI", 14, "bold")).pack(side="left")
        tk.Label(hdr,
                 text=f"Trial  {self._idx + 1} / {len(self._stimuli)}",
                 bg=self._C_BG, fg=self._C_DIM,
                 font=self._F_SMALL).pack(side="right")

        # Input row
        inp_row = tk.Frame(frame, bg=self._C_BG, padx=32, pady=14)
        inp_row.pack(fill="x")
        tk.Label(inp_row, text="Type what you memorized:",
                 bg=self._C_BG, fg=self._C_DIM, font=self._F_SMALL).pack(anchor="w")

        txt = tk.Text(inp_row, bg=self._C_BG, fg=self._C_FG,
                      insertbackground=self._C_FG,
                      font=("Segoe UI", 20), height=2, wrap="word",
                      relief="flat", padx=14, pady=12,
                      highlightbackground=self._C_DIM, highlightthickness=1)
        txt.pack(fill="x", pady=(6, 0))
        txt.focus_set()

        # ── Keystroke tracking (works for both physical & on-screen keyboard) ──
        def _track_changes(event=None):
            if not txt.edit_modified():
                return
            txt.edit_modified(False)
            current = txt.get("1.0", "end-1c")
            prev    = self._prev_typed_text
            now     = datetime.datetime.now()
            if not self._first_key_pressed and current != prev:
                self._trial_start_time  = now
                self._first_key_pressed = True
            rel_time = round((now - self._trial_start_time).total_seconds(), 3) \
                       if self._trial_start_time else 0.0
            if len(current) > len(prev):
                added = current[len(prev):]
                self._spacebar_count += added.count(" ")
                for ch in added:
                    self._keystroke_log.append([ch if ch != " " else "<SP>", rel_time])
            elif len(current) < len(prev):
                n_del = len(prev) - len(current)
                self._backspace_count += n_del
                for _ in range(n_del):
                    self._keystroke_log.append(["<BS>", rel_time])
            self._prev_typed_text = current

        txt.bind("<<Modified>>", _track_changes, add="+")

        def _save():
            typed      = txt.get("1.0", "end").strip()
            end_time   = datetime.datetime.now()
            start_time = self._trial_start_time
            duration   = round((end_time - start_time).total_seconds(), 3) \
                         if start_time else ""
            self._responses.append({
                "participant_name"   : self._participant_name,
                "trial_number"       : self._idx + 1,
                "method"             : "overt",
                "stimulus"           : self._current_stim(),
                "typed_response"     : typed,
                "is_correct"         : typed.strip().lower() == (self._current_stim() or "").strip().lower(),
                "trial_start_time"   : start_time.isoformat() if start_time else "",
                "trial_end_time"     : end_time.isoformat(),
                "typing_duration_sec": duration,
                "spacebar_count"     : self._spacebar_count,
                "backspace_count"    : self._backspace_count,
                "char_count"         : len(typed),
                "word_count"         : len(typed.split()) if typed else 0,
                "letter_timestamps"  : json.dumps(self._keystroke_log),
            })
            self._advance()

        tk.Button(inp_row, text="  ✔  Save & Next  ",
                  command=_save,
                  bg=self._C_BG, fg=self._C_FG,
                  activebackground="#EFEFEF", activeforeground=self._C_FG,
                  relief="solid", bd=1,
                  font=("Segoe UI", 13, "bold"),
                  padx=18, pady=8, cursor="hand2").pack(anchor="e", pady=(10, 0))

        # Keyboard
        kb = tk.Frame(frame, bg=self._C_BG)
        kb.pack(fill="both", expand=True, padx=8, pady=4)
        OnScreenKeyboard(kb, txt, layout="normal", notepad_app=None).pack(fill="both", expand=True)

    # ─────────────────────────────────────────────────────────────────────────
    #  PHASE 4 — COVERT: word visible + keyboard
    # ─────────────────────────────────────────────────────────────────────────
    def _show_covert_typing(self):
        frame = self._use_frame()
        stim  = self._current_stim()
        self._reset_trial_tracking()

        # Header
        hdr = tk.Frame(frame, bg=self._C_BG, padx=24, pady=14,
                       highlightbackground=self._C_DIM, highlightthickness=1)
        hdr.pack(fill="x")
        tk.Label(hdr, text="COVERT  ·  Type what you see",
                 bg=self._C_BG, fg=self._C_FG,
                 font=("Segoe UI", 14, "bold")).pack(side="left")
        tk.Label(hdr,
                 text=f"Trial  {self._idx + 1} / {len(self._stimuli)}",
                 bg=self._C_BG, fg=self._C_DIM,
                 font=self._F_SMALL).pack(side="right")

        # Permanently visible stimulus
        stim_row = tk.Frame(frame, bg=self._C_BG, pady=8,
                            highlightbackground=self._C_DIM, highlightthickness=1)
        stim_row.pack(fill="x")
        tk.Label(stim_row, text="Read & type →",
                 bg=self._C_BG, fg=self._C_DIM, font=self._F_SMALL).pack()
        tk.Label(stim_row, text=stim,
                 bg=self._C_BG, fg=self._C_FG, font=self._F_STIM,
                 wraplength=int(self.winfo_screenwidth() * 0.85),
                 justify="center").pack(pady=(2, 0))

        # Input row
        inp_row = tk.Frame(frame, bg=self._C_BG, padx=32, pady=4)
        inp_row.pack(fill="x")
        tk.Label(inp_row, text="Your typed response:",
                 bg=self._C_BG, fg=self._C_DIM, font=self._F_SMALL).pack(anchor="w")

        txt = tk.Text(inp_row, bg=self._C_BG, fg=self._C_FG,
                      insertbackground=self._C_FG,
                      font=("Segoe UI", 20), height=2, wrap="word",
                      relief="flat", padx=14, pady=8,
                      highlightbackground=self._C_DIM, highlightthickness=1)
        txt.pack(fill="x", pady=(2, 0))
        txt.focus_set()

        # ── Keystroke tracking (works for both physical & on-screen keyboard) ──
        def _track_changes(event=None):
            if not txt.edit_modified():
                return
            txt.edit_modified(False)
            current = txt.get("1.0", "end-1c")
            prev    = self._prev_typed_text
            now     = datetime.datetime.now()
            if not self._first_key_pressed and current != prev:
                self._trial_start_time  = now
                self._first_key_pressed = True
            rel_time = round((now - self._trial_start_time).total_seconds(), 3) \
                       if self._trial_start_time else 0.0
            if len(current) > len(prev):
                added = current[len(prev):]
                self._spacebar_count += added.count(" ")
                for ch in added:
                    self._keystroke_log.append([ch if ch != " " else "<SP>", rel_time])
            elif len(current) < len(prev):
                n_del = len(prev) - len(current)
                self._backspace_count += n_del
                for _ in range(n_del):
                    self._keystroke_log.append(["<BS>", rel_time])
            self._prev_typed_text = current

        txt.bind("<<Modified>>", _track_changes, add="+")

        def _save():
            typed      = txt.get("1.0", "end").strip()
            end_time   = datetime.datetime.now()
            start_time = self._trial_start_time
            duration   = round((end_time - start_time).total_seconds(), 3) \
                         if start_time else ""
            self._responses.append({
                "participant_name"   : self._participant_name,
                "trial_number"       : self._idx + 1,
                "method"             : "covert",
                "stimulus"           : self._current_stim(),
                "typed_response"     : typed,
                "is_correct"         : typed.strip().lower() == (self._current_stim() or "").strip().lower(),
                "trial_start_time"   : start_time.isoformat() if start_time else "",
                "trial_end_time"     : end_time.isoformat(),
                "typing_duration_sec": duration,
                "spacebar_count"     : self._spacebar_count,
                "backspace_count"    : self._backspace_count,
                "char_count"         : len(typed),
                "word_count"         : len(typed.split()) if typed else 0,
                "letter_timestamps"  : json.dumps(self._keystroke_log),
            })
            self._advance()

        tk.Button(inp_row, text="  ✔  Save & Next  ",
                  command=_save,
                  bg=self._C_BG, fg=self._C_FG,
                  activebackground="#EFEFEF", activeforeground=self._C_FG,
                  relief="solid", bd=1,
                  font=("Segoe UI", 13, "bold"),
                  padx=18, pady=8, cursor="hand2").pack(anchor="e", pady=(10, 0))

        # Keyboard
        kb = tk.Frame(frame, bg=self._C_BG)
        kb.pack(fill="both", expand=True, padx=8, pady=4)
        OnScreenKeyboard(kb, txt, layout="normal", notepad_app=None).pack(fill="both", expand=True)

    # ─────────────────────────────────────────────────────────────────────────
    #  Trial advance
    # ─────────────────────────────────────────────────────────────────────────
    def _advance(self):
        self._idx += 1
        if self._idx < len(self._stimuli):
            self._show_fixation()   # fixation before every new trial
        else:
            self._save_csv_and_finish()

    # ─────────────────────────────────────────────────────────────────────────
    #  HEATMAP — generated from tracker gaze data during this experiment
    # ─────────────────────────────────────────────────────────────────────────
    def _generate_experiment_heatmap(self, csv_filepath: str):
        """
        Read gaze points that were logged by tracker.py during this experiment
        session and generate a heatmap PNG named after the experiment CSV.

        The tracker's gaze CSV contains a `timestamp` column (seconds since
        session start).  We use the wall-clock start time we recorded in
        _begin() to extract only the rows that fall within the experiment
        window.
        """
        if not self._gaze_csv_path or not os.path.exists(self._gaze_csv_path):
            print("[HeatMap] No gaze CSV found — skipping experiment heatmap.")
            return
        if not self._exp_start_time:
            return

        try:
            import pandas as pd
            df = pd.read_csv(self._gaze_csv_path)
            if "gaze_x" not in df.columns or "gaze_y" not in df.columns:
                print("[HeatMap] Gaze CSV missing required columns.")
                return

            # The tracker's `timestamp` is seconds elapsed since session start.
            # Use the row count written before vs after to slice the window.
            # Simpler: use all rows if tracker just started; otherwise use the
            # last N rows added since experiment began.
            # We compare mtime of the CSV to our start time to estimate rows.
            # Most robust: grab all gaze data (the heatmap covers the whole session).
            gaze_pts = list(zip(df["gaze_x"].tolist(), df["gaze_y"].tolist()))

            if not gaze_pts:
                print("[HeatMap] No gaze points collected.")
                return

            # Output: same folder as the experiment CSV, same base name
            hm_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heatmaps")
            os.makedirs(hm_dir, exist_ok=True)
            base   = os.path.basename(os.path.splitext(csv_filepath)[0])
            outf   = os.path.join(hm_dir, base + "_heatmap.png")
            label  = os.path.basename(csv_filepath)

            sw = self.winfo_screenwidth()  or 1920
            sh = self.winfo_screenheight() or 1080

            _try_generate_heatmap(gaze_pts, outf, sw, sh, label)
            print(f"[HeatMap] Generating experiment heatmap -> {outf}")

        except Exception as e:
            print(f"[HeatMap] Could not generate experiment heatmap: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    #  PHASE 5 — SAVE CSV & FINISH
    # ─────────────────────────────────────────────────────────────────────────

    def _save_csv_and_finish(self):
        # ── Determine output path ────────────────────────────────────────────
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_dir  = os.path.join(base_dir, "csv")
        os.makedirs(csv_dir, exist_ok=True)

        ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = self._participant_name.strip().lower().replace(" ", "_") or "unknown"
        filename  = f"experiment_{self._method}_{safe_name}_{ts}.csv"
        filepath  = os.path.join(csv_dir, filename)

        # ── Write CSV ────────────────────────────────────────────────────────
        fieldnames = [
            "participant_name", "trial_number", "method", "stimulus",
            "typed_response", "is_correct", "trial_start_time", "trial_end_time",
            "typing_duration_sec", "spacebar_count", "backspace_count",
            "char_count", "word_count", "letter_timestamps",
        ]
        try:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self._responses)
            save_ok = True

            # Automatically run latex.py to generate the LaTeX report
            try:
                latex_script = os.path.join(base_dir, "latex.py")
                subprocess.Popen([sys.executable, latex_script, filepath])
            except Exception as e:
                print(f"[ERROR] Failed to auto-generate LaTeX: {e}")
        except Exception as exc:
            save_ok  = False
            err_msg  = str(exc)

        # ── Generate gaze heatmap for this experiment session ─────────────────
        if save_ok:
            self._generate_experiment_heatmap(filepath)

        # ── Confirmation screen ──────────────────────────────────────────────
        self._use_canvas(); self._clear(); self.update_idletasks()
        c  = self._canvas
        cx, cy = self._cx(), self._cy()
        h      = self._ch()

        if save_ok:
            c.create_text(cx, cy - 90,
                          text="✓  Experiment Complete",
                          fill=self._C_GRN, font=self._F_HEAD, anchor="center")
            c.create_text(cx, cy - 45,
                          text=f"{self._method.upper()}  ·  {len(self._responses)} trial(s) saved to CSV",
                          fill=self._C_DIM, font=self._F_BODY, anchor="center")
            c.create_text(cx, cy,
                          text="Saved to:",
                          fill=self._C_DIM, font=self._F_SMALL, anchor="center")
            c.create_text(cx, cy + 32,
                          text=f"csv/{filename}",
                          fill=self._C_FG, font=("Segoe UI", 13, "bold"), anchor="center")
        else:
            c.create_text(cx, cy - 60,
                          text="⚠  Could not save CSV",
                          fill="#DC2626", font=self._F_HEAD, anchor="center")
            c.create_text(cx, cy,
                          text=err_msg,
                          fill=self._C_DIM, font=self._F_SMALL, anchor="center",
                          width=int(self._cw() * 0.7))

        # Buttons
        btns = tk.Frame(c, bg=self._C_BG)
        tk.Button(btns, text="  ↺  Restart  ",
                  command=self._show_name_entry,
                  bg=self._C_BG, fg=self._C_FG,
                  activebackground="#EFEFEF", activeforeground=self._C_FG,
                  relief="solid", bd=1,
                  font=("Segoe UI", 13, "bold"),
                  padx=16, pady=8, cursor="hand2").pack(side="left", padx=6)
        tk.Button(btns, text="  ✕  Close  ",
                  command=self.destroy,
                  bg=self._C_BG, fg=self._C_FG,
                  activebackground="#EFEFEF", activeforeground=self._C_FG,
                  relief="solid", bd=1, font=("Segoe UI", 13),
                  padx=16, pady=8, cursor="hand2").pack(side="left", padx=6)

        c.create_window(cx, cy + 90, window=btns, anchor="center")





# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # To support WebView2, the Tkinter event loop must run in a Single-Threaded Apartment (STA) thread.
    # Since pythonnet initializes the main thread as MTA, we start the Tkinter app in a new STA thread.
    try:
        import clr
        clr.AddReference('System.Threading')
        from System.Threading import Thread, ApartmentState, ThreadStart
        
        def start_gui():
            app = Launcher()
            app.mainloop()
            
        t = Thread(ThreadStart(start_gui))
        t.ApartmentState = ApartmentState.STA
        t.Start()
        t.Join()
    except Exception as e:
        # Fallback to running on the main thread if pythonnet/CLR is not available
        app = Launcher()
        app.mainloop()