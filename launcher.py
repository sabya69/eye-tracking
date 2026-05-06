
import tkinter as tk
from tkinter import font as tkfont, filedialog
import subprocess, sys, os, datetime, random, json

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
            font=FB, cursor="hand2")

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

        text_wrapper = tk.Frame(frame, bg=SURFACE)
        text_wrapper.place(relx=0, rely=0, relwidth=1, relheight=0.33)

        self.kb_wrapper = tk.Frame(frame, bg=BG)
        self.kb_wrapper.place(relx=0, rely=0.33, relwidth=1, relheight=0.67)

        self._tfont = tkfont.Font(family="Consolas", size=13)
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
                    cursor="hand2",
                    command=lambda k=key: self._press(k)
                )
                btn.grid(row=0, column=c, sticky="nsew", padx=4)
            row_frame.rowconfigure(0, weight=1)

        # Info label
        pad.rowconfigure(len(self.ROWS), weight=0)
        pad.columnconfigure(0, weight=1)
        tk.Label(pad, text="After Clicking the keys using tracker you can type in notepad",
                 bg=BG, fg=MUTED, font=FS).grid(row=len(self.ROWS), column=0, pady=(10,0))

    def _press(self, key):
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
if __name__ == "__main__":
    app = Launcher()
    app.mainloop()