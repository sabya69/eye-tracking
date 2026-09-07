import os
import sys
import json
import time
import threading
import subprocess
import urllib.request
import urllib.parse
import tkinter as tk
from tkinter import messagebox, font as tkfont, simpledialog

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# ── Color Palette & Styles (Light Mode / White Theme AAC Layout) ─────────────
BG_COLOR       = "#F8FAFC"   # Clean light background
SURFACE_COLOR  = "#FFFFFF"   # Pure white surface
TEXT_MAIN      = "#0F172A"   # High contrast dark slate text
TEXT_MUTED     = "#475569"   # Muted gray text

GREEN_TILE     = "#16A34A"   # Primary green phrase tile
GREEN_TILE_HOV = "#15803D"
ORANGE_TILE    = "#EA580C"   # Quick answer orange phrase tile
ORANGE_TILE_HOV= "#C2410C"

RED_BTN        = "#DC2626"   # Emergency red button
RED_BTN_HOV    = "#B91C1C"
BLUE_BTN       = "#2563EB"   # Utility blue
PURPLE_BTN     = "#7C3AED"   # Add phrase purple

BORDER_COLOR   = "#E2E8F0"

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emergency_config.json")
LOG_FILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emergency_sent_log.txt")

# ── Default Emergency Configuration ─────────────────────────────────────────
DEFAULT_CONFIG = {
    "emergency_contact": "+1 (555) 019-2831",
    "dispatch_channel": "simulation",  # 'simulation', 'twilio', 'webhook'
    "twilio_account_sid": "",
    "twilio_auth_token": "",
    "twilio_from_number": "",
    "webhook_url": "",
    "phrases": [
        {"text": "I need immediate help!", "category": "green"},
        {"text": "Please call emergency services!", "category": "green"},
        {"text": "I am in severe pain.", "category": "green"},
        {"text": "I feel dizzy / unwell.", "category": "green"},
        {"text": "Please bring me water.", "category": "green"},
        {"text": "I need my medication.", "category": "green"},
        {"text": "Please adjust my position.", "category": "green"},
        {"text": "I need to go to the bathroom.", "category": "green"},
        {"text": "Please call my family.", "category": "green"},
        {"text": "I am okay, don't worry.", "category": "green"},
        {"text": "It's good to see you.", "category": "green"},
        {"text": "Excuse me.", "category": "green"},
        {"text": "Yes", "category": "orange"},
        {"text": "No", "category": "orange"},
        {"text": "Maybe", "category": "orange"},
        {"text": "Of course", "category": "orange"},
        {"text": "Please", "category": "orange"},
        {"text": "Thank you", "category": "orange"},
        {"text": "Why?", "category": "orange"},
        {"text": "Why not?", "category": "orange"}
    ]
}


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "emergency_contact" in data and "phrases" in data:
                    for k, v in DEFAULT_CONFIG.items():
                        if k not in data:
                            data[k] = v
                    return data
        except Exception as e:
            print(f"[Emergency] Error loading config: {e}")
    save_config(DEFAULT_CONFIG)
    return DEFAULT_CONFIG


def save_config(config_data):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)
    except Exception as e:
        print(f"[Emergency] Error saving config: {e}")


def speak_text(text):
    """Text-to-speech out load in background thread."""
    if not text:
        return
    def _run_tts():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            return
        except Exception:
            pass

        if sys.platform == "win32":
            try:
                ps_script = f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{text}");'
                subprocess.run(["powershell", "-Command", ps_script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                print(f"[TTS Fallback Error]: {e}")

    threading.Thread(target=_run_tts, daemon=True).start()


def play_alarm_sound():
    """Play loud emergency alarm sound."""
    def _alarm():
        try:
            if HAS_WINSOUND:
                for _ in range(3):
                    winsound.Beep(2500, 400)
                    time.sleep(0.1)
        except Exception:
            pass
    threading.Thread(target=_alarm, daemon=True).start()


def dispatch_real_phone_signal(contact_num, message_text, config_data):
    """Dispatches emergency signal via Twilio, Webhook, or Local Simulation."""
    channel = config_data.get("dispatch_channel", "simulation")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Always log locally
    log_entry = f"[{timestamp}] [Channel: {channel}] TO: {contact_num} | MESSAGE: {message_text}\n"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        print(f"[Log Write Error]: {e}")

    play_alarm_sound()

    status_msg = f"Alert logged locally for {contact_num}."

    if channel == "twilio":
        sid = config_data.get("twilio_account_sid", "").strip()
        auth = config_data.get("twilio_auth_token", "").strip()
        from_num = config_data.get("twilio_from_number", "").strip()

        if sid and auth and from_num:
            try:
                import base64
                url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
                payload = urllib.parse.urlencode({
                    "From": from_num,
                    "To": contact_num,
                    "Body": f"🚨 EMERGENCY ALERT: {message_text}"
                }).encode("utf-8")
                req = urllib.request.Request(url, data=payload, method="POST")
                creds = base64.b64encode(f"{sid}:{auth}".encode("utf-8")).decode("utf-8")
                req.add_header("Authorization", f"Basic {creds}")

                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status in (200, 201):
                        status_msg = f"📱 Real SMS sent to {contact_num} via Twilio!"
            except Exception as e:
                status_msg = f"Twilio SMS Failed: {e}"

    elif channel == "webhook":
        wh_url = config_data.get("webhook_url", "").strip()
        if wh_url:
            try:
                payload = json.dumps({
                    "to": contact_num,
                    "message": message_text,
                    "timestamp": timestamp,
                    "event": "EMERGENCY_IMPAIRED_PERSON_HELP"
                }).encode("utf-8")
                req = urllib.request.Request(wh_url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    status_msg = f"🌐 Webhook Alert Dispatched successfully ({resp.status})!"
            except Exception as e:
                status_msg = f"Webhook Alert Failed: {e}"

    return status_msg


class AlertSettingsDialog(tk.Toplevel):
    """Configuration dialog for phone alert channel settings (Twilio / Webhook / Local)."""
    def __init__(self, parent, config_data):
        super().__init__(parent)
        self.title("Phone Alert Dispatch Setup")
        self.configure(bg=SURFACE_COLOR)
        self.geometry("640x520")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.config_data = config_data

        tk.Label(self, text="⚙ Emergency Phone Alert Setup", font=("Segoe UI", 16, "bold"),
                 bg=SURFACE_COLOR, fg=TEXT_MAIN).pack(pady=(20, 10))

        # Channel selection
        chan_frame = tk.Frame(self, bg=SURFACE_COLOR)
        cat_lbl = tk.Label(chan_frame, text="Alert Method:", font=("Segoe UI", 12, "bold"), bg=SURFACE_COLOR, fg=TEXT_MAIN)
        cat_lbl.pack(anchor="w")

        self.chan_var = tk.StringVar(value=config_data.get("dispatch_channel", "simulation"))
        
        channels = [
            ("Simulation / Local Alarm & TTS (Default)", "simulation"),
            ("Twilio SMS API (Real Cellular SMS)", "twilio"),
            ("Custom Webhook (IFTTT / Fast2SMS API)", "webhook")
        ]

        for label, val in channels:
            tk.Radiobutton(chan_frame, text=label, variable=self.chan_var, value=val,
                           bg=SURFACE_COLOR, fg=TEXT_MAIN, selectcolor=SURFACE_COLOR,
                           font=("Segoe UI", 11)).pack(anchor="w", padx=10, pady=2)
        chan_frame.pack(fill="x", padx=30, pady=10)

        # Twilio Fields
        tw_frame = tk.LabelFrame(self, text="Twilio SMS Setup", font=("Segoe UI", 11, "bold"),
                                 bg=SURFACE_COLOR, fg=ORANGE_TILE, bd=1, padx=15, pady=8)
        tw_frame.pack(fill="x", padx=30, pady=6)

        tk.Label(tw_frame, text="Account SID:", bg=SURFACE_COLOR, fg=TEXT_MUTED).grid(row=0, column=0, sticky="e")
        self.tw_sid_var = tk.StringVar(value=config_data.get("twilio_account_sid", ""))
        tk.Entry(tw_frame, textvariable=self.tw_sid_var, width=45, bg=BG_COLOR).grid(row=0, column=1, padx=6, pady=3)

        tk.Label(tw_frame, text="Auth Token:", bg=SURFACE_COLOR, fg=TEXT_MUTED).grid(row=1, column=0, sticky="e")
        self.tw_auth_var = tk.StringVar(value=config_data.get("twilio_auth_token", ""))
        tk.Entry(tw_frame, textvariable=self.tw_auth_var, width=45, show="*", bg=BG_COLOR).grid(row=1, column=1, padx=6, pady=3)

        tk.Label(tw_frame, text="From Number:", bg=SURFACE_COLOR, fg=TEXT_MUTED).grid(row=2, column=0, sticky="e")
        self.tw_from_var = tk.StringVar(value=config_data.get("twilio_from_number", ""))
        tk.Entry(tw_frame, textvariable=self.tw_from_var, width=45, bg=BG_COLOR).grid(row=2, column=1, padx=6, pady=3)

        # Webhook Fields
        wh_frame = tk.LabelFrame(self, text="Webhook Setup", font=("Segoe UI", 11, "bold"),
                                 bg=SURFACE_COLOR, fg=BLUE_BTN, bd=1, padx=15, pady=8)
        wh_frame.pack(fill="x", padx=30, pady=6)

        tk.Label(wh_frame, text="Webhook URL:", bg=SURFACE_COLOR, fg=TEXT_MUTED).grid(row=0, column=0, sticky="e")
        self.wh_url_var = tk.StringVar(value=config_data.get("webhook_url", ""))
        tk.Entry(wh_frame, textvariable=self.wh_url_var, width=45, bg=BG_COLOR).grid(row=0, column=1, padx=6, pady=3)

        # Save Button
        btn_f = tk.Frame(self, bg=SURFACE_COLOR)
        btn_f.pack(pady=15)

        tk.Button(btn_f, text="Save Settings", font=("Segoe UI", 13, "bold"),
                  bg=GREEN_TILE, fg="#FFFFFF", activebackground=GREEN_TILE_HOV,
                  relief="flat", padx=24, pady=8, cursor="hand2", command=self._save).pack(side="left", padx=10)
        
        tk.Button(btn_f, text="Cancel", font=("Segoe UI", 13),
                  bg=BORDER_COLOR, fg=TEXT_MAIN, relief="flat", padx=20, pady=8,
                  cursor="hand2", command=self.destroy).pack(side="left", padx=10)

    def _save(self):
        self.config_data["dispatch_channel"] = self.chan_var.get()
        self.config_data["twilio_account_sid"] = self.tw_sid_var.get().strip()
        self.config_data["twilio_auth_token"] = self.tw_auth_var.get().strip()
        self.config_data["twilio_from_number"] = self.tw_from_var.get().strip()
        self.config_data["webhook_url"] = self.wh_url_var.get().strip()
        save_config(self.config_data)
        messagebox.showinfo("Settings Saved", "Emergency alert setup updated successfully!", parent=self)
        self.destroy()


class CustomPhraseDialog(tk.Toplevel):
    """On-screen gaze & keyboard friendly dialog with integrated visual keyboard."""
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Add Custom Emergency Phrase")
        self.configure(bg=SURFACE_COLOR)
        self.geometry("980x660")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.result = None

        tk.Label(self, text="Write Your Custom Phrase", font=("Segoe UI", 18, "bold"),
                 bg=SURFACE_COLOR, fg=TEXT_MAIN).pack(pady=(16, 8))

        # Text Field Display
        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(self, textvariable=self.entry_var, font=("Segoe UI", 20, "bold"),
                              bg=BG_COLOR, fg=TEXT_MAIN, insertbackground=TEXT_MAIN,
                              bd=2, relief="groove", justify="left")
        self.entry.pack(padx=30, pady=8, fill="x", ipady=8)
        self.entry.focus_set()

        # Category Selection Frame
        cat_frame = tk.Frame(self, bg=SURFACE_COLOR)
        cat_frame.pack(pady=6)
        
        self.cat_var = tk.StringVar(value="green")
        tk.Radiobutton(cat_frame, text="Standard Message (Green)", variable=self.cat_var, value="green",
                       bg=SURFACE_COLOR, fg=TEXT_MAIN, selectcolor=SURFACE_COLOR, activebackground=SURFACE_COLOR,
                       activeforeground=TEXT_MAIN, font=("Segoe UI", 12, "bold")).pack(side="left", padx=20)
        tk.Radiobutton(cat_frame, text="Quick Response (Orange)", variable=self.cat_var, value="orange",
                       bg=SURFACE_COLOR, fg=TEXT_MAIN, selectcolor=SURFACE_COLOR, activebackground=SURFACE_COLOR,
                       activeforeground=TEXT_MAIN, font=("Segoe UI", 12, "bold")).pack(side="left", padx=20)

        # ── On-Screen Virtual Keyboard Container ────────────────────────────
        kb_frame = tk.Frame(self, bg=SURFACE_COLOR, padx=20, pady=10)
        kb_frame.pack(fill="both", expand=True)

        rows = [
            ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
            ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
            ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
            ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "?"]
        ]

        for r_idx, row_keys in enumerate(rows):
            rf = tk.Frame(kb_frame, bg=SURFACE_COLOR)
            rf.pack(pady=4)
            for k in row_keys:
                btn = tk.Button(rf, text=k, font=("Segoe UI", 14, "bold"), width=4, height=1,
                                bg="#F1F5F9", fg=TEXT_MAIN, activebackground="#CBD5E1",
                                activeforeground=TEXT_MAIN, relief="solid", bd=1, cursor="hand2",
                                command=lambda char=k: self._press_key(char))
                btn.pack(side="left", padx=4)

        # Special Action Keys Row
        special_rf = tk.Frame(kb_frame, bg=SURFACE_COLOR)
        special_rf.pack(pady=8)

        tk.Button(special_rf, text="␣ SPACE", font=("Segoe UI", 13, "bold"), width=16, height=1,
                  bg="#F1F5F9", fg=TEXT_MAIN, activebackground="#CBD5E1",
                  activeforeground=TEXT_MAIN, relief="solid", bd=1, cursor="hand2",
                  command=lambda: self._press_key(" ")).pack(side="left", padx=6)

        tk.Button(special_rf, text="⌫ BACKSPACE", font=("Segoe UI", 13, "bold"), width=14, height=1,
                  bg="#F1F5F9", fg=TEXT_MAIN, activebackground="#CBD5E1",
                  activeforeground=TEXT_MAIN, relief="solid", bd=1, cursor="hand2",
                  command=self._backspace).pack(side="left", padx=6)

        tk.Button(special_rf, text="CLEAR", font=("Segoe UI", 13, "bold"), width=10, height=1,
                  bg="#F1F5F9", fg=TEXT_MAIN, activebackground="#CBD5E1",
                  activeforeground=TEXT_MAIN, relief="solid", bd=1, cursor="hand2",
                  command=lambda: self.entry_var.set("")).pack(side="left", padx=6)

        # ── Dialog Action Buttons (Add / Cancel) ─────────────────────────────
        btn_frame = tk.Frame(self, bg=SURFACE_COLOR, pady=12)
        btn_frame.pack(fill="x", side="bottom")

        tk.Button(btn_frame, text="Add Phrase", font=("Segoe UI", 15, "bold"),
                  bg=GREEN_TILE, fg="#FFFFFF", activebackground=GREEN_TILE_HOV,
                  activeforeground="#FFFFFF", relief="flat", padx=28, pady=8,
                  cursor="hand2", command=self._on_add).pack(side="left", padx=(40, 10))

        tk.Button(btn_frame, text="Cancel", font=("Segoe UI", 15),
                  bg=BORDER_COLOR, fg=TEXT_MAIN, activebackground="#CBD5E1",
                  activeforeground=TEXT_MAIN, relief="flat", padx=28, pady=8,
                  cursor="hand2", command=self.destroy).pack(side="right", padx=(10, 40))

        self.bind("<Return>", lambda e: self._on_add())
        self.bind("<Escape>", lambda e: self.destroy())

    def _press_key(self, char):
        self.entry_var.set(self.entry_var.get() + char)

    def _backspace(self):
        curr = self.entry_var.get()
        if curr:
            self.entry_var.set(curr[:-1])

    def _on_add(self):
        phrase = self.entry_var.get().strip()
        if phrase:
            self.result = {"text": phrase, "category": self.cat_var.get()}
            self.destroy()


class EmergencyWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Emergency Communication")
        self.configure(bg=BG_COLOR)
        self.attributes('-fullscreen', True)
        
        self.bind("<Escape>", lambda e: self.attributes('-fullscreen', False))
        self.bind("<F11>", lambda e: self.attributes('-fullscreen', not self.attributes('-fullscreen')))

        self.config_data = load_config()
        self.remove_mode = False

        self._build_ui()
        self._refresh_phrases_grid()

    def _build_ui(self):
        # ── Top Display Header ───────────────────────────────────────────────
        header = tk.Frame(self, bg=SURFACE_COLOR, padx=24, pady=16, highlightbackground=BORDER_COLOR, highlightthickness=1)
        header.pack(fill="x")

        # Title & Emergency Contact info
        left_hdr = tk.Frame(header, bg=SURFACE_COLOR)
        left_hdr.pack(side="left")

        tk.Label(left_hdr, text="🚨 Emergency Communication", font=("Segoe UI", 20, "bold"),
                 bg=SURFACE_COLOR, fg=RED_BTN).pack(anchor="w")

        chan = self.config_data.get("dispatch_channel", "simulation").upper()
        self.contact_var = tk.StringVar(value=f"Emergency Contact: {self.config_data.get('emergency_contact', '')}  [{chan} MODE]")
        self.contact_lbl = tk.Label(left_hdr, textvariable=self.contact_var, font=("Segoe UI", 12),
                                    bg=SURFACE_COLOR, fg=TEXT_MUTED)
        self.contact_lbl.pack(anchor="w")

        right_hdr = tk.Frame(header, bg=SURFACE_COLOR)
        right_hdr.pack(side="right")

        tk.Button(right_hdr, text="⚙ Alert Setup", font=("Segoe UI", 12, "bold"),
                  bg=SURFACE_COLOR, fg=TEXT_MAIN, activebackground=BORDER_COLOR,
                  activeforeground=TEXT_MAIN, relief="solid", bd=1, padx=14, pady=8,
                  cursor="hand2", command=self._open_alert_settings).pack(side="left", padx=6)

        tk.Button(right_hdr, text="✏ Edit Contact", font=("Segoe UI", 12, "bold"),
                  bg=SURFACE_COLOR, fg=TEXT_MAIN, activebackground=BORDER_COLOR,
                  activeforeground=TEXT_MAIN, relief="solid", bd=1, padx=14, pady=8,
                  cursor="hand2", command=self._edit_contact).pack(side="left", padx=6)

        tk.Button(right_hdr, text="🚨 SEND ALERT NOW", font=("Segoe UI", 13, "bold"),
                  bg=RED_BTN, fg="#FFFFFF", activebackground=RED_BTN_HOV,
                  activeforeground="#FFFFFF", relief="flat", padx=18, pady=8,
                  cursor="hand2", command=self._send_emergency_alert).pack(side="left", padx=6)

        # ── Phrase Display Box ───────────────────────────────────────────────
        disp_frame = tk.Frame(self, bg=BG_COLOR, padx=24, pady=14)
        disp_frame.pack(fill="x")

        self.display_var = tk.StringVar()
        self.display_entry = tk.Entry(disp_frame, textvariable=self.display_var,
                                      font=("Segoe UI", 22, "bold"), bg=SURFACE_COLOR,
                                      fg=TEXT_MAIN, insertbackground=TEXT_MAIN,
                                      bd=2, relief="groove")
        self.display_entry.pack(side="left", fill="x", expand=True, ipady=10, padx=(0, 10))

        tk.Button(disp_frame, text="Clear ✕", font=("Segoe UI", 14, "bold"),
                  bg=SURFACE_COLOR, fg=TEXT_MUTED, activebackground=BORDER_COLOR,
                  activeforeground=TEXT_MAIN, relief="solid", bd=1, padx=18, pady=10,
                  cursor="hand2", command=lambda: self.display_var.set("")).pack(side="right")

        # Status / Notification Banner
        self.status_var = tk.StringVar(value="Select or write a phrase to communicate.")
        self.status_lbl = tk.Label(self, textvariable=self.status_var, font=("Segoe UI", 12, "italic"),
                                   bg=BG_COLOR, fg=BLUE_BTN, pady=4)
        self.status_lbl.pack()

        # ── Phrases Grid Container ───────────────────────────────────────────
        self.grid_container = tk.Frame(self, bg=BG_COLOR, padx=24, pady=8)
        self.grid_container.pack(fill="both", expand=True)

        # ── Bottom Control Toolbar ───────────────────────────────────────────
        toolbar = tk.Frame(self, bg=SURFACE_COLOR, padx=24, pady=14, highlightbackground=BORDER_COLOR, highlightthickness=1)
        toolbar.pack(fill="x", side="bottom")

        # Back Button
        tk.Button(toolbar, text="◀ Back", font=("Segoe UI", 15, "bold"),
                  bg=RED_BTN, fg="#FFFFFF", activebackground=RED_BTN_HOV,
                  activeforeground="#FFFFFF", relief="flat", padx=24, pady=10,
                  cursor="hand2", command=self.destroy).pack(side="left", padx=6)

        # Add Phrase Button
        tk.Button(toolbar, text="+ Add phrase", font=("Segoe UI", 15, "bold"),
                  bg=GREEN_TILE, fg="#FFFFFF", activebackground=GREEN_TILE_HOV,
                  activeforeground="#FFFFFF", relief="flat", padx=24, pady=10,
                  cursor="hand2", command=self._add_phrase_dialog).pack(side="left", padx=6)

        # Remove Phrase Button
        self.remove_btn = tk.Button(toolbar, text="- Remove phrase", font=("Segoe UI", 15, "bold"),
                                    bg=RED_BTN, fg="#FFFFFF", activebackground=RED_BTN_HOV,
                                    activeforeground="#FFFFFF", relief="flat", padx=24, pady=10,
                                    cursor="hand2", command=self._toggle_remove_mode)
        self.remove_btn.pack(side="left", padx=6)

        # Speak TTS Button
        tk.Button(toolbar, text="🔊 Speak", font=("Segoe UI", 15, "bold"),
                  bg=BLUE_BTN, fg="#FFFFFF", activebackground="#1D4ED8",
                  activeforeground="#FFFFFF", relief="flat", padx=24, pady=10,
                  cursor="hand2", command=self._speak_current_phrase).pack(side="right", padx=6)

        # Send Message Button
        tk.Button(toolbar, text="📱 Send Alert", font=("Segoe UI", 15, "bold"),
                  bg=ORANGE_TILE, fg="#FFFFFF", activebackground=ORANGE_TILE_HOV,
                  activeforeground="#FFFFFF", relief="flat", padx=24, pady=10,
                  cursor="hand2", command=self._send_emergency_alert).pack(side="right", padx=6)

    def _refresh_phrases_grid(self):
        """Re-render the phrase buttons in a clean 4-column grid."""
        for child in self.grid_container.winfo_children():
            child.destroy()

        phrases = self.config_data.get("phrases", [])
        cols = 4

        for idx, item in enumerate(phrases):
            text = item.get("text", "")
            cat = item.get("category", "green")

            r = idx // cols
            c = idx % cols

            if cat == "orange":
                bg = ORANGE_TILE
                hov = ORANGE_TILE_HOV
            else:
                bg = GREEN_TILE
                hov = GREEN_TILE_HOV

            if self.remove_mode:
                bg = "#DC2626"
                hov = "#B91C1C"

            btn = tk.Button(
                self.grid_container, text=text, font=("Segoe UI", 15, "bold"),
                bg=bg, fg="#FFFFFF", activebackground=hov, activeforeground="#FFFFFF",
                relief="flat", wraplength=260, justify="center", cursor="hand2",
                command=lambda t=text, item=item: self._on_phrase_click(t, item)
            )
            btn.grid(row=r, column=c, padx=8, pady=8, sticky="nsew")

        for i in range(cols):
            self.grid_container.grid_columnconfigure(i, weight=1)
        total_rows = (len(phrases) + cols - 1) // cols
        for i in range(max(total_rows, 1)):
            self.grid_container.grid_rowconfigure(i, weight=1)

    def _on_phrase_click(self, text, item):
        if self.remove_mode:
            phrases = self.config_data.get("phrases", [])
            if item in phrases:
                phrases.remove(item)
                save_config(self.config_data)
                self.status_var.set(f"Removed phrase: '{text}'")
                self.remove_mode = False
                self.remove_btn.configure(bg=RED_BTN, text="- Remove phrase")
                self._refresh_phrases_grid()
            return

        self.display_var.set(text)
        self.status_var.set(f"Selected: '{text}'")
        speak_text(text)

    def _toggle_remove_mode(self):
        self.remove_mode = not self.remove_mode
        if self.remove_mode:
            self.remove_btn.configure(bg="#D97706", text="Click phrase to remove (Cancel)")
            self.status_var.set("⚠ Removal Mode Active: Click any phrase button to delete it.")
        else:
            self.remove_btn.configure(bg=RED_BTN, text="- Remove phrase")
            self.status_var.set("Select or write a phrase to communicate.")
        self._refresh_phrases_grid()

    def _add_phrase_dialog(self):
        dlg = CustomPhraseDialog(self)
        self.wait_window(dlg)
        if dlg.result:
            phrases = self.config_data.get("phrases", [])
            phrases.append(dlg.result)
            save_config(self.config_data)
            self._refresh_phrases_grid()
            self.display_var.set(dlg.result["text"])
            self.status_var.set(f"Added new phrase: '{dlg.result['text']}'")
            speak_text(dlg.result["text"])

    def _open_alert_settings(self):
        dlg = AlertSettingsDialog(self, self.config_data)
        self.wait_window(dlg)
        chan = self.config_data.get("dispatch_channel", "simulation").upper()
        self.contact_var.set(f"Emergency Contact: {self.config_data.get('emergency_contact', '')}  [{chan} MODE]")

    def _edit_contact(self):
        curr = self.config_data.get("emergency_contact", "")
        new_num = simpledialog.askstring("Emergency Contact", "Enter Emergency Contact Phone Number:",
                                         initialvalue=curr, parent=self)
        if new_num is not None and new_num.strip():
            self.config_data["emergency_contact"] = new_num.strip()
            save_config(self.config_data)
            chan = self.config_data.get("dispatch_channel", "simulation").upper()
            self.contact_var.set(f"Emergency Contact: {new_num.strip()}  [{chan} MODE]")
            self.status_var.set(f"Updated Emergency Contact: {new_num.strip()}")

    def _speak_current_phrase(self):
        txt = self.display_var.get().strip()
        if not txt:
            txt = "Emergency! I need assistance!"
            self.display_var.set(txt)
        speak_text(txt)
        self.status_var.set(f"🔊 Spoken: '{txt}'")

    def _send_emergency_alert(self):
        txt = self.display_var.get().strip()
        if not txt:
            txt = "EMERGENCY ALERT: I need immediate assistance!"
            self.display_var.set(txt)

        contact = self.config_data.get("emergency_contact", "Unspecified Contact")
        
        # Dispatch signal across selected channel
        dispatch_status = dispatch_phone_signal(contact, txt, self.config_data)

        speak_text(f"Emergency Alert sent. {txt}")

        self.status_var.set(f"✅ {dispatch_status}")
        messagebox.showinfo("Emergency Alert Dispatched",
                            f"Emergency Message Dispatched!\n\nTo: {contact}\nMessage: '{txt}'\nStatus: {dispatch_status}\n\nLogged to emergency_sent_log.txt",
                            parent=self)


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    app = EmergencyWindow(root)
    app.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
