#!/usr/bin/env python3
"""
Eye-Gaze / Text-Entry Experimental Analyzer
============================================

Select one or more CSV files and generate a visually rich PDF report.

Designed for datasets containing fields such as:
    method
    trial_number
    stimulus / target_phrase
    typed_response / entered_phrase
    is_correct
    trial_start_time
    trial_end_time
    typing_duration_sec
    spacebar_count
    backspace_count
    char_count
    word_count
    letter_timestamps

The program is intentionally tolerant of column-name variations.

Install:
    pip install pandas numpy matplotlib scipy reportlab

Run:
    python analysis.py

Or:
    python analysis.py file1.csv file2.csv
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats
    SCIPY = True
except Exception:
    SCIPY = False

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether
)

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    TK = True
except Exception:
    TK = False


# ---------------------------------------------------------------------------
# Visual theme
# ---------------------------------------------------------------------------

NAVY = HexColor("#17233C")
BLUE = HexColor("#2563EB")
CYAN = HexColor("#06B6D4")
GREEN = HexColor("#10B981")
ORANGE = HexColor("#F59E0B")
RED = HexColor("#EF4444")
PURPLE = HexColor("#7C3AED")
SLATE = HexColor("#475569")
LIGHT = HexColor("#F1F5F9")
LIGHT_BLUE = HexColor("#E8F0FE")
WHITE = colors.white


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

ALIASES = {
    "participant": [
        "participant", "participant_id", "participantid", "subject",
        "subject_id", "user", "user_id", "tester", "person"
    ],
    "method": [
        "method", "technique", "input_method", "keyboard_type",
        "condition", "system", "interface", "device"
    ],
    "trial": [
        "trial", "trial_number", "trial_no", "trialid", "trial_id",
        "attempt", "session_trial"
    ],
    "target": [
        "stimulus", "target", "target_phrase", "target_text",
        "expected", "expected_phrase", "expected_text", "phrase"
    ],
    "entered": [
        "typed_response", "typed_text", "entered_phrase",
        "entered_text", "entered", "input", "typed_phrase",
        "actual_phrase", "response", "transcription"
    ],
    "correct": [
        "is_correct", "correct", "success", "successful",
        "trial_correct", "accuracy_flag"
    ],
    "start_time": [
        "trial_start_time", "start_time", "start_timestamp",
        "session_start", "begin_time"
    ],
    "end_time": [
        "trial_end_time", "end_time", "end_timestamp",
        "session_end", "finish_time"
    ],
    "duration": [
        "typing_duration_sec", "typing_duration_seconds",
        "duration_seconds", "duration_sec", "duration_s",
        "completion_time", "completion_time_seconds",
        "elapsed_seconds", "duration"
    ],
    "char_count": [
        "char_count", "characters_typed", "total_characters",
        "total_characters_typed", "typed_characters",
        "character_count", "characters"
    ],
    "word_count": [
        "word_count", "words", "total_words", "words_typed"
    ],
    "spaces": [
        "spacebar_count", "spaces", "space_count", "space_counted"
    ],
    "backspaces": [
        "backspace_count", "backspaces", "backspace",
        "correction_count", "corrections"
    ],
    "timestamps": [
        "letter_timestamps", "keystroke_timestamps",
        "key_timestamps", "timestamps", "keypress_timestamps"
    ],
    "wpm": ["wpm", "words_per_minute", "typing_speed_wpm"],
    "cps": ["cps", "characters_per_second", "typing_speed_cps"],
    "accuracy": ["accuracy", "accuracy_percent", "accuracy_percentage"],
    "dwell": ["dwell_time", "dwell_seconds", "average_dwell_time"],
    "yaw": ["yaw", "yaw_deg", "head_yaw"],
    "pitch": ["pitch", "pitch_deg", "head_pitch"],
    "raw_x": ["raw_gaze_x", "gaze_x_raw", "raw_x", "gaze_x"],
    "raw_y": ["raw_gaze_y", "gaze_y_raw", "raw_y", "gaze_y"],
    "smooth_x": ["smooth_gaze_x", "smoothed_gaze_x", "gaze_x_smooth", "smooth_x"],
    "smooth_y": ["smooth_gaze_y", "smoothed_gaze_y", "gaze_y_smooth", "smooth_y"],
}


def norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def detect_columns(df):
    normalized = {norm(c): c for c in df.columns}
    found = {}
    for logical, aliases in ALIASES.items():
        for alias in aliases:
            if norm(alias) in normalized:
                found[logical] = normalized[norm(alias)]
                break
    return found


def num(df, col):
    if not col or col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def clean(v, digits=2):
    if v is None:
        return "N/A"
    try:
        if pd.isna(v):
            return "N/A"
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


def pct(a, b):
    if b == 0 or pd.isna(b):
        return np.nan
    return 100 * a / b


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

def parse_keystroke_timestamps(value):
    """
    Supports examples such as:
      s: 0.000, m: 4.947, i: 7.895
    or:
      [{"key":"s","time":0.0}, ...]
    or:
      {"s":0.0,"m":4.947}
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    if isinstance(value, list):
        items = value
    else:
        text = str(value).strip()
        if not text:
            return []

        # JSON
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                items = obj
            elif isinstance(obj, dict):
                return [(str(k), float(v)) for k, v in obj.items()
                        if _floatable(v)]
            else:
                items = None
        except Exception:
            items = None

        if items is None:
            # Generic key/time extraction.
            # Matches:
            #   s: 0.000
            #   s=0.000
            #   s -> 0.000
            #   <BS>: 4.2
            pattern = r"([A-Za-z0-9_<>\[\] \-.,']+?)\s*(?::|=|->)\s*(-?\d+(?:\.\d+)?)"
            matches = re.findall(pattern, text)
            if not matches:
                # Fallback: extract numbers as monotonically ordered timestamps.
                nums = re.findall(r"-?\d+(?:\.\d+)?", text)
                return [(f"k{i+1}", float(t)) for i, t in enumerate(nums)]
            return [(k.strip(), float(t)) for k, t in matches]

    result = []
    for item in items:
        if isinstance(item, dict):
            key = item.get("key", item.get("letter", item.get("event", "")))
            t = item.get("time", item.get("timestamp", item.get("t")))
            if _floatable(t):
                result.append((str(key), float(t)))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            if _floatable(item[1]):
                result.append((str(item[0]), float(item[1])))
    return result


def _floatable(x):
    try:
        float(x)
        return True
    except Exception:
        return False


def add_derived_metrics(df, cols):
    df = df.copy()

    # Duration
    if "duration" not in cols and "start_time" in cols and "end_time" in cols:
        start = pd.to_datetime(df[cols["start_time"]], errors="coerce")
        end = pd.to_datetime(df[cols["end_time"]], errors="coerce")
        df["__duration"] = (end - start).dt.total_seconds()
        cols["duration"] = "__duration"

    # Character count
    if "char_count" not in cols and "entered" in cols:
        df["__char_count"] = df[cols["entered"]].fillna("").astype(str).str.len()
        cols["char_count"] = "__char_count"

    # Word count
    if "word_count" not in cols and "entered" in cols:
        df["__word_count"] = (
            df[cols["entered"]].fillna("").astype(str).str.split().str.len()
        )
        cols["word_count"] = "__word_count"

    # Backspaces / spaces may be encoded in timestamps.
    if "timestamps" in cols:
        events = df[cols["timestamps"]].apply(parse_keystroke_timestamps)
        if "backspaces" not in cols:
            df["__backspaces"] = events.apply(
                lambda x: sum(1 for k, _ in x if "<BS>" in k.upper() or "BACKSPACE" in k.upper())
            )
            cols["backspaces"] = "__backspaces"
        if "spaces" not in cols:
            df["__spaces"] = events.apply(
                lambda x: sum(1 for k, _ in x if k.strip() in {"<SP>", "SPACE", " "})
            )
            cols["spaces"] = "__spaces"

    # WPM
    if "wpm" not in cols and "duration" in cols and "char_count" in cols:
        d = num(df, cols["duration"])
        c = num(df, cols["char_count"])
        df["__wpm"] = (c / 5) / (d / 60).replace(0, np.nan)
        cols["wpm"] = "__wpm"

    # CPS
    if "cps" not in cols and "duration" in cols and "char_count" in cols:
        d = num(df, cols["duration"])
        c = num(df, cols["char_count"])
        df["__cps"] = c / d.replace(0, np.nan)
        cols["cps"] = "__cps"

    # Accuracy: first use target/entered comparison instead of trusting a
    # potentially inconsistent is_correct field.
    if "target" in cols and "entered" in cols:
        target = df[cols["target"]].fillna("").astype(str)
        entered = df[cols["entered"]].fillna("").astype(str)

        def edit_distance(a, b):
            # Standard Levenshtein distance.
            prev = list(range(len(b) + 1))
            for i, ca in enumerate(a, 1):
                cur = [i]
                for j, cb in enumerate(b, 1):
                    cur.append(min(
                        cur[-1] + 1,
                        prev[j] + 1,
                        prev[j - 1] + (ca != cb)
                    ))
                prev = cur
            return prev[-1]

        distances = []
        normalized = []
        exact = []

        for a, b in zip(target, entered):
            dist = edit_distance(a, b)
            distances.append(dist)
            normalized.append(dist / max(1, len(a)) * 100)
            exact.append(a == b)

        df["__edit_distance"] = distances
        df["__normalized_error_pct"] = normalized
        df["__final_response_correct"] = exact
        df["__text_accuracy"] = [
            max(0, 100 - x) for x in normalized
        ]

        cols["edit_distance"] = "__edit_distance"
        cols["normalized_error"] = "__normalized_error_pct"
        cols["final_correct"] = "__final_response_correct"
        cols["text_accuracy"] = "__text_accuracy"

    # Backspace correction rate
    if "backspaces" in cols and "char_count" in cols:
        b = num(df, cols["backspaces"])
        c = num(df, cols["char_count"])
        df["__correction_rate"] = b / c.replace(0, np.nan) * 100
        cols["correction_rate"] = "__correction_rate"

    # Event-level timing metrics
    if "timestamps" in cols:
        event_data = []
        for value in df[cols["timestamps"]]:
            ts = parse_keystroke_timestamps(value)
            times = [t for _, t in ts]
            keys = [k for k, _ in ts]
            intervals = np.diff(times) if len(times) > 1 else np.array([])
            positive = intervals[intervals >= 0]

            event_data.append({
                "event_count": len(ts),
                "iki_mean": float(np.mean(positive)) if len(positive) else np.nan,
                "iki_median": float(np.median(positive)) if len(positive) else np.nan,
                "iki_sd": float(np.std(positive, ddof=1)) if len(positive) > 1 else 0,
                "iki_min": float(np.min(positive)) if len(positive) else np.nan,
                "iki_max": float(np.max(positive)) if len(positive) else np.nan,
                "pause_count_3s": int(np.sum(positive >= 3)),
                "pause_count_5s": int(np.sum(positive >= 5)),
                "longest_pause": float(np.max(positive)) if len(positive) else np.nan,
                "first_event": times[0] if times else np.nan,
                "last_event": times[-1] if times else np.nan,
                "backspace_events": sum(
                    1 for k in keys if "<BS>" in k.upper() or "BACKSPACE" in k.upper()
                ),
                "space_events": sum(
                    1 for k in keys if k.strip() in {"<SP>", "SPACE", " "}
                ),
            })

        event_df = pd.DataFrame(event_data)
        for c in event_df.columns:
            new = "__" + c
            df[new] = event_df[c]
            cols[c] = new

    return df, cols


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def method_stats(df, cols, metric):
    mcol = cols.get("method")
    vcol = cols.get(metric)
    if not mcol or not vcol:
        return None
    tmp = pd.DataFrame({
        "method": df[mcol].astype(str),
        "value": num(df, vcol)
    }).dropna()

    if tmp.empty:
        return None

    return tmp.groupby("method")["value"].agg(
        ["count", "mean", "std", "median", "min", "max"]
    ).reset_index()


def test_two_methods(df, cols, metric):
    if not SCIPY:
        return None

    mcol = cols.get("method")
    vcol = cols.get(metric)
    if not mcol or not vcol:
        return None

    d = pd.DataFrame({
        "method": df[mcol].astype(str),
        "value": num(df, vcol)
    }).dropna()

    methods = list(d["method"].unique())
    if len(methods) != 2:
        return None

    a = d[d.method == methods[0]].value
    b = d[d.method == methods[1]].value

    if len(a) < 2 or len(b) < 2:
        return {
            "methods": methods,
            "test": "Insufficient observations",
            "stat": np.nan,
            "p": np.nan
        }

    stat, p = stats.ttest_ind(a, b, equal_var=False)
    return {
        "methods": methods,
        "test": "Welch t-test",
        "stat": stat,
        "p": p
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_plots(df, cols, out):
    paths = []
    mcol = cols.get("method")

    def methods_exist():
        return mcol and df[mcol].dropna().astype(str).nunique() >= 2

    # WPM
    if cols.get("wpm"):
        fig, ax = plt.subplots(figsize=(9, 5))
        s = num(df, cols["wpm"])
        if methods_exist():
            tmp = pd.DataFrame({"method": df[mcol], "wpm": s}).dropna()
            means = tmp.groupby("method").wpm.mean()
            ax.bar(means.index, means.values)
            ax.set_xlabel("Method")
        else:
            ax.plot(range(1, len(s.dropna()) + 1), s.dropna(), marker="o")
            ax.set_xlabel("Trial")
        ax.set_ylabel("WPM")
        ax.set_title("Typing Speed")
        ax.grid(axis="y", alpha=.25)
        p = out / "01_typing_speed.png"
        plot_save(fig, p)
        paths.append(("Typing Speed", p))

    # Accuracy
    if cols.get("text_accuracy"):
        fig, ax = plt.subplots(figsize=(9, 5))
        s = num(df, cols["text_accuracy"])
        if methods_exist():
            tmp = pd.DataFrame({"method": df[mcol], "accuracy": s}).dropna()
            groups = [g.accuracy.values for _, g in tmp.groupby("method")]
            labels = list(tmp.groupby("method").groups.keys())
            ax.boxplot(groups, labels=labels)
            ax.set_xlabel("Method")
        else:
            ax.bar(range(1, len(s.dropna()) + 1), s.dropna())
            ax.set_xlabel("Trial")
        ax.set_ylabel("Text accuracy (%)")
        ax.set_ylim(0, 105)
        ax.set_title("Final Text Accuracy")
        ax.grid(axis="y", alpha=.25)
        p = out / "02_accuracy.png"
        plot_save(fig, p)
        paths.append(("Final Text Accuracy", p))

    # Completion time
    if cols.get("duration"):
        fig, ax = plt.subplots(figsize=(9, 5))
        s = num(df, cols["duration"])
        if methods_exist():
            tmp = pd.DataFrame({"method": df[mcol], "time": s}).dropna()
            means = tmp.groupby("method").time.mean()
            ax.bar(means.index, means.values)
            ax.set_xlabel("Method")
        else:
            ax.plot(range(1, len(s.dropna()) + 1), s.dropna(), marker="o")
            ax.set_xlabel("Trial")
        ax.set_ylabel("Seconds")
        ax.set_title("Completion Time")
        ax.grid(axis="y", alpha=.25)
        p = out / "03_completion_time.png"
        plot_save(fig, p)
        paths.append(("Completion Time", p))

    # Corrections
    if cols.get("backspaces"):
        fig, ax = plt.subplots(figsize=(9, 5))
        s = num(df, cols["backspaces"])
        if methods_exist():
            tmp = pd.DataFrame({"method": df[mcol], "backspaces": s}).dropna()
            means = tmp.groupby("method").backspaces.mean()
            ax.bar(means.index, means.values)
            ax.set_xlabel("Method")
        else:
            ax.bar(range(1, len(s.dropna()) + 1), s.dropna())
            ax.set_xlabel("Trial")
        ax.set_ylabel("Backspaces")
        ax.set_title("Correction Behavior")
        ax.grid(axis="y", alpha=.25)
        p = out / "04_corrections.png"
        plot_save(fig, p)
        paths.append(("Correction Behavior", p))

    # WPM vs accuracy
    if cols.get("wpm") and cols.get("text_accuracy"):
        fig, ax = plt.subplots(figsize=(9, 5))
        x = num(df, cols["wpm"])
        y = num(df, cols["text_accuracy"])
        valid = pd.DataFrame({"wpm": x, "accuracy": y}).dropna()
        ax.scatter(valid.wpm, valid.accuracy, s=70)
        ax.set_xlabel("WPM")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Speed–Accuracy Relationship")
        ax.grid(alpha=.25)
        p = out / "05_speed_accuracy.png"
        plot_save(fig, p)
        paths.append(("Speed–Accuracy Relationship", p))

    # IKI
    if cols.get("iki_mean"):
        fig, ax = plt.subplots(figsize=(9, 5))
        s = num(df, cols["iki_mean"])
        ax.plot(range(1, len(s.dropna()) + 1), s.dropna(), marker="o")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Mean inter-keystroke interval (s)")
        ax.set_title("Keystroke Rhythm / Mean IKI")
        ax.grid(alpha=.25)
        p = out / "06_iki.png"
        plot_save(fig, p)
        paths.append(("Keystroke Rhythm", p))

    # Pause profile
    if cols.get("longest_pause"):
        fig, ax = plt.subplots(figsize=(9, 5))
        s = num(df, cols["longest_pause"])
        ax.bar(range(1, len(s.dropna()) + 1), s.dropna())
        ax.set_xlabel("Trial")
        ax.set_ylabel("Longest pause (s)")
        ax.set_title("Longest Inter-Keystroke Pause")
        ax.grid(axis="y", alpha=.25)
        p = out / "07_pauses.png"
        plot_save(fig, p)
        paths.append(("Pause Analysis", p))

    # Character/word volume
    if cols.get("char_count") and cols.get("duration"):
        fig, ax = plt.subplots(figsize=(9, 5))
        c = num(df, cols["char_count"])
        d = num(df, cols["duration"])
        valid = pd.DataFrame({"characters": c, "duration": d}).dropna()
        ax.scatter(valid.characters, valid.duration, s=70)
        ax.set_xlabel("Characters")
        ax.set_ylabel("Completion time (s)")
        ax.set_title("Task Length vs Completion Time")
        ax.grid(alpha=.25)
        p = out / "08_task_length_time.png"
        plot_save(fig, p)
        paths.append(("Task Length vs Completion Time", p))

    # Gaze trajectory if available
    if cols.get("raw_x") and cols.get("raw_y"):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        x = num(df, cols["raw_x"])
        y = num(df, cols["raw_y"])
        ax.plot(x, y, linewidth=1, label="Raw")
        if cols.get("smooth_x") and cols.get("smooth_y"):
            sx = num(df, cols["smooth_x"])
            sy = num(df, cols["smooth_y"])
            ax.plot(sx, sy, linewidth=1.5, label="Smoothed")
        ax.invert_yaxis()
        ax.set_xlabel("Gaze X")
        ax.set_ylabel("Gaze Y")
        ax.set_title("Gaze Trajectory")
        ax.legend()
        p = out / "09_gaze_trajectory.png"
        plot_save(fig, p)
        paths.append(("Gaze Trajectory", p))

    return paths


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def styles():
    s = getSampleStyleSheet()

    s.add(ParagraphStyle(
        name="Cover", parent=s["Title"], fontSize=26, leading=31,
        textColor=WHITE, alignment=TA_LEFT, spaceAfter=12
    ))
    s.add(ParagraphStyle(
        name="CoverSub", parent=s["BodyText"], fontSize=11, leading=16,
        textColor=HexColor("#D9E5FF")
    ))
    s.add(ParagraphStyle(
        name="H1x", parent=s["Heading1"], fontSize=17, leading=21,
        textColor=NAVY, spaceBefore=8, spaceAfter=8
    ))
    s.add(ParagraphStyle(
        name="H2x", parent=s["Heading2"], fontSize=11, leading=14,
        textColor=BLUE, spaceBefore=8, spaceAfter=5
    ))
    s.add(ParagraphStyle(
        name="Bodyx", parent=s["BodyText"], fontSize=9.2, leading=13,
        textColor=SLATE, spaceAfter=6
    ))
    s.add(ParagraphStyle(
        name="Smallx", parent=s["BodyText"], fontSize=7.2, leading=9,
        textColor=SLATE
    ))
    s.add(ParagraphStyle(
        name="Metric", parent=s["BodyText"], fontSize=20, leading=23,
        textColor=NAVY, alignment=TA_CENTER
    ))
    s.add(ParagraphStyle(
        name="MetricLabel", parent=s["BodyText"], fontSize=7.5, leading=9,
        textColor=SLATE, alignment=TA_CENTER
    ))
    return s


def table(data, widths=None, header=True):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("GRID", (0,0), (-1,-1), .35, HexColor("#CBD5E1")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("FONTSIZE", (0,0), (-1,-1), 7.5),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0,0), (-1,0), NAVY),
            ("TEXTCOLOR", (0,0), (-1,0), WHITE),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ]
    t.setStyle(TableStyle(commands))
    return t


def metric_cards(items):
    cells = []
    for label, value in items:
        cells.append([
            Paragraph(str(value), PDF_STYLES["Metric"]),
            Paragraph(label, PDF_STYLES["MetricLabel"])
        ])
    t = Table([cells], colWidths=[160*mm/len(cells)]*len(cells))
    t.setStyle(TableStyle([
        ("BOX", (0,0), (-1,-1), .5, HexColor("#CBD5E1")),
        ("INNERGRID", (0,0), (-1,-1), .5, HexColor("#CBD5E1")),
        ("BACKGROUND", (0,0), (-1,-1), LIGHT),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
    ]))
    return t


def page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(SLATE)
    canvas.drawString(15*mm, 9*mm, "Automated Experimental Analysis")
    canvas.drawRightString(A4[0]-15*mm, 9*mm, f"Page {doc.page}")
    canvas.restoreState()


# Global for metric_cards.
PDF_STYLES = None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_pdf(df, cols, source_files, out):
    global PDF_STYLES
    PDF_STYLES = styles()

    plot_dir = out / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plots = make_plots(df, cols, plot_dir)

    pdf = out / "eye_gaze_experimental_analysis_report.pdf"

    doc = SimpleDocTemplate(
        str(pdf), pagesize=A4,
        rightMargin=15*mm, leftMargin=15*mm,
        topMargin=15*mm, bottomMargin=15*mm,
        title="Eye-Gaze Text Entry Experimental Analysis"
    )

    story = []

    # Cover
    cover = Table([[
        Paragraph("EYE-GAZE / TEXT-ENTRY<br/>EXPERIMENTAL ANALYSIS",
                  PDF_STYLES["Cover"]),
    ]], colWidths=[170*mm], rowHeights=[65*mm])
    cover.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), NAVY),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 14),
        ("RIGHTPADDING", (0,0), (-1,-1), 14),
    ]))
    story.append(cover)
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "A data-driven report of typing speed, accuracy, corrections, "
        "keystroke timing, pauses, task difficulty and method-level performance.",
        PDF_STYLES["Bodyx"]
    ))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%d %B %Y, %H:%M:%S')}",
        PDF_STYLES["Smallx"]
    ))

    # Overview cards
    story.append(Spacer(1, 12))
    story.append(metric_cards([
        ("Rows / Trials", len(df)),
        ("Methods", df[cols["method"]].nunique() if cols.get("method") else "N/A"),
        ("Participants", df[cols["participant"]].nunique() if cols.get("participant") else "N/A"),
        ("Variables", len(df.columns)),
    ]))

    story.append(PageBreak())

    # Executive summary
    story.append(Paragraph("1. Executive Summary", PDF_STYLES["H1x"]))

    summary_items = []
    if cols.get("wpm"):
        summary_items.append(("Mean WPM", clean(num(df, cols["wpm"]).mean())))
    if cols.get("text_accuracy"):
        summary_items.append(("Text Accuracy", clean(num(df, cols["text_accuracy"]).mean()) + "%"))
    if cols.get("duration"):
        summary_items.append(("Mean Time", clean(num(df, cols["duration"]).mean()) + " s"))
    if cols.get("backspaces"):
        summary_items.append(("Mean Backspaces", clean(num(df, cols["backspaces"]).mean())))

    if summary_items:
        story.append(metric_cards(summary_items[:4]))

    story.append(Spacer(1, 10))

    findings = []

    if cols.get("wpm"):
        s = num(df, cols["wpm"]).dropna()
        if len(s):
            findings.append(
                f"Average typing speed was <b>{s.mean():.2f} WPM</b>, "
                f"with a range of {s.min():.2f}–{s.max():.2f} WPM."
            )

    if cols.get("text_accuracy"):
        s = num(df, cols["text_accuracy"]).dropna()
        if len(s):
            findings.append(
                f"Final-response text accuracy averaged <b>{s.mean():.2f}%</b>."
            )

    if cols.get("edit_distance"):
        s = num(df, cols["edit_distance"]).dropna()
        if len(s):
            findings.append(
                f"Mean Levenshtein edit distance was <b>{s.mean():.2f}</b> characters."
            )

    if cols.get("backspaces"):
        s = num(df, cols["backspaces"]).dropna()
        if len(s):
            findings.append(
                f"The user produced an average of <b>{s.mean():.2f} backspaces per trial</b>."
            )

    if cols.get("iki_mean"):
        s = num(df, cols["iki_mean"]).dropna()
        if len(s):
            findings.append(
                f"Mean inter-keystroke interval was <b>{s.mean():.2f} seconds</b>, "
                "providing a direct measure of typing rhythm."
            )

    for f in findings:
        story.append(Paragraph("• " + f, PDF_STYLES["Bodyx"]))

    # Dataset
    story.append(Paragraph("2. Dataset & Variable Inventory", PDF_STYLES["H1x"]))
    overview = [
        ["Property", "Value"],
        ["Files loaded", str(len(source_files))],
        ["Rows", str(len(df))],
        ["Columns", str(len(df.columns))],
        ["Methods", str(df[cols["method"]].nunique()) if cols.get("method") else "Not detected"],
        ["Participants", str(df[cols["participant"]].nunique()) if cols.get("participant") else "Not detected"],
        ["Trials", str(df[cols["trial"]].nunique()) if cols.get("trial") else "Not detected"],
    ]
    story.append(table(overview, [70*mm, 90*mm]))

    story.append(Paragraph("Detected columns", PDF_STYLES["H2x"]))
    detected = [["Logical Variable", "CSV Column"]]
    for k, v in cols.items():
        detected.append([k, v])
    story.append(table(detected, [65*mm, 95*mm]))

    # Core performance
    story.append(PageBreak())
    story.append(Paragraph("3. Core Typing Performance", PDF_STYLES["H1x"]))

    rows = [["Metric", "Mean", "SD", "Median", "Min", "Max", "N"]]
    metric_list = [
        ("WPM", "wpm"),
        ("CPS", "cps"),
        ("Completion time (s)", "duration"),
        ("Characters", "char_count"),
        ("Words", "word_count"),
        ("Backspaces", "backspaces"),
        ("Spaces", "spaces"),
        ("Text accuracy (%)", "text_accuracy"),
        ("Edit distance", "edit_distance"),
        ("Normalized error (%)", "normalized_error"),
        ("Correction rate (%)", "correction_rate"),
        ("Mean IKI (s)", "iki_mean"),
        ("Median IKI (s)", "iki_median"),
        ("Longest pause (s)", "longest_pause"),
    ]

    for label, key in metric_list:
        if not cols.get(key):
            continue
        s = num(df, cols[key]).dropna()
        if len(s):
            rows.append([
                label,
                clean(s.mean()),
                clean(s.std(ddof=1) if len(s) > 1 else 0),
                clean(s.median()),
                clean(s.min()),
                clean(s.max()),
                str(len(s))
            ])

    story.append(table(rows, [42*mm, 20*mm, 20*mm, 20*mm, 20*mm, 20*mm, 18*mm]))

    story.append(Paragraph("Metric definitions", PDF_STYLES["H2x"]))
    story.append(Paragraph(
        "<b>WPM</b> = (characters typed / 5) / duration in minutes.<br/>"
        "<b>CPS</b> = characters typed / duration in seconds.<br/>"
        "<b>Edit distance</b> = minimum insertions, deletions and substitutions "
        "needed to transform the entered response into the target.<br/>"
        "<b>Correction rate</b> = backspaces / final character count × 100.<br/>"
        "<b>IKI</b> = time between consecutive recorded keystroke events.",
        PDF_STYLES["Bodyx"]
    ))

    # Trial-level detail
    story.append(PageBreak())
    story.append(Paragraph("4. Trial-Level Analysis", PDF_STYLES["H1x"]))

    headers = ["Trial", "Method", "Target", "Entered", "Time", "WPM", "Accuracy", "BS", "Edit"]
    rows = [headers]

    for _, r in df.iterrows():
        trial = r[cols["trial"]] if cols.get("trial") else ""
        method = r[cols["method"]] if cols.get("method") else ""
        target = str(r[cols["target"]]) if cols.get("target") and not pd.isna(r[cols["target"]]) else ""
        entered = str(r[cols["entered"]]) if cols.get("entered") and not pd.isna(r[cols["entered"]]) else ""

        rows.append([
            str(trial),
            str(method),
            target[:28],
            entered[:28],
            clean(r[cols["duration"]]) if cols.get("duration") else "N/A",
            clean(r[cols["wpm"]]) if cols.get("wpm") else "N/A",
            clean(r[cols["text_accuracy"]]) + "%" if cols.get("text_accuracy") else "N/A",
            clean(r[cols["backspaces"]], 0) if cols.get("backspaces") else "N/A",
            clean(r[cols["edit_distance"]], 0) if cols.get("edit_distance") else "N/A",
        ])

    story.append(table(
        rows,
        [12*mm, 25*mm, 32*mm, 32*mm, 16*mm, 16*mm, 20*mm, 12*mm, 15*mm]
    ))

    # Keystroke analysis
    story.append(PageBreak())
    story.append(Paragraph("5. Keystroke-Level Analysis", PDF_STYLES["H1x"]))

    if cols.get("timestamps"):
        rows = [[
            "Trial", "Events", "Mean IKI", "Median IKI",
            "IKI SD", "3s Pauses", "5s Pauses", "Longest Pause"
        ]]

        for _, r in df.iterrows():
            rows.append([
                str(r[cols["trial"]]) if cols.get("trial") else "",
                clean(r[cols["event_count"]], 0) if cols.get("event_count") else "N/A",
                clean(r[cols["iki_mean"]]) if cols.get("iki_mean") else "N/A",
                clean(r[cols["iki_median"]]) if cols.get("iki_median") else "N/A",
                clean(r[cols["iki_sd"]]) if cols.get("iki_sd") else "N/A",
                clean(r[cols["pause_count_3s"]], 0) if cols.get("pause_count_3s") else "N/A",
                clean(r[cols["pause_count_5s"]], 0) if cols.get("pause_count_5s") else "N/A",
                clean(r[cols["longest_pause"]]) if cols.get("longest_pause") else "N/A",
            ])

        story.append(table(rows, [
            15*mm, 20*mm, 23*mm, 23*mm,
            20*mm, 20*mm, 20*mm, 25*mm
        ]))

        story.append(Paragraph(
            "The timestamp stream permits analysis of typing rhythm rather than only "
            "final trial performance. Long inter-keystroke intervals can indicate "
            "hesitation, search time, correction, or task difficulty. They should not "
            "automatically be interpreted as cognitive hesitation without additional evidence.",
            PDF_STYLES["Bodyx"]
        ))
    else:
        story.append(Paragraph(
            "No keystroke timestamp column was detected. If the CSV contains a "
            "letter_timestamps field, it can be parsed for IKI and pause analysis.",
            PDF_STYLES["Bodyx"]
        ))

    # Corrections
    story.append(Paragraph("6. Error & Correction Behavior", PDF_STYLES["H1x"]))
    if cols.get("backspaces"):
        s = num(df, cols["backspaces"]).dropna()
        story.append(Paragraph(
            f"Total backspaces across the dataset: <b>{s.sum():.0f}</b>. "
            f"Mean per trial: <b>{s.mean():.2f}</b>. "
            f"Maximum in a trial: <b>{s.max():.0f}</b>.",
            PDF_STYLES["Bodyx"]
        ))

    if cols.get("edit_distance"):
        s = num(df, cols["edit_distance"]).dropna()
        story.append(Paragraph(
            f"Mean final-response edit distance: <b>{s.mean():.2f}</b>. "
            "This is preferable to relying only on an is_correct flag because it "
            "quantifies how far the final response is from the target.",
            PDF_STYLES["Bodyx"]
        ))

    if cols.get("correct"):
        raw = df[cols["correct"]].astype(str).str.lower().str.strip()
        raw_correct = raw.isin(["true", "1", "yes", "correct", "success"])
        story.append(Paragraph(
            f"The CSV's <b>{cols['correct']}</b> field reports "
            f"<b>{raw_correct.mean()*100:.1f}%</b> correct trials. "
            "The report separately calculates final-response correctness from target "
            "and entered text so inconsistent correctness flags can be identified.",
            PDF_STYLES["Bodyx"]
        ))

    # Method comparison
    story.append(PageBreak())
    story.append(Paragraph("7. Method Comparison", PDF_STYLES["H1x"]))

    mcol = cols.get("method")
    if mcol and df[mcol].dropna().astype(str).nunique() >= 2:
        for label, key in [
            ("WPM", "wpm"),
            ("Accuracy", "text_accuracy"),
            ("Completion time", "duration"),
            ("Backspaces", "backspaces"),
            ("Mean IKI", "iki_mean"),
        ]:
            result = method_stats(df, cols, key)
            if result is None:
                continue

            rows = [[label, "N", "Mean", "SD", "Median", "Min", "Max"]]
            for _, r in result.iterrows():
                rows.append([
                    r["method"], str(int(r["count"])),
                    clean(r["mean"]), clean(r["std"]),
                    clean(r["median"]), clean(r["min"]), clean(r["max"])
                ])

            story.append(Paragraph(label, PDF_STYLES["H2x"]))
            story.append(table(rows, [42*mm, 15*mm, 23*mm, 23*mm, 23*mm, 23*mm, 23*mm]))

            test = test_two_methods(df, cols, key)
            if test and not pd.isna(test["p"]):
                significance = "statistically significant" if test["p"] < .05 else "not statistically significant"
                story.append(Paragraph(
                    f"{test['test']}: statistic = {test['stat']:.3f}, "
                    f"p = {test['p']:.4f}. The difference is <b>{significance}</b> "
                    "at α = 0.05.",
                    PDF_STYLES["Smallx"]
                ))
    else:
        story.append(Paragraph(
            "Only one method was detected in the current data, so a method comparison "
            "cannot be performed. Multiple methods such as Traditional Keyboard and "
            "Eye-Gaze Keyboard are required.",
            PDF_STYLES["Bodyx"]
        ))

    # Task difficulty
    story.append(PageBreak())
    story.append(Paragraph("8. Task / Phrase Difficulty", PDF_STYLES["H1x"]))

    if cols.get("target"):
        target = df[cols["target"]].fillna("").astype(str)
        lengths = target.str.len()
        spaces = target.str.count(r"\s")
        punctuation = target.apply(lambda x: sum(1 for c in x if not c.isalnum() and not c.isspace()))

        rows = [["Trial", "Target Length", "Words", "Spaces", "Punctuation", "Time", "WPM"]]
        for i, (_, r) in enumerate(df.iterrows()):
            text = str(r[cols["target"]]) if not pd.isna(r[cols["target"]]) else ""
            rows.append([
                str(r[cols["trial"]]) if cols.get("trial") else str(i+1),
                str(len(text)),
                str(len(text.split())),
                str(sum(1 for c in text if c.isspace())),
                str(sum(1 for c in text if not c.isalnum() and not c.isspace())),
                clean(r[cols["duration"]]) if cols.get("duration") else "N/A",
                clean(r[cols["wpm"]]) if cols.get("wpm") else "N/A"
            ])

        story.append(table(rows, [15*mm, 28*mm, 22*mm, 22*mm, 28*mm, 25*mm, 25*mm]))

        story.append(Paragraph(
            "Phrase length, spaces and punctuation provide task-complexity descriptors. "
            "Completion time should therefore be interpreted alongside WPM/CPS rather "
            "than used alone to compare phrases of different lengths.",
            PDF_STYLES["Bodyx"]
        ))

    # Correlation
    story.append(Paragraph("9. Relationship Analysis", PDF_STYLES["H1x"]))
    pairs = [
        ("wpm", "text_accuracy", "WPM vs Accuracy"),
        ("wpm", "duration", "WPM vs Duration"),
        ("duration", "char_count", "Duration vs Characters"),
        ("backspaces", "edit_distance", "Backspaces vs Edit Distance"),
        ("iki_mean", "wpm", "Mean IKI vs WPM"),
        ("longest_pause", "wpm", "Longest Pause vs WPM"),
    ]

    corr_rows = [["Relationship", "Pearson r", "p-value", "N"]]
    if SCIPY:
        for a, b, label in pairs:
            if not cols.get(a) or not cols.get(b):
                continue
            x = num(df, cols[a])
            y = num(df, cols[b])
            valid = pd.DataFrame({"x": x, "y": y}).dropna()
            if len(valid) >= 3:
                r, p = stats.pearsonr(valid.x, valid.y)
                corr_rows.append([label, clean(r, 3), clean(p, 4), str(len(valid))])

    if len(corr_rows) > 1:
        story.append(table(corr_rows, [80*mm, 30*mm, 30*mm, 20*mm]))
    else:
        story.append(Paragraph(
            "Insufficient observations for correlation analysis. Correlation values "
            "from very small samples should not be treated as reliable evidence.",
            PDF_STYLES["Bodyx"]
        ))

    # Gaze/head section
    story.append(PageBreak())
    story.append(Paragraph("10. Gaze & Head-Pose Availability", PDF_STYLES["H1x"]))

    gaze_available = cols.get("raw_x") and cols.get("raw_y")
    pose_available = cols.get("yaw") or cols.get("pitch")
    dwell_available = cols.get("dwell")

    if gaze_available:
        story.append(Paragraph(
            "Gaze coordinates were detected. The report can analyze trajectory and, "
            "when smoothed coordinates are also available, raw-vs-smoothed movement.",
            PDF_STYLES["Bodyx"]
        ))
    else:
        story.append(Paragraph(
            "No raw gaze X/Y coordinate stream was detected in this CSV. "
            "Therefore gaze trajectory, gaze jitter and EMA smoothing effectiveness "
            "cannot be measured from this file.",
            PDF_STYLES["Bodyx"]
        ))

    if pose_available:
        story.append(Paragraph(
            "Head-pose information was detected and can be summarized using yaw/pitch statistics.",
            PDF_STYLES["Bodyx"]
        ))
    else:
        story.append(Paragraph(
            "No yaw/pitch head-pose stream was detected.",
            PDF_STYLES["Bodyx"]
        ))

    if dwell_available:
        story.append(Paragraph(
            "Dwell-time information was detected.",
            PDF_STYLES["Bodyx"]
        ))
    else:
        story.append(Paragraph(
            "No dwell-time field was detected, so dwell-to-click performance cannot "
            "be evaluated from this dataset.",
            PDF_STYLES["Bodyx"]
        ))

    # Visuals
    if plots:
        story.append(PageBreak())
        story.append(Paragraph("11. Visual Analysis", PDF_STYLES["H1x"]))
        for title, path in plots:
            story.append(Paragraph(title, PDF_STYLES["H2x"]))
            story.append(Image(str(path), width=170*mm, height=92*mm))
            story.append(Spacer(1, 5))

    # Data quality
    story.append(PageBreak())
    story.append(Paragraph("12. Data Quality & Research Notes", PDF_STYLES["H1x"]))

    missing = []
    for logical, col in cols.items():
        if col in df.columns:
            missing.append([
                logical,
                col,
                f"{df[col].isna().mean()*100:.1f}%"
            ])

    if missing:
        story.append(table(
            [["Variable", "CSV Column", "Missing"]] + missing,
            [50*mm, 80*mm, 30*mm]
        ))

    warnings = []

    # is_correct disagreement
    if cols.get("correct") and cols.get("target") and cols.get("entered"):
        flag = df[cols["correct"]].astype(str).str.lower().str.strip().isin(
            ["true", "1", "yes", "correct", "success"]
        )
        actual = df["__final_response_correct"]
        disagreement = (flag != actual).sum()
        if disagreement:
            warnings.append(
                f"{disagreement} trial(s) have a disagreement between the CSV correctness "
                "flag and exact final-response comparison. Review the original application logic."
            )

    if len(df) < 10:
        warnings.append(
            f"Only {len(df)} rows are available. This is suitable for validating the "
            "analysis pipeline but is too small for strong statistical/generalization claims."
        )

    if mcol and df[mcol].nunique(dropna=True) < 2:
        warnings.append(
            "Only one input method is present. Comparative claims require at least two conditions."
        )

    if warnings:
        story.append(Paragraph("Important observations", PDF_STYLES["H2x"]))
        for w in warnings:
            story.append(Paragraph("• " + w, PDF_STYLES["Bodyx"]))

    story.append(Paragraph(
        "Research interpretation: descriptive differences are not automatically evidence "
        "of superiority. For a thesis-quality comparison, collect repeated trials across "
        "multiple participants under matched task conditions and analyze participant-level "
        "paired observations.",
        PDF_STYLES["Bodyx"]
    ))

    # Appendix
    story.append(PageBreak())
    story.append(Paragraph("Appendix A — Source Files", PDF_STYLES["H1x"]))
    for f in source_files:
        story.append(Paragraph(Path(f).name, PDF_STYLES["Bodyx"]))

    story.append(Paragraph("Appendix B — Processed Dataset", PDF_STYLES["H1x"]))
    story.append(Paragraph(
        "The output folder contains combined_processed_data.csv. Derived columns are "
        "prefixed with __ so the original CSV fields can be distinguished from calculated metrics.",
        PDF_STYLES["Bodyx"]
    ))

    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)

    return pdf, plots


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def pick_files():
    if not TK:
        raise RuntimeError(
            "Tkinter is unavailable. Pass CSV paths on the command line."
        )

    root = tk.Tk()
    root.withdraw()
    root.update()
    files = filedialog.askopenfilenames(
        title="Select CSV file(s)",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()
    return list(files)


def popup(title, message, error=False):
    if not TK:
        print(message)
        return
    try:
        root = tk.Tk()
        root.withdraw()
        if error:
            messagebox.showerror(title, message)
        else:
            messagebox.showinfo(title, message)
        root.destroy()
    except Exception:
        print(message)


def main():
    try:
        files = sys.argv[1:] if len(sys.argv) > 1 else pick_files()

        if not files:
            print("No CSV selected.")
            return

        for f in files:
            if not Path(f).exists():
                raise FileNotFoundError(f)

        print("Loading CSV files...")
        frames = []
        for f in files:
            try:
                d = pd.read_csv(f)
            except UnicodeDecodeError:
                d = pd.read_csv(f, encoding="latin1")
            d["__source_file"] = Path(f).name
            frames.append(d)

        df = pd.concat(frames, ignore_index=True, sort=False)

        print(f"Loaded {len(df)} rows from {len(files)} file(s).")
        print("Columns:")
        for c in df.columns:
            print("  ", c)

        cols = detect_columns(df)
        df, cols = add_derived_metrics(df, cols)

        print("\nDetected variables:")
        for k, v in cols.items():
            print(f"  {k}: {v}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(files[0]).resolve().parent / f"gaze_analysis_report_{timestamp}"
        out.mkdir(parents=True, exist_ok=True)

        # Audit files
        df.to_csv(out / "combined_processed_data.csv", index=False)

        with open(out / "detected_columns.json", "w", encoding="utf-8") as f:
            json.dump(cols, f, indent=2, default=str)

        with open(out / "source_files.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(str(Path(x).resolve()) for x in files))

        print("\nGenerating PDF and visualizations...")
        pdf, plots = build_pdf(df, cols, files, out)

        print("\n==============================================")
        print("ANALYSIS COMPLETE")
        print("==============================================")
        print("PDF:")
        print(pdf)
        print("\nProcessed CSV:")
        print(out / "combined_processed_data.csv")
        print("\nOutput folder:")
        print(out)
        print(f"\nVisualizations: {len(plots)}")

        popup(
            "Analysis Complete",
            f"Report generated successfully.\n\n{pdf}\n\n"
            f"Processed data:\n{out / 'combined_processed_data.csv'}"
        )

    except Exception as e:
        traceback.print_exc()
        popup(
            "Analysis Failed",
            f"{type(e).__name__}: {e}\n\nCheck the terminal for details.",
            error=True
        )


if __name__ == "__main__":
    main()
