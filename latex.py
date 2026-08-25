"""
latex.py  —  Convert experiment CSV files -> LaTeX reports (custom design)

Usage:
    python latex.py                       # converts ALL CSVs in csv/ folder
    python latex.py <path/to/file.csv>    # converts a specific CSV file

Output: latex/<same_filename>.tex
"""

import os
import sys
import ast
import json
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR    = os.path.join(SCRIPT_DIR, "csv")
LATEX_DIR  = os.path.join(SCRIPT_DIR, "latex")

# How many keystroke entries to print per line in the timestamp table
STAMPS_PER_LINE = 8


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters in a plain-text string."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&",  r"\&"),
        ("%",  r"\%"),
        ("$",  r"\$"),
        ("#",  r"\#"),
        ("_",  r"\_"),
        ("{",  r"\{"),
        ("}",  r"\}"),
        ("~",  r"\textasciitilde{}"),
        ("^",  r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _parse_timestamps(raw) -> list:
    """
    Parse the letter_timestamps field.
    Accepts Python-list-style strings or proper JSON.
    Returns a list of [char, time] pairs, or [] on failure.
    """
    if pd.isna(raw) or str(raw).strip() == "":
        return []
    raw = str(raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        return ast.literal_eval(raw)
    except Exception:
        return []


def _format_timestamps_latex(pairs: list, per_line: int = STAMPS_PER_LINE) -> str:
    """
    Convert [[char, time], ...] pairs into multi-line LaTeX tabular cell.
    Groups per_line keystrokes per row, renders <SP> as SP, <BS> as BS.
    """
    if not pairs:
        return r"\textit{n/a}"

    tokens = []
    for item in pairs:
        ch  = str(item[0])
        t   = float(item[1])
        if ch in ("<SP>", " "):
            label = "SP"
        elif ch == "<BS>":
            label = "BS"
        else:
            label = _latex_escape(ch)
        tokens.append(f"{label}:{t:.3f}")

    rows = []
    for i in range(0, len(tokens), per_line):
        chunk = tokens[i : i + per_line]
        rows.append(",\\ ".join(chunk))

    inner = "\\\\\n".join(rows)
    return (
        r"\begin{tabular}[t]{@{}l@{}}" + "\n"
        + inner + "\n"
        + r"\end{tabular}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core converter
# ─────────────────────────────────────────────────────────────────────────────

def csv_to_latex(csv_path: str) -> str:
    """Convert a single experiment CSV to the custom LaTeX design."""
    os.makedirs(LATEX_DIR, exist_ok=True)

    df = pd.read_csv(csv_path)

    # ── Metadata from CSV data / filename ─────────────────────────────────────
    basename = os.path.splitext(os.path.basename(csv_path))[0]
    parts    = basename.split("_")

    if "participant_name" in df.columns and not df["participant_name"].empty:
        participant_display = str(df["participant_name"].iloc[0]).capitalize()
    else:
        participant_display = parts[2].capitalize() if len(parts) > 2 else "?"

    # ── Summary counts ────────────────────────────────────────────────────────
    total_trials   = len(df)
    correct_col    = "is_correct" if "is_correct" in df.columns else None
    correct_trials = int(df[correct_col].astype(str).str.lower().eq("true").sum()) \
                     if correct_col else 0

    # ── Section 1 — Trial Performance rows ───────────────────────────────────
    trial_rows = []
    for _, row in df.iterrows():
        trial      = int(row.get("trial_number",       0))
        method     = str(row.get("method",             "")).capitalize()
        stimulus   = _latex_escape(str(row.get("stimulus",        "")))
        typed_resp = _latex_escape(str(row.get("typed_response",  "")))
        correct    = str(row.get("is_correct", "")).lower()
        correct_str = "Yes" if correct == "true" else "No"
        duration   = f"{float(row.get('typing_duration_sec', 0)):.3f}"
        spaces     = int(row.get("spacebar_count",  0))
        backspaces = int(row.get("backspace_count", 0))
        chars      = int(row.get("char_count",      0))

        trial_rows.append(
            f"{trial} &\n"
            f"{method} &\n"
            f"{stimulus} &\n"
            f"{typed_resp} &\n"
            f"{correct_str} &\n"
            f"{duration} &\n"
            f"{spaces} &\n"
            f"{backspaces} &\n"
            f"{chars}\n"
            r"\\"
        )

    trial_table_body = "\n\n".join(trial_rows)

    # ── Section 2 — Keystroke Timestamps rows ────────────────────────────────
    has_timestamps = "letter_timestamps" in df.columns
    ts_rows = []
    if has_timestamps:
        for _, row in df.iterrows():
            trial  = int(row.get("trial_number", 0))
            pairs  = _parse_timestamps(row.get("letter_timestamps", ""))
            ts_str = _format_timestamps_latex(pairs)
            ts_rows.append(
                f"{trial} &\n"
                f"{ts_str}\n"
                r"\\" + r"[0.12cm]"
            )

    # ── Build keystroke section only if data exists ───────────────────────────
    if has_timestamps and ts_rows:
        ts_table_body = "\n\n".join(ts_rows)
        keystroke_section = (
            "\n\\vspace{0.35cm}\n\n"
            "% =========================================================\n"
            "% KEYSTROKE TIMESTAMPS\n"
            "% =========================================================\n\n"
            "\\noindent\n"
            "{\\large\\bfseries 2. Keystroke Timestamps}\n\n"
            "\\vspace{0.08cm}\n\n"
            "\\begin{center}\n"
            "\\begin{tabularx}{\\textwidth}{\n"
            "    C{0.55in}\n"
            "    X\n"
            "}\n"
            "\\toprule\n\n"
            "\\rowcolor{headergray}\n"
            "\\textbf{Trial} &\n"
            "\\textbf{Keystroke Timestamps (seconds)}\n"
            "\\\\\n\n"
            "\\midrule\n\n"
            + ts_table_body + "\n\n"
            "\\bottomrule\n"
            "\\end{tabularx}\n"
            "\\end{center}\n"
        )
    else:
        keystroke_section = ""

    # ── Full document ─────────────────────────────────────────────────────────
    full_doc = (
        "\\documentclass[10pt,a4paper]{article}\n\n"
        "\\usepackage[a4paper,landscape,margin=0.55in]{geometry}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{array}\n"
        "\\usepackage{tabularx}\n"
        "\\usepackage[table]{xcolor}\n"
        "\\usepackage{ragged2e}\n"
        "\\usepackage{helvet}\n\n"
        "\\renewcommand{\\familydefault}{\\sfdefault}\n\n"
        "% Simple colors\n"
        "\\definecolor{headergray}{RGB}{235,235,235}\n"
        "\\definecolor{textgray}{RGB}{80,80,80}\n\n"
        "% Column types\n"
        "\\newcolumntype{C}[1]{>{\\centering\\arraybackslash}p{#1}}\n"
        "\\newcolumntype{L}[1]{>{\\RaggedRight\\arraybackslash}p{#1}}\n\n"
        "\\setlength{\\tabcolsep}{5pt}\n"
        "\\renewcommand{\\arraystretch}{1.3}\n\n"
        "\\begin{document}\n\n"
        "% =========================================================\n"
        "% TITLE\n"
        "% =========================================================\n\n"
        "\\begin{center}\n"
        "    {\\Large\\bfseries Typing Experiment Data}\\\\\n"
        "    \\vspace{2pt}\n"
        f"    Participant: \\textbf{{{participant_display}}}\n"
        "\\end{center}\n\n"
        "\\vspace{0.25cm}\n\n"
        "% =========================================================\n"
        "% TRIAL PERFORMANCE\n"
        "% =========================================================\n\n"
        "\\noindent\n"
        "{\\large\\bfseries 1. Trial Performance}\n\n"
        "\\vspace{0.08cm}\n\n"
        "\\begin{center}\n"
        "\\begin{tabularx}{\\textwidth}{\n"
        "    C{0.45in}\n"
        "    C{0.65in}\n"
        "    L{2.35in}\n"
        "    L{2.55in}\n"
        "    C{0.65in}\n"
        "    C{0.65in}\n"
        "    C{0.55in}\n"
        "    C{0.75in}\n"
        "    C{0.55in}\n"
        "}\n"
        "\\toprule\n\n"
        "\\rowcolor{headergray}\n"
        "\\textbf{Trial} &\n"
        "\\textbf{Method} &\n"
        "\\textbf{Stimulus} &\n"
        "\\textbf{Typed Response} &\n"
        "\\textbf{Correct} &\n"
        "\\textbf{Time (s)} &\n"
        "\\textbf{Space} &\n"
        "\\textbf{Backspace} &\n"
        "\\textbf{Chars}\n"
        "\\\\\n\n"
        "\\midrule\n\n"
        + trial_table_body + "\n\n"
        "\\bottomrule\n"
        "\\end{tabularx}\n"
        "\\end{center}\n"
        + keystroke_section +
        "\n\\vfill\n\n"
        "% =========================================================\n"
        "% SIMPLE FOOTER\n"
        "% =========================================================\n\n"
        "\\noindent\n"
        f"\\textbf{{Participant:}} {participant_display}\n"
        "\\hfill\n"
        f"\\textbf{{Total Trials:}} {total_trials}\n"
        "\\hfill\n"
        f"\\textbf{{Correct Trials:}} {correct_trials}/{total_trials}\n\n"
        "\\end{document}\n"
    )

    out_path = os.path.join(LATEX_DIR, basename + ".tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_doc)

    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Determine which CSV files to process
    if len(sys.argv) > 1:
        # Specific file(s) passed as arguments
        csv_files = [os.path.abspath(p) for p in sys.argv[1:]]
        missing   = [p for p in csv_files if not os.path.isfile(p)]
        if missing:
            print(f"[ERROR] File(s) not found: {missing}")
            sys.exit(1)
    else:
        # Auto-discover all CSVs in the csv/ folder
        if not os.path.isdir(CSV_DIR):
            print(f"[ERROR] csv/ folder not found at: {CSV_DIR}")
            print("        Run the experiment first to generate CSV files.")
            sys.exit(1)
        csv_files = sorted(
            [os.path.join(CSV_DIR, f)
             for f in os.listdir(CSV_DIR) if f.endswith(".csv")]
        )
        if not csv_files:
            print("[ERROR] No CSV files found in csv/ folder.")
            sys.exit(1)

    # Convert each file
    print(f"\nConverting {len(csv_files)} CSV file(s) -> LaTeX\n" + "-"*50)
    for csv_path in csv_files:
        try:
            out = csv_to_latex(csv_path)
            print(f"  OK  {os.path.basename(csv_path)}")
            print(f"      -> {os.path.relpath(out, SCRIPT_DIR)}")
        except Exception as e:
            print(f"  FAIL  {os.path.basename(csv_path)}  [{e}]")

    print(f"\nDone. LaTeX files saved to:  latex/\n")


if __name__ == "__main__":
    main()
