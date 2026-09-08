import os, sys, ast, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR    = os.path.join(SCRIPT_DIR, "csv")
OUT_DIR    = os.path.join(SCRIPT_DIR, "analysis_output")
PLOT_DIR   = os.path.join(OUT_DIR, "plots")

# ─── Aesthetic constants ──────────────────────────────────────────────────────
DARK_BG     = "#0f1117"
CARD_BG     = "#1a1d27"
GRID_COLOR  = "#2a2d3a"
TEXT_COLOR  = "#e0e0e8"
ACCENT_1    = "#6366f1"   # indigo
ACCENT_2    = "#22d3ee"   # cyan
ACCENT_3    = "#f472b6"   # pink
ACCENT_4    = "#34d399"   # emerald
ACCENT_5    = "#fb923c"   # orange
ACCENT_6    = "#a78bfa"   # violet
PALETTE     = [ACCENT_1, ACCENT_2, ACCENT_3, ACCENT_4, ACCENT_5, ACCENT_6]
METHOD_PAL  = {"overt": ACCENT_2, "covert": ACCENT_3}


def setup_style():
    """Apply a premium dark theme for all matplotlib/seaborn plots."""
    plt.rcParams.update({
        "figure.facecolor":   DARK_BG,
        "axes.facecolor":     CARD_BG,
        "axes.edgecolor":     GRID_COLOR,
        "axes.labelcolor":    TEXT_COLOR,
        "axes.grid":          True,
        "grid.color":         GRID_COLOR,
        "grid.alpha":         0.5,
        "text.color":         TEXT_COLOR,
        "xtick.color":        TEXT_COLOR,
        "ytick.color":        TEXT_COLOR,
        "legend.facecolor":   CARD_BG,
        "legend.edgecolor":   GRID_COLOR,
        "legend.labelcolor":  TEXT_COLOR,
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Segoe UI", "Arial", "DejaVu Sans"],
        "font.size":          11,
        "axes.titlesize":     14,
        "axes.titleweight":   "bold",
        "figure.titlesize":   16,
        "figure.titleweight": "bold",
        "savefig.facecolor":  DARK_BG,
        "savefig.edgecolor":  DARK_BG,
        "savefig.dpi":        200,
        "savefig.bbox":       "tight",
    })
    sns.set_palette(PALETTE)

# ═════════════════════════════════════════════════════════════════════════════
#  1. DATA LOADING & CLEANING
# ═════════════════════════════════════════════════════════════════════════════

def parse_timestamps(raw) -> list:
    """Parse letter_timestamps → list of [char, time_sec] pairs."""
    if pd.isna(raw) or str(raw).strip() in ("", "[]"):
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


def load_all_csvs(csv_dir: str) -> pd.DataFrame:
    """Load every CSV in *csv_dir*, concatenate, and clean."""
    frames = []
    for fn in sorted(os.listdir(csv_dir)):
        if not fn.endswith(".csv"):
            continue
        path = os.path.join(csv_dir, fn)
        try:
            df = pd.read_csv(path)
            df["_source_file"] = fn
            frames.append(df)
        except Exception as e:
            print(f"  [WARN] Skipping {fn}: {e}")
    if not frames:
        print("[ERROR] No CSV files found."); sys.exit(1)

    data = pd.concat(frames, ignore_index=True)

    # Normalise method column
    data["method"] = data["method"].astype(str).str.strip().str.lower()

    # Normalise participant name
    data["participant_name"] = (
        data["participant_name"].astype(str).str.strip().str.lower()
    )

    # Parse timestamps once
    data["_timestamps"] = data["letter_timestamps"].apply(parse_timestamps)

    # Mark abandoned trials (empty response or zero duration)
    data["_abandoned"] = (
        data["typed_response"].isna()
        | (data["typed_response"].astype(str).str.strip() == "")
        | data["typing_duration_sec"].isna()
        | (data["typing_duration_sec"] == 0)
    )

    print(f"  Loaded {len(data)} trials from {len(frames)} files  "
          f"({data['_abandoned'].sum()} abandoned)")
    return data

# ═════════════════════════════════════════════════════════════════════════════
#  2. METRIC COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

# ── 2a. Character Error Rate (Levenshtein) ────────────────────────────────────

def levenshtein_ops(s: str, t: str) -> tuple:
    """Return (substitutions, deletions, insertions) via Wagner-Fischer."""
    m, n = len(s), len(t)
    # dp[i][j] = (cost, subs, dels, ins)
    dp = [[(0, 0, 0, 0)] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, n + 1):
        dp[0][j] = (j, 0, 0, j)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i-1] == t[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                sub = dp[i-1][j-1]
                dele = dp[i-1][j]
                ins = dp[i][j-1]
                candidates = [
                    (sub[0]+1, sub[1]+1, sub[2],   sub[3]),    # substitution
                    (dele[0]+1, dele[1],  dele[2]+1, dele[3]),  # deletion
                    (ins[0]+1,  ins[1],   ins[2],   ins[3]+1),  # insertion
                ]
                dp[i][j] = min(candidates, key=lambda x: x[0])
    _, S, D, I = dp[m][n]
    return S, D, I


def compute_cer(stimulus: str, response: str) -> float:
    """Character Error Rate (%) using Levenshtein distance."""
    stim = str(stimulus).strip().lower()
    resp = str(response).strip().lower()
    if not stim:
        return 0.0 if not resp else 100.0
    S, D, I = levenshtein_ops(stim, resp)
    return (S + D + I) / len(stim) * 100


# ── 2b. Keystroke-level metrics ───────────────────────────────────────────────

def extract_keystroke_metrics(ts_pairs: list) -> dict:
    """
    From a list of [char, time] pairs, compute:
      - total_keystrokes
      - backspace_keystrokes
      - char_keystrokes (non-backspace producing keys)
      - ikis: list of inter-key intervals
      - correction_times: list of time deltas for corrections
    """
    result = {
        "total_keystrokes": 0,
        "backspace_keystrokes": 0,
        "char_keystrokes": 0,
        "ikis": [],
        "correction_times": [],
    }
    if not ts_pairs or len(ts_pairs) == 0:
        return result

    times = []
    chars = []
    for pair in ts_pairs:
        try:
            ch = str(pair[0])
            t  = float(pair[1])
            chars.append(ch)
            times.append(t)
        except (IndexError, ValueError, TypeError):
            continue

    n = len(times)
    result["total_keystrokes"] = n

    bs_count = sum(1 for c in chars if c == "<BS>")
    result["backspace_keystrokes"] = bs_count
    result["char_keystrokes"] = n - bs_count

    # Inter-key intervals
    ikis = []
    for i in range(1, n):
        iki = times[i] - times[i-1]
        if iki >= 0:
            ikis.append(iki)
    result["ikis"] = ikis

    # Correction times: time from the erroneous key to the <BS> that fixes it
    correction_times = []
    for i, ch in enumerate(chars):
        if ch == "<BS>" and i > 0:
            # Look back to find the most recent non-BS key
            for j in range(i - 1, -1, -1):
                if chars[j] != "<BS>":
                    ct = times[i] - times[j]
                    if ct >= 0:
                        correction_times.append(ct)
                    break
    result["correction_times"] = correction_times

    return result


def compute_trial_metrics(row: pd.Series) -> dict:
    """Compute all per-trial metrics from a single row."""
    metrics = {}

    dur   = float(row.get("typing_duration_sec", 0) or 0)
    chars = int(row.get("char_count", 0) or 0)
    stim  = str(row.get("stimulus", ""))
    resp  = str(row.get("typed_response", ""))
    correct = str(row.get("is_correct", "")).lower() == "true"
    ts    = row.get("_timestamps", [])

    # ── Typing Speed ──────────────────────────────────────────────────────
    if dur > 0 and chars > 0:
        metrics["wpm"] = (chars / 5) / (dur / 60)
        metrics["cpm"] = chars / (dur / 60)
    else:
        metrics["wpm"] = 0.0
        metrics["cpm"] = 0.0

    # ── Task Completion Time ──────────────────────────────────────────────
    metrics["tct_sec"] = dur

    # ── Final Text Accuracy ───────────────────────────────────────────────
    metrics["is_correct"] = correct

    # ── CER ───────────────────────────────────────────────────────────────
    metrics["cer"] = compute_cer(stim, resp)

    # ── Keystroke-level ───────────────────────────────────────────────────
    ks = extract_keystroke_metrics(ts)

    total_ks = ks["total_keystrokes"]
    bs_ks    = ks["backspace_keystrokes"]
    ikis     = ks["ikis"]
    corr_ts  = ks["correction_times"]

    # KSPC
    if chars > 0 and total_ks > 0:
        metrics["kspc"] = total_ks / chars
    else:
        metrics["kspc"] = np.nan

    # Backspace Rate
    if total_ks > 0:
        metrics["backspace_rate"] = (bs_ks / total_ks) * 100
    else:
        metrics["backspace_rate"] = 0.0

    # IKI statistics
    if ikis:
        metrics["mean_iki"]  = np.mean(ikis)
        metrics["sd_iki"]    = np.std(ikis, ddof=1) if len(ikis) > 1 else 0.0
        metrics["long_pause_rate"] = (
            sum(1 for x in ikis if x > 3.0) / len(ikis) * 100
        )
    else:
        metrics["mean_iki"]  = np.nan
        metrics["sd_iki"]    = np.nan
        metrics["long_pause_rate"] = np.nan

    # Correction Time
    if corr_ts:
        metrics["mean_correction_time"] = np.mean(corr_ts)
    else:
        metrics["mean_correction_time"] = np.nan

    # ── Composite metrics ─────────────────────────────────────────────────

    # Input Efficiency Index (IEI)
    acc_frac = 1.0 if correct else 0.0
    bs_ratio = (bs_ks / total_ks) if total_ks > 0 else 0.0
    metrics["iei"] = metrics["wpm"] * acc_frac * (1 - bs_ratio)

    # Cognitive Load Proxy Index (CLPI)
    w1, w2, w3 = 0.4, 0.3, 0.3
    cv_iki = (metrics["sd_iki"] / metrics["mean_iki"]
              if metrics.get("mean_iki") and metrics["mean_iki"] > 0
              and not np.isnan(metrics.get("sd_iki", np.nan))
              else 0.0)
    lpr = metrics.get("long_pause_rate", 0.0)
    lpr = lpr / 100 if not np.isnan(lpr) else 0.0
    bs_char = (bs_ks / chars) if chars > 0 else 0.0
    metrics["clpi"] = w1 * cv_iki + w2 * lpr + w3 * bs_char

    # Store raw counts for downstream aggregation
    metrics["total_keystrokes"] = total_ks
    metrics["backspace_keystrokes"] = bs_ks
    metrics["stimulus_length"] = len(stim)

    return metrics

# ═════════════════════════════════════════════════════════════════════════════
#  3. AGGREGATION
# ═════════════════════════════════════════════════════════════════════════════

PRIMARY_METRICS = [
    "wpm", "cpm", "tct_sec", "cer", "kspc",
    "backspace_rate", "mean_iki",
]
SECONDARY_METRICS = [
    "sd_iki", "long_pause_rate", "mean_correction_time",
    "iei", "clpi",
]
ALL_METRICS = PRIMARY_METRICS + SECONDARY_METRICS


def build_trial_table(data: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics for every non-abandoned trial."""
    active = data[~data["_abandoned"]].copy()
    records = []
    for idx, row in active.iterrows():
        m = compute_trial_metrics(row)
        m["participant_name"] = row["participant_name"]
        m["method"]           = row["method"]
        m["trial_number"]     = row["trial_number"]
        m["stimulus"]         = row["stimulus"]
        m["typed_response"]   = row["typed_response"]
        m["source_file"]      = row["_source_file"]
        records.append(m)
    return pd.DataFrame(records)


def build_participant_summary(trials: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics per participant."""
    agg = {}
    for col in ALL_METRICS:
        agg[col] = "mean"
    agg["is_correct"] = "mean"  # proportion correct
    agg["trial_number"] = "count"

    summary = trials.groupby(["participant_name", "method"]).agg(agg)
    summary = summary.rename(columns={
        "trial_number": "num_trials",
        "is_correct": "accuracy_pct",
    })
    summary["accuracy_pct"] *= 100
    return summary.reset_index()


def build_method_comparison(trials: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics per method (overt / covert)."""
    rows = []
    for method in ["overt", "covert"]:
        sub = trials[trials["method"] == method]
        if sub.empty:
            continue
        row = {"method": method, "num_trials": len(sub), "num_participants": sub["participant_name"].nunique()}
        for col in ALL_METRICS:
            vals = sub[col].dropna()
            row[f"{col}_mean"] = vals.mean() if len(vals) else np.nan
            row[f"{col}_std"]  = vals.std()  if len(vals) > 1 else np.nan
            row[f"{col}_median"] = vals.median() if len(vals) else np.nan
        # Accuracy
        row["accuracy_pct"] = sub["is_correct"].mean() * 100
        rows.append(row)
    return pd.DataFrame(rows)


def run_statistical_tests(trials: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney U tests for overt vs covert on each metric."""
    overt  = trials[trials["method"] == "overt"]
    covert = trials[trials["method"] == "covert"]
    results = []
    test_metrics = ALL_METRICS + ["is_correct"]
    for col in test_metrics:
        a = overt[col].dropna().values
        b = covert[col].dropna().values
        if len(a) < 2 or len(b) < 2:
            results.append({
                "metric": col,
                "overt_n": len(a), "covert_n": len(b),
                "overt_median": np.median(a) if len(a) else np.nan,
                "covert_median": np.median(b) if len(b) else np.nan,
                "U_statistic": np.nan, "p_value": np.nan,
                "significant_005": "N/A (insufficient data)",
            })
            continue
        U, p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
        results.append({
            "metric": col,
            "overt_n": len(a), "covert_n": len(b),
            "overt_median": np.median(a),
            "covert_median": np.median(b),
            "U_statistic": U, "p_value": round(p, 6),
            "significant_005": "Yes" if p < 0.05 else "No",
        })
    return pd.DataFrame(results)

# ═════════════════════════════════════════════════════════════════════════════
#  4. VISUALISATION SUITE
# ═════════════════════════════════════════════════════════════════════════════

def _save(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"    [OK] {name}")


def _method_colors(data):
    """Return a list of colours matching the method column."""
    return [METHOD_PAL.get(m, ACCENT_1) for m in data["method"]]


def plot_wpm_by_participant(trials):
    """Grouped bar — WPM per participant, coloured by method."""
    fig, ax = plt.subplots(figsize=(14, 6))
    pt = trials.groupby(["participant_name", "method"])["wpm"].mean().reset_index()
    pt = pt.sort_values("wpm", ascending=False)

    bars = ax.bar(
        range(len(pt)), pt["wpm"],
        color=[METHOD_PAL.get(m, ACCENT_1) for m in pt["method"]],
        edgecolor="none", width=0.7, alpha=0.9,
    )
    ax.set_xticks(range(len(pt)))
    ax.set_xticklabels(
        [f"{r.participant_name}\n({r.method})" for _, r in pt.iterrows()],
        rotation=45, ha="right", fontsize=9,
    )
    ax.set_ylabel("Words Per Minute (WPM)")
    ax.set_title("WPM by Participant")

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=METHOD_PAL["overt"], label="Overt"),
               Patch(facecolor=METHOD_PAL["covert"], label="Covert")]
    ax.legend(handles=handles, loc="upper right", framealpha=0.8)
    fig.tight_layout()
    _save(fig, "wpm_by_participant.png")


def plot_wpm_boxplot(trials):
    """Box + strip — Overt vs Covert WPM distribution."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=trials, x="method", y="wpm", palette=METHOD_PAL,
                linewidth=1.5, fliersize=0, ax=ax, boxprops=dict(alpha=0.6))
    sns.stripplot(data=trials, x="method", y="wpm", palette=METHOD_PAL,
                  size=7, alpha=0.85, jitter=0.2, ax=ax, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Words Per Minute (WPM)")
    ax.set_xlabel("")
    ax.set_title("WPM Distribution: Overt vs Covert")
    fig.tight_layout()
    _save(fig, "wpm_by_method_boxplot.png")


def plot_cer_by_participant(trials):
    """Grouped bar — CER per participant."""
    fig, ax = plt.subplots(figsize=(14, 6))
    pt = trials.groupby(["participant_name", "method"])["cer"].mean().reset_index()
    pt = pt.sort_values("cer", ascending=True)
    ax.bar(
        range(len(pt)), pt["cer"],
        color=[METHOD_PAL.get(m, ACCENT_1) for m in pt["method"]],
        edgecolor="none", width=0.7, alpha=0.9,
    )
    ax.set_xticks(range(len(pt)))
    ax.set_xticklabels(
        [f"{r.participant_name}\n({r.method})" for _, r in pt.iterrows()],
        rotation=45, ha="right", fontsize=9,
    )
    ax.set_ylabel("Character Error Rate (%)")
    ax.set_title("CER by Participant")
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=METHOD_PAL["overt"], label="Overt"),
               Patch(facecolor=METHOD_PAL["covert"], label="Covert")]
    ax.legend(handles=handles, loc="upper right", framealpha=0.8)
    fig.tight_layout()
    _save(fig, "cer_by_participant.png")


def plot_cer_boxplot(trials):
    """Box + strip — Overt vs Covert CER distribution."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=trials, x="method", y="cer", palette=METHOD_PAL,
                linewidth=1.5, fliersize=0, ax=ax, boxprops=dict(alpha=0.6))
    sns.stripplot(data=trials, x="method", y="cer", palette=METHOD_PAL,
                  size=7, alpha=0.85, jitter=0.2, ax=ax, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Character Error Rate (%)")
    ax.set_xlabel("")
    ax.set_title("CER Distribution: Overt vs Covert")
    fig.tight_layout()
    _save(fig, "cer_by_method_boxplot.png")


def plot_accuracy_by_method(trials):
    """Stacked bar — correct vs incorrect trials per method."""
    fig, ax = plt.subplots(figsize=(8, 6))
    counts = trials.groupby("method")["is_correct"].value_counts().unstack(fill_value=0)
    if True not in counts.columns:
        counts[True] = 0
    if False not in counts.columns:
        counts[False] = 0
    counts = counts[[True, False]]
    counts.columns = ["Correct", "Incorrect"]

    x = range(len(counts))
    ax.bar(x, counts["Correct"], color=ACCENT_4, edgecolor="none", label="Correct", alpha=0.9)
    ax.bar(x, counts["Incorrect"], bottom=counts["Correct"],
           color=ACCENT_3, edgecolor="none", label="Incorrect", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in counts.index])
    ax.set_ylabel("Number of Trials")
    ax.set_title("Trial Accuracy by Method")
    ax.legend(framealpha=0.8)

    # Annotate percentages
    for i, method in enumerate(counts.index):
        total = counts.loc[method].sum()
        pct = counts.loc[method, "Correct"] / total * 100 if total > 0 else 0
        ax.text(i, total + 0.3, f"{pct:.0f}%", ha="center", fontsize=11,
                fontweight="bold", color=ACCENT_4)
    fig.tight_layout()
    _save(fig, "accuracy_by_method.png")


def plot_iki_distribution(trials):
    """Violin — IKI distribution by method."""
    fig, ax = plt.subplots(figsize=(8, 6))
    valid = trials.dropna(subset=["mean_iki"])
    if valid.empty:
        plt.close(fig); return
    sns.violinplot(data=valid, x="method", y="mean_iki", palette=METHOD_PAL,
                   inner="box", linewidth=1.2, ax=ax, alpha=0.7)
    sns.stripplot(data=valid, x="method", y="mean_iki", palette=METHOD_PAL,
                  size=6, alpha=0.8, jitter=0.15, ax=ax, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Mean Inter-Key Interval (s)")
    ax.set_xlabel("")
    ax.set_title("IKI Distribution: Overt vs Covert")
    fig.tight_layout()
    _save(fig, "iki_distribution.png")


def plot_kspc_comparison(trials):
    """Bar — KSPC per participant."""
    fig, ax = plt.subplots(figsize=(14, 6))
    pt = trials.groupby(["participant_name", "method"])["kspc"].mean().reset_index()
    pt = pt.sort_values("kspc")
    ax.bar(
        range(len(pt)), pt["kspc"],
        color=[METHOD_PAL.get(m, ACCENT_1) for m in pt["method"]],
        edgecolor="none", width=0.7, alpha=0.9,
    )
    ax.axhline(y=1.0, color=ACCENT_4, linestyle="--", linewidth=1.5, alpha=0.7, label="Ideal KSPC = 1.0")
    ax.set_xticks(range(len(pt)))
    ax.set_xticklabels(
        [f"{r.participant_name}\n({r.method})" for _, r in pt.iterrows()],
        rotation=45, ha="right", fontsize=9,
    )
    ax.set_ylabel("Keystrokes Per Character (KSPC)")
    ax.set_title("KSPC by Participant")
    ax.legend(framealpha=0.8)
    fig.tight_layout()
    _save(fig, "kspc_comparison.png")


def plot_backspace_rate(trials):
    """Bar — Backspace rate per participant."""
    fig, ax = plt.subplots(figsize=(14, 6))
    pt = trials.groupby(["participant_name", "method"])["backspace_rate"].mean().reset_index()
    pt = pt.sort_values("backspace_rate")
    ax.bar(
        range(len(pt)), pt["backspace_rate"],
        color=[METHOD_PAL.get(m, ACCENT_1) for m in pt["method"]],
        edgecolor="none", width=0.7, alpha=0.9,
    )
    ax.set_xticks(range(len(pt)))
    ax.set_xticklabels(
        [f"{r.participant_name}\n({r.method})" for _, r in pt.iterrows()],
        rotation=45, ha="right", fontsize=9,
    )
    ax.set_ylabel("Backspace Rate (%)")
    ax.set_title("Backspace Rate by Participant")
    fig.tight_layout()
    _save(fig, "backspace_rate_comparison.png")


def plot_tct_vs_stimulus_length(trials):
    """Scatter + regression — TCT vs stimulus length."""
    fig, ax = plt.subplots(figsize=(10, 6))
    valid = trials.dropna(subset=["tct_sec", "stimulus_length"])
    valid = valid[valid["stimulus_length"] > 0]
    if valid.empty:
        plt.close(fig); return
    for method in ["overt", "covert"]:
        sub = valid[valid["method"] == method]
        if sub.empty:
            continue
        ax.scatter(sub["stimulus_length"], sub["tct_sec"],
                   c=METHOD_PAL[method], s=60, alpha=0.8, label=method.capitalize(),
                   edgecolors="white", linewidth=0.5)
        # Regression line
        if len(sub) > 2:
            z = np.polyfit(sub["stimulus_length"], sub["tct_sec"], 1)
            p = np.poly1d(z)
            xs = np.linspace(sub["stimulus_length"].min(), sub["stimulus_length"].max(), 100)
            ax.plot(xs, p(xs), color=METHOD_PAL[method], linewidth=2, alpha=0.6, linestyle="--")
    ax.set_xlabel("Stimulus Length (characters)")
    ax.set_ylabel("Task Completion Time (s)")
    ax.set_title("TCT vs Stimulus Length")
    ax.legend(framealpha=0.8)
    fig.tight_layout()
    _save(fig, "tct_by_stimulus_length.png")


def plot_metrics_heatmap(trials):
    """Heatmap — correlation matrix of primary metrics."""
    fig, ax = plt.subplots(figsize=(10, 8))
    cols = [c for c in PRIMARY_METRICS if c in trials.columns]
    corr = trials[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.8, linecolor=GRID_COLOR,
                ax=ax, cbar_kws={"shrink": 0.8},
                annot_kws={"fontsize": 10, "color": TEXT_COLOR})
    ax.set_title("Primary Metrics — Correlation Matrix")
    fig.tight_layout()
    _save(fig, "metrics_heatmap.png")


def plot_iei_clpi_scatter(trials):
    """Scatter — IEI vs CLPI by method."""
    fig, ax = plt.subplots(figsize=(9, 6))
    valid = trials.dropna(subset=["iei", "clpi"])
    if valid.empty:
        plt.close(fig); return
    for method in ["overt", "covert"]:
        sub = valid[valid["method"] == method]
        if sub.empty:
            continue
        ax.scatter(sub["clpi"], sub["iei"], c=METHOD_PAL[method],
                   s=70, alpha=0.85, label=method.capitalize(),
                   edgecolors="white", linewidth=0.5)
    ax.set_xlabel("Cognitive Load Proxy Index (CLPI)")
    ax.set_ylabel("Input Efficiency Index (IEI)")
    ax.set_title("IEI vs CLPI by Method")
    ax.legend(framealpha=0.8)
    fig.tight_layout()
    _save(fig, "iei_clpi_scatter.png")


def plot_summary_dashboard(trials, participant_summary):
    """Multi-panel 2×3 dashboard of key metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Eye-Gaze Text Entry — Summary Dashboard", fontsize=18, fontweight="bold", y=0.98)

    # (0,0) WPM by method
    ax = axes[0, 0]
    sns.boxplot(data=trials, x="method", y="wpm", palette=METHOD_PAL,
                linewidth=1.2, fliersize=0, ax=ax, boxprops=dict(alpha=0.6))
    sns.stripplot(data=trials, x="method", y="wpm", palette=METHOD_PAL,
                  size=5, alpha=0.8, jitter=0.2, ax=ax, edgecolor="white", linewidth=0.3)
    ax.set_title("WPM"); ax.set_xlabel(""); ax.set_ylabel("WPM")

    # (0,1) CER by method
    ax = axes[0, 1]
    sns.boxplot(data=trials, x="method", y="cer", palette=METHOD_PAL,
                linewidth=1.2, fliersize=0, ax=ax, boxprops=dict(alpha=0.6))
    sns.stripplot(data=trials, x="method", y="cer", palette=METHOD_PAL,
                  size=5, alpha=0.8, jitter=0.2, ax=ax, edgecolor="white", linewidth=0.3)
    ax.set_title("CER (%)"); ax.set_xlabel(""); ax.set_ylabel("CER (%)")

    # (0,2) Accuracy
    ax = axes[0, 2]
    acc = trials.groupby("method")["is_correct"].mean() * 100
    colors = [METHOD_PAL.get(m, ACCENT_1) for m in acc.index]
    bars = ax.bar(acc.index, acc.values, color=colors, edgecolor="none", alpha=0.9)
    for bar, val in zip(bars, acc.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold", color=TEXT_COLOR)
    ax.set_title("Accuracy (%)"); ax.set_xlabel(""); ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)

    # (1,0) KSPC by method
    ax = axes[1, 0]
    valid = trials.dropna(subset=["kspc"])
    sns.boxplot(data=valid, x="method", y="kspc", palette=METHOD_PAL,
                linewidth=1.2, fliersize=0, ax=ax, boxprops=dict(alpha=0.6))
    sns.stripplot(data=valid, x="method", y="kspc", palette=METHOD_PAL,
                  size=5, alpha=0.8, jitter=0.2, ax=ax, edgecolor="white", linewidth=0.3)
    ax.axhline(y=1.0, color=ACCENT_4, linestyle="--", linewidth=1.2, alpha=0.6)
    ax.set_title("KSPC"); ax.set_xlabel(""); ax.set_ylabel("KSPC")

    # (1,1) Mean IKI by method
    ax = axes[1, 1]
    valid = trials.dropna(subset=["mean_iki"])
    sns.boxplot(data=valid, x="method", y="mean_iki", palette=METHOD_PAL,
                linewidth=1.2, fliersize=0, ax=ax, boxprops=dict(alpha=0.6))
    sns.stripplot(data=valid, x="method", y="mean_iki", palette=METHOD_PAL,
                  size=5, alpha=0.8, jitter=0.2, ax=ax, edgecolor="white", linewidth=0.3)
    ax.set_title("Mean IKI (s)"); ax.set_xlabel(""); ax.set_ylabel("IKI (s)")

    # (1,2) Backspace Rate by method
    ax = axes[1, 2]
    sns.boxplot(data=trials, x="method", y="backspace_rate", palette=METHOD_PAL,
                linewidth=1.2, fliersize=0, ax=ax, boxprops=dict(alpha=0.6))
    sns.stripplot(data=trials, x="method", y="backspace_rate", palette=METHOD_PAL,
                  size=5, alpha=0.8, jitter=0.2, ax=ax, edgecolor="white", linewidth=0.3)
    ax.set_title("Backspace Rate (%)"); ax.set_xlabel(""); ax.set_ylabel("Backspace Rate (%)")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "summary_dashboard.png")


# ═════════════════════════════════════════════════════════════════════════════
#  5. TEXT REPORT
# ═════════════════════════════════════════════════════════════════════════════

def generate_report(data, trials, psummary, mcomp, stat_tests) -> str:
    """Generate a human-readable analysis report."""
    lines = []
    lines.append("=" * 72)
    lines.append("  EYE-GAZE TEXT ENTRY — EVALUATION METRICS REPORT")
    lines.append("=" * 72)
    lines.append("")

    # Overview
    lines.append("1. DATA OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"  Total CSV files loaded    : {data['_source_file'].nunique()}")
    lines.append(f"  Total trials (all)        : {len(data)}")
    lines.append(f"  Abandoned trials          : {data['_abandoned'].sum()}")
    lines.append(f"  Valid trials analysed      : {len(trials)}")
    lines.append(f"  Unique participants        : {trials['participant_name'].nunique()}")
    lines.append(f"  Overt trials               : {(trials['method'] == 'overt').sum()}")
    lines.append(f"  Covert trials              : {(trials['method'] == 'covert').sum()}")
    lines.append("")

    # Method comparison
    lines.append("2. METHOD COMPARISON (Overt vs Covert)")
    lines.append("-" * 40)
    for _, row in mcomp.iterrows():
        method = row["method"].upper()
        lines.append(f"  [{method}]  ({int(row['num_trials'])} trials, "
                     f"{int(row['num_participants'])} participants)")
        lines.append(f"    WPM             : {row.get('wpm_mean', 0):.2f} ± {row.get('wpm_std', 0):.2f}")
        lines.append(f"    CER (%)         : {row.get('cer_mean', 0):.2f} ± {row.get('cer_std', 0):.2f}")
        lines.append(f"    Accuracy (%)    : {row.get('accuracy_pct', 0):.1f}")
        lines.append(f"    TCT (s)         : {row.get('tct_sec_mean', 0):.2f} ± {row.get('tct_sec_std', 0):.2f}")
        lines.append(f"    KSPC            : {row.get('kspc_mean', 0):.3f} ± {row.get('kspc_std', 0):.3f}")
        lines.append(f"    Backspace Rate  : {row.get('backspace_rate_mean', 0):.2f}%")
        lines.append(f"    Mean IKI (s)    : {row.get('mean_iki_mean', 0):.3f} ± {row.get('mean_iki_std', 0):.3f}")
        lines.append(f"    IEI             : {row.get('iei_mean', 0):.3f}")
        lines.append(f"    CLPI            : {row.get('clpi_mean', 0):.4f}")
        lines.append("")

    # Statistical tests
    lines.append("3. STATISTICAL TESTS (Mann-Whitney U, α = 0.05)")
    lines.append("-" * 40)
    for _, row in stat_tests.iterrows():
        sig = row["significant_005"]
        marker = " *" if sig == "Yes" else ""
        if not np.isnan(row.get("U_statistic", np.nan)):
            lines.append(f"  {row['metric']:25s}  U = {row['U_statistic']:8.1f}  "
                         f"p = {row['p_value']:.4f}{marker}")
        else:
            lines.append(f"  {row['metric']:25s}  {row['significant_005']}")
    lines.append("")

    # Non-computable metrics
    lines.append("4. NON-COMPUTABLE METRICS (data not available)")
    lines.append("-" * 40)
    lines.append("  • Gaze Selection Accuracy  — requires gaze coordinate logs")
    lines.append("  • Dwell Time               — requires gaze coordinate logs")
    lines.append("  • Fixation Duration         — requires gaze coordinate logs")
    lines.append("  • GSJI                      — requires raw + smoothed gaze trajectories")
    lines.append("  • SUS Score                 — requires post-task questionnaire")
    lines.append("  • NASA-TLX                  — requires post-task questionnaire")
    lines.append("")

    lines.append("=" * 72)
    lines.append("  Report generated by analysis.py")
    lines.append("=" * 72)
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
#  6. MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    setup_style()
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n[1/5] Loading data …")
    if len(sys.argv) > 1:
        # Single file mode
        csv_path = os.path.abspath(sys.argv[1])
        if not os.path.isfile(csv_path):
            print(f"[ERROR] File not found: {csv_path}"); sys.exit(1)
        # Create temp dir with the single file
        import shutil, tempfile
        tmp = tempfile.mkdtemp()
        shutil.copy2(csv_path, tmp)
        data = load_all_csvs(tmp)
        shutil.rmtree(tmp)
    else:
        data = load_all_csvs(CSV_DIR)

    # ── Compute per-trial metrics ─────────────────────────────────────────
    print("\n[2/5] Computing metrics …")
    trials = build_trial_table(data)
    print(f"  Computed metrics for {len(trials)} trials")

    # ── Aggregate ─────────────────────────────────────────────────────────
    print("\n[3/5] Aggregating …")
    psummary   = build_participant_summary(trials)
    mcomp      = build_method_comparison(trials)
    stat_tests = run_statistical_tests(trials)

    # ── Export CSVs ───────────────────────────────────────────────────────
    print("\n[4/5] Exporting CSVs ...")
    trials.to_csv(os.path.join(OUT_DIR, "per_trial_metrics.csv"), index=False)
    print(f"    [OK] per_trial_metrics.csv ({len(trials)} rows)")
    psummary.to_csv(os.path.join(OUT_DIR, "per_participant_summary.csv"), index=False)
    print(f"    [OK] per_participant_summary.csv ({len(psummary)} rows)")
    mcomp.to_csv(os.path.join(OUT_DIR, "method_comparison.csv"), index=False)
    print(f"    [OK] method_comparison.csv")
    stat_tests.to_csv(os.path.join(OUT_DIR, "statistical_tests.csv"), index=False)
    print(f"    [OK] statistical_tests.csv")

    report = generate_report(data, trials, psummary, mcomp, stat_tests)
    report_path = os.path.join(OUT_DIR, "analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"    [OK] analysis_report.txt")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n[5/5] Generating plots ...")
    plot_wpm_by_participant(trials)
    plot_wpm_boxplot(trials)
    plot_cer_by_participant(trials)
    plot_cer_boxplot(trials)
    plot_accuracy_by_method(trials)
    plot_iki_distribution(trials)
    plot_kspc_comparison(trials)
    plot_backspace_rate(trials)
    plot_tct_vs_stimulus_length(trials)
    plot_metrics_heatmap(trials)
    plot_iei_clpi_scatter(trials)
    plot_summary_dashboard(trials, psummary)

    # ── Print report to console ───────────────────────────────────────────
    try:
        print("\n" + report)
    except UnicodeEncodeError:
        print("\n" + report.encode("ascii", errors="replace").decode("ascii"))
    print(f"\n[OK] All outputs saved to: {OUT_DIR}\n")


if __name__ == "__main__":
    main()
