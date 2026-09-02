

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe in subprocesses
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Custom "attention" colormap  (transparent → cyan → yellow → red)
# ---------------------------------------------------------------------------
_CMAP_COLORS = [
    (0.00, (0.00, 0.00, 0.00, 0.00)),   # fully transparent
    (0.20, (0.00, 0.60, 1.00, 0.40)),   # translucent cyan
    (0.50, (0.00, 1.00, 0.60, 0.70)),   # teal-green
    (0.75, (1.00, 0.85, 0.00, 0.88)),   # amber
    (1.00, (1.00, 0.10, 0.00, 1.00)),   # opaque red
]

def _make_cmap():
    r = [(p, c[0], c[0]) for p, c in _CMAP_COLORS]
    g = [(p, c[1], c[1]) for p, c in _CMAP_COLORS]
    b = [(p, c[2], c[2]) for p, c in _CMAP_COLORS]
    a = [(p, c[3], c[3]) for p, c in _CMAP_COLORS]
    return LinearSegmentedColormap("gaze_heat", {"red": r, "green": g,
                                                  "blue": b, "alpha": a})

GAZE_CMAP = _make_cmap()


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------
def generate_heatmap(
    gaze_points: list,          # list of (norm_x, norm_y) tuples, values 0-1
    output_path: str,           # full path to save PNG
    screen_w: int = 1920,
    screen_h: int = 1080,
    session_label: str = "",    # e.g. file name shown in title
    sigma_fraction: float = 0.03,   # gaussian blur relative to screen size
    grid_res: int = 400,        # heatmap grid resolution
):
    """
    Render a full-resolution gaze heatmap and save it as a PNG.

    Parameters
    ----------
    gaze_points   : list of (x, y) where x,y in [0, 1]
    output_path   : destination file path for the PNG
    screen_w/h    : physical screen resolution (for aspect ratio)
    session_label : a descriptive title string
    sigma_fraction: controls Gaussian blur width as fraction of grid size
    grid_res      : internal grid resolution (pixels)
    """
    if not gaze_points:
        print("[HeatMap] No gaze data - skipping heatmap generation.")
        return

    xs = np.array([p[0] for p in gaze_points], dtype=float)
    ys = np.array([p[1] for p in gaze_points], dtype=float)

    # Clamp to valid range
    xs = np.clip(xs, 0.0, 1.0)
    ys = np.clip(ys, 0.0, 1.0)

    # -- Build density grid ---------------------------------------------------
    aspect = screen_w / screen_h
    grid_w = grid_res
    grid_h = int(grid_res / aspect)

    density = np.zeros((grid_h, grid_w), dtype=float)
    xi = np.clip((xs * (grid_w - 1)).astype(int), 0, grid_w - 1)
    yi = np.clip((ys * (grid_h - 1)).astype(int), 0, grid_h - 1)
    np.add.at(density, (yi, xi), 1)

    # Gaussian smoothing
    sigma = sigma_fraction * grid_res
    density = gaussian_filter(density, sigma=sigma)

    # Normalise
    dmax = density.max()
    if dmax > 0:
        density /= dmax

    # -- Layout ---------------------------------------------------------------
    fig_w = 16
    fig_h = fig_w / aspect + 2.0   # extra height for title + stats bar

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#0a0a14")
    ax  = fig.add_axes([0.0, 0.08, 1.0, 0.86])   # leave room at top & bottom

    # -- Background grid (screen regions) -------------------------------------
    ax.set_facecolor("#0a0a14")
    for gx in np.arange(0, 1.01, 1/3):
        ax.axvline(gx, color="#1e1e36", linewidth=0.8, zorder=1)
    for gy in np.arange(0, 1.01, 1/3):
        ax.axhline(gy, color="#1e1e36", linewidth=0.8, zorder=1)

    # Region labels
    regions = [
        (1/6, 1/6, "TOP-LEFT"),   (0.5, 1/6, "TOP-CENTER"),   (5/6, 1/6, "TOP-RIGHT"),
        (1/6, 0.5, "MID-LEFT"),   (0.5, 0.5, "CENTER"),        (5/6, 0.5, "MID-RIGHT"),
        (1/6, 5/6, "BTM-LEFT"),   (0.5, 5/6, "BOTTOM-CENTER"), (5/6, 5/6, "BTM-RIGHT"),
    ]
    for rx, ry, label in regions:
        ax.text(rx, ry, label, color="#2a2a4a", fontsize=7.5,
                ha="center", va="center", fontfamily="monospace",
                transform=ax.transData, zorder=2)

    # -- Heatmap --------------------------------------------------------------
    ax.imshow(density, extent=[0, 1, 1, 0],   # note: y-axis inverted (top=0)
              cmap=GAZE_CMAP, aspect="auto",
              interpolation="bilinear", vmin=0, vmax=1, zorder=3)

    # -- Gaze scatter (raw dots) ----------------------------------------------
    if len(xs) <= 5000:          # skip scatter for very long sessions (perf)
        ax.scatter(xs, ys, c="white", s=0.8, alpha=0.08, linewidths=0, zorder=4)

    # -- Hotspot markers (top 5 high-density cells) ---------------------------
    flat_idx = np.argsort(density.ravel())[::-1][:5]
    for rank, fidx in enumerate(flat_idx):
        hy, hx = np.unravel_index(fidx, density.shape)
        nx = hx / grid_w
        ny = hy / grid_h
        circle = plt.Circle((nx, ny), 0.02, color="white",
                             fill=False, linewidth=1.5, alpha=0.7, zorder=5)
        ax.add_patch(circle)
        ax.text(nx, ny - 0.025, f"#{rank+1}", color="white",
                fontsize=7, ha="center", va="bottom", zorder=6,
                path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    # -- Colorbar -------------------------------------------------------------
    cb_ax = fig.add_axes([0.02, 0.015, 0.40, 0.025])
    sm    = plt.cm.ScalarMappable(cmap=GAZE_CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb    = fig.colorbar(sm, cax=cb_ax, orientation="horizontal")
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_ticklabels(["Low", "Medium", "High"], color="#aaaaaa", fontsize=8)
    cb.outline.set_edgecolor("#333355")

    # -- Stats text -----------------------------------------------------------
    n_pts  = len(gaze_points)
    cx_pct = float(xs.mean()) * 100
    cy_pct = float(ys.mean()) * 100
    spread = float(xs.std() + ys.std())

    stats_str = (
        f"Total gaze points: {n_pts:,}   |   "
        f"Mean gaze: ({cx_pct:.1f}%, {cy_pct:.1f}%)   |   "
        f"Spread (std): {spread:.3f}"
    )
    fig.text(0.55, 0.025, stats_str, color="#888888", fontsize=8,
             ha="center", va="bottom", fontfamily="monospace")

    # -- Title ----------------------------------------------------------------
    base = os.path.basename(session_label) if session_label else "Typing Session"
    fig.text(0.5, 0.97, "Gaze Heatmap  -  Typing Session",
             ha="center", va="top", fontsize=14, color="#e0e0f0",
             fontweight="bold")
    fig.text(0.5, 0.955, base,
             ha="center", va="top", fontsize=9, color="#7070b0",
             fontfamily="monospace")

    # -- Axes cosmetics -------------------------------------------------------
    ax.set_xlim(0, 1); ax.set_ylim(1, 0)   # origin top-left
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#222244")

    # -- Save -----------------------------------------------------------------
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0a0a14")
    plt.close(fig)
    print(f"[HeatMap] Saved -> {output_path}")


# ---------------------------------------------------------------------------
# Convenience: generate from a CSV file (standalone use)
# ---------------------------------------------------------------------------
def generate_from_csv(csv_path: str, typed_file: str = "",
                      screen_w: int = 1920, screen_h: int = 1080):
    """
    Load gaze_x / gaze_y from a CSV log and generate a heatmap.
    Output path is derived from `typed_file` (or csv_path if empty).
    """
    import pandas as pd

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[HeatMap] Cannot read CSV {csv_path}: {e}")
        return

    if "gaze_x" not in df.columns or "gaze_y" not in df.columns:
        print("[HeatMap] CSV missing gaze_x / gaze_y columns.")
        return

    pts = list(zip(df["gaze_x"].tolist(), df["gaze_y"].tolist()))

    if typed_file:
        base   = os.path.splitext(typed_file)[0]
        outf   = base + "_heatmap.png"
        label  = typed_file
    else:
        base   = os.path.splitext(csv_path)[0]
        outf   = base + "_heatmap.png"
        label  = csv_path

    generate_heatmap(pts, outf,
                     screen_w=screen_w, screen_h=screen_h,
                     session_label=label)


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        generate_from_csv(sys.argv[1],
                          typed_file=sys.argv[2] if len(sys.argv) >= 3 else "")
    else:
        print("Usage: python heatmap_generator.py <gaze_log.csv> [typed_text.txt]")
