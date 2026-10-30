"""
Perplexity figures of the paper (style of Test LMs.ipynb cells 16, 71 and 73), into images/other_imgs/:

    validation_perplexity.png   validation PPL per epoch (5-15) of B1, B2, M1, M2, M3, from the runs' metrics.txt
    extrapolation.png           test PPL at L = 512 ... 4096 (Table 1), log y-axis
    damping_b_sweep.png         test PPL of SC-RFA (M2) vs. the damping coefficient b (Table 2)

    python figures/curves.py [out_dir]

Reads the WikiText-103 runs in $RFA_RUNS_DIR (default: <repo>/runs). The two table figures plot the
paper's numbers, which eval_extrapolation.py reproduces from the trained runs.
"""
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker

from common import IMAGES, STYLE, latest_run, style_axes

LENGTHS = [512, 1024, 2048, 4096]

# Table 1 (published numbers for the unchanged models incl. B2, rerun numbers for B1).
EXTRAPOLATION = {
    "RoPE (B1)":   [27.93, 40.55, 100.63, 202.36],
    "ALiBi (B2)":  [28.59, 27.30, 26.54, 26.30],
    "RFA (M1)":    [28.01, 27.58, 29.99, 38.46],
    "SC-RFA (M2)": [27.54, 26.73, 29.46, 37.19],
    "SC-RFA (M3)": [27.91, 26.68, 26.37, 28.16],
}

# Table 2: SC-RFA (M2) with damping coefficient b (b = 0.05 is M2), test PPL at each L.
DAMPING = {
    5e-4: [27.60, 28.88, 37.34, 51.48],
    5e-3: [27.60, 28.71, 35.35, 43.90],
    5e-2: [27.54, 26.73, 29.46, 37.19],
    5e-1: [27.61, 26.38, 26.37, 29.72],
    5e0:  [27.91, 26.68, 26.37, 28.16],
}


def val_history(run):
    rows = []
    for line in open(os.path.join(latest_run(run), "metrics.txt")):
        parts = line.split()
        if parts and parts[0].isdigit():
            rows.append(float(parts[3]))
    return rows


def plot_validation(models, path, start_epoch=5):
    plt.figure(figsize=(10, 6))
    hist = {label: val_history(run) for label, run in models.items()}
    for label in sorted(hist, key=lambda k: hist[k][-1], reverse=True):
        y = hist[label]
        color, ls, m = STYLE[label]
        plt.plot(range(start_epoch, len(y) + 1), y[start_epoch - 1:], label=f"{label} ({y[-1]:.2f})",
                 color=color, linestyle=ls, marker=m, linewidth=2.0, markersize=6, markeredgewidth=1.5)
    plt.xticks(fontsize=14)
    style_axes("Epoch", "Validation Perplexity", "upper right")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_extrapolation(data, path):
    plt.figure(figsize=(10, 6))
    for label in sorted(data, key=lambda k: data[k][-1], reverse=True):
        color, ls, m = STYLE[label]
        plt.plot(LENGTHS, data[label], label=f"{label} ({data[label][-1]:.2f})", color=color, linestyle=ls,
                 marker=m, linewidth=2.5, markersize=10, markeredgecolor="white", markeredgewidth=1.0)
    plt.yscale("log")  # keeps the RFA curves readable next to RoPE's 202
    ticks = [25, 30, 40, 60, 100, 150, 200]
    plt.gca().yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    plt.yticks(ticks, [str(t) for t in ticks])
    plt.xticks(LENGTHS, fontsize=14)
    style_axes("Context Length (Tokens)", "Test Perplexity", "upper left")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_damping(data, path):
    b_values = list(data)
    styles = [("#0072B2", "-", "o"), ("#E69F00", "--", "s"), ("#CC79A7", "-.", "^"), ("#009E73", ":", "D")]
    plt.figure(figsize=(10, 6))
    for k, (L, (color, ls, m)) in enumerate(zip(LENGTHS, styles)):
        plt.plot(b_values, [data[b][k] for b in b_values], label=f"$L = {L}$", color=color, linestyle=ls,
                 marker=m, linewidth=3.0, markersize=10, markeredgecolor="white", markeredgewidth=1.5)
    plt.xscale("log")
    plt.gca().xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    plt.xticks(b_values, [rf"$5 \times 10^{{{e}}}$" for e in (-4, -3, -2, -1, 0)], fontsize=14)
    plt.ylim(25, 55)
    style_axes("Damping Coefficient $b$", "Test Perplexity", "upper right", legend_title="Context Length")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(IMAGES, "other_imgs")
    os.makedirs(out_dir, exist_ok=True)
    # Validation curves from the reruns (the published per-epoch histories are not on disk).
    plot_validation({"ALiBi (B2)": "B2", "RoPE (B1)": "B1", "RFA (M1)": "M1", "SC-RFA (M2)": "M2",
                     "SC-RFA (M3)": "M3"}, os.path.join(out_dir, "validation_perplexity.png"))
    plot_extrapolation(EXTRAPOLATION, os.path.join(out_dir, "extrapolation.png"))
    plot_damping(DAMPING, os.path.join(out_dir, "damping_b_sweep.png"))
    print("saved to", out_dir)
