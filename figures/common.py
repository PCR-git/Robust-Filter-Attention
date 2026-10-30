"""Shared helpers for the paper-figure scripts: run lookup, model loading and plot styling."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from eval_extrapolation import latest_checkpoint  # noqa: E402
from rfa.training.presets import build_language_model  # noqa: E402

RUNS = os.environ.get("RFA_RUNS_DIR", os.path.join(REPO, "runs"))
IMAGES = os.path.join(REPO, "images")

STYLE = {  # same colors/markers per model as the published figures
    "ALiBi (B2)":       ("#0072B2", "--", "^"),
    "RoPE (B1)":        ("#E69F00", "-", "o"),
    "RFA (M1)":         ("#CC79A7", "-.", "s"),
    "SC-RFA (M3)":      ("#F0E442", "-", "o"),
    "SC-RFA (M2)":      ("#56B4E9", ":", "D"),
    "RoPE + decay (B3)": ("#D55E00", "--", "v"),
    "SC-RoPE + decay (B4)": ("#009E73", "-.", "p"),
}


def latest_run(name, dataset="wikitext103"):
    """Most recent run directory <RUNS>/<dataset>/<name>__<date>."""
    root = os.path.join(RUNS, dataset)
    dirs = sorted(d for d in os.listdir(root) if d.startswith(name + "__"))
    if not dirs:
        raise FileNotFoundError(f"no run {name}__* in {root} (set RFA_RUNS_DIR to the training output folder)")
    return os.path.join(root, dirs[-1])


def load_checkpoint(run_dir):
    return torch.load(latest_checkpoint(run_dir), map_location="cpu", weights_only=False)


def load_model(run_dir, device):
    """Model of the run's latest checkpoint, rebuilt from the args stored in it, in eval mode."""
    ckpt = load_checkpoint(run_dir)
    args = ckpt["args"]
    args.device = device
    model = build_language_model(args).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval(), args


def style_axes(xlabel, ylabel, legend_loc, legend_title=None):
    plt.xlabel(xlabel, fontsize=16, labelpad=10)
    plt.ylabel(ylabel, fontsize=16, labelpad=10)
    plt.yticks(fontsize=14)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = plt.legend(loc=legend_loc, fontsize=13, frameon=True, framealpha=0.9, edgecolor="lightgray",
                        labelspacing=0.5, title=legend_title)
    if legend_title:
        legend.get_title().set_fontsize(14)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
