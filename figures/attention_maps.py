"""
Last-layer attention maps at L = 4096 (Appendix, "Analysis of Attention Matrices"; Test LMs.ipynb cells 77-78),
into images/attn_mats/attn_mats_<name>_<start>_to_<end>[_power].png:

    python figures/attention_maps.py                     # B1, B2, M1, M2: full range, linear scale (the paper's figures)
    python figures/attention_maps.py M2 --all-variants   # also PowerNorm(0.2) and the 3000-4096 zoom

Input: the first 4096-token block of the WikiText-103 test split. Each panel is one head; the color scale is
clipped at the 99.5th percentile (linear) or uses PowerNorm(gamma=0.2). For RFA models the map is the
softmax attention before the decay exp(-mu * lag) is applied, and the head titles give mu * L and the
surviving weight exp(-mu * L) at the right edge of the view. Needs a GPU, the trained runs in
$RFA_RUNS_DIR, and the data in ./datasets.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import PowerNorm

from common import IMAGES, REPO, latest_run, load_model
from eval_extrapolation import test_blocks
from training.lm_data import load_tokenizer

FILE_NAMES = {"M2": "M2_b0.05"}  # names used by the paper's \includegraphics
L = 4096


@torch.no_grad()
def last_layer_attention(model, input_ids):
    """[H, L, L] softmax weights of the last block, and the per-head decay rates mu (None for baselines)."""
    _, infos = model(input_ids, causal=True, return_metadata=True)
    info = infos[-1]
    A = info["attn_mat"][0]
    if "mu_v" in info:
        return A.permute(2, 0, 1).float().cpu().numpy(), info["mu_v"].float().cpu().numpy()  # RFA: [N, N, H]
    return A.float().cpu().numpy(), None  # baseline: [H, N, N]


def plot(A, mu, view_range, use_powernorm, path):
    start, end = view_range
    fig, axes = plt.subplots(2, 4, figsize=(20, 10) if mu is None else (22, 12), layout="constrained")
    for i, ax in enumerate(axes.flatten()):
        if use_powernorm:
            ax.imshow(A[i], cmap="magma", interpolation="antialiased", interpolation_stage="data",
                      norm=PowerNorm(gamma=0.2))
        else:
            ax.imshow(A[i], cmap="magma", interpolation="antialiased", interpolation_stage="data",
                      vmin=0, vmax=np.percentile(A[i], 99.5))
        ax.set_xlim(start, end)
        ax.set_ylim(end, start)
        if mu is None:
            ax.set_title(f"Head {i + 1}", fontsize=28)
        else:
            mu_L = mu[i] * end
            ax.set_title(f"Head {i + 1}\nμL: {mu_L:.2f}\nSurv: {np.exp(-mu_L) * 100:.1f}%", fontsize=22)
        ax.axis("off")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="*", default=["B1", "B2", "M1", "M2"], help="run names in $RFA_RUNS_DIR/wikitext103")
    p.add_argument("--all-variants", action="store_true", help="also PowerNorm and the 3000-4096 zoom")
    p.add_argument("--out-dir", default=os.path.join(IMAGES, "attn_mats"))
    p.add_argument("--data-dir", default=os.path.join(REPO, "datasets"))
    cli = p.parse_args()

    device = torch.device("cuda")
    tokenizer = load_tokenizer(os.path.join(REPO, "gpt2_tokenizer"))
    input_ids = test_blocks("wikitext103", tokenizer, L, cli.data_dir, 4)[0][None].to(device)
    variants = [(False, (0, L))]
    if cli.all_variants:
        variants += [(True, (0, L)), (False, (3000, L)), (True, (3000, L))]
    os.makedirs(cli.out_dir, exist_ok=True)

    for name in cli.models:
        model, _ = load_model(latest_run(name), device)
        A, mu = last_layer_attention(model, input_ids)
        del model
        torch.cuda.empty_cache()
        for use_powernorm, (start, end) in variants:
            file = f"attn_mats_{FILE_NAMES.get(name, name)}_{start}_to_{end}{'_power' if use_powernorm else ''}.png"
            plot(A, mu, (start, end), use_powernorm, os.path.join(cli.out_dir, file))
            print("saved", file)


if __name__ == "__main__":
    main()
