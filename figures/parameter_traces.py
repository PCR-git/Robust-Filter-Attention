"""
Per-head SDE parameters of the last block over training (Appendix figures "Measurement and Process Noise
Parameters", "Robustness and inverse temperature" and "Integrative dynamics" (a); style of Test LMs.ipynb
cell 47), into images/Noise_params_m1/ and images/Noise_params_m2/:

    query_measurement_noise_<M>.png   gamma^2
    key_measurement_noise_<M>.png     eta^2
    ss_process_noise_<M>.png          sigma^2 / 2 mu
    robustness_threshold_<M>.png      nu / d
    inverse_temp_<M>.png              beta (tau)
    phase_param_M2.png                alpha = eta^2 - sigma^2 / 2 mu

    python figures/parameter_traces.py

Reads the history stored in the latest checkpoint of the M1 and M2 runs in $RFA_RUNS_DIR (train_lm.py logs
the parameters every 100 steps). Heads with mu = 0 have no steady state and are left out of the
sigma^2 / 2 mu and alpha plots.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from common import IMAGES, latest_run, load_checkpoint

LOG_EVERY = 100  # training steps between logged values (single_epoch_lm's log_every)
COLORS = ["#0072B2", "#E69F00", "#CC79A7", "#56B4E9", "#F0E442", "#000000", "#D55E00", "#999999"]

FIGURES = [  # file stem, ylabel, function of the history arrays [n_logged, n_heads], models
    ("query_measurement_noise", r"$\gamma^2$ (Query Noise)", lambda h: h["gamma"], ("M1", "M2")),
    ("key_measurement_noise", r"$\eta^2$ (Key Noise)", lambda h: h["eta"], ("M1", "M2")),
    ("ss_process_noise", r"Steady state process noise ($\sigma^2/2\mu$)", lambda h: h["sigma_tilde"], ("M1", "M2")),
    ("robustness_threshold", r"Robustness ($\nu/d$)", lambda h: h["nu_over_d"], ("M1", "M2")),
    ("inverse_temp", r"Inv. Temp ($\beta$)", lambda h: h["tau"], ("M1", "M2")),
    ("phase_param", r"Phase Parameter ($\alpha$)", lambda h: h["eta"] - h["sigma_tilde"], ("M2",)),
]


def plot_heads(data, ylabel, path):
    plt.figure(figsize=(7, 4))
    steps = np.arange(len(data)) * LOG_EVERY
    for h in range(data.shape[1]):
        if np.isfinite(data[:, h]).all():
            plt.plot(steps, data[:, h], color=COLORS[h % len(COLORS)], label=f"Head {h + 1}", linewidth=1.2, alpha=0.9)
    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    plt.grid(True)
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.labelsize": 12, "legend.fontsize": 9,
                         "grid.alpha": 0.2, "grid.linestyle": "-"})
    for model in ("M1", "M2"):
        history = {k: np.array(v, dtype=float) for k, v in load_checkpoint(latest_run(model))["history"].items()
                   if k in ("mu", "sigma_tilde", "eta", "gamma", "tau", "nu_over_d")}
        out_dir = os.path.join(IMAGES, f"Noise_params_{model.lower()}")
        os.makedirs(out_dir, exist_ok=True)
        for stem, ylabel, value, models in FIGURES:
            if model in models:
                plot_heads(value(history), ylabel, os.path.join(out_dir, f"{stem}_{model}.png"))
        print("saved to", out_dir)
