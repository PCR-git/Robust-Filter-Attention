# Robust Filter Attention (RFA)

Robust Filter Attention (RFA) is an attention mechanism for sequence modeling under stochastic dynamics. It integrates a learnable linear time-invariant (LTI) state-space model directly into attention, enabling noise-aware aggregation and principled temporal propagation of information.

Rather than comparing queries and keys directly, RFA evolves past representations forward under learned dynamics and
computes attention weights based on residual consistency in Mahalanobis geometry, yielding robust, uncertainty-aware attention.

---

# Key Features

- Learned LTI dynamics embedded in attention.
- State propagation via matrix exponentials and eigenbasis diagonalization.
- Precision-weighted aggregation using analytically propagated covariances.
- Residual-based reweighting for robustness to model mismatch and outliers.
- Iterative refinement of latent state estimates via stacked layers.

## Repository Structure

```
robust_filter_attention/
├── rfa/                      # Core RFA mechanism (self-contained)
│   ├── config.py             #   RFAConfig: all options of an RFA layer
│   ├── attention.py          #   MultiheadIsotropicRFA (Algorithm 1 of the paper)
│   ├── kernels.py            #   Pure functions: rotations, DLE covariance, residual norm, logits
│   ├── init.py               #   Frequency / decay / complex projection initialization
│   ├── norm.py               #   Complex RMS norm
│   └── legacy.py             #   Loading checkpoints saved by the pre-refactor code
├── model/                    # Transformer blocks, networks, baselines (RoPE, ALiBi, ...)
├── training/                 # Training loops, LM datasets, named model presets (presets.py)
├── train_lm.py               # Language-model training script
├── visualization/            # Attention and trajectory visualizations
├── dynamics/                 # Dynamical system simulators for testing
├── utils/                    # Misc utilities
└── tests/                    # Regression and smoke tests
```

Note: This repository contains the research codebase used for the experiments and ablations in the paper.

## Using the RFA layer

```python
from rfa import RFAConfig, MultiheadIsotropicRFA

config = RFAConfig(d_model=256, n_heads=8, head_dim=32,
                   frequency_init="spectral", decay_init="spectral", damping=0.05)  # SC-RFA (M2)
layer = MultiheadIsotropicRFA(config)
out, info = layer(x)  # x: [B, N, 256]; add `out` to the residual stream
```

The notebooks build layers from their `args` namespace through `RFAConfig.from_args`
(see `model/blocks.py`). Checkpoints saved before the refactor load with
`model.load_legacy_state_dict(model, state_dict)`.

## Training language models

```
python train_lm.py --dataset wikitext103 --preset M3 --compile
python train_lm.py --dataset babylm --preset B1 --compile
python train_lm.py --dataset wikitext103 --preset M2 --set damping=0.5   # override any args field
```

Each run writes to `<out-dir>/<dataset>/<name>__<date>/` (`--out-dir` defaults to
`C:/rfa_runs` on Windows, outside the synced source folder; `--name` to `<backbone>__<preset>`):
model weights of the last two epochs (`lm_epoch_N.pt`), `resume.pt` with the optimizer state
while the run is unfinished (`--resume .../resume.pt`), progress plots, and `metrics.txt`
(per-epoch train/validation loss, validation perplexity, epoch time; test perplexity at the end).
`run_queue.ps1` runs a list of trainings one after another.

Presets (`training/presets.py`) cover the paper's models: baselines B1 (RoPE), B2 (ALiBi),
B3 (RoPE + decay), B4 (SC-RoPE + decay); RFA models M1, M2, M3; and the ablations M2.1-M2.8:
M2.7 (minimal SC-RFA: SC rotary attention plus only the additive precision bias, i.e. a learned
logarithmic distance bias) and M2.8 (M2.6 without value rotation, which is rotary dot-product
attention implemented inside RFA).

Speed: `--compile` applies `torch.compile` to the backbone (on Windows this needs
`triton-windows`, see requirements.txt). At batch 16 it makes RFA 1.8x faster and uses
2.5 GB less memory, since the pairwise element-wise operations get fused; RoPE gains 1.2x.

Memory: `LanguageModel.loss()` computes the next-token loss in chunks, so the full
[batch, length, vocab] logits are never stored (about 3.4 GB saved at batch 16).
`--grad-checkpointing` additionally recomputes each block during the backward pass
(peak memory at batch 16 drops from about 7 GB to under 2 GB, at about 30% more time).

Defaults match the published runs (N(0, 1) token embedding, d->d projections around the
blocks), except that the baseline RoPE/SC-RoPE (B1, B3, B4) is now a correct rotation (the
original paired the wrong features) and M2.4 removes both value rotations. For new work,
`--small-embed-init` (std 0.02) and `--set io_projections=False` give the standard setup.

## Tests

```
python tests/check_regression.py   # compares against golden outputs of the original implementation
python tests/test_smoke.py         # behavior not covered by the golden files
```

The files in `tests/golden/` were generated by `tests/make_golden.py` and
`tests/make_golden_baselines.py` from the original (pre-refactor) code, for the paper
models, ablations and baselines. `check_regression.py` checks that initialization,
outputs, all gradients and diagnostic outputs are unchanged, except for the intentional
changes listed in `tests/adapter.py` (M2.4, and B1/B3/B4 because of the RoPE fix).

## Data

Training on Wikitext-103 or BabyLM 2025 requires (a) downloading these datasets and placing them in ./datasets and (b) downloading the GPT2 Tokenizer to ./gpt2_tokenizer. The model can be easily tested using Main - Dynamics Training, which generates its own synthetic data (a simulation of a simple LTI system).
