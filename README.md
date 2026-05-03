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
├── utils/                    # Complex-valued tensor utilities
├── dynamics/                 # Dynamical system simulators for testing
├── isotropic_rfa/            # Core RFA attention computations
├── model/                    # Neural network components
├── training/                 # Training utilities and loops
└── visualization/            # Attention and trajectory visualizations
```

Note: This repository contains the original research codebase used for all experiments and ablations in the paper. A streamlined version is planned for the final release.

Training on Wikitext-103 or BabyLM 2025 requires (a) downloading these datasets and placing them in ./datasets and (b) downloading the GPT2 Tokenizer to ./gpt2_tokenizer. The model can be easily tested using Main - Dynamics Training, which generates its own synthetic data (a simulation of a simple LTI system).


