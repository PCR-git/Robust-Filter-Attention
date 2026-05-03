# Adaptive Filter Attention (AFA)

Adaptive Filter Attention (AFA) is a novel attention mechanism designed for filtering and forecasting in stochastic dynamical systems. Unlike standard attention mechanisms,
AFA integrates an explicit, learnable linear time-invariant (LTI) dynamics model into the attention computation, allowing it to more robustly filter noise and infer latent state transitions.

---

## Key Idea

AFA uses a learned LTI dynamics model to propagate latent states before computing similarity. Instead of comparing a query with keys directly, AFA compares a
predicted latent state (from the key, evolved to the query’s time point) to the query. The attention weights are then computed as a function of 
how consistent the transition is under the learned dynamics, effectively filtering out observations that do not align with plausible state evolution.

This framework introduces:
- Treating inputs as latent states within a learned autonomous dynamical system rather than fixed exogenous signals.
- Propagating keys forward through system dynamics before computing similarity, aligning attention with temporal evolution.
- Efficient linear dynamics computation via matrix exponentials and diagonalization.
- Precision-weighted maximum likelihood estimation combining evolved past measurements with analytically computed covariances.
- Adaptive reweighting of precision matrices using residual-based Mahalanobis distances to correct for model error.
- Replacing softmax with a robust, uncertainty-aware weighting scheme akin to an M-estimator.
- Joint latent state estimation and prediction ensuring attention weights reflect confidence-consistent similarities.

## Repository Structure

```
adaptive_filter_attention/
│
├── utils/                    # Complex-valued tensor utilities
│
├── dynamics/                 # Tools for simulating dynamical systems for testing
│
├── precision_attention/      # Core of AFA computations, including residuals, estimates, and propagated precision matrices
|                               estimates, and propagated precision matrices
│
├── model/                    # Neural network components
│
├── training/                 # Training utilities and loops
│
└── visualization/            # Tools to visualize attention and predictions
```

Also includes: Animations of predicted trajectories and attention matrices during training, for a simple 2D ODE
