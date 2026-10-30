"""
Named model configurations from the paper, as overrides of the notebook-style `args`.

    args = make_args("M3", device=device)     # defaults + preset + any overrides
    model = build_language_model(args)

Presets starting with "B" are dot-product baselines (TransformerNetwork); the others are RFA.
"""

from argparse import Namespace

import torch.nn as nn

# Defaults shared by every model (paper setup: 6 blocks, d=256, 8 heads, trained at L=512)
BASE_ARGS = dict(
    # Architecture
    vocab_size=50257, d_e=256, d_k_total=256, d_v_total=256, n_heads=8, num_blocks=6,
    seq_len=512,                      # training context; also sets the decay scale when scale_decay_by_time_interval
    use_real_input_norm=True, use_real_output_norm=True, use_outer_residual=True,
    grad_checkpointing=False,         # recompute blocks in backward (much less memory, ~30% slower)
    io_projections=True,              # extra d->d maps around the blocks (as in the published runs)
    compute_metadata=False,           # return attention matrices etc. in eval mode (for plots)

    # RFA dynamics
    use_SC_RoPE=False, damping=0.05, zero_frac=0.25,
    use_log_linear_decay=True, scale_decay_by_time_interval=True, max_fixed_decay=5.0,
    omega_rescale=1.0, learn_rotations=False,

    # RFA noise model
    use_ss_process_noise=False, target_ss_process_var=0.1, target_measurement_var=1.0,

    # RFA attention weights and ablation switches
    use_robust_weight=True, additive_bias_type=1, multiplicative_bias_type=1,
    use_full_residual_norm=1, rotate_values=True, zero_rotations=False, lambda_real_zero=False,
    zero_process_noise=False, zero_key_measurement_noise=False,

    # Numerics
    epsilon=1e-5, min_exponent=-10, max_exponent=0,

    # Baseline attention (only used by "B" presets)
    use_rope=True, use_alibi=False, use_relative_decay_vanilla=False,
)

_SC = dict(use_SC_RoPE=True, scale_decay_by_time_interval=False, use_log_linear_decay=False)
_NO_DLE = dict(zero_process_noise=True, zero_key_measurement_noise=True, additive_bias_type=0, multiplicative_bias_type=0)

PRESETS = {
    # --- Baselines ---
    "B1":   dict(use_rope=True, use_alibi=False),                                   # RoPE
    "B2":   dict(use_rope=False, use_alibi=True),                                   # ALiBi
    "B3":   dict(use_rope=True, use_alibi=False, use_relative_decay_vanilla=True,   # RoPE + decay
                 use_SC_RoPE=False, damping=0.05),
    "B4":   dict(use_rope=False, use_alibi=False, use_relative_decay_vanilla=True,  # SC-RoPE + decay
                 use_SC_RoPE=True, damping=0.05),
    # --- RFA models ---
    "M1":   dict(use_log_linear_decay=False, damping=0.05),                         # Isotropic RFA
    "M2":   dict(_SC, damping=0.05),                                                # SC-RFA (in-window)
    "M3":   dict(_SC, damping=5.0, zero_frac=0.0, target_ss_process_var=0.5),       # SC-RFA (extrapolation)
    # --- Structural ablations of M2 ---
    "M2.1": dict(_SC, damping=0.05, use_robust_weight=False),                       # exponential weight
    "M2.2": dict(_SC, damping=0.05, **_NO_DLE),                                     # flat prior
    "M2.3": dict(_SC, damping=0.05, multiplicative_bias_type=0),                    # no multiplicative gate
    "M2.4": dict(_SC, damping=0.05, rotate_values=False),                           # no value rotations
    "M2.5": dict(_SC, damping=0.05, zero_rotations=True),                           # no rotations
    "M2.6": dict(_SC, damping=0.0, use_robust_weight=False, use_full_residual_norm=0, # pure rotation
                 lambda_real_zero=True, max_fixed_decay=0.0, **_NO_DLE),
    # Minimal SC-RFA: rotary dot-product attention plus only the additive precision bias
    # -log(sigma^2 dt + eta^2 + gamma^2) (zero decay, exponential weight, no gate, no value rotation).
    # damping=0.05 is kept so sigma^2 is initialized coupled to the spectrum (2 * 0.05 * omega_max * 0.1).
    # (Trained 2026-10-23 as run "M2.8", before the presets were renumbered to match the paper.)
    "M2.7": dict(_SC, damping=0.05, lambda_real_zero=True, use_robust_weight=False, use_full_residual_norm=0,
                 multiplicative_bias_type=0, rotate_values=False),
    # Rotary limit: M2.6 without value rotation, i.e. exactly rotary dot-product attention (conjugate-pair
    # frequencies, learned temperature), RoPE implemented inside RFA. Add --set use_SC_RoPE=False for the
    # standard (unpartitioned) RoPE spectrum in every head.
    # (Trained 2026-10-18/19 as runs "M2.7" and "M2.7_rope_spec", before the renumbering.)
    "M2.8": dict(_SC, damping=0.0, use_robust_weight=False, use_full_residual_norm=0,
                 lambda_real_zero=True, max_fixed_decay=0.0, rotate_values=False, **_NO_DLE),
}


def make_args(preset, **overrides):
    """Namespace of BASE_ARGS, updated with PRESETS[preset], then with `overrides`."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    d = dict(BASE_ARGS)
    d.update(PRESETS[preset])
    d.update(overrides)
    d["preset"] = preset
    return Namespace(**d)


def is_baseline(args):
    return args.preset.startswith("B")


def backbone_config(args):
    """Typed BackboneConfig of the model described by `args` (paper layout: LayerNorm, FFN width 4 d)."""
    from rfa.model import BackboneConfig
    from rfa.model.compat import attention_config_from_args, rfa_config_from_args, rfa_backbone_config
    if is_baseline(args):
        attention = attention_config_from_args(args, args.d_e, args.n_heads, 2 * args.d_e)
        return BackboneConfig(attention=attention, n_blocks=args.num_blocks, ffn_dim=4 * args.d_e, norm="layer",
                              io_projections=args.io_projections)
    attention = rfa_config_from_args(args, args.n_heads, args.d_e,
                                     query_key_dim_total=args.d_k_total, value_dim_total=args.d_v_total)
    return rfa_backbone_config(args, attention, args.num_blocks, 4 * args.d_v_total, nn.LayerNorm, args.io_projections)


def build_backbone(args):
    from rfa.model import Backbone
    return Backbone(backbone_config(args), grad_checkpointing=args.grad_checkpointing)


def build_language_model(args, embed_init_std=None):
    """embed_init_std=None is the published runs' N(0, 1) embedding init; 0.02 is the usual choice."""
    from rfa.model import LanguageModel
    return LanguageModel(build_backbone(args), vocab_size=args.vocab_size, embed_dim=args.d_e,
                         embed_init_std=embed_init_std)
