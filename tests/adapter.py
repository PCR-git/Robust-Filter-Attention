"""
Adapter between the golden (pre-refactor) format and the current code.

As the refactor changes constructor signatures and parameter names, only this
file should need to change; check_regression.py stays fixed.
"""

from tests.configs import make_args, D_E, N_HEADS, N_BLOCKS, HIDDEN

# Configs whose behavior was changed on purpose; the checker reports them instead of comparing.
INTENTIONAL_CHANGES = {
    "M2.4_no_value_rot": "rotate_values=False now skips both value rotations (original skipped only the "
                         "counter-rotation); new behavior is checked in test_smoke.py",
    # The original RoPE/SCRoPE paired feature 2k+1 with 2k+3 instead of 2k (not a rotation).
    # Verified that patching the old rotation back in reproduces these goldens exactly.
    "B1_rope": "RoPE fixed to a true pairwise rotation (checked in test_smoke.py)",
    "B3_rope_decay": "RoPE fixed to a true pairwise rotation (checked in test_smoke.py)",
    "B4_sc_rope": "SC-RoPE fixed to a true pairwise rotation (checked in test_smoke.py)",
}


def _config(overrides):
    from rfa import RFAConfig
    return RFAConfig.from_args(make_args(overrides), d_model=D_E, n_heads=N_HEADS, head_dim=D_E // N_HEADS)


def build_network(overrides):
    from model import RFATransformerNetwork
    args = make_args(overrides)
    return RFATransformerNetwork(
        args, num_blocks=N_BLOCKS, n_heads=N_HEADS, input_dim=D_E,
        query_key_dim_total=args.d_k_total, value_dim_total=args.d_v_total, hidden_dim=HIDDEN,
    )


def _unused_norm(key, overrides):
    """The original blocks created both real norms even when use_real_input/output_norm turned them off."""
    args = make_args(overrides)
    return (".norm1." in key and not args.use_real_input_norm) or (".norm2." in key and not args.use_real_output_norm)


def convert_state(legacy_state, overrides):
    from rfa import convert_legacy_state_dict
    new = convert_legacy_state_dict(legacy_state, _config(overrides))
    return {k: v for k, v in new.items() if not _unused_norm(k, overrides)}


def legacy_to_new_key(legacy_key, overrides):
    """Map a legacy state_dict key to the current one (None = dropped)."""
    from rfa.legacy import convert_legacy_key
    if _unused_norm(legacy_key, overrides):
        return None
    return convert_legacy_key(legacy_key, _config(overrides))


def set_compute_metadata(model, flag):
    model.args.compute_metadata = flag


def legacy_meta_key(new_key, overrides, call):
    """Legacy metadata key to compare against, or None to skip (intentional changes)."""
    config = _config(overrides)
    # Legacy reported softplus() of whichever process-noise parameter was unused; the derived
    # quantity is now reported instead, so only the one that is actually learned is compared.
    if new_key == "sigma_tilde_sq_v" and config.process_noise == "variance":
        return None
    if new_key == "sigma_sq_v" and config.process_noise == "steady_state":
        return None
    # Legacy masked A_prior with args.causal instead of the forward `causal` argument.
    if new_key == "A_prior" and not call.get("causal", True):
        return None
    return new_key
