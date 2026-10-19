"""Load checkpoints saved by the pre-refactor MultiheadIsotropicRFA."""

from .config import RFAConfig

# Legacy buffers/parameters that no longer exist (recomputed on the fly, or unused features).
_DROPPED = ("t_measure", "abs_idx", "causal_mask", "P_prior_log", "P_scale_log")

# Parameters of removed features: a checkpoint containing them cannot be represented exactly.
_UNSUPPORTED = ("t_shift_param", "mu_k", "omega_k", "sigma_k", "eta_k", "gamma_k",
                "c_exp_v", "beta_exp_v", "a_per_v", "b_per_v", "omega_per_v")

_RENAMED = {
    "omega_v": "omega", "mu_v": "mu",
    "eta_v": "eta_raw", "gamma_v": "gamma_raw",
    "tau_param": "tau_raw", "nu_param": "nu_raw",
}


def convert_legacy_key(key, config: RFAConfig):
    """New name for a legacy state_dict key, or None if the entry was dropped."""
    prefix, _, name = key.rpartition(".")
    prefix = prefix + "." if prefix else ""
    if name in _UNSUPPORTED:
        raise ValueError(f"Checkpoint entry '{key}' belongs to a feature removed in the refactor "
                         f"(learned t_shift, separate key params, or colored prior) and cannot be loaded.")
    if name in _DROPPED:
        return None
    if name in ("sigma_v", "sigma_tilde_v"):
        used = "sigma_v" if config.process_noise == "variance" else "sigma_tilde_v"
        return prefix + "sigma_raw" if name == used else None
    return prefix + _RENAMED.get(name, name)


def convert_legacy_state_dict(state_dict, config: RFAConfig):
    """
    Convert a legacy state_dict (of a whole model or a single layer) to the current parameter names.
    Assumes every RFA layer in the model shares `config` (as in the training notebooks).
    """
    new = {}
    for key, value in state_dict.items():
        new_key = convert_legacy_key(key, config)
        if new_key is None:
            continue
        if key.rpartition(".")[2] == "mu_v":
            value = value * config.decay_scale  # decay scaling is now folded into the buffer
        new[new_key] = value
    return new


def load_legacy_state_dict(model, state_dict, strict=True):
    """Load a pre-refactor checkpoint into `model` (any module containing MultiheadIsotropicRFA layers)."""
    from .attention import MultiheadIsotropicRFA
    configs = {m.config for m in model.modules() if isinstance(m, MultiheadIsotropicRFA)}
    if len(configs) != 1:
        raise ValueError(f"Expected all RFA layers to share one config, found {len(configs)}")
    return model.load_state_dict(convert_legacy_state_dict(state_dict, configs.pop()), strict=strict)
