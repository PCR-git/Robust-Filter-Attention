"""
Test configurations for the RFA regression tests.

BASE_ARGS mirrors the defaults in 'Main - Wikitext_103 Training.ipynb' (= model M1).
Each entry of CONFIGS is (args overrides, forward-call options).
"""

from argparse import Namespace

import torch

D_E = 32
N_HEADS = 4
N_BLOCKS = 2
HIDDEN = 64
BATCH = 2
SEQ = 20

BASE_ARGS = dict(
    device=torch.device("cpu"),
    d_e=D_E, d_k_total=D_E, d_v_total=D_E, n_heads=N_HEADS,
    seq_len=512,  # training length: used for decay scaling / colored-prior periods
    max_learned_decay=5.0, max_fixed_decay=5.0,
    max_exponent=0, min_exponent=-10, epsilon=1e-5,
    compute_metadata=False, compute_pulled_forward_estimates=False,
    causal=True, t_equal=True,
    sep_params=False, new_settings=False,
    lambda_real_zero=False,
    use_full_residual_norm=1, use_robust_weight=True,
    additive_bias_type=1, multiplicative_bias_type=1,
    t_shift=None, learn_t_shift=False,  # notebook used True; removed in the refactor
    learn_rotations=False, learn_decay=False,
    rotate_values=True,
    zero_process_noise=False, zero_key_measurement_noise=False,
    use_total_precision_gate=1, use_inner_residual=False, use_outer_residual=True,
    use_complex_input_norm=0, use_complex_output_norm=False,
    use_real_input_norm=True, use_real_output_norm=True,
    add_gaussian_noise=False,
    use_complex_conj_constraint=True,
    use_colored_prior=False,
    scale_decay_by_time_interval=True,
    zero_rotations=False,
    use_ss_process_noise=False,
    damping=0.05, omega_rescale=1.0,
    use_SC_RoPE=False, use_log_linear_decay=True, zero_frac=0.25,
    target_ss_process_var=0.1, target_measurement_var=1.0, sigma_sq_v_floor=0,
)

M2 = dict(use_SC_RoPE=True, scale_decay_by_time_interval=False, damping=0.05, use_log_linear_decay=False)
M3 = dict(use_SC_RoPE=True, scale_decay_by_time_interval=False, damping=5.0, use_log_linear_decay=False,
          zero_frac=0.0, use_ss_process_noise=False, target_ss_process_var=0.5)


def with_(base, **kw):
    d = dict(base)
    d.update(kw)
    return d


CONFIGS = {
    # --- Paper models ---
    "M1":   ({}, {}),
    "M2":   (M2, {}),
    "M3":   (M3, {}),
    # --- Paper ablations (relative to M2) ---
    "M2.1_exp_weight":   (with_(M2, use_robust_weight=False), {}),
    "M2.2_flat_prior":   (with_(M2, zero_process_noise=True, zero_key_measurement_noise=True,
                                additive_bias_type=0, multiplicative_bias_type=0), {}),
    "M2.3_no_mult_gate": (with_(M2, multiplicative_bias_type=0), {}),
    "M2.4_no_value_rot": (with_(M2, rotate_values=False), {}),
    "M2.5_no_rotations": (with_(M2, zero_rotations=True), {}),
    "M2.6_pure_rotation": (with_(M2, damping=0.0, use_robust_weight=False, use_full_residual_norm=0,
                                 lambda_real_zero=True, zero_process_noise=True, zero_key_measurement_noise=True,
                                 additive_bias_type=0, multiplicative_bias_type=0, max_fixed_decay=0.0), {}),
    # --- Other supported options ---
    "M2_noncausal":      (M2, {"causal": False}),
    "M2_irregular_time": (with_(M2, t_equal=False), {"irregular_time": True}),
    "M2_ss_process":     (with_(M2, use_ss_process_noise=True), {}),
    "M3_ss_process":     (with_(M3, use_ss_process_noise=True), {}),
    "M3_omega_half":     (with_(M3, omega_rescale=0.5), {}),
    "M2_learn_rot":      (with_(M2, learn_rotations=True), {}),
    "M2_cnorm_qkv":      (with_(M2, use_complex_input_norm=1, use_complex_output_norm=True), {}),
    "M1_no_real_norms":  ({"use_real_input_norm": False, "use_real_output_norm": False}, {}),
}


def make_args(overrides):
    d = dict(BASE_ARGS)
    d.update(overrides)
    return Namespace(**d)
