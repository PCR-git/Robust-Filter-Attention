from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RFAConfig:
    """
    Configuration of a multi-head Isotropic RFA layer.

    Shapes: d_model is the real input/output width. Each head works in a complex
    space of dimension head_dim (stored as head_dim real/imaginary pairs), i.e. the
    projections map d_model -> 2 * n_heads * head_dim.
    """

    d_model: int
    n_heads: int
    head_dim: int

    # --- Dynamics: eigenvalues lambda_k = -mu + i*omega_k (per head) ---
    frequency_init: Literal["rope", "spectral"] = "rope"  # "spectral" = SC-RFA: spectrum partitioned across heads
    decay_init: Literal["spectral", "log_linear"] = "spectral"  # "spectral": mu_h = damping * omega_h,max
    damping: float = 0.05            # SC-RFA damping coefficient b
    zero_decay_frac: float = 0.25    # Fraction of (lowest-frequency) heads with mu = 0
    decay_scale: float = 1.0         # Multiplies the initial decay rates
    omega_rescale: float = 1.0       # Multiplies all rotation frequencies
    learn_rotations: bool = False

    # --- Noise model ---
    # "variance":     learn the process noise variance sigma^2 (paper experiments)
    # "steady_state": learn the steady-state variance sigma^2 / (2 mu) (better conditioned)
    process_noise: Literal["variance", "steady_state"] = "variance"
    init_steady_state_var: float = 0.1
    init_measurement_var: float = 1.0
    init_nu_over_d: float = 4.0

    # --- Attention weights ---
    robust_weight: bool = True       # Student-t (power-law) weight; False = exponential weight

    # --- Ablations ---
    dle_bias: bool = True            # Additive log-precision bias log(P_dt); False = no bias
    dle_gate: bool = True            # Multiplicative precision in the Mahalanobis distance; False = 1/sqrt(d)
    full_residual_norm: bool = True  # False = use only the cross term (dot-product attention)
    rotate_values: bool = True       # False = aggregate values without rotation or counter-rotation (M2.4)
    zero_rotations: bool = False
    zero_decay: bool = False
    zero_process_noise: bool = False
    zero_key_measurement_noise: bool = False

    # --- Numerics ---
    epsilon: float = 1e-5
    cov_epsilon: float = 1e-2
    min_exponent: float = -10.0
    max_exponent: float = 0.0

    def __post_init__(self):
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even (conjugate frequency pairs), got {self.head_dim}")

    @classmethod
    def from_args(cls, args, d_model, n_heads, head_dim):
        """Build a config from the argparse-style `args` used by the training notebooks."""
        removed = {
            "learn_t_shift": True, "learn_decay": True, "sep_params": True, "use_colored_prior": True,
            "use_inner_residual": True, "add_gaussian_noise": True, "use_complex_conj_constraint": False,
            "compute_pulled_forward_estimates": True,
        }
        for name, bad in removed.items():
            if getattr(args, name, not bad) == bad:
                raise ValueError(f"args.{name}={bad} is no longer supported (feature removed in the refactor).")
        if getattr(args, "use_complex_input_norm", 0) or getattr(args, "use_complex_output_norm", False):
            raise ValueError("args.use_complex_input_norm / use_complex_output_norm are no longer supported (feature removed).")
        if getattr(args, "additive_bias_type", 1) not in (0, 1) or getattr(args, "multiplicative_bias_type", 1) not in (0, 1):
            raise ValueError("Only additive/multiplicative_bias_type 0 (off) or 1 (DLE) are supported.")

        if getattr(args, "scale_decay_by_time_interval", False):
            decay_scale = args.max_fixed_decay / (args.seq_len - 1)
        else:
            decay_scale = 1.0

        use_sc = args.use_SC_RoPE
        return cls(
            d_model=d_model, n_heads=n_heads, head_dim=head_dim,
            frequency_init="spectral" if use_sc else "rope",
            decay_init="log_linear" if (args.use_log_linear_decay and not use_sc) else "spectral",
            damping=args.damping,
            zero_decay_frac=args.zero_frac,
            decay_scale=decay_scale,
            omega_rescale=args.omega_rescale,
            learn_rotations=args.learn_rotations,
            process_noise="steady_state" if args.use_ss_process_noise else "variance",
            init_steady_state_var=args.target_ss_process_var,
            init_measurement_var=args.target_measurement_var,
            robust_weight=args.use_robust_weight,
            dle_bias=args.additive_bias_type == 1,
            dle_gate=args.multiplicative_bias_type == 1,
            full_residual_norm=bool(args.use_full_residual_norm),
            rotate_values=args.rotate_values,
            zero_rotations=args.zero_rotations,
            zero_decay=args.lambda_real_zero,
            zero_process_noise=args.zero_process_noise,
            zero_key_measurement_noise=args.zero_key_measurement_noise,
            epsilon=args.epsilon,
            min_exponent=args.min_exponent,
            max_exponent=args.max_exponent,
        )
