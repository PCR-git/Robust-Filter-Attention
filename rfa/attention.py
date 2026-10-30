import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RFAConfig
from .init import (inv_softplus, init_complex_matrix, init_complex_linear_,
                   init_rope, init_spectrally_coupled_rope, init_decay_per_head)
from .kernels import rotate, rotation_angles, pairwise_lags, lag_covariance, residual_norm_sq, robust_logits


class MultiheadIsotropicRFA(nn.Module):
    """
    Multi-head Isotropic Robust Filter Attention (Algorithm 1 of the paper).

    Each head models tokens as noisy observations of a latent state following a linear SDE
    with eigenvalues lambda_k = -mu + i*omega_k and isotropic noise. Keys are transported
    to the query time, attention weights come from the robust likelihood of each transported
    key under the DLE covariance, and values are aggregated in the rotated (stationary)
    frame, then rotated back.

    Parameters (per head):
        omega      rotation frequencies [H, head_dim/2] (buffer unless learn_rotations)
        mu         decay rate [H] (fixed buffer)
        sigma_raw  process noise; softplus gives sigma^2 or sigma~^2 depending on config.process_noise
        eta_raw    key-side measurement noise, eta^2 = softplus(eta_raw) + eps
        gamma_raw  query-side noise floor, gamma^2 = softplus(gamma_raw) + eps
        tau_raw    softmax inverse temperature, softplus(tau_raw) + eps
        nu_raw     Student-t degrees of freedom, nu = d * softplus(nu_raw) + 2 + eps
    """

    def __init__(self, config: RFAConfig):
        super().__init__()
        self.config = c = config
        H, D = c.n_heads, c.head_dim
        width = H * D

        # Projections into / out of the complex eigenbasis (real and imaginary parts stacked)
        self.W_q = nn.Linear(c.d_model, 2 * width)
        self.W_k = nn.Linear(c.d_model, 2 * width)
        self.W_v = nn.Linear(c.d_model, 2 * width)
        self.W_o = nn.Linear(2 * width, c.d_model)

        # --- Dynamics ---
        n_freqs = D // 2  # frequencies come in conjugate pairs (+omega, -omega) so that A is real
        omega_sc, mu = init_spectrally_coupled_rope(H, n_freqs, b=c.damping, zero_frac=c.zero_decay_frac)
        omega = omega_sc if c.frequency_init == "spectral" else init_rope(H, n_freqs)
        if c.decay_init == "log_linear":
            mu = init_decay_per_head(H, min_decay=0.01, max_decay=0.95, zero_frac=c.zero_decay_frac)
        if c.zero_rotations:
            omega = torch.zeros_like(omega)
        if c.learn_rotations:
            self.omega = nn.Parameter(omega)
        else:
            self.register_buffer("omega", omega)
        self.register_buffer("mu", mu * c.decay_scale)

        # --- Noise ---
        # Note: eta, gamma (and sigma~ in "steady_state" mode) store the target value as the *raw*
        # parameter, so their initial variance is softplus(target). In "variance" mode sigma^2 is
        # initialized to 2 * mu * target using the decay rate before decay_scale is applied.
        if c.process_noise == "variance":
            sigma_init = inv_softplus(2 * mu * c.init_steady_state_var)
        else:
            sigma_init = torch.full((H,), c.init_steady_state_var)
        self.sigma_raw = nn.Parameter(sigma_init)
        self.eta_raw = nn.Parameter(torch.full((H,), c.init_measurement_var))
        self.gamma_raw = nn.Parameter(torch.full((H,), c.init_measurement_var))

        # --- Softmax temperature and robustness ---
        self.tau_raw = nn.Parameter(torch.ones(H))
        self.nu_raw = nn.Parameter(torch.full((H,), c.init_nu_over_d))

        init_complex_linear_(self.W_q, init_complex_matrix(c.d_model, width), "in")
        init_complex_linear_(self.W_k, init_complex_matrix(c.d_model, width), "in")
        init_complex_linear_(self.W_v, init_complex_matrix(c.d_model, width), "in")
        init_complex_linear_(self.W_o, init_complex_matrix(width, c.d_model), "out")

    # ------------------------------------------------------------------

    def sde_params(self):
        """Constrained per-head parameters: dict of [H] tensors (omega is [H, head_dim])."""
        c = self.config
        mu = torch.zeros_like(self.mu) if c.zero_decay else self.mu
        omega = c.omega_rescale * torch.stack([self.omega, -self.omega], dim=-1).flatten(-2)

        process_var = F.softplus(self.sigma_raw)
        eta_sq = F.softplus(self.eta_raw) + c.epsilon
        gamma_sq = F.softplus(self.gamma_raw) + c.epsilon
        if c.zero_process_noise:
            process_var = process_var * 0
        if c.zero_key_measurement_noise:
            eta_sq = eta_sq * 0

        d = c.head_dim
        nu = d * F.softplus(self.nu_raw) + 2.0 + c.epsilon
        tau = F.softplus(self.tau_raw) + c.epsilon
        return dict(mu=mu, omega=omega, process_var=process_var, eta_sq=eta_sq, gamma_sq=gamma_sq, nu=nu, tau=tau)

    def _project_in(self, W, Z):
        B, N, _ = Z.shape
        return W(Z).view(B, N, 2, self.config.n_heads, self.config.head_dim).permute(0, 1, 3, 4, 2)

    def forward(self, Z_q, Z_k=None, Z_v=None, t_measure=None, causal=True, return_metadata=False):
        """
        Args:
            Z_q, Z_k, Z_v: real inputs [B, N, d_model] (Z_k, Z_v default to Z_q).
            t_measure:     observation times [B, N]; None means regular positions 0..N-1.
            causal:        restrict attention to j <= i.
            return_metadata: also return attention matrices and other diagnostics.

        Returns:
            out [B, N, d_model] (to be added to the residual stream), and a dict with the
            per-head SDE parameters (plus diagnostics if return_metadata).
        """
        c = self.config
        Z_k = Z_q if Z_k is None else Z_k
        Z_v = Z_q if Z_v is None else Z_v
        B, N, _ = Z_q.shape
        p = self.sde_params()

        if t_measure is None:
            t = torch.arange(N, device=Z_q.device, dtype=p["mu"].dtype)[None]  # [1, N]
        else:
            t = t_measure.to(p["mu"].dtype)

        # --- Dynamics: rotations, decay kernel E and DLE covariance Sigma (all [B|1, N, N, H]) ---
        cos, sin = rotation_angles(p["omega"], t)  # [B|1, N, H, D]
        lags = pairwise_lags(t)
        decay_exp = torch.clamp(-p["mu"] * lags, min=c.min_exponent, max=c.max_exponent)
        E = torch.exp(decay_exp)
        Sigma = lag_covariance(p["mu"], lags, decay_exp, p["process_var"], p["eta_sq"], p["gamma_sq"],
                               c.process_noise, mu_eps=c.epsilon, cov_eps=c.cov_epsilon)

        # --- Project and rotate into the stationary frame (Phi^-) ---
        Q = rotate(self._project_in(self.W_q, Z_q), cos, -sin)
        K = rotate(self._project_in(self.W_k, Z_k), cos, -sin)
        V = self._project_in(self.W_v, Z_v)
        if c.rotate_values:
            V = rotate(V, cos, -sin)

        # --- Robust attention weights ---
        R_sq = residual_norm_sq(Q, K, E, full=c.full_residual_norm)
        logits = robust_logits(R_sq, Sigma, p["nu"], c.head_dim, c.robust_weight, c.dle_bias, c.dle_gate)
        logits = p["tau"] * logits
        if causal:
            mask = torch.ones(N, N, dtype=torch.bool, device=Z_q.device).tril()
            logits = logits.masked_fill(~mask[None, :, :, None], float("-inf"))
        A = torch.softmax(logits, dim=2)  # normalize over keys j

        # --- Aggregate decayed values, then rotate back (Phi^+) ---
        est = torch.einsum("bijh,bjhdc->bihdc", A * E, V)
        if c.rotate_values:
            est = rotate(est, cos, sin)

        out = self.W_o(est.permute(0, 1, 4, 2, 3).reshape(B, N, -1))  # [re heads..., im heads...]

        info = self._param_summary(p)
        if return_metadata:
            info.update(self._metadata(p, A, E, Sigma, R_sq, est, causal))
        return out, info

    # ------------------------------------------------------------------

    def _param_summary(self, p):
        """Per-head physical parameters (keys kept compatible with the training/plotting code)."""
        mu, d = p["mu"], self.config.head_dim
        if self.config.process_noise == "variance":
            sigma_sq = p["process_var"]
            steady = torch.where(mu > 0, sigma_sq / (2 * mu).clamp_min(1e-12), torch.full_like(mu, float("inf")))
        else:
            steady = p["process_var"]
            sigma_sq = 2 * mu * steady
        return {
            "mu_v": mu, "sigma_sq_v": sigma_sq, "sigma_tilde_sq_v": steady,
            "eta_sq_v": p["eta_sq"], "gamma_sq_v": p["gamma_sq"],
            "tau": p["tau"], "nu_over_d": p["nu"] / d,
        }

    def _metadata(self, p, A, E, Sigma, R_sq, est, causal):
        c = self.config
        H, D = c.n_heads, c.head_dim
        prior_precision = 1 / Sigma
        log_total_precision = robust_logits(R_sq, Sigma, p["nu"], D, c.robust_weight, c.dle_bias, c.dle_gate, kappa=1)
        if causal:
            mask = torch.ones(A.shape[1], A.shape[2], dtype=torch.bool, device=A.device).tril()[None, :, :, None]
            prior_precision = prior_precision.masked_fill(~mask, 0.0)
            log_total_precision = log_total_precision.masked_fill(~mask, float("-inf"))
        eigenvals = torch.stack([-p["mu"][:, None].expand(H, D), p["omega"]]).reshape(2, 1, H * D, 1)
        return {
            "attn_mat": A,                        # softmax weights [B, N, N, H]
            "decayed_attn_mat": A * E,            # weights actually applied to values
            "A_prior": prior_precision.squeeze(0),  # DLE precision 1/Sigma per lag
            "P_tot": torch.logsumexp(log_total_precision, dim=2),  # log accumulated precision [B, N, H]
            "est_latent": est,                    # filtered estimate in the eigenbasis [B, N, H, D, 2]
            "eigenvals": eigenvals,               # [re/im, 1, H*D, 1]
            "x_hat": None, "gate": None,          # removed features, kept for plotting-code compatibility
        }
