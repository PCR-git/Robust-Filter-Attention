"""
Pure functions implementing the Isotropic RFA computations.

Layout conventions:
    complex vectors     [..., D, 2]   (last dim = real, imaginary)
    pairwise tensors    [B, N, N, H]  (query i, key j, head h; B may be 1 for shared time grids)
"""

import torch
import torch.nn.functional as F


def rotate(x, cos, sin):
    """Complex multiplication x * (cos + i sin). x: [..., D, 2]; cos, sin: broadcastable to [..., D]."""
    re = x[..., 0] * cos - x[..., 1] * sin
    im = x[..., 0] * sin + x[..., 1] * cos
    return torch.stack([re, im], dim=-1)


def rotation_angles(omega, t):
    """omega [H, D], t [B, N] -> (cos, sin), each [B, N, H, D]."""
    theta = omega * t[..., None, None]
    return torch.cos(theta), torch.sin(theta)


def pairwise_lags(t):
    """t [B, N] -> |t_i - t_j| as [B, N, N, 1]."""
    return (t[:, :, None] - t[:, None, :]).abs()[..., None]


def lag_covariance(mu, lags, decay_exp, process_var, eta_sq, gamma_sq, process_noise, mu_eps=1e-5, cov_eps=1e-2):
    """
    Isotropic DLE covariance of a transported key as a function of lag:

        Sigma(dt) = Var_process(dt) + eta^2 e^{-2 mu dt} + gamma^2 + cov_eps

    with Var_process(dt) = sigma^2 (1 - e^{-2 mu dt}) / (2 mu)   ("variance")
                     or  = sigma~^2 (1 - e^{-2 mu dt})            ("steady_state"),
    and the Brownian limit sigma^2 * dt for mu -> 0 in both cases.

    mu, process_var, eta_sq, gamma_sq: [H]. lags: [B, N, N, 1]. decay_exp = clamp(-mu * lags): [B, N, N, H].
    """
    E = torch.exp(decay_exp)
    unitary = mu.abs() < mu_eps
    one_minus_E2 = -torch.expm1(2 * decay_exp)
    if process_noise == "variance":
        denom = torch.where(unitary, torch.ones_like(mu), 2 * mu)
        var_process = process_var * torch.where(unitary, lags, one_minus_E2 / denom)
    elif process_noise == "steady_state":
        var_process = torch.where(unitary, process_var * lags, process_var * one_minus_E2)
    else:
        raise ValueError(f"Unknown process_noise {process_noise!r}")
    return var_process + eta_sq * E**2 + gamma_sq + cov_eps


def residual_norm_sq(Q_rot, K_rot, E, full=True):
    """
    ||q_i - E_ij k_j||^2 = ||q_i||^2 + E_ij^2 ||k_j||^2 - 2 E_ij Re<q_i, k_j>,  in the rotated frame.

    Q_rot, K_rot: [B, N, H, D, 2]; E: [B, N, N, H]. Returns [B, N, N, H].
    With full=False only the cross term -E_ij Re<q_i, k_j> is returned (dot-product attention).
    """
    cross = 2 * E * torch.einsum("bihdc,bjhdc->bijh", Q_rot, K_rot)
    if not full:
        return -cross / 2
    q_sq = Q_rot.pow(2).sum(dim=(-1, -2))[:, :, None, :]
    k_sq = K_rot.pow(2).sum(dim=(-1, -2))[:, None, :, :]
    return q_sq + E**2 * k_sq - cross


def robust_logits(R_sq, Sigma, nu, d, robust_weight=True, dle_bias=True, dle_gate=True, kappa=None):
    """
    Attention logits from the Student-t (robust_weight=True) or exponential likelihood
    of each transported key:

        robust:      log P - kappa * log(1 + R^2 / (nu Sigma)),   kappa = (nu + d) / d
        exponential: log P - R^2 / (nu Sigma)

    where P = 1 / Sigma. dle_bias=False drops log P; dle_gate=False replaces 1/(nu Sigma) by 1/sqrt(d).
    Pass kappa=1 for the unsharpened log-likelihood (used for the total-precision diagnostic).
    """
    bias = -torch.log(Sigma) if dle_bias else 0.0
    gate = 1.0 / (nu * Sigma) if dle_gate else 1.0 / d**0.5
    if robust_weight:
        if kappa is None:
            kappa = (nu + d) / d
        return bias - kappa * torch.log(1 + gate * R_sq)
    return bias - gate * R_sq
