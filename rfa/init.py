"""Initialization helpers: complex projections, rotation frequencies and decay rates."""

import numpy as np
import torch


def inv_softplus(x):
    x = torch.as_tensor(x)
    return torch.log(torch.expm1(x) + 1e-6)


def init_complex_matrix(d_in, d_out):
    """Isotropic complex init: random magnitude, uniform phase. Returns (2, d_in, d_out)."""
    scale = np.sqrt(2 / (d_in + d_out))
    mag = scale * torch.randn(d_in, d_out)
    phase = 2 * np.pi * torch.rand(d_in, d_out)
    return torch.stack([mag * torch.cos(phase), mag * torch.sin(phase)])


def init_complex_linear_(layer, weight, layer_type):
    """
    Copy a complex weight (2, d_in, d_out) into a real nn.Linear.
      'in':  real d_in -> complex d_out, weight [2*d_out, d_in] = [W_r; W_i]
      'out': complex d_in -> real d_out (real part), weight [d_out, 2*d_in] = [W_r, W_i] / sqrt(2)
    """
    real_w, imag_w = weight[0].T, weight[1].T
    if layer_type == "in":
        W = torch.cat((real_w, imag_w), dim=0)
    elif layer_type == "out":
        W = torch.cat((real_w, imag_w), dim=1) / np.sqrt(2)
    else:
        raise ValueError(f"layer_type must be 'in' or 'out', got {layer_type!r}")
    with torch.no_grad():
        layer.weight.copy_(W)
        if layer.bias is not None:
            layer.bias.zero_()


def init_rope(n_heads, n_freqs, base=10000.0):
    """Standard RoPE frequencies, identical for every head. Returns [n_heads, n_freqs]."""
    omega = 1.0 / (base ** (torch.arange(n_freqs).float() / n_freqs))
    return omega.repeat(n_heads).view(n_heads, n_freqs)


def init_spectrally_coupled_rope(n_heads, n_freqs, b=5.0, zero_frac=0.0, base=10000.0):
    """
    SC-RFA: partition one RoPE spectrum across heads (head 0 = lowest frequencies)
    and couple each head's decay to its maximum frequency, mu_h = b * omega_h,max.
    The lowest `zero_frac` of heads get mu = 0.

    Returns (omega [n_heads, n_freqs], mu [n_heads]).
    """
    total = n_heads * n_freqs
    omega_master = 1.0 / (base ** (torch.arange(total).float() / total))
    omega = torch.flip(omega_master, dims=[0]).view(n_heads, n_freqs)
    mu = b * omega[:, -1]
    n_zero = int(n_heads * zero_frac)
    mu[:n_zero] = 0.0
    return omega, mu


def init_decay_per_head(n_heads, min_decay=0.001, max_decay=5.0, zero_frac=0.25):
    """Decay rates log-linearly spaced over heads, with the first `zero_frac` heads at 0."""
    n_zero = int(n_heads * zero_frac)
    decays = torch.exp(torch.linspace(np.log(min_decay), np.log(max_decay), n_heads - n_zero))
    return torch.cat([torch.zeros(n_zero), decays])
