import torch
import torch.nn as nn


class ComplexRMSNorm(nn.Module):
    """
    RMS normalization of complex per-head vectors [..., H, D, 2], with a learned real gain per feature.
    """

    def __init__(self, n_heads, head_dim, eps=1e-6, affine=True):
        super().__init__()
        self.eps = eps
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, n_heads, head_dim, 1))
        else:
            self.register_parameter("gamma", None)

    def forward(self, Z):
        # Mean of |z|^2 over complex features: 2 * mean over (D, re/im)
        mean_sq = 2.0 * Z.pow(2).mean(dim=(-2, -1), keepdim=True)
        Z = Z * torch.rsqrt(mean_sq + self.eps)
        return Z if self.gamma is None else self.gamma * Z
