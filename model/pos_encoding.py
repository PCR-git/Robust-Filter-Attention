import math

import torch
import torch.nn as nn

from rfa.init import init_spectrally_coupled_rope
from rfa.kernels import rotate

##########################################################################################
##########################################################################################

################################## POSITIONAL ENCODING ###################################

##########################################################################################
##########################################################################################

class RoPE(nn.Module):
    """
    Rotary positional embedding. Feature pairs (2k, 2k+1) of each head are rotated by
    angle theta_k * position. Input/output shape: [B, L, H, D_head].
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dimension must be even, got {dim}")
        theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # [D/2]
        self.register_buffer("theta", theta)

    def angles(self, seq_len, device, dtype):
        pos = torch.arange(seq_len, device=device, dtype=dtype)
        return pos[:, None, None] * self.theta.to(dtype)  # [L, 1, D/2] (broadcast over heads)

    def forward(self, x):
        B, L, H, D = x.shape
        theta = self.angles(L, x.device, x.dtype)
        return rotate(x.view(B, L, H, D // 2, 2), torch.cos(theta), torch.sin(theta)).view(B, L, H, D)

##########################################################################################
##########################################################################################

class SCRoPE(RoPE):
    """
    Spectrally-coupled RoPE: the RoPE spectrum is partitioned across heads
    (head 0 = lowest frequencies), as in SC-RFA.
    """

    def __init__(self, dim: int, n_heads: int, base: float = 10000.0):
        nn.Module.__init__(self)
        if dim % 2 != 0:
            raise ValueError(f"RoPE dimension must be even, got {dim}")
        theta, _ = init_spectrally_coupled_rope(n_heads, dim // 2, base=base)  # [H, D/2]
        self.register_buffer("theta", theta)

    def angles(self, seq_len, device, dtype):
        pos = torch.arange(seq_len, device=device, dtype=dtype)
        return pos[:, None, None] * self.theta.to(dtype)  # [L, H, D/2]

##########################################################################################
##########################################################################################

def get_alibi_slopes(n):
    """Standard ALiBi slopes, shape [1, n, 1, 1]."""
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        slopes = get_slopes_power_of_2(n)
    else:
        # For non-power-of-2 heads, interpolate
        closest_power_of_2 = 2**math.floor(math.log2(n))
        slopes = get_slopes_power_of_2(closest_power_of_2) + \
                 get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:n-closest_power_of_2]

    return torch.tensor(slopes).view(1, n, 1, 1)
