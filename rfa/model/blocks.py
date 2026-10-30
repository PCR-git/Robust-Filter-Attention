import torch.nn as nn

from rfa import RFAConfig, MultiheadIsotropicRFA
from .config import AttentionConfig, BackboneConfig
from .layers import MultiHeadAttentionLayer

NORMS = {"layer": nn.LayerNorm, "rms": nn.RMSNorm}


def build_attention(config):
    """RFA layer for an RFAConfig, baseline dot-product attention for an AttentionConfig."""
    if isinstance(config, RFAConfig):
        return MultiheadIsotropicRFA(config)
    if isinstance(config, AttentionConfig):
        return MultiHeadAttentionLayer(config)
    raise TypeError(f"Unknown attention config type {type(config).__name__}")


class Block(nn.Module):
    """
    Pre-norm residual block: x + attn(norm1(x)), then x + ffn(norm2(x)), with a ReLU feedforward network.
    The attention is RFA or a baseline, depending on the type of config.attention.
    """

    def __init__(self, config: BackboneConfig):
        super().__init__()
        d = config.d_model
        self.attn = build_attention(config.attention)
        norm = NORMS[config.norm] if config.norm else None
        self.norm1 = norm(d) if norm and config.attn_norm else None
        self.norm2 = norm(d) if norm and config.ffn_norm else None
        self.ffn = nn.Sequential(nn.Linear(d, config.ffn_dim), nn.ReLU(), nn.Linear(config.ffn_dim, d))

    def forward(self, x, t_measure=None, causal=True, return_metadata=False):
        """x: [B, N, d_model]. Returns (out [B, N, d_model], the attention layer's info dict)."""
        h = x if self.norm1 is None else self.norm1(x)
        attn_out, info = self.attn(h, t_measure=t_measure, causal=causal, return_metadata=return_metadata)
        x = x + attn_out
        h = x if self.norm2 is None else self.norm2(x)
        return x + self.ffn(h), info
