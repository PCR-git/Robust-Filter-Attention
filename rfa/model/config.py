"""Typed configuration of the transformer backbones (the RFA layer itself is configured by rfa.RFAConfig)."""

from dataclasses import dataclass
from typing import Literal, Optional, Union

from rfa import RFAConfig


@dataclass(frozen=True)
class AttentionConfig:
    """
    Baseline multi-head dot-product attention with optional positional mechanisms:

        rope:          None | "rope" | "sc_rope"   rotary embedding of queries and keys
        alibi:         linear distance bias on the logits
        decay_damping: exponential decay of the attention weights, mu_h = damping * omega_h,max
                       (applied after the softmax, as in RFA); None = no decay

    The baselines of the paper are B1 = rope, B2 = alibi, B3 = rope + decay, B4 = sc_rope + decay.
    """

    d_model: int
    n_heads: int
    width: int                       # total query/key/value width across heads
    rope: Optional[Literal["rope", "sc_rope"]] = "rope"
    alibi: bool = False
    decay_damping: Optional[float] = None

    def __post_init__(self):
        if self.width % self.n_heads != 0:
            raise ValueError(f"width={self.width} is not divisible by n_heads={self.n_heads}")
        if self.rope not in (None, "rope", "sc_rope"):
            raise ValueError(f"rope must be None, 'rope' or 'sc_rope', got {self.rope!r}")


@dataclass(frozen=True)
class BackboneConfig:
    """
    A stack of pre-norm residual blocks, x + attn(norm1(x)) then x + ffn(norm2(x)), with a ReLU
    feedforward network, a final norm, and optional d->d projections around the stack.
    """

    attention: Union[RFAConfig, AttentionConfig]
    n_blocks: int
    ffn_dim: int
    norm: Optional[Literal["layer", "rms"]] = "layer"  # None = no normalization layers at all
    attn_norm: bool = True           # normalize the block input before attention
    ffn_norm: bool = True            # normalize before the feedforward network
    io_projections: bool = True      # needed when the backbone is used directly as a regressor; redundant
                                     # inside a LanguageModel (embedding and LM head already provide them)

    def __post_init__(self):
        if self.norm not in (None, "layer", "rms"):
            raise ValueError(f"norm must be None, 'layer' or 'rms', got {self.norm!r}")

    @property
    def d_model(self):
        return self.attention.d_model
