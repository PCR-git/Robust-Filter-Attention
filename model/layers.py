import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfa.init import init_spectrally_coupled_rope
from .pos_encoding import RoPE, SCRoPE, get_alibi_slopes

##########################################################################################
##########################################################################################

#################################### ATTENTION LAYERS ####################################

##########################################################################################
##########################################################################################

class MultiHeadAttentionLayer(nn.Module):
    """
    Baseline multi-head dot-product attention with optional positional mechanisms:

        rope:   None | "rope" | "sc_rope"   rotary embedding of queries and keys
        alibi:  linear distance bias on the logits
        decay:  exponential decay of the attention weights, mu_h = damping * omega_h,max
                (applied after the softmax, as in RFA)

    The baselines of the paper are B1 = rope, B2 = alibi, B3 = rope + decay, B4 = sc_rope + decay.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, rope=None, alibi=False, decay_damping=None):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Projections for Q, K, V, and final output
        self.query_projection = nn.Linear(input_dim, hidden_dim)
        self.key_projection = nn.Linear(input_dim, hidden_dim)
        self.value_projection = nn.Linear(input_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, input_dim)

        if rope == "rope":
            self.rope = RoPE(self.head_dim)
        elif rope == "sc_rope":
            self.rope = SCRoPE(self.head_dim, n_heads=num_heads)
        elif rope is not None:
            raise ValueError(f"rope must be None, 'rope' or 'sc_rope', got {rope!r}")
        else:
            self.rope = None

        self.alibi = alibi
        if alibi:
            self.register_buffer("alibi_slopes", get_alibi_slopes(num_heads))  # [1, H, 1, 1]

        self.decay = decay_damping is not None
        if self.decay:
            _, mu = init_spectrally_coupled_rope(num_heads, self.head_dim // 2, b=decay_damping)
            self.register_buffer("mu", mu)

    @classmethod
    def from_args(cls, input_dim, hidden_dim, num_heads, args):
        """Baseline selection from the notebook flags (see the class docstring for B1-B4)."""
        if getattr(args, "use_relative_decay_vanilla", False):
            rope = "sc_rope" if args.use_SC_RoPE else "rope"
            damping = args.damping
        else:
            rope = "rope" if args.use_rope else None
            damping = None
        return cls(input_dim, hidden_dim, num_heads, rope=rope, alibi=args.use_alibi, decay_damping=damping)

    def forward(self, query, keys, values, causal=True):
        B, L, _ = query.shape
        Q = self.query_projection(query).view(B, L, self.num_heads, self.head_dim)
        K = self.key_projection(keys).view(B, L, self.num_heads, self.head_dim)
        V = self.value_projection(values).view(B, L, self.num_heads, self.head_dim)

        if self.rope is not None:
            Q, K = self.rope(Q), self.rope(K)

        # [B, H, L, D_head]
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        pos = torch.arange(L, device=query.device)
        dist = (pos[:, None] - pos[None, :]).abs()  # [L, L]
        if self.alibi:
            scores = scores - self.alibi_slopes * dist
        if causal:
            scores = scores.masked_fill(~torch.ones(L, L, dtype=torch.bool, device=query.device).tril(), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        if self.decay:
            attn_weights = attn_weights * torch.exp(-self.mu.view(1, -1, 1, 1) * dist)

        context = torch.matmul(attn_weights, V).transpose(1, 2).reshape(B, L, self.hidden_dim)
        return self.output_projection(context), attn_weights
