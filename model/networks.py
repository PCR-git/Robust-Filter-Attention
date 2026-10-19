import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .blocks import Block, NORMS
from .config import BackboneConfig


class Backbone(nn.Module):
    """
    Stack of residual Blocks (RFA or baseline attention, see BackboneConfig), with a final norm and
    optional d->d input/output projections. Maps [B, N, d_model] -> [B, N, d_model].

    grad_checkpointing recomputes each block in the backward pass instead of storing its
    activations (much less memory, ~30% more time); it can be switched at any time.
    """

    def __init__(self, config: BackboneConfig, grad_checkpointing=False):
        super().__init__()
        self.config = config
        self.grad_checkpointing = grad_checkpointing
        d = config.d_model
        self.input_proj = nn.Linear(d, d) if config.io_projections else nn.Identity()
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_blocks)])
        self.final_norm = NORMS[config.norm](d) if config.norm else None
        self.output_proj = nn.Linear(d, d) if config.io_projections else nn.Identity()

    def forward(self, x, t_measure=None, causal=True, return_metadata=False):
        """
        Args:
            x:          [B, N, d_model]
            t_measure:  observation times [B, N] (RFA only); None means regular positions 0..N-1
            causal:     restrict attention to j <= i
            return_metadata: RFA layers also return attention matrices and other diagnostics

        Returns:
            out [B, N, d_model], and a list with each block's attention info dict (RFA: per-head SDE
            parameters, plus diagnostics if return_metadata; baseline: "attn_mat").
        """
        x = self.input_proj(x)
        infos = []
        for block in self.blocks:
            if self.training and self.grad_checkpointing:
                x, info = checkpoint(block, x, t_measure, causal, return_metadata, use_reentrant=False)
            else:
                x, info = block(x, t_measure=t_measure, causal=causal, return_metadata=return_metadata)
            infos.append(info)
        if self.final_norm is not None:
            x = self.final_norm(x)
        return self.output_proj(x), infos


class LanguageModel(nn.Module):
    """
    A generic wrapper that turns any backbone into a causal LM (token embedding tied to the output head).

    Use `loss()` for training: it computes the next-token cross-entropy in chunks without
    materializing the full [B, L, vocab] logits, which otherwise dominate memory.
    """

    def __init__(self, backbone, vocab_size=50257, embed_dim=256, embed_init_std=None, loss_chunk_size=2048):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.backbone = backbone
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.loss_chunk_size = loss_chunk_size

        # Tie weights. embed_init_std=None keeps the PyTorch default N(0, 1) used for the published
        # results; with tying this makes the initial logits large (initial loss ~3x ln(vocab)),
        # and a small init such as 0.02 is the usual choice.
        self.lm_head.weight = self.token_embedding.weight
        if embed_init_std is not None:
            nn.init.normal_(self.token_embedding.weight, std=embed_init_std)

    def forward(self, input_ids, **kwargs):
        """Returns (logits [B, L, vocab], backbone output info)."""
        out, info = self.backbone(self.token_embedding(input_ids), **kwargs)
        return self.lm_head(out), info

    def loss(self, input_ids, labels=None, **kwargs):
        """
        Mean next-token cross-entropy (position i predicts labels[i+1]; label -100 is ignored).
        Returns (loss, backbone output info).
        """
        labels = input_ids if labels is None else labels
        out, info = self.backbone(self.token_embedding(input_ids), **kwargs)
        h = out[:, :-1].reshape(-1, out.shape[-1])
        y = labels[:, 1:].reshape(-1)

        def chunk_loss(h_chunk, y_chunk):
            logits = F.linear(h_chunk, self.lm_head.weight, self.lm_head.bias)
            return F.cross_entropy(logits, y_chunk, ignore_index=-100, reduction="sum")

        total = out.new_zeros(())
        for s in range(0, h.shape[0], self.loss_chunk_size):
            h_chunk, y_chunk = h[s:s + self.loss_chunk_size], y[s:s + self.loss_chunk_size]
            if torch.is_grad_enabled():
                # Logits of each chunk are recomputed in the backward pass instead of stored
                total = total + checkpoint(chunk_loss, h_chunk, y_chunk, use_reentrant=False)
            else:
                total = total + chunk_loss(h_chunk, y_chunk)
        return total / (y != -100).sum(), info
