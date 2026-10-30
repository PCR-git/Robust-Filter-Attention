"""
Capture golden outputs of the ORIGINAL baseline attention layers (B1-B4) before cleaning them up.
Run once, from the repo root:  python tests/make_golden_baselines.py
"""
import os, sys
import torch, torch.nn as nn
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tests.configs import make_args

D, H, SEQ = 32, 4, 20
BASELINES = {  # args overrides as in the training notebook
    "B1_rope":     dict(use_rope=True, use_alibi=False, use_relative_decay_vanilla=False),
    "B2_alibi":    dict(use_rope=False, use_alibi=True, use_relative_decay_vanilla=False),
    "B3_rope_decay": dict(use_rope=True, use_alibi=False, use_SC_RoPE=False, scale_decay_by_time_interval=False,
                        damping=0.05, use_relative_decay_vanilla=True),
    "B4_sc_rope":  dict(use_rope=False, use_alibi=False, use_SC_RoPE=True, scale_decay_by_time_interval=False,
                        damping=0.05, use_relative_decay_vanilla=True),
}

def build(overrides):
    from rfa.model import TransformerNetwork
    return TransformerNetwork(D, 2 * D, 4 * D, H, make_args(overrides), num_blocks=2, Norm=nn.LayerNorm)

def run(model, x, w):
    out, attn = model(x, causal=True)
    loss = (out * w).sum()
    loss.backward()
    return out.detach(), loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}, attn[-1].detach()

if __name__ == "__main__":
    results = {}
    for name, ov in BASELINES.items():
        torch.manual_seed(0)
        model = build(ov)
        init_state = {k: v.clone() for k, v in model.state_dict().items()}
        model = model.double()
        gen = torch.Generator().manual_seed(1)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.1 * torch.randn(p.shape, generator=gen, dtype=p.dtype))
        x = torch.randn(2, SEQ, D, generator=gen, dtype=torch.float64)
        w = torch.randn(2, SEQ, D, generator=gen, dtype=torch.float64)
        state = {k: v.clone() for k, v in model.state_dict().items()}
        out, loss, grads, attn = run(model, x, w)
        results[name] = dict(overrides=ov, init_state=init_state, state=state, x=x, w=w, out=out, loss=loss, grads=grads, attn=attn)
        print(f"[ok] {name}")
    torch.save(results, os.path.join(REPO, "tests", "golden", "golden_baselines.pt"))
