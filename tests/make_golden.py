"""
Capture golden outputs from the ORIGINAL (pre-refactor) RFA implementation.

This was run once, from the repo root, BEFORE refactoring:
    python tests/make_golden.py
It only works against the original code (args-based MultiheadIsotropicRFA and the
isotropic_rfa package); it is kept as a record of how tests/golden/golden.pt was made.

For every config in CONFIGS this records:
  - init_state:  state_dict right after construction with a fixed seed (checks init is unchanged)
  - state:       a randomly perturbed float64 state_dict (exercises generic parameter values)
  - inputs, outputs and parameter gradients (train mode)
  - eval-mode metadata (attention matrices etc.)

Configs that crash in the original code are recorded with their error message.
"""

import os
import sys
import copy
import traceback
from argparse import Namespace

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tests.configs import CONFIGS, make_args, D_E, N_HEADS, N_BLOCKS, BATCH, SEQ, HIDDEN

GOLDEN_PATH = os.path.join(REPO, "tests", "golden", "golden.pt")


def build_network(args):
    from model import RFATransformerNetwork
    return RFATransformerNetwork(
        args, num_blocks=N_BLOCKS, n_heads=N_HEADS, input_dim=D_E,
        query_key_dim_total=args.d_k_total, value_dim_total=args.d_v_total, hidden_dim=HIDDEN,
    )


def perturb_(model, gen):
    """Add noise to every parameter so tests don't sit at symmetric init values."""
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn(p.shape, generator=gen, dtype=p.dtype))


def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().clone()
    return x


def capture(name, overrides, call):
    args = make_args(overrides)

    torch.manual_seed(0)
    model = build_network(args)
    init_state = {k: to_cpu(v) for k, v in model.state_dict().items()}

    model = model.double()
    gen = torch.Generator().manual_seed(1)
    perturb_(model, gen)
    state = {k: to_cpu(v) for k, v in model.state_dict().items()}

    x = torch.randn(BATCH, SEQ, D_E, generator=gen, dtype=torch.float64)
    w = torch.randn(BATCH, SEQ, D_E, generator=gen, dtype=torch.float64)
    t_measure = None
    if call.get("irregular_time"):
        steps = torch.rand(BATCH, SEQ, generator=gen, dtype=torch.float64) * 2.0 + 0.1
        t_measure = torch.cumsum(steps, dim=1) - steps[:, :1]

    fwd_kwargs = dict(t_measure=t_measure, t_shift=call.get("t_shift"), causal=call.get("causal", True))

    # --- Train-mode forward + backward ---
    model.train()
    torch.manual_seed(2)
    out, out_dict = model(x, **fwd_kwargs)
    loss = (out * w).sum()
    loss.backward()
    grads = {n: to_cpu(p.grad) for n, p in model.named_parameters() if p.grad is not None}
    train_scalars = {k: to_cpu(v) for k, v in out_dict.items() if isinstance(v, torch.Tensor)}

    # --- Eval-mode forward with metadata ---
    model.eval()
    model.zero_grad()
    eval_args_metadata = args.compute_metadata
    for m in model.modules():
        if hasattr(m, "args"):
            m.args.compute_metadata = True
    with torch.no_grad():
        torch.manual_seed(3)
        out_eval, meta = model(x, **fwd_kwargs)
    for m in model.modules():
        if hasattr(m, "args"):
            m.args.compute_metadata = eval_args_metadata
    meta = {k: to_cpu(v) for k, v in meta.items() if isinstance(v, torch.Tensor)}

    return {
        "overrides": overrides,
        "call": call,
        "init_state": init_state,
        "state": state,
        "x": x, "w": w, "t_measure": t_measure,
        "out": to_cpu(out), "loss": to_cpu(loss), "grads": grads,
        "train_dict": train_scalars,
        "out_eval": to_cpu(out_eval), "meta": meta,
    }


def main():
    torch.use_deterministic_algorithms(False)
    results = {}
    for name, (overrides, call) in CONFIGS.items():
        try:
            results[name] = capture(name, overrides, call)
            print(f"[ok]     {name}")
        except Exception as e:
            msg = "".join(traceback.format_exception_only(type(e), e)).strip()
            results[name] = {"overrides": overrides, "call": call, "error": msg}
            print(f"[BROKEN] {name}: {msg}")
    os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
    torch.save(results, GOLDEN_PATH)
    print(f"\nSaved {len(results)} configs to {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
