"""
Regression check: current RFA code vs. golden outputs of the original code.

    python tests/check_regression.py            # all configs
    python tests/check_regression.py M2 M3      # selected configs

For each config:
  init   - state_dict after seeded construction matches the original init
  fwd    - train-mode output and loss match (perturbed float64 weights)
  grad   - all parameter gradients match; dropped legacy params must have had zero gradient
  eval   - eval-mode output and shared metadata tensors match
"""

import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tests import adapter

GOLDEN_PATH = os.path.join(REPO, "tests", "golden", "golden.pt")
RTOL, ATOL = 1e-8, 1e-10


def close(a, b):
    if a is None or b is None:
        return a is None and b is None, "None mismatch"
    if a.shape != b.shape:
        return False, f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    a, b = a.double(), b.double()
    if torch.allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True):
        return True, ""
    err = (a - b).abs().max().item()
    return False, f"max abs err {err:.3e}"


def check_config(name, g):
    problems = []
    ov = g["overrides"]

    # --- init ---
    torch.manual_seed(0)
    model = adapter.build_network(ov)
    cur_init = model.state_dict()
    exp_init = adapter.convert_state(g["init_state"], ov)
    missing = set(exp_init) - set(cur_init)
    extra = set(cur_init) - set(exp_init)
    if missing or extra:
        problems.append(f"init keys: missing={sorted(missing)} extra={sorted(extra)}")
    for k in set(exp_init) & set(cur_init):
        ok, msg = close(cur_init[k], exp_init[k])
        if not ok:
            problems.append(f"init {k}: {msg}")

    # --- forward/backward ---
    model = model.double()
    model.load_state_dict(adapter.convert_state(g["state"], ov), strict=True)
    call = g["call"]
    fwd_kwargs = dict(t_measure=g["t_measure"], causal=call.get("causal", True))

    model.train()
    torch.manual_seed(2)
    out, _ = model(g["x"], **fwd_kwargs)
    loss = (out * g["w"]).sum()
    loss.backward()
    for label, a, b in [("out", out, g["out"]), ("loss", loss, g["loss"])]:
        ok, msg = close(a.detach(), b)
        if not ok:
            problems.append(f"fwd {label}: {msg}")

    cur_grads = {n: p.grad for n, p in model.named_parameters()}
    legacy_keys = {lk: adapter.legacy_to_new_key(lk, ov) for lk in g["grads"]}
    for lk, lg in g["grads"].items():
        nk = legacy_keys[lk]
        if nk is None:
            if lg.abs().max() > 0:
                problems.append(f"grad {lk}: dropped but had nonzero legacy gradient")
            continue
        cg = cur_grads.get(nk)
        if cg is None:
            if lg.abs().max() > 0:
                problems.append(f"grad {nk}: no gradient (legacy nonzero)")
            continue
        ok, msg = close(cg, lg)
        if not ok:
            problems.append(f"grad {nk}: {msg}")
    for nk, cg in cur_grads.items():
        if cg is not None and cg.abs().max() > 0 and nk not in legacy_keys.values():
            problems.append(f"grad {nk}: nonzero but absent in legacy")

    # --- eval + metadata ---
    model.eval()
    adapter.set_compute_metadata(model, True)
    with torch.no_grad():
        torch.manual_seed(3)
        out_eval, meta = model(g["x"], **fwd_kwargs)
    adapter.set_compute_metadata(model, False)
    ok, msg = close(out_eval, g["out_eval"])
    if not ok:
        problems.append(f"eval out: {msg}")
    compared = 0
    for k, v in meta.items():
        lk = adapter.legacy_meta_key(k, ov, call)
        if isinstance(v, torch.Tensor) and lk in g["meta"]:
            compared += 1
            ok, msg = close(v, g["meta"][lk])
            if not ok:
                problems.append(f"meta {k}: {msg}")
    if compared < 8:
        problems.append(f"meta: only {compared} keys compared (metadata missing?)")

    return problems


def check_baseline(name, g):
    """Baseline (B1-B4) TransformerNetwork vs golden outputs of the original baseline code."""
    from tests.make_golden_baselines import build, run
    problems = []
    torch.manual_seed(0)
    model = build(g["overrides"])
    cur_init = model.state_dict()
    params = dict(model.named_parameters())
    for k in set(g["init_state"]) - set(cur_init):
        problems.append(f"init: missing {k}")
    for k in set(cur_init) - set(g["init_state"]):
        if k in params:
            problems.append(f"init: new parameter {k}")
    for k in set(cur_init) & set(g["init_state"]):
        ok, msg = close(cur_init[k], g["init_state"][k])
        if not ok:
            problems.append(f"init {k}: {msg}")
    model = model.double()
    model.load_state_dict(g["state"], strict=False)
    out, loss, grads, attn = run(model, g["x"], g["w"])
    for label, a, b in [("out", out, g["out"]), ("loss", loss, g["loss"]), ("attn", attn, g["attn"])]:
        ok, msg = close(a, b)
        if not ok:
            problems.append(f"fwd {label}: {msg}")
    if set(grads) != set(g["grads"]):
        problems.append(f"grad keys differ: {sorted(set(grads) ^ set(g['grads']))}")
    for k in set(grads) & set(g["grads"]):
        ok, msg = close(grads[k], g["grads"][k])
        if not ok:
            problems.append(f"grad {k}: {msg}")
    return problems


def main(selected):
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    golden_baselines = torch.load(GOLDEN_PATH.replace("golden.pt", "golden_baselines.pt"), weights_only=False)
    golden.update({name: dict(g, baseline=True) for name, g in golden_baselines.items()})
    names = selected or list(golden)
    n_fail = 0
    for name in names:
        g = golden[name]
        if "error" in g:
            print(f"[skip]  {name}  (broken in original: {g['error']})")
            continue
        if name in adapter.REMOVED_FEATURES:
            print(f"[skip]  {name}  ({adapter.REMOVED_FEATURES[name]})")
            continue
        if name in adapter.INTENTIONAL_CHANGES:
            print(f"[changed] {name}  ({adapter.INTENTIONAL_CHANGES[name]})")
            continue
        try:
            problems = check_baseline(name, g) if g.get("baseline") else check_config(name, g)
        except Exception as e:
            problems = [f"exception: {type(e).__name__}: {e}"]
        if problems:
            n_fail += 1
            print(f"[FAIL]  {name}")
            for p in problems[:15]:
                print(f"          {p}")
            if len(problems) > 15:
                print(f"          ... {len(problems) - 15} more")
        else:
            print(f"[pass]  {name}")
    print(f"\n{n_fail} failing config(s)")
    return n_fail


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1:]) else 0)
