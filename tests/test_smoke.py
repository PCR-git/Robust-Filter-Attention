"""
Smoke tests for behavior not covered by the golden regression files.

    python tests/test_smoke.py
"""

import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tests.configs import make_args, M2
from model import (RFATransformerNetwork, TransformerNetwork, LanguageModel, RFA_Block,
                   print_layer_diagnostics, get_model_sde_snapshots)
from rfa import RFAConfig, MultiheadIsotropicRFA


def rfa_net(overrides=None, d=32, heads=4, blocks=2):
    args = make_args(dict(M2, **(overrides or {})))
    return RFATransformerNetwork(args, num_blocks=blocks, n_heads=heads, input_dim=d,
                                 query_key_dim_total=d, value_dim_total=d, hidden_dim=2 * d)


def test_qk_norm_mode():
    """use_complex_input_norm=2 crashed in the original code (V_norm undefined)."""
    net = rfa_net({"use_complex_input_norm": 2})
    out, _ = net(torch.randn(2, 10, 32))
    assert out.shape == (2, 10, 32) and torch.isfinite(out).all()


def test_removed_flags_raise():
    for flag, value in [("learn_t_shift", True), ("learn_decay", True), ("sep_params", True),
                        ("use_colored_prior", True), ("use_inner_residual", True),
                        ("add_gaussian_noise", True), ("additive_bias_type", 2), ("use_outer_residual", False)]:
        try:
            rfa_net({flag: value})
        except ValueError:
            continue
        raise AssertionError(f"{flag}={value} should raise")


def test_no_value_rotation_ablation():
    """M2.4: with rotate_values=False the estimate is the decayed-attention average of the *unrotated* values."""
    torch.manual_seed(0)
    layer = MultiheadIsotropicRFA(RFAConfig(d_model=16, n_heads=2, head_dim=8, frequency_init="spectral",
                                            rotate_values=False)).double().eval()
    x = torch.randn(2, 12, 16, dtype=torch.float64)
    _, meta = layer(x, return_metadata=True)
    V = layer._project_in(layer.W_v, x, "cn_v")
    expected = torch.einsum("bijh,bjhdc->bihdc", meta["decayed_attn_mat"], V)
    assert torch.allclose(meta["est_latent"], expected)

    # ...and with rotations on, the frame change actually matters
    rot = MultiheadIsotropicRFA(RFAConfig(d_model=16, n_heads=2, head_dim=8, frequency_init="spectral")).double()
    rot.load_state_dict(layer.state_dict())
    _, meta_rot = rot(x, return_metadata=True)
    assert not torch.allclose(meta_rot["est_latent"], expected)


def test_m27_is_rotary_attention():
    """Preset M2.7 (M2.6 without value rotation) equals plain rotary dot-product attention."""
    import math
    from training.presets import make_args as preset_args
    from model import build_rfa_layer
    from rfa.kernels import rotate, rotation_angles
    for spectral in (True, False):
        torch.manual_seed(0)
        args = preset_args("M2.7", d_e=32, d_k_total=32, d_v_total=32, n_heads=4, use_SC_RoPE=spectral)
        layer = build_rfa_layer(args, n_heads=4, input_dim=32, query_key_dim_total=32, value_dim_total=32).double()
        with torch.no_grad():
            for p in layer.parameters():
                p.add_(0.1 * torch.randn_like(p))
        x = torch.randn(2, 24, 32, dtype=torch.float64)
        out, _ = layer(x)

        # Hand-written rotary attention with the layer's weights
        B, N, H, D = 2, 24, 4, 8
        proj = lambda W: W(x).view(B, N, 2, H, D).permute(0, 1, 3, 4, 2)
        p = layer.sde_params()
        assert torch.all(p["mu"] == 0)
        cos, sin = rotation_angles(p["omega"], torch.arange(N, dtype=torch.float64)[None])
        Q, K, V = rotate(proj(layer.W_q), cos, -sin), rotate(proj(layer.W_k), cos, -sin), proj(layer.W_v)
        logits = p["tau"] * torch.einsum("bihdc,bjhdc->bijh", Q, K) / math.sqrt(D)
        causal = torch.ones(N, N, dtype=torch.bool).tril()[None, :, :, None]
        A = torch.softmax(logits.masked_fill(~causal, float("-inf")), dim=2)
        ref = layer.W_o(torch.einsum("bijh,bjhdc->bihdc", A, V).permute(0, 1, 4, 2, 3).reshape(B, N, -1))
        assert torch.allclose(out, ref), f"spectral={spectral}"

        # Logits depend only on i - j (same token everywhere)
        same = x[:, :1].expand(B, N, 32)
        Qs = rotate(layer.W_q(same).view(B, N, 2, H, D).permute(0, 1, 3, 4, 2), cos, -sin)
        Ks = rotate(layer.W_k(same).view(B, N, 2, H, D).permute(0, 1, 3, 4, 2), cos, -sin)
        L = torch.einsum("bihdc,bjhdc->bijh", Qs, Ks)[0]
        assert max(torch.diagonal(L, offset=o).std(dim=-1).max().item() for o in range(-20, 21)) < 1e-10


def test_rope_is_relative_rotation():
    """RoPE/SC-RoPE preserve norms, and q.k scores depend only on the position offset."""
    from model import RoPE, SCRoPE
    torch.manual_seed(0)
    H, D, L = 4, 8, 64
    q, k = torch.randn(1, 1, H, D, dtype=torch.float64), torch.randn(1, 1, H, D, dtype=torch.float64)
    for rope in (RoPE(D).double(), SCRoPE(D, n_heads=H).double()):
        Q, K = rope(q.expand(1, L, H, D).clone()), rope(k.expand(1, L, H, D).clone())
        assert torch.allclose(Q.norm(dim=-1), q.norm(dim=-1).expand(1, L, H))
        scores = (Q[0, 5:] * K[0, :-5]).sum(-1)  # offset 5 at every position: [L-5, H]
        assert torch.allclose(scores, scores[:1].expand_as(scores))


def test_long_sequence_beyond_old_cap():
    """The original code had a hard 4096-token limit from precomputed buffers."""
    layer = MultiheadIsotropicRFA(RFAConfig(d_model=8, n_heads=1, head_dim=8, frequency_init="spectral"))
    out, _ = layer(torch.randn(1, 4200, 8))
    assert torch.isfinite(out).all()


def test_no_large_buffers_in_state_dict():
    sd = rfa_net().state_dict()
    biggest = max(v.numel() for v in sd.values())
    assert biggest < 10_000, f"unexpectedly large state_dict entry ({biggest} elements)"


def test_language_model_and_baselines():
    lm = LanguageModel(rfa_net(), vocab_size=100, embed_dim=32)
    logits, info = lm(torch.randint(0, 100, (2, 12)), t_measure=None, causal=True)
    assert logits.shape == (2, 12, 100) and "mu_v" in info

    base_args = make_args({"use_rope": True, "use_alibi": False, "use_relative_decay_vanilla": False})
    base = LanguageModel(TransformerNetwork(32, 32, 64, 4, base_args, num_blocks=2), vocab_size=100, embed_dim=32)
    logits, _ = base(torch.randint(0, 100, (2, 12)))
    assert logits.shape == (2, 12, 100)


def _lm_grads(lm, ids, use_loss_method):
    lm.zero_grad()
    if use_loss_method:
        loss, _ = lm.loss(ids, ids, causal=True)
    else:
        logits, _ = lm(ids, causal=True)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
    loss.backward()
    return loss.detach(), {n: p.grad.clone() for n, p in lm.named_parameters() if p.grad is not None}


def test_chunked_loss_and_checkpointing_are_exact():
    """LanguageModel.loss (chunked CE) and block checkpointing give the same loss and gradients as the plain path."""
    torch.manual_seed(0)
    ids = torch.randint(0, 100, (2, 30))
    for backbone in ("rfa", "baseline"):
        if backbone == "rfa":
            net = rfa_net()
        else:
            net = TransformerNetwork(32, 64, 64, 4, make_args({"use_rope": True, "use_alibi": False}), num_blocks=2)
        lm = LanguageModel(net, vocab_size=100, embed_dim=32).double()
        lm.loss_chunk_size = 7  # several chunks, last one partial
        ref_loss, ref_grads = _lm_grads(lm, ids, use_loss_method=False)
        for ckpt in (False, True):
            lm.backbone.grad_checkpointing = ckpt
            loss, grads = _lm_grads(lm, ids, use_loss_method=True)
            assert torch.allclose(loss, ref_loss), (backbone, ckpt)
            assert all(torch.allclose(grads[n], ref_grads[n]) for n in ref_grads), (backbone, ckpt)
        lm.backbone.grad_checkpointing = False


def test_metadata_toggle_and_diagnostics():
    net = rfa_net()
    net.eval()
    net.args.compute_metadata = True
    _, meta = net(torch.randn(1, 10, 32))
    assert meta["attn_mat"].shape == (1, 10, 10, 4)
    net.args.compute_metadata = False
    _, meta = net(torch.randn(1, 10, 32))
    assert "attn_mat" not in meta
    print_layer_diagnostics(net)
    assert len(get_model_sde_snapshots(net)) == 2


def _small_preset(name):
    from training.presets import make_args as preset_args
    return preset_args(name, d_e=32, d_k_total=32, d_v_total=32, n_heads=4, num_blocks=2)


def test_typed_backbone_matches_notebook_classes():
    """The presets' typed Backbone has the same parameters, init and outputs as the notebook constructors."""
    import torch.nn as nn
    from training.presets import build_backbone
    for preset in ("M2", "M3", "B1", "B2", "B4"):
        args = _small_preset(preset)
        torch.manual_seed(0)
        typed = build_backbone(args)
        torch.manual_seed(0)
        if preset.startswith("B"):
            old = TransformerNetwork(32, 64, 128, 4, args, num_blocks=2, Norm=nn.LayerNorm)
        else:
            old = RFATransformerNetwork(args, num_blocks=2, n_heads=4, input_dim=32, query_key_dim_total=32,
                                        value_dim_total=32, hidden_dim=128, Norm=nn.LayerNorm)
        sd_typed, sd_old = typed.state_dict(), old.state_dict()
        assert list(sd_typed) == list(sd_old), preset
        assert all(torch.equal(sd_typed[k], sd_old[k]) for k in sd_old), preset
        x = torch.randn(2, 10, 32)
        out_typed, infos = typed(x)
        out_old, _ = old(x)
        assert torch.equal(out_typed, out_old) and len(infos) == 2, preset


def test_backbone_metadata_per_block():
    from training.presets import build_backbone
    net = build_backbone(_small_preset("M2")).eval()
    _, infos = net(torch.randn(1, 10, 32), return_metadata=True)
    assert [info["attn_mat"].shape for info in infos] == [(1, 10, 10, 4)] * 2
    _, infos = net(torch.randn(1, 10, 32))
    assert "attn_mat" not in infos[0] and "mu_v" in infos[0]


def test_rfa_block():
    block = RFA_Block(make_args(M2), n_heads=4, input_dim=32, query_key_dim_total=32, value_dim_total=32)
    out, _ = block(torch.randn(2, 10, 32))
    assert out.shape == (2, 10, 32) and "layers.0.W_q.weight" in block.state_dict()


def test_load_legacy_checkpoint():
    """A pre-refactor state_dict (from the golden file) loads via load_legacy_state_dict; t_shift ones are refused."""
    from rfa import load_legacy_state_dict
    golden = torch.load(os.path.join(REPO, "tests", "golden", "golden.pt"), weights_only=False)
    net = rfa_net().double()
    load_legacy_state_dict(net, golden["M2"]["state"])
    out, _ = net(golden["M2"]["x"])
    assert torch.allclose(out, golden["M2"]["out"], rtol=1e-8, atol=1e-10)

    bad = dict(golden["M2"]["state"], **{"blocks.0.attn.t_shift_param": torch.zeros(1, 4)})
    try:
        load_legacy_state_dict(rfa_net(), bad)
    except ValueError:
        return
    raise AssertionError("checkpoint with t_shift_param should be refused")


def test_compiled_matches_eager():
    """In-place torch.compile of the backbone: same loss, float32-level gradient agreement, unchanged state_dict keys."""
    try:
        import triton  # noqa: F401
    except ImportError:
        print("  (Triton not installed, skipped)")
        return
    if not torch.cuda.is_available():
        print("  (no GPU, skipped)")
        return
    import copy
    from training.presets import build_backbone
    for make_net in (rfa_net, lambda: build_backbone(_small_preset("M2"))):  # notebook class, typed preset backbone
        torch.manual_seed(0)
        eager = LanguageModel(make_net(), vocab_size=100, embed_dim=32).cuda()
        comp = copy.deepcopy(eager)
        comp.backbone.compile()
        ids = torch.randint(0, 100, (2, 64), device="cuda")
        (le, ge), (lc, gc) = _lm_grads(eager, ids, True), _lm_grads(comp, ids, True)
        assert torch.allclose(le, lc, rtol=1e-4)
        assert all((ge[n] - gc[n]).norm() <= 1e-2 * ge[n].norm() + 1e-6 for n in ge)
        assert list(comp.state_dict()) == list(eager.state_dict())


def test_gpu_paper_size():
    """Forward/backward at the paper's size (d=256, 8 heads, 6 blocks, L=512) if a GPU is available."""
    if not torch.cuda.is_available():
        print("  (no GPU, skipped)")
        return
    net = rfa_net(d=256, heads=8, blocks=6).cuda()
    x = torch.randn(16, 512, 256, device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(3):
        out, _ = net(x)
        out.square().mean().backward()
    torch.cuda.synchronize()
    print(f"  fwd+bwd x3: {time.time() - t0:.2f}s, peak mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[pass]  {name}")
            except Exception as e:
                failed += 1
                print(f"[FAIL]  {name}: {type(e).__name__}: {e}")
    print(f"\n{failed} failing test(s)")
    sys.exit(1 if failed else 0)
