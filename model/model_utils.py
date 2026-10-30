import torch
from tqdm import tqdm

from rfa import MultiheadIsotropicRFA

##########################################################################################
##########################################################################################

def resolve_multihead_dims(n_heads, query_key_dim=None, value_dim=None, query_key_dim_total=None, value_dim_total=None):
    """
    Standardizes dimensions for Multi-Head Attention, ensuring total
    embedding sizes are compatible with the head count.

    You must supply either:
        - query_key_dim (per-head) or query_key_dim_total (total), but not both, and
        - value_dim (per-head) or value_dim_total (total), but not both.

    Returns:
        (query_key_dim, value_dim, query_key_dim_total, value_dim_total)
    """
    # Validate head count
    if n_heads <= 0 or not isinstance(n_heads, int):
        raise ValueError(f"n_heads must be a positive integer, got {n_heads}")

    # Query/key dims
    if (query_key_dim is not None) and (query_key_dim_total is not None):
        raise ValueError("Specify either query_key_dim or query_key_dim_total, not both.")
    if (query_key_dim is None) and (query_key_dim_total is None):
        raise ValueError("Must specify one of query_key_dim or query_key_dim_total.")

    if query_key_dim is not None:
        query_key_dim_total = n_heads * query_key_dim
    else:
        if query_key_dim_total % n_heads != 0:
            raise ValueError(f"query_key_dim_total={query_key_dim_total} is not divisible by n_heads={n_heads}")
        query_key_dim = query_key_dim_total // n_heads

    # Value dims
    if (value_dim is not None) and (value_dim_total is not None):
        raise ValueError("Specify either value_dim or value_dim_total, not both.")
    if (value_dim is None) and (value_dim_total is None):
        raise ValueError("Must specify one of value_dim or value_dim_total.")

    if value_dim is not None:
        value_dim_total = n_heads * value_dim
    else:
        if value_dim_total % n_heads != 0:
            raise ValueError(f"value_dim_total={value_dim_total} is not divisible by n_heads={n_heads}")
        value_dim = value_dim_total // n_heads

    return query_key_dim, value_dim, query_key_dim_total, value_dim_total

##########################################################################################
##########################################################################################

def autoregressive_sample(model, start_seq, max_gen_len, t_measure=None, t_equal=True, causal=True):
    """
    Performs autoregressive generation and collects total precision metadata.

    The precision reflects the accumulated scale of evidence at each step.
    High precision indicates the model is in a stable 'Integrative' regime, while
    low precision indicates 'High Innovation' or uncertainty.
    (Requires args.compute_metadata = True so that 'P_tot' is returned.)
    """
    if t_equal:
        t_measure = None

    precisions = []

    with torch.no_grad():
        model.eval()
        current_seq = start_seq
        total_seq = start_seq
        window_size = start_seq.size(1)

        for i in tqdm(range(max_gen_len)):
            # Forward pass: Filters context using the SDE prior
            out, output_dict = model(current_seq, t_measure=t_measure, causal=causal)

            # Extract log-precision (P_tot) for the latest token
            # P_tot shape is [B, L, H]. We take the last index and mean across heads.
            p_step = torch.mean(output_dict['P_tot'][:, -1, :], dim=-1) # [B]
            precisions.append(p_step)

            # Extract prediction for t_{i+1}
            next_token = out[:, -1, :].unsqueeze(1)
            total_seq = torch.cat([total_seq, next_token], dim=1)
            current_seq = total_seq[:, -window_size:, :]

    new_seq = total_seq[:, -max_gen_len:, :]

    return total_seq, new_seq, precisions

##########################################################################################
##########################################################################################

def get_rfa_layers(model):
    return [m for m in model.modules() if isinstance(m, MultiheadIsotropicRFA)]


def get_model_sde_snapshots(model):
    """Per-layer dict of the physical per-head SDE parameters (numpy arrays)."""
    snapshot = {}
    with torch.no_grad():
        for layer_idx, module in enumerate(get_rfa_layers(model)):
            p = module.sde_params()
            summary = module._param_summary(p)
            snapshot[f"layer_{layer_idx}"] = {
                "mu": summary["mu_v"].cpu().numpy(),
                "sigma_sq": summary["sigma_sq_v"].cpu().numpy(),
                "sigma_tilde_sq": summary["sigma_tilde_sq_v"].cpu().numpy(),
                "eta_sq": summary["eta_sq_v"].cpu().numpy(),
                "gamma_sq": summary["gamma_sq_v"].cpu().numpy(),
                "tau": summary["tau"].cpu().numpy(),
                "nu_over_d": summary["nu_over_d"].cpu().numpy(),
            }
    return snapshot

##########################################################################################
##########################################################################################

def print_layer_diagnostics(model):
    """Table of per-head decay, noise, filtering regime (alpha = eta^2 - sigma~^2) and horizon 1/mu."""
    print(f"\n{'='*100}")
    print(f"{'Layer':<6} | {'Head':<4} | {'mu':<8} | {'sigma^2':<10} | {'eta^2':<10} | {'sigma^2/2mu':<12} | {'Alpha':<10} | {'Horizon'}")
    print("-" * 100)

    for layer_idx, module in enumerate(get_rfa_layers(model)):
        with torch.no_grad():
            s = module._param_summary(module.sde_params())
        for h in range(module.config.n_heads):
            mu, sig, eta, steady = (s[k][h].item() for k in ("mu_v", "sigma_sq_v", "eta_sq_v", "sigma_tilde_sq_v"))
            if mu < 1e-7:
                steady_str, alpha_str, horizon = "N/A", "N/A", "Inf"
            else:
                steady_str, alpha_str, horizon = f"{steady:<12.4f}", f"{eta - steady:<10.4f}", f"{1.0 / mu:.1f}"
            print(f"L{layer_idx:<4} | H{h:<3} | {mu:<8.4f} | {sig:<10.4f} | {eta:<10.4f} | {steady_str:<12} | {alpha_str:<10} | {horizon}")
        print("-" * 100)
    print(f"{'='*100}\n")
