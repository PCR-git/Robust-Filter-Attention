import math

import torch
import torch.nn as nn
from tqdm import tqdm

##########################################################################################
##########################################################################################

# Per-head RFA parameters logged to `history` (output-dict key -> history key)
_RFA_LOG_KEYS = {"mu_v": "mu", "sigma_sq_v": "sigma", "sigma_tilde_sq_v": "sigma_tilde",
                 "eta_sq_v": "eta", "gamma_sq_v": "gamma", "tau": "tau", "nu_over_d": "nu_over_d"}


def single_epoch_lm(model, train_loader, history, optimizer, args, scheduler=None, log_every=100):
    """
    One epoch of next-token language-model training (RFA or baseline backbone).
    Appends the loss of every step to history['loss'], and for RFA models, every `log_every`
    steps, the per-head SDE parameters of the last block (history['mu'], ['sigma'], ...).
    """
    model.train()
    output_dict = None

    pbar = tqdm(train_loader, desc="Training", leave=False, mininterval=5.0)
    for it, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)

        optimizer.zero_grad()
        loss, output_dict = model.loss(input_ids, labels, causal=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        history['loss'].append(loss.item())
        if it % 100 == 0:
            pbar.set_postfix({"loss": f"{history['loss'][-1]:.4f}"})

        # Backbone returns one info dict per block; the notebooks' RFATransformerNetwork only the last one
        info = output_dict[-1] if isinstance(output_dict, list) else output_dict
        if it % log_every == 0 and isinstance(info, dict) and "mu_v" in info:
            for key, hist_key in _RFA_LOG_KEYS.items():
                history.setdefault(hist_key, []).append(info[key].detach().cpu().numpy())

    return output_dict, history

##########################################################################################
##########################################################################################

@torch.no_grad()
def evaluate_lm(model, loader, args):
    """Token-weighted mean next-token loss and perplexity over `loader`: (avg_nll, ppl)."""
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)
        loss, _ = model.loss(input_ids, labels, causal=True)
        n = (labels[:, 1:] != -100).sum().item()
        total_nll += loss.item() * n
        total_tokens += n
    avg_nll = total_nll / total_tokens
    return avg_nll, math.exp(avg_nll)
