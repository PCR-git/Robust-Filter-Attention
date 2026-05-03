import torch
import torch.nn as nn
from tqdm import tqdm

##########################################################################################
##########################################################################################

def single_epoch_rfa_lm(model, train_loader, history, optimizer, criterion, args, scheduler=None):
    """
    Standard Next-Step Language Modeling training epoch.
    Handles internal token shifting and CrossEntropy loss.
    """
    model.train()
    vocab_size = model.lm_head.out_features

    pbar = tqdm(train_loader, desc="Training", leave=False, mininterval=5.0)
    for it, batch in enumerate(pbar):
        batch = {k: v.to(args.device) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        optimizer.zero_grad()

        # Forward pass (t_measure=None triggers the backbone to use arange(L))
        logits, output_dict = model(input_ids, t_measure=None, causal=True)

        # --- NEXT-STEP LOGIC SHIFT ---
        # Shift logits: we predict tokens 1 to N using tokens 0 to N-1
        # Logits shape: [Batch, Seq_Len, Vocab] -> [Batch, Seq_Len - 1, Vocab]
        shift_logits = logits[:, :-1, :].contiguous()

        # Shift labels: tokens 1 to N are the targets
        # Labels shape: [Batch, Seq_Len] -> [Batch, Seq_Len - 1]
        shift_labels = labels[:, 1:].contiguous()

        # Flatten for CrossEntropy: (Batch * (Seq_Len-1), Vocab) vs (Batch * (Seq_Len-1))
        loss = criterion(
            shift_logits.view(-1, vocab_size), 
            shift_labels.view(-1)
        )

        loss.backward()

        # Clip Gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # SDE stability clipping
        sde_params = [p for n, p in model.named_parameters() 
                      if any(k in n for k in ['mu_', 'sigma_', 'eta_', 'gamma_'])]
        if sde_params:
            torch.nn.utils.clip_grad_norm_(sde_params, max_norm=1e-4)

        optimizer.step()
        if scheduler:
            scheduler.step()
            
        if it % 100 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # LOGGING
        history['loss'].append(loss.item())

        # Track physical parameters if available in output_dict
        if 'mu_v' in output_dict:
            # Capture the 1D noise params (mean across batch, but keep per-head)
            history['mu'].append(output_dict['mu_v'].detach().cpu().numpy())
            history['sigma'].append(output_dict['sigma_sq_v'].detach().cpu().numpy())
            history['sigma_tilde'].append(output_dict['sigma_tilde_sq_v'].detach().cpu().numpy())
            history['eta'].append(output_dict['eta_sq_v'].detach().cpu().numpy())
            history['gamma'].append(output_dict['gamma_sq_v'].detach().cpu().numpy())

            # Capture inverse temp and robustness param
            history['tau'].append(output_dict['tau'].detach().cpu().numpy())
            history['nu_over_d'].append(output_dict['nu_over_d'].detach().cpu().numpy())

    return output_dict, history

##########################################################################################
##########################################################################################

def single_epoch_standard_lm(model, train_loader, history, optimizer, criterion, args, scheduler=None):
    """
    Standard Next-Step Language Modeling training epoch for a Vanilla Transformer.
    """
    model.train()
    vocab_size = model.lm_head.out_features

    pbar = tqdm(train_loader, desc="Training (Baseline)", leave=False, mininterval=5.0)
    for it, batch in enumerate(pbar):
        batch = {k: v.to(args.device) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        optimizer.zero_grad()

        # Forward pass (Standard Transformer ignores t_measure/t_shift)
        logits, attn_weights = model(input_ids)

        # --- NEXT-STEP LOGIC SHIFT ---
        # Shift logits: we predict tokens 1 to N using tokens 0 to N-1
        shift_logits = logits[:, :-1, :].contiguous()

        # Shift labels: tokens 1 to N are the targets
        shift_labels = labels[:, 1:].contiguous()

        # Compute Cross-Entropy
        loss = criterion(
            shift_logits.view(-1, vocab_size), 
            shift_labels.view(-1)
        )

        loss.backward()

        # Standard Global Gradient Clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        
        if scheduler:
            scheduler.step()
            
        if it % 100 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # LOGGING
        history['loss'].append(loss.item())
        
        if it % 10 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return attn_weights, history

