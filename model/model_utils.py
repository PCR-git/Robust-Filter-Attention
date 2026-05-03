import numpy as np
import torch
import torch.nn.functional as F
from utils import apply_interleaved_rope
from tqdm import tqdm

##########################################################################################
##########################################################################################

# def inv_softplus(target_val):
#     return torch.log(torch.exp(torch.tensor(target_val)) - 1.0 + 1e-6)

# def inv_sigmoid(target_val):
#     return torch.log(target_val/(1.0 - target_val))

def inv_softplus(x):
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    return torch.log(torch.expm1(x) + 1e-6)

def inv_sigmoid(x):
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    # Use the property that logit(x) = log(x / (1-x))
    return torch.log(x / (1.0 - x + 1e-8))

##########################################################################################
##########################################################################################

def compute_lambda(lambda_real_input, lambda_imag_input, args, scale_r, scale_i=1.0, epsilon=1e-5):
    """
    Computes stable eigenvalue pairs (lambda = -mu +/- i*omega) for each head.
    
    Ensures the system is Hurwitz (real part < 0) and captures oscillatory 
    modes via complex conjugate pairs.
    """
    
    device = lambda_real_input.device
    
    n_heads, d_half = lambda_imag_input.shape
    
#     if args.t_equal == True:
#         total_time = self.args.seq_len - 1 # (Unless unequal time intervals)
#     else:
#         total_time = t_measure[:,-1] - t_measure[:,0]

    if args.learn_decay == True:
#         # Compute Real Part (mu): Constrained to be positive for stability.
#         lambda_real = scale_r * F.softplus(lambda_real_input) + epsilon # Use softplus
#         lambda_real = scale_r * torch.exp(lambda_real_input) + epsilon # Use exponential
#         lambda_real = scale_r * lambda_real_input**2 + epsilon # Use square
        lambda_real = scale_r * torch.sigmoid(lambda_real_input) + epsilon # Use sigmoid
    else:
        lambda_real = scale_r * lambda_real_input

    # Compute Imaginary Part (+/- omega): Creates conjugate pairs.
    # Resulting shape: [n_heads, d_v_head]
    imag_parts = scale_i * torch.stack([lambda_imag_input, -lambda_imag_input], dim=-1)
    imag_parts = imag_parts.reshape(n_heads, 2 * d_half)

    return lambda_real, imag_parts

##########################################################################################
##########################################################################################

# def compute_lambda(lambda_real_input, lambda_imag_input, args, scale_c=10, scale_r=1, scale_i=1, epsilon=1e-5):
#     """
#     Computes stable eigenvalue pairs (lambda = -mu +/- i*omega) for each head.
    
#     Ensures the system is Hurwitz (real part < 0) and captures oscillatory 
#     modes via complex conjugate pairs.
#     """
    
#     device = lambda_real_input.device

#     # Normalize by sequence length to ensure a good scale during training.
#     # Note: If we test on a different sequence length, the mu we learned will no longer be consistent.
#     scale_r = 5*(scale_c/2) / (args.seq_len-1)
#     scale_i = 5*(scale_c/2) * 2*np.pi / (args.seq_len-1)
    
#     n_heads, d_half = lambda_imag_input.shape

#     # Compute Real Part (mu): Constrained to be positive for stability.
    
# #     lambda_real = torch.sigmoid(lambda_real_input) * scale_r + epsilon # Constrain to be positive
#     lambda_real = lambda_real_input**2 * scale_r + epsilon
# #     print(lambda_real)
# #     lambda_real = (torch.sigmoid(lambda_real_input) - 1.0) * scale_r - epsilon # Constrain to be negative
    
#     # Toggle for Unitary RFA limit (mu = 0), equivalent to RoPE generalization.
#     if args.lambda_real_zero == 1:
#         lambda_real = lambda_real*0

#     # Compute Imaginary Part (+/- omega): Creates conjugate pairs.
#     # Resulting shape: [n_heads, d_v_head]
#     imag_parts = torch.stack([lambda_imag_input, -lambda_imag_input], dim=-1) * scale_i  
#     imag_parts = imag_parts.reshape(n_heads, 2 * d_half)

#     return lambda_real, imag_parts

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
        dict with keys:
            query_key_dim, query_key_dim_total, value_dim, value_dim_total
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

# # Sample the model autoregressively
# def autoregressive_sample(model, start_seq, max_gen_len, t_measure=None, t_shift=None, t_equal=True, causal=True):
#     """
#     Performs autoregressive generation by treating the model as a sequential 
#     state estimator. 
    
#     Each step predicts the mean of the marginal distribution z_{si}^{+} 
#     based on the accumulated precision (P_tot) and dynamics (Phi) 
#     derived in Section 3.4.

#     Args:
#         model: The MultiheadIsotropicRFA model.
#         start_seq: Initial seed/context tensor (B, 1, L, d_e).
#         max_gen_len: Number of new tokens to generate.

#     Returns:
#         total_seq: The full sequence (seed + generated tokens).
#         new_seq: Only the generated tokens.
#     """
    
#     # Handles unequal time intervals:
#     if t_equal == True:
#         t_measure = None

#     with torch.no_grad():
#         model.eval()
#         current_seq = start_seq
#         total_seq = start_seq
        
#         window_size = start_seq.size(1)

#         for i in tqdm(range(max_gen_len)):
#             # Forward pass through the model. Performs filtering (RFA)
#             out, output_dict = model(current_seq, t_measure=t_measure, t_shift=t_shift, causal=causal) 

#             # Extract the prediction for t_{i+1} (the last token in the output sequence)
#             next_token = out[:, -1, :].unsqueeze(1)

#             # Append the prediction to the total sequence
#             total_seq = torch.cat([total_seq, next_token], dim=1)

#             # Update the sliding context window to maintain O(N^2) complexity.
#             # This 'current_seq' will be used as the history for the next step.
#             current_seq = total_seq[:, -window_size:, :]

#     # Isolate the generation portion (z_{s, i+1} ... z_{s, i+max_gen_len})
#     new_seq = total_seq[:, -max_gen_len:, :]

#     return total_seq, new_seq

def autoregressive_sample(model, start_seq, max_gen_len, t_measure=None, t_shift=None, t_equal=True, causal=True):
    """
    Performs autoregressive generation and collects total precision metadata.
    
    The precision reflects the accumulated scale of evidence at each step.
    High precision indicates the model is in a stable 'Integrative' regime, while
    low precision indicates 'High Innovation' or uncertainty.
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
            out, output_dict = model(current_seq, t_measure=t_measure, t_shift=t_shift, causal=causal) 

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

