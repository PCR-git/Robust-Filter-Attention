import numpy as np
import torch
import torch.nn as nn

from utils import complex_matmul

##########################################################################################
##########################################################################################
        
def initialize_linear_layers(m):
    if isinstance(m, nn.Linear):
        # Apply Xavier Uniform to the weight matrix
        nn.init.xavier_uniform_(m.weight)
        # Initialize bias to zero (standard practice)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

##########################################################################################
##########################################################################################

def init_complex_matrix(d_1, d_2, bias=False):
    """
    Isotropic initialization of a complex-valued matrix and optional bias.
    
    Returns:
        W: torch.Tensor of shape (2, 1, d_1, d_2) for weights
        b: torch.Tensor of shape (2, 1, d_2) for bias (if bias=True)
    """
    scale = np.sqrt(2 / (d_1 + d_2))
    mag = scale * torch.randn(d_1, d_2)
    phase = 2 * np.pi * torch.rand(d_1, d_2)

    real = mag * torch.cos(phase)
    imag = mag * torch.sin(phase)
    W = torch.stack([real, imag]).unsqueeze(1)  # (2, 1, d_1, d_2)

    if bias:
        mag_b = scale * torch.randn(d_2)
        phase_b = 2 * np.pi * torch.rand(d_2)
        real_b = mag_b * torch.cos(phase_b)
        imag_b = mag_b * torch.sin(phase_b)
        b = torch.stack([real_b, imag_b]).unsqueeze(1)  # (2, 1, d_2)
        return W, b

    return W

##########################################################################################
##########################################################################################
  
def init_complexlinear(linear_layer, weight_tensor, layer_type):
    """
    Initializes a Complex Linear Layer from complex weight (and optional bias) tensors.

    weight_tensor: shape (2, 1, d_in, d_out)
    bias_tensor: shape (2, 1, d_out)
    """
    real_w = weight_tensor[0, 0].T  # (d_out, d_in)
    imag_w = weight_tensor[1, 0].T
    if layer_type == 'in':
        W = torch.cat((real_w,imag_w),axis=0)
    else:
        W = torch.cat((real_w,imag_w),axis=1) / np.sqrt(2)

    with torch.no_grad():
        linear_layer.weight.copy_(W)
        
        if linear_layer.bias is not None:
            nn.init.constant_(linear_layer.bias, 0.0)
            
##########################################################################################
##########################################################################################

def init_rope(n_heads, dim_target, base=10000.0):
    """
    RoPE initialization
    """

    omega_init = 1.0 / (base ** (torch.arange(dim_target).float() / dim_target))
    omega_init_stacked = omega_init.repeat(n_heads).view(n_heads, dim_target)
    
    return omega_init_stacked

##########################################################################################
##########################################################################################

# def init_spectrally_coupled_rope(n_heads, dim_target, b=4.0, zero_frac=0.25, base=10000.0):
#     """
#     Initializes frequencies and decays in a coupled, sharded manner.
#     Heads are ordered from SLOWEST (Head 0) to FASTEST (Head 7).
#     """
#     # Create Master Spectrum (Fastest to Slowest initially)
#     master_dim = n_heads * dim_target
#     omega_master = 1.0 / (base ** (torch.arange(master_dim).float() / master_dim))
    
#     # Flip to get Slowest to Fastest
#     omega_master = torch.flip(omega_master, dims=[0])
    
#     # Shard so Head 0 gets the lowest frequencies
#     omega_sharded = omega_master.view(n_heads, dim_target)
    
#     # Calculate Decays based on the Max Frequency per head
#     # After flipping, the max omega for each head is at the end of the shard
#     max_omegas = omega_sharded[:, -1]
#     total_mus = b * max_omegas   
#     # b = 1.0 corresponds to the critically damped regime
    
#     # Apply Zero Fraction to the SLOWEST heads (now at the start)
#     n_unitary = int(n_heads * zero_frac)
#     if n_unitary > 0:
#         total_mus[:n_unitary] = 0.0
        
#     return omega_sharded, total_mus

def init_spectrally_coupled_rope(n_heads, dim_target, zero_frac=0.0, base=10000.0, b=5.0, c=2.0, L=512):
    """
    Initializes frequencies and decays in a coupled, sharded manner.
    Heads are ordered from SLOWEST (Head 0) to FASTEST (Head 7).
    """
    # Create Master Spectrum (Fastest to Slowest initially)
    master_dim = n_heads * dim_target
    omega_master = 1.0 / (base ** (torch.arange(master_dim).float() / master_dim))
    
    # Flip to get Slowest to Fastest
    omega_master = torch.flip(omega_master, dims=[0])
    
    # Shard so Head 0 gets the lowest frequencies
    omega_sharded = omega_master.view(n_heads, dim_target)
    
    # Calculate Decays based on the Max Frequency per head
    # After flipping, the max omega for each head is at the end of the shard
    max_omegas = omega_sharded[:, -1]
    total_mus = b * max_omegas   
    # b = 1.0 corresponds to the critically damped regime
    
    # Apply Zero Fraction to the SLOWEST heads (now at the start)
    n_unitary = int(n_heads * zero_frac)
    if n_unitary > 0:
        total_mus[:n_unitary] = 0.0
        
#     mu_floor = c / L
#     total_mus = torch.maximum(b * max_omegas, torch.full_like(max_omegas, mu_floor))
        
    return omega_sharded, total_mus



# def init_spectrally_coupled_rope_V2(
#     n_heads,
#     dim_target,
#     n_zero_heads=0,
#     base=10000.0,
#     b=5.0,
# ):
#     """
#     Spectrally-coupled RoPE with optional zero-frequency heads.

#     - First n_zero_heads have ω = 0
#     - Remaining heads use standard RoPE spectrum, compressed to fit remaining dims
#     - Preserves exact RoPE functional form and spacing

#     Heads are ordered from SLOWEST → FASTEST.
#     """

#     total_dim = n_heads * dim_target
#     zero_dim = n_zero_heads * dim_target
#     nonzero_dim = total_dim - zero_dim

#     assert nonzero_dim > 0, "Need at least one nonzero-frequency head"

#     # --- Build standard RoPE spectrum over reduced dimension
#     idx = torch.arange(nonzero_dim).float()
#     omega_nonzero = 1.0 / (base ** (idx / nonzero_dim))

#     # Flip so slow → fast (same as your original)
#     omega_nonzero = torch.flip(omega_nonzero, dims=[0])

#     # --- Prepend zeros
#     omega_master = torch.cat([
#         torch.zeros(zero_dim),
#         omega_nonzero
#     ])

#     # --- Reshape into heads
#     omega_sharded = omega_master.view(n_heads, dim_target)

#     # --- SC-RFA coupling
#     max_omegas = omega_sharded[:, -1]
#     total_mus = b * max_omegas

#     return omega_sharded, total_mus

##########################################################################################
##########################################################################################

# Log linear initialization of decay per head
def init_decay_per_head(n_heads, min_decay=0.001, max_decay=5.0, zero_frac=0.25):

    # Assuming self.n_heads is 8
    n_unitary = int(n_heads * zero_frac) # Fraction of heads fixed at zero
    n_decay = n_heads - n_unitary

    # Fix some heads at exactly zero
    unitary_mus = torch.zeros(n_unitary)
    
    # Linear spread in log space
    log_mus = torch.linspace(np.log(min_decay), np.log(max_decay), n_decay)

    # Log-linear spread for the filtering heads
    decay_mus = torch.exp(log_mus)

    # Combine and register
    total_mus = torch.cat([unitary_mus, decay_mus])
    
    return total_mus

##########################################################################################
##########################################################################################

def init_linear_bias_slopes(n_heads):
    """
    Computes the geometric slope sequence introduced in the ALiBi paper.
    For n heads, it creates a sequence of slopes from 2^-8/n to 2^-8.
    """
    def get_slopes_power_of_2(n):
        start = (2**(-8/n))
        ratio = start
        return [start*ratio**i for i in range(n)]
    
    slopes = torch.Tensor(get_slopes_power_of_2(n_heads))
    
    # Inverse Softplus: y = log(exp(x) - 1)
    # We add a small epsilon to ensure the log argument is > 0
    # and clamp to avoid log(0) for extremely small slopes.
    eps = 1e-9
    initial_sigma_raw = torch.log(torch.exp(slopes) - 1.0 + eps)
    
    return initial_sigma_raw

##########################################################################################
##########################################################################################

def initialize_to_correct_model(module, D1, S1, Si1, sigma_process, sigma_measure, args):
    """
    Initialize to correct model parameter values (for testing)
    """

    with torch.no_grad():

        module.W_q.weight[0:2,0:2].copy_(Si1[0])
        module.W_q.weight[128:130,0:2].copy_(Si1[1])
        module.W_k.weight[0:2,0:2].copy_(Si1[0])
        module.W_k.weight[128:130,0:2].copy_(Si1[1])
        module.W_v.weight[0:2,0:2].copy_(Si1[0])
        module.W_v.weight[128:130,0:2].copy_(Si1[1])
        module.W_o.weight[0:2,0:2].copy_(S1[0])
        module.W_o.weight[0:2,128:130].copy_(-S1[1])
        
    print('Model initialized to match true dynamics.')
