import torch
import torch.nn as nn

# Frequency / decay initializers live in the rfa package; re-exported here for the baseline layers.
from rfa.init import init_rope, init_spectrally_coupled_rope, init_decay_per_head

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

def init_linear_bias_slopes(n_heads):
    """
    Computes the geometric slope sequence introduced in the ALiBi paper,
    returned in inverse-softplus space.
    For n heads, it creates a sequence of slopes from 2^-8/n to 2^-8.
    """
    def get_slopes_power_of_2(n):
        start = (2**(-8/n))
        ratio = start
        return [start*ratio**i for i in range(n)]

    slopes = torch.Tensor(get_slopes_power_of_2(n_heads))

    # Inverse Softplus: y = log(exp(x) - 1)
    # We add a small epsilon to ensure the log argument is > 0
    eps = 1e-9
    initial_sigma_raw = torch.log(torch.exp(slopes) - 1.0 + eps)

    return initial_sigma_raw
