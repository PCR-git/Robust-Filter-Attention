import numpy as np
import torch

##############################################
##############################################

def complex_matmul(A, B):
    """
    Complex matrix multiplication for tensors representing complex numbers.
    """
    
    # Get real and imaginary components
    A_real, A_imag = A[0], A[1]
    B_real, B_imag = B[0], B[1]
    
    # Compute real and imaginary parts of output
    real_part = torch.matmul(A_real, B_real) - torch.matmul(A_imag, B_imag)
    imag_part = torch.matmul(A_real, B_imag) + torch.matmul(A_imag, B_real)
    
    return torch.stack([real_part, imag_part], dim=0)

###############################################
###############################################

def apply_interleaved_rope(x, cos, sin):
    """
    x:   [B, L, H, D, 2] (Interleaved Query/Key/Value)
    cos: [1, L, 1, D]    (Precomputed Cosine)
    sin: [1, L, 1, D]    (Precomputed Sine)
    """
    # x[..., 0] is Real, x[..., 1] is Imag
    # Standard complex rotation: (R + iI) * (cos + i*sin)
    # = (R*cos - I*sin) + i(R*sin + I*cos)
    
    res_real = x[..., 0] * cos - x[..., 1] * sin
    res_imag = x[..., 0] * sin + x[..., 1] * cos
    
    # Return as [..., D, 2]
    return torch.stack([res_real, res_imag], dim=-1)
