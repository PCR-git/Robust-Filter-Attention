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
