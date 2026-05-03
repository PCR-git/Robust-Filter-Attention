import numpy as np
import torch
import torch.nn as nn

from utils import complex_matmul
from model import get_complex_weights

##########################################################################################
##########################################################################################

class Complex_MSE_Loss(nn.Module):
    """
    Mean Square Error Loss for complex-valued arrays
    """
    def __init__(self):
        super().__init__()

    def forward(self, out, target):
        real_diff = out[0] - target[0] # Error of real part
        imag_diff = out[1] - target[1] # Error of imaginary part
        squared_error = real_diff**2 + imag_diff**2
        mean_SE = torch.mean(squared_error) # Sum square error of real and imaginary parts and take mean
#         max_SE = torch.max(squared_error) # Sum square error of real and imaginary parts and take mean
        return mean_SE
#         return mean_SE + max_SE

##########################################################################################
##########################################################################################

class Batched_Complex_MSE_Loss(nn.Module):
    """
    Batched Mean Square Error Loss for complex-valued arrays
    """
    def __init__(self):
        super().__init__()

    def forward(self, out, target):
        real_diff = out[:,0] - target[:,0] # Error of real part
        imag_diff = out[:,1] - target[:,1] # Error of imaginary part
        return torch.mean(real_diff**2 + imag_diff**2) # Sum square error of real and imaginary parts and take mean

def inverse_penalty(model, loss_p, args):
    """
    Penalty to keep the product of W_v and W_p close to the identity matrix.
    Assumes each module has W_v and W_p as ComplexLinearLayer or ComplexLinearHermitian.
    """
    penalty = 0.0
    num_layers = 0

    for name, module in model.named_modules():
        if hasattr(module, 'W_v') and hasattr(module, 'W_p'):
            Wv = module.W_v
            Wp = module.W_p

            # Extract real and imag parts of weights
            Wv_r, Wv_i = Wv.real.weight, Wv.imag.weight
            Wp_r, Wp_i = Wp.real.weight, Wp.imag.weight

            # Stack as (2, *, *) for complex_matmul
            Wv_mat = torch.stack((Wv_r, Wv_i), dim=0)
            Wp_mat = torch.stack((Wp_r, Wp_i), dim=0)

            # Product W_v @ W_p (or W_p @ W_v depending on dimensionality)
            prod = complex_matmul(Wv_mat, Wp_mat)

            # Compare to complex identity
            layer_penalty = loss_p(prod, module.complex_identity)
            penalty += layer_penalty
            num_layers += 1

    if num_layers > 0:
        penalty = penalty / num_layers

    return penalty

##########################################################################################
##########################################################################################

def inverse_net_penalty(model, loss_p, args):
    """
    Penalty to keep the product of W_p and W_v near the identity matrix.
    """
    penalty = 0.0
    num_layers = 0

    for name, module in model.named_modules():
        if hasattr(module, 'W_p') and hasattr(module, 'W_v'):
            # Extract complex weights from ComplexLinear layers            
            W_v = get_complex_weights(module, 'W_v')
            W_p = get_complex_weights(module, 'W_p')

            # Compute product (W_v @ W_p), assuming shape: (2, out_dim, in_dim)
            prod = complex_matmul(W_v, W_p)

            # Penalty = distance from identity
#             layer_penalty = loss_p(prod, module.complex_identity)
            layer_penalty = loss_p(prod[0], module.complex_identity[0].squeeze(0)) + loss_p(prod[1], module.complex_identity[1].squeeze(0))
        
            penalty += layer_penalty
            num_layers += 1

    if num_layers > 0:
        penalty /= num_layers

    return penalty

##########################################################################################
##########################################################################################

def lambda_L1_penalty(model, args):
    """
    """
    penalty = 0.0
    num_layers = 0

    for name, module in model.named_modules():
        if hasattr(module, 'lambda1'):
#         if hasattr(module, 'lambda1') and hasattr(module, 'lambda_Omega'):
            
            lambda1_penalty = torch.norm(module.lambda1, 1)
#             lambda_Omega_penalty = torch.norm(module.lambda_Omega, 1)
#             lambda_Gamma_penalty = torch.norm(module.lambda_Gamma, 1)
            
            layer_penalty = lambda1_penalty
#             layer_penalty = lambda1_penalty + lambda_Omega_penalty
            penalty += layer_penalty
            num_layers += 1
            
    if num_layers > 0:
        penalty /= num_layers
    
    return penalty
            
            