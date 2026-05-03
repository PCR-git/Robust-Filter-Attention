import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import math

from utils import complex_exp, batched_complex_matmul, batched_complex_hadamard, batched_complex_matmul_full
from precision_attention import compute_residual_norm, compute_exp_kernel, compute_covariance_kernel
from model import init_complexlinear, init_complex_matrix, compute_lambda_h, compute_lambda_shared
from model import RealPositionalEncoding, ComplexPositionalEncoding, LearnedRealPositionalEncoding, LearnedComplexPositionalEncoding, RoPE

##########################################################################################
##########################################################################################

#################################### OTHER LAYERS ####################################

##########################################################################################
##########################################################################################
    
# --- Small network to generate data-dependent lambda parameters ---
class DataDependentLambdaGenerator(nn.Module):
    """
    A small neural network that takes a sequence of vectors and generates
    the lambda parameters for the attention mechanism.

    The network first pools the sequence to get a fixed-size representation,
    then uses a few linear layers to map this representation to the required
    lambda parameters.
    """
    def __init__(self, n_heads, embed_dim, d_v_head, d_k_head, sep_params, epsilon=1e-5):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.d_v_head = d_v_head
        self.sep_params = sep_params

        # The total number of parameters to output
        total_output_size = 0
        total_output_size += n_heads  # lambda_real_v
        total_output_size += n_heads * (d_v_head // 2)  # lambda_imag_v
        total_output_size += n_heads  # lambda_omega_sqrt_v
        total_output_size += n_heads  # lambda_gamma_sqrt_v
        
        if self.sep_params == 1:
            total_output_size += n_heads  # lambda_real_k
            total_output_size += n_heads * (d_k_head // 2)  # lambda_imag_k
            total_output_size += n_heads  # lambda_omega_sqrt_k
            total_output_size += n_heads  # lambda_gamma_sqrt_k

        # Small MLP to generate the parameters
        self.generator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, total_output_size)
        )
        
        self.epsilon = epsilon

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
        
        Returns:
            A dictionary containing the generated lambda parameters with the
            correct shapes.
        """
        # Pool the sequence to a single vector per batch item
        pooled_x = torch.mean(x, dim=-2).squeeze(-2)  # Shape: (batch_size, embed_dim)

        # Generate a single tensor containing all parameters
        params_flat = self.generator(pooled_x)

        # Split the tensor into the individual parameters and reshape
        current_idx = 0
        
        # lambda_real_v
        size_real_v = self.n_heads
        lambda_real_v = params_flat[:, current_idx:current_idx + size_real_v]
        current_idx += size_real_v

        # lambda_imag_v
        size_imag_v = self.n_heads * (self.d_v_head // 2)
        lambda_imag_v = params_flat[:, current_idx:current_idx + size_imag_v].view(
            -1, self.n_heads, self.d_v_head // 2
        )
        current_idx += size_imag_v
        
        # lambda_omega_sqrt_v
        size_omega_v = self.n_heads
        lambda_omega_sqrt_v = params_flat[:, current_idx:current_idx + size_omega_v]
        current_idx += size_omega_v
        
        # lambda_gamma_sqrt_v
        size_gamma_v = self.n_heads
        lambda_gamma_sqrt_v = params_flat[:, current_idx:current_idx + size_gamma_v] + self.epsilon
        current_idx += size_gamma_v

        # Conditional parameters for 'k' if sep_params is enabled
        if self.sep_params == 1:
            # lambda_real_k
            size_real_k = self.n_heads
            lambda_real_k = params_flat[:, current_idx:current_idx + size_real_k]
            current_idx += size_real_k
            
            # lambda_imag_k
            size_imag_k = self.n_heads * (self.d_k_head // 2)
            lambda_imag_k = params_flat[:, current_idx:current_idx + size_imag_k].view(
                -1, self.n_heads, self.d_k_head // 2
            )
            current_idx += size_imag_k

            # lambda_omega_sqrt_k
            size_omega_k = self.n_heads
            lambda_omega_sqrt_k = params_flat[:, current_idx:current_idx + size_omega_k]
            current_idx += size_omega_k

            # lambda_gamma_sqrt_k
            size_gamma_k = self.n_heads
            lambda_gamma_sqrt_k = params_flat[:, current_idx:current_idx + size_gamma_k] + self.epsilon
            current_idx += size_gamma_k
        else:
            # Return dummy tensors if sep_params is not 1
            lambda_real_k, lambda_imag_k = None, None
            lambda_omega_sqrt_k, lambda_gamma_sqrt_k = None, None

        return {
            "lambda_real_v": lambda_real_v,
            "lambda_imag_v": lambda_imag_v,
            "lambda_omega_sqrt_v": lambda_omega_sqrt_v,
            "lambda_gamma_sqrt_v": lambda_gamma_sqrt_v,
            "lambda_real_k": lambda_real_k,
            "lambda_imag_k": lambda_imag_k,
            "lambda_omega_sqrt_k": lambda_omega_sqrt_k,
            "lambda_gamma_sqrt_k": lambda_gamma_sqrt_k,
        }
    
##########################################################################################
##########################################################################################

class MagnitudeProjection(nn.Module):
    """
    A PyTorch module that calculates the magnitude of the input tensor along
    the last dimension, projects it back to the original dimension using a
    linear layer, and adds the result to the input.

    This can be used to inject information about the overall vector magnitude
    into the vector itself.

    Args:
        input_dim (int): The dimension of the input tensor's last axis.
        device (torch.device): The device (e.g., 'cuda', 'cpu') to place
                                the linear layer on.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        # The linear layer takes a scalar magnitude (1) and projects it
        # to the input's dimension.
        self.mag_proj = nn.Linear(1, input_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Performs the magnitude projection and addition.

        Args:
            inputs (torch.Tensor): The input tensor with shape (..., input_dim).

        Returns:
            torch.Tensor: The modified tensor with the projected magnitude
                          added, with the same shape as the input.
        """
        # Calculate the magnitude of the input vector along the last dimension.
        # This results in a tensor of shape (..., 1).
        mag = torch.sqrt(torch.sum(inputs**2, dim=-1, keepdim=True))

        # Pass the scalar magnitude through a linear layer to
        # project it back to the original dimension.
        mag_proj_output = self.mag_proj(mag)

        # Add the projected magnitude back to the original input.
        # PyTorch's broadcasting will handle the addition correctly.
        output = inputs + mag_proj_output

        return output
    
##########################################################################################
##########################################################################################       

class MagnitudeTransform(nn.Module):
    """
    A PyTorch module that can calculate the magnitude of an input tensor and
    project it back to the original dimension. This is useful for scenarios
    where you want to inject magnitude information into a tensor, either
    in a single step or across intermediate layers.

    Args:
        input_dim (int): The dimension of the input tensor's last axis.
        device (torch.device): The device (e.g., 'cuda', 'cpu') to place
                               the linear layers on.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        # Linear layer for the multiplicative component of the projection.
        self.mag_mul = nn.Linear(1, input_dim)
        # Linear layer for the additive component of the projection.
        self.mag_add = nn.Linear(1, input_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Performs the magnitude calculation and projection in a single step,
        maintaining the original functionality of the class.

        Args:
            inputs (torch.Tensor): The input tensor with shape (..., input_dim).

        Returns:
            torch.Tensor: The modified tensor with the projected magnitude
                          added, with the same shape as the input.
        """
        mag = self.get_magnitude(inputs)
        return self.add_projection(inputs, mag)

    def get_magnitude(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Calculates and returns the magnitude of the input tensor along
        the last dimension.

        Args:
            inputs (torch.Tensor): The input tensor with shape (..., input_dim).

        Returns:
            torch.Tensor: A tensor of shape (..., 1) containing the magnitudes.
        """
        return torch.sqrt(torch.sum(inputs**2, dim=-1, keepdim=True))

    def add_projection(self, inputs: torch.Tensor, magnitude: torch.Tensor) -> torch.Tensor:
        """
        Adds the projected magnitude back to the input tensor. This version
        uses both a multiplicative and an additive linear layer.

        Args:
            inputs (torch.Tensor): The tensor to add the projection to,
                                   with shape (..., input_dim).
            magnitude (torch.Tensor): The pre-calculated magnitude tensor
                                      with shape (..., 1).

        Returns:
            torch.Tensor: The modified tensor with the projected magnitude
                          added, with the same shape as the input.
        """
        # Pass the scalar magnitude through both linear layers
        mag_mul_output = self.mag_mul(magnitude)
        mag_add_output = self.mag_add(magnitude)

        # Apply the multiplicative and additive components to the input
        output = inputs * mag_mul_output + mag_add_output

        return output
    
##########################################################################################
##########################################################################################
    
def predict_multiple_steps(lambda_h, est_eigenbasis, W_p, t_forward):
    """
    Given a sequence of time steps t_forward, make multiple future predictions 
    """

    mag_f, phase_f = complex_exp_v2(lambda_h*t_forward.unsqueeze(0).unsqueeze(2).unsqueeze(3))  # Forward
    mat_exp_f = mag_f*phase_f
    preds_p = batched_complex_hadamard_full(mat_exp_f.unsqueeze(0), est_eigenbasis[:,:,-1,:,:].unsqueeze(2))
    preds = batched_complex_matmul_full(W_p.unsqueeze(0), preds_p)
    
    return preds

##########################################################################################
##########################################################################################

def get_complex_weights(module, layer_name):
    """
    Extracts and stacks the real and imaginary weights from a ComplexLinearLayer
    or ComplexLinearHermitianLayer.

    Parameters:
        module (nn.Module): The module containing the layers.
        layer_name (str): Name of the ComplexLinear layer (e.g., 'W_p').

    Returns:
        Complex weight tensor of shape (2, out_dim, in_dim).
    """

    layer = getattr(module, layer_name)

    try:
        # Regular ComplexLinearLayer
        real_weight = layer.real.weight
        imag_weight = layer.imag.weight
    except:
        # Unwrap source layer weights
        real_weight = layer.source.real.weight.T
        imag_weight = -layer.source.imag.weight.T

    W = torch.stack([real_weight, imag_weight], dim=0)

    return W
