import numpy as np
import torch
import torch.nn as nn

from rfa import RFAConfig, MultiheadIsotropicRFA
from .model_utils import resolve_multihead_dims
from .layers import MultiHeadAttentionLayer

##########################################################################################
##########################################################################################

class SelfAttentionBlock(nn.Module):
    """
    A minimal wrapper for a single multi-head attention layer.
    Removes residuals, norms, and FFN to isolate attention performance.
    """

    def __init__(self, input_dim, qkv_dim, num_heads, args):
        super(SelfAttentionBlock, self).__init__()

        # Baseline attention (B1-B4, selected by the args flags)
        self.attn = MultiHeadAttentionLayer.from_args(input_dim, qkv_dim, num_heads, args)

    def forward(self, x, t_measure=None, causal=True):
        """
        Forward pass directly into the attention layer.
        """
        # We accept t_measure as a dummy variable to match the RFA API
        # but we ONLY use causal for the actual baseline attention logic.
        
        # We pass x as Q, K, and V for self-attention
        out, attn_weights = self.attn(x, x, x, causal=causal)

        return out, attn_weights
    
##########################################################################################
##########################################################################################

def build_rfa_layer(args, n_heads, input_dim, query_key_dim=None, value_dim=None, query_key_dim_total=None, value_dim_total=None):
    """Build a MultiheadIsotropicRFA layer from notebook-style `args` and head dimensions."""
    if query_key_dim is None or value_dim is None:
        query_key_dim, value_dim, _, _ = resolve_multihead_dims(n_heads, query_key_dim, value_dim, query_key_dim_total, value_dim_total)
    if query_key_dim != value_dim:
        raise ValueError(f"RFA requires equal query/key and value head dims, got {query_key_dim} and {value_dim}")
    config = RFAConfig.from_args(args, d_model=input_dim, n_heads=n_heads, head_dim=query_key_dim)
    return MultiheadIsotropicRFA(config)

##########################################################################################
##########################################################################################

class RFA_Block(nn.Module):
    """
    A single multi-head RFA layer, without residuals, norms or FFN.
    """

    def __init__(self, args, n_heads, input_dim, query_key_dim=None, value_dim=None, query_key_dim_total=None, value_dim_total=None):
        super(RFA_Block, self).__init__()
        # (ModuleList kept so that checkpoint keys stay 'layers.0.*')
        self.layers = nn.ModuleList([build_rfa_layer(args, n_heads, input_dim, query_key_dim, value_dim, query_key_dim_total, value_dim_total)])
        self.args = args

    def forward(self, inputs, t_measure=None, causal=True):
        return_metadata = not self.training and getattr(self.args, "compute_metadata", False)
        return self.layers[0](inputs, t_measure=t_measure, causal=causal, return_metadata=return_metadata)

##########################################################################################
##########################################################################################

class TransformerBlock(nn.Module):
    """
    Custom transformer block using vanilla attention, with residual connections,
    layer normalization, and a feedforward network.
    """

    def __init__(self, input_dim, qkv_dim, hidden_dim, num_heads, args, Norm=nn.RMSNorm):
        """
        Initializes a single transformer block.

        Args:
            input_dim (int): Dimensionality of input and output.
            hidden_dim (int): Dimensionality of the hidden feedforward layer.
            args: Additional parameters passed to the AttentionLayer.
        """
        super(TransformerBlock, self).__init__()

        # Baseline attention (B1-B4, selected by the args flags)
        self.attn = MultiHeadAttentionLayer.from_args(input_dim, qkv_dim, num_heads, args)
        
        # Layer norms before attention and MLP
        if Norm == None:
            self.norm1 = self.norm2 = None
        else:
            self.norm1 = Norm(input_dim)
            self.norm2 = Norm(input_dim)

        # Feedforward network: Linear -> ReLU -> Linear
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, causal=True):
        """
        Forward pass through the transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            out (torch.Tensor): Output tensor of the same shape as input.
            attn (torch.Tensor): Attention weights.
        """
        
        # === Self-attention block ===
        
        # Layer norm before attention (pre-norm)
        if self.norm1 == None:
            x_norm = x
        else:
            x_norm = self.norm1(x)

        # Self-attention with residual connection
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm, causal=causal)
        x = x + attn_out  # Residual connection
        
#         eta = torch.sigmoid(self.g_param)
#         x = (1 - eta) * x + eta * attn_out

        # Layer norm before feedforward network
        if self.norm2 == None:
            x_norm = x
        else:
            x_norm = self.norm2(x)

        # Feedforward network with residual connection
        ffn_out = self.ffn(x_norm)
        out = x + ffn_out  # Residual connection

        return out, attn_weights

##########################################################################################
##########################################################################################    

class RFATransformerBlock(nn.Module):
    """
    Multihead Simplified RFA Transformer Block
    """

    def __init__(self, args, n_heads, input_dim, query_key_dim=None, value_dim=None, query_key_dim_total=None, value_dim_total=None, hidden_dim=None, Norm=nn.RMSNorm):
        """
        Initializes a single Multihead Simplified RFA transformer block (with real inputs/outputs):

        Args:
            input_dim (int): Dimensionality of input and output.
            hidden_dim (int): Dimensionality of the hidden feedforward layer.
            args: Additional parameters passed to the AttentionLayer.
        """
        super(RFATransformerBlock, self).__init__()

        # Self-attention layer
        self.attn = build_rfa_layer(args, n_heads, input_dim, query_key_dim, value_dim, query_key_dim_total, value_dim_total)

        # Layer norms before attention and MLP
        if Norm == None:
            self.norm1 = self.norm2 = None
        else:
            self.norm1 = Norm(input_dim)
            self.norm2 = Norm(input_dim)

        # Feedforward network: Linear -> ReLU -> Linear
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self.args = args

    def forward(self, x, t_measure=None, causal=True, return_metadata=False):
        """
        Forward pass through the RFA transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            out (torch.Tensor): Output tensor of the same shape as input.
            output_dict (dict): Per-head SDE parameters and, if return_metadata, diagnostics.
        """

        # Norm before attention (pre-norm)
        x_norm = self.norm1(x) if self.args.use_real_input_norm else x

        # Self-attention with residual connection
        attn_out, output_dict = self.attn(x_norm, t_measure=t_measure, causal=causal, return_metadata=return_metadata)
        if self.args.use_outer_residual:
            x = x + attn_out

        x_norm = self.norm2(x) if self.args.use_real_output_norm else x

        ffn_out = self.ffn(x_norm) # Feedforward network with residual connection
        out = x + ffn_out  # Residual connection

        return out, output_dict
