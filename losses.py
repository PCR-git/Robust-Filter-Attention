import numpy as np
import torch
import torch.nn as nn

from model import resolve_multihead_dims
from model import init_complexlinear, init_complex_matrix
from model import ComplexLinearLayer, MultiHeadAttentionLayer, SpectralCoupledHeuristicAttention
from model import ComplexRMSNorm, MultiheadIsotropicRFA

##########################################################################################
##########################################################################################

class SelfAttentionBlock(nn.Module):
    """
    A minimal wrapper for a single multi-head attention layer.
    Removes residuals, norms, and FFN to isolate attention performance.
    """

    def __init__(self, input_dim, qkv_dim, num_heads, args):
        super(SelfAttentionBlock, self).__init__()

#         self.attn = MultiHeadAttentionLayer(input_dim, qkv_dim, num_heads, args)
        
        # --- Self-attention layer selection ---
        # Logic for ablations (RoPE + Decay) and (SCRoPE + Decay)
        if args.use_relative_decay_vanilla == True:
            self.attn = SpectralCoupledHeuristicAttention(input_dim, qkv_dim, num_heads, args)
        # Standard Baseline logic (B1, B2)
        else:
            self.attn = MultiHeadAttentionLayer(input_dim, qkv_dim, num_heads, args)

    def forward(self, x, t_measure=None, t_shift=None, causal=True):
        """
        Forward pass directly into the attention layer.
        """
        # We accept t_measure/t_shift as dummy variables to match RFA API
        # but we ONLY use causal for the actual baseline attention logic.
        
        # We pass x as Q, K, and V for self-attention
        out, attn_weights = self.attn(x, x, x, causal=causal)

        return out, attn_weights
    
##########################################################################################
##########################################################################################

class RFA_Block(nn.Module):
    """
    Neural network with a single multihead simplified precision attention block
    """

    # Initialize the network and specify input/output dimensions:
    def __init__(self, args, n_heads, input_dim, query_key_dim=None, value_dim=None, query_key_dim_total=None, value_dim_total=None):
        super(RFA_Block, self).__init__()

        self.layers = nn.ModuleList([MultiheadIsotropicRFA(args, n_heads, input_dim, query_key_dim, value_dim, query_key_dim_total, value_dim_total)])

     # Build the network:
    def forward(self, inputs, t_measure=None, t_shift=None, causal=True):
        
        layer = self.layers[0]
        out, output_dict = layer(inputs, inputs, inputs, t_measure, t_shift=t_shift, causal=causal)

        return out, output_dict

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

#         # Self-attention layer
#         self.attn = MultiHeadAttentionLayer(input_dim, qkv_dim, num_heads, args)
        
        # --- Self-attention layer selection ---
        # Logic for ablations (RoPE + Decay) and (SCRoPE + Decay)
        if args.use_relative_decay_vanilla == True:
            self.attn = SpectralCoupledHeuristicAttention(input_dim, qkv_dim, num_heads, args)
        # Standard Baseline logic (B1, B2)
        else:
            self.attn = MultiHeadAttentionLayer(input_dim, qkv_dim, num_heads, args)
        
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

# class ComplexTransformerBlock(nn.Module):
#     """
#     Custom transformer block using vanilla attention, with residual connections,
#     layer normalization, and a feedforward network.
#     """

#     def __init__(self, input_dim, qkv_dim, hidden_dim, num_heads, args, Norm=ComplexRMSNorm):
#         """
#         Initializes a single transformer block.

#         Args:
#             input_dim (int): Dimensionality of input and output.
#             hidden_dim (int): Dimensionality of the hidden feedforward layer.
#             args: Additional parameters passed to the AttentionLayer.
#         """
#         super(ComplexTransformerBlock, self).__init__()

#         # Self-attention layer
#         self.attn = ComplexMultiHeadAttentionLayer(input_dim, qkv_dim, num_heads, args)
        
#         # Layer norms before attention and MLP
#         if Norm == None:
#             self.norm1 = self.norm2 = None
#         else:
#             self.norm1 = Norm(input_dim)
#             self.norm2 = Norm(input_dim)

#         # Feedforward network: Linear -> ReLU -> Linear
#         self.ffn = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, input_dim)
#         )
        
#         self.g_param = nn.Parameter(torch.zeros(input_dim))

#     def forward(self, x):
#         """
#         Forward pass through the transformer block.

#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)

#         Returns:
#             out (torch.Tensor): Output tensor of the same shape as input.
#             attn (torch.Tensor): Attention weights.
#         """
        
#         # === Self-attention block ===
        
#         # Layer norm before attention (pre-norm)
#         if self.norm1 == None:
#             x_norm = x
#         else:
#             x_norm = self.norm1(x)

#         # Self-attention with residual connection
#         attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm)
#         x = x + attn_out  # Residual connection
        
# #         eta = torch.sigmoid(self.g_param)
# #         x = (1 - eta) * x + eta * attn_out

#         # Layer norm before feedforward network
#         if self.norm2 == None:
#             x_norm = x
#         else:
#             x_norm = self.norm2(x)

#         # Feedforward network with residual connection
#         ffn_out = self.ffn(x_norm)
#         out = x + ffn_out  # Residual connection

#         return out, attn_weights
    
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
        
        if query_key_dim==None or value_dim==None or query_key_dim_total==None or value_dim_total==None:
            # Set query_key and value dims, depending on whether user provided total dims, or head dims
            query_key_dim, value_dim, query_key_dim_total, value_dim_total = resolve_multihead_dims(n_heads, query_key_dim, value_dim, query_key_dim_total, value_dim_total)

        # Self-attention layer
        self.attn = MultiheadIsotropicRFA(args, n_heads, input_dim, query_key_dim, value_dim, query_key_dim_total, value_dim_total)

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

    def forward(self, x, t_measure=None, t_shift=None, causal=True):
        """
        Forward pass through the RFA transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, seq_len, input_dim)

        Returns:
            out (torch.Tensor): Output tensor of the same shape as input.
            attn (torch.Tensor): Attention weights.
        """
        
#         x_mag = torch.sqrt(torch.sum(x**2,axis=-1,keepdims=True))

        # Norm before attention (pre-norm)
        if self.args.use_real_input_norm == True:
            x_norm = self.norm1(x)
        else:
            x_norm = x

        # Self-attention with residual connection
        attn_out, output_dict = self.attn(x_norm, x_norm, x_norm, t_measure, t_shift=t_shift, causal=causal)
        
        if self.args.use_outer_residual == True:
            x = x + attn_out
        else:
            pass

        if self.args.use_real_output_norm == True:
            x_norm = self.norm2(x)
        else:
            x_norm = x
            
        ffn_out = self.ffn(x_norm) # Feedforward network with residual connection
        out = x + ffn_out  # Residual connection

        return out, output_dict
    
