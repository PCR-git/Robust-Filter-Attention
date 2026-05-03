import numpy as np
import torch
import torch.nn as nn

from model import init_complexlinear, init_complex_matrix
from model import ComplexLinearLayer, MultiHeadAttentionLayer
from model import TransformerBlock
from model import MultiheadIsotropicRFA, RFATransformerBlock
from model import resolve_multihead_dims

##########################################################################################
##########################################################################################

class TransformerNetwork(nn.Module):
    """
    A network composed of multiple stacked Transformer blocks.

    Each block consists of attention, feedforward layers, and residual connections.
    """

    def __init__(self, input_dim, qkv_dim, hidden_dim, num_heads, args, num_blocks=3, Norm=nn.RMSNorm):
        """
        Args:
            input_dim (int): Input and output dimensionality for transformer blocks.
            hidden_dim (int): Hidden dimensionality in the feedforward layers.
            args: Additional arguments passed to each TransformerBlock and AttentionLayer.
            num_blocks (int): Number of stacked Transformer blocks.
        """
        super().__init__()
        
        self.args = args

        # Initial linear layer
        self.input_proj = nn.Linear(input_dim, input_dim)

        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(input_dim, qkv_dim, hidden_dim, num_heads, args, Norm)
            for _ in range(num_blocks)
        ])

        # Optional final LayerNorm
        if Norm == None:
            self.final_norm = None
        else:
            self.final_norm = Norm(input_dim)

        self.output_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x, causal=True, **kwargs):
        """
        Forward pass through the Transformer network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            out (torch.Tensor): Final output (possibly projected).
            attn_list (list): List of attention weights from each block.
        """
        attn_list = []

        x = self.input_proj(x)

        for block in self.blocks:
            x, attn = block(x, causal=causal)
            attn_list.append(attn)

        # Apply final normalization
        if self.final_norm == None:
            pass
        else:
            x = self.final_norm(x)

        x = self.output_proj(x)

        return x, attn_list

##########################################################################################
##########################################################################################

# class ComplexTransformerNetwork(nn.Module):
#     """
#     A network composed of a stack of complex-real Transformer blocks, i.e. using complex-valued attention internally,
#     but operating on real-valued inputs and outputs. Optionally ends with a projection layer.

#     Attributes:
#         blocks (nn.ModuleList): A list of ComplexRealTransformerBlock modules applied sequentially.
#         final_norm: Normalization applied after the stack of blocks.
#         use_output_layer (bool): Whether to apply a final linear projection.
#         output_layer (nn.Linear): Optional linear projection layer if use_output_layer is True.
#     """

#     def __init__(self, input_dim, qkv_dim, hidden_dim, num_heads, args, num_blocks=2, Norm=nn.RMSNorm):
#         """
#         Initializes the transformer network.

#         Args:
#             input_dim (int): Dimensionality of the input and output vectors.
#             hidden_dim (int): Internal dimension used by the attention blocks.
#             args (Namespace): Additional model hyperparameters (e.g., device, config flags).
#             num_blocks (int): Number of stacked Transformer blocks.
#         """
#         super().__init__()
        
#         # Initial linear layer
#         self.input_proj = nn.Linear(input_dim, input_dim)

#         # Stack of Transformer blocks, each using complex-valued attention
#         self.blocks = nn.ModuleList([
#             ComplexRealTransformerBlock(input_dim, qkv_dim, hidden_dim, num_heads, args)
#             for _ in range(num_blocks)
#         ])

#         # Final normalization applied to the output of the last block
#         self.final_norm = Norm(input_dim)

#         # Output linear projection
#         self.output_proj = nn.Linear(input_dim, input_dim)

#     def forward(self, x):
#         """
#         Forward pass through the Transformer network.

#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)

#         Returns:
#             x (torch.Tensor): Output tensor, optionally projected to output_dim.
#             attn_list (list): List of attention weight tensors from each block.
#         """
#         attn_list = []
        
#         x = self.input_proj(x)

#         # Sequentially apply each Transformer block
#         for block in self.blocks:
#             x, attn = block(x)
#             attn_list.append(attn)

#         # Apply final normalization
#         x = self.final_norm(x)

#         # Output projection layer
#         x = self.output_proj(x)

#         return x, attn_list

##########################################################################################
##########################################################################################

class RFATransformerNetwork(nn.Module):
    """
    Robust Filter Attention (RFA) Transformer Network.

    Implements a sequence-to-sequence model composed of stacked RFA Transformer blocks,
    which leverage continuous-time SDE dynamics for positional encoding and uncertainty-aware filtering.

    Attributes:
        args (Namespace): Configuration and hyperparameters for the model.
        input_proj (nn.Linear): Initial linear projection of the input features.
        blocks (nn.ModuleList): A stack of RFATransformerBlock modules applied sequentially.
        final_norm (nn.Module | None): Normalization layer (e.g., RMSNorm) applied after the stack of blocks.
        output_proj (nn.Linear): Final linear projection of the processed sequence to the output dimension.
    """

    def __init__(self, args, num_blocks=2, n_heads=1, input_dim=None, query_key_dim=None, value_dim=None, query_key_dim_total=None, value_dim_total=None, hidden_dim=None, Norm=nn.RMSNorm):
        """
        Initializes the RFA Transformer network architecture.

        Args:
            args (Namespace): Global model configuration and hyperparameters.
            num_blocks (int): The number of stacked RFATransformerBlock layers. Defaults to 2.
            n_heads (int): The number of attention heads in the Multi-Head Attention layer. Defaults to 1.
            input_dim (int): Dimensionality of the raw input features (and the residual path). Must be provided.
            query_key_dim (int | None): Dimensionality of query/key vectors per head. Mutually exclusive with query_key_dim_total.
            value_dim (int | None): Dimensionality of value vectors per head. Mutually exclusive with value_dim_total.
            query_key_dim_total (int | None): Total dimensionality of Q/K space across all heads. Mutually exclusive with query_key_dim.
            value_dim_total (int | None): Total dimensionality of V space across all heads. Mutually exclusive with value_dim.
            hidden_dim (int | None): Internal expansion dimension for the Feed-Forward Network (FFN).
            Norm (Type[nn.Module]): Normalization class (e.g., nn.RMSNorm, nn.LayerNorm). Can be None.
        """
        
        super().__init__()
        
        self.args = args
        
        # Set query_key and value dims, depending on whether user provided total dims, or head dims
        query_key_dim, value_dim, query_key_dim_total, value_dim_total = resolve_multihead_dims(n_heads, query_key_dim, value_dim, query_key_dim_total, value_dim_total)
        
        # Initial linear layer
        self.input_proj = nn.Linear(input_dim, input_dim)

        # Stack of Transformer blocks, each using complex-valued attention
        self.blocks = nn.ModuleList([      
            RFATransformerBlock(args, n_heads, input_dim, query_key_dim, value_dim, query_key_dim_total, value_dim_total, hidden_dim, Norm)
            for _ in range(num_blocks)
        ])

        # Final normalization applied to the output of the last block
        if Norm == None:
            self.final_norm = None
        else:
            self.final_norm = Norm(input_dim)

        # Output linear projection
        self.output_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x, t_measure=None, t_shift=None, causal=True):
        """
        Forward pass through the RFA Transformer network stack.

        Args:
            x (torch.Tensor): Input sequence tensor, shape (batch_size, seq_len, input_dim).
            t_measure (torch.Tensor): Time of measurement for each token, shape (batch_size, seq_len) or (seq_len,). 
                                      Used to compute time lag (Delta t) for SDE propagation.

        Returns:
            out (torch.Tensor): The final output tensor after processing and output projection,
                                shape (batch_size, seq_len, input_dim).
            output_dict (dict): Dictionary containing outputs from the last RFA block (e.g., final attention weights, norms, etc.).
        """
        
        x = self.input_proj(x) # Input projection

        # Sequentially apply each Transformer block
        for block in self.blocks:
            x, output_dict = block(x, t_measure, t_shift=t_shift, causal=causal)

        # Apply final normalization
        if self.final_norm == None:
            pass
        else:
            x = self.final_norm(x)

        out = self.output_proj(x) # Output projection

        return out, output_dict

##########################################################################################
##########################################################################################

class LanguageModel(nn.Module):
    """
    A generic wrapper that turns any transformer/attention block 
    into a Causal LM.
    """
    
    def __init__(self, backbone, vocab_size=50257, embed_dim=256):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.backbone = backbone  # This can be RFA, Standard, 1-layer, or full Transformer
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids, t_measure=None, t_shift=None, **kwargs):
        x = self.token_embedding(input_ids)
        
        # The backbone handles the attention/mixing logic
        # We pass **kwargs so that t_measure or t_shift only get used if the backbone needs them
        out, output_dict = self.backbone(x, t_measure=t_measure, t_shift=t_shift, **kwargs)
        
        logits = self.lm_head(out)
        
        return logits, output_dict
     
##########################################################################################
##########################################################################################     
    
class ETT_NextStep_Predictor(nn.Module):
    def __init__(self, backbone, input_dim=7, embed_dim=256):
        super().__init__()
        self.backbone = backbone
        self.input_dim = input_dim

        # Project 7D sensor vector to Latent
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Project Latent back to 7D sensor vector
        self.output_head = nn.Linear(embed_dim, input_dim)

    def forward(self, x, t_measure=None, **kwargs):
        # x shape: [Batch, Seq_Len, 7]
        
        # 1. Project to Latent
        x_latent = self.input_proj(x)
        
        # 2. Backbone pass (Must be causal=True)
        # We set t_shift=1 because every token i is predicting token i+1
        out_latent, output_dict = self.backbone(
            x_latent, 
            t_measure=t_measure, 
            t_shift=1, 
            causal=True, 
            **kwargs
        )
        
        # 3. Map all positions back to sensor space
        # [Batch, Seq_Len, 7]
        predictions = self.output_head(out_latent)
        
        return predictions, output_dict

##########################################################################################
########################################################################################## 

# class ETT_Seq_to_Seq_Forecaster(nn.Module):
#     def __init__(self, backbone, input_dim=7, embed_dim=256, seq_len=96, pred_len=48):
#         super().__init__()
#         self.backbone = backbone
#         self.seq_len = seq_len
#         self.pred_len = pred_len
#         self.input_dim = input_dim

#         # Input projection to latent dimension
#         self.input_proj = nn.Linear(input_dim, embed_dim)
        
#         # Global head for ETT benchmarks
#         self.forecast_head = nn.Linear(seq_len * embed_dim, pred_len * input_dim)

#     def forward(self, x, t_measure=None, t_shift=None, causal=False, **kwargs):
#         # x: [Batch, Seq_Len, Input_Dim]
#         batch_size, seq_len, _ = x.shape
        
#         # Project features to latent space
#         x_latent = self.input_proj(x)
        
#         # Backbone Forward Pass
#         # We pass causal=False for smoothing. 
#         # t_shift is used by RFA; Standard Transformer will ignore it via **kwargs.
#         out_latent, backbone_output = self.backbone(
#             x_latent, 
#             t_measure=t_measure, 
#             t_shift=t_shift, 
#             causal=causal, 
#             **kwargs
#         )
        
#         # Output Standardization
#         # If backbone_output is a list (Standard), wrap it in a dict for consistency.
#         if isinstance(backbone_output, list):
#             output_dict = {'attn_weights': backbone_output}
#         else:
#             output_dict = backbone_output
        
#         # Global Projection (Flattening)
#         # Standardize the flatten to handle variable seq_len if necessary
#         flat_latent = out_latent.reshape(batch_size, -1)
        
#         # Generate Forecast Window
#         forecast = self.forecast_head(flat_latent)
        
#         # Final shape: [Batch, 48, 7]
#         return forecast.view(batch_size, self.pred_len, self.input_dim), output_dict
    
class ETT_Seq_to_Seq_Forecaster(nn.Module):
    def __init__(self, backbone, input_dim=7, embed_dim=256, seq_len=96, pred_len=48):
        super().__init__()
        self.backbone = backbone
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim

        # Input projection to latent dimension
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # --- FACTORIZED FORECAST HEAD ---
        # 1. Spatial/Feature Projector: Maps latent dim back to 7 features
        # [128 -> 7] (or 256 -> 7 based on your args.d_e)
        self.feature_map = nn.Linear(embed_dim, input_dim)
        
        # 2. Temporal Projector: Maps history length to forecast length
        # [96 -> 48]
        self.temporal_map = nn.Linear(seq_len, pred_len)
        # --------------------------------

    def forward(self, x, t_measure=None, t_shift=None, causal=False, **kwargs):
        # x: [Batch, Seq_Len, Input_Dim]
        batch_size, seq_len, _ = x.shape
        
        # Project features to latent space
        x_latent = self.input_proj(x)
        
        # Backbone Forward Pass
        out_latent, backbone_output = self.backbone(
            x_latent, 
            t_measure=t_measure, 
            t_shift=t_shift, 
            causal=causal, 
            **kwargs
        )
        
        # 1. Feature Mapping (Apply to each token independently)
        # [Batch, 96, 128] -> [Batch, 96, 7]
        out_features = self.feature_map(out_latent)
        
        # 2. Temporal Projection (Apply to the time dimension)
        # Permute to [Batch, 7, 96] so the Linear layer acts on the 96 steps
        out_features = out_features.transpose(1, 2)
        
        # [Batch, 7, 96] -> [Batch, 7, 48]
        forecast = self.temporal_map(out_features)
        
        # 3. Permute back to [Batch, 48, 7]
        forecast = forecast.transpose(1, 2)
        
        # Output Standardization for RFA/Transformer logging
        if isinstance(backbone_output, list):
            output_dict = {'attn_weights': backbone_output}
        else:
            output_dict = backbone_output
            
        return forecast, output_dict
    