import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .blocks import TransformerBlock, RFATransformerBlock
from .model_utils import resolve_multihead_dims

##########################################################################################
##########################################################################################

class TransformerNetwork(nn.Module):
    """
    A network composed of multiple stacked Transformer blocks.

    Each block consists of attention, feedforward layers, and residual connections.
    """

    def __init__(self, input_dim, qkv_dim, hidden_dim, num_heads, args, num_blocks=3, Norm=nn.RMSNorm, io_projections=True):
        """
        Args:
            input_dim (int): Input and output dimensionality for transformer blocks.
            hidden_dim (int): Hidden dimensionality in the feedforward layers.
            args: Additional arguments passed to each TransformerBlock and AttentionLayer.
            num_blocks (int): Number of stacked Transformer blocks.
            io_projections (bool): Linear d->d maps before the first and after the last block. Needed when
                the network is used directly as a regressor; redundant inside a LanguageModel (embedding
                and LM head already provide them), where the standard layout omits them.
        """
        super().__init__()
        
        self.args = args

        # Initial linear layer
        self.input_proj = nn.Linear(input_dim, input_dim) if io_projections else nn.Identity()

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

        self.output_proj = nn.Linear(input_dim, input_dim) if io_projections else nn.Identity()

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
            if self.training and getattr(self.args, "grad_checkpointing", False):
                # Recompute the block in the backward pass instead of storing its activations
                x, attn = checkpoint(block, x, causal, use_reentrant=False)
            else:
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

    def __init__(self, args, num_blocks=2, n_heads=1, input_dim=None, query_key_dim=None, value_dim=None, query_key_dim_total=None, value_dim_total=None, hidden_dim=None, Norm=nn.RMSNorm, io_projections=True):
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
            io_projections (bool): Linear d->d maps before the first and after the last block. Needed when
                the network is used directly as a regressor; redundant inside a LanguageModel (embedding
                and LM head already provide them), where the standard layout omits them.
        """
        
        super().__init__()
        
        self.args = args
        
        # Set query_key and value dims, depending on whether user provided total dims, or head dims
        query_key_dim, value_dim, query_key_dim_total, value_dim_total = resolve_multihead_dims(n_heads, query_key_dim, value_dim, query_key_dim_total, value_dim_total)
        
        # Initial linear layer
        self.input_proj = nn.Linear(input_dim, input_dim) if io_projections else nn.Identity()

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
        self.output_proj = nn.Linear(input_dim, input_dim) if io_projections else nn.Identity()

    def forward(self, x, t_measure=None, causal=True):
        """
        Forward pass through the RFA Transformer network stack.

        Args:
            x (torch.Tensor): Input sequence tensor, shape (batch_size, seq_len, input_dim).
            t_measure (torch.Tensor | None): Observation times, shape (batch_size, seq_len).
                                             None means regular positions 0..seq_len-1.

        Returns:
            out (torch.Tensor): The final output tensor after processing and output projection,
                                shape (batch_size, seq_len, input_dim).
            output_dict (dict): Outputs from the last RFA block. Diagnostics (attention matrices etc.)
                                are included in eval mode when args.compute_metadata is True.
        """

        return_metadata = not self.training and getattr(self.args, "compute_metadata", False)

        x = self.input_proj(x) # Input projection

        # Sequentially apply each Transformer block
        for block in self.blocks:
            if self.training and getattr(self.args, "grad_checkpointing", False):
                # Recompute the block in the backward pass instead of storing its activations
                x, output_dict = checkpoint(block, x, t_measure, causal, False, use_reentrant=False)
            else:
                x, output_dict = block(x, t_measure=t_measure, causal=causal, return_metadata=return_metadata)

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
    into a Causal LM (token embedding tied to the output head).

    Use `loss()` for training: it computes the next-token cross-entropy in chunks without
    materializing the full [B, L, vocab] logits, which otherwise dominate memory.
    """
    
    def __init__(self, backbone, vocab_size=50257, embed_dim=256, embed_init_std=None, loss_chunk_size=2048):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.backbone = backbone  # This can be RFA, Standard, 1-layer, or full Transformer
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.loss_chunk_size = loss_chunk_size
        
        # Tie weights. embed_init_std=None keeps the PyTorch default N(0, 1) used for the published
        # results; with tying this makes the initial logits large (initial loss ~3x ln(vocab)),
        # and a small init such as 0.02 is the usual choice.
        self.lm_head.weight = self.token_embedding.weight
        if embed_init_std is not None:
            nn.init.normal_(self.token_embedding.weight, std=embed_init_std)

    def forward(self, input_ids, **kwargs):
        """Returns (logits [B, L, vocab], backbone output dict)."""
        out, output_dict = self.backbone(self.token_embedding(input_ids), **kwargs)
        return self.lm_head(out), output_dict

    def loss(self, input_ids, labels=None, **kwargs):
        """
        Mean next-token cross-entropy (position i predicts labels[i+1]; label -100 is ignored).
        Returns (loss, backbone output dict).
        """
        labels = input_ids if labels is None else labels
        out, output_dict = self.backbone(self.token_embedding(input_ids), **kwargs)
        h = out[:, :-1].reshape(-1, out.shape[-1])
        y = labels[:, 1:].reshape(-1)

        def chunk_loss(h_chunk, y_chunk):
            logits = F.linear(h_chunk, self.lm_head.weight, self.lm_head.bias)
            return F.cross_entropy(logits, y_chunk, ignore_index=-100, reduction="sum")

        total = out.new_zeros(())
        for s in range(0, h.shape[0], self.loss_chunk_size):
            h_chunk, y_chunk = h[s:s + self.loss_chunk_size], y[s:s + self.loss_chunk_size]
            if torch.is_grad_enabled():
                # Logits of each chunk are recomputed in the backward pass instead of stored
                total = total + checkpoint(chunk_loss, h_chunk, y_chunk, use_reentrant=False)
            else:
                total = total + chunk_loss(h_chunk, y_chunk)
        return total / (y != -100).sum(), output_dict
     
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
        
        # 2. Backbone pass (Must be causal=True); token i's output is the prediction of token i+1
        out_latent, output_dict = self.backbone(
            x_latent,
            t_measure=t_measure,
            causal=True,
            **kwargs
        )
        
        # 3. Map all positions back to sensor space
        # [Batch, Seq_Len, 7]
        predictions = self.output_head(out_latent)
        
        return predictions, output_dict

##########################################################################################
########################################################################################## 

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

    def forward(self, x, t_measure=None, causal=False, **kwargs):
        # x: [Batch, Seq_Len, Input_Dim]
        batch_size, seq_len, _ = x.shape

        # Project features to latent space
        x_latent = self.input_proj(x)

        # Backbone Forward Pass
        out_latent, backbone_output = self.backbone(
            x_latent,
            t_measure=t_measure,
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
