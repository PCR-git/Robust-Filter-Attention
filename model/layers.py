import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import math

from model import init_complexlinear, init_complex_matrix
from model import RoPE, SCRoPE, get_alibi_slopes, init_spectrally_coupled_rope, LearnableRoPE

##########################################################################################
##########################################################################################

class ComplexLinearLayer(nn.Module):
    """
    Complex-valued linear layer
    """ 
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.real = nn.Linear(in_features, out_features, bias=bias)
        self.imag = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, inputs):
        # input: (B, 2, in_features, 1)
        real_input = inputs[:, 0]
        imag_input = inputs[:, 1]

        real_out = self.real(real_input) - self.imag(imag_input)
        imag_out = self.real(imag_input) + self.imag(real_input)

        return torch.stack((real_out, imag_out), dim=1)

##########################################################################################
##########################################################################################

class ComplextoRealLinearLayer(nn.Module):
    """
    Complex-valued linear layer that returns only the real part of the output
    """ 
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.real = nn.Linear(in_features, out_features, bias=bias)
        self.imag = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, inputs):
        # input: (B, 2, in_features, 1)
        real_input = inputs[:, 0]
        imag_input = inputs[:, 1]

        real_out = self.real(real_input) - self.imag(imag_input)

        return real_out

##########################################################################################
##########################################################################################

class ComplexLinearHermitianLayer(nn.Module):
    def __init__(self, source_layer: ComplexLinearLayer):
        super().__init__()
        self.source = source_layer  # reference, not a copy

    def forward(self, x):
        real_input = x[:, 0]
        imag_input = x[:, 1]

        W_r = self.source.real.weight
        W_i = -self.source.imag.weight

        real_out = F.linear(real_input, W_r) + F.linear(imag_input, W_i)
        imag_out = F.linear(imag_input, W_r) - F.linear(real_input, W_i)

        return torch.stack((real_out, imag_out), dim=1)

##########################################################################################
##########################################################################################

#################################### ATTENTION LAYERS ####################################

##########################################################################################
##########################################################################################
    
class MultiHeadAttentionLayer(nn.Module):
    """
    Multi-head attention layer with Rotary Positional Embeddings (RoPE).
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, args):
        super(MultiHeadAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Ensure hidden_dim is divisible by num_heads
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads

        # Projections for Q, K, V, and final output
        self.query_projection = nn.Linear(input_dim, hidden_dim)
        self.key_projection = nn.Linear(input_dim, hidden_dim)
        self.value_projection = nn.Linear(input_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, input_dim)

        # Scaling factor is based on the dimension of a single head
        self.scale = math.sqrt(self.head_dim)
        
        max_len = 4096 
        causal_mask = torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(1)
        self.register_buffer('causal_mask', causal_mask)
        
        self.args = args

        # --- RoPE Initialization ---
        if self.args.use_rope == True:
            if self.head_dim % 2 != 0:
                 raise ValueError(f"Head dimension ({self.head_dim}) must be even to use RoPE.")
            self.rope = RoPE(dim=self.head_dim, max_seq_len=max_len)
            
        # --- ALiBi Initialization ---
        if self.args.use_alibi == True:
            self.register_buffer('alibi_slopes', get_alibi_slopes(num_heads))     
            
    def forward(self, query, keys, values, causal=True):
        batch_size, seq_len, _ = query.shape

        # Project and Reshape to [B, L, H, D_head] 
        # (Matches RoPE's expected input shape immediately)
        Q = self.query_projection(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key_projection(keys).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value_projection(values).view(batch_size, seq_len, self.num_heads, self.head_dim)

        if self.args.use_rope:
            Q = self.rope(Q, seq_len)
            K = self.rope(K, seq_len)

        # Transpose for Batch MatMul: [B, H, L, D_head]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Standard Scaled Dot Product Attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if self.args.use_alibi == True:
            context_position = torch.arange(seq_len, device=query.device).unsqueeze(0)
            memory_position = torch.arange(seq_len, device=query.device).unsqueeze(1)
            relative_distance = -torch.abs(memory_position - context_position) # [L, L]
            alibi_bias = self.alibi_slopes * relative_distance.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores + alibi_bias

        if causal:
            # Apply causal mask
            mask = self.causal_mask[:, :, :seq_len, :seq_len]
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, V) # [B, H, L, D_head]

        # Output Projection
        out = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        return self.output_projection(out), attn_weights

##########################################################################################
##########################################################################################     
    
# # # MHA with learnable RoPE
    
# class MultiHeadAttentionLayer(nn.Module):
#     def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, args):
#         super(MultiHeadAttentionLayer, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
        
#         assert hidden_dim % num_heads == 0
#         self.head_dim = hidden_dim // num_heads

#         self.query_projection = nn.Linear(input_dim, hidden_dim)
#         self.key_projection = nn.Linear(input_dim, hidden_dim)
#         self.value_projection = nn.Linear(input_dim, hidden_dim)
#         self.output_projection = nn.Linear(hidden_dim, input_dim)

#         self.scale = math.sqrt(self.head_dim)
        
#         max_len = 4096 
#         causal_mask = torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
#         self.register_buffer('causal_mask', causal_mask)
        
#         self.args = args

#         # --- RoPE Initialization ---
#         if getattr(self.args, 'use_rope', False):
#             if self.head_dim % 2 != 0:
#                  raise ValueError(f"Head dimension ({self.head_dim}) must be even for RoPE.")
            
#             self.rope = LearnableRoPE(dim=self.head_dim, max_seq_len=max_len)
            
#             # Toggle learnability based on flag
#             if not getattr(self.args, 'learnable_rope', False):
#                 self.rope.theta.requires_grad = False
            
#         # --- ALiBi Initialization ---
#         if getattr(self.args, 'use_alibi', False):
#             # Assuming get_alibi_slopes is defined elsewhere
#             self.register_buffer('alibi_slopes', get_alibi_slopes(num_heads))     
            
#     def forward(self, query, keys, values, causal=True):
#         batch_size, seq_len, _ = query.shape

#         # Project and Reshape to [B, L, H, D_head]
#         Q = self.query_projection(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
#         K = self.key_projection(keys).view(batch_size, seq_len, self.num_heads, self.head_dim)
#         V = self.value_projection(values).view(batch_size, seq_len, self.num_heads, self.head_dim)

#         # Apply RoPE in [B, L, H, D_head] format
#         if self.args.use_rope:
#             Q = self.rope(Q)
#             K = self.rope(K)

#         # Transpose for Attention: [B, H, L, D_head]
#         Q = Q.transpose(1, 2)
#         K = K.transpose(1, 2)
#         V = V.transpose(1, 2)

#         attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
#         # Apply ALiBi Bias
#         if getattr(self.args, 'use_alibi', False):
#             # context_position is "columns", memory_position is "rows"
#             # [1, 1, L, L]
#             dist = torch.arange(seq_len, device=query.device)
#             relative_distance = dist[None, :] - dist[:, None] 
#             relative_distance = -torch.abs(relative_distance).view(1, 1, seq_len, seq_len)
            
#             # alibi_slopes shape: [1, H, 1, 1]
#             attn_scores = attn_scores + (self.alibi_slopes.view(1, -1, 1, 1) * relative_distance)

#         if causal:
#             mask = self.causal_mask[:, :, :seq_len, :seq_len]
#             attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

#         attn_weights = F.softmax(attn_scores, dim=-1)
#         context = torch.matmul(attn_weights, V)

#         # Reshape back to [B, L, Hidden]
#         out = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

#         return self.output_projection(out), attn_weights      
    
##########################################################################################
##########################################################################################   

class SpectralCoupledHeuristicAttention(nn.Module):
    """
    Spectrally-Coupled RoPE + Heuristic Decay.
    
    This variant isolates the geometric effects of spectral sharding and 
    exponential decay without the Bayesian uncertainty modeling of RFA.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, args):
        super(SpectralCoupledHeuristicAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads

        self.query_projection = nn.Linear(input_dim, hidden_dim)
        self.key_projection = nn.Linear(input_dim, hidden_dim)
        self.value_projection = nn.Linear(input_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, input_dim)

        self.scale = math.sqrt(self.head_dim)
        
        max_len = 4096 
        causal_mask = torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(1)
        self.register_buffer('causal_mask', causal_mask)
        
        self.args = args

        if self.head_dim % 2 != 0:
             raise ValueError(f"Head dimension ({self.head_dim}) must be even.")
        
        # Determine whether to use sharded (SC) or standard frequencies
        if self.args.use_SC_RoPE == True:
            self.rope = SCRoPE(dim=self.head_dim, n_heads=num_heads, max_seq_len=max_len)
        else:
            self.rope = RoPE(dim=self.head_dim, max_seq_len=max_len)
            
        # Initialize mus using spectral coupling logic for parity with RFA
        dim_target = self.head_dim // 2
        _, total_mu = init_spectrally_coupled_rope(
            num_heads, 
            dim_target, 
            b=self.args.damping
        )
        self.register_buffer("mu", total_mu)
            
    def forward(self, query, keys, values, causal=True):
        batch_size, seq_len, _ = query.shape

        Q = self.query_projection(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key_projection(keys).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value_projection(values).view(batch_size, seq_len, self.num_heads, self.head_dim)

        Q = self.rope(Q, seq_len)
        K = self.rope(K, seq_len)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply exponential decay based on relative distance
        t = torch.arange(seq_len, device=query.device)
        rel_dist = torch.abs(t[:, None] - t[None, :])
        
        mu = self.mu.view(1, self.num_heads, 1, 1) 
        decay_multiplier = torch.exp(-mu * rel_dist)
        attn_scores = attn_scores * decay_multiplier
        
        if self.args.use_alibi == True:
            context_position = torch.arange(seq_len, device=query.device).unsqueeze(0)
            memory_position = torch.arange(seq_len, device=query.device).unsqueeze(1)
            relative_distance = -torch.abs(memory_position - context_position)
            alibi_bias = self.alibi_slopes * relative_distance.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores + alibi_bias

        if causal:
            mask = self.causal_mask[:, :, :seq_len, :seq_len]
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, V)

        out = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        return self.output_projection(out), attn_weights

##########################################################################################
##########################################################################################   

# class ComplexMultiHeadAttentionLayer(nn.Module):
#     """
#     Complex Multi-head attention layer with Rotary Positional Embeddings (RoPE).
#     """
#     def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, args):
#         super(ComplexMultiHeadAttentionLayer, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
        
#         # Ensure hidden_dim is divisible by num_heads
#         assert hidden_dim % num_heads == 0
#         self.head_dim = hidden_dim // num_heads

#         # Projections for Q, K, V, and final output
#         self.query_projection = nn.Linear(input_dim, hidden_dim)
#         self.key_projection = nn.Linear(input_dim, hidden_dim)
#         self.value_projection = nn.Linear(input_dim, hidden_dim)
#         self.output_projection = nn.Linear(hidden_dim, input_dim)

#         # Scaling factor is based on the dimension of a single head
#         self.scale = math.sqrt(self.head_dim)

#         # Causal mask for autoregressive attention, a bit more flexible
#         causal_mask = torch.tril(torch.ones(args.seq_len, args.seq_len)).unsqueeze(0).unsqueeze(1)
#         self.register_buffer('causal_mask', causal_mask)
        
#         self.args = args

#         # --- RoPE INSTANTIATION ---
#         if self.head_dim % 2 != 0:
#              raise ValueError(f"Head dimension ({self.head_dim}) must be even to use RoPE.")
#         self.rope = RoPE(dim=self.head_dim, max_seq_len=args.seq_len)

#     def forward(self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
        
#         batch_size = query.size(0)
#         seq_len = query.size(1) # L_q
        
#         # 1. Project Q, K, V
#         proj_query = self.query_projection(query)
#         proj_keys = self.key_projection(keys)
#         proj_values = self.value_projection(values)

#         # 2. Reshape and transpose to [B, H, L, D_head]
#         proj_query = proj_query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
#         proj_keys = proj_keys.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
#         proj_values = proj_values.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

#         # 3. Apply RoPE
#         # RoPE expects [B, L, H, D_head], so transpose the L and H dimensions
#         Q_rot = proj_query.transpose(1, 2) 
#         K_rot = proj_keys.transpose(1, 2)
        
#         Q_rot = self.rope(Q_rot, seq_len)
#         K_rot = self.rope(K_rot, seq_len)
        
#         # Permute back to [B, H, L, D_head] for matrix multiplication
#         proj_query = Q_rot.transpose(1, 2)
#         proj_keys = K_rot.transpose(1, 2)
        
#         # 4. Compute attention scores
#         # (B, H, Lq, D_head) x (B, H, D_head, Lk) -> (B, H, Lq, Lk)
#         attn_scores = torch.matmul(proj_query, proj_keys.transpose(-2, -1))

#         # 5. Mask and Normalize
#         attn_scores = attn_scores.masked_fill(self.causal_mask[:, :, :attn_scores.size(-2), :attn_scores.size(-1)] == 0, float('-inf'))
#         scaled_attn_scores = attn_scores / self.scale
#         attn_weights = F.softmax(scaled_attn_scores, dim=-1)

#         # 6. Weighted sum of values
#         # (B, H, Lq, Lk) x (B, H, Lk, D_head) -> (B, H, Lq, D_head)
#         context_vector = torch.matmul(attn_weights, proj_values)
        
#         # 7. Concatenate heads and project out
#         # (B, H, Lq, D_head) -> (B, Lq, H, D_head) -> (B, Lq, hidden_dim)
#         context_vector = context_vector.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
#         out = self.output_projection(context_vector)

#         return out, attn_weights

##########################################################################################
##########################################################################################  

# class MultiHeadComplexAttentionLayer(nn.Module):
#     """
#     Multi-head complex-valued attention layer.
#     """
#     def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, args):
#         super(MultiHeadComplexAttentionLayer, self).__init__()

#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads

#         # Ensure hidden_dim is divisible by num_heads
#         assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
#         self.head_dim = hidden_dim // num_heads

#         # Define complex linear layers for projections.
#         # These project to the full hidden_dim, then are split into heads later.
#         self.W_q = ComplexLinearLayer(input_dim, hidden_dim)
#         self.W_k = ComplexLinearLayer(input_dim, hidden_dim)
#         self.W_v = ComplexLinearLayer(input_dim, hidden_dim)
#         # Output projection takes the concatenated heads (hidden_dim) and projects back to input_dim.
#         self.W_o = ComplexLinearLayer(hidden_dim, input_dim)
#         # Initialize complex linear layers
#         Wqi, bqi = init_complex_matrix(input_dim, hidden_dim, bias=True)
#         Wki, bki = init_complex_matrix(input_dim, hidden_dim, bias=True)
#         Wvi, bvi = init_complex_matrix(input_dim, hidden_dim, bias=True)
#         Wpi, bpi = init_complex_matrix(hidden_dim, input_dim, bias=True)

#         init_complexlinear(self.W_q, Wqi, bqi)
#         init_complexlinear(self.W_k, Wki, bki)
#         init_complexlinear(self.W_v, Wvi, bvi)
#         init_complexlinear(self.W_o, Wpi, bpi)

#         # Scaling factor is based on the dimension of a single head.
#         self.scale = math.sqrt(self.head_dim)

#         # Causal mask for autoregressive attention, suitable for broadcasting.
#         causal_mask = torch.tril(torch.ones(args.seq_len, args.seq_len)).unsqueeze(0).unsqueeze(1)
#         self.register_buffer('causal_mask', causal_mask)
        
#         self.noise_floor = nn.Parameter(torch.tensor(1.0))
#         self.tau = nn.Parameter(torch.tensor(0.0))

#         self.args = args

#     def forward(self, Z_q, Z_k, Z_v):
#         """
#         Forward pass for multi-head complex-valued attention.

#         Inputs:
#             Z_q, Z_k, Z_v: complex-valued input tensors of shape
#                            (batch_size, 2, seq_len, input_dim)
#                            where the second dimension holds [real, imag]

#         Returns:
#             out: complex-valued output tensor of shape
#                  (batch_size, 2, seq_len, input_dim)
#             attention_weights: real-valued attention weights of shape
#                                (batch_size, num_heads, seq_len, seq_len)
#         """
        
#         # Determine if the input was originally real-valued
#         using_real = False
#         if len(Z_q.size()) == 3:
#             using_real = True
#             Z_q = torch.stack((Z_q, torch.zeros_like(Z_q).to(Z_q.device)), axis=1)
#             Z_k = torch.stack((Z_k, torch.zeros_like(Z_k).to(Z_k.device)), axis=1)
#             Z_v = torch.stack((Z_v, torch.zeros_like(Z_v).to(Z_v.device)), axis=1)

#         batch_size = Z_q.size(0)
        
#         # Project Q, K, V to the hidden dimension
#         # Shape: (B, 2, L, input_dim) -> (B, 2, L, hidden_dim)
#         proj_q = self.W_q(Z_q)
#         proj_k = self.W_k(Z_k)
#         proj_v = self.W_v(Z_v)

#         # Reshape and transpose for multi-head attention.
#         # Original: (B, 2, L, hidden_dim) -> (B, 2, L, num_heads, head_dim)
#         # Permute to bring num_heads to dim 2: (B, 2, num_heads, L, head_dim)
#         proj_q_multih = proj_q.view(batch_size, 2, -1, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
#         proj_k_multih = proj_k.view(batch_size, 2, -1, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
#         proj_v_multih = proj_v.view(batch_size, 2, -1, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

#         # Conjugate-transpose of K for dot-product attention
#         # Shape: (B, 2, heads, L, head_dim) -> (B, 2, heads, head_dim, L)
#         proj_k_conjT = proj_k_multih.transpose(-2, -1).contiguous()
#         proj_k_conjT[:, 1] *= -1 # Negate imaginary part at dimension 1

#         # Compute dot-product attention scores within each head
#         if self.args.metric_type == 'RealDotProduct':
#             qk_complex = batched_complex_matmul_full(proj_q_multih, proj_k_conjT)
#             attn_scores = F.softplus(self.tau) * qk_complex[:, 0]
# #         elif self.args.metric_type == 'MagDotProduct':
# #             qk_complex = batched_complex_matmul_full(proj_q_multih, proj_k_conjT)
# #             attn_scores = F.softplus(self.tau) * torch.sum(torch.abs(qk_complex)**2, dim=1)
#         elif self.args.metric_type == 'InverseMahalanobis':
#             Q_real = proj_q_multih[:, 0]
#             Q_imag = proj_q_multih[:, 1]
#             K_real = proj_k_multih[:, 0]
#             K_imag = proj_k_multih[:, 1]
            
#             # The rest of the logic from compute_residual_norm is integrated here.
#             Q_mag_sq = (Q_real**2 + Q_imag**2).sum(dim=-1)
#             K_mag_sq = (K_real**2 + K_imag**2).sum(dim=-1)

#             K_real_T = K_real.transpose(-1, -2)
#             K_imag_T = K_imag.transpose(-1, -2)

#             term_RR = torch.matmul(Q_real, K_real_T)
#             term_II = torch.matmul(Q_imag, K_imag_T)
#             real_dot_product_QK = term_RR + term_II

#             Q_mag_sq_expanded = Q_mag_sq.unsqueeze(-1)
#             K_mag_sq_expanded = K_mag_sq.unsqueeze(-2)

#             R_squared_magnitude = Q_mag_sq_expanded + K_mag_sq_expanded - 2 * real_dot_product_QK

#             attn_scores = - F.softplus(self.tau) * torch.log(self.noise_floor**2 + R_squared_magnitude + self.args.epsilon)

        
#         # attn_scores shape: (batch_size, num_heads, L_q, L_k)

#         # Apply causal mask. The mask is broadcasted to the heads and batch dimensions.
#         attn_scores = attn_scores.masked_fill(self.causal_mask[:, :, :attn_scores.size(-2), :attn_scores.size(-1)] == 0, float('-inf'))

#         # Scale and normalize scores
#         scaled_attn_scores = attn_scores / self.scale
#         attn_weights = F.softmax(scaled_attn_scores, dim=-1)

#         # To perform a weighted sum with complex values, we treat the real weights
#         # as a complex number with a zero imaginary part and multiply with the complex values.
#         # Shape change for attn_weights: (B, 1, heads, L) -> (B, 2, heads, 1, L)
#         attn_weights_complex = torch.stack((attn_weights, torch.zeros_like(attn_weights).to(attn_weights.device)), dim=1)

#         # Weighted sum of values.
#         # (B, 2, heads, 1, L) x (B, 2, heads, L, head_dim) -> (B, 2, heads, 1, head_dim)
#         est_v = batched_complex_matmul_full(attn_weights_complex, proj_v_multih)
        
#         # Concatenate heads and project to the final output dimension
#         # (B, 2, heads, 1, head_dim) -> (B, 2, 1, heads*head_dim) -> (B, 2, 1, hidden_dim)
#         context_vector = est_v.view(batch_size, 2, -1, self.hidden_dim)
        
#         # Final output projection
#         out = self.W_o(context_vector)
        
#         if using_real:
#             out = out[:, 0]

#         return out, attn_weights

