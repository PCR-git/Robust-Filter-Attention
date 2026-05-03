import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import math

# from utils import complex_exp, batched_complex_matmul, batched_complex_hadamard, batched_complex_matmul_full
# from precision_attention import compute_exp_kernel, compute_covariance_kernel
from model import init_complexlinear, init_complex_matrix, init_spectrally_coupled_rope
    
##########################################################################################
##########################################################################################

################################## POSITIONAL ENCODING ###################################

##########################################################################################
##########################################################################################

class RealPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for real-valued inputs,
    as introduced in "Attention Is All You Need" (Vaswani et al., 2017).

    Adds deterministic, non-learned position-dependent signals to the input.
    """

    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model (int): Dimensionality of the input embeddings.
            max_len (int): Maximum sequence length supported.
        """
        super().__init__()
        
        # Create zero tensor of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)  # (L, D)

        # Position indices: shape (L, 1)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # Frequency terms for sinusoids: shape (D/2,)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))

        # Apply sin to even dimensions (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cos to odd dimensions (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, L, D) to broadcast across batches
        pe = pe.unsqueeze(0)

        # Register as buffer (non-trainable, saved with model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encoding to the input tensor.

        Args:
            x (Tensor): Input of shape (B, L, D)

        Returns:
            Tensor: Positionally encoded input of shape (B, L, D)
        """
        # Add positional encoding up to sequence length x.size(1)
        return x + self.pe[:, :x.size(1)]
    
##########################################################################################
##########################################################################################

class ComplexPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding adapted for complex-valued inputs.
    Generates real and imaginary components separately, maintaining the same
    structure across both even and odd indices.

    This is intended for inputs of shape (batch, 2, seq_len, embed_dim),
    where the second dimension holds the real and imaginary parts respectively.
    """

    def __init__(self, seq_len, embed_dim, device='cpu'):
        super().__init__()

        # Create a position index (seq_len, 1)
        position = torch.arange(seq_len).unsqueeze(1)

        # Compute frequency divisors (embed_dim/2,) using exponential decay
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * 
                             -(np.log(10000.0) / embed_dim))

        # Phase encoding: shape (seq_len, embed_dim/2)
        phase = position * div_term

        # Allocate real and imaginary positional embeddings
        pe_real = torch.zeros(seq_len, embed_dim)
        pe_imag = torch.zeros(seq_len, embed_dim)

        # Apply cosine to real part (both even and odd positions)
        pe_real[:, 0::2] = torch.cos(phase)
        pe_real[:, 1::2] = torch.cos(phase)  # optional: symmetric version

        # Apply sine to imaginary part (both even and odd positions)
        pe_imag[:, 0::2] = torch.sin(phase)
        pe_imag[:, 1::2] = torch.sin(phase)  # optional: symmetric version

        # Stack into shape (2, seq_len, embed_dim), where index 0 is real and 1 is imag
        pe = torch.stack([pe_real, pe_imag], dim=0)

        # Register as a buffer so it's saved in the model but not updated by optimizer
        self.register_buffer("pos_enc", pe.to(device))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input of shape (batch, 2, seq_len, embed_dim)
        
        Returns:
            torch.Tensor: Input with complex sinusoidal positional encoding added.
        """
        
        # Convert real-valued inputs to complex by appending zero imaginary parts, if necessary
        if x.size()[1] == 1:
            x = torch.cat((x, torch.zeros_like(x)),dim=1)
        
        # Broadcast positional encoding across batch dimension
        return x + self.pos_enc.unsqueeze(0)  # shape (1, 2, seq_len, embed_dim)

##########################################################################################
##########################################################################################

class LearnedRealPositionalEncoding(nn.Module):
    """
    Learned real-valued positional encoding for complex input tensors shaped (B, 2, L, D).
    The same real encoding is added to both the real and imaginary parts.
    """
    def __init__(self, seq_len, embed_dim):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # Create a learnable (L, D) real-valued positional encoding
        self.pe = nn.Parameter(torch.randn(seq_len, embed_dim) * 0.02)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, 2, L, D), complex-valued input

        Returns:
            x + positional encoding (broadcasted over batch and complex dims)
        """
        # Shape: (1, 1, L, D) to broadcast over batch and complex dims
        pe_expanded = self.pe.unsqueeze(0).unsqueeze(0)
        return x + pe_expanded

##########################################################################################
##########################################################################################

class LearnedComplexPositionalEncoding(nn.Module):
    """
    Learned complex-valued positional encoding for input of shape (B, 2, L, D).
    Encodes position using separate learned real and imaginary components.
    """

    def __init__(self, seq_len, embed_dim):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # Learnable complex positional encoding: shape (2, L, D)
        self.pe = nn.Parameter(torch.randn(2, seq_len, embed_dim) * 0.02)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, 2, L, D) — complex input

        Returns:
            x + learned complex positional encoding (broadcasted)
        """
        # Expand shape to (1, 2, L, D) for broadcasting over batch dim
        return x + self.pe.unsqueeze(0)
    
##########################################################################################
##########################################################################################

class RoPE(nn.Module):
    # --- FIXED RoPE IMPLEMENTATION ---
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            print('Error: embed dim must be even.')
            pass

        self.dim = dim
        self.max_seq_len = max_seq_len
        theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("theta", theta)
        self.register_buffer("positions", torch.arange(max_seq_len, dtype=torch.float))
        # angles.shape: [1, max_seq_len, dim/2]
        angles = torch.outer(self.positions, self.theta).unsqueeze(0)
        self.register_buffer("cos", torch.cos(angles))
        self.register_buffer("sin", torch.sin(angles))

    def _rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Applies the rotation matrix using the complex representation [x_r, x_i] -> [x_r*cos - x_i*sin, x_r*sin + x_i*cos].
        Assumes cos and sin are already expanded to the full feature dimension.
        """
        # x_r, x_i shape: [..., D_head/2]
        x_r, x_i = x[..., ::2], x[..., 1::2]
        
        # x_permuted shape: [..., D_head]. Equivalent to [-x_i, x_r]
        x_permuted = torch.cat((-x_i, x_r), dim=-1)

        # Result: x * cos + x_permuted * sin
        # The key change: cos and sin are used directly without repeating/interleaving, as this was done in forward.
        return x * cos + x_permuted * sin

    def forward(self, x: torch.Tensor, seq_len: int):
        # x shape: [B, L, H, D_head]
        cos_sliced = self.cos[:, :seq_len, :].to(x.device) # [1, L, D_head/2]
        sin_sliced = self.sin[:, :seq_len, :].to(x.device) # [1, L, D_head/2]
        
        # Expand D_head/2 -> D_head
        cos_final = cos_sliced.repeat_interleave(2, dim=-1)
        sin_final = sin_sliced.repeat_interleave(2, dim=-1)
        
        # Prepare for broadcasting across the Head dimension
        if x.ndim == 4:
            cos_final = cos_final.unsqueeze(2) # [1, L, 1, D_head]
            sin_final = sin_final.unsqueeze(2) # [1, L, 1, D_head]
        
        return self._rotate(x, cos_final, sin_final)
    
##########################################################################################
##########################################################################################    
    
class SCRoPE(nn.Module):
    def __init__(self, dim: int, n_heads: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError('Error: head dim must be even.')

        self.dim = dim
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        # --- Spectral Sharding ---
        # theta shape: [n_heads, head_dim/2]
        # Head 0 gets slowest frequencies, Head 7 gets fastest
        theta, _ = init_spectrally_coupled_rope(n_heads, dim // 2, base=base)
        
        self.register_buffer("theta", theta)
        self.register_buffer("positions", torch.arange(max_seq_len, dtype=torch.float))

        # Compute angles using broadcasting: [L, 1, 1] * [1, H, D/2] -> [L, H, D/2]
        angles = self.positions.view(-1, 1, 1) * self.theta.unsqueeze(0)
        
        # Buffers shape: [L, H, D/2]
        self.register_buffer("cos", torch.cos(angles))
        self.register_buffer("sin", torch.sin(angles))

    def _rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Standard interleaved RoPE logic: [x_r, x_i, x_r, x_i...]
        x_r, x_i = x[..., ::2], x[..., 1::2]
        x_permuted = torch.cat((-x_i, x_r), dim=-1)
        return x * cos + x_permuted * sin

    def forward(self, x: torch.Tensor, seq_len: int):
        # x shape: [B, L, H, D_head]
        # Slice the precomputed buffers: [L, H, D/2]
        cos_sliced = self.cos[:seq_len, :, :].unsqueeze(0) # [1, L, H, D/2]
        sin_sliced = self.sin[:seq_len, :, :].unsqueeze(0) # [1, L, H, D/2]
        
        # Expand D/2 -> D to match interleaved real/imaginary pairs
        cos_final = cos_sliced.repeat_interleave(2, dim=-1)
        sin_final = sin_sliced.repeat_interleave(2, dim=-1)
        
        return self._rotate(x, cos_final, sin_final)

##########################################################################################
##########################################################################################
    
def get_alibi_slopes(n):
        """Standard ALiBi slope calculation."""
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            slopes = get_slopes_power_of_2(n)
        else:
            # For non-power-of-2 heads, interpolate
            closest_power_of_2 = 2**math.floor(math.log2(n))
            slopes = get_slopes_power_of_2(closest_power_of_2) + \
                     get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:n-closest_power_of_2]
        
        return torch.tensor(slopes).view(1, n, 1, 1)
    
##########################################################################################
##########################################################################################   
    
class xPos(nn.Module):
    def __init__(self, n_heads, head_dim, max_seq_len=4096, smoothing=0.4, base=10000.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dim_target = head_dim // 2

        # Standard RoPE Frequencies (Identical for all heads)
        # Power-law distribution from base to 2pi
        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        
        # Standard xPos Decay Base (Linear Ramp across dimensions)
        # Resulting gamma is in [0, 1]. Lower index = more decay.
        gamma = (torch.arange(0, head_dim, 2).float() * smoothing + 0.4) / (head_dim * smoothing + 0.4)

        # Register as [1, 1, 1, D/2] for broadcasting across [B, L, H, D/2]
        self.register_buffer("theta", theta.view(1, 1, 1, -1))
        self.register_buffer("gamma", gamma.view(1, 1, 1, -1))
        self.register_buffer("positions", torch.arange(max_seq_len, dtype=torch.float))

    def forward(self, x, seq_len, is_key=False):
        # x shape: [B, L, H, D_head]
        batch_size, sequence_length, n_heads, head_dim = x.shape
        pos = self.positions[:seq_len].view(1, sequence_length, 1, 1) # [1, L, 1, 1]
        
        # Compute Phase: theta broadcasted over pos
        angles = pos * self.theta # [1, L, 1, D/2]
        
        # Compute Exponential Decay: gamma^n or gamma^-n
        # gamma is [1, 1, 1, D/2], pos is [1, L, 1, 1]
        exponent = -pos if is_key else pos
        decay = torch.pow(self.gamma, exponent) # [1, L, 1, D/2]
        
        # Apply to Trig components and expand D/2 -> D_head
        # Result shapes: [1, L, 1, D_head]
        cos = (torch.cos(angles) * decay).repeat_interleave(2, dim=-1)
        sin = (torch.sin(angles) * decay).repeat_interleave(2, dim=-1)

        # Standard Interleaved Rotary Logic
        x_r, x_i = x[..., ::2], x[..., 1::2]
        x_permuted = torch.cat((-x_i, x_r), dim=-1)
        
        return x * cos + x_permuted * sin
    
    

class LearnableRoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError('embed dim must be even.')

        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Initialize theta based on the standard RoPE log-linear schedule
        # We make it a Parameter so it's optimized during training
        theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)) # standard rope init
        
        theta = torch.randn(dim // 2) # Gaussian noise
        self.theta = nn.Parameter(theta) 
        
        # Positions remain a fixed buffer as they represent the time steps t_i
        self.register_buffer("positions", torch.arange(max_seq_len, dtype=torch.float))

    def _rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Complex representation: [x_r*cos - x_i*sin, x_r*sin + x_i*cos]
        x_r, x_i = x[..., ::2], x[..., 1::2]
        
        # Interleave the results to maintain the [r, i, r, i] structure
        out_r = x_r * cos - x_i * sin
        out_i = x_r * sin + x_i * cos
        
        # Stack and flatten to reconstruct the original D_head dimension
        return torch.stack([out_r, out_i], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor):
        # x shape: [B, L, H, D_head]
        seq_len = x.shape[1]
        
        # Dynamically compute angles using the learnable theta
        # angles.shape: [L, dim/2]
        angles = torch.outer(self.positions[:seq_len], self.theta)
        
#         plt.scatter(self.theta.detach().cpu().numpy())
#         plt.show()
        
        cos = torch.cos(angles) # [L, D_head/2]
        sin = torch.sin(angles) # [L, D_head/2]

        # Prepare for broadcasting: [1, L, 1, D_head/2]
        if x.ndim == 4:
            cos = cos.unsqueeze(0).unsqueeze(2)
            sin = sin.unsqueeze(0).unsqueeze(2)
        
        return self._rotate(x, cos, sin)
    
    