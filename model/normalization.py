import torch
import torch.nn as nn

##########################################################################################
##########################################################################################

################################## NORMALIZATION LAYERS ##################################

##########################################################################################
##########################################################################################

class ComplexRMSNorm(nn.Module):
    """
    Complex-valued Root Mean Square Layer Normalization (RMSNorm).
    
    Performs L2 norm normalization (RMS) on the complex vector, followed by 
    a complex affine transformation (scaling and shifting).
    
    """

    def __init__(self, feature_dim: int, n_head: int, eps: float = 1e-6, affine=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_head = n_head
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1,1,n_head,int(feature_dim/n_head),1))
#             self.beta = nn.Parameter(torch.ones(1,1,1,int(feature_dim/n_head),n_head))
        else:
            self.register_parameter('gamma', None)

    def forward(self, Z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Z: complex-valued input tensor

        Returns:
            Z_norm: normalized output tensor
        """
        
        # --- 1. Calculate the Complex RMS (Root Mean Square) ---
        
        # Calculate the squared magnitude of the complex vector: |Z|^2 = Z_r^2 + Z_i^2
        Z_squared = Z.pow(2)
        squared_norm = 2.0 * torch.mean(Z_squared,dim=[-2,-1],keepdims=True) # Use mean (multiply by two since each real-imag pair is a single complex number)
        
        # RMS is sqrt(Mean(|Z|^2)). Add epsilon for stability.
        rms = torch.rsqrt(squared_norm + self.eps)

        Z_norm = Z * rms

        if self.affine == True:
            Z_out = self.gamma * Z_norm
#             Z_out = self.gamma * Z_norm - self.beta
        else:
            Z_out = Z_norm
        
        return Z_out
