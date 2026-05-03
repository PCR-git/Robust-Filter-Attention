import numpy as np
import torch

###############################################################
###############################################################

def construct_mapping(X, m, d_e, args):
    """
    Constructs a pair of complex-valued projection matrices (represented as real/imaginary 
    tensors) to map between an observation space of size 'm' and an embedding space of 'd_e'.
    
    The 'Up' projection (Pu) is randomly initialized, and the 'Down' projection (Pd) 
    is derived as its Moore-Penrose pseudoinverse to ensure a stable, least-squares 
    mapping back to the original space.
    """

    # Initialize 'Up' projection matrix (Pu) with Gaussian noise
    Pu = torch.randn(2, d_e, m).to(args.device)
    # Pre-allocate 'Down' projection matrix (Pd)
    Pd = torch.zeros(2, m, d_e).to(args.device)

    # Create a complex representation of Pu for linear algebra operations
    Pu_complex = Pu[0] + 1j * Pu[1]
    # Compute the pseudoinverse of the projection to ensure reconstruction stability
    Pd_complex = torch.linalg.pinv(Pu_complex)
    # Extract components back into the stacked real format
    Pd[0] = Pd_complex.real
    Pd[1] = Pd_complex.imag

    # Add a singleton head/batch dimension to facilitate broadcasting in AFA layers
    Pu = Pu.unsqueeze(1)
    Pd = Pd.unsqueeze(1)

    return Pu, Pd