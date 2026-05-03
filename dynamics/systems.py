import numpy as np
import torch
import torch.nn as nn

from utils import complex_matmul

##########################################################################################
##########################################################################################

class linear_spiral(nn.Module):
    """
    Stable 2D linear system
    Given current position, outputs current velocity
    """

    def __init__(self,params):
        super().__init__()
        
        # System parameters:
        D = params[0]
        S = params[1]
        Si = params[2]

        D = D.unsqueeze(1)
        S = S.unsqueeze(1)
        Si = Si.unsqueeze(1)

        self.A = complex_matmul(S,complex_matmul(D,Si))[0] # State transition matrix

    def forward(self, t, x):
        out = torch.matmul(self.A, x.squeeze())

        return out

##########################################################################################
##########################################################################################
    
class linear_spiral_3D(nn.Module):
    """
    Stable 2D linear system
    Given current position, outputs current velocity
    """

    def __init__(self,params,device):
        super().__init__()
        
        # System parameters:
        D = torch.zeros(2,3,3).to(device)
        S = torch.zeros(2,3,3).to(device)
        Si = torch.zeros(2,3,3).to(device)

        D[:,0:2,0:2] = params[0]
        D[0,2,2] = -1.0

        S[:,0:2,0:2] = params[1]
        Si[:,0:2,0:2] = params[2]
        S[0,2,2] = 1.0
        Si[0,2,2] = 1.0

        self.D = D.unsqueeze(1)
        self.S = S.unsqueeze(1)
        self.Si = Si.unsqueeze(1)
        
        self.A = complex_matmul(self.S,complex_matmul(self.D,self.Si))[0] # State transition matrix

    def forward(self, t, x):
        out = torch.matmul(self.A, x.squeeze())

        return out

##########################################################################################
##########################################################################################

class Lorenz(nn.Module):
    """
    Lorenz System
    Given current position, outputs current velocity
    """

    def __init__(self,params,device):
        super().__init__()
        
        # System parameters:
        self.sigma = params[0]
        self.beta = params[1]
        self.rho = params[2]
        self.device = device

    def forward(self, t, x):

        out = torch.tensor([self.sigma*(x[1]-x[0]), x[0]*(self.rho-x[2])-x[1], x[0]*x[1]-self.beta*x[2]]).to(self.device) # Equations of motion

        return out

##########################################################################################
##########################################################################################
    
def rand_coupling_matrix(n, omega, seed, device):
    """
    Get random coupling matrix, for use in Van der Pol Oscillator
    """

    torch.manual_seed(seed)

    u = torch.randperm(n * n)[:n]
    xi = u // n
    xj = u % n

    G = torch.zeros((n, n), device=device)

    G[xi, xj] = -1.0 * torch.rand(n, device=device)
    G[range(n), range(n)] = omega
    G[range(n - 1), range(1, n)] = -1.0 * torch.rand(n - 1, device=device)
    G[n - 1, 0] = -1.0 * torch.rand(1, device=device)

    return G

##########################################################################################
##########################################################################################
    
class Van_der_Pol_osc:
    """
    Van der Pol Oscillator
    """
    
    def __init__(self, params, K):
        self.mu = params[0]
        self.K = K

    def __call__(self, t, X):
        dim = len(X) // 2
        x = X[:dim]
        y = X[dim:]

        dx = y
        dy = self.mu * (1 - x**2) * y - self.K @ x  # K @ x is matrix-vector product

        return torch.cat((dx, dy), dim=0)
