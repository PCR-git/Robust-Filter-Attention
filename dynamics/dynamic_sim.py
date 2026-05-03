import numpy as np
import torch

from utils import complex_matmul

##########################################################################################
##########################################################################################

def simulate_stochastic_LTI(A, x0, N_t, args, sigma_process=0.1, sigma_measure=1):
    """
    Simulates a linear time-invariant system with zero-mean Gaussian noise using Euler integration.
    """
    m = args.m
    dt = args.dt
    device = args.device

    # Precompute noise
    process_noise = sigma_process * torch.randn((N_t+1, m, 1), device=device)
    measurement_noise = sigma_measure * torch.randn((N_t+1, m, 1), device=device)

    # Allocate trajectory array
    X = torch.zeros((N_t+1, m, 1), device=device)

    # Initial condition with optional initial process noise
    x = x0
    X[0] = x  # Store initial state as first timestep

    # Simulate over time (start from t = 1)
    for t in range(1, N_t+1):
        xp = torch.matmul(A, x) # Compute velocity
        x = x + xp * dt # Euler integration
        x += process_noise[t] * np.sqrt(dt) # Add process noise
        X[t] = x # Append current state to array

    X_measure = X + measurement_noise # Add measuremnt noise
        
    return X, X_measure

##########################################################################################
##########################################################################################
    
# (This function is faster and more accurate)
def simulate_diagonalized_stochastic_LTI(D, S, Si, x0, t_vector, sigma_process, sigma_measure, device):
    """
    Simulates a diagonalizable LTI SDE using the analytical exact discretization.
    Utilizes complex eigen-parameters to ensure zero numerical drift.

    Args:
        D (Tensor): Complex diagonal matrix [2, m, m] (Real/Imag parts).
        S (Tensor): Complex eigenvector matrix [2, m, m].
        Si (Tensor): Complex inverse eigenvector matrix [2, m, m].
        x0 (Tensor): Initial state vector [1, m, 1].
        t_vector (Tensor): Timestamps for sampling.
        sigma_process (float): Diffusion coefficient for process noise.
        sigma_measure (float): Standard deviation for measurement noise.
        device (torch.device): Computation device.
    """
    # Ensure shapes are correct for 2D system
    m = x0.shape[1]
    num_steps = t_vector.shape[0]
    dt = t_vector[1] - t_vector[0]

    # Extract physical dynamics from the diagonal matrix D
    # mu: decay rates (Real), omega: rotational frequencies (Imag)
    mu = D[0].diagonal()
    omega = D[1].diagonal()
    
    # Compute the discrete-time transition operator in the eigenbasis: exp(D * dt)
    # For lambda = mu + i*omega, the operator is exp(mu*dt) * (cos(w*dt) + i*sin(w*dt))
    phi_real = torch.exp(mu * dt) * torch.cos(omega * dt)
    phi_imag = torch.exp(mu * dt) * torch.sin(omega * dt)
    Phi_D = torch.stack([torch.diag(phi_real), torch.diag(phi_imag)]) # Shape [2, m, m]

    # Transform the transition operator back to the original coordinate frame
    # Phi_complex = S @ Phi_D @ Si
    Phi_complex = complex_matmul(S, complex_matmul(Phi_D, Si))
    # For a real-valued system, the resulting transition matrix Phi is the real part
    Phi = Phi_complex[0] # Shape [m, m]

    # Compute the integrated process noise variance (DLE solution) per eigenvalue
    # This accounts for the decay/rotation of noise within the interval dt
    q_diag = torch.zeros(m, device=device)
    for i in range(m):
        # mu is typically negative for stable systems in your D definition
        decay_const = -mu[i] 
        if abs(decay_const) > 1e-8:
            # OU-process variance: (sigma^2 / 2mu) * (1 - exp(-2mu * dt))
            q_diag[i] = (sigma_process**2 / (2 * decay_const)) * (1 - torch.exp(-2 * decay_const * dt))
        else:
            # Brownian motion limit as decay approaches zero
            q_diag[i] = (sigma_process**2) * dt
            
    # Scale the stochastic increment by the integrated standard deviation
    q_dt_std = torch.sqrt(q_diag)

    # Pre-allocate trajectory tensors
    X = torch.zeros(num_steps, m, 1, device=device)
    X[0] = x0.squeeze(0) # Shape [m, 1]
    
    # Iterative trajectory generation using exact discretization
    for i in range(1, num_steps):
        # Sample independent Gaussian noise in the complex eigenbasis
        # We divide by sqrt(2) to distribute variance across real and imaginary parts
        z_real = torch.randn(m, 1, device=device) * q_dt_std.unsqueeze(-1) / np.sqrt(2)
        z_imag = torch.randn(m, 1, device=device) * q_dt_std.unsqueeze(-1) / np.sqrt(2)
        Z_increment = torch.stack([z_real, z_imag], dim=0) # Shape [2, m, 1]
        
        # Project eigen-noise back to the physical basis: noise_phys = Re(S @ Z)
        # This ensures noise respects the non-orthogonal geometry of the system
        noise_complex = complex_matmul(S, Z_increment)
        noise_physical = noise_complex[0] # Shape [m, 1]
        
        # Exact update: x[t] = Phi @ x[t-1] + noise
        X[i] = torch.matmul(Phi, X[i-1]) + noise_physical

    # Generate observations by adding measurement noise (Sensor error)
    X_measure = X + sigma_measure * torch.randn_like(X)
    
    return X, X_measure

##########################################################################################
##########################################################################################

class DynamicSim:
    """
    General dynamic simulation class
    Uses simple Euler update
    """
    
    def __init__(self, device):
        self.device = device    # Device
        self.model = None            # Placeholder for a callable model

    def set_model(self, model):
        """
        Set the dynamical system model (must be a callable: model(t, X)).
        """
        self.model = model

    def eq_of_Motion(self, t, X):
        """
        Evaluate the equations of motion. Must return dx/dt.
        """
        if self.model is None:
            raise NotImplementedError("No model set for equation of motion.")
        return self.model(t, X)

    def simulate_ODE(self, x0, tf, t0, dt, sigma_process, sigma_process_0, sigma_measure):
        m = x0.shape[0]
        N_t = int((tf - t0) / dt) + 1  # Number of time steps
        t_v = torch.linspace(t0, tf, N_t, device=self.device)

        # Precompute noise
        process_noise = sigma_process * torch.randn((N_t, m), device=self.device)
        measurement_noise = sigma_measure * torch.randn((N_t, m), device=self.device)

        # Allocate trajectory and velocity arrays
        X = torch.zeros((N_t, m), device=self.device)
        Xp = torch.zeros((N_t, m), device=self.device)

        # Initial condition with optional initial process noise
        x = x0 + sigma_process_0 * torch.randn((m,), device=self.device)
        X[0, :] = x  # Store initial state

        # Simulate over time (start from t = 1)
        for i in range(1, N_t):
            t = t_v[i]
            xp = self.eq_of_Motion(t, x) # Compute velocity
            x = x + xp * dt # Euler update
            x = x + process_noise[i] * torch.sqrt(torch.tensor(dt)) # Add process noise
            X[i, :] = x # Concatenate to trajectory (array of states)
            Xp[i, :] = xp # Array of velocities

        X_measure = X + measurement_noise
        
        return X.unsqueeze(-1), X_measure.unsqueeze(-1), Xp.unsqueeze(-1)  # Return true and noisy trajectory
    