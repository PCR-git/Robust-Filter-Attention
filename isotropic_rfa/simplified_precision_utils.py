
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

##########################################################################################
##########################################################################################

def get_safe_exp_tot(t_measure, lambda_h, args):
    """
    Computes numerically stable exponents (lambda * t) for the SDE solution.
    
    Clamps the real part to prevent vanishing or exploding gradients 
    during exponential state transitions.
    """

    lambda_real = lambda_h[0]
    lambda_imag = lambda_h[1]
    
    # Broadcast timestamps to match head/feature dimensions
    t_exp = t_measure.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4).squeeze(0)
    
    # Compute real part and clamp for stability
    exp_tot_real = lambda_real * t_exp
    exp_tot_real_safe = torch.clamp(exp_tot_real, min=args.min_exponent, max=args.max_exponent)
    
    # Compute imaginary part (rotational phase)
    exp_tot_imag = lambda_imag * t_exp
    
    # Combine into a single complex exponent tensor
    exp_tot_safe = torch.stack((exp_tot_real_safe, exp_tot_imag))

    return exp_tot_safe

##########################################################################################
##########################################################################################

def compute_exp_kernel_isotropic(omega, t_measure, exp_rel_safe):
    """
    Computes the complex-valued rotation (Phi_tilde) and 
    the real-valued decay (E_rel). Together, they form the state transition 
    matrix exponential: exp(Lambda * delta_t).

    Args:
        omega: Rotational frequencies (Im(lambda)) for each head/feature [H, d_v_head//2].
        t_measure: Timestamps [seq_len].
        exp_rel_safe: Clamped real-valued decay exponents (Re(lambda) * Delta_T) [L, L, H].

    Returns:
        Phi_tilde_plus: Complex rotation components [seq_len, n_heads, d_v_head//2, 2].
                        Stored as (cos, sin) pairs for use in apply_interleaved_rope.
        E_rel: Real-valued decay factors [1, L, H].
               Represents e^{-mu * |t_i - t_j|}.
    """
    
    # --- Compute Rotational Kernel (Phase) ---
    # Reshape timestamps for broadcasting: [B, L, 1, 1]
    t_exp_rot = t_measure.unsqueeze(-1).unsqueeze(-1)
    
    # Phase angle theta = omega * t
    omega_t = omega * t_exp_rot
    
    # Store rotation as (cos(theta), sin(theta)) to avoid complex tensor overhead.
    # This represents the Unitary part of the dynamics (Phi_tilde).
    Phi_tilde_plus = torch.stack((torch.cos(omega_t), torch.sin(omega_t)),axis=-1)
    
    # --- Compute Decay Kernel (Magnitude) ---
    # E_rel represents the 'memory' factor
    # It scales the attention weights based on temporal distance.
    E_rel = torch.exp(exp_rel_safe)

    return Phi_tilde_plus, E_rel

##########################################################################################
##########################################################################################

def compute_covariance_matrix(mu, sigma_squared, eta_squared, gamma_squared, E_rel, args, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Computes the L x L isotropic covariance kernel V_ij = alpha * E_rel^2 + beta.
    This is the simplified analytic solution to the DLE used when mu > 0.
    """
    
    # Precompute shared denominators to avoid redundant ops
    denom = 2 * mu.squeeze(0)
    
    # alpha = eta^2 - sigma^2 / (2 * mu). Scales the time-decaying noise component.
    alpha = (eta_squared - sigma_squared/denom).unsqueeze(0).unsqueeze(1)
    
    # beta = gamma^2 + sigma^2 / (2 * mu). The stationary/residual noise floor.
    beta = (sigma_squared/denom + gamma_squared).unsqueeze(0).unsqueeze(1)
    
    # V_ij [L, L, H]: Combined uncertainty prior
    V_ij = alpha * E_rel**2 + beta
    
    return V_ij.squeeze(0) # [L, L, H]

# -----------------------------------------------

def compute_covariance_matrix_LHopital(mu_h, delta_t, E_rel, exp_rel_safe, sigma_squared, eta_squared, gamma_squared, args, epsilon: float = 1e-5, cov_epsilon: float = 1e-2) -> torch.Tensor:
    """
    Computes the analytic covariance kernel V_ij for the Isotropic RFA model.
    
    The covariance follows the derivation in Section 3.3.1:
    V_ij = sigma^2 * (1 - e^{-2mu|delta_t|}) / (2mu) + eta^2 * e^{-2mu|delta_t|} + gamma^2
    
    This implementation handles the 'Unitary Limit' (mu -> 0) using L'Hôpital's rule 
    to prevent division by zero, transitioning the kernel from an OU-process 
    variance to a Brownian motion variance (linear growth in Delta_T).

    Args:
        mu: Real decay rate (Re(lambda)).
        delta_T: Temporal distance matrix |t_i - t_j|.
        exp_rel_safe: Clamped exponent (mu * Delta_T) for numerical stability.
        sigma_squared: Process noise (diffusion).
        eta_squared: Key measurement noise.
        gamma_squared: Query measurement (anchor) noise.
        epsilon: Threshold for L'Hôpital transition.

    Returns:
        V_ij: The total isotropic covariance matrix [L, L, n_heads].
    """

    sigma_h = sigma_squared.view_as(mu_h)
    eta_h = eta_squared.view_as(mu_h)
    gamma_h = gamma_squared.view_as(mu_h)
    
    # Term 1: Propagated Process Noise (Brownian/OU component)
    numerator = -torch.expm1(2 * exp_rel_safe)
#     numerator1 = 1 - E_rel**2 # Faster but potentially less accurate
    
    # L'Hopital Masking
    mask = torch.abs(mu_h) < epsilon
    denom_safe = torch.where(mask, torch.ones_like(mu_h), 2 * mu_h)
    
    stable_frac = torch.where(mask, delta_t, numerator / denom_safe)
    term1 = sigma_h * stable_frac
    
    # Term 2: Decay-weighted Key Measurement Noise
    term2 = eta_h * E_rel**2
    
    # Term 3: Stationary Query Measurement Noise (Anchor noise)
    term3 = gamma_h
    
    # Total Prior Covariance V_ij
    V_ij = term1 + term2 + term3 + cov_epsilon
    
    return V_ij

##########################################################################################
##########################################################################################

def compute_covariance_hybrid_regime(mu_h, delta_t, E_rel, exp_rel_safe, 
                                       process_parameter, eta_squared, gamma_squared, 
                                       args, epsilon: float = 1e-5, 
                                       cov_epsilon: float = 1e-2) -> torch.Tensor:
    """
    Computes the prior covariance V_ij by switching between stationary 
    and non-stationary regimes based on the decay rate mu.
    
    Regime 1 (mu > epsilon): Stationary OU-process. 
        The process_parameter represents the memory floor (V_inf).
    Regime 2 (mu < epsilon): Non-stationary Brownian motion. 
        The process_parameter represents the diffusion slope (sigma^2).
    """

    # process_parameter acts as V_inf for dissipative heads 
    # and as sigma^2 for unitary heads.
    p_h = process_parameter.view_as(mu_h)
    eta_h = eta_squared.view_as(mu_h)
    gamma_h = gamma_squared.view_as(mu_h)
    
    # --- Term 1: Process Noise Component ---
    mask = torch.abs(mu_h) < epsilon
    
    # Dissipative regime: V_inf * (1 - exp(-2 * mu * delta_t))
    # Note: exp_rel_safe is assumed to be -mu * delta_t
    diffusion_factor = -torch.expm1(2 * exp_rel_safe)
    dissipative_term = p_h * diffusion_factor
    
    # Unitary regime: sigma^2 * delta_t
    unitary_term = p_h * delta_t
    
    term1 = torch.where(mask, unitary_term, dissipative_term)
    
    # --- Term 2 & 3: Measurement Noise ---
    term2 = eta_h * E_rel**2
    term3 = gamma_h
    
    # Total Prior Covariance V_ij
    V_ij = term1 + term2 + term3 + cov_epsilon
    
    return V_ij

##########################################################################################
##########################################################################################

def compute_covariance_matrix_spectral_full(
    mu_h, delta_t, exp_rel_safe, E_rel, 
    sigma_sq, eta_sq, gamma_sq,
    noise_params_raw,
    args, epsilon=1e-4, cov_epsilon=1e-2
):
    """
    Computes the Generalized Spectral Precision Kernel
    Handles analytical solutions for:
    1. Constant (White) Noise
    2. Exponential (Colored) Noise
    3. Periodic (Harmonic) Noise
    (This implementation allows a single set of learnable coefficients for each:
    sigma_sq for constant, beta_exp and c_exp for exponential, and a_per, b_per, omega_per for periodic)
    
    Includes 2D L'Hôpital stability for (mu, omega, beta) -> 0.
    """
    
    # --- Constrain Parameters ---
    c_exp = F.softplus(noise_params_raw['c_exp']) # Exponential coeff
    beta_exp = F.softplus(noise_params_raw['beta_exp']) # Exponent
    omega_per = noise_params_raw['omega_per'] # Frequency

    # --- Scale a_per and b_per so that V also stays positive ---
    # By calculating a global_floor based on the minimum of the initial and steady-state variances, and scaling the periodic amplitude using a tanh saturation, we ensure that the sinusoidal terms can never dip the total covariance into negative territory.
    v_zero = eta_sq + gamma_sq # The sensing resolution limit (initial noise)
    v_inf = (sigma_sq / (2 * mu_h + args.epsilon)) + gamma_sq # The dynamical drift limit (steady-state noise)
    global_floor = torch.min(v_zero, v_inf) * 0.95 # Use a safety margin to prevent ever reaching exactly zero.
    # Treats (a, b) as a single complex amplitude |a + ib|.
    a_raw = noise_params_raw['a_per']
    b_raw = noise_params_raw['b_per']
    amplitude = torch.sqrt(a_raw**2 + b_raw**2 + 1e-8) # Max amplitude of periodic components
    # Smoothly scale the amplitude to stay within the budget.
    scale = (global_floor / amplitude) * (torch.tanh(amplitude / global_floor))
    a_per = a_raw * scale
    b_per = b_raw * scale
    # ------------------------------
    
    # --- Constant Process Noise (Standard OU/Brownian) ---
    # Limit as mu -> 0: sigma_sq * delta_t
    numerator_const = -torch.expm1(2 * exp_rel_safe)
    mask_const = torch.abs(mu_h) < epsilon
    denom_const = torch.where(mask_const, torch.ones_like(mu_h), 2 * mu_h)
    
    V_const = sigma_sq * torch.where(
        mask_const, 
        delta_t, 
        numerator_const / denom_const
    )

    # --- Exponential Structured Noise ---
    # Limit as (2mu + beta) -> 0: c_exp * delta_t * exp(-2mu * delta_t)
    rate_exp = 2 * mu_h + beta_exp
    mask_exp = torch.abs(rate_exp) < epsilon
    
    # Factorized form for stability: c * E^2 * [ (exp(rate*dt) - 1) / rate ]
    exp_rate_dt_m1 = torch.expm1(rate_exp * delta_t)
    denom_exp = torch.where(mask_exp, torch.ones_like(rate_exp), rate_exp)
    
    stable_exp_frac = torch.where(mask_exp, delta_t, exp_rate_dt_m1 / denom_exp)
    V_exp = c_exp * (E_rel**2) * stable_exp_frac

    # --- Periodic Structured Noise ---
    # Limit as (mu, omega) -> 0: a_per * delta_t
    # This aligns confidence with learned seasonalities (e.g. 24h cycles in ETT).
    two_mu = 2 * mu_h
    r_sq = two_mu**2 + omega_per**2
    mask_per = r_sq < epsilon
    
    sin_wt = torch.sin(omega_per * delta_t)
    cos_wt = torch.cos(omega_per * delta_t)
    
    # Numerator terms
    diff_cos_exp = cos_wt - E_rel**2
    term_a_num = a_per * (two_mu * diff_cos_exp + omega_per * sin_wt)
    term_b_num = b_per * (two_mu * sin_wt - omega_per * diff_cos_exp)
    
    denom_per = torch.where(mask_per, torch.ones_like(r_sq), r_sq)
    
    V_periodic = torch.where(
        mask_per,
        a_per * delta_t, # Taylor expansion limit
        (term_a_num + term_b_num) / denom_per
    )

    # --- Measurement Noise & Floor ---
    # eta: Key-side noise (decays with distance)
    # gamma: Query-side noise (static anchor floor)
    V_meas = eta_sq * E_rel**2 + gamma_sq

    # --- 5. Total Combined Covariance ---
    V_total = V_const + V_exp + V_periodic + V_meas + cov_epsilon
    
    # Final Safety Check: Ensure the prior variance is strictly positive
    # Negative variance would imply 'imaginary' uncertainty.
    return torch.clamp(V_total, min=cov_epsilon)

##########################################################################################
##########################################################################################

def compute_covariance_matrix_residual_diffusion(mu_h, delta_t, exp_rel_safe, E_rel, sigma_squared, sigma_tilde_squared, eta_squared, gamma_squared, args, allow_BM_branch=True, cov_epsilon: float = 1e-2) -> torch.Tensor:
    """
    Computes analytic covariance V_ij using a Residual Diffusion formulation.
    (This is for trying to learn the decays.)
    
    To avoid the gradient instability of the 1/(2*mu) term in the DLE solution, 
    we decompose process noise into two branches: 
    1. A stationary OU branch (sigma_tilde) that vanishes as mu -> 0.
    2. A non-stationary Brownian branch (sigma) that provides the L'Hôpital limit 
       (linear growth in Delta_T) required for the Unitary Limit.
       
    This additive approach ensures a smooth, differentiable transition between 
    regimes without the 'ejection' gradients caused by mu in the denominator.
    """

    # Reshape params
    sigma_h = sigma_squared.view_as(mu_h)
    sigma_tilde_h = sigma_tilde_squared.view_as(mu_h)
    eta_h = eta_squared.view_as(mu_h)
    gamma_h = gamma_squared.view_as(mu_h)
    
    # Term 1: Propagated Process Noise (Residual Diffusion)
    # The OU branch captures mean-reversion; the BM branch handles the mu -> 0 limit.
    OU_branch = - sigma_tilde_h * torch.expm1(2 * exp_rel_safe)
    
    BM_branch = sigma_h * delta_t
    term1 = (OU_branch + BM_branch) if allow_BM_branch else OU_branch
    
     # Term 2: Decay-weighted Key Measurement Noise (eta^2 * e^{-2mu*dt})
    term2 = eta_h * E_rel**2
    
    # Term 3: Stationary Query Measurement Noise (Anchor noise floor)
    term3 = gamma_h
    
    # Total Prior Covariance V_ij
    V_ij = term1 + term2 + term3 + cov_epsilon

    return V_ij

##########################################################################################
##########################################################################################

def compute_covariance_matrix_measurement_gg_process(alpha_sq, beta_sq, zeta_sq, Delta_T, E_rel, args, allow_BM_branch=True, cov_epsilon: float = 1e-2) -> torch.Tensor:
    """
    Computes the L x L isotropic covariance kernel V_ij = alpha * E_rel^2 + beta.
    This is the simplified analytic solution to the DLE used when mu > 0.
    We pass in alpha_sq > 0, beta_sq > 0, which ensures measurement noise dominates process noise (low-pass filter)
    """
    
#     # Precompute shared denominators to avoid redundant ops
#     denom = 2 * mu.squeeze(0)
    
#     sigma_ss = sigma_squared/denom
    
#     # alpha = eta^2 - sigma^2 / (2 * mu). Scales the time-decaying noise component.
#     alpha = (eta_squared - sigma_ss).unsqueeze(0).unsqueeze(1)
    
#     # beta = gamma^2 + sigma^2 / (2 * mu). The stationary/residual noise floor.
#     beta = (sigma_ss + gamma_squared).unsqueeze(0).unsqueeze(1)
    
    # V_ij [L, L, n_heads]: Combined uncertainty prior
    
    OU_branch = alpha_sq * E_rel**2 + beta_sq
    
    if allow_BM_branch == 1:
        BM_branch = zeta_sq * Delta_T
        V_ij = OU_branch + BM_branch
    elif allow_BM_branch == 2:
        # Optionally, add in a BM branch
        BM_branch = zeta_sq * Delta_T
        BM_gate = torch.exp(-mu).view(1, 1, 1, -1)
        V_ij = (1 - BM_gate) * OU_branch + BM_gate * BM_branch
    else:
        V_ij = OU_branch
    
    return V_ij + cov_epsilon # [1, L, L, n_heads]

##########################################################################################
##########################################################################################

# def build_factorized_kernels(Phi_tilde_minus_k, Phi_tilde_minus_v, Q, K, V, args):
#     """
#     Build factorized kernels for use in simplified RFA
#     """
    
#     Q_tilde = batched_complex_hadamard(Phi_tilde_minus_k, Q)
#     K_tilde = batched_complex_hadamard(Phi_tilde_minus_k, K)
    
#     if args.rotate_values == 1:
#         V_tilde = batched_complex_hadamard(Phi_tilde_minus_v, V)
#     else:
#         V_tilde = V
#         print('Not rotating values.')
    
#     return Q_tilde, K_tilde, V_tilde

##########################################################################################
##########################################################################################

def compute_residual_norm_isotropic(Q_tilde, K_tilde, E_rel_k, args):
    """
    Computes the stable squared residual norm |R_qk|^2 using the factorized formula 
    from the Isotropic Robust Filter Attention (RFA).

    |R_qk|^2 = |Q_tilde|^2 + E_qk[i,j]^2 * |K_tilde|^2 - 2 * E_qk[i,j] * Re(Q_tilde^* K_tilde)

    Inputs:
        Q: Unrotated query
        Q_tilde (torch.Tensor): Rotated Query vectors (Phi^- * Q). [B, 2, N, d_k, H]
        K_tilde (torch.Tensor): Rotated Key vectors (Phi^- * K). [B, 2, N, d_k, H]
        E_rel_k (torch.Tensor): Decay matrix E_qk[i,j] = e^(alpha*(t_i-t_j)). [N, N, H]

    Returns:
        R_qk_abs_squared: Tensor of shape (B, m, m, H), per-head squared residuals.
    """

    # Calculate total magnitude squared: |Z_tilde|^2 = sum_d (Re^2 + Im^2)
    # Sum over Real/Imag (dim -1) and Feature (dim -2) dimensions.
    
    # We could normalize either Q or Q_tilde
    Q_mag_sq_sum = torch.sum(Q_tilde**2, dim=[-1, -2]) # Normalize rotated Q_tilde

    # --- Term 1: |Z_q|^2 (Broadcast over Keys) ---
    T1 = Q_mag_sq_sum.unsqueeze(2)

    # --- Term 2: E_qk^2 * |Z_k|^2 (Broadcast over Queries) ---
    K_mag_sq_sum = torch.sum(K_tilde**2, dim=[-1, -2])

    T2 = (E_rel_k**2) * K_mag_sq_sum.unsqueeze(1) # [B, N, N, H]

    # --- Term 3: -2 * E_rel_k * Re(Q_tilde^H K_tilde) ---
    # Term 3: Cross-term Re(Q*K) using matmul for O(N^2 d) efficiency
    
#     # Extract real and imaginary components for concatenation
#     Q_tilde_re, Q_tilde_im = Q_tilde[..., 0], Q_tilde[..., 1]
#     K_tilde_re, K_tilde_im = K_tilde[..., 0], K_tilde[..., 1]
#     # Flatten the complex/feature dimensions to perform a standard dot product.
#     # We move Heads (dim 3) to the batch position for matmul optimization.
#     Q_c = torch.cat((Q_tilde_re, Q_tilde_im), axis=-1).permute(0,2,1,3).contiguous() # [B, H, N, 2d]
#     K_c = torch.cat((K_tilde_re, K_tilde_im), axis=-1).permute(0,2,3,1).contiguous() # [B, H, 2d, N]
#     # Compute dot product and permute back to [B, m, m, H]
#     dot_product = torch.matmul(Q_c,K_c).permute(0,2,3,1).contiguous()

    dot_product = torch.einsum('bihdc,bjhdc->bijh', Q_tilde, K_tilde)

    # Apply the temporal decay factor (E_rel_k) derived from the SDE transition
    T3 = 2 * E_rel_k * dot_product # [B, N, N H]

    if args.use_full_residual_norm:
        # Full residiual norm: |R_qk|^2 = |Q|^2 + |K|^2 - 2Re(Q*K)
        R_qk_abs_squared = T1 + T2 - T3
    else:
        # Fallback to standard dot-product attention
        R_qk_abs_squared = - T3 / 2

    return R_qk_abs_squared
