import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt

from utils import apply_interleaved_rope

from isotropic_rfa import compute_covariance_matrix, compute_covariance_matrix_LHopital, compute_covariance_hybrid_regime, compute_covariance_matrix_spectral_full, compute_covariance_matrix_residual_diffusion, get_safe_exp_tot
from isotropic_rfa import compute_exp_kernel_isotropic, compute_residual_norm_isotropic

from model import inv_softplus, inv_sigmoid, resolve_multihead_dims
from model import initialize_linear_layers, init_complex_matrix, init_complexlinear
from model import init_rope, init_spectrally_coupled_rope, init_decay_per_head, init_linear_bias_slopes
from model import ComplexLinearLayer, ComplexLinearHermitianLayer
from model import ComplexRMSNorm

# from model import apply_projection_mask

##########################################################################################
##########################################################################################

class MultiheadIsotropicRFA(nn.Module):
    def __init__(self, args, n_heads, input_dim, query_key_dim=None, value_dim=None, query_key_dim_total=None, value_dim_total=None):
        """
        Implements the Multi-Head Isotropic Robust Filter Attention (RFA) block.

        This module integrates a complex-valued, linear time-invariant Stochastic
        Differential Equation (LTI-SDE) model into the Transformer's attention mechanism.

        The architecture is designed for O(N^2 + Nd) memory complexity by
        enforcing isotropic decay and noise in the diagonalized eigenbasis.

        Key Learned Parameters (per head):
        - Dynamics (lambda_real, lambda_imag): Defines the SDE's decay and rotational frequencies.
        - Noise (sigma, eta, gamma): Defines process noise, measurement noise, and anchor noise.
        - Robustness (nu): Sets the statistical threshold for outlier suppression.
        - Inverse temperature (tau): Scales the final attention logits.

        The core function is to derive the attention matrix from the SDE's propagated
        uncertainty via the Differential Lyapunov Equation (DLE).
        """

        super().__init__()
        
        # max_len = self.args.seq_len
        max_len = 4096 # Or whatever your absolute upper limit is
        
        self.args = args
        
        self.register_buffer("t_measure", torch.arange(max_len))
        self.register_buffer("abs_idx", torch.abs(self.t_measure.view(max_len, 1) - self.t_measure.view(1, max_len)))
        if query_key_dim==None or value_dim==None or query_key_dim_total==None or value_dim_total==None:
            # Set query_key and value dims, depending on whether user provided total dims, or head dims
            query_key_dim, value_dim, query_key_dim_total, value_dim_total = resolve_multihead_dims(n_heads, query_key_dim, value_dim, query_key_dim_total=query_key_dim_total, value_dim_total=value_dim_total)

        # Store dimensions as instance attributes
        self.n_heads = n_heads
        self.d_e = input_dim
        self.d_k_head = query_key_dim
        self.d_v_head = value_dim
        self.d_k_total = query_key_dim_total
        self.d_v_total = value_dim_total

        ################################################

        # Linear Layers
        self.W_q = nn.Linear(self.args.d_e, self.args.d_k_total*2)
        self.W_k = nn.Linear(self.args.d_e, self.args.d_k_total*2)
        self.W_v = nn.Linear(self.args.d_e, self.args.d_v_total*2)
        self.W_o = nn.Linear(self.args.d_v_total*2, self.args.d_e)
        
        ################################################
        # Define LTI params
        target_ss_process_var = 0.1 # Initial steady state process noise
        target_measurement_var = 1.0 # Initial measurement noise
        # Initialize the model as a conservative smoother.
        ##########################
        # Initialize omega
        if self.args.use_complex_conj_constraint == True:
            dim_target_v = int(self.d_v_head/2)
        else:
            dim_target_v = self.d_v_head
            
        if self.args.use_SC_RoPE == True:
            omega_init_stacked_v, total_mu_v = init_spectrally_coupled_rope(self.n_heads, dim_target_v, b=self.args.damping)
        else:
            _, total_mu_v = init_spectrally_coupled_rope(self.n_heads, dim_target_v, b=self.args.damping)
            omega_init_stacked_v = init_rope(self.n_heads, dim_target_v)
        
#         omega_init_stacked_v = torch.randn(self.n_heads, dim_target_v) # Gaussian noise initialize of omega (not used)
        
        if self.args.zero_rotations == True:
            omega_init_stacked_v = torch.zeros_like(omega_init_stacked_v)
        if self.args.learn_rotations == True:
            self.omega_v = nn.Parameter(omega_init_stacked_v)
        else:
            self.register_buffer("omega_v", omega_init_stacked_v)
#         ##########################
        # Initialize decay
        if self.args.use_log_linear_decay == True and self.args.use_SC_RoPE == False:
            total_mu_v = init_decay_per_head(self.n_heads, min_decay=0.01, max_decay=0.95, zero_frac=0.25)
       
        if self.args.learn_decay == True:
            self.mu_v = nn.Parameter(inv_sigmoid(total_mu_v))
        else:
            self.register_buffer("mu_v", total_mu_v)
#         ##########################
        # Initialize noise params
        # This initialization creates a Low-Pass Filter effect at initialization.
        # The model will prefer to average across the history rather than reacting violently to every new token.
        initial_sigma_sq_v = 2 * total_mu_v.detach().clone() * target_ss_process_var # We initialize sigma^2 to a multiple of mu, to ensure that regardless of how fast a head decays, it starts with a consistent steady-state uncertainty.
        initial_sigma_sq_inverse_softplus_v = inv_softplus(initial_sigma_sq_v)
        
        self.sigma_v = nn.Parameter(initial_sigma_sq_inverse_softplus_v)
        self.sigma_tilde_v = nn.Parameter(torch.ones(self.n_heads) * target_ss_process_var)
        self.eta_v = nn.Parameter(torch.ones(self.n_heads) * target_measurement_var)
        self.gamma_v = nn.Parameter(torch.ones(self.n_heads) * target_measurement_var)

        # It should actually be this, but we didn't realize util ablations were already running, so kept the above for consistency:
#         self.noise_params_v = nn.ParameterDict({
#             'sigma': nn.Parameter(initial_sigma_sq_inverse_softplus_v),
#             'sigma_tilde': nn.Parameter(torch.ones(self.n_heads) * inv_softplus(target_ss_process_var)),
#             'eta': nn.Parameter(torch.ones(self.n_heads) * inv_softplus(target_measurement_var)),
#             'gamma': nn.Parameter(torch.ones(self.n_heads) * inv_softplus(target_measurement_var))
# })
        ##########################
        if self.args.use_colored_prior:
            # Initialize spectral coefficients
            self.c_exp_v = nn.Parameter(torch.zeros(self.n_heads) - 4.0) # Start with low colored noise
            beta_exp_inverse_softplus_v  = inv_softplus(total_mu_v.detach().clone())
            self.beta_exp_v = nn.Parameter(beta_exp_inverse_softplus_v) # Initialize beta_exp to match the initial mu
            # Since beta_exp_v < 2 mu, the model starts in the Slow Noise Regime.
            self.a_per_v = nn.Parameter(torch.zeros(self.n_heads)) # Start with no periodic terms
            self.b_per_v = nn.Parameter(torch.zeros(self.n_heads)) # Start with no periodic terms
            periods = torch.exp(torch.linspace(np.log(4.0), np.log(self.args.seq_len), self.n_heads))
            self.omega_per_v = nn.Parameter((2 * np.pi) / periods)
        ################################################
        # Optionally, define separate dynamic query/key parameters
        if self.args.sep_params == True:
            ##########################
            # Define LTI params
            ##########################
            # Initialize omega
            if self.args.use_complex_conj_constraint == True:
                dim_target_k = int(self.d_k_head/2)
            else:
                dim_target_k = self.d_k_head

            if self.args.use_SC_RoPE == True:
                omega_init_stacked_k, total_mu_k = init_spectrally_coupled_rope(self.n_heads, dim_target_k, b=self.args.damping)
            else:
                _, total_mu_k = init_spectrally_coupled_rope(self.n_heads, dim_target_k, b=self.args.damping)
                omega_init_stacked_k = init_rope(self.n_heads, dim_target_k)
                
            if self.args.zero_rotations == True:
                omega_init_stacked_k = torch.zeros_like(omega_init_stacked_k)
            if self.args.learn_rotations == True:
                self.omega_k = nn.Parameter(omega_init_stacked_k)
            else:
                self.register_buffer("omega_k", omega_init_stacked_k)
                
#             ##########################
            # Initialize decay
            if self.args.use_log_linear_decay == True and self.args.use_SC_RoPE == False:
                total_mu_k = init_decay_per_head(self.n_heads, min_decay=0.01, max_decay=0.95, zero_frac=0.25)
                    
            if self.args.learn_decay == True:
                self.mu_k = nn.Parameter(inv_sigmoid(total_mu_k))
            else:
                self.register_buffer("mu_k", total_mu_k)
#             ##########################
            # Initialize noise params
            initial_sigma_sq_k = 2 * total_mu_k.detach().clone() * target_ss_process_var
            initial_sigma_sq_inverse_softplus_k = inv_softplus(initial_sigma_sq_k)
            self.sigma_k = nn.Parameter(initial_sigma_sq_inverse_softplus_k)
            self.sigma_tilde_k = nn.Parameter(torch.ones(self.n_heads) * target_ss_process_var)
            self.eta_k = nn.Parameter(torch.ones(self.n_heads) * target_measurement_var)
            self.gamma_k = nn.Parameter(torch.ones(self.n_heads) * target_measurement_var)

#             self.noise_params_k = nn.ParameterDict({
#                 'sigma': nn.Parameter(initial_sigma_sq_inverse_softplus_k),
#                 'sigma_tilde': nn.Parameter(torch.ones(self.n_heads) * inv_softplus(target_ss_process_var)),
#                 'eta': nn.Parameter(torch.ones(self.n_heads) * inv_softplus(target_measurement_var)),
#                 'gamma': nn.Parameter(torch.ones(self.n_heads) * inv_softplus(target_measurement_var))
#     })
            ##########################
            if self.args.use_colored_prior == True:
                # Initialize spectral coefficients for the D_k suite
                self.c_exp_k = nn.Parameter(torch.zeros(self.n_heads) - 4.0) # Start with low colored noise
#                 beta_exp_inverse_softplus_k = torch.log(torch.exp(total_mu_k.detach().clone()) - 1.0 + 1e-6)
                beta_exp_inverse_softplus_k = inv_softplus(total_mu_k.detach().clone())
                self.beta_exp_k = nn.Parameter(beta_exp_inverse_softplus_k) # Initialize beta_exp to match the initial mu
                self.a_per_k = nn.Parameter(torch.zeros(self.n_heads))
                self.b_per_k = nn.Parameter(torch.zeros(self.n_heads))
                periods = torch.exp(torch.linspace(np.log(4.0), np.log(self.args.seq_len), self.n_heads))
                self.omega_per_k = nn.Parameter((2 * np.pi) / periods)           
        ################################################
        # Initialize robustness parameter and softmax inverse temperature
        self.tau_param = nn.Parameter(torch.ones(self.n_heads)) # Softmax inverse temperature
        nu_over_d_target = 4.0 # Lower numbers make the starting "Student's t" distribution have much heavier tails
        self.nu_param = nn.Parameter(torch.ones(self.n_heads) * nu_over_d_target) 
#         self.nu_param = nn.Parameter(torch.ones(self.n_heads) * inv_softplus(nu_over_d_target - 2.0/self.d_v_total))
        # This initialization for nu ensures that the model starts "Gaussian-like" rather than aggresively downweighting outliers.
        
        if self.args.learn_t_shift == True:
            self.t_shift_param = nn.Parameter(torch.zeros(self.n_heads).unsqueeze(0))

        ################################################

        # Initialize linear layers using complex initialization
        W_q_init = init_complex_matrix(self.d_e, self.d_k_total)
        W_k_init = init_complex_matrix(self.d_e, self.d_k_total)
        W_v_init = init_complex_matrix(self.d_e, self.d_v_total)
        W_o_init = init_complex_matrix(self.d_v_total, self.d_e)

        init_complexlinear(self.W_q, W_q_init, layer_type='in')
        init_complexlinear(self.W_k, W_k_init, layer_type='in')
        init_complexlinear(self.W_v, W_v_init, layer_type='in')
        init_complexlinear(self.W_o, W_o_init, layer_type='out')

        ################################################
        # Initialize complex-valued normalization layers
        if self.args.use_complex_input_norm == 0:
            pass
        else:
            self.cn_q = ComplexRMSNorm(self.d_k_total, self.n_heads)
            self.cn_k = ComplexRMSNorm(self.d_k_total, self.n_heads)
            self.cn_v = ComplexRMSNorm(self.d_v_total, self.n_heads)
        
        if self.args.use_complex_output_norm == True:
            self.cn_o = ComplexRMSNorm(self.d_v_total, self.n_heads)
        else:
            pass
        ################################################
        
        # Causal mask:
        causal_mask = torch.tril(torch.ones(max_len, max_len))
        self.register_buffer("causal_mask", causal_mask.view(1, max_len, max_len, 1))

        # Relative weighting of estimate and prediction in output (IRLS step size)
#         self.P_prior_param_log = nn.Parameter(torch.rand(1, 1, self.n_heads, self.d_v_head, 1)) # Anisotropic
#         self.P_scale_log = nn.Parameter(torch.rand(1, 1, self.n_heads, self.d_v_head, 1)) # Anisotropic
        self.P_prior_log = nn.Parameter(torch.zeros(1, 1, n_heads, 1, 1))
        self.P_scale_log = nn.Parameter(torch.zeros(1, 1, n_heads, 1, 1) - 2.0) # Initialize to not trust P_tot
    ############################################
        
    def _get_sde_kernels(self, mu_raw, omega_raw, noise_params_raw, t_measure, delta_t):
        """
        Transforms learned SDE parameters into the physical kernels required for RFA.
        Processes eigenvalues, noise variances, state transitions, and precision.
        """
        
        if self.args.scale_decay_by_time_interval == True:
            # Note: If, at test time, we use a longer sequence length, we must still use the value used during training        
            total_time = self.args.seq_len - 1 # (Assumes equal time intervals)
            scale_c = self.args.max_learned_decay if self.args.learn_decay else self.args.max_fixed_decay
            scale_r = scale_c / total_time
        else:
            scale_r = 1.0
           
        # Compute stable eigenvalues (Hurwitz constraint)
        if self.args.learn_decay == True:
            mu = scale_r * torch.sigmoid(mu_raw) + self.args.epsilon # Use sigmoid to maintain positivity and bound scale
        else:
            mu = scale_r * mu_raw

        # Compute Imaginary Part (+/- omega): Creates conjugate pairs.
        if self.args.use_complex_conj_constraint == True:
            n_heads, d_half = omega_raw.shape
            omega = torch.stack([omega_raw, -omega_raw], dim=-1)
            omega = omega.reshape(n_heads, 2 * d_half)
        else:
            omega = omega_raw

        # Compute clamped exponents for numerical stability in e^(mu * delta_t)
        mu_h = mu.view(1, 1, -1)
        
        exp_rel_safe = torch.clamp(-mu_h * delta_t, min=self.args.min_exponent, max=self.args.max_exponent)
        
        # Generate state transition kernels (Rotation Phi and Decay E)
        Phi_tilde_plus, E_rel = compute_exp_kernel_isotropic(omega, t_measure, exp_rel_safe)

        # Extract 'D_k' (Diffusion-Kernel) components and constrain to be non-negative
        sigma_sq = F.softplus(noise_params_raw['sigma']) # Process noise
        # Steady state effective process noise (for learnable decay case)
        sigma_tilde_sq = F.softplus(noise_params_raw.get('sigma_tilde', torch.zeros_like(sigma_sq)))
        eta_sq = F.softplus(noise_params_raw['eta']) + self.args.epsilon # Key-side measurement noise
        gamma_sq = F.softplus(noise_params_raw['gamma']) + self.args.epsilon # Query-side measurement noise

        # Zeroing logic for specific ablations
        if self.args.lambda_real_zero == True:
            mu = mu*0
        if self.args.zero_process_noise:
            sigma_sq = sigma_sq * 0
            sigma_tilde_sq = sigma_tilde_sq * 0
        if self.args.zero_key_measurement_noise:
            eta_sq = eta_sq * 0
        
        # Compute analytic DLE solution for the isotropic covariance kernel V_ij
        if self.args.learn_decay == True:
            V_ij = compute_covariance_matrix_residual_diffusion(mu_h, delta_t, exp_rel_safe, E_rel, sigma_sq, sigma_tilde_sq, eta_sq, gamma_sq, self.args)
        else:
            if self.args.use_colored_prior == True:
                V_ij = compute_covariance_matrix_spectral_full(
                    mu_h, delta_t, exp_rel_safe, E_rel, 
                    sigma_sq, eta_sq, gamma_sq, noise_params_raw,
                    self.args, epsilon=1e-5, cov_epsilon=1e-2)
            else:
                if self.args.use_ss_process_noise == True:
                    V_ij = compute_covariance_hybrid_regime(mu_h, delta_t, E_rel, exp_rel_safe, sigma_tilde_sq, eta_sq, gamma_sq, self.args)
                else:
                    V_ij = compute_covariance_matrix_LHopital(mu_h, delta_t, E_rel, exp_rel_safe, sigma_sq, eta_sq, gamma_sq, self.args)

        return mu, omega, Phi_tilde_plus, E_rel, V_ij, sigma_sq, sigma_tilde_sq, eta_sq, gamma_sq
         
    ############################################
        
    def _compute_attention_matrix(self, R_qk_abs_sq, V_ij_k, V_ij_v, delta_t, abs_idx, sigma_sq_k, eta_sq_k, gamma_sq_k, seq_len):
        """
        Computes the attention matrix A by combining the dynamical prior (Bias) 
        and the M-estimator (Mahalanobis distance).
        
        We derive the attention energy from the negative log-likelihood (NLL) 
        of a d-dimensional isotropic Multivariate Student's t-distribution:
        Energy = - (1/2) * [ log|Σ| + (ν+d) * log(1 + (rᵀΣ⁻¹r)/ν) ]
        
        For isotropic Σ = V*I, this simplifies to:
        Energy = - (d/2) log(V) - ((ν+d)/2) * log(1 + R² / (νV))
        
        Dividing through by d/2:
        Energy (per dimension) ~ - log(V) - ((ν+d)/d) * log(1 + R² / (νV))
        
        We map this to the following code variables:
        B (DLE Bias)        = -log(V)         -> Log-prior of the precision
        P (Prior Precision) = 1 / V     -> Inverse scale for the Mahalanobis distance
        DoF_rescale         = (ν + d) / d     -> Normalized DOF adjustment (factor of 0.5 absorbed)
        """
        
        d = self.d_k_head # Embed dim
        
        # Student's t degrees of freedom nu (constrain to be at least 2, so variance exists)
        # We parameterize as a multiple of d: nu = d softplus(theta) + 2, so robustness threshold scales
        # with the dimensionality of the latent space, preventing the 'curse of dimensionality' from making
        # the M-estimator excessively sparse in high-dimensional heads.
        nu = d * F.softplus(self.nu_param) + 2.0 + self.args.epsilon
        
#         tau = 1.0
        tau = F.softplus(self.tau_param) + self.args.epsilon # Softmax inverse temperature
#         DoF_rescale = 1.0
        DoF_rescale = (nu + d) / d

        # --- Additive Bias (B): Log-Prior of the Precision ---
        if self.args.additive_bias_type == 0:
            B = 0 # Zero bias
        elif self.args.additive_bias_type == 1:
            B = - torch.log(V_ij_v) # DLE Bias
        elif self.args.additive_bias_type == 2:
            m = sigma_sq_k / (eta_sq_k + gamma_sq_k)
            B = - m * delta_t # Linear bias
        else:
            pass
        
        # --- Multiplicative Scaling (P): Prior Precision ---
        if self.args.multiplicative_bias_type == 0:
            P = 1/(d**0.5) # Constant
        elif self.args.multiplicative_bias_type == 1:
            P = 1/(nu * V_ij_k) # DLE Bias (a sigmoid)
        elif self.args.multiplicative_bias_type == 2:
            P = 1/(sigma_sq_k * delta_t + eta_sq_k) # Linear bias
        else:
            pass
        
        if self.args.t_equal == True:
            if self.args.multiplicative_bias_type == 1:
                P = P[:, abs_idx, :]
            if self.args.multiplicative_bias_type == 2:
                P = P[abs_idx, :]
            if self.args.additive_bias_type == 1:
                B = B[:, abs_idx, :]
            if self.args.additive_bias_type == 2:
                B = B[abs_idx, :]
        
        # --- Robust Scoring (M-Estimator) ---
        if self.args.use_robust_weight == True:
            # Student's t-style robust weight: log(P) - log(1 + d^2)
            #   P_ij_v/(1 + R_qk_abs_sq * P_ij_k)
            # = exp[ log(P_ij_v) - log(1 + R_qk_abs_sq * P_ij_k)]
            # = exp[-log(V_ij_v) - log(1 + R_qk_abs_sq / V_ij_k)]
            log_denominator = torch.log(1 + P * R_qk_abs_sq)
            base_attn_scores = B - DoF_rescale * log_denominator
            energy_linear = B - log_denominator # Only needed for total_precision_gate or add Gaussian noise
        else:
            # Gaussian weight: log(P) - d^2 (Standard Softmax Attention)
            #   P_ij_v * exp(-R_qk_abs_sq * P_ij_k)
            # = exp[ log(P_ij_v) - R_qk_abs_sq * P_ij_k]
            # = exp[-log(V_ij_v) - R_qk_abs_sq / V_ij_k]
#             base_attn_scores = B - DoF_rescale * P * R_qk_abs_sq
            base_attn_scores = B - P * R_qk_abs_sq
            energy_linear = B - P * R_qk_abs_sq # Only needed for total_precision_gate or add Gaussian noise

        # Compute this ephemeral O(N^2) tensor and reduce it immediately
        if self.causal == True:
            current_mask = self.causal_mask[:, :seq_len, :seq_len, :]
            precision_marginal_log = torch.logsumexp(energy_linear.masked_fill(current_mask == 0, float('-inf')), dim=-2)
        else:
            precision_marginal_log = torch.logsumexp(energy_linear, dim=-2)
            
        # --- Normalization and Masking ---    
        attention_scores = tau * base_attn_scores # Apply softmax inverse temperature

        if self.causal == True:
            # Causal masking before exponentiation for numerical stability
            current_mask = self.causal_mask[:, :seq_len, :seq_len, :]
            attention_scores = attention_scores.masked_fill(current_mask == 0, float('-inf'))
                    
#         # Measure of total accumulated precision for each target token i
#         unnormalized_attention = torch.exp(base_attn_scores_linear_masked) # Only needed if using total_precision_gate or add_Gaussian_noise

        # Final normalized attention matrix A[i, j]
        A_hat = torch.softmax(attention_scores, dim=2) # Apply softmax

        return A_hat, precision_marginal_log

    ############################################
    
    def _compute_estimate(self, A_hat, E_rel_v, V_tilde, Phi_tilde_plus_v):
        """
        Aggregates rotated values and transforms them back to the original frame.

        1. Weighted sum: Sums source features scaled by attention scores (einsum).
        2. Counter-rotation: Restores global frame for the final estimate.
        """
        
#         A = A _hat * E_rel_v # Scale attention weights by the relative decay
#
#         # Move head dimension back
#         A_permute = A.permute(0,3,1,2).contiguous()
#         V_tilde_permute = V_tilde.permute(0,2,1,3,4).contiguous()
#         # Stack real/imag components into one dimension
#         V_tilde_stack = V_tilde_permute.reshape(*V_tilde_permute.shape[:-2], V_tilde_permute.size()[-2]*2)
#         # Multiply rotated values by attention matrix
#         est_rotated_stack = torch.matmul(A_permute, V_tilde_stack)
#         # Unstack real/imaginary components and move head dimension forward
#         est_rotated = est_rotated_stack.reshape(V_tilde_permute.size()).permute(0,2,1,3,4).contiguous()
#        
#         # Compute weighted sum across the source sequence (j).
#         # Equivalent to: A.permute(...) @ V_tilde.reshape(...) then un-permuting.
#         # b=batch, i=target_seq, j=source_seq, h=heads, d=d_head, c=complex(2)
#         est_rotated = torch.einsum('bijh,bjhdc->bihdc', A, V_tilde)
        
#         if self.args.rotate_values == True:
#             est_rotated = torch.einsum('bijh,bijh,bjhdc->bihdc', A_hat, E_rel_v, V_tilde)
#         else:
#             A = A_hat * E_rel_v 
#             est_rotated = torch.einsum('bijh,bjhdc->bihdc', A, V_tilde)
        
        est_rotated = torch.einsum('bijh,bijh,bjhdc->bihdc', A_hat, E_rel_v, V_tilde)
    
        if self.args.rotate_values == True:
            # Rotate back to original frame
            cos_v = Phi_tilde_plus_v[...,0]
            sin_v = Phi_tilde_plus_v[...,1]
            est_counter_rotated = apply_interleaved_rope(est_rotated, cos_v, sin_v)
        else:
            est_counter_rotated = est_rotated

        return est_counter_rotated
    
    def _add_Gaussian_noise(self, precision_marginal_log, gamma_sq_v, est_v):
        """
        Adds SDE-derived stochasticity for probabilistic generative sampling.

        Maps total precision (P_tot) to complex variance and samples 
        independent isotropic noise for the final timestep.
        """
        
        # Retrieve the scaling factors used in the forward pass
        d = self.d_k_head
        nu = d * (F.softplus(self.nu_param) + 2.0 + self.args.epsilon)
        
#         # Information Fusion: Gating SDE estimate vs. Raw observation
#         P_tot = torch.sum(torch.exp(energy_linear),-2) # Total precision

#         p_phys = P_tot[:, -1]

#         # Calculate Variance (divide by d to to get per-dimension variance)
#         v_phys = 1.0 / (p_phys + self.args.epsilon)
        
#         P_tot_log = torch.logsumexp(energy_linear[:, -1, :, :], dim=-2) # [B, H]
        P_tot_log = precision_marginal_log[:, -1, :].detach()

        v_phys = torch.exp(-P_tot_log)
        
        # Total variance including measurement noise (gamma)
        # nu_factor is the Student's t variance correction (nu / nu-2)
        nu_factor = torch.clamp(nu / (nu - 2.0 + self.args.epsilon), max=5.0)

        # Total physical variance
#         v_final = (v_phys * nu_factor) + gamma_sq_v
        v_final = (v_phys * nu_factor.view(1, -1)) + gamma_sq_v.view(1, -1)
        
        # Isotropic complex noise
        # Divide by 2.0 because we split variance between real and imag
        std_dev = torch.sqrt(v_final / 2.0)

        # Reshape for [B, 1, H, 1, 1] to match est_v [B, L, H, D, 2]
        std_dev_reshape = std_dev.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        
        # Sample and add noise only to the final generated step
        noise = torch.randn_like(est_v[:, -1:]) * std_dev_reshape
        est_v[:, -1:] = est_v[:, -1:] + noise

        return est_v
    
    def _apply_total_precision_gate(self, precision_marginal_log, est_v, V):
        """
        Convex combination of the SDE estimate and raw observation.

        Acts as a 'Confidence Gate': As accumulated precision (P_tot) increases, 
        the model trusts its filtered state over the noisy raw input (V).
        
        The gate is defined as:
        P_tot/ (P_tot + P_prior) = 1 / (1 + P_prior/P_tot)
        = 1/(1 + e^{-(ln(P_tot) - ln(P_prior)}) = sigmoid[ln(P_tot) - ln(P_prior)]
        """
        
#         # Information Fusion: Gating SDE estimate vs. Raw observation
#         P_tot = torch.sum(unnormalized_attention,-2) # Total precision

        # Map precision ratio to a [0, 1] gate using sigmoid in log-space
#         P_tot_detach = P_tot.detach() # Detach P_tot to decouple confidence calibration from feature learning
#         P_tot_log = torch.log(P_tot_detach + self.args.epsilon)
#         P_tot_log = torch.logsumexp(energy_linear.detach(), dim=-2) # Stabler implementation
        P_tot_log = precision_marginal_log.detach()

        # P_scale determines how "sensitive" the gate is to precision changes
        P_scale = torch.exp(self.P_scale_log)
        
#         # Gate g -> 1 means 'Trust SDE', g -> 0 means 'Trust Input'
# #         g = torch.sigmoid(P_scale * P_tot_log - self.P_prior_log)
#         g = torch.sigmoid(P_scale * P_tot_log - self.P_prior_log)

        # Extract nu for scaling
        nu = self.d_k_head * (F.softplus(self.nu_param) + 2.0 + self.args.epsilon)

        # Apply the nu_factor CORRECTION in log-space
        # In noise function, you multiply variance by (nu / nu-2)
        # Here, we multiply precision by (nu-2 / nu)
        # In log-space, this is + log(nu-2) - log(nu)
        nu_correction_log = torch.log(nu - 2.0 + self.args.epsilon) - torch.log(nu + self.args.epsilon)

        # Calibrated Log-Precision
        # We do NOT add log(tau). We ONLY add the nu correction.
        calibrated_P_log = (P_tot_log + nu_correction_log.view(1, 1, -1)).unsqueeze(-1).unsqueeze(-1)

        # The Sigmoid Gate
        # P_scale handles the learnable sensitivity
        g = torch.sigmoid(P_scale * calibrated_P_log - self.P_prior_log)

        # Final latent state estimate
        # This keeps the gradient flow through est_v and V intact, 
        # while P_tot acts only as a weight.
        est_latent = g * est_v + (1.0 - g) * V # Update using convex combination

        return est_latent, g
    
    def _project_forward(self, est, dt_pred, omega_v, mu_v):
        """
        Projects the current state estimate dt_pred steps into the future via the LTI transition matrix.
        """
        
        # Scale rotation (frequencies) and decay (rates) by time delta
        omega_v_dt = omega_v[None, None, :, :] * dt_pred # [1, 1, H, D] * [B, 1, 1, 1]
        mu_v_dt    = mu_v[None, None, :, None] * dt_pred # [1, 1, H, 1] * [B, 1, 1, 1]

        # Compute rotation components (cos, sin)
        cos_v_dt = torch.cos(omega_v_dt) # [B, 1, H, D]
        sin_v_dt = torch.sin(omega_v_dt) # [B, 1, H, D]
        
        # Compute exponential decay factor (magnitude)
        decay = torch.exp(-mu_v_dt)[..., None] # Adds the complex-dim [B, 1, H, 1, 1]
        
        # Apply RoPE-style phase shift to rotate state forward
        rot = apply_interleaved_rope(est, cos_v_dt, sin_v_dt) # [B, L, H, D, 2]
        
        # Scale rotated state by decay to get future filtered prediction
        pred_p = decay * rot # [B, L, H, D, 2]
        
        return pred_p
    
    def _compute_pulled_forward_estimates(self, V_tilde, cos_v, sin_v, E_rel_v, current_mask):
        """
        Only for visualization. (This is expensive so only run occasionally.)
        
        Projects historical latent states into all future coordinate frames.

        Transforms past states via SDE transition kernels (Phi, E_rel) to visualize 
        belief trajectories—showing how the representation of token j evolves as 
        it is 'pulled forward' to time i.
        """
        
        # Apply the rotation to V_tilde
        cos_v_all = cos_v.unsqueeze(2)
        sin_v_all = sin_v.unsqueeze(2)
        rotated_V = apply_interleaved_rope(V_tilde.unsqueeze(1), cos_v_all, sin_v_all)

        # Apply the relative scaling (E_rel_v)
        Z_ij_hat_all_multihead = E_rel_v.unsqueeze(-1).unsqueeze(-1) * rotated_V

        # Reshape and concatenate multi-head/complex components
        # Prepares the tensor for projection back to the original input space.
        Z = Z_ij_hat_all_multihead.permute(0,1,2,5,3,4).contiguous() # Move complex dim
        Z_ij_hat_all = Z.view(*Z.size()[:-2], -1) # Stack heads

        # Mask future timesteps and flatten for the output projection
        if self.args.causal == True:
            Z_ij_hat_all = Z_ij_hat_all * current_mask.unsqueeze(-1) # Apply causal mask
        Z_ij_hat_all = Z_ij_hat_all.view(*Z_ij_hat_all.size()[:-2], -1) # Stack real/imaginary partys

        # Project to original basis via W_o
        x_hat = self.W_o(Z_ij_hat_all).unsqueeze(-1)

        return x_hat
    
    def _get_output_metadata(self, V_tilde, cos_v, sin_v, E_rel_v, est_latent, mu_v, omega_v, A_hat, V_ij_v, precision_marginal_log, sigma_sq_v, sigma_tilde_sq_v, eta_sq_v, gamma_sq_v, g, abs_idx, seq_len):
        """Handles logging and visualization data."""
        
        tau = F.softplus(self.tau_param) + self.args.epsilon # Softmax inverse temperature
        d = self.d_k_head
        nu_over_d = (d * F.softplus(self.nu_param) + 2.0 + self.args.epsilon)/d
        
#         mu_v = mu_v / self.args.delta_t # Have to divide by dt since we absorbed dt into mu

        if not self.training and self.args.compute_metadata==True:
        
            A = A_hat * E_rel_v # Decay attention matrix
        
            if self.args.causal == True:
                current_mask = self.causal_mask[:, :seq_len, :seq_len, :]
            else:
                current_mask = None

            # Get pulled-forward estimates (for testing/visualization)
            # This creates a massive O(N^2 d) tensor to "project" every past state into every future frame.
            if self.args.compute_pulled_forward_estimates == True:
                x_hat = self._compute_pulled_forward_estimates(V_tilde, cos_v, sin_v, E_rel_v, current_mask)
            else:
                x_hat = None
            
            # Collect the eigenvalues for plotting
            mu_v_expanded = mu_v.unsqueeze(-1).expand(-1, int(omega_v.size()[-1]/2))  # Expand real to match imag
            mu_v_vec = mu_v_expanded.repeat_interleave(2, dim=1) # Create interleaved complex conj pairs
            lambda_v = torch.stack([-mu_v_vec.T, omega_v.T], dim=0).unsqueeze(1) # Stack real/imag
            L = lambda_v.transpose(-2, -1).contiguous() # Flip dimensions to stack heads
            eigenvals = L.view(*L.size()[:-2], -1).unsqueeze(-1) # Merge heads

            # Prior for visualization (1/V represents the model's prior confidence over time)
            if self.args.t_equal == True:
                V_ij_v = V_ij_v[:, abs_idx, :]
            
            A_prior = 1/V_ij_v
            if self.args.causal == True:
                A_prior = A_prior.masked_fill(current_mask == 0, float('0')).squeeze(0)
            
#             unnormalized_attention = torch.exp(energy_linear)
#             P_tot = torch.sum(unnormalized_attention,-2) # Total precision
#             P_tot = torch.exp(torch.logsumexp(energy_linear, dim=-2))
            P_tot = precision_marginal_log
    
        else:
            x_hat = None
            eigenvals = None
            A_hat = None
            A = None
            A_prior = None
            unnormalized_attention = None
            P_tot = None
            est_latent = None
        
        # # Map latent estimate back to original basis
        # est_latent_permute = est_latent.permute(0,1,4,2,3).contiguous() # Move real/imag dimension back
        # est_latent_reshape = est_latent_permute.view(batch_size, seq_len, 2, self.d_v_total) # Merge heads
        # est_latent_stack = est_latent_reshape.view(batch_size, 1, seq_len, self.d_v_total*2) # Stack complex numbers into last dimension
        # est_out = self.W_o(est_latent_stack) # Map back to original basis

        # Return dictionary of metadata
        output_dict = {
            'est_latent': est_latent,
            'attn_mat': A_hat,
            'decayed_attn_mat': A,
            'A_prior': A_prior,
            'x_hat': x_hat, 
            'mu_v': mu_v,
            'sigma_sq_v': sigma_sq_v,
            'sigma_tilde_sq_v': sigma_tilde_sq_v,
            'eta_sq_v': eta_sq_v,
            'gamma_sq_v': gamma_sq_v,
            'eigenvals': eigenvals,
            'P_tot': P_tot,
            'tau': tau,
            'nu_over_d': nu_over_d,
            'gate': g
        }

        return output_dict
    
    ############################################

    def forward(self, Z_q, Z_k, Z_v, t_measure=None, t_shift=None, causal=True):
        """
        Executes the Robust Filter Attention (RFA) process to generate the filtered state estimate.

        The function performs:
        1. Projection: Maps real features (Z_q, Z_k, Z_v) to the complex latent eigenbasis.
        2. Kernel Computation: Calculates the time-dependent state transition kernels (Phi_tilde)
           and the covariance kernels (K_cov) using the learned dynamic parameters.
        3. Weight Calculation: Determines the robust attention matrix (A_hat) by scaling
           the calculated precision prior (1/K_cov) based on the Mahalanobis residual distance (R_qk_abs_sq).
        4. Aggregation: Computes the complex, precision-weighted sum of the rotated values (V_tilde)
           and rotates the result back to the forward domain to produce the final filtered estimate (out).

        Args:
            Z_q (Tensor): Input features for Queries (shape B x L x d_e).
            Z_k (Tensor): Input features for Keys.
            Z_v (Tensor): Input features for Values.
            t_measure (Tensor, optional): Timestamps for observations. Defaults to regular sampling if None.
            t_shift: Predict t_shift time into the future

        Returns:
            out (Tensor): Final output, projected back to the real domain.
            output_dict (dict): Intermediate tensors including latent estimates and attention matrix.
        """
        
#         #######################
#         ## For testing in 2D ##
#         if self.args.weight_mask == True:
#             apply_projection_mask(self)
#         #######################
        
        batch_size, seq_len, _ = Z_q.shape
        device = Z_q.device
        self.causal = causal
        
        #######################################
        
        if self.args.t_equal == True:
            t_measure = self.t_measure[0:seq_len]
#             t_measure = torch.arange(seq_len, device=device)
            delta_t = t_measure.float().unsqueeze(-1) # [N, 1]
#             delta_t = t_measure.float().view(1, -1, 1, 1)
#             abs_idx = torch.abs(t_measure.view(seq_len, 1) - t_measure.view(1, seq_len))
            abs_idx = self.abs_idx[0:seq_len, 0:seq_len]
            t_measure = t_measure.unsqueeze(0) # [1, N]      
        else:
            # Full 2D Track (Batched / Irregular Time) ---
            # t_measure is [B, N]. Compute delta_T: [B, N, N, 1]
            delta_t = torch.abs(t_measure.unsqueeze(-1) - t_measure.unsqueeze(-2)).unsqueeze(-1)
            abs_idx = None
    
        # ------
        # If t_shift is provided, we perform 'Predictive Filtering'.
        # We shift the kernel's horizon so the attention weights (A) and 
        # uncertainty (V_ij) are computed for a future time point (t_i + t_shift).
        if self.args.learn_t_shift == True:
            t_shift = F.softplus(self.t_shift_param)
            delta_t = delta_t + t_shift # Necessary for correct computation of V_ij & E_rel
            t_shift_v = t_shift.unsqueeze(0).unsqueeze(-1)
        else:
            if t_shift == None or t_shift == 0:
                pass
            else:
                t_shift = torch.as_tensor(t_shift, device=device).view(-1, 1) # [B, 1]
                delta_t = delta_t + t_shift # Necessary for correct computation of V_ij & E_rel
                t_shift_v = t_shift
                # In order to avoid double counting E_rel in the pred step, only rotate forward, no decay
        
        #######################################

        # Linear projections; split into heads & real/imaginary parts
        Q = self.W_q(Z_q).view(batch_size, seq_len, 2, self.n_heads, self.d_k_head)
        K = self.W_k(Z_k).view(batch_size, seq_len, 2, self.n_heads, self.d_k_head)
        V = self.W_v(Z_v).view(batch_size, seq_len, 2, self.n_heads, self.d_v_head)

        # Move real/imaginary index to the end
        Q = Q.permute(0,1,3,4,2).contiguous()
        K = K.permute(0,1,3,4,2).contiguous()
        V = V.permute(0,1,3,4,2).contiguous()    

        #######################################

        # Optionally, apply complex-valued normalization on inputs (per head)
        if self.args.use_complex_input_norm == 1: # Normalize query, key, and value
            Q_norm = self.cn_q(Q)
            K_norm = self.cn_k(K)
            V_norm = self.cn_v(V)
        elif self.args.use_complex_input_norm == 0: # No normalization
            Q_norm = Q
            K_norm = K
            V_norm = V
        elif self.args.use_complex_input_norm == 2: # Normalize only query and key
            Q_norm = self.cn_q(Q)
            K_norm = self.cn_k(K)
        else:
            print('Eror: args.use_complex_input_norm must be 0, 1, or 2.')
            
        #######################################
        
        noise_params_v = {
        'sigma': self.sigma_v,
        'eta': self.eta_v,
        'gamma': self.gamma_v,
        'sigma_tilde': self.sigma_tilde_v
        }
        if self.args.use_colored_prior:
            noise_params_v.update({
                'c_exp': self.c_exp_v,
                'beta_exp': self.beta_exp_v,
                'a_per': self.a_per_v,
                'b_per': self.b_per_v,
                'omega_per': self.omega_per_v
            })

        # Dynamic Kernels: Compute transition (Phi, E) and precision (V_ij)
        mu_v, omega_v, Phi_tilde_plus_v, E_rel_v, V_ij_v, sigma_sq_v, sigma_tilde_sq_v, eta_sq_v, gamma_sq_v = self._get_sde_kernels(self.mu_v, self.omega_v, noise_params_v, t_measure, delta_t)

        # Handle Parameter Separation (Optional separate K/Q dynamics)
        if self.args.sep_params == True:
            noise_params_k = {
            'sigma': self.sigma_k,
            'eta': self.eta_k,
            'gamma': self.gamma_k,
            'sigma_tilde': self.sigma_tilde_k
            }
            if self.args.use_colored_prior:
                noise_params_k.update({
                    'c_exp': self.c_exp_k,
                    'beta_exp': self.beta_exp_k,
                    'a_per': self.a_per_k,
                    'b_per': self.b_per_k,
                    'omega_per': self.omega_per_k
                })
            mu_k, omega_k, Phi_tilde_plus_k, E_rel_k, V_ij_k, sigma_sq_k, sigma_tilde_sq_k, eta_sq_k, gamma_sq_k = self._get_sde_kernels(self.mu_k, self.omega_k, noise_params_k, t_measure, delta_t)
        else:
            V_ij_k = V_ij_v
            Phi_tilde_plus_k = Phi_tilde_plus_v
            E_rel_k = E_rel_v
            sigma_sq_k = sigma_sq_v
            eta_sq_k = eta_sq_v
            gamma_sq_k = gamma_sq_v
        
        # Rotational Alignment: Apply RoPE-style phase shifts (Phi)
        cos_k = Phi_tilde_plus_k[...,0]
        sin_k = Phi_tilde_plus_k[...,1]
        cos_v = Phi_tilde_plus_v[...,0]
        sin_v = Phi_tilde_plus_v[...,1]
        
        Q_tilde = apply_interleaved_rope(Q_norm, cos_k, -sin_k)
        K_tilde = apply_interleaved_rope(K_norm, cos_k, -sin_k)
        V_tilde = apply_interleaved_rope(V_norm, cos_v, -sin_v)
        
        if self.args.t_equal == True:
            E_rel_k = E_rel_k[:, abs_idx, :]
            E_rel_v = E_rel_v[:, abs_idx, :]
        
        # Attention Scoring: Mahalanobis distance + Precision Bias
        R_qk_abs_sq = compute_residual_norm_isotropic(Q_tilde, K_tilde, E_rel_k, self.args)
        
        A_hat, precision_marginal_log = self._compute_attention_matrix(R_qk_abs_sq, V_ij_k, V_ij_v, delta_t, abs_idx, sigma_sq_k, eta_sq_k, gamma_sq_k, seq_len)
        
        # Filtered Estimation: Weighted sum of rotated history
        est_v = self._compute_estimate(A_hat, E_rel_v, V_tilde, Phi_tilde_plus_v)

        #############################################
        
        if self.args.add_gaussian_noise == True:
            # Add Guassian noise to last step (for test-time sampling)
            est_v = self._add_Gaussian_noise(precision_marginal_log, gamma_sq_v, est_v)
        else:
            pass

        # Add residual connection
        if self.args.use_inner_residual == 1:
            if self.args.use_total_precision_gate == 0:
                est_latent = est_v # No residual connection
                g = None
            elif self.args.use_total_precision_gate == 1: # Precision gate
                est_latent, g = self._apply_total_precision_gate(precision_marginal_log, est_v, V_norm)
            elif self.args.use_total_precision_gate == 2: # Learned gate
                g = torch.sigmoid(self.P_scale)
                est_latent = g * est_v + (1-g) * V_norm
#                 est_latent = g * est_v + V # No gate on residual connection
            else:
                print('Error: args.use_total_precision_gate must be 0, 1, or 2.')
        else:
            est_latent = est_v # No residual
            g = None
        
        # Optionally, use complex normalization on outputs
        if self.args.use_complex_output_norm == True:
            est_norm = self.cn_o(est_latent)
        else:
            est_norm = est_latent
            
        # If t_shift was applied to kernels, decay is already in est_norm, so we only need to rotate forward.
        if t_shift == None:
            pred_p = est_norm
        else:
            # Reuse _project_forward but nullify the decay part
            pred_p = self._project_forward(est_norm, t_shift_v, omega_v, mu_v * 0)

        # -------------------
        # Output projections
        
        # Move real/imag dimension back
        pred_p_permute = pred_p.permute(0,1,4,2,3).contiguous()
        # Merge heads
        pred_p_reshape = pred_p_permute.view(batch_size, seq_len, 2, self.d_v_total)     
        # Stack complex numbers into last dimension
        pred_p_stack = pred_p_reshape.view(batch_size, seq_len, self.d_v_total*2)

#         # Move real/imag dimension back, merge heads, and stack complex numbers into last dimension
#         pred_p_stack = pred_p.permute(0, 1, 4, 2, 3).reshape(batch_size, 1, seq_len, -1)
        
        # Map back to original basis and get real part
        out = self.W_o(pred_p_stack)
        
        # -----------------
        
        # Return dictionary of metadata
        output_dict = self._get_output_metadata(V_tilde, cos_v, sin_v, E_rel_v, est_latent, mu_v, omega_v, A_hat, V_ij_v, precision_marginal_log, sigma_sq_v, sigma_tilde_sq_v, eta_sq_v, gamma_sq_v, g, abs_idx, seq_len)

        return out, output_dict
