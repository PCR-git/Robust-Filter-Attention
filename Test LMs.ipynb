import numpy as np
import torch
import torch.nn as nn

from matplotlib import pyplot as plt
import matplotlib.cm as cm
plt.rcParams['figure.figsize'] = [10, 10]
plt.rc('font', size=20)

from utils import complex_matmul

# from model import compute_lambda
from model import ComplexLinearLayer, RFATransformerBlock
from model import MultiheadIsotropicRFA
from model import RFA_Block
from model import RFATransformerNetwork

##########################################################################################
##########################################################################################

def plot_trajectory(X_true,X_measure,X_est):
    """
    Plot actual, measured, and estimated trajectories.
    """

    # Actual trajectory
    X_true_plt = X_true.squeeze().detach().cpu().numpy()
    plt.plot(X_true_plt.T[0],X_true_plt.T[1],'black', label='Ground Truth')

    # Noisy trajectory
    X_noisy = X_measure.squeeze().detach().cpu().numpy()
    plt.plot(X_noisy.T[0], X_noisy.T[1], 'b--', label='Measured')

    # Predicted trajectory
    X_est_plt = X_est.squeeze(0)[0].squeeze().detach().cpu().numpy()
    plt.plot(X_est_plt.T[0], X_est_plt.T[1], 'r--', label='Predicted')

    plt.grid()

##########################################################################################
##########################################################################################

def compute_state_matrix(module,Pu,Pd,R1,R1i,args):
    """
    Reconstructs the global real-valued state transition matrix A from the learned 
    diagonalized SDE parameters. 
    
    This function performs the change-of-basis transformation A = S * Lambda * S^-1, 
    mapping the decoupled latent complex eigenmodes back into the original 
    real-valued feature space [d_e, d_e].
    """

    with torch.no_grad():
        # Get stable eigenvalues
        mu_v, omega_v = compute_lambda(module.mu_v, module.omega_v, args)
        mu_v_expanded = mu_v.unsqueeze(-1).expand(-1, int(omega_v.size()[-1]/2))
        mu_v_vec = mu_v_expanded.repeat_interleave(2, dim=1)

        lambda_model = torch.stack([-mu_v_vec.flatten(), omega_v.flatten()], dim=0)

        # Construct Diagonal Lambda Matrix DD: [2, d_v_total, d_v_total]
        DD = torch.stack((
            torch.diag(lambda_model[0]), 
            torch.diag(lambda_model[1])
        ))

        # Extract Complex Weights for all heads (d_v_total)
        # W_v weights are [2*d_v_total, d_e] -> stack to [2, d_v_total, d_e]
        W_v_r = module.W_v.weight[0:args.d_v_total, :]
        W_v_i = module.W_v.weight[args.d_v_total:2*args.d_v_total, :] 
        W_v = torch.stack((W_v_r, W_v_i), dim=0)

        # W_o weights are [d_e, 2*d_v_total] -> stack to [2, d_e, d_v_total]
        W_o_r = module.W_o.weight[:, 0:args.d_v_total] 
        W_o_i = module.W_o.weight[:, args.d_v_total:2*args.d_v_total] 
        W_o = torch.stack((W_o_r, W_o_i), dim=0)

        # Correct phase for operator reconstruction (maps C space back to R space)
        W_o_rec = W_o.clone()
        W_o_rec[1] *= -1 

        # 4. Complex Matmul: (W_o_rec) * (DD * W_v)
        y_re = torch.matmul(DD[0], W_v[0]) - torch.matmul(DD[1], W_v[1])
        y_im = torch.matmul(DD[0], W_v[1]) + torch.matmul(DD[1], W_v[0])
        y = torch.stack((y_re, y_im))

        # Outside: W_o_rec * y
        A_complex_re = torch.matmul(W_o_rec[0], y[0]) - torch.matmul(W_o_rec[1], y[1])
        A_complex_im = torch.matmul(W_o_rec[0], y[1]) + torch.matmul(W_o_rec[1], y[0])

        # Final state matrix A is the real part: [d_e, d_e]
        A = A_complex_re

    return A

##########################################################################################
##########################################################################################

def plot_state_matrix(A,marker,size=32,color=None):
    """
    Plot entries of the state matrix (real values in blue, imaginary in red)
    """
    
    if A.size()[0] == 1:
        A_real = A[0].flatten().detach().cpu().numpy()
        if color == None:
            plt.scatter(range(4),A_real, c='b', marker=marker,s=size)
        else:
            plt.scatter(range(4),A_real, c=color, marker=marker,s=size)
    else:
        A_real = A[0].flatten().detach().cpu().numpy()
        A_imag = A[1].flatten().detach().cpu().numpy()

        if color == None:
            plt.scatter(range(4),A_real, c='b', marker=marker,s=size)
            plt.scatter(range(4),A_imag, c='r', marker=marker,s=size)
        else:
            plt.scatter(range(4),A_real, c=color, marker=marker,s=size)
            plt.scatter(range(4),A_imag, c=color, marker=marker,s=size)

##########################################################################################
##########################################################################################

def visualize_results(model, train_dataset, history, 
                      R1, R1i, Pu, Pd, A, epoch, folder, args, t_equal=True, t_shift=None, causal=True,
                      plot_losses_flag=False, plot_log_losses_flag=True, plot_traj_flag=True,
                      plot_pulled_forward_estimates_flag=True, plot_last_attn_mat_flag=False,
                      plot_total_precision_flag=False, plot_attn_prior_flag=False, plot_eigenvals_flag=True,
                      plot_decay_per_epoch=True, plot_noise_params_by_type=False, plot_noise_params_by_head=False,
                      plot_tau_and_nu_flag=False, plot_gates_per_epoch=False):
    """
    Visualize results during training using history dictionary for scalar tracking.
    """
    
    # --- Map History Dict to Local Variables ---
    all_losses = np.array(history['loss'])
    all_mus = np.array(history['mu'])
    all_sigmas = np.array(history['sigma'])
    all_sigma_tildes = np.array(history['sigma_tilde'])
    all_etas = np.array(history['eta'])
    all_gammas = np.array(history['gamma'])
    all_taus = np.array(history['tau'])
    all_nu_over_ds = np.array(history['nu_over_d'])
    all_gs = np.array(history['gs_mean'])
    
    # Compute mean epoch losses on the fly for the log-mean plot
    its_per_epoch = len(train_dataset) // args.batch_size
    num_epochs_done = len(all_losses) // its_per_epoch
    if num_epochs_done > 0:
        mean_epoch_losses = [np.mean(all_losses[i*its_per_epoch:(i+1)*its_per_epoch]) 
                             for i in range(num_epochs_done)]
    else:
        mean_epoch_losses = [np.mean(all_losses)]
    
    main_attention_block, last_inner_layer_of_main_attention_block = _get_visual_modules(model)

    # Error checking for the helper function's output
    if main_attention_block is None:
        print("Error: Could not identify main attention block or its last inner layer for visualization. Skipping visualization.")
        return 

    module = last_inner_layer_of_main_attention_block # 'module' refers to the last inner shared attention block

    with torch.no_grad():

        # Get random choice of input
#         rand_idx = np.random.choice(args.num_samp)
        rand_idx = 1
#         train_data, X_true, X_measure, t_measure = train_dataset.__getitem__(rand_idx)
#         inputs = train_data.unsqueeze(0)[:, :-1]

        inputs, target, X_true, X_measure, t_measure = train_dataset.__getitem__(rand_idx)
    
        # Handles unequal time intervals:
        if t_equal == True:
            t_measure = None
        else:
            t_measure = t_measure.unsqueeze(0)

        # Forward pass of model
#         out, output_dict = model.forward(inputs)
        out, output_dict = model(inputs.unsqueeze(0), t_measure=t_measure, t_shift=t_shift, causal=causal)
        
        # Get output metadata
        est = output_dict['est_latent']
        attn_mat = output_dict['attn_mat']
        decayed_attn_mat = output_dict['decayed_attn_mat']
        A_prior = output_dict['A_prior']
        x_hat = output_dict['x_hat']
        eigenvals = output_dict['eigenvals']
        P_tot = output_dict['P_tot']
        gate = output_dict['gate']
        
        est = est.unsqueeze(-1)
        out = out.unsqueeze(1).unsqueeze(-1)
        
        #########################################

        if plot_losses_flag == True:
            # Plot loss
            plt.plot(all_losses)
            plt.title('Loss')
            plt.grid()
            plt.show()

        if plot_log_losses_flag == True:
            # Plot log loss
            log_losses = np.log(all_losses)
            plt.plot(log_losses)
            plt.title('Log Loss')

#             # Calculate baseline from current batch
#             one_step_diff = inputs[:,:,1:] - inputs[:,:,:-1]
#             baseline_val = torch.log(torch.mean(one_step_diff**2)).detach().cpu().numpy()
#             plt.plot(range(len(all_losses)), baseline_val * np.ones(len(all_losses)), 'r--')
            
#             y_min = np.min(log_losses) - 0.5
#             y_max = np.max(log_losses) + 0.5
#             # Only include baseline in view if it's reasonably close
#             plt.ylim(y_min, max(y_max, baseline_val + 0.5))
            
            plt.minorticks_on()
            plt.xlabel("Iteration")
            plt.ylabel("Log Loss")
            plt.grid()
            plt.show()

        #########################################
        
        plt.axis('equal')
        
        # Set plotting dims
        x_max = torch.max(X_true[:,0]).detach().cpu().numpy()
        x_min = torch.min(X_true[:,0]).detach().cpu().numpy()
        y_max = torch.max(X_true[:,1]).detach().cpu().numpy()
        y_min = torch.min(X_true[:,1]).detach().cpu().numpy()
        margin = 2
        
        # Plot trajectory
        if plot_traj_flag:
        
            pred_map = torch.matmul(R1i,out) # Reverse random mapping
            X_est = torch.matmul(Pd,pred_map) # Map back to lower dim

            plot_trajectory(X_true[args.n:].unsqueeze(0),X_measure[1:].unsqueeze(0),X_est)

            plt.xlim(x_min-margin, x_max+margin)
            plt.ylim(y_min-margin, y_max+margin)

    #           plt.savefig(folder + 'trajecs//' + str(epoch) + '.png', bbox_inches='tight')
#             plt.title('Trajectory')
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.show()
    
        #########################################
        
        # Plot pulled-forward estimates
        if args.compute_metadata and plot_pulled_forward_estimates_flag:
            
            x_hat = torch.stack((x_hat, torch.zeros_like(x_hat)),dim=1) # Add zero imaginary part
            
            # Plot state estimates at n_example data points
    #           fig, ax = plt.subplots(figsize=(dim, dim))

            plot_trajectory(X_true[:-args.n].unsqueeze(0),0*X_measure[1:].unsqueeze(0),0*X_est) # Plot actual trajectory

            Xo_h = torch.matmul(R1i,x_hat) # Reverse random mapping
            Xo = torch.matmul(Pd,Xo_h).detach().cpu()[0,0].squeeze(0).squeeze(-1) # State estimates torch.Size([1, 2, 101, 101, 2, 1])
            markers = ['o', 'v', 's', 'd', 'P']
            colors = ['pink', 'red', 'black', 'yellow', 'blue']
            mi = 0
            for i in range(args.seq_len):
                if np.mod(i,int((args.seq_len)/args.n_example)) == 0 and i > 0:
                    xi = Xo[i,i,:]
                    x_est = Xo[i,0:i+1,:].numpy()
                    plt.scatter(x_est.T[0],x_est.T[1], s=10, marker=markers[np.mod(mi,len(markers))], color=colors[np.mod(mi,len(colors))])
                    plt.scatter(xi[0],xi[1], s=100, marker='x', color=colors[np.mod(mi,len(colors))])
                    mi += 1

            plt.xlim(x_min-margin, x_max+margin)
            plt.ylim(y_min-margin, y_max+margin)
    
            plt.gca().set_aspect('equal', adjustable='box')

            plt.xlabel("x")
            plt.ylabel("y")
#             plt.title('State Estimates')
    #           plt.savefig(folder + 'ests//' + str(epoch) + '.png', bbox_inches='tight')
            
            plt.show()
        
        else:
            pass

        #########################################

        # Plot LAST attention matrix in network
        
        if plot_last_attn_mat_flag:               
            # Simplified model:
            for head in range(attn_mat.size()[-1]):
                A = attn_mat[:,:,:,head]
                A_avg = A.squeeze().detach().cpu().numpy() # If using simplified model
                plt.imshow(A_avg**0.25) # (Power is just to increase contrast for better visualization)
        #         plt.savefig(folder + 'attn//' + str(epoch) + '.png', bbox_inches='tight')
                plt.title('Attention Matrix, Head: ' + str(head))
                plt.show()

        if plot_last_attn_mat_flag:               
            # Set a professional style
            plt.style.use('seaborn-v0_8-white') 

            # Extract attention map
            # Assuming attn_mat shape is [Batch, Seq, Seq, Heads]
            # We take the first batch item for visualization
            for head in range(attn_mat.size()[-1]):
                A_hat = attn_mat[0, :, :, head] # Indexing into batch dim if present
                A_hat_avg = A_hat.detach().cpu().numpy()

                fig, ax = plt.subplots(figsize=(10, 10))

                # 'magma' or 'viridis' are perceptually uniform and professional.
                # Power scaling (0.25) is kept for contrast as requested.
                im = ax.imshow(np.power(A_hat_avg, 0.25), cmap='magma', interpolation='nearest')
#                 im = ax.imshow(A_avg, cmap='magma', interpolation='nearest')

                # Professional formatting
#                 ax.set_title(f'Attention Matrix: Head {head+1}', fontsize=12, fontweight='bold', pad=10)
                ax.set_xlabel('Key Index (Past States)', fontsize=16)
                ax.set_ylabel('Query Index (Current State)', fontsize=16)

#                 # Add a colorbar that fits the height of the plot
#                 cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#                 cbar.ax.set_ylabel('Attention Weight (Scaled)', rotation=-90, va="bottom", fontsize=9)

                # Optional: Add head-specific info if SC-RFA regimes are known
                # ax.text(0.5, -0.15, "High-Frequency Regime", transform=ax.transAxes, ha='center', style='italic')

                plt.tight_layout()
                # plt.savefig(f'{folder}/attn/epoch_{epoch}_head_{head}.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                
            # # DECAYED Attention Matrix
                
            # Set a professional style
            plt.style.use('seaborn-v0_8-white') 

            # Extract attention map
            # Assuming attn_mat shape is [Batch, Seq, Seq, Heads]
            # We take the first batch item for visualization
            for head in range(attn_mat.size()[-1]):
                A = decayed_attn_mat[0, :, :, head] # Indexing into batch dim if present
                A_avg = A.detach().cpu().numpy()

                fig, ax = plt.subplots(figsize=(10, 10))

                # 'magma' or 'viridis' are perceptually uniform and professional.
                # Power scaling (0.25) is kept for contrast as requested.
                im = ax.imshow(np.power(A_avg, 0.25), cmap='magma', interpolation='nearest')
#                 im = ax.imshow(A_avg, cmap='magma', interpolation='nearest')

                # Professional formatting
#                 ax.set_title(f'Attention Matrix: Head {head+1}', fontsize=12, fontweight='bold', pad=10)
                ax.set_xlabel('Key Index (Past States)', fontsize=16)
                ax.set_ylabel('Query Index (Current State)', fontsize=16)

#                 # Add a colorbar that fits the height of the plot
#                 cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#                 cbar.ax.set_ylabel('Attention Weight (Scaled)', rotation=-90, va="bottom", fontsize=9)

                # Optional: Add head-specific info if SC-RFA regimes are known
                # ax.text(0.5, -0.15, "High-Frequency Regime", transform=ax.transAxes, ha='center', style='italic')

                plt.tight_layout()
                # plt.savefig(f'{folder}/attn/epoch_{epoch}_head_{head}.png', dpi=300, bbox_inches='tight')
                plt.show()

        #########################################
        
#         # Plot ALL Attention matrices in network
        
#         if plot_all_attn_mats_flag:
#             attn_matrix_count = 0
#             for module_name, module_plt in model.named_modules():            
#                 if hasattr(module_plt, 'attn_mat') and module_plt.attn_mat is not None:

#                     Q_matrix = module_plt.attn_mat # Get the stored attention matrix

#                     # Q_matrix is expected to be (Batch, seq_len, seq_len)
#                     # Squeeze the batch dimension for plotting if it's a single batch
#                     Q_ij_viz = Q_matrix.detach().cpu().numpy() # (seq_len, seq_len)

#                     plt.imshow(Q_ij_viz**0.25)
#                     plt.title(f'Attention Matrix - Layer {attn_matrix_count}')
#                     plt.show()
#                     attn_matrix_count += 1
        
        #########################################
    
        if plot_attn_prior_flag:
            # Plot attention priors
            for head in range(A_prior.size()[-1]):
                A_prior_i = A_prior[:,:,head].detach().cpu().numpy()
                plt.imshow(A_prior_i)
                plt.title('Attn Prior, Head: ' + str(head))
                plt.show()
    
        #########################################
        
        # Plot row sum of unnormalized attention (measure of total confidence)
        if plot_total_precision_flag:
            for head in range(args.n_heads):
                P_tot_head = P_tot.detach().cpu().numpy()[0,:,head]
                plt.plot(P_tot_head)
            plt.grid()
            plt.title('Total Confidence (row sum of unnormalized attention)')
            plt.show()
        
        #########################################
        
#         if g != None and plot_gate == 1:
#             plt.plot(g)
        
        #########################################        

#         if plot_eigenvals_flag:
#             # Plot eigenvalues per epoch
#             lambdas = eigenvals.detach().cpu().numpy() # Unsorted lambdas

#             fig, ax1 = plt.subplots()

#             # Left axis: Decay (blue circles)
#             ax1.scatter(np.arange(args.d_v), lambdas[0], color='b', marker='o')
#             ax1.set_ylabel('Decay ($\mu$)', color='blue')
#             ax1.tick_params(axis='y', labelcolor='blue')

#             # Right axis: Rotation (red crosses)
#             ax2 = ax1.twinx()
#             ax2.scatter(np.arange(args.d_v), lambdas[1], color='r', marker='x')
#             ax2.set_ylabel('Rotation ($\omega$)', color='red')
#             ax2.tick_params(axis='y', labelcolor='red')
            
#             ax1.set_xlabel('Embedding Dimension Index')

#             # Use ax1 for the grid to ensure it stays in the background
#             ax1.grid(True)

#             plt.show()

#         if plot_eigenvals_flag:
#                 # Plot eigenvalues per epoch
#                 lambdas = eigenvals.detach().cpu().numpy() # Unsorted lambdas

#                 fig, ax1 = plt.subplots()
               
#                 # Left axis: Decay (blue circles)
#                 BLUE_PROF = '#1B4F72' 
#                 RED_PROF = '#943126'
#                 ax1.scatter(np.arange(args.d_v), -lambdas[0], color=BLUE_PROF, marker='o')
#                 ax1.set_ylabel('Decay ($\mu$)', color=BLUE_PROF)
#                 ax1.tick_params(axis='y', labelcolor=BLUE_PROF)

#                 # Right axis: Rotation (red crosses)
#                 ax2 = ax1.twinx()
#                 ax2.scatter(np.arange(args.d_v), lambdas[1], color=RED_PROF, marker='x')
#                 ax2.set_ylabel('Rotation ($\omega$)', color=RED_PROF)
#                 ax2.tick_params(axis='y', labelcolor=RED_PROF)

#                 # Align grid and zero-line
#                 # Force ax2 to include 0 and draw the dotted line
#                 ax2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)

#                 # Setting limits to ensure 0 is the baseline for the rotation grid
#                 # This helps the grid lines of ax1 and ax2 align at the zero-point
#                 curr_y_min, curr_y_max = ax2.get_ylim()
#                 ax2.set_ylim(min(0, curr_y_min), curr_y_max)

#                 ax1.set_xlabel('Embedding Dimension Index')

#                 # Standard grid on ax1
#                 ax1.grid(True, which='both', linestyle='-', alpha=0.3)

#                 plt.show()

        import matplotlib.ticker as ticker

        if plot_eigenvals_flag:
            # Plot eigenvalues per epoch
            lambdas = eigenvals.detach().cpu().numpy() # Unsorted lambdas

            # Aspect ratio: 2/3 as high as wide
            fig, ax1 = plt.subplots(figsize=(9, 6))

            # Left axis: Decay (blue circles)
            BLUE_PROF = '#1B4F72' 
            RED_PROF = '#943126'
            ax1.scatter(np.arange(args.d_v), -lambdas[0], color=BLUE_PROF, marker='o', alpha=0.8)
            ax1.set_ylabel(r'Decay ($\mu$)', color=BLUE_PROF, fontsize=12, fontweight='bold')
            ax1.tick_params(axis='y', labelcolor=BLUE_PROF)

            # Right axis: Rotation (red crosses)
            ax2 = ax1.twinx()
            ax2.scatter(np.arange(args.d_v), lambdas[1], color=RED_PROF, marker='x', linewidths=1.2)
            ax2.set_ylabel(r'Rotation ($\omega$)', color=RED_PROF, fontsize=12, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor=RED_PROF)

            # --- Ticker Logic for X-Axis ---
            # Major ticks every 64 (with labels)
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(64))
            # Minor ticks every 8 (no labels, just ticks)
            ax1.xaxis.set_minor_locator(ticker.MultipleLocator(8))

            # Make the ticks visible and clean
            ax1.tick_params(axis='x', which='major', labelsize=10, width=1.5, length=6)
            ax1.tick_params(axis='x', which='minor', width=1, length=3)

            # --- Prominent Head Partitions & Labels ---
            head_size = args.d_v // 8
            y_max = ax1.get_ylim()[1]

            for i in range(0, 9):
                ax1.axvline(x=i * head_size, color='#424949', linestyle='--', linewidth=1.5, alpha=0.6)

            for i in range(8):
                label_x = (i * head_size) + (head_size / 2)
                ax1.text(label_x, y_max * 1.02, f'Head {i+1}', 
                         ha='center', va='bottom', fontsize=12, color='#212F3D', fontweight='bold')

            # Align grid and zero-line
            ax2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)

            # Setting limits to ensure 0 is the baseline for the rotation grid
            curr_y_min, curr_y_max = ax2.get_ylim()
            ax2.set_ylim(min(0, curr_y_min), curr_y_max)

            ax1.set_xlabel('Embedding Dimension Index', fontsize=11, fontweight='bold')

            # Standard grid on ax1
            ax1.grid(True, which='major', linestyle='-', alpha=0.2)

            plt.xlim(-2, 258)

            # Adjust layout to prevent text clipping
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

        #########################################
        
        if plot_decay_per_epoch:
            # Plot decay per epoch
            for i in range(args.n_heads):
                # We label each line as "Head 0", "Head 1", etc.
                plt.plot(all_mus[:, i], label=f'Head {i}')
            
            plt.xlabel("Iteration")
            plt.ylabel("Decay")
            plt.grid()
#             plt.grid(True, which='both', linestyle='--', alpha=0.5)
            plt.title('Decay per Iteration (Values)')
            
            plt.legend(loc='best')
                
            plt.tight_layout()
            plt.show()
            
#             # Plot log decay per epoch
#             for i in range(args.n_heads):
#                 plt.plot(np.log(all_mus[:,i]))
#             plt.xlabel("Epoch")
#             plt.ylabel("Log Decay")
#             plt.grid()
#             plt.title('Log Decay per Epoch')
#             plt.show()
            
        #########################################
        
        if plot_noise_params_by_type == True:
            # We define a helper list to iterate through the four noise types
            noise_data = [
                (all_sigmas, r'$\sigma$', 'Process Noise'),
                (all_sigma_tildes, r'$\tilde{\sigma}$', 'Steady State Noise'),
                (all_etas, r'$\eta$', 'Key Measurement Noise'),
                (all_gammas, r'$\gamma$', 'Query Measurement Noise')
            ]

            for data, tex_label, title_label in noise_data:
                plt.figure(figsize=(8, 5))
                for H in range(args.n_heads):
                    plt.plot(data[:, H], label=f'Head {H}')

                plt.xlabel('Iteration')
                plt.ylabel('Value')
                plt.title(f'{title_label} ({tex_label}) - All Heads')

                # Use a multi-column legend if you have many heads
                plt.legend(loc='best', fontsize='x-small', ncol=2)
                plt.grid(True, alpha=0.3)
                plt.show()
        
        if plot_noise_params_by_head == True:
            for H in range(args.n_heads):        
                plt.plot(all_sigmas[:, H], label=r'$\sigma$')
                plt.plot(all_sigma_tildes[:, H], label=r'$\tilde{\sigma}$')
                plt.plot(all_etas[:, H], label=r'$\eta$')
                plt.plot(all_gammas[:, H], label=r'$\gamma$')

                plt.xlabel('Iteration')
                plt.ylabel('Value')
                plt.title(f'Noise Params (Head {H})')
                plt.legend()
                plt.grid()
                plt.show()
        
        #########################################   
        
        if plot_tau_and_nu_flag == True:
            # --- Plot Inverse Temperature ---
            for h in range(args.n_heads):
                plt.plot(all_taus[:, h], label=f'Head {h}')
            
            plt.grid(True, alpha=0.3)
            plt.title(fr'Inverse Temperature ($\frac{{\tau}}{{d}}$)')
            plt.xlabel('Iteration')
            plt.ylabel(r'Value')
            plt.legend(loc='best', fontsize='x-small', ncol=2)
            plt.show()

            # --- Plot Robustness Parameter ---
            for h in range(args.n_heads):
                plt.plot(all_nu_over_ds[:, h], label=f'Head {h}')
                
            plt.grid(True, alpha=0.3)
            plt.title(fr'Robustness Parameter ($\frac{{\nu}}{{d}}$)')
            plt.xlabel('Iteration')
            plt.ylabel(r'Value')
            plt.legend(loc='best', fontsize='x-small', ncol=2)
            plt.show()
            
        #########################################
        
        if plot_gates_per_epoch == True:
            for i in range(args.n_heads):
                plt.imshow(all_gs[:, :, i].squeeze(-1).squeeze(-1))
                plt.show()

        #########################################

#         # Plot values of state matrix
#         _, A_model = compute_state_matrix(module,Pu,Pd,R1,R1i,args)
#         plot_state_matrix(A_model, marker='x',size=80)
#         plot_state_matrix(A, marker='o')

#         plt.title('State Matrix Comparison')
#         plt.grid()
#         plt.show()
        
        #########################################
        
#         # Compute eigenvals of effective A and print
#         A_model_np = A_model.detach().cpu().numpy()
#         complex_matrix = A_model_np[0] + 1j * A_model_np[1]  # shape (2, 2)
#         eigenvals = np.linalg.eigvals(complex_matrix)
#         print(eigenvals)

#         # Access W_v and W_o from the main_attention_block as these are typically at the Nlayer level
#         W_v = get_complex_weights(main_attention_block, 'W_v')
#         W_o = get_complex_weights(main_attention_block, 'W_o')
#         print('W_o * W_v: (should be near I:')
#         print(complex_matmul(W_o, W_v).squeeze().detach().cpu()[0][0:args.m,0:args.m]) # Should be near identity
        
##########################################################################################
##########################################################################################

def visualize_results_attn(model, train_dataset, all_losses, R1, R1i, Pu, Pd, A, epoch, args):
    """
    Visualize results during training
    Plots the following:
        Noisy and true trajectory, and prediction of model
        State estimates and n_example data points
        Attention matrix
        Values of state matrix
    """
    
    folder = None
    plt.axis('equal')
    
    with torch.no_grad():

        # print(module.lambda1)

        # Get prediction for random choice of input
#         rand_idx = np.random.choice(args.num_samp)
        rand_idx = 1

        inputs, target, X_true, X_measure, t_measure = train_dataset.__getitem__(rand_idx)

        out, attn_list = model.forward(inputs.unsqueeze(0), args.causal)

        out = out.unsqueeze(1)

        # Set plotting dims
        x_max = torch.max(X_true[:,0]).detach().cpu().numpy()
        x_min = torch.min(X_true[:,0]).detach().cpu().numpy()
        y_max = torch.max(X_true[:,1]).detach().cpu().numpy()
        y_min = torch.min(X_true[:,1]).detach().cpu().numpy()
        margin = 2

        #########################################
        
        # Plot trajectory
#         fig, ax = plt.subplots(figsize=(dim, dim))
        
        pred_map = torch.matmul(R1i,out.unsqueeze(-1)) # Reverse random mapping
        est = torch.matmul(Pd,pred_map) # Map back to lower dim
        plot_trajectory(X_true[args.n:].unsqueeze(0),X_measure[1:].unsqueeze(0),est)
        
        plt.xlim(x_min-margin, x_max+margin)
        plt.ylim(y_min-margin, y_max+margin)
        
#         plt.savefig(folder + 'trajecs//' + str(epoch) + '.png', bbox_inches='tight')
        plt.show()
    
        #########################################

        # # Plot losses
        # plt.plot(all_losses)
        # plt.grid()
        # plt.show()
        #       plt.plot(mean_epoch_losses)
        #       plt.grid()
        #       plt.show()
        
        #########################################
        
#         # Plot attention matrices
#         for attn in attn_list:
# #             print(attn.size())
#             for head in range(args.n_heads):
#                 plt.imshow(attn[head].detach().cpu().numpy())
#                 plt.show()
                
        # Use a scientific colormap like 'magma' (black-to-red-to-white) 
        # which is great for seeing low-intensity "noise" or "diffusion"
        CMAP = 'magma' 

        for layer_idx, attn in enumerate(attn_list):
            for head in range(args.n_heads):
                fig, ax = plt.subplots(figsize=(10, 10))

                # Convert to numpy and plot
                # Using aspect='equal' to keep the matrix square
                img = ax.imshow((attn[head].detach().cpu().numpy())**0.25, 
                                cmap=CMAP, 
                                interpolation='nearest', 
                                origin='upper')

                # Professional Title showing both Layer and Head
#                 ax.set_title(f'Attention Matrix (Layer {layer_idx+1}, Head {head+1})', 
#                              fontsize=12, fontweight='bold', pad=12)

                # Standard Axis Labels
                ax.set_xlabel('Key Index (Past States)', fontsize=16)
                ax.set_ylabel('Query Index (Current State)', fontsize=16)

                # Remove the frame for a cleaner look
                for spine in ax.spines.values():
                    spine.set_visible(False)

                plt.tight_layout()
        plt.show()
        
        #########################################

        # Plot log mean loss per epoch
        plt.plot(np.log(all_losses))
        
        one_step_diff = inputs[:,1:]- inputs[:,:-1]
        baseline_loss = torch.log(torch.mean(one_step_diff**2)).detach().cpu().numpy()
        plt.plot(range(epoch),baseline_loss*np.ones(epoch), 'r--')
        
        plt.minorticks_on()
        plt.grid()
        plt.show()

##########################################################################################
##########################################################################################        
        
def _get_visual_modules(model: nn.Module):
    """
    Identifies and returns the main model and its
    last inner attention layer from a given model instance,
    for use in visualize_results.

    This helper function adapts to different top-level model architectures.

    Args:
        model (nn.Module): The top-level model instance.

    Returns:
        tuple[nn.Module | None, nn.Module | None]: A tuple containing:
            - The main model instance for visualization.
            - The last attention layer instance within that main block.
            Returns (None, None) if the required modules cannot be found.
    """
    
    main_attention_block = None
    last_inner_layer_of_main_attention_block = None
            
   # Model is RFATransformerBlock
    if isinstance(model, RFATransformerBlock):
        if hasattr(model, 'attn'):
            if isinstance(model.attn, MultiheadIsotropicRFA):
                main_attention_block = model.attn
    
    # Model is RFATransformerNetwork
    # (This network contains 'blocks', which are RFATransformerBlock instances)
    elif isinstance(model, RFATransformerNetwork):
        # We'll take the attention block from the *first* RFATransformerBlock in the network
        if hasattr(model, 'blocks') and len(model.blocks) > 0:
            first_transformer_block = model.blocks[0]
            if hasattr(first_transformer_block, 'attn'):
                if isinstance(first_transformer_block.attn, MultiheadIsotropicRFA):
                    main_attention_block = first_transformer_block.attn            
            
    ###################
            
    # If a main_attention_block was found, now try to get its last inner layer
    if main_attention_block is not None:
        if isinstance(main_attention_block, MultiheadIsotropicRFA): 
            last_inner_layer_of_main_attention_block = main_attention_block
        else:
            print(f"Warning: Main attention block ({main_attention_block.__class__.__name__}) found, "
                  f"but its last inner layer (FullPrecisionAttentionBlockShared) could not be retrieved. "
                  f"Some visualizations might not work.")
            
    ###################

    # Model is RFA_Block
    if isinstance(model, RFA_Block):
        if hasattr(model, 'layers') and len(model.layers) > 0 and \
           isinstance(model.layers[0], MultiheadIsotropicRFA):
            main_attention_block = model.layers[0]
            last_inner_layer_of_main_attention_block = model.layers[0]
            
    return main_attention_block, last_inner_layer_of_main_attention_block

##########################################################################################
##########################################################################################

# def plot_eigenvals(A,eigenvals=None):
#     """
#     Compute and plot eigenvalues of A
#     """
    
#     if eigenvals == None:
#         A_np = A.detach().cpu().numpy()
#         complex_matrix = A_np[0] + 1j * A_np[1]  # shape (2, 2)
#         eigenvals = np.linalg.eigvals(complex_matrix).squeeze()
#         eig_r = eigenvals.real
#         eig_i = eigenvals.imag
#     else:
#         eigenvals = eigenvals.squeeze().detach().cpu().numpy()
#         eig_r = eigenvals[0]
#         eig_i = eigenvals[1]

#     eig_abs = np.flip(np.sort(np.abs(eig_r.T)))
#     plt.plot(eig_abs)

#     eig_abs = np.flip(np.sort(np.abs(eig_i.T)))
#     plt.plot(eig_abs)
    
#     plt.grid()
#     plt.show()