import numpy as np
import torch
import os

from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [10, 10]
plt.rc('font', size=20)

##########################################################################################
##########################################################################################

def plot_training_progress_lm(history, epoch, model_path, save=True):
    """
    Plots training loss per iteration and validation perplexity per epoch.
    Saves the figure to the specified model path.
    """
    plt.figure(figsize=(12, 5))

#     # --- Plot 1: Iteration Loss ---
#     loss = history['loss']
#     plt.subplot(1, 2, 1)
#     plt.plot(loss, label='Train Loss (Iter)', color='blue', alpha=0.3)
    
#     # Add a rolling average to see the trend through the noise
#     if len(history['loss']) > 100:
#         window_size = 100
#         rolling_avg = np.convolve(loss, np.ones(window_size)/window_size, mode='valid')
#         plt.plot(range(window_size - 1, len(history['loss'])), rolling_avg, label='Rolling Avg (100)', color='red')
    
#     plt.title('Training Loss')
#     plt.xlabel('Iteration')
#     plt.ylabel('Cross Entropy')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
    # --- Plot 1: Log Loss ---
    loss = np.array(history['loss'])
    log_loss = np.log(loss)
    plt.subplot(1, 2, 1)
    plt.plot(log_loss, label='Train Loss (Iter)', color='blue', alpha=0.3)
    
    # Add a rolling average to see the trend through the noise
    if len(history['loss']) > 100:
        window_size = 100
        rolling_avg = np.convolve(log_loss, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size - 1, len(history['loss'])), rolling_avg, label='Rolling Avg (100)', color='red')
    
    plt.title('Training Perplexity')
    plt.xlabel('Iteration')
    plt.ylabel('Perplexity (Log Cross Entropy)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- Plot 2: Validation Perplexity ---
    plt.subplot(1, 2, 2)
    # Ensure x-axis matches the number of perplexity entries we have
    epochs_range = range(1, len(history['val_ppl']) + 1)
    plt.plot(epochs_range, history['val_ppl'], marker='o', color='green', label='Val PPL')
    
    plt.title('Validation Perplexity (PPL)')
    plt.xlabel('Epoch')
    plt.ylabel('PPL')
    plt.yscale('log') # PPL movements are often exponential
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    
    print('PPL history:', history['val_ppl'])

    plt.tight_layout()
    
    # Ensure the directory exists before saving
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
    
    if save == True:
        save_name = os.path.join(model_path, f'training_progress_epoch_{epoch+1}.png')
        plt.savefig(save_name)
        plt.show() 
        plt.close() # Close figure to free up memory
    
##########################################################################################
##########################################################################################


def visualize_rfa_lm(model, val_dataset, history, epoch, folder, args,
                    save=False,
                    plot_log_losses_flag=True,
                    plot_last_attn_mat_flag=True,
                    plot_decay_per_iteration=True,
                    plot_noise_params=True,
                    plot_tau_and_nu_flag=True):
    """
    Plots RFA dynamics with colorblind-safe, professional styling.
    Uses Okabe-Ito inspired palette to avoid red/green confusion.
    """
    model.eval()
    
    # --- Colorblind-Friendly Palette (Okabe-Ito inspired) ---
    # Optimized for contrast and readability.
    cb_colors = [
        '#0072B2', # Deep Blue
        '#E69F00', # Orange
        '#CC79A7', # Soft Magenta (Replaces Red)
        '#56B4E9', # Sky Blue
        '#F0E442', # Yellow
        '#000000', # Black
        '#D55E00', # Vermillion
        '#999999'  # Gray
    ]
    
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "grid.alpha": 0.2,
        "grid.linestyle": "-"
    })

    # --- Data Preparation ---
    # Extract everything from history into numpy arrays immediately
    history_data = {k: np.array(v) for k, v in history.items()}

    def plot_heads(data, ylabel, filename, is_memory_floor=False, is_alpha=False, plot_gates_per_epoch=True):
        plt.figure(figsize=(7, 4))
        for h in range(args.n_heads):
            # Omit unitary heads (0, 1) for floor or alpha calculations
            if (is_memory_floor or is_alpha) and h < 2: 
                continue
                
            y = data[:, h]
            plt.plot(y, color=cb_colors[h % len(cb_colors)], label=f'Head {h}', 
                     linewidth=1.2, alpha=0.9)

        if is_alpha:
            plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
            # Adjust text placement based on data length
            text_x = len(data) * 0.02
            plt.text(text_x, 0.15, 'Integrative', fontsize=8, alpha=0.6, va='bottom')
            plt.text(text_x, -0.15, 'Diffusive', fontsize=8, alpha=0.6, va='top')

        plt.xlabel('Iteration')
        plt.ylabel(ylabel)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
        plt.grid(True)
        
        if save:
            os.makedirs(folder, exist_ok=True)
            plt.savefig(os.path.join(folder, f'{filename}_epoch_{epoch}.pdf'), 
                        bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()

    with torch.no_grad():
        # Get the raw item from the dataset
        rand_idx = np.random.choice(len(val_dataset))
        raw_item = val_dataset[rand_idx]
        
        if isinstance(raw_item, dict):
            item = raw_item['input_ids']
        else:
            item = raw_item

        if not isinstance(item, torch.Tensor):
            item = torch.tensor(item)
            
        inputs = item.unsqueeze(0).to(args.device)
        _, output_dict = model(inputs, t_measure=None, causal=True)
        attn_mat = output_dict.get('attn_mat', None)
        
        # --- 1. Log Loss Plot ---
        if plot_log_losses_flag:
            plt.figure(figsize=(7, 4))
            losses = np.log(history_data['loss'])
            plt.plot(losses, color=cb_colors[0], alpha=0.15, label='Log Iter Loss')
            if len(losses) > 100:
                smooth = np.convolve(losses, np.ones(100)/100, mode='valid')
                plt.plot(smooth, color='#000000', linewidth=1.5, label='Moving Avg')
            plt.xlabel('Iteration')
            plt.ylabel('log(Loss)')
            plt.legend(loc='upper right', frameon=False)
            plt.grid(True)
            if save:
                plt.savefig(os.path.join(folder, f'log_loss_epoch_{epoch}.pdf'), bbox_inches='tight')
            plt.show()
            plt.close()

        # --- 2. Attention Matrices ---
        if plot_last_attn_mat_flag and attn_mat is not None:
             n_heads = attn_mat.size(1)
             fig, axes = plt.subplots(1, n_heads, figsize=(n_heads * 4, 4))
             if n_heads == 1: axes = [axes]
             for h in range(n_heads):
                 A = attn_mat[0, h].cpu().numpy()
                 axes[h].imshow(A**0.25, cmap='magma') 
                 axes[h].set_axis_off()
                 axes[h].set_title(f"H{h}", fontsize=10)
             plt.tight_layout()
             plt.show() 
             plt.close()

        # --- 3. Learned Decay Tracking ---
        if plot_decay_per_iteration:
             plot_heads(history_data['mu'], r"Damping Rate ($\mu$)", 'decay_evolution')

        # --- 4. Noise Params ---
        if plot_noise_params:
            plot_heads(history_data['sigma'], r'$\sigma^2$', 'sigma_sq')
            plot_heads(history_data['eta'], r'$\eta^2$', 'eta_sq')
            plot_heads(history_data['gamma'], r'$\gamma^2$', 'gamma_sq')
            
            # Steady state process noise: Sigma^2 / 2Mu 
            mu = history_data['mu']
            sigma = history_data['sigma']
            floor = sigma / (2 * mu + 1e-10)
            plot_heads(floor, r'Steady state process noise ($\sigma^2/2\mu$)', 'memory_floor', is_memory_floor=True)

            # --- 4c. Phase Transition Parameter (Alpha) ---
            # Corrected: Uses the local variables extracted from history_data
            eta = history_data['eta']
            alpha = eta - floor
            plot_heads(alpha, r'Phase Parameter ($\alpha$)', 'alpha_phase', is_alpha=True)
                
        # --- 5. Tau and Nu Tracking ---
        if plot_tau_and_nu_flag:
            plot_heads(history_data['tau'], r'Inv. Temp ($\tau$)', 'tau_tracking')
            plot_heads(history_data['nu_over_d'], r'Robustness ($\nu/d$)', 'nu_tracking')

#         if plot_gates_per_epoch == True:
#             # Set up a shared X-axis for all heads for better comparison
#             for i in range(args.n_heads):
#                 # Slice for the last query token only
#                 # Shape: [epochs, sequence_length]
#                 all_gs = np.array(history['gs_mean'])
#                 last_token_gates = all_gs[:, -1, i] 
                
#                 plt.figure(figsize=(10, 4))
#                 plt.imshow(last_token_gates, aspect='auto', cmap='viridis')
#                 plt.colorbar(label='Gate Value (Precision)')
#                 plt.title(f'Head {i}: Precision Gate for Last Token across Training')
#                 plt.xlabel('Historical Token Lag ($\Delta t$)')
#                 plt.ylabel('Training Epoch')
#                 plt.show()
                