import numpy as np
import torch
import torch.nn as nn

from matplotlib import pyplot as plt
import matplotlib.cm as cm
plt.rcParams['figure.figsize'] = [10, 10]
plt.rc('font', size=20)

##########################################################################################
##########################################################################################

def single_epoch_rfa_dynamics(model, train_loader, history, optimizer, criterion, params_list, args, t_equal=True, t_shift=None, causal=True, scheduler=None):
    """
    Single epoch of training
    """
    
#     all_dicts = [] # Dictionary for outputs
        
    # Iterate through training data
    for it, (inputs, target, X_true, X_measure, t_measure) in enumerate(train_loader):

        optimizer.zero_grad() # Zero out gradients
        
        # Handles unequal time intervals:
        if t_equal == True:
            t_measure = None
        
#         with torch.autograd.set_detect_anomaly(True):
        
        out, output_dict = model(inputs, t_measure=t_measure, t_shift=t_shift, causal=causal) # Forward pass of model

        loss = criterion(out, target) # Compute loss
        
        loss.backward()
        
        ### Clip the Gradients ###
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clip grads (optional)
            
        ##########################
        # The sde parameters can be unstable, so we clip their gradients separately
        # Collect all sde parameters from the model
        sde_physics_params = [
            p for n, p in model.named_parameters() 
            if any(key in n for key in ['mu_', 'sigma_', 'eta_', 'gamma_'])
        ]
        
        if sde_physics_params:
            # We use a very tight norm to ensure the SDE trajectory 
            # and noise floor evolve slowly and stay numerically stable.
            torch.nn.utils.clip_grad_norm_(sde_physics_params, max_norm=1e-4)
        ##########################

        optimizer.step()

        # Update the learning rate
        if scheduler is not None:
            scheduler.step()
        
        #############################
        
        # Capture the loss
        history['loss'].append(loss.item())
        
        # Capture the 1D noise params (mean across batch, but keep per-head)
        history['mu'].append(output_dict['mu_v'].detach().cpu().numpy())
        history['sigma'].append(output_dict['sigma_sq_v'].detach().cpu().numpy())
        history['sigma_tilde'].append(output_dict['sigma_tilde_sq_v'].detach().cpu().numpy())
        history['eta'].append(output_dict['eta_sq_v'].detach().cpu().numpy())
        history['gamma'].append(output_dict['gamma_sq_v'].detach().cpu().numpy())
        
        # Capture inverse temp and robustness param
        history['tau'].append(output_dict['tau'].detach().cpu().numpy())
        history['nu_over_d'].append(output_dict['nu_over_d'].detach().cpu().numpy())
        
        # Capture a summary of the gate
        if output_dict['gate'] is not None:
            # Store mean gate value per head for this iteration
            history['gs_mean'].append(output_dict['gate'].mean(axis=0).detach().cpu().numpy())

    return output_dict, history

##########################################################################################
##########################################################################################

def single_epoch_attn_dynamics(model, train_loader, optimizer, loss, params_list, args, scheduler=None):
    """
    Single epoch of training
    """
    
    epoch_losses = [] 

    # Iterate through training data
    for it, (inputs, target, X_true, X_measure, t_measure) in enumerate(train_loader):

        optimizer.zero_grad() # Zero out gradients

        # Single iteration of training
#         epoch_losses[it] = single_iter_attn(model, optimizer, loss, inputs, outputs, args)

        out, _ = model(inputs, args.causal)

        loss_i = loss(out, target) # Compute loss
        
        epoch_losses.append(loss_i.item()) # Record losses

        loss_i.backward() # Backprop

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clip gradient
        
        optimizer.step()
        
        # Update the learning rate
        if scheduler is not None:
            scheduler.step()
        
    return epoch_losses

##########################################################################################
##########################################################################################

def hook_fn(grad):
    """
    Hook function to get gradients and plot
    """
    
#     print(grad)
    grad_mean = grad.mean(dim=[0, 1, 2])
#     print(f"Mean gradient per component: {grad_mean}")
    plt.scatter(np.arange(args.embed_dim),grad_mean.detach().cpu().numpy())
    plt.show()
#     print(f"Max gradient per component: {grad.abs().max(dim=[0, 1, 2])}")
    return grad
