import torch
import numpy as np

##########################################################################################
##########################################################################################

def get_nth_measurement(X_measure_full, t_v, N_t, n):
    """
    Get evenly spaced measurements along the trajectory (at a spacing of n points)
    """
#     idxs = torch.arange(0, N_t, n) # Indices of sample states
#     X_measure = X_measure_full[idxs]  # Measured states
#     t_measure =  t_v[idxs] # Measured times
    idxs = torch.arange(0, N_t+n, n) # Indices of sample states
    X_measure = X_measure_full[idxs]  # Measured states
    t_measure =  t_v[idxs] # Measured times
    return idxs, t_measure, X_measure

##########################################################################################
##########################################################################################

def get_random_measurements(X_measure_full, t_v, args):
    """
    Get Npts random measurements from the trajectory
    """
    idxs = np.sort(np.random.choice(len(t_v), args.seq_len+1, replace=False)) # Indices of sample states
    X_measure = X_measure_full[idxs] # Measured states
    t_measure =  t_v[idxs] # Measured times
    return idxs, t_measure, X_measure
