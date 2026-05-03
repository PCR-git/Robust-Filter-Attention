import numpy as np
import torch
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from utils import complex_matmul
from dynamics import simulate_stochastic_LTI
from dynamics import get_nth_measurement, get_random_measurements

##########################################################################################
##########################################################################################

def construct_random_mapping(S1, Si1, args):
    """
    Embed data in higher dimension.
    Optionally, map through random orthogonal matrix.
    """
    
    m = args.m
    
    # Construct matrices to map between higher/lower dimensions
    Pu = torch.zeros(1, args.d_e, m).to(args.device)
    Pd = torch.zeros(1, m, args.d_e).to(args.device)
    Pu[:,0:m,0:m] = torch.eye(m, m) # Map to higher dim
    Pd[:,0:m,0:m] = torch.eye(m, m) # Map back to lower dim

    if args.rand_embed == 1:
        # Construct random orthogonal matrix and its inverse
        rand_mat = torch.randn(args.d_e,args.d_e).to(args.device)/torch.sqrt(torch.tensor(args.d_e, dtype=torch.float32))
        Q, _ = torch.linalg.qr(rand_mat) # Q is an orthogonal matrix
        R1 = Q # Random complex orthogonal matrix
        R1i = Q.T # For orthogonal matrices, inverse is transpose
    else:
        # Alternatively, instead of a random matrix, just use identity
        R1 = torch.zeros(1, args.d_e, args.d_e).to(args.device).to(args.device)
        R1i = torch.zeros(1, args.d_e, args.d_e).to(args.device).to(args.device)
        R1[:,0:args.m,0:args.m] = torch.eye(args.m,args.m)
        R1i[:,0:args.m,0:args.m] = torch.eye(args.m,args.m)
    
    return Pu, Pd, R1, R1i

##########################################################################################
##########################################################################################

def construct_data_dynamics(A, Pu, Pd, R1, R1i, Npts, t_vector, sigma_process, sigma_measure, args, x0=None, epsilon=1e-5):
    """
    Collect simulated data and map to higher dimension
    """

    if x0 == None:
        rm = 20.0
        rd = 3.0
        theta0 = torch.rand(1) * 2 * np.pi
        r0 = rm + (torch.rand(1) * 2*rd) - rd
        x0 = torch.tensor([r0 * torch.cos(theta0), r0 * torch.sin(theta0)])
        x0 = x0.to(args.device).unsqueeze(0).unsqueeze(-1)

    # Simulate system
    X_true, X_measure_full = simulate_stochastic_LTI(A, x0, Npts, args, sigma_process=sigma_process, sigma_measure = sigma_measure) # Simulate system
    
    if args.t_equal == 1: # If equal time intervals
        idxs, t_measure, X_measure = get_nth_measurement(X_measure_full, t_vector, Npts, n=args.n)
    else: # If unequal time intervals
        idxs, t_measure, X_measure = get_random_measurements(X_measure_full, t_vector, args)
    
    X_high = torch.matmul(Pu,X_measure) # Map to higher dim

    X_random = torch.matmul(R1,X_high) # Map to random basis

    X_random = X_random.squeeze(-1)
    X_high = X_high.squeeze(-1)
    X_true = X_true.squeeze(-1)
    X_measure = X_measure.squeeze(-1)

    return X_random, X_high, X_true, X_measure, t_measure

##########################################################################################
##########################################################################################

class TrainDataset_dynamics(Dataset):
    def __init__(self, Train_Data, X_true_all, X_measure_all, t_measure_all, k_step=1):
        self.Train_Data = Train_Data # [Batch, Seq+1, Dim]
        self.X_true_all = X_true_all
        self.X_measure_all = X_measure_all
        self.t_measure_all = t_measure_all
        self.k_step = k_step

    def __len__(self):
        return self.Train_Data.size(0)

#     def __getitem__(self, idx):
#         # Slice the Data: Inputs (0 to N-1) and Targets (1 to N)
#         # This handles the autoregressive "next-step" requirement
#         x = self.Train_Data[idx, :-1]
#         y = self.Train_Data[idx, 1:]
        
#         t_measure = self.t_measure_all[idx]
#         t_in = t_measure[:-1]

#         # Optional: keep X_true for evaluation metrics
#         x_true = self.X_true_all[idx]
#         x_measure = self.X_measure_all[idx]
        
#         return x, y, x_true, x_measure, t_in
    
    def __getitem__(self, idx):
        t_measure = self.t_measure_all[idx]
        
        # Standard "Shift-by-K" logic (Overlap)
        x = self.Train_Data[idx, :-self.k_step]
        y = self.Train_Data[idx, self.k_step:]
        t_in = t_measure[:-self.k_step]

        return x, y, self.X_true_all[idx], self.X_measure_all[idx], t_in

#     def __getitem__(self, idx):
#         t_measure = self.t_measure_all[idx]
        
#         if self.task_type == 'autoregressive':
#             # Standard "Shift-by-K" logic (Overlap)
#             x = self.Train_Data[idx, :-self.k_step]
#             y = self.Train_Data[idx, self.k_step:]
#             t_in = t_measure[:-self.k_step]
            
#         elif self.task_type == 'seq2seq':
#             # ETT Logic: Non-overlapping blocks
#             # x is the 'Look-back' window
#             x = self.Train_Data[idx, :self.lookback]
#             # y is the 'Forecast' window (starts exactly where x ends)
#             y = self.Train_Data[idx, self.lookback : self.lookback + self.forecast]
            
#             t_in = t_measure[:self.lookback]
            
#             # PHYSICAL KEY: For Seq2Seq, we calculate the dt from the LAST 
#             # point of the look-back to each point in the forecast.
#             t_last_obs = t_measure[self.lookback - 1]
#             t_forecast = t_measure[self.lookback : self.lookback + self.forecast]
    
        return x, y, self.X_true_all[idx], self.X_measure_all[idx], t_in
    
##########################################################################################
##########################################################################################

def create_train_loader_dynamics(A, S1, Si1, Pu, Pd, R1, R1i, t_vector, sigma_process, sigma_measure, args, k_step=1):
    """
    Simulates dynamics and packages them into a DataLoader ready for 
    Project-then-Filter training.
    """
    
    # Initialize storage tensors (Batch, Sequence, Dim)
    # We use args.seq_len + 1 to account for the autoregressive target shift
    Train_Data = torch.zeros(args.num_samp, args.total_L, args.d_e).to(args.device)
    X_true_all = torch.zeros(args.num_samp, args.N_t_tot + 1, 2).to(args.device)
    X_measure_all = torch.zeros(args.num_samp, args.total_L, 2).to(args.device)
    t_measure_all = torch.zeros(args.num_samp, args.total_L).to(args.device)

    print(f"--- Generating {args.num_samp} Dynamic Trajectories ---")
    
    for it in tqdm(range(args.num_samp)):
        # Construct data for one iteration using simulation function
        # This returns the raw trajectory points
        X_random, X_high, X_true, X_measure, t_measure = construct_data_dynamics(
            A, Pu, Pd, R1, R1i, 
            args.N_t_tot, # Ensure simulation length covers target needs
            t_vector, 
            sigma_process, 
            sigma_measure,
            args
        )

        Train_Data[it] = X_random
        X_true_all[it] = X_true
        X_measure_all[it] = X_measure
        t_measure_all[it] = t_measure

    # Create the Custom Dataset
    train_dataset = TrainDataset_dynamics(
        Train_Data, 
        X_true_all, 
        X_measure_all, 
        t_measure_all,
        k_step
    )

    # Create the DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    return train_loader, train_dataset, X_true_all, X_measure_all, t_measure_all
