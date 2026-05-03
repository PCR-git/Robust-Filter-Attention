import numpy as np
import torch
import os
import random

###############################################
###############################################

def count_parameters(model):
    """
    Count the number of trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

###############################################
###############################################

def get_layers(model, layer_class):
    """
    Recursively collect all layers of a given type from a model.

    """
    return [module for module in model.modules() if isinstance(module, layer_class)]

###############################################
###############################################

def seed_everything(seed=2025):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # for multi-GPU
    
    # CRITICAL for GPU reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For PyTorch 2.x/torch.compile stability
    torch.use_deterministic_algorithms(False) # Set to True only if you don't mind a speed hit