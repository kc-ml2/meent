import random
import os
import numpy as np
import torch

import constants

# 160, 192
# 40, 48
def carve_pattern(
        upper_idx,  
        lower_idx, 
        pattern, # (1, 1, 1, n)
        fill_value=constants.AIR
    ):
    B = pattern.shape[0]
    C = pattern.shape[1]
    height = lower_idx - upper_idx
    width = pattern.shape[-1]
    
    if isinstance(pattern, np.ndarray): 
        canvas = np.full((B, C, width, width), fill_value)
    elif isinstance(pattern, torch.Tensor):
        canvas =  torch.full((B, C, width, width), fill_value)

    if isinstance(pattern, np.ndarray):
        pattern = pattern.repeat(height, 2)
    elif isinstance(pattern, torch.Tensor):
        pattern = pattern.repeat(1, 1, height, 1)

    canvas[:, :, upper_idx:lower_idx, :] = pattern
    
    return canvas # (1, 1, n, n)


def to_blob(n_times, patterns): # expects Tensor input
    return torch.stack(
        [np.repeat(i, n_times) for i in patterns]
    )

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True