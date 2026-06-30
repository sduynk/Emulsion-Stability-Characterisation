import torch
import numpy as np
import random
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score
import os

def set_seed(seed: int = 42) -> None:
   
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Random seed set as {seed}")
    


def filter_params(model):
    """
    Filters the parameters of a model into two groups: decay_parameters and no_decay_parameters.

    Parameters:
        model (torch.nn.Module): The model whose parameters will be filtered.

    Returns:
        dict: A dictionary containing two lists of parameters: decay_parameters and no_decay_parameters.
    """
    decay = []
    no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'bn' in name or 'norm' in name:
            no_decay.append(param)
        else:
            decay.append(param)
    
    return {"decay": decay, "no_decay": no_decay}
