#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 17:53:55 2021

@author: ghiggi
"""
import torch 
import random
import numpy as np 

def check_torch_device(device):
    """Check torch device validity."""
    if isinstance(device, str): 
        device = torch.device(device)
    if not isinstance(device, torch.device): 
        raise TypeError("Specify device as torch.device or as string : 'cpu','cuda',...")
    return device

def set_pytorch_deterministic(seed=100):
    """Set seeds for deterministic training with pytorch."""
    # TODO
    # - https://pytorch.org/docs/stable/generated/torch.set_deterministic.html#torch.set_deterministic
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    return 

def get_torch_dtype(numeric_precision): 
    """Provide torch dtype based on numeric precision string."""
    dtypes = {'float64': torch.float64,
              'float32': torch.float32,
              'float16': torch.float16,
              'bfloat16': torch.bfloat16
              }
    return dtypes[numeric_precision]

