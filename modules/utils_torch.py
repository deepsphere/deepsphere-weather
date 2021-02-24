#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 17:53:55 2021

@author: ghiggi
"""
import torch 
import random
import numpy as np 
import time

def check_torch_device(device):
    """Check torch device validity."""
    if isinstance(device, str): 
        device = torch.device(device)
    if not isinstance(device, torch.device): 
        raise TypeError("Specify device as torch.device or as string : 'cpu','cuda',...")
    return device

def check_pin_memory(pin_memory, num_workers, device):
    """ Check pin_memory possibility."""
    if not isinstance(pin_memory, bool):
        raise TypeError("'pin_memory' must be either True or False. If num_workers > 0, set to False ;) ")
    # CPU case
    if device.type == 'cpu':
        if pin_memory:
            print("- GPU is not available. 'pin_memory' set to False.")
            pin_memory = False
    # GPU case with multiprocess
    if num_workers > 0 and pin_memory: 
        print("- Pinned memory can't be shared across processes! \n It is not possible to pin tensors into memory in each worker process. \n If num_workers > 0, pin_memory is set to False.")
        pin_memory = False    
    return pin_memory

def check_prefetch_in_GPU(prefetch_in_GPU, num_workers, device):
    """Check prefetch_in_GPU possibility."""
    if not isinstance(prefetch_in_GPU, bool):
        raise TypeError("'prefetch_in_GPU' must be either True or False. If num_workers > 0, set to False ;) ")
    # CPU case
    if device.type == 'cpu':
        if prefetch_in_GPU:
            print("- GPU is not available. 'prefetch_in_GPU' set to False.")
            prefetch_in_GPU = False
    # GPU case with multiprocess
    elif num_workers > 0 and prefetch_in_GPU: 
        print("- Prefetch in GPU with multiprocessing is currently unstable.\n\
            It is generally not recommended to return CUDA tensors within multi-process data loading\
                loading because of many subtleties in using CUDA and sharing CUDA tensors.")
        prefetch_in_GPU = False    
    else: # num_workers = 0 
        prefetch_in_GPU = prefetch_in_GPU
    return prefetch_in_GPU

def check_asyncronous_GPU_transfer(asyncronous_GPU_transfer, device):
    """Check asyncronous_GPU_transfer possibility."""
    if not isinstance(asyncronous_GPU_transfer, bool):
        raise TypeError("'asyncronous_GPU_transfer' must be either True or False.")
    # CPU case
    if device.type == 'cpu':
        if asyncronous_GPU_transfer:
            print("- GPU is not available. 'asyncronous_GPU_transfer' set to False.")
            asyncronous_GPU_transfer = False
    return asyncronous_GPU_transfer

def check_prefetch_factor(prefetch_factor, num_workers):     
    """Check prefetch_factor validity."""
    if not isinstance(prefetch_factor, int):
        raise TypeError("'prefetch_factor' must be positive integer.")
    if prefetch_factor < 0: 
        raise ValueError("'prefetch_factor' must be positive.")
    if num_workers == 0 and prefetch_factor !=2:
        prefetch_factor = 2 # bug in pytorch ... need to set to 2 
    return prefetch_factor

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

#-----------------------------------------------------------------------------.
# ####################
#### Timing utils ####
# ####################
def get_synchronized_cuda_time():
    """Get time after CUDA synchronization.""" 
    torch.cuda.synchronize()
    return time.time()

def get_time_function(device):
    """General function returing a time() function.""" 
    if isinstance(device, str): 
        device = torch.device(device)
    if device.type == "cpu":
        return time.time
    else:
        return get_synchronized_cuda_time