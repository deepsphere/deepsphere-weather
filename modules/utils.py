import os
import math
import xarray as xr
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import cm, colors

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dropout, Conv2D, Lambda
import tensorflow.keras.backend as K


def load_test_data(path, lead_time, years=slice('2017', '2018')):
    """ Function to load test labels
    
    Parameters
    ----------
    path : str
        Path to data folder
    lead_time : int
        Prediction interval (in hours)
    years : slice(str)
        Start and end years defining the testing period
    
    Returns
    -------
    test_data : xr.Dataset
        Z500 and T850 for the selected test years
    """
    zpath = path + '/geopotential_500'
    tpath = path + '/temperature_850'
    
    z = xr.open_mfdataset(zpath+'/*.nc', combine='by_coords')['z']
    t = xr.open_mfdataset(tpath+'/*.nc', combine='by_coords')['t']

    try:
        z = z.drop('level')
    except ValueError:
        pass

    try:
        t = t.drop('level')
    except ValueError:
        pass

    dataset = xr.merge([z, t], compat='override')

    return dataset.sel(time=years).isel(time=slice(lead_time, None))
        

def init_device(model, gpu=None):
    """Initialize device based on cpu/gpu and number of gpu
    Parameters
    ----------
    device : str
        cpu or gpu
    gpu : list of int
        List of gpus that should be used. 
    model : torch.nn.Module
        The model to place on the device(s)
        
    Returns
    -------
    torch.Module, torch.device: 
        The model placed on device and the device
    """
    
    if torch.cuda.is_available():
        if gpu is None:
            device = torch.device("cuda")
            model = model.to(device)
            model = nn.DataParallel(model)
        elif len(gpu) == 1:
            device = torch.device("cuda:{}".format(gpu[0]))
            model = model.to(device)
        else:
            device = torch.device("cuda:{}".format(gpu[0]))
            model = model.to(device)
            model = nn.DataParallel(model, device_ids=[i for i in gpu])
    else:
        device = torch.device("cpu")
        model = model.to(device)

    return model, device


class Adam_tf(Optimizer):
    """Implements the Adam algorithm variant that tensorflow uses

    Parameters
    ----------
    params iterable): iterable of parameters to optimize or dicts defining
        parameter groups
    lr : float, optional): 
        Learning rate (default: 1e-3)
    betas (Tuple[float, float], optional): 
        Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
    eps (float, optional): 
        Term added to the denominator to improve numerical stability (default: 1e-7)
    weight_decay (float, optional): 
        Weight decay (L2 penalty) (default: 0)
    amsgrad (boolean, optional): 
        Whether to use the AMSGrad variant of this algorithm from [2] (default: False)
        
    References
    ----------
    [1] Adam: A Method for Stochastic Optimization: https://arxiv.org/abs/1412.6980
    [2] On the Convergence of Adam and Beyond: https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-7,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(myAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam_tf, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Parameters
        ---------
        closure : callable (optional)
            A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data) #, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data) #, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data) #, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt()).add_(group['eps'])

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

def transfer_weights(tf_model, torch_model):
    """ Example function on how to transfer weights from tensorflow model to pytorch model
    
    Parameters
    ----------
    tf_model : tensorflow model
        Weight donor
    torch_model : pytorch model (torch.nn.Modules)
        Weight recipient
    
    Returns
    -------
    torch_model : pytorch model
        Model with updated weights
    
    """
    def convert_weight(W):
        return torch.tensor(W.transpose(3, 2, 0, 1), dtype=torch.float)

    def convert_bias(B):
        return torch.tensor(B, dtype=torch.float)

    all_tf_weights = tf_model.get_weights()
    
    torch_model.conv1.conv.weight.data = convert_weight(all_tf_weights[0])
    torch_model.conv1.conv.bias.data = convert_bias(all_tf_weights[1])

    torch_model.conv2.conv.weight.data = convert_weight(all_tf_weights[2])
    torch_model.conv2.conv.bias.data = convert_bias(all_tf_weights[3])

    torch_model.conv3.conv.weight.data = convert_weight(all_tf_weights[4])
    torch_model.conv3.conv.bias.data = convert_bias(all_tf_weights[5])

    torch_model.conv4.conv.weight.data = convert_weight(all_tf_weights[6])
    torch_model.conv4.conv.bias.data = convert_bias(all_tf_weights[7])

    torch_model.conv5.conv.weight.data = convert_weight(all_tf_weights[8])
    torch_model.conv5.conv.bias.data = convert_bias(all_tf_weights[9])
    
    return torch_model