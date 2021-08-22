#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import torch
import os
import copy
from datetime import datetime
import math
import numpy as np
import tqdm

import torch.nn.functional as F

from modules.dataloader_autoregressive import get_AR_batch
from modules.dataloader_autoregressive import AutoregressiveDataset
from modules.dataloader_autoregressive import AutoregressiveDataLoader

def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i : i + n].view(tensor.shape))
        i += n
    return outList


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(model, 
             # Data
             da_dynamic,
             da_static = None,              
             da_bc = None, 
             # Scaler options
             scaler = None,
             # Dataloader options
             batch_size = 64, 
             num_workers = 0,
             prefetch_factor = 2, 
             prefetch_in_gpu = False,  
             pin_memory = False,
             asyncronous_gpu_transfer = True,
             device = 'cpu',
             numeric_precision = "float32", 
             # Autoregressive settings  
             input_k = [-3,-2,-1], 
             output_k = [0],
             forecast_cycle = 1,                           
             AR_iterations = 2, 
             stack_most_recent_prediction = True,
             **kwargs):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))

    dataset = AutoregressiveDataset(da_dynamic = da_dynamic,  
                                    da_bc = da_bc,
                                    da_static = da_static,
                                    scaler = scaler, 
                                    # Autoregressive settings  
                                    input_k = input_k,
                                    output_k = output_k,
                                    forecast_cycle = forecast_cycle,                           
                                    AR_iterations = AR_iterations, 
                                    stack_most_recent_prediction = stack_most_recent_prediction, 
                                    # GPU settings 
                                    device = device,
                                    # Precision settings
                                    numeric_precision = numeric_precision)
    
    dataloader = AutoregressiveDataLoader(dataset = dataset, 
                                          batch_size = batch_size,  
                                          drop_last_batch = False, 
                                          random_shuffle = False,
                                          num_workers = num_workers,
                                          prefetch_factor = prefetch_factor, 
                                          prefetch_in_gpu = prefetch_in_gpu,  
                                          pin_memory = pin_memory,
                                          asyncronous_gpu_transfer = asyncronous_gpu_transfer, 
                                          device = device)

    n = 0

    with torch.no_grad():
        ##--------------------------------------------------------------------.     
        # Iterate along training batches       
        for batch_dict in dataloader: 
            # batch_dict = next(iter(batch_dict))
            ##----------------------------------------------------------------.      
            ### Perform autoregressive loop
            dict_Y_predicted = {}
            for i in range(AR_iterations+1):
                # Retrieve X and Y for current AR iteration
                # - Torch Y stays in CPU with training_mode=False
                torch_X, _ = get_AR_batch(AR_iteration = i, 
                                        batch_dict = batch_dict, 
                                        dict_Y_predicted = dict_Y_predicted,
                                        device = device, 
                                        asyncronous_gpu_transfer = asyncronous_gpu_transfer,
                                        training_mode=False)
            
                input_var = torch.autograd.Variable(torch_X)
                b = input_var.data.size(0)

                momentum = b / (n + b)
                for module in momenta.keys():
                    module.momentum = momentum

                dict_Y_predicted[i] = model(input_var, **kwargs)
                n += b
                del torch_X, input_var

    model.apply(lambda module: _set_momenta(module, momenta))

def bn_update_with_loader(model, 
                          loader, 
                          AR_iterations = 2, 
                          asyncronous_gpu_transfer = True,
                          device = 'cpu',
                          **kwargs):
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0

    with torch.no_grad():
        ##--------------------------------------------------------------------.     
        # Iterate along training batches       
        for batch_dict in loader: 
            # batch_dict = next(iter(batch_dict))
            ##----------------------------------------------------------------.      
            ### Perform autoregressive loop
            dict_Y_predicted = {}
            for i in range(AR_iterations+1):
                # Retrieve X and Y for current AR iteration
                # - Torch Y stays in CPU with training_mode=False
                torch_X, _ = get_AR_batch(AR_iteration = i, 
                                        batch_dict = batch_dict, 
                                        dict_Y_predicted = dict_Y_predicted,
                                        device = device, 
                                        asyncronous_gpu_transfer = asyncronous_gpu_transfer,
                                        training_mode=False)
            
                input_var = torch.autograd.Variable(torch_X)
                b = input_var.data.size(0)

                momentum = b / (n + b)
                for module in momenta.keys():
                    module.momentum = momentum

                dict_Y_predicted[i] = model(input_var, **kwargs)
                n += b
                del torch_X, input_var
            
            del dict_Y_predicted

    model.apply(lambda module: _set_momenta(module, momenta))
