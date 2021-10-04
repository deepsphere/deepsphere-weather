#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 18:42:12 2021

@author: ghiggi
"""
import os
import torch
import time
import pickle
import dask
import numpy as np
from tabulate import tabulate 

from modules.dataloader_autoregressive import AutoregressiveDataset
from modules.dataloader_autoregressive import AutoregressiveDataLoader
from modules.dataloader_autoregressive import get_aligned_ar_batch
from modules.dataloader_autoregressive import remove_unused_Y
from modules.dataloader_autoregressive import cylic_iterator
from modules.utils_autoregressive import check_ar_settings
from modules.utils_autoregressive import check_input_k
from modules.utils_autoregressive import check_output_k 
from modules.utils_training import AR_TrainingInfo
from modules.utils_torch import check_device
from modules.utils_torch import check_pin_memory
from modules.utils_torch import check_asyncronous_gpu_transfer
from modules.utils_torch import check_prefetch_in_gpu
from modules.utils_torch import check_prefetch_factor
from modules.utils_torch import check_ar_training_strategy
from modules.utils_torch import get_time_function
from modules.utils_xr import xr_is_aligned
from modules.loss import reshape_tensors_4_loss 

from modules.utils_swag import bn_update_with_loader
##----------------------------------------------------------------------------.
# TODOs
# - ONNX for saving model weights 
# - Record the loss per variable 
# - Compute additional metrics (R2, bias, rsd)

#-----------------------------------------------------------------------------.
# ############################
#### Autotune num_workers ####
# ############################       
def timing_AR_Training(dataset,
                       model, 
                       optimizer, 
                       criterion, 
                       ar_scheduler, 
                       ar_training_strategy = "AR",
                       # DataLoader options
                       batch_size = 32,
                       shuffle = True, 
                       shuffle_seed = 69,
                       num_workers = 0, 
                       prefetch_in_gpu = False,
                       prefetch_factor = 2,
                       pin_memory = False,
                       asyncronous_gpu_transfer = True,
                       # Timing options 
                       training_mode = True,
                       n_repetitions = 10,
                       verbose = True):
    """
    Time execution and memory consumption of AR training.

    Parameters
    ----------
    dataset : AutoregressiveDataset
        AutoregressiveDataset
    model : pytorch model
        pytorch model.
    optimizer : pytorch optimizer
        pytorch optimizer.
    criterion : pytorch criterion
        pytorch criterion
    num_workers : 0, optional
        Number of processes that generate batches in parallel.
        0 means ONLY the main process will load batches (that can be a bottleneck).
        1 means ONLY one worker (just not the main process) will load data 
        A high enough number of workers usually assures that CPU computations 
        are efficiently managed. However, increasing num_workers increase the 
        CPU memory consumption.
        The Dataloader prefetch into the CPU prefetch_factor*num_workers batches.
        The default is 0.        
    batch_size : int, optional
        Number of samples within a batch. The default is 32.
    shuffle : bool, optional
        Wheter to random shuffle the samples at each epoch or when ar_iterations are updated.
        The default is True.
    shuffle_seed : int, optional
        Empower deterministic random shuffling.
        The shuffle_seed is increased by 1 when ar_iterations are updated. 
    prefetch_factor: int, optional 
        Number of sample loaded in advance by each worker.
        The default is 2.
    prefetch_in_gpu: bool, optional 
        Whether to prefetch 'prefetch_factor'*'num_workers' batches of data into GPU instead of CPU.
        By default it prech 'prefetch_factor'*'num_workers' batches of data into CPU (when False)
        The default is False.
    pin_memory : bool, optional
        When True, it prefetch the batch data into the pinned memory.  
        pin_memory=True enables (asynchronous) fast data transfer to CUDA-enabled GPUs.
        Useful only if training on GPU.
        The default is False.
    asyncronous_gpu_transfer: bool, optional 
        Only used if 'prefetch_in_gpu' = True. 
        Indicates whether to transfer data into GPU asynchronously 
    training_mode : bool, optional 
        Whether to compute the gradients or time the "validation mode".
        The default is True.
    n_repetitions : int, optional
        Number of runs to time. The default is 10.
    verbose : bool, optional
        Wheter to print the timing summary. The default is True.

    Returns
    -------
    timing_info : dict
        Dictionary with timing information of AR training.

    """
    ##------------------------------------------------------------------------.
    # Check at least 1 pass is done 
    if n_repetitions < 1: 
        n_repetitions = 1
    ##------------------------------------------------------------------------.
    if not isinstance(num_workers, int):
        raise TypeError("'num_workers' must be a integer larger than 0.")
    if num_workers < 0: 
        raise ValueError("'num_workers' must be a integer larger than 0.")
    if not isinstance(training_mode, bool):
        raise TypeError("'training_mode' must be either True or False.")
    ##------------------------------------------------------------------------.    
    # Retrieve informations 
    ar_iterations = dataset.ar_iterations
    device = dataset.device                      
    # Retrieve function to get time 
    get_time = get_time_function(device)
    # Retrieve custom ar_batch_fun fuction
    ar_batch_fun = dataset.ar_batch_fun  
    ##------------------------------------------------------------------------.
    # Get dimension infos
    dim_info = dataset.dim_info
    feature_info = dataset.feature_info
    dim_info_dynamic = dim_info['dynamic']
    # feature_names_dynamic = list(feature_info['dynamic'])
    ##------------------------------------------------------------------------.
    # Initialize model 
    if training_mode:
        model.train()
    else: 
        model.eval()  
    ##------------------------------------------------------------------------.
    # Initialize list 
    Dataloader_timing = []  
    ar_batch_timing = []  
    ar_data_removal_timing = []  
    ar_forward_timing = []
    ar_loss_timing = []
    Backprop_timing = [] 
    Total_timing = []
    ##-------------------------------------------------------------------------.
    # Initialize DataLoader 
    trainingDataLoader = AutoregressiveDataLoader(dataset = dataset,                                                   
                                                  batch_size = batch_size,  
                                                  drop_last_batch = True,
                                                  shuffle = shuffle,
                                                  shuffle_seed = shuffle_seed,
                                                  num_workers = num_workers,
                                                  prefetch_factor = prefetch_factor, 
                                                  prefetch_in_gpu = prefetch_in_gpu,  
                                                  pin_memory = pin_memory,
                                                  asyncronous_gpu_transfer = asyncronous_gpu_transfer, 
                                                  device = device)
    trainingDataLoader_iter = iter(trainingDataLoader)
    ##-------------------------------------------------------------------------.
    # Measure the size of model parameters 
    if device.type != 'cpu':
        model_params_size = torch.cuda.memory_allocated()/1000/1000
    else: 
        model_params_size = 0
    ##-------------------------------------------------------------------------.
    # Repeat training n_repetitions
    with torch.set_grad_enabled(training_mode):
        for count in range(n_repetitions):  
            t_start = get_time()
            # Retrieve batch
            t_i = get_time()
            training_batch_dict = next(trainingDataLoader_iter)
            Dataloader_timing.append(get_time() - t_i)
            # Perform AR iterations 
            dict_training_Y_predicted = {}
            dict_training_loss_per_ar_iteration = {}
            ##---------------------------------------------------------------------.
            # Initialize stuff for AR loop timing 
            tmp_ar_data_removal_timing = 0
            tmp_ar_batch_timing = 0 
            tmp_ar_forward_timing = 0 
            tmp_ar_loss_timing = 0
            tmp_ar_backprop_timing = 0
            batch_memory_size = 0 
            for ar_iteration in range(ar_iterations+1):
                # Retrieve X and Y for current AR iteration   
                t_i = get_time()
                torch_X, torch_Y = ar_batch_fun(ar_iteration = ar_iteration, 
                                                batch_dict = training_batch_dict, 
                                                dict_Y_predicted = dict_training_Y_predicted,
                                                device = device, 
                                                asyncronous_gpu_transfer = asyncronous_gpu_transfer)
                tmp_ar_batch_timing = tmp_ar_batch_timing + (get_time() - t_i)
                ##-----------------------------------------------------------------.                                
                # Measure model parameters + batch size in MB 
                if device.type != 'cpu' and ar_iteration == 0:
                    batch_memory_size = torch.cuda.memory_allocated()/1000/1000 - model_params_size
                ##-----------------------------------------------------------------.
                # Forward pass and store output for stacking into next AR iterations
                t_i = get_time()
                dict_training_Y_predicted[ar_iteration] = model(torch_X)
                tmp_ar_forward_timing = tmp_ar_forward_timing + (get_time() - t_i) 
                ##-----------------------------------------------------------------.
                # Compute loss for current forecast iteration 
                # - The criterion currently expects [data_points, nodes, features]
                #   So we collapse all other dimensions to a 'data_points' dimension  
                # TODO to generalize AR_Training to whatever tensor formats:
                # - reshape_tensors_4_loss should be done in criterion 
                # - criterion should get args: dim_info, dynamic_features, 
                # - criterion can perform internally "per-variable loss", "per-variable masking"
                t_i = get_time()   
                Y_pred, Y_obs = reshape_tensors_4_loss(Y_pred = dict_training_Y_predicted[ar_iteration],
                                                       Y_obs = torch_Y,
                                                       dim_info_dynamic = dim_info_dynamic)
                dict_training_loss_per_ar_iteration[ar_iteration] = criterion(Y_obs, Y_pred)
                tmp_ar_loss_timing = tmp_ar_loss_timing + (get_time() - t_i)
                ##-----------------------------------------------------------------.
                # If ar_training_strategy is "AR", perform backward pass at each AR iteration 
                if ar_training_strategy == "AR":
                    # - Detach gradient of Y_pred (to avoid RNN-style optimization)
                    if training_mode: 
                        dict_training_Y_predicted[ar_iteration] = dict_training_Y_predicted[ar_iteration].detach()   # TODO: should not be detached after backward?
                    # - AR weight the loss (aka weight sum the gradients ...)
                    t_i = get_time() 
                    dict_training_loss_per_ar_iteration[ar_iteration] = dict_training_loss_per_ar_iteration[ar_iteration]*ar_scheduler.ar_weights[ar_iteration]
                    tmp_ar_loss_timing = tmp_ar_loss_timing + (get_time() - t_i)
                    # - Measure model size requirements
                    if device.type != 'cpu':
                        model_memory_allocation = torch.cuda.memory_allocated()/1000/1000
                    else: 
                        model_memory_allocation = 0    
                    # - Backpropagate to compute gradients (the derivative of the loss w.r.t. the parameters)
                    if training_mode: 
                        t_i = get_time()  
                        dict_training_loss_per_ar_iteration[ar_iteration].backward()
                        tmp_ar_backprop_timing = tmp_ar_backprop_timing + (get_time() - t_i)
                    # - Update the total (AR weighted) loss
                    t_i = get_time()  
                    if ar_iteration == 0:
                        training_total_loss = dict_training_loss_per_ar_iteration[ar_iteration] 
                    else: 
                        training_total_loss += dict_training_loss_per_ar_iteration[ar_iteration]
                    tmp_ar_loss_timing = tmp_ar_loss_timing + (get_time() - t_i)
                ##------------------------------------------------------------.
                # Remove unnecessary stored Y predictions 
                t_i = get_time()
                remove_unused_Y(ar_iteration = ar_iteration, 
                                dict_Y_predicted = dict_training_Y_predicted,
                                dict_Y_to_remove = training_batch_dict['dict_Y_to_remove'])
                del Y_pred, Y_obs, torch_X, torch_Y
                if ar_iteration == ar_iterations:
                    del dict_training_Y_predicted
                tmp_ar_data_removal_timing = tmp_ar_data_removal_timing + (get_time()- t_i)
                    
            ##-------------------------------------------------------------------.       
            # If ar_training_strategy is RNN, perform backward pass after all AR iterations            
            if ar_training_strategy == "RNN":
                t_i = get_time()
                # - Compute total (AR weighted) loss 
                for i, (ar_iteration, loss) in enumerate(dict_training_loss_per_ar_iteration.items()):
                    if i == 0:
                        training_total_loss = ar_scheduler.ar_weights[ar_iteration] * loss 
                    else: 
                        training_total_loss += ar_scheduler.ar_weights[ar_iteration] * loss
                tmp_ar_loss_timing = tmp_ar_loss_timing + (get_time() - t_i)
                # - Measure model size requirements
                if device.type != 'cpu':
                    model_memory_allocation = torch.cuda.memory_allocated()/1000/1000
                else: 
                    model_memory_allocation = 0    
                if training_mode:       
                    # - Perform backward pass 
                    t_i = get_time()
                    training_total_loss.backward()
                    tmp_ar_backprop_timing = tmp_ar_backprop_timing + (get_time() - t_i)
            ##--------------------------------------------------------------------.   
            # Update the network weights 
            if training_mode:  
                t_i = get_time()       
                # - Update the network weights 
                optimizer.step()  
                ##----------------------------------------------------------------.
                # Zeros all the gradients for the next batch training 
                # - By default gradients are accumulated in buffers (and not overwritten)
                optimizer.zero_grad(set_to_none=True)  
                tmp_ar_backprop_timing = tmp_ar_backprop_timing + (get_time() - t_i)
                    
            ##--------------------------------------------------------------------.
            # Summarize timing 
            ar_batch_timing.append(tmp_ar_batch_timing)
            ar_data_removal_timing.append(tmp_ar_data_removal_timing)
            ar_forward_timing.append(tmp_ar_forward_timing)
            ar_loss_timing.append(tmp_ar_loss_timing)  
            Backprop_timing.append(tmp_ar_backprop_timing)
            ##--------------------------------------------------------------------.
            # - Total time elapsed
            Total_timing.append(get_time() - t_start)

    ##------------------------------------------------------------------------.
    # Create timing info dictionary 
    timing_info = {'Run': list(range(n_repetitions)), 
                   'Total': Total_timing, 
                   'Dataloader': Dataloader_timing,
                   'AR Batch': ar_batch_timing,
                   'Delete': ar_data_removal_timing,
                   'Forward': ar_forward_timing,
                   'Loss': ar_loss_timing,
                   'Backward': Backprop_timing}
    ##-------------------------------------------------------------------------. 
    memory_info = {'Model parameters': model_params_size,
                   'Batch': batch_memory_size,
                   'Forward pass': model_memory_allocation}  
    
    ##-------------------------------------------------------------------------. 
    # Create timing table 
    if verbose:
        table = []
        headers = ['Run', 'Total', 'Dataloader','AR Batch', 'Delete', 'Forward', 'Loss', 'Backward']
        for count in range(n_repetitions):
            table.append([count,    
                         round(Total_timing[count], 4),
                         round(Dataloader_timing[count], 4),
                         round(ar_batch_timing[count], 4),
                         round(ar_data_removal_timing[count], 4),
                         round(ar_forward_timing[count], 4),
                         round(ar_loss_timing[count], 4),
                         round(Backprop_timing[count], 4)
                          ])
        print(tabulate(table, headers=headers))   
        if device.type != 'cpu':
            print("- Model parameters requires {:.2f} MB in GPU".format(memory_info['Model parameters']))                     
            print("- A batch with {} samples for {} AR iterations allocate {:.2f} MB in GPU".format(batch_size, ar_iterations, memory_info['Batch']))
            print("- The model forward pass allocates {:.2f} MB in GPU.".format(memory_info['Forward pass']))    
    ##------------------------------------------------------------------------.    
    ### Reset model to training mode
    model.train() 
    ##------------------------------------------------------------------------. 
    ### Delete Dataloader to avoid deadlocks
    del trainingDataLoader_iter
    del trainingDataLoader
    ##------------------------------------------------------------------------.               
    return timing_info, memory_info

def tune_num_workers(dataset,
                     model, 
                     optimizer, 
                     criterion, 
                     num_workers_list, 
                     ar_scheduler,                 
                     ar_training_strategy = "AR",  
                     # DataLoader options
                     batch_size = 32, 
                     shuffle = True,
                     shuffle_seed = 69, 
                     prefetch_in_gpu = False,
                     prefetch_factor = 2,
                     pin_memory = False,
                     asyncronous_gpu_transfer = True,
                     # Timing options
                     training_mode = True, 
                     n_repetitions = 10,
                     n_pass_to_skip = 4, 
                     summary_stat = "max", 
                     verbose = True):
    """
    Search for the best value of 'num_workers'.

    Parameters
    ----------
    dataset : AutoregressiveDataset
        AutoregressiveDataset
    model : pytorch model
        pytorch model.
    optimizer : pytorch optimizer
        pytorch optimizer.
    criterion : pytorch criterion
        pytorch criterion
    ar_scheduler : 
        Scheduler regulating the changes in loss weights (per AR iteration) during RNN/AR training 
    ar_training_strategy : str 
        Either "AR" or "RNN" 
        "AR" perform the backward pass at each AR iteration 
        "RNN" perform the backward pass after all AR iterations
    num_workers_list : list
        A list of num_workers to time.
    batch_size : int, optional
        Number of samples within a batch. The default is 32.
    prefetch_factor: int, optional 
        Number of sample loaded in advance by each worker.
        The default is 2.
    prefetch_in_gpu: bool, optional 
        Whether to prefetch 'prefetch_factor'*'num_workers' batches of data into GPU instead of CPU.
        By default it prech 'prefetch_factor'*'num_workers' batches of data into CPU (when False)
        The default is False.
    pin_memory : bool, optional
        When True, it prefetch the batch data into the pinned memory.  
        pin_memory=True enables (asynchronous) fast data transfer to CUDA-enabled GPUs.
        Useful only if training on GPU.
        The default is False.
    asyncronous_gpu_transfer: bool, optional 
        Only used if 'prefetch_in_gpu' = True. 
        Indicates whether to transfer data into GPU asynchronously 
    training_mode : bool, optional 
        Whether to compute the gradients or time the "validation mode".
        The default is True.
    n_repetitions : int, optional
        Number of runs to time. The default is 10.
    n_pass_to_skip : int 
        The default is 2. 
        Avoid timing also the worker initialization when num_workers > 0
    summary_stat : bool, optional 
        Statical function to summarize timing 
        The default is 'max'.
        Valid values are ('min','mean','median','max').
        The first 'n_pass_to_skip' batch pass are excluded because they might not 
        be representative of the actual performance (workers initializations)
    verbose : bool, optional
        Wheter to print the timing summary. The default is True.

    Returns
    -------
    optimal_num_workers : int
        Optimal num_workers to use for efficient data loading.
    """
    ##------------------------------------------------------------------------.
    # Check at least 1 pass is done 
    if n_pass_to_skip < 0:
        n_pass_to_skip = 0 
    n_repetitions = n_repetitions + n_pass_to_skip
    ##------------------------------------------------------------------------.
    # Checks arguments 
    if isinstance(num_workers_list, int):
        num_workers_list = [num_workers_list]
    # Define summary statistic 
    if summary_stat == "median":
        summary_fun = np.median 
    elif summary_stat == "max":
        summary_fun = np.max
    elif summary_stat == "min":
        summary_fun = np.min
    elif summary_stat == "mean":
        summary_fun = np.mean
    else:
        raise ValueError("Valid summary_stat values are ('min','mean','median','max').")
    ##------------------------------------------------------------------------.
    # Initialize dictionary 
    Dataloader_timing = {i: [] for i in num_workers_list }
    ar_batch_timing = {i: [] for i in num_workers_list }
    ar_data_removal_timing = {i: [] for i in num_workers_list }
    ar_forward_timing = {i: [] for i in num_workers_list }
    ar_loss_timing = {i: [] for i in num_workers_list }
    Backprop_timing = {i: [] for i in num_workers_list }
    Total_timing = {i: [] for i in num_workers_list }
    Memory_Info = {i: [] for i in num_workers_list }
    ##------------------------------------------------------------------------.
    # Time AR training for specified num_workers in num_workers_list
    for num_workers in num_workers_list:  
        timing_info, memory_info = timing_AR_Training(dataset = dataset, 
                                                      model = model, 
                                                      optimizer = optimizer,
                                                      criterion = criterion,
                                                      ar_scheduler = ar_scheduler, 
                                                      ar_training_strategy = ar_training_strategy, 
                                                      # DataLoader options
                                                      batch_size = batch_size, 
                                                      shuffle = shuffle,
                                                      shuffle_seed = shuffle_seed, 
                                                      num_workers = num_workers, 
                                                      prefetch_in_gpu = prefetch_in_gpu,
                                                      prefetch_factor = prefetch_factor,
                                                      pin_memory = pin_memory,
                                                      asyncronous_gpu_transfer = asyncronous_gpu_transfer,
                                                      # Timing options 
                                                      training_mode = training_mode, 
                                                      n_repetitions = n_repetitions,
                                                      verbose = False) 
        Dataloader_timing[num_workers] = timing_info['Dataloader'][n_pass_to_skip:]
        ar_batch_timing[num_workers] = timing_info['AR Batch'][n_pass_to_skip:]
        ar_data_removal_timing[num_workers] = timing_info['Delete'][n_pass_to_skip:]
        ar_forward_timing[num_workers] = timing_info['Forward'][n_pass_to_skip:]
        ar_loss_timing[num_workers] = timing_info['Loss'][n_pass_to_skip:]
        Backprop_timing[num_workers] = timing_info['Backward'][n_pass_to_skip:]
        Total_timing[num_workers] = timing_info['Total'][n_pass_to_skip:]
        Memory_Info[num_workers] = memory_info

    ##------------------------------------------------------------------------. 
    ### Summarize timing results      
    headers = ['N. workers', 'Total', 'Dataloader','AR Batch', 'Delete', 'Forward', 'Loss', 'Backward']
    table = []
    dtloader = []
    for num_workers in num_workers_list:
        dtloader.append(summary_fun(Dataloader_timing[num_workers]).round(4))
        table.append([num_workers,    
                      summary_fun(Total_timing[num_workers]).round(4),
                      summary_fun(Dataloader_timing[num_workers]).round(4),
                      summary_fun(ar_batch_timing[num_workers]).round(4),
                      summary_fun(ar_data_removal_timing[num_workers]).round(4),
                      summary_fun(ar_forward_timing[num_workers]).round(4),
                      summary_fun(ar_loss_timing[num_workers]).round(4),
                      summary_fun(Backprop_timing[num_workers]).round(4)])
    ##------------------------------------------------------------------------.
    # Select best num_workers
    optimal_num_workers = num_workers_list[np.argmin(dtloader)]
    ##------------------------------------------------------------------------.
    # Print timing results  
    if verbose:
        print(tabulate(table, headers=headers)) 
        if dataset.device.type != 'cpu':
            memory_info = Memory_Info[optimal_num_workers]
            print("- Model parameters requires {:.2f} MB in GPU.".format(memory_info['Model parameters']))                     
            print("- A batch with {} samples for {} AR iterations allocate {:.2f} MB in GPU.".format(batch_size, dataset.ar_iterations, memory_info['Batch']))
            print("- The model forward pass allocates {:.2f} MB in GPU.".format(memory_info['Forward pass']))    
    ##------------------------------------------------------------------------.
    return optimal_num_workers
        
#----------------------------------------------------------------------------.              
# #########################
#### Training function ####
# #########################
def AutoregressiveTraining(model, 
                           model_fpath,        
                           # Loss settings 
                           criterion,
                           ar_scheduler,
                           early_stopping,
                           optimizer, 
                           # Data
                           training_data_dynamic, 
                           training_data_bc = None,   
                           data_static = None,
                           validation_data_dynamic = None,
                           validation_data_bc = None, 
                           bc_generator = None, 
                           scaler = None,
                           # AR_batching_function
                           ar_batch_fun = get_aligned_ar_batch,
                           # Dataloader options
                           prefetch_in_gpu = False,
                           prefetch_factor = 2,
                           drop_last_batch = True,
                           shuffle = True, 
                           shuffle_seed = 69, 
                           num_workers = 0, 
                           autotune_num_workers = False, 
                           pin_memory = False,
                           asyncronous_gpu_transfer = True,
                           # Autoregressive settings  
                           input_k = [-3,-2,-1], 
                           output_k = [0],
                           forecast_cycle = 1,                           
                           ar_iterations = 6, 
                           stack_most_recent_prediction = True,
                           # Training settings 
                           ar_training_strategy = "AR", 
                           lr_scheduler = None, 
                           training_batch_size = 128,
                           validation_batch_size = 128, 
                           epochs = 10, 
                           scoring_interval = 10, 
                           save_model_each_epoch = False,
                           ar_training_info = None, 
                           # SWAG settings
                           swag = False,
                           swag_model = None,
                           swag_freq = 10,
                           swa_start = 8,
                           # GPU settings 
                           device = 'cpu'):
    """AutoregressiveTraining.
    
    ar_batch_fun : callable 
            Custom function that batch/stack together data across AR iterations. 
            The custom function must return a tuple of length 2 (X, Y), but X and Y 
            can be whatever desired objects (torch.Tensor, dict of Tensor, ...). 
            The custom function must have the following arguments: 
                def ar_batch_fun(ar_iteration, batch_dict, dict_Y_predicted,
                                 device = 'cpu', asyncronous_gpu_transfer = True)
            The default ar_batch_fun function is the pre-implemented get_aligned_ar_batch() which return 
            two torch.Tensor: one for X (input) and one four Y (output). Such function expects 
            the dynamic and bc batch data to have same dimensions and shape.
    if early_stopping=None, no ar_iteration update
    """
    with dask.config.set(scheduler='synchronous'):
        ##------------------------------------------------------------------------.
        time_start_training = time.time()
        ## Checks arguments 
        device = check_device(device)
        pin_memory = check_pin_memory(pin_memory=pin_memory, num_workers=num_workers, device=device)  
        asyncronous_gpu_transfer = check_asyncronous_gpu_transfer(asyncronous_gpu_transfer=asyncronous_gpu_transfer, device=device) 
        prefetch_in_gpu = check_prefetch_in_gpu(prefetch_in_gpu=prefetch_in_gpu, num_workers=num_workers, device=device) 
        prefetch_factor = check_prefetch_factor(prefetch_factor=prefetch_factor, num_workers=num_workers)
        ar_training_strategy = check_ar_training_strategy(ar_training_strategy)
        ##------------------------------------------------------------------------.  
        # Check ar_scheduler 
        if len(ar_scheduler.ar_weights) > ar_iterations+1:
            raise ValueError("The AR scheduler has {} AR weights, but ar_iterations is specified to be {}".format(len(ar_scheduler.ar_weights), ar_iterations))
        if ar_iterations == 0: 
            if ar_scheduler.method != "constant":
                print("Since 'ar_iterations' is 0, ar_scheduler 'method' is changed to 'constant'.")
                ar_scheduler.method = "constant"
        ##------------------------------------------------------------------------.   
        # Check that autoregressive settings are valid 
        # - input_k and output_k must be numpy arrays hereafter ! 
        print("- Defining AR settings:")
        input_k = check_input_k(input_k=input_k, ar_iterations=ar_iterations)   
        output_k = check_output_k(output_k=output_k)
        check_ar_settings(input_k = input_k,
                          output_k = output_k,
                          forecast_cycle = forecast_cycle,                           
                          ar_iterations = ar_iterations, 
                          stack_most_recent_prediction = stack_most_recent_prediction)    
        ##------------------------------------------------------------------------.  
        # Check training data 
        if training_data_dynamic is None:  
            raise ValueError("'training_data_dynamic' must be provided !")
        ##------------------------------------------------------------------------.  
        ## Check validation data 
        if validation_data_dynamic is not None: 
            if not xr_is_aligned(training_data_dynamic, validation_data_dynamic, exclude="time"):
                 raise ValueError("training_data_dynamic' and 'validation_data_dynamic' does not"
                                  "share same dimensions (order and values)(excluding 'time').")
        if validation_data_bc is not None: 
            if training_data_dynamic is None: 
                raise ValueError("If 'validation_data_bc' is provided, also 'training_data_dynamic' must be specified.")
            if not xr_is_aligned(training_data_bc, validation_data_bc, exclude="time"):
                raise ValueError("training_data_bc' and 'validation_data_bc' does not"
                                  "share same dimensions (order and values)(excluding 'time').")
    
        ##------------------------------------------------------------------------.   
        ## Check early stopping
        if validation_data_dynamic is None:
            if early_stopping is not None: 
                if early_stopping.stopping_metric == "total_validation_loss":
                    print("Validation dataset is not provided."
                           "Stopping metric of early_stopping set to 'total_training_loss'")
                    early_stopping.stopping_metric = "total_training_loss"  
        ##------------------------------------------------------------------------.
        ## Decide wheter to tune num_workers    
        if autotune_num_workers and (num_workers > 0): 
            num_workers_list = list(range(0, num_workers))
        else: 
            num_workers_list = [num_workers]
        ##------------------------------------------------------------------------.
        # Ensure criterion and model are on device 
        model.to(device)
        criterion.to(device)
        ##------------------------------------------------------------------------.
        # Zeros gradients     
        optimizer.zero_grad(set_to_none=True)    
        ##------------------------------------------------------------------------.
        ### Create Datasets 
        t_i = time.time()
        trainingDataset = AutoregressiveDataset(data_dynamic = training_data_dynamic,  
                                                data_bc = training_data_bc,
                                                data_static = data_static,
                                                bc_generator = bc_generator,  
                                                scaler = scaler,
                                                # Custom AR batching function
                                                ar_batch_fun = ar_batch_fun,
                                                training_mode = True,
                                                # Autoregressive settings  
                                                input_k = input_k,
                                                output_k = output_k,
                                                forecast_cycle = forecast_cycle,  
                                                ar_iterations = ar_scheduler.current_ar_iterations,
                                                stack_most_recent_prediction = stack_most_recent_prediction, 
                                                # GPU settings 
                                                device = device)
        if validation_data_dynamic is not None:
            validationDataset = AutoregressiveDataset(data_dynamic = validation_data_dynamic,  
                                                      data_bc = validation_data_bc,
                                                      data_static = data_static,  
                                                      bc_generator = bc_generator,  
                                                      scaler = scaler,
                                                      # Custom AR batching function
                                                      ar_batch_fun = ar_batch_fun,
                                                      training_mode = True,
                                                      # Autoregressive settings  
                                                      input_k = input_k,
                                                      output_k = output_k,
                                                      forecast_cycle = forecast_cycle,                           
                                                      ar_iterations = ar_scheduler.current_ar_iterations,
                                                      stack_most_recent_prediction = stack_most_recent_prediction, 
                                                      # GPU settings 
                                                      device = device)
        else: 
            validationDataset = None
        print('- Creation of AutoregressiveDatasets: {:.0f}s'.format(time.time() - t_i))
        ##------------------------------------------------------------------------.
        ### Time execution         
        # - Time AR training    
        print("- Timing AR training with {} AR iterations:".format(trainingDataset.ar_iterations))
        training_num_workers = tune_num_workers(dataset = trainingDataset,
                                                model = model, 
                                                optimizer = optimizer, 
                                                criterion = criterion, 
                                                num_workers_list = num_workers_list, 
                                                ar_scheduler = ar_scheduler,
                                                ar_training_strategy = ar_training_strategy, 
                                                # DataLoader options
                                                batch_size = training_batch_size, 
                                                shuffle = shuffle,
                                                shuffle_seed = shuffle_seed, # This cause training on same batch n_repetitions times
                                                prefetch_in_gpu = prefetch_in_gpu,
                                                prefetch_factor = prefetch_factor,
                                                pin_memory = pin_memory,
                                                asyncronous_gpu_transfer = asyncronous_gpu_transfer,
                                                # Timing options
                                                training_mode = True, 
                                                n_repetitions = 5,
                                                verbose = True)
        print('  --> Selecting num_workers={} for TrainingDataLoader.'.format(training_num_workers))
    
        # - Time AR validation 
        if validationDataset is not None: 
            print()
            print("- Timing AR validation with {} AR iterations:".format(validationDataset.ar_iterations))
            validation_num_workers = tune_num_workers(dataset = validationDataset,
                                                      model = model, 
                                                      optimizer = optimizer, 
                                                      criterion = criterion, 
                                                      num_workers_list = num_workers_list, 
                                                      ar_scheduler = ar_scheduler,
                                                      ar_training_strategy = ar_training_strategy, 
                                                      # DataLoader options
                                                      batch_size = validation_batch_size, 
                                                      shuffle = shuffle,
                                                      shuffle_seed = shuffle_seed,
                                                      prefetch_in_gpu = prefetch_in_gpu,
                                                      prefetch_factor = prefetch_factor,
                                                      pin_memory = pin_memory,
                                                      asyncronous_gpu_transfer = asyncronous_gpu_transfer,
                                                      # Timing options
                                                      training_mode = False, 
                                                      n_repetitions = 5,
                                                      verbose = True)
            print('  --> Selecting num_workers={} for ValidationDataLoader.'.format(validation_num_workers))
    
        ##------------------------------------------------------------------------.
        ## Create DataLoaders
        # - Prefetch (prefetch_factor*num_workers) batches parallelly into CPU
        # - At each AR iteration, the required data are transferred asynchronously to GPU 
        # - If static data are provided, they are prefetched into the GPU 
        # - Some data are duplicated in CPU memory because of the data overlap between forecast iterations.
        #   However this mainly affect boundary conditions data, because dynamic data
        #   after few AR iterations are the predictions of previous AR iteration.
        t_i = time.time()
        trainingDataLoader = AutoregressiveDataLoader(dataset = trainingDataset,                                                   
                                                      batch_size = training_batch_size,  
                                                      drop_last_batch = drop_last_batch,
                                                      shuffle = shuffle,
                                                      shuffle_seed = shuffle_seed, 
                                                      num_workers = training_num_workers,
                                                      prefetch_factor = prefetch_factor, 
                                                      prefetch_in_gpu = prefetch_in_gpu,  
                                                      pin_memory = pin_memory,
                                                      asyncronous_gpu_transfer = asyncronous_gpu_transfer, 
                                                      device = device)
        if validation_data_dynamic is not None:
            validationDataLoader = AutoregressiveDataLoader(dataset = validationDataset, 
                                                            batch_size = validation_batch_size,  
                                                            drop_last_batch = drop_last_batch,
                                                            shuffle = shuffle,
                                                            shuffle_seed = shuffle_seed, 
                                                            num_workers = validation_num_workers,
                                                            prefetch_in_gpu = prefetch_in_gpu,  
                                                            prefetch_factor = prefetch_factor, 
                                                            pin_memory = pin_memory,
                                                            asyncronous_gpu_transfer = asyncronous_gpu_transfer, 
                                                            device = device)
            validationDataLoader_iter = cylic_iterator(validationDataLoader)
            print('- Creation of AutoregressiveDataLoaders: {:.0f}s'.format(time.time() - t_i))
        else: 
            validationDataset = None
            validationDataLoader_iter = None
        
        ##------------------------------------------------------------------------.
        # Initialize AR_TrainingInfo instance if not provided 
        # - Initialization occurs when a new model training starts
        # - Passing an AR_TrainingInfo instance allows to continue model training from where it stopped !
        #   --> The ar_scheduler of previous training must be provided to ar_Training() !
        if ar_training_info is not None: 
            if not isinstance(ar_training_info, AR_TrainingInfo):
                raise TypeError("If provided, 'ar_training_info' must be an instance of AR_TrainingInfo class.")
                # TODO: Check AR scheduler weights are compatible ! or need numpy conversion
                # ar_scheduler = ar_training_info.ar_scheduler
        else: 
            ar_training_info = AR_TrainingInfo(ar_iterations=ar_iterations,
                                               epochs = epochs,
                                               ar_scheduler = ar_scheduler)  
    
        ##------------------------------------------------------------------------.
        # Get dimension and feature infos
        # TODO: this is only used by the loss, --> future refactoring
        dim_info = trainingDataset.dim_info
        dim_order = trainingDataset.dim_order
        feature_info = trainingDataset.feature_info
        feature_order = trainingDataset.feature_order
        dim_info_dynamic = dim_info['dynamic']
        # feature_dynamic = list(feature_info['dynamic'])
    
        ##------------------------------------------------------------------------.
        # Retrieve custom ar_batch_fun fuction
        ar_batch_fun = trainingDataset.ar_batch_fun  
        ##------------------------------------------------------------------------.
        # Set model layers (i.e. batchnorm) in training mode 
        model.train()
        optimizer.zero_grad(set_to_none=True)  
        ##------------------------------------------------------------------------.
        # Iterate along epochs
        print("")
        print("========================================================================================")
        flag_stop_training = False
        t_i_scoring = time.time()
        for epoch in range(epochs):
            ar_training_info.new_epoch()
            ##--------------------------------------------------------------------.
            # Iterate along training batches 
            trainingDataLoader_iter = iter(trainingDataLoader)
            ##--------------------------------------------------------------------.
            # Compute collection points for SWAG training
            num_batches = len(trainingDataLoader_iter)
            batch_indices = range(num_batches)
            swag_training = swag and swag_model and epoch >= swa_start
            if swag_training:
                freq = int(num_batches/(swag_freq-1))
                collection_indices = list(range(0, num_batches, freq))
            ##--------------------------------------------------------------------.     
            for batch_count in batch_indices:
                ##----------------------------------------------------------------.   
                # Retrieve the training batch
                training_batch_dict = next(trainingDataLoader_iter)
                ##----------------------------------------------------------------.      
                # Perform autoregressive training loop
                # - The number of AR iterations is determined by ar_scheduler.ar_weights 
                # - If ar_weights are all zero after N forecast iteration:
                #   --> Load data just for F forecast iteration 
                #   --> Autoregress model predictions just N times to save computing time
                dict_training_Y_predicted = {}
                dict_training_loss_per_ar_iteration = {}
                for ar_iteration in range(ar_scheduler.current_ar_iterations+1):
                    # Retrieve X and Y for current AR iteration
                    # - ar_batch_fun() function stack together the required data from the previous AR iteration
                    torch_X, torch_Y = ar_batch_fun(ar_iteration = ar_iteration, 
                                                    batch_dict = training_batch_dict, 
                                                    dict_Y_predicted = dict_training_Y_predicted,
                                                    asyncronous_gpu_transfer = asyncronous_gpu_transfer,
                                                    device = device)                                         
                    ##-------------------------------------------------------------.                               
                    # # Print memory usage dataloader
                    # if device.type != 'cpu':
                    #     # torch.cuda.synchronize()
                    #     print("{}: {:.2f} MB".format(ar_iteration, torch.cuda.memory_allocated()/1000/1000)) 
    
                    ##-------------------------------------------------------------.
                    # Forward pass and store output for stacking into next AR iterations
                    dict_training_Y_predicted[ar_iteration] = model(torch_X)
    
                    ##-------------------------------------------------------------.
                    # Compute loss for current forecast iteration 
                    # - The criterion expects [data_points, nodes, features]
                    # - Collapse all other dimensions to a 'data_points' dimension  
                    Y_pred, Y_obs = reshape_tensors_4_loss(Y_pred = dict_training_Y_predicted[ar_iteration],
                                                           Y_obs = torch_Y,
                                                           dim_info_dynamic = dim_info_dynamic)
                    dict_training_loss_per_ar_iteration[ar_iteration] = criterion(Y_obs, Y_pred)
    
                    ##-------------------------------------------------------------.
                    # If ar_training_strategy is "AR", perform backward pass at each AR iteration 
                    if ar_training_strategy == "AR":
                        # - Detach gradient of Y_pred (to avoid RNN-style optimization)
                        dict_training_Y_predicted[ar_iteration] = dict_training_Y_predicted[ar_iteration].detach() # TODO: should not be detached after backward?
                        # - AR weight the loss (aka weight sum the gradients ...)
                        current_ar_loss = dict_training_loss_per_ar_iteration[ar_iteration]
                        current_ar_loss = current_ar_loss*ar_scheduler.ar_weights[ar_iteration]
                        # - Backpropagate to compute gradients (the derivative of the loss w.r.t. the parameters)
                        current_ar_loss.backward()
                        del current_ar_loss
                     
                    ##------------------------------------------------------------.
                    # Remove unnecessary stored Y predictions 
                    remove_unused_Y(ar_iteration = ar_iteration, 
                                    dict_Y_predicted = dict_training_Y_predicted,
                                    dict_Y_to_remove = training_batch_dict['dict_Y_to_remove'])
                    
                    del Y_pred, Y_obs, torch_X, torch_Y
                    if ar_iteration == ar_scheduler.current_ar_iterations:
                        del dict_training_Y_predicted
    
                    ##------------------------------------------------------------.
                    # # Print memory usage dataloader + model 
                    # if device.type != 'cpu':
                    #     torch.cuda.synchronize()
                    #     print("{}: {:.2f} MB".format(ar_iteration, torch.cuda.memory_allocated()/1000/1000)) 
    
                ##----------------------------------------------------------------.
                # - Compute total (AR weighted) loss 
                for i, (ar_iteration, loss) in enumerate(dict_training_loss_per_ar_iteration.items()):
                    if i == 0:
                        training_total_loss = ar_scheduler.ar_weights[ar_iteration] * loss 
                    else: 
                        training_total_loss += ar_scheduler.ar_weights[ar_iteration] * loss
                ##----------------------------------------------------------------.       
                # - If ar_training_strategy is RNN, perform backward pass after all AR iterations            
                if ar_training_strategy == "RNN":
                    # - Perform backward pass using training_total_loss (after all AR iterations)
                    training_total_loss.backward()
    
                ##----------------------------------------------------------------.     
                # - Update the network weights 
                optimizer.step()  
    
                ##----------------------------------------------------------------.
                # Zeros all the gradients for the next batch training 
                # - By default gradients are accumulated in buffers (and not overwritten)
                optimizer.zero_grad(set_to_none=True)   
                
                ##----------------------------------------------------------------. 
                # - Update training statistics                                                                        # TODO: This require CPU-GPU synchronization
                if ar_training_info.iteration_from_last_scoring == scoring_interval:
                    ar_training_info.update_training_stats(total_loss = training_total_loss,
                                                           dict_loss_per_ar_iteration = dict_training_loss_per_ar_iteration, 
                                                           ar_scheduler = ar_scheduler, 
                                                           lr_scheduler = lr_scheduler)
                ##----------------------------------------------------------------.
                # Printing infos (if no validation data available) 
                if validationDataset is None:
                    if batch_count % scoring_interval == 0:
                        print("Epoch: {} | Batch: {}/{} | AR: {} | Loss: {} | "
                                "ES: {}/{}".format(epoch, batch_count, num_batches, 
                                                   ar_iteration,
                                                   round(dict_training_loss_per_ar_iteration[ar_iteration].item(),5), # TODO: This require CPU-GPU synchronization
                                                   early_stopping.counter, early_stopping.patience)
                             )
                    ##-------------------------------------------------------------.
                    # The following code can be used to debug training if loss diverge to nan 
                    if dict_training_loss_per_ar_iteration[0].item() > 10000:                                         # TODO: This require CPU-GPU synchronization
                        ar_training_info_fpath = os.path.join(os.path.dirname(model_fpath), "AR_TrainingInfo.pickle")
                        with open(ar_training_info_fpath, 'wb') as handle:
                            pickle.dump(ar_training_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        raise ValueError("The training has diverged. The training info can be recovered using: \n"
                                        "with open({!r}, 'rb') as handle: \n" 
                                        "    ar_training_info = pickle.load(handle)".format(ar_training_info_fpath))
                                    
                ##-----------------------------------------------------------------.
                # TODO: SWAG Description
                if swag_training:
                    if batch_count in collection_indices:
                        swag_model.collect_model(model)
                        
                ##-----------------------------------------------------------------. 
                ### Run validation 
                if validationDataset is not None:
                    if ar_training_info.iteration_from_last_scoring == scoring_interval:
                        # Set model layers (i.e. batchnorm) in evaluation mode 
                        model.eval() 
                        
                        # Retrieve batch for validation
                        validation_batch_dict = next(validationDataLoader_iter)
                        
                        # Initialize 
                        dict_validation_loss_per_ar_iteration = {}
                        dict_validation_Y_predicted = {}
                        
                        #----------------------------------------------------------.
                        # SWAG: collect, sample and update batch norm statistics
                        if swag_training:
                            swag_model.collect_model(model)
                            with torch.no_grad():
                                swag_model.sample(0.0)
    
                            bn_update_with_loader(swag_model, trainingDataLoader,
                                                  ar_iterations = ar_scheduler.current_ar_iterations,
                                                  asyncronous_gpu_transfer = asyncronous_gpu_transfer,
                                                  device = device)
                                                 
                        #----------------------------------------------------------.
                        # Disable gradient calculations 
                        # - And do not update network weights  
                        with torch.set_grad_enabled(False): 
                            # Autoregressive loop 
                            for ar_iteration in range(ar_scheduler.current_ar_iterations+1):
                                # Retrieve X and Y for current AR iteration
                                torch_X, torch_Y = ar_batch_fun(ar_iteration = ar_iteration, 
                                                                batch_dict = validation_batch_dict, 
                                                                dict_Y_predicted = dict_validation_Y_predicted,
                                                                asyncronous_gpu_transfer = asyncronous_gpu_transfer,
                                                                device = device)
                            
                                ##------------------------------------------------.
                                # Forward pass and store output for stacking into next AR iterations
                                dict_validation_Y_predicted[ar_iteration] = swag_model(torch_X) if swag_training else model(torch_X)
                    
                                ##------------------------------------------------.
                                # Compute loss for current forecast iteration 
                                # - The criterion expects [data_points, nodes, features] 
                                Y_pred, Y_obs = reshape_tensors_4_loss(Y_pred = dict_validation_Y_predicted[ar_iteration],
                                                                       Y_obs = torch_Y,
                                                                       dim_info_dynamic = dim_info_dynamic)
                                dict_validation_loss_per_ar_iteration[ar_iteration] = criterion(Y_obs, Y_pred)
                                
                                ##------------------------------------------------.
                                # Remove unnecessary stored Y predictions 
                                remove_unused_Y(ar_iteration = ar_iteration, 
                                                dict_Y_predicted = dict_validation_Y_predicted,
                                                dict_Y_to_remove = validation_batch_dict['dict_Y_to_remove'])
                                del Y_pred, Y_obs, torch_X, torch_Y
                                if ar_iteration == ar_scheduler.current_ar_iterations:
                                    del dict_validation_Y_predicted
    
                        ##--------------------------------------------------------.    
                        ### Compute total (AR weighted) loss 
                        for i, (ar_iteration, loss) in enumerate(dict_validation_loss_per_ar_iteration.items()):
                            if i == 0:
                                validation_total_loss = ar_scheduler.ar_weights[ar_iteration] * loss 
                            else: 
                                validation_total_loss += ar_scheduler.ar_weights[ar_iteration] * loss
                        
                        ##--------------------------------------------------------. 
                        ### Update validation info                                                                                        # TODO: This require CPU-GPU synchronization
                        ar_training_info.update_validation_stats(total_loss = validation_total_loss,
                                                                 dict_loss_per_ar_iteration = dict_validation_loss_per_ar_iteration)
                        
                        ##--------------------------------------------------------.
                        ### Reset model to training mode
                        model.train() 
                        
                        ##--------------------------------------------------------.
                        ### Print scoring 
                        t_f_scoring = round(time.time() - t_i_scoring)
                        print("Epoch: {} | Batch: {}/{} | AR: {} | Loss: {} | "
                              "ES: {}/{} | Elapsed time: {}s".format(epoch, batch_count, num_batches, 
                                                                     ar_iteration,
                                                                     round(dict_validation_loss_per_ar_iteration[ar_iteration].item(),5), # TODO: This require CPU-GPU synchronization
                                                                     early_stopping.counter, early_stopping.patience, 
                                                                     t_f_scoring)
                              )
                        t_i_scoring = time.time()
                        ##---------------------------------------------------------.
                        # The following code can be used to debug training if loss diverge to nan 
                        if dict_validation_loss_per_ar_iteration[0].item() > 10000:  # TODO: This require CPU-GPU synchronization
                            ar_training_info_fpath = os.path.join(os.path.dirname(model_fpath), "AR_TrainingInfo.pickle")
                            with open(ar_training_info_fpath, 'wb') as handle:
                                pickle.dump(ar_training_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
                            raise ValueError("The training has diverged. The training info can be recovered using: \n"
                                            "with open({!r}, 'rb') as handle: \n" 
                                            "    ar_training_info = pickle.load(handle)".format(ar_training_info_fpath))
                        ##--------------------------------------------------------.
                
                ##----------------------------------------------------------------. 
                # - Update learning rate 
                if lr_scheduler is not None:
                    lr_scheduler.step() 
                    
                ##----------------------------------------------------------------. 
                # - Update the AR weights 
                ar_scheduler.step()
    
                ##----------------------------------------------------------------. 
                # - Evaluate stopping metrics and update AR scheduler if the loss has plateau
                if ar_training_info.iteration_from_last_scoring == scoring_interval:
                    # Reset counter for scoring 
                    ar_training_info.reset_counter()  
                    ##-------------------------------------------------------------.
                    # If the model has not improved (based on early stopping settings)
                    # - If current_ar_iterations < ar_iterations --> Update AR scheduler
                    # - If current_ar_iterations = ar_iterations --> Stop training 
                    if early_stopping is not None and early_stopping(ar_training_info):
                        # - If current_ar_iterations < ar_iterations --> Update AR scheduler
                        if ar_scheduler.current_ar_iterations < ar_iterations: 
                            ##----------------------------------------------------.
                            # Update the AR scheduler
                            ar_scheduler.update()
                            # Reset iteration counter from last AR weight update
                            ar_training_info.reset_iteration_from_last_ar_update()
                            # Reset early stopping 
                            early_stopping.reset()
                            # Print info
                            current_ar_training_info = "(epoch: {}, iteration: {}, total_iteration: {})".format(ar_training_info.epoch, 
                                                                                                                ar_training_info.epoch_iteration,
                                                                                                                ar_training_info.iteration)
                            print("") 
                            print("========================================================================================")
                            print("- Updating training to {} AR iterations {}.".format(ar_scheduler.current_ar_iterations, current_ar_training_info))
                            ##----------------------------------------------------.           
                            # Update Datasets (to prefetch the correct amount of data)
                            # - Training
                            del trainingDataLoader, trainingDataLoader_iter         # to avoid deadlocks
                            trainingDataset.update_ar_iterations(ar_scheduler.current_ar_iterations)
                            # - Validation
                            if validationDataset is not None: 
                                del validationDataLoader, validationDataLoader_iter # to avoid deadlocks
                                validationDataset.update_ar_iterations(ar_scheduler.current_ar_iterations)
                            ##----------------------------------------------------.                              
                            ## Time execution         
                            # - Time AR training  
                            print("")  
                            print("- Timing AR training with {} AR iterations:".format(trainingDataset.ar_iterations))
                            training_num_workers = tune_num_workers(dataset = trainingDataset,
                                                                    model = model, 
                                                                    optimizer = optimizer, 
                                                                    criterion = criterion, 
                                                                    num_workers_list = num_workers_list, 
                                                                    ar_scheduler = ar_scheduler,
                                                                    ar_training_strategy = ar_training_strategy, 
                                                                    # DataLoader options
                                                                    batch_size = training_batch_size, 
                                                                    shuffle = shuffle,
                                                                    shuffle_seed = shuffle_seed, # This cause training on same batch n_repetitions times
                                                                    prefetch_in_gpu = prefetch_in_gpu,
                                                                    prefetch_factor = prefetch_factor,
                                                                    pin_memory = pin_memory,
                                                                    asyncronous_gpu_transfer = asyncronous_gpu_transfer,
                                                                    # Timing options
                                                                    training_mode = True, 
                                                                    n_repetitions = 5,
                                                                    verbose = True)
                            print('--> Selecting num_workers={} for TrainingDataLoader.'.format(training_num_workers))
                            # - Time AR validation 
                            if validationDataset is not None: 
                                print("")
                                print("- Timing AR validation with {} AR iterations:".format(validationDataset.ar_iterations))
                                validation_num_workers = tune_num_workers(dataset = validationDataset,
                                                                          model = model, 
                                                                          optimizer = optimizer, 
                                                                          criterion = criterion, 
                                                                          num_workers_list = num_workers_list,
                                                                          ar_scheduler = ar_scheduler,
                                                                          ar_training_strategy = ar_training_strategy,  
                                                                          # DataLoader options
                                                                          batch_size = validation_batch_size, 
                                                                          shuffle = shuffle,
                                                                          shuffle_seed = shuffle_seed, 
                                                                          prefetch_in_gpu = prefetch_in_gpu,
                                                                          prefetch_factor = prefetch_factor,
                                                                          pin_memory = pin_memory,
                                                                          asyncronous_gpu_transfer = asyncronous_gpu_transfer,
                                                                          # Timing options
                                                                          training_mode = False, 
                                                                          n_repetitions = 5,
                                                                          verbose = True)
                                print('--> Selecting num_workers={} for ValidationDataLoader.'.format(validation_num_workers))
                            ##----------------------------------------------------------------.
                            # Update DataLoaders (to prefetch the correct amount of data)
                            shuffle_seed += 1
                            trainingDataLoader = AutoregressiveDataLoader(dataset = trainingDataset,                                                   
                                                                          batch_size = training_batch_size,  
                                                                          drop_last_batch = drop_last_batch,
                                                                          shuffle = shuffle,
                                                                          shuffle_seed = shuffle_seed, 
                                                                          num_workers = training_num_workers,
                                                                          prefetch_factor = prefetch_factor, 
                                                                          prefetch_in_gpu = prefetch_in_gpu,  
                                                                          pin_memory = pin_memory,
                                                                          asyncronous_gpu_transfer = asyncronous_gpu_transfer, 
                                                                          device = device)
                            trainingDataLoader_iter = cylic_iterator(trainingDataLoader)
                            if validationDataset is not None: 
                                validationDataset.update_ar_iterations(ar_scheduler.current_ar_iterations)
                                validationDataLoader = AutoregressiveDataLoader(dataset = validationDataset, 
                                                                                batch_size = validation_batch_size,  
                                                                                drop_last_batch = drop_last_batch,
                                                                                shuffle = shuffle,
                                                                                shuffle_seed = shuffle_seed, 
                                                                                num_workers = validation_num_workers,
                                                                                prefetch_in_gpu = prefetch_in_gpu,  
                                                                                prefetch_factor = prefetch_factor, 
                                                                                pin_memory = pin_memory,
                                                                                asyncronous_gpu_transfer = asyncronous_gpu_transfer,
                                                                                device = device)
                                validationDataLoader_iter = cylic_iterator(validationDataLoader)
                                
                        ##--------------------------------------------------------.     
                        # - If current_ar_iterations = ar_iterations --> Stop training 
                        else: 
                            # Stop training 
                            flag_stop_training = True
                            break
                        
                ##----------------------------------------------------------------.     
                # - Update iteration count 
                ar_training_info.step()   
                            
            ##--------------------------------------------------------------------. 
            ### Print epoch training statistics  
            ar_training_info.print_epoch_info()
            
            if flag_stop_training:
                break 
            ##--------------------------------------------------------------------. 
            # Option to save the model each epoch
            if save_model_each_epoch:
                model_weights = swag_model.state_dict() if swag_training else model.state_dict()
                torch.save(model_weights, model_fpath[:-3] + '_epoch_{}'.format(epoch) + '.h5')
          
        ##-------------------------------------------------------------------------.
        ### Save final model
        print(" ")
        print("========================================================================================")
        print("- Training ended !")
        print("- Total elapsed time: {:.2f} hours.".format((time.time()-time_start_training)/60/60))
        print("- Saving model to {}".format(model_fpath))
        model_weights = swag_model.state_dict() if (swag and swag_model) else model.state_dict() 
        torch.save(model_weights, f=model_fpath)   
        
        ##-------------------------------------------------------------------------.
        ### Save AR TrainingInfo  
        print("========================================================================================")
        print("- Saving training information")
        with open(os.path.join(os.path.dirname(model_fpath), "AR_TrainingInfo.pickle"), 'wb') as handle:
            pickle.dump(ar_training_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        ##-------------------------------------------------------------------------.
        ## Remove Dataset and DataLoaders to avoid deadlocks 
        del validationDataset
        del validationDataLoader
        del validationDataLoader_iter
        del trainingDataset
        del trainingDataLoader
        del trainingDataLoader_iter
        ##------------------------------------------------------------------------.
        # Return training info object 
        return ar_training_info
    
    #-----------------------------------------------------------------------------.
