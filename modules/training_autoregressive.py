#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 18:42:12 2021

@author: ghiggi
"""
import torch
import time
import numpy as np
from tabulate import tabulate 

from modules.dataloader_autoregressive import AutoregressiveDataset
from modules.dataloader_autoregressive import AutoregressiveDataLoader
from modules.dataloader_autoregressive import get_AR_batch
from modules.dataloader_autoregressive import remove_unused_Y
from modules.dataloader_autoregressive import cylic_iterator
from modules.utils_autoregressive import check_AR_settings
from modules.utils_autoregressive import check_input_k
from modules.utils_autoregressive import check_output_k 
from modules.utils_io import check_AR_DataArrays 
from modules.utils_training import AR_TrainingInfo
from modules.utils_torch import check_torch_device
##----------------------------------------------------------------------------.
# TODOs
# - ONNX for saving model weights 
# - Record the loss per variable 
# - Add something related to memory consumption --> torch.cuda.memory_summary() 
##----------------------------------------------------------------------------. 
###################
### Loss utils ####
###################
def reshape_tensors_4_loss(Y_pred, Y_obs, dim_names):
    """Reshape tensors for loss computation."""
    vars_to_flatten = np.array(dim_names)[np.isin(dim_names,['node','feature'], invert=True)].tolist()
    Y_pred = Y_pred.rename(*dim_names).align_to(...,'node','feature').flatten(vars_to_flatten, 'data_points').rename(None)
    Y_obs = Y_obs.rename(*dim_names).align_to(...,'node','feature').flatten(vars_to_flatten, 'data_points').rename(None)
    return Y_pred, Y_obs

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

#-----------------------------------------------------------------------------.
# ############################
#### Autotune num_workers ####
# ############################       
def timing_AR_Training(dataset,
                       model, 
                       optimizer, 
                       criterion, 
                       # DataLoader options
                       batch_size = 32, 
                       num_workers = 0, 
                       prefetch_in_GPU = False,
                       prefetch_factor = 2,
                       pin_memory = False,
                       asyncronous_GPU_transfer = True,
                       # Timing options 
                       training_mode = True,
                       n_repetitions = 10,
                       verbose = True):
    """
    Time AR training.

    Parameters
    ----------
    dataset : AutoregressiveDataLoader
        AutoregressiveDataLoader
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
    prefetch_factor: int, optional 
        Number of sample loaded in advance by each worker.
        The default is 2.
    prefetch_in_GPU: bool, optional 
        Whether to prefetch 'prefetch_factor'*'num_workers' batches of data into GPU instead of CPU.
        By default it prech 'prefetch_factor'*'num_workers' batches of data into CPU (when False)
        The default is False.
    pin_memory : bool, optional
        When True, it prefetch the batch data into the pinned memory.  
        pin_memory=True enables (asynchronous) fast data transfer to CUDA-enabled GPUs.
        Useful only if training on GPU.
        The default is False.
    asyncronous_GPU_transfer: bool, optional 
        Only used if 'prefetch_in_GPU' = True. 
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
    if not isinstance(num_workers, int):
        raise TypeError("'num_workers' must be a integer larger than 0.")
    if num_workers < 0: 
        raise ValueError("'num_workers' must be a integer larger than 0.")
    if not isinstance(training_mode, bool):
        raise TypeError("'training_mode' must be either True or False.")
    ##------------------------------------------------------------------------.    
    # Retrieve informations 
    AR_iterations = dataset.AR_iterations
    device = dataset.device                      
    dict_Y_to_remove = dataset.dict_Y_to_remove 
    # Retrieve function to get time 
    get_time = get_time_function(device)
    ##------------------------------------------------------------------------.
    # Get dimension infos
    dim_info = dataset.dim_info
    dim_names = tuple(dim_info.keys())
    ##------------------------------------------------------------------------.
    # Initialize model 
    if training_mode:
        model.train()
        torch.set_grad_enabled(True)
    else: 
        model.eval()  
        torch.set_grad_enabled(False)
    ##------------------------------------------------------------------------.
    # Initialize list 
    Dataloader_timing = []  
    AR_batch_timing = []  
    AR_data_removal_timing = []  
    AR_forward_timing = []
    AR_loss_timing = []
    Backprop_timing = [] 
    Total_timing = []
    ##------------------------------------------------------------------------.
    # Initialize DataLoader 
    trainingDataLoader = AutoregressiveDataLoader(dataset = dataset,                                                   
                                                  batch_size = batch_size,  
                                                  drop_last_batch = True,
                                                  random_shuffle = True,
                                                  num_workers = num_workers,
                                                  prefetch_factor = prefetch_factor, 
                                                  prefetch_in_GPU = prefetch_in_GPU,  
                                                  pin_memory = pin_memory,
                                                  asyncronous_GPU_transfer = asyncronous_GPU_transfer, 
                                                  device = device)
    trainingDataLoader_iter = iter(trainingDataLoader)
    # Repeat training n_repetitions
    for count in range(n_repetitions):  
        t_start = get_time()
        # Retrieve batch
        t_i = get_time()
        training_batch_dict = next(trainingDataLoader_iter)
        Dataloader_timing.append(get_time() - t_i)
        # Perform AR iterations 
        dict_training_Y_predicted = {}
        dict_training_loss_per_AR_iteration = {}
        ##-----------------------------------------------------------------.
        # Initialize stuff for AR loop timing 
        tmp_AR_data_removal_timing = 0
        tmp_AR_batch_timing = 0 
        tmp_AR_forward_timing = 0 
        tmp_AR_loss_timing = 0
        for i in range(AR_iterations+1):
            # Retrieve X and Y for current AR iteration   
            t_i = get_time()
            torch_X, torch_Y = get_AR_batch(AR_iteration = i, 
                                            batch_dict = training_batch_dict, 
                                            dict_Y_predicted = dict_training_Y_predicted,
                                            device = device, 
                                            asyncronous_GPU_transfer = asyncronous_GPU_transfer)
            tmp_AR_batch_timing = tmp_AR_batch_timing + (get_time() - t_i)
            ##------------------------------------------------------------.
            # Forward pass and store output for stacking into next AR iterations
            t_i = get_time()
            dict_training_Y_predicted[i] = model(torch_X)
            tmp_AR_forward_timing = tmp_AR_forward_timing + (get_time() - t_i) 
            ##------------------------------------------------------------.
            # Compute loss for current forecast iteration 
            # - The criterion expects [data_points, nodes, features]
            # - Collapse all other dimensions to a 'data_points' dimension  
            t_i = get_time()   
            Y_pred, Y_obs = reshape_tensors_4_loss(Y_pred = dict_training_Y_predicted[i],
                                                   Y_obs = torch_Y,
                                                   dim_names = dim_names)
            dict_training_loss_per_AR_iteration[i] = criterion(Y_obs, Y_pred)
            tmp_AR_loss_timing = tmp_AR_loss_timing + (get_time() - t_i)
            ##------------------------------------------------------------.
            # Remove unnecessary stored Y predictions 
            t_i = get_time()
            remove_unused_Y(AR_iteration = i, 
                            dict_Y_predicted = dict_training_Y_predicted,
                            dict_Y_to_remove = dict_Y_to_remove)
            del Y_pred, Y_obs, torch_X, torch_Y
            if i == AR_iterations:
                del dict_training_Y_predicted
            tmp_AR_data_removal_timing = tmp_AR_data_removal_timing + (get_time()- t_i)
            
        ##--------------------------------------------------------------------.  
        # Summarize timing 
        AR_batch_timing.append(tmp_AR_batch_timing)
        AR_data_removal_timing.append(tmp_AR_data_removal_timing)
        AR_forward_timing.append(tmp_AR_forward_timing)
        
        ##--------------------------------------------------------------------.    
        ### Compute total (AR weighted) loss 
        t_i = get_time()
        for i, (AR_iteration, loss) in enumerate(dict_training_loss_per_AR_iteration.items()):
            if i == 0:
                training_total_loss = loss 
            else: 
                training_total_loss += loss
        tmp_AR_loss_timing = tmp_AR_loss_timing + (get_time() - t_i)
        AR_loss_timing.append(tmp_AR_loss_timing)    
        
        ##--------------------------------------------------------------------.       
        ### Backprogate the gradients and update the network weights 
        if training_mode: 
            t_i = get_time()
            # Backward pass 
            # - Update the network weights 
            # Zeros all the gradients
            # - By default gradients are accumulated in buffers (and not overwritten)
            optimizer.zero_grad()   
            training_total_loss.backward()
            optimizer.step()  
            # - Time backward pass 
            Backprop_timing.append(get_time() - t_i)
        else: 
            Backprop_timing.append(0)
        ##--------------------------------------------------------------------.
        # - Total time elapsed
        Total_timing.append(get_time() - t_start)

    ##------------------------------------------------------------------------.
    # Create timing info dictionary 
    timing_info = {'Run': list(range(n_repetitions)), 
                   'Total': Total_timing, 
                   'Dataloader': Dataloader_timing,
                   'AR Batch': AR_batch_timing,
                   'Delete': AR_data_removal_timing,
                   'Forward': AR_forward_timing,
                   'Loss': AR_loss_timing,
                   'Backward': Backprop_timing}
    ##------------------------------------------------------------------------. 
    # Create timing table 
    if verbose:
        table = []
        headers = ['Run', 'Total', 'Dataloader','AR Batch', 'Delete', 'Forward', 'Loss', 'Backward']
        for count in range(n_repetitions):
            table.append([count,    
                         round(Total_timing[count], 4),
                         round(Dataloader_timing[count], 4),
                         round(AR_batch_timing[count], 4),
                         round(AR_data_removal_timing[count], 4),
                         round(AR_forward_timing[count], 4),
                         round(AR_loss_timing[count], 4),
                         round(Backprop_timing[count], 4)
                          ])
        print(tabulate(table, headers=headers))   
    
    ##------------------------------------------------------------------------.    
    # Reactivate gradient computations 
    torch.set_grad_enabled(True)
    ##------------------------------------------------------------------------.               
    return timing_info   

def tune_num_workers(dataset,
                     model, 
                     optimizer, 
                     criterion, 
                     num_workers_list, 
                     # DataLoader options
                     batch_size = 32, 
                     prefetch_in_GPU = False,
                     prefetch_factor = 2,
                     pin_memory = False,
                     asyncronous_GPU_transfer = True,
                     # Timing options
                     training_mode = True, 
                     n_repetitions = 10,
                     verbose = True):
    """
    Search for the best value of 'num_workers'.

    Parameters
    ----------
    dataset : AutoregressiveDataLoader
        AutoregressiveDataLoader
    model : pytorch model
        pytorch model.
    optimizer : pytorch optimizer
        pytorch optimizer.
    criterion : pytorch criterion
        pytorch criterion
    num_workers_list : list
        A list of num_workers to time.
    batch_size : int, optional
        Number of samples within a batch. The default is 32.
    prefetch_factor: int, optional 
        Number of sample loaded in advance by each worker.
        The default is 2.
    prefetch_in_GPU: bool, optional 
        Whether to prefetch 'prefetch_factor'*'num_workers' batches of data into GPU instead of CPU.
        By default it prech 'prefetch_factor'*'num_workers' batches of data into CPU (when False)
        The default is False.
    pin_memory : bool, optional
        When True, it prefetch the batch data into the pinned memory.  
        pin_memory=True enables (asynchronous) fast data transfer to CUDA-enabled GPUs.
        Useful only if training on GPU.
        The default is False.
    asyncronous_GPU_transfer: bool, optional 
        Only used if 'prefetch_in_GPU' = True. 
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
    optimal_num_workers : int
        Optimal num_workers to use for efficient data loading.
    """
    ##------------------------------------------------------------------------.
    # Checks arguments 
    if isinstance(num_workers_list, int):
        num_workers_list = [num_workers_list]
    ##------------------------------------------------------------------------.
    # Initialize dictionary 
    Dataloader_timing = {i: [] for i in num_workers_list }
    AR_batch_timing = {i: [] for i in num_workers_list }
    AR_data_removal_timing = {i: [] for i in num_workers_list }
    AR_forward_timing = {i: [] for i in num_workers_list }
    AR_loss_timing = {i: [] for i in num_workers_list }
    Backprop_timing = {i: [] for i in num_workers_list }
    Total_timing = {i: [] for i in num_workers_list }
    ##------------------------------------------------------------------------.
    # Time AR training for specified num_workers in num_workers_list
    for num_workers in num_workers_list:  
        timing_info = timing_AR_Training(dataset = dataset, 
                                         model = model, 
                                         optimizer = optimizer,
                                         criterion = criterion,
                                         # DataLoader options
                                         batch_size = batch_size, 
                                         num_workers = num_workers, 
                                         prefetch_in_GPU = prefetch_in_GPU,
                                         prefetch_factor = prefetch_factor,
                                         pin_memory = pin_memory,
                                         asyncronous_GPU_transfer = asyncronous_GPU_transfer,
                                         # Timing options 
                                         training_mode = training_mode, 
                                         n_repetitions = n_repetitions,
                                         verbose = False) 
        Dataloader_timing[num_workers] = timing_info['Dataloader']
        AR_batch_timing[num_workers] = timing_info['AR Batch']
        AR_data_removal_timing[num_workers] = timing_info['Delete']
        AR_forward_timing[num_workers] = timing_info['Forward']
        AR_loss_timing[num_workers] = timing_info['Loss']
        Backprop_timing[num_workers] = timing_info['Backward']
        Total_timing[num_workers] = timing_info['Total']

    ##------------------------------------------------------------------------. 
    ### Summarize timing results      
    headers = ['N. workers', 'Total', 'Dataloader','AR Batch', 'Delete', 'Forward', 'Loss', 'Backward']
    table = []
    dtloader = []
    for num_workers in num_workers_list:
        dtloader.append(np.median(Dataloader_timing[num_workers]).round(4))
        table.append([num_workers,    
                      np.median(Total_timing[num_workers]).round(4),
                      np.median(Dataloader_timing[num_workers]).round(4),
                      np.median(AR_batch_timing[num_workers]).round(4),
                      np.median(AR_data_removal_timing[num_workers]).round(4),
                      np.median(AR_forward_timing[num_workers]).round(4),
                      np.median(AR_loss_timing[num_workers]).round(4),
                      np.median(Backprop_timing[num_workers]).round(4)])
    
    ##------------------------------------------------------------------------.
    # Print timing results  
    if verbose:
        print(tabulate(table, headers=headers)) 
    ##------------------------------------------------------------------------.
    # Select best num_workers
    optimal_num_workers = num_workers_list[np.argmin(dtloader)]
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
                           AR_scheduler,
                           early_stopping,
                           optimizer, 
                           # Data
                           da_training_dynamic,
                           da_validation_dynamic = None,
                           da_static = None,              
                           da_training_bc = None,         
                           da_validation_bc = None, 
                           scaler = None,
                           # Dataloader options
                           prefetch_in_GPU = False,
                           prefetch_factor = 2,
                           drop_last_batch = True,
                           random_shuffle = True, 
                           num_workers = 0, 
                           autotune_num_workers = False, 
                           pin_memory = False,
                           asyncronous_GPU_transfer = True,
                           # Autoregressive settings  
                           input_k = [-3,-2,-1], 
                           output_k = [0],
                           forecast_cycle = 1,                           
                           AR_iterations = 2, 
                           stack_most_recent_prediction = True,
                           # Training settings 
                           LR_scheduler = None, 
                           training_batch_size = 128,
                           validation_batch_size = 128, 
                           epochs = 10, 
                           numeric_precision = "float64",
                           scoring_interval = 10, 
                           save_model_each_epoch = False,
                           # GPU settings 
                           device = 'cpu'):
    """AutoregressiveTraining."""
    ##------------------------------------------------------------------------.
    ## Checks arguments 
    device = check_torch_device(device)
    if device.type == 'cpu':
        if pin_memory:
            print("- GPU is not available. 'pin_memory' set to False.")
            pin_memory = False
        if prefetch_in_GPU: 
            print("- GPU is not available. 'prefetch_in_GPU' set to False.")
            prefetch_in_GPU = False
        if asyncronous_GPU_transfer: 
            print("- GPU is not available. 'asyncronous_GPU_transfer' set to False.")
            asyncronous_GPU_transfer = False
    ##------------------------------------------------------------------------.   
    # Check that autoregressive settings are valid 
    # - input_k and output_k must be numpy arrays hereafter ! 
    print("- Defining AR settings:")
    input_k = check_input_k(input_k=input_k, AR_iterations=AR_iterations)   
    output_k = check_output_k(output_k=output_k)
    check_AR_settings(input_k = input_k,
                      output_k = output_k,
                      forecast_cycle = forecast_cycle,                           
                      AR_iterations = AR_iterations, 
                      stack_most_recent_prediction = stack_most_recent_prediction)    
    ##------------------------------------------------------------------------.
    # Check that DataArrays are valid 
    check_AR_DataArrays(da_training_dynamic = da_training_dynamic,
                        da_validation_dynamic = da_validation_dynamic, 
                        da_training_bc = da_training_bc,
                        da_validation_bc = da_validation_bc, 
                        da_static = da_static)
    
    ##------------------------------------------------------------------------.
    ## Decide wheter to tune num_workers    
    if autotune_num_workers and (num_workers > 0): 
        num_workers_list = list(range(0, num_workers))
    else: 
        num_workers_list = [num_workers]
        
    ##------------------------------------------------------------------------.
    # Zeros gradients     
    optimizer.zero_grad()    
    ##------------------------------------------------------------------------.
    ### Create Datasets 
    t_i = time.time()
    trainingDataset = AutoregressiveDataset(da_dynamic = da_training_dynamic,  
                                            da_bc = da_training_bc,
                                            da_static = da_static,
                                            scaler = scaler,
                                            # Autoregressive settings  
                                            input_k = input_k,
                                            output_k = output_k,
                                            forecast_cycle = forecast_cycle,  
                                            AR_iterations = AR_scheduler.current_AR_iterations,
                                            stack_most_recent_prediction = stack_most_recent_prediction, 
                                            # GPU settings 
                                            device = device,
                                            # Precision settings
                                            numeric_precision = numeric_precision)
    if da_validation_dynamic is not None:
        validationDataset = AutoregressiveDataset(da_dynamic = da_validation_dynamic,  
                                                  da_bc = da_validation_bc,
                                                  da_static = da_static,   
                                                  scaler = scaler,
                                                  # Autoregressive settings  
                                                  input_k = input_k,
                                                  output_k = output_k,
                                                  forecast_cycle = forecast_cycle,                           
                                                  AR_iterations = AR_scheduler.current_AR_iterations,
                                                  stack_most_recent_prediction = stack_most_recent_prediction, 
                                                  # GPU settings 
                                                  device = device,
                                                  # Precision settings
                                                  numeric_precision = numeric_precision)
    else: 
        validationDataset = None
    print('- Creation of AutoregressiveDatasets: {:.0f}s'.format(time.time() - t_i))
    ##------------------------------------------------------------------------.
    ### Time execution         
    # - Time AR training    
    print("- Timing AR training with {} AR iterations:".format(trainingDataset.AR_iterations))
    training_num_workers = tune_num_workers(dataset = trainingDataset,
                                            model = model, 
                                            optimizer = optimizer, 
                                            criterion = criterion, 
                                            num_workers_list = num_workers_list, 
                                            # DataLoader options
                                            batch_size = training_batch_size, 
                                            prefetch_in_GPU = prefetch_in_GPU,
                                            prefetch_factor = prefetch_factor,
                                            pin_memory = pin_memory,
                                            asyncronous_GPU_transfer = asyncronous_GPU_transfer,
                                            # Timing options
                                            training_mode = True, 
                                            n_repetitions = 5,
                                            verbose = True)
    print('  --> Selecting num_workers={} for TrainingDataLoader.'.format(training_num_workers))
    
    # - Time AR validation 
    if validationDataset is not None: 
        print()
        print("- Timing AR validation with {} AR iterations:".format(validationDataset.AR_iterations))
        validation_num_workers = tune_num_workers(dataset = validationDataset,
                                                  model = model, 
                                                  optimizer = optimizer, 
                                                  criterion = criterion, 
                                                  num_workers_list = num_workers_list, 
                                                  # DataLoader options
                                                  batch_size = validation_batch_size, 
                                                  prefetch_in_GPU = prefetch_in_GPU,
                                                  prefetch_factor = prefetch_factor,
                                                  pin_memory = pin_memory,
                                                  asyncronous_GPU_transfer = asyncronous_GPU_transfer,
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
                                                  random_shuffle = random_shuffle,
                                                  num_workers = training_num_workers,
                                                  prefetch_factor = prefetch_factor, 
                                                  prefetch_in_GPU = prefetch_in_GPU,  
                                                  pin_memory = pin_memory,
                                                  asyncronous_GPU_transfer = asyncronous_GPU_transfer, 
                                                  device = device)
    if da_validation_dynamic is not None:
        validationDataLoader = AutoregressiveDataLoader(dataset = validationDataset, 
                                                        batch_size = validation_batch_size,  
                                                        drop_last_batch = drop_last_batch,
                                                        random_shuffle = random_shuffle,
                                                        num_workers = validation_num_workers,
                                                        prefetch_in_GPU = prefetch_in_GPU,  
                                                        prefetch_factor = prefetch_factor, 
                                                        pin_memory = pin_memory,
                                                        asyncronous_GPU_transfer = asyncronous_GPU_transfer, 
                                                        device = device)
        validationDataLoader_iter = cylic_iterator(validationDataLoader)
        print('- Creation of AutoregressiveDataLoaders: {:.0f}s'.format(time.time() - t_i))
    else: 
        validationDataset = None
        validationDataLoader = None
        
    ##------------------------------------------------------------------------.
    # Initialize AR TrainingInfo object 
    training_info = AR_TrainingInfo(AR_iterations=AR_iterations,
                                    epochs = epochs)  
    
    ##------------------------------------------------------------------------.
    # Get dimension infos
    dim_info = trainingDataset.dim_info
    dim_names = tuple(dim_info.keys())
    ##------------------------------------------------------------------------.
    # Iterate along epochs
    flag_stop_training = False
    for epoch in range(epochs):
        training_info.new_epoch()
        model.train() # Set model layers (i.e. batchnorm) in training mode 
        ##--------------------------------------------------------------------.
        # Iterate along training batches 
        trainingDataLoader_iter = iter(trainingDataLoader)
        for batch_count in range(len(trainingDataLoader_iter)):  
            ##----------------------------------------------------------------.   
            # Retrieve the training batch
            training_batch_dict = next(trainingDataLoader_iter)
            ##----------------------------------------------------------------.      
            # Perform autoregressive training loop
            # - The number of AR iterations is determined by AR_scheduler.AR_weights 
            # - If AR_weights are all zero after N forecast iteration:
            #   --> Load data just for F forecast iteration 
            #   --> Autoregress model predictions just N times to save computing time
            dict_training_Y_predicted = {}
            dict_training_loss_per_AR_iteration = {}
            for i in range(AR_scheduler.current_AR_iterations+1):
                # Retrieve X and Y for current AR iteration
                torch_X, torch_Y = get_AR_batch(AR_iteration = i, 
                                                batch_dict = training_batch_dict, 
                                                dict_Y_predicted = dict_training_Y_predicted,
                                                asyncronous_GPU_transfer = asyncronous_GPU_transfer,
                                                device = device)
                        
                ##------------------------------------------------------------.
                # Forward pass and store output for stacking into next AR iterations
                dict_training_Y_predicted[i] = model(torch_X)
                
                ##------------------------------------------------------------.
                # Compute loss for current forecast iteration 
                # - The criterion expects [data_points, nodes, features]
                # - Collapse all other dimensions to a 'data_points' dimension  
                Y_pred, Y_obs = reshape_tensors_4_loss(Y_pred = dict_training_Y_predicted[i],
                                                       Y_obs = torch_Y,
                                                       dim_names = dim_names)
                dict_training_loss_per_AR_iteration[i] = criterion(Y_obs, Y_pred)
                                                                    
                ##------------------------------------------------------------.
                # Remove unnecessary stored Y predictions 
                remove_unused_Y(AR_iteration = i, 
                                dict_Y_predicted = dict_training_Y_predicted,
                                dict_Y_to_remove = training_batch_dict['dict_Y_to_remove'])
                del Y_pred, Y_obs, torch_X, torch_Y
                if i == AR_scheduler.current_AR_iterations:
                    del dict_training_Y_predicted

            ##----------------------------------------------------------------.    
            ### Compute total (AR weighted) loss 
            for i, (AR_iteration, loss) in enumerate(dict_training_loss_per_AR_iteration.items()):
                if i == 0:
                    training_total_loss = AR_scheduler.AR_weights[AR_iteration] * loss 
                else: 
                    training_total_loss += AR_scheduler.AR_weights[AR_iteration] * loss
              
            ##----------------------------------------------------------------.       
            ### Backprogate the gradients and update the network weights 
            # Backward pass 
            # - Update the network weights 
            # Zeros all the gradients
            # - By default gradients are accumulated in buffers (and not overwritten)
            optimizer.zero_grad()   
            training_total_loss.backward()
            optimizer.step()  
            
            ##----------------------------------------------------------------. 
            # - Update training statistics
            if training_info.score_interval == scoring_interval:
                training_info.update_training_stats(total_loss = training_total_loss,
                                                    dict_loss_per_AR_iteration = dict_training_loss_per_AR_iteration, 
                                                    AR_scheduler = AR_scheduler, 
                                                    LR_scheduler = LR_scheduler) 
    
            ##----------------------------------------------------------------. 
            ### Run validation 
            if validationDataset is not None:
                if training_info.score_interval == scoring_interval:
                    # Set model layers (i.e. batchnorm) in evaluation mode 
                    model.eval() 
                    
                    # Retrieve batch for validation
                    validation_batch_dict = next(validationDataLoader_iter)
                    
                    # Initialize 
                    dict_validation_loss_per_AR_iteration = {}
                    dict_validation_Y_predicted = {}
                    #-#-------------------------------------------------------.
                    # Disable gradient calculations 
                    # - And do not update network weights  
                    with torch.set_grad_enabled(False): 
                        # Autoregressive loop 
                        for i in range(AR_scheduler.current_AR_iterations+1):
                            # Retrieve X and Y for current AR iteration
                            torch_X, torch_Y = get_AR_batch(AR_iteration = i, 
                                                            batch_dict = validation_batch_dict, 
                                                            dict_Y_predicted = dict_validation_Y_predicted,
                                                            asyncronous_GPU_transfer = asyncronous_GPU_transfer,
                                                            device = device)
                        
                            ##------------------------------------------------.
                            # Forward pass and store output for stacking into next AR iterations
                            dict_validation_Y_predicted[i] = model(torch_X)
                
                            ##------------------------------------------------.
                            # Compute loss for current forecast iteration 
                            # - The criterion expects [data_points, nodes, features] 
                            Y_pred, Y_obs = reshape_tensors_4_loss(Y_pred = dict_validation_Y_predicted[i],
                                                                   Y_obs = torch_Y,
                                                                   dim_names = dim_names)
                            dict_validation_loss_per_AR_iteration[i] = criterion(Y_obs, Y_pred)
                            
                            ##------------------------------------------------.
                            # Remove unnecessary stored Y predictions 
                            remove_unused_Y(AR_iteration = i, 
                                            dict_Y_predicted = dict_validation_Y_predicted,
                                            dict_Y_to_remove = validation_batch_dict['dict_Y_to_remove'])
                            del Y_pred, Y_obs, torch_X, torch_Y
                            if i == AR_scheduler.current_AR_iterations:
                                del dict_validation_Y_predicted

                    ##--------------------------------------------------------.    
                    ### Compute total (AR weighted) loss 
                    for i, (AR_iteration, loss) in enumerate(dict_validation_loss_per_AR_iteration.items()):
                        if i == 0:
                            validation_total_loss = AR_scheduler.AR_weights[AR_iteration] * loss 
                        else: 
                            validation_total_loss += AR_scheduler.AR_weights[AR_iteration] * loss
                    
                    ##---------------------------------------------------------. 
                    ### Update validation info 
                    training_info.update_validation_stats(total_loss = validation_total_loss,
                                                          dict_loss_per_AR_iteration = dict_validation_loss_per_AR_iteration)

            ##-----------------------------------------------------------------. 
            # - Update learning rate 
            if LR_scheduler is not None:
                LR_scheduler.step() 
                
            ##-----------------------------------------------------------------. 
            # - Update the AR weights 
            AR_scheduler.step()
            
            ##-----------------------------------------------------------------. 
            # - Evaluate stopping metrics  
            # --> Update AR scheduler if the loss has plateau
            if training_info.score_interval == scoring_interval:
                # Reset counter for scoring 
                training_info.reset_score_interval()  
                ##-------------------------------------------------------------.
                # If the model has not improved (based on early stopping settings)
                # - If current_AR_iterations < AR_iterations --> Update AR scheduler
                # - If current_AR_iterations = AR_iterations --> Stop training 
                if early_stopping(training_info):
                    # - If current_AR_iterations < AR_iterations --> Update AR scheduler
                    if AR_scheduler.current_AR_iterations < AR_iterations: 
                        ##----------------------------------------------------.
                        # Update the AR scheduler
                        AR_scheduler.update()
                        # Reset early stopiing counter 
                        early_stopping.reset()  
                        # Print info
                        current_training_info = "(epoch: {}, iteration: {}, total_iteration: {})".format(training_info.current_epoch, 
                                                                                                         training_info.current_epoch_iteration,
                                                                                                         training_info.iteration)
                        print("--> Updating training to {} AR iterations {}.".format(AR_scheduler.current_AR_iterations, current_training_info))
                        ##----------------------------------------------------.           
                        # Update Datasets (to prefetch the correct amount of data)
                        # - Training
                        del trainingDataLoader, trainingDataLoader_iter
                        trainingDataset.update_AR_iterations(AR_scheduler.current_AR_iterations)
                        # - Validation
                        if validationDataset is not None: 
                            del validationDataLoader, validationDataLoader_iter
                            validationDataset.update_AR_iterations(AR_scheduler.current_AR_iterations)
                        ##----------------------------------------------------.                              
                        ## Time execution         
                        # - Time AR training    
                        print("- Timing AR training with {} AR iterations:".format(trainingDataset.AR_iterations))
                        training_num_workers = tune_num_workers(dataset = trainingDataset,
                                                                model = model, 
                                                                optimizer = optimizer, 
                                                                criterion = criterion, 
                                                                num_workers_list = num_workers_list, 
                                                                # DataLoader options
                                                                batch_size = training_batch_size, 
                                                                prefetch_in_GPU = prefetch_in_GPU,
                                                                prefetch_factor = prefetch_factor,
                                                                pin_memory = pin_memory,
                                                                asyncronous_GPU_transfer = asyncronous_GPU_transfer,
                                                                # Timing options
                                                                training_mode = True, 
                                                                n_repetitions = 5,
                                                                verbose = True)
                        print('  --> Selecting num_workers={} for TrainingDataLoader.'.format(training_num_workers))
                        # - Time AR validation 
                        if validationDataset is not None: 
                            print()
                            print("- Timing AR validation with {} AR iterations:".format(validationDataset.AR_iterations))
                            validation_num_workers = tune_num_workers(dataset = validationDataset,
                                                                      model = model, 
                                                                      optimizer = optimizer, 
                                                                      criterion = criterion, 
                                                                      num_workers_list = num_workers_list, 
                                                                      # DataLoader options
                                                                      batch_size = validation_batch_size, 
                                                                      prefetch_in_GPU = prefetch_in_GPU,
                                                                      prefetch_factor = prefetch_factor,
                                                                      pin_memory = pin_memory,
                                                                      asyncronous_GPU_transfer = asyncronous_GPU_transfer,
                                                                      # Timing options
                                                                      training_mode = False, 
                                                                      n_repetitions = 5,
                                                                      verbose = True)
                            print('  --> Selecting num_workers={} for ValidationDataLoader.'.format(validation_num_workers))
                        ##----------------------------------------------------------------.
                        # Update DataLoaders (to prefetch the correct amount of data)
                        trainingDataLoader = AutoregressiveDataLoader(dataset = trainingDataset,                                                   
                                                                      batch_size = training_batch_size,  
                                                                      drop_last_batch = drop_last_batch,
                                                                      random_shuffle = random_shuffle,
                                                                      num_workers = training_num_workers,
                                                                      prefetch_factor = prefetch_factor, 
                                                                      prefetch_in_GPU = prefetch_in_GPU,  
                                                                      pin_memory = pin_memory,
                                                                      asyncronous_GPU_transfer = asyncronous_GPU_transfer, 
                                                                      device = device)
                        trainingDataLoader_iter = cylic_iterator(trainingDataLoader)
                        if validationDataset is not None: 
                            validationDataset.update_AR_iterations(AR_scheduler.current_AR_iterations)
                            validationDataLoader = AutoregressiveDataLoader(dataset = validationDataset, 
                                                                        batch_size = validation_batch_size,  
                                                                        drop_last_batch = drop_last_batch,
                                                                        random_shuffle = random_shuffle,
                                                                        num_workers = validation_num_workers,
                                                                        prefetch_in_GPU = prefetch_in_GPU,  
                                                                        prefetch_factor = prefetch_factor, 
                                                                        pin_memory = pin_memory,
                                                                        asyncronous_GPU_transfer = asyncronous_GPU_transfer,
                                                                        device = device)
                            validationDataLoader_iter = cylic_iterator(validationDataLoader)
                            
                    ##--------------------------------------------------------.     
                    # - If current_AR_iterations = AR_iterations --> Stop training 
                    else: 
                        # Stop training 
                        flag_stop_training = True
                        break
                    
            ##----------------------------------------------------------------.     
            # - Update iteration count 
            training_info.step()   
                        
        ##--------------------------------------------------------------------. 
        ### Print epoch training statistics  
        training_info.print_epoch_info()
        
        if flag_stop_training:
            break 
        ##--------------------------------------------------------------------. 
        # Option to save the model each epoch
        if save_model_each_epoch:
            torch.save(model.state_dict(), model_fpath[:-3] + '_epoch_{}'.format(epoch) + '.h5')
      
    ##------------------------------------------------------------------------.
    # Save final model 
    torch.save(model.state_dict(), f=model_fpath)    
    ##-------------------------------------------------------------------------.
    # Return training info object 
    return training_info
