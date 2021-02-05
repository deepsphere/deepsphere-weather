#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 18:42:12 2021

@author: ghiggi
"""
import pdb
import torch
import time
import numpy as np
from torch import optim

from modules.dataloader_autoregressive import AutoregressiveDataset
from modules.dataloader_autoregressive import AutoregressiveDataLoader
from modules.dataloader_autoregressive import get_AR_batch
from modules.dataloader_autoregressive import remove_unused_Y
from modules.dataloader_autoregressive import cylic_iterator
from modules.utils_autoregressive import get_dict_stack_info
from modules.utils_autoregressive import check_AR_settings
from modules.utils_autoregressive import check_input_k
from modules.utils_autoregressive import check_output_k 
from modules.utils_io import check_DataArrays_dimensions

from modules.utils_training import TrainingInfo
from modules.utils_torch import check_torch_device
##----------------------------------------------------------------------------.
# TODO DataLoader Options    
# - sampler                    # Could be useful for bootstraping ? 
# - worker_init_fn             # to initialize dask scheduler? to set RNG?
# - persistent_workers = False # but True might be appropriate

##----------------------------------------------------------------------------.
# TODOs
# - ONNX for saving model weights 
# - Record the loss per variable 
# - Loss --> Reshape data before applying loss

##----------------------------------------------------------------------------.               
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
                           # Dataloader options
                           prefetch_in_GPU = False,
                           prefetch_factor = 2,
                           drop_last_batch = True,
                           random_shuffle = True, 
                           num_workers = 0, 
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
    ##------------------------------------------------------------------------.
    # TODO 
    ## Checks arguments 
    device = check_torch_device(device)

    if device.type == 'cpu':
        if pin_memory is True:
            print("GPU is not available. 'pin_memory' set to False.")
            pin_memory = False
        if prefetch_in_GPU is True: 
            print("GPU is not available. 'prefetch_in_GPU' set to False.")
            prefetch_in_GPU = False
        if asyncronous_GPU_transfer is True: 
            print("GPU is not available. 'asyncronous_GPU_transfer' set to False.")
            asyncronous_GPU_transfer = False
    ##------------------------------------------------------------------------.   
    # Check that autoregressive settings are valid 
    # - input_k and output_k must be numpy arrays hereafter ! 
    input_k = check_input_k(input_k=input_k, AR_iterations=AR_iterations)   
    output_k = check_output_k(output_k=output_k)
    check_AR_settings(input_k = input_k,
                      output_k = output_k,
                      forecast_cycle = forecast_cycle,                           
                      AR_iterations = AR_iterations, 
                      stack_most_recent_prediction = stack_most_recent_prediction)    
    ##------------------------------------------------------------------------.
    # Check that DataArrays are valid 
    check_DataArrays_dimensions(da_training_dynamic = da_training_dynamic,
                                da_validation_dynamic = da_validation_dynamic, 
                                da_training_bc = da_training_bc,
                                da_validation_bc = da_validation_bc, 
                                da_static = da_static)
    ##------------------------------------------------------------------------.
    ## Autotune DataLoaders 
    # --> Tune the number of num_workers for best performance 
    # TODO
    # - Performe 10 iterations for measuring timing at the beginning 
    # --> Time the dataloader at multiple iterations (in the first it might open the dataset on each worker) 
    # - num_workers=os.cpu_count() # num_worker = 4 * num_GPU
    # - torch.cuda.memory_summary() 
    # - Time data-loading 
    # - Time forward 
    # - Time backward 
        
    ##------------------------------------------------------------------------.
    ## Create DataLoaders
    # - Prefetch (prefetch_factor*num_workers) batches parallelly into CPU
    # - At each AR iteration, the required data are transferred asynchronously to GPU 
    # - If static data are provided, they are prefetched into the GPU 
    # - Some data are duplicated in CPU memory because of the data overlap between forecast iterations.
    #   However this mainly affect boundary conditions data, because dynamic data
    #   after few AR iterations are the predictions of previous AR iteration.
    #
    # Create training Autoregressive Dataset and DataLoader    
    t_i = time.time()
    trainingDataset = AutoregressiveDataset(da_dynamic = da_training_dynamic,  
                                            da_bc = da_training_bc,
                                            da_static = da_static,
                                            # Autoregressive settings  
                                            input_k = input_k,
                                            output_k = output_k,
                                            forecast_cycle = forecast_cycle,  
                                            AR_iterations = AR_scheduler.current_AR_iterations,
                                            max_AR_iterations = AR_iterations,
                                            stack_most_recent_prediction = stack_most_recent_prediction, 
                                            # GPU settings 
                                            device = device,
                                            # Precision settings
                                            numeric_precision = numeric_precision)
    print('- Creation of Training AutoregressiveDataset: {:.0f}s'.format(time.time() - t_i))
    
    t_i = time.time()
    trainingDataLoader = AutoregressiveDataLoader(dataset = trainingDataset,                                                   
                                                  batch_size = training_batch_size,  
                                                  drop_last_batch = drop_last_batch,
                                                  random_shuffle = random_shuffle,
                                                  num_workers = num_workers,
                                                  prefetch_factor = prefetch_factor, 
                                                  prefetch_in_GPU = prefetch_in_GPU,  
                                                  pin_memory = pin_memory,
                                                  asyncronous_GPU_transfer = asyncronous_GPU_transfer, 
                                                  device = device)
    print('- Creation of Training AutoregressiveDataLoader: {:.0f}s'.format(time.time() - t_i))
    
    ### Create validation Autoregressive Dataset and DataLoader
    if da_validation_dynamic is not None:
        t_i = time.time()
        validationDataset = AutoregressiveDataset(da_dynamic = da_validation_dynamic,  
                                                  da_bc = da_validation_bc,
                                                  da_static = da_static,   
                                                  # Autoregressive settings  
                                                  input_k = input_k,
                                                  output_k = output_k,
                                                  forecast_cycle = forecast_cycle,                           
                                                  AR_iterations = AR_scheduler.current_AR_iterations,
                                                  max_AR_iterations = AR_iterations,
                                                  stack_most_recent_prediction = stack_most_recent_prediction, 
                                                  # GPU settings 
                                                  device = device,
                                                  # Precision settings
                                                  numeric_precision = numeric_precision)
        print('- Creation of Validation AutoregressiveDataset: {:.0f}s'.format(time.time() - t_i))
        validationDataLoader = AutoregressiveDataLoader(dataset = validationDataset, 
                                                        batch_size = validation_batch_size,  
                                                        drop_last_batch = drop_last_batch,
                                                        random_shuffle = random_shuffle,
                                                        num_workers = num_workers,
                                                        prefetch_in_GPU = prefetch_in_GPU,  
                                                        prefetch_factor = prefetch_factor, 
                                                        pin_memory = pin_memory,
                                                        asyncronous_GPU_transfer = asyncronous_GPU_transfer, 
                                                        device = device)
        print('- Creation of Validation AutoregressiveDataLoader: {:.0f}s'.format(time.time() - t_i))
    else: 
        validationDataset = None
        validationDataLoader = None
        
    ##------------------------------------------------------------------------.
    # Retrieve information for autoregress/stack the predicted data 
    _, dict_Y_to_remove = get_dict_stack_info(AR_iterations = AR_iterations, 
                                              forecast_cycle = forecast_cycle, 
                                              input_k = input_k, 
                                              output_k = output_k, 
                                              stack_most_recent_prediction = stack_most_recent_prediction)

    ##------------------------------------------------------------------------.
    # Initialize training info object 
    training_info = TrainingInfo(AR_iterations=AR_iterations,
                                 epochs = epochs)

    ##------------------------------------------------------------------------.
    # Zeros gradients     
    optimizer.zero_grad() 
    
    ##------------------------------------------------------------------------.
    # Initialize infinite validationDataLoader iterator
    validation_data_available = validationDataLoader is not None
    if validation_data_available:
        validationDataLoader_iter = cylic_iterator(validationDataLoader)
        
    ##------------------------------------------------------------------------.
    # Iterate along epochs
    for epoch in range(epochs):
        training_info.new_epoch()
        model.train() # Set model layers (i.e. batchnorm) in training mode 
        ##--------------------------------------------------------------------. 
        # Iterate along training batches       
        for training_batch_dict in trainingDataLoader:   
            print(".", end="")
            ##----------------------------------------------------------------.      
            # Perform autoregressive training loop
            # - The number of AR iterations is determined by AR_scheduler.AR_weights 
            # - If AR_weights are all zero after N forecast iteration:
            #   --> Load data just for F forecast iteration 
            #   --> Autoregress model predictions just N times to save computing time
            dict_training_Y_predicted = {}
            dict_training_loss_per_leadtime = {}
            for i in range(AR_scheduler.current_AR_iterations+1):
                # Retrieve X and Y for current AR iteration
                torch_X, torch_Y = get_AR_batch(AR_iteration = i, 
                                                batch_dict = training_batch_dict, 
                                                dict_Y_predicted = dict_training_Y_predicted,
                                                device = device, 
                                                asyncronous_GPU_transfer = asyncronous_GPU_transfer)
                        
                ##------------------------------------------------------------.
                # Forward pass and store output for stacking into next AR iterations
                dict_training_Y_predicted[i] = model(torch_X)
                
                ##------------------------------------------------------------.
                # Compute loss for current forecast iteration 
                # - The torch tensors are [sample, time, nodes, features]
                # - The criterion expects [samples, nodes, features]
                # - Collapse time dimension with sample dimension 
                Y_dims = torch_Y.shape
                reshape_dims = (-1, Y_dims[2], Y_dims[3])
                dict_training_loss_per_leadtime[i] = criterion(dict_training_Y_predicted[i].reshape(*reshape_dims), 
                                                               torch_Y.reshape(*reshape_dims))  
                # dict_training_loss_per_leadtime[i] = criterion(dict_training_Y_predicted[i], torch_Y) 

                ##------------------------------------------------------------.
                # Remove unnecessary stored Y predictions 
                remove_unused_Y(AR_iteration = i, 
                                dict_Y_predicted = dict_training_Y_predicted,
                                dict_Y_to_remove = dict_Y_to_remove)
                if i == AR_scheduler.current_AR_iterations:
                    del dict_training_Y_predicted

            ##----------------------------------------------------------------.    
            ### Compute total (AR weighted) loss 
            for i, (leadtime, loss) in enumerate(dict_training_loss_per_leadtime.items()):
                if i == 0:
                    training_total_loss = AR_scheduler.AR_weights[leadtime] * loss 
                else: 
                    training_total_loss += AR_scheduler.AR_weights[leadtime] * loss
              
            ##----------------------------------------------------------------.       
            ### Backprogate the gradients and update the network weights 
            # Backward pass 
            training_total_loss.backward()          
 
            # Zeros all the gradients
            # - By default gradients are accumulated in buffers (and not overwritten)
            optimizer.zero_grad()   
            
            # - Update the network weights 
            optimizer.step() 
            
            ##----------------------------------------------------------------. 
            # - Update training statistics
            if training_info.score_interval == scoring_interval:
                training_info.update_training_stats(total_loss = training_total_loss,
                                                    dict_loss_per_leadtime = dict_training_loss_per_leadtime, 
                                                    AR_scheduler = AR_scheduler, 
                                                    LR_scheduler = LR_scheduler) 
    
            ##----------------------------------------------------------------. 
            ### Run validation 
            if validation_data_available:
                if training_info.score_interval == scoring_interval:
                    # Set model layers (i.e. batchnorm) in evaluation mode 
                    model.eval() 
                    
                    # Retrieve batch for validation
                    validation_batch_dict = next(validationDataLoader_iter)
                    
                    # Initialize 
                    dict_validation_loss_per_leadtime = {}
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
                                                            device = device, 
                                                            asyncronous_GPU_transfer = asyncronous_GPU_transfer)
                        
                            ##------------------------------------------------.
                            # Forward pass and store output for stacking into next AR iterations
                            dict_validation_Y_predicted[i] = model(torch_X)
                
                            ##------------------------------------------------.
                            # Compute loss for current forecast iteration 
                            # - The criterion expects [samples, nodes, features]
                            Y_dims = torch_Y.shape
                            reshape_dims = (-1, Y_dims[2], Y_dims[3])
                            dict_validation_loss_per_leadtime[i] = criterion(dict_validation_Y_predicted[i].reshape(*reshape_dims), 
                                                                             torch_Y.reshape(*reshape_dims))  
                            
                            ##------------------------------------------------.
                            # Remove unnecessary stored Y predictions 
                            remove_unused_Y(AR_iteration = i, 
                                            dict_Y_predicted = dict_validation_Y_predicted,
                                            dict_Y_to_remove = dict_Y_to_remove)
                            if i == AR_scheduler.current_AR_iterations:
                                del dict_validation_Y_predicted

                    ##--------------------------------------------------------.    
                    ### Compute total (AR weighted) loss 
                    for i, (leadtime, loss) in enumerate(dict_validation_loss_per_leadtime.items()):
                        if i == 0:
                            validation_total_loss = AR_scheduler.AR_weights[leadtime] * loss 
                        else: 
                            validation_total_loss += AR_scheduler.AR_weights[leadtime] * loss
                    
                    ##---------------------------------------------------------. 
                    ### Update validation info 
                    training_info.update_validation_stats(total_loss = validation_total_loss,
                                                          dict_loss_per_leadtime = dict_validation_loss_per_leadtime)

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
                if early_stopping(training_info) is True:
                   
                    if AR_scheduler.current_AR_iterations < AR_iterations: 
                        # Update the AR scheduler
                        print("Updating AR scheduler")
                        AR_scheduler.update()
                        # Update Datasets (to prefetch the correct amount of data)
                        # - TODO: check that also if prefetch is on... it works 
                        trainingDataset.update_AR_iterations(AR_scheduler.current_AR_iterations)
                        if validationDataset is not None: 
                            validationDataset.update_AR_iterations(AR_scheduler.current_AR_iterations)
                    else: 
                        # Stop training 
                        break
                    
            ##----------------------------------------------------------------.     
            # - Update iteration count 
            training_info.step()               

        ##--------------------------------------------------------------------. 
        ### Print epoch training statistics  
        training_info.print_epoch_info()
        
        ##--------------------------------------------------------------------. 
        # Option to save the model each epoch
        if save_model_each_epoch is True:
            torch.save(model.state_dict(), model_fpath[:-3] + '_epoch_{}'.format(epoch) + '.h5')
      
    ##------------------------------------------------------------------------.
    # Save final model 
    torch.save(model.state_dict(), f=model_fpath)
    ##-------------------------------------------------------------------------.
    # Return training info object 
    return training_info
