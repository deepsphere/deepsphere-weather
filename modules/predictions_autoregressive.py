#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 19:49:16 2021

@author: ghiggi
"""
import torch
import numpy as np
import time
from modules.dataloader_autoregressive import get_AR_batch
from modules.dataloader_autoregressive import remove_unused_Y
from modules.utils_autoregressive import get_dict_stack_info
from modules.utils_autoregressive import check_AR_settings
from modules.utils_autoregressive import check_input_k
from modules.utils_autoregressive import check_output_k 

# ## Predictions with Dask cluster 
# https://examples.dask.org/machine-learning/torch-prediction.html

 
 prediction_ds.assign_coords({'lat': out_lat, 'lon': out_lon})

 next_batch_.append(inputs[:, :, in_feat:].detach().cpu().clone().numpy())
 output.detach().cpu().clone().numpy()[:, :, :out_feat])
 output.detach().cpu().clone().permute(0, 2, 3, 1).numpy()
 # Unormalize
 preds = preds * dg.dataset.std.values[:out_features] + dg.dataset.mean.values[:out_features]
 
 
def AutoregressivePredictions(model, 
                              # Data
                              ds_dynamic,
                              ds_static = None,              
                              ds_bc = None, 
                              scaler = None,
                              # Dataloader options
                              batch_size = 64, 
                              preload_data_in_CPU = False, 
                              num_workers = 0, 
                              pin_memory = False,
                              asyncronous_GPU_transfer = True,
                              device = 'cpu',
                              numeric_precision = numeric_precision, 
                              # Autoregressive settings  
                              input_k = [-3,-2,-1], 
                              output_k = [0],
                              forecast_cycle = 1,                           
                              AR_iterations = 50, 
                              stack_most_recent_prediction = True,
                              # Save options 
                              zarr_fpath = None, 
                              rounding = None
                              compressor = None
                              chunking = None):
    ##------------------------------------------------------------------------.
    # TODO 
    # TODO: code ... select_most_recent_prediction ... 
    # scaler 
    # zarr append 
        
    ## Checks arguments 
    if device == 'cpu' and pin_memory is True:
        pin_memory = False
    
    ##------------------------------------------------------------------------. 
    # Check Datasets are in the expected format  
    check_Datasets(ds_training_dynamic = ds_dynamic,
                   ds_static = ds_static,              
                   ds_training_bc = ds_bc,         
 
    ##------------------------------------------------------------------------.
    # Check that autoregressive settings are valid 
    # - input_k and output_k must be numpy arrays hereafter ! 
    input_k = check_input_k(input_k=input_k, AR_iterations=AR_iterations)   
    output_k = check_output_k(output_k = output_k) 
    check_AR_settings(input_k = input_k,
                      output_k = output_k,
                      forecast_cycle = forecast_cycle,                           
                      AR_iterations = AR_iterations, 
                      stack_most_recent_prediction = stack_most_recent_prediction)
                      
    ##------------------------------------------------------------------------.
    ### Load all data into CPU memory here if asked 
    if preload_data_in_CPU is True:
        ## Load Dynamic data
        t_i = time.time()
        ds_dynamic = ds_dynamic.compute()
        print("- Preload xarray Dataset of dynamic data into CPU memory: {.2f}s".format(time.time() - t_i))        
        ##--------------------------------------------------------------------.
        ## Boundary conditions data
        if ds_bc is not None: 
            t_i = time.time()
            ds_bc = ds_bc.compute()
            print("- Preload xarray Dataset of boundary conditions into CPU memory: {.2f}s".format(time.time() - t_i))             
    ##------------------------------------------------------------------------. 
    ### Conversion to DataArray and order dimensions 
    # - For dynamic and bc: ['time', 'node', 'features']
    # - For static: ['node', 'features']
    t_i = time.time()
    da_dynamic = da_dynamic.to_array(dim='feature', name='Dynamic').transpose('time', 'node', 'feature')
    if ds_bc is not None:
        da_bc = ds_bc.to_array(dim='feature', name='BC').transpose('time', 'node', 'feature')
    else: 
        da_bc = None
    if ds_static is not None: 
        da_static = ds_static.to_array(dim='feature', name='Static').transpose('node','feature') 
    else: 
        da_static = None
    print('- Conversion to xarray DataArrays: {:.2f}s'.format(time.time() - t_i))
    #-------------------------------------------------------------------------.
    # Define batch dimensions 
    dim_info = {}
    dim_info['batch_dim'] = 0
    dim_info['time'] = 1
    dim_info['node'] = 2
    dim_info['feature'] = 3    
    ##------------------------------------------------------------------------. 
    ### Create training Autoregressive Dataset and DataLoader    
    t_i = time.time()
    dataset = AutoregressiveDataset(da_dynamic = da_dynamic,  
                                    da_bc = da_bc,
                                    da_static = da_static,
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
    print('- Creation of AutoregressiveDataset: {:.2f}s'.format(time.time() - t_i))
    t_i = time.time()
    dataloader = AutoregressiveDataLoader(dataset = dataset, 
                                          batch_size = training_batch_size,  
                                          random_shuffle = False,
                                          num_workers = num_workers,
                                          pin_memory = pin_memory,
                                          dim_info = dim_info)
    
    print('- Creation of AutoregressiveDataLoader: {:.2f}s'.format(time.time() - t_i))
    ##------------------------------------------------------------------------.
    features = da_dynamic.feature 
    
          range(AR_iterations+1)                
    # Actual
    dataset. 
    
    start = np.datetime64(dg.years[0], 'h') + np.timedelta64(initial_lead_time, 'h')
    stop = start + np.timedelta64(dg.n_samples, 'h')
    times = np.arange(start, stop)
 
    #-------------------------------------------------------------------------.
    # If zarr fpath provided, create the required folder   
    if zarr_fpath is not None:
        if not os.path.exists(os.path.dirname(zarr_fpath)):
            os.makedirs(os.path.dirname(zarr_fpath))
                
    #-------------------------------------------------------------------------,
    # Initialize 
    model.to(device)  
    model.eval()
    t_i = time.time()
    list_da = []
    with torch.set_grad_enabled(False):
        ##--------------------------------------------------------------------.     
        # Iterate along training batches       
        for batch_dict in dataloader:            
            ##----------------------------------------------------------------.      
            # Perform autoregressive training loop
            # - The number of AR iterations is determined by AR_scheduler.AR_weights 
            # - If AR_weights are all zero after N forecast iteration:
            #   --> Load data just for F forecast iteration 
            #   --> Autoregress model predictions just N times to save computing time
            dict_Y_predicted = {}
            for i in range(AR_iterations+1):
                # Retrieve X and Y for current AR iteration
                torch_X, torch_Y = get_AR_batch(AR_iteration = i, 
                                                batch_dict = batch_dict, 
                                                dict_Y_predicted = dict_Y_predicted,
                                                device = device, 
                                                asyncronous_GPU_transfer = asyncronous_GPU_transfer)
                
                ##------------------------------------------------------------.
                # Forward pass and store output for stacking into next AR iterations
                dict_Y_predicted[i] = model(torch_X)
            
            # Select Y to stack (along forecast_iteration dimension)
            ##----------------------------------------------------------------.
            # Create xarray Dataset 
            forecasted_tensor =     dict_Y_predicted
            if rounding is not None: 
                forecasted_tensor = np.round(forecasted_tensor, rounding)
            
            # TODO: look at dimensions of CDS vs ECMWF 
            'forecast_time'
            'forecast_iteration'
            'forecast_leadtime' = 'forecast_iteration' * deltat
            'forecasted_time' = 'forecast_time' + 'forecast_leadtime'
            
            da=xr.DataArray(forecasted_tensor,           
                            dims=['lead_time', 'time', 'node', 'feature'],
                            coords={'lead_time': lead_times, 
                                    'time': valid_time, 
                                    'node': nodes, 
                                    'level': nlevs},
            # 'node': np.arange(nodes)
            ds = da.to_dataset(dim='feature')
            #-----------------------------------------------------------------.
            # Retransform data to original dimensions
            if scaler is not None: 
               ds = scaler.inverse_transform(ds)
            ##----------------------------------------------------------------.
            # If zarr_fpath not provided, keep predictions in memory 
            if zarr_fpath is not None:
                list_ds.append(ds)
            # Else, write forecast to zarr store  
            else:
                # Specify chunking 
                
                # Compressor 
                compressor = None
                encoding  = {var: {"compressor": compressor} for var in ['z', 't']} 
                
                if not os.path.exists(zarr_fpath):
                    ds.to_zarr(zarr_fpath, encoding=encoding, mode='w')
                else:                        
                    ds.to_zarr(zarr_fpath, append_dim='forecast_time')
                                
    #-------------------------------------------------------------------------.
    if zarr_fpath is not None:
        ds_forecasts = xr.open_zarr(zarr_fpath, chunks="auto")
    else: 
        ds_forecasts = xr.merge(list_ds)
    #-------------------------------------------------------------------------.    
    print("- Forecast generation: {.2f}s".format(t_i - time.time()))
    #-------------------------------------------------------------------------.    
    return ds_forecasts  
 


