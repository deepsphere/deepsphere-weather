#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 11:38:23 2022

@author: ghiggi
"""
import os
import glob 
import torch
import xarray as xr
from modules.utils_swag import bn_update
from xforecasting import AutoregressivePredictions

     
def AutoregressiveSWAGPredictions(model, exp_dir, 
                                  # Data
                                  training_data_dynamic,
                                  training_data_bc = None,  
                                  data_static = None,   
                                  test_data_dynamic = None,                                       
                                  test_data_bc = None, 
                                  bc_generator = None,
                                  # Scaler options
                                  scaler_transform = None,   
                                  scaler_inverse = None,     
                                  # Dataloader options
                                  batch_size = 64, 
                                  num_workers = 0,
                                  prefetch_factor = 2, 
                                  prefetch_in_gpu = False,  
                                  pin_memory = False,
                                  asyncronous_gpu_transfer = True,
                                  device = 'cpu',
                                  # Autoregressive settings  
                                  input_k = [-3,-2,-1], 
                                  output_k = [0],
                                  forecast_cycle = 1,                           
                                  ar_iterations = 50, 
                                  stack_most_recent_prediction = True,
                                  # Prediction options 
                                  forecast_reference_times = None, 
                                  keep_first_prediction = True, 
                                  ar_blocks = None,
                                  # SWAG settings
                                  no_cov_mat=False,
                                  sampling_scale = 0.1,
                                  nb_samples = 10,
                                  # Save options  
                                  rounding = None,
                                  compressor = "auto",
                                  chunks = "auto"):
    """ Caution: the following function is in development !"""                                 
    sampling_scale_str = str(sampling_scale).replace(".", "")
    
    for i in range(1, nb_samples+1): 
        print(f"- Sample {i}")
        forecast_zarr_fpath = os.path.join(exp_dir, f"model_predictions/spatial_chunks/test_pred_{sampling_scale_str}_temp{i}.zarr")
        with torch.no_grad():
            model.sample(sampling_scale, cov=(no_cov_mat))

        bn_update(model,
                # Data
                data_dynamic = training_data_dynamic,
                data_bc = training_data_bc,    
                data_static = data_static,      
                bc_generator = bc_generator,         
                scaler = scaler_transform, 
                # Dataloader options
                device = device,
                batch_size = batch_size,  # number of forecasts per batch
                num_workers = num_workers, 
                # tune_num_workers = False, 
                prefetch_factor = prefetch_factor, 
                prefetch_in_gpu = prefetch_in_gpu,  
                pin_memory = pin_memory,
                asyncronous_gpu_transfer = asyncronous_gpu_transfer,
                # Autoregressive settings  
                input_k = input_k, 
                output_k = output_k, 
                forecast_cycle = forecast_cycle,                         
                ar_iterations = ar_iterations, 
                stack_most_recent_prediction = stack_most_recent_prediction
                )

    _ = AutoregressivePredictions(model = model, 
                                  # Data
                                  data_static = data_static,
                                  data_dynamic = test_data_dynamic,
                                  data_bc = test_data_bc,         
                                  bc_generator = bc_generator, 
                                  scaler_transform = scaler_transform,   
                                  scaler_inverse = scaler_inverse,    
                                  # Dataloader options
                                  device = device,
                                  batch_size = batch_size,  # number of forecasts per batch
                                  num_workers = num_workers, 
                                  # tune_num_workers = False, 
                                  prefetch_factor = prefetch_factor, 
                                  prefetch_in_gpu = prefetch_in_gpu,  
                                  pin_memory = pin_memory,
                                  asyncronous_gpu_transfer = asyncronous_gpu_transfer,
                                  # Autoregressive settings
                                  input_k = input_k, 
                                  output_k = output_k, 
                                  forecast_cycle = forecast_cycle,                         
                                  stack_most_recent_prediction = stack_most_recent_prediction, 
                                  ar_iterations = ar_iterations,        # How many time to autoregressive iterate
                                  # Prediction options 
                                  forecast_reference_times = forecast_reference_times, 
                                  keep_first_prediction = keep_first_prediction, 
                                  ar_blocks = ar_blocks,
                                  # Save options 
                                  zarr_fpath = forecast_zarr_fpath,  # None --> do not write to disk
                                  rounding = rounding,             # Default None. Accept also a dictionary 
                                  compressor = compressor,      # Accept also a dictionary per variable
                                  chunks = chunks)         

    ##-------------------------------------------------------------------------.
    # Ensemble the predicitons along dim "member"
    zarr_members_fpaths = glob.glob(os.path.join(exp_dir, f"model_predictions/spatial_chunks/test_pred_{sampling_scale_str}_*"))
    list_ds_member = [xr.open_zarr(fpath) for fpath in zarr_members_fpaths]
    ds_ensemble = xr.concat(list_ds_member, dim="member")
    
    del list_ds_member
        
    ##-------------------------------------------------------------------------.
    # Save ensemble
    forecast_zarr_fpath = os.path.join(exp_dir, f"model_predictions/spatial_chunks/test_pred_{sampling_scale_str}.zarr")
    if not os.path.exists(forecast_zarr_fpath):
        ds_ensemble.to_zarr(forecast_zarr_fpath, mode='w') # Create
    else:                        
        ds_ensemble.to_zarr(forecast_zarr_fpath, append_dim='member') # Append
    ds_ensemble = xr.open_zarr(forecast_zarr_fpath, chunks="auto")

    ##-------------------------------------------------------------------------.
    # Remove individual members
    for member in zarr_members_fpaths:
        shutil.rmtree(member)
    
    ##-------------------------------------------------------------------------.
    # Compute median of ensemble
    forecast_zarr_fpath = os.path.join(exp_dir, f"model_predictions/spatial_chunks/test_pred_{sampling_scale_str}_median.zarr")
    df_median = ds_ensemble.median(dim="member")
    del ds_ensemble
    df_median.to_zarr(forecast_zarr_fpath, mode='w') # Create
    df_median = xr.open_zarr(forecast_zarr_fpath, chunks="auto")
    
    return df_median