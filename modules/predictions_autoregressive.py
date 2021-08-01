#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 19:49:16 2021

@author: ghiggi
"""
import os
import glob
import shutil
import time
import torch
import zarr
import numpy as np
import xarray as xr
 
from modules.dataloader_autoregressive import get_AR_batch
from modules.dataloader_autoregressive import remove_unused_Y
from modules.dataloader_autoregressive import AutoregressiveDataset
from modules.dataloader_autoregressive import AutoregressiveDataLoader
from modules.utils_autoregressive import get_dict_stack_info
from modules.utils_autoregressive import check_AR_settings
from modules.utils_autoregressive import check_input_k
from modules.utils_autoregressive import check_output_k 
from modules.utils_zarr import check_chunks
from modules.utils_zarr import check_compressor
from modules.utils_zarr import check_rounding
from modules.utils_zarr import rechunk_Dataset
from modules.utils_io import check_AR_DataArrays
from modules.utils_torch import check_device
from modules.utils_torch import check_pin_memory
from modules.utils_torch import check_asyncronous_GPU_transfer
from modules.utils_torch import check_prefetch_in_GPU
from modules.utils_torch import check_prefetch_factor
from modules.utils_swag import bn_update

##----------------------------------------------------------------------------.
# conda install -c conda-forge zarr
# conda install -c conda-forge cfgrib
# conda install -c conda-forge rechunker

##----------------------------------------------------------------------------.
### Standard format for coordinates 
# import cf2cdm
# dir(cf2cdm.datamodels)
# cf2cdm.translate_coords(ds, cf2cdm.CDS)
# cf2cdm.translate_coords(ds, cf2cdm.ECMWF)

# CDS coordinates 
# - realization              int64  
# - forecast_reference_time  datetime64[ns]     # (base time)
# - leadtime                 timedelta64[ns]    
# - lat                      float64
# - lon                      float64
# - time                     datetime64[ns]     # aka forecasted_time
# -->  - leadtime_idx 

##----------------------------------------------------------------------------.
# ## Predictions with Dask cluster 
# https://examples.dask.org/machine-learning/torch-prediction.html

# Terminology to maybe change 
# scaler_transform
# scaler_inverse
#----------------------------------------------------------------------------.
###############
### Checks ####
###############
##----------------------------------------------------------------------------.
def check_timedelta_unit(timedelta_unit):
    """Check timedelta_unit validity."""
    if not isinstance(timedelta_unit, str):
        raise TypeError("'timedelta_unit' must be a string.")
    valid_timedelta_unit = list(get_timedelta_types().keys())
    if timedelta_unit not in valid_timedelta_unit:
        raise ValueError("Specify a valid 'timedelta_unit': {}".format(valid_timedelta_unit))
    return timedelta_unit

def get_timedelta_types():
    """Return {time_delta_unit: timedelta_type} dictionary."""
    timedelta_types = {'nanosecond': 'timedelta64[ns]',
                       'microsecond': 'timedelta64[ms]',
                       'second': 'timedelta64[s]',
                       'minute': 'timedelta64[m]',
                       'hour': 'timedelta64[h]',
                       'day': 'timedelta64[D]',
                       'month': 'timedelta64[M]',
                       'year': 'timedelta64[Y]'}
    return timedelta_types

#-----------------------------------------------------------------------------.
def AutoregressivePredictions(model, 
                              # Data
                              da_dynamic,
                              da_static = None,              
                              da_bc = None, 
                              # Scaler options
                              scaler = None,
                              scaler_transform = True,  # transform_input ???
                              scaler_inverse = True,    # backtransform_predictions ????
                              # Dataloader options
                              batch_size = 64, 
                              num_workers = 0,
                              prefetch_factor = 2, 
                              prefetch_in_GPU = False,  
                              pin_memory = False,
                              asyncronous_GPU_transfer = True,
                              device = 'cpu',
                              numeric_precision = "float32", 
                              # Autoregressive settings  
                              input_k = [-3,-2,-1], 
                              output_k = [0],
                              forecast_cycle = 1,                           
                              AR_iterations = 50, 
                              stack_most_recent_prediction = True,
                              # Save options 
                              zarr_fpath = None, 
                              rounding = None,
                              compressor = "auto",
                              chunks = "auto",
                              timedelta_unit='hour'):
    """AutoregressivePredictions.
    
    leadtime_idx = 0 correspond to the first forecast timestep
    leadtime = 0 correspond to the ground truth 
    leadtime = forecast_cycle correspond to the first forecast timestep
    forecast_reference_time = time at 'leadtime = 0' 
    
    Currently works only if the model predict on timestep at each forecast cycle ! (no multi-temporal output)
    """
    ##------------------------------------------------------------------------.
    # Work only if output_k are not replicated and are already time-ordered  !
    ##------------------------------------------------------------------------.
    ## Checks arguments 
    device = check_device(device)
    pin_memory = check_pin_memory(pin_memory=pin_memory, num_workers=num_workers, device=device)  
    asyncronous_GPU_transfer = check_asyncronous_GPU_transfer(asyncronous_GPU_transfer=asyncronous_GPU_transfer, device=device) 
    prefetch_in_GPU = check_prefetch_in_GPU(prefetch_in_GPU=prefetch_in_GPU, num_workers=num_workers, device=device) 
    prefetch_factor = check_prefetch_factor(prefetch_factor=prefetch_factor, num_workers=num_workers)
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
    # Check that DataArrays are valid 
    check_AR_DataArrays(da_training_dynamic = da_dynamic,
                        da_training_bc = da_bc,
                        da_static = da_static)     
    ##------------------------------------------------------------------------.
    # Check zarr settings 
    # If zarr fpath provided, create the required folder   
    if zarr_fpath is not None:
        if not os.path.exists(os.path.dirname(zarr_fpath)):
            os.makedirs(os.path.dirname(zarr_fpath))
    if zarr_fpath is not None: 
        variable_names = da_dynamic['feature'].values.tolist()
        ##--------------------------------------------------------------------.
        # Check chunking 
        default_chunks = {'node': -1,
                          'forecast_reference_time': 1,
                          'leadtime': 1}
        chunks = check_chunks(chunks=chunks, 
                              default_chunks=default_chunks,
                              variable_names=variable_names)       
        ##--------------------------------------------------------------------.
        # Check compressor (used as encoding for writing to zarr)
        default_compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
        compressor = check_compressor(compressor = compressor,  
                                      default_compressor = default_compressor,
                                      variable_names = variable_names)

        # Check rounding 
        rounding = check_rounding(rounding = rounding,
                                  variable_names = variable_names)

    ##------------------------------------------------------------------------.            
    ### Check timedelta_unit
    timedelta_unit = check_timedelta_unit(timedelta_unit=timedelta_unit)

    ##------------------------------------------------------------------------. 
    ### Prepare scalers for transform and inverse 
    if scaler is not None:
        if scaler_transform:
            scaler_transform = scaler
        else: 
            scaler_transform = None
        if scaler_inverse:   
            scaler_inverse = scaler 
        else: 
            scaler_inverse = None
    else: 
        scaler_transform = None 
        scaler_inverse = None                   
    ##------------------------------------------------------------------------.                            
    ### Create training Autoregressive Dataset and DataLoader    
    dataset = AutoregressiveDataset(da_dynamic = da_dynamic,  
                                    da_bc = da_bc,
                                    da_static = da_static,
                                    scaler = scaler_transform, 
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
                                          prefetch_in_GPU = prefetch_in_GPU,  
                                          pin_memory = pin_memory,
                                          asyncronous_GPU_transfer = asyncronous_GPU_transfer, 
                                          device = device)
    ##------------------------------------------------------------------------.
    # Retrieve dimension info of the forecast 
    nodes = da_dynamic.node
    features = da_dynamic.feature 
    
    t_res_timedelta = np.diff(da_dynamic.time.values)[0]*forecast_cycle   
    t_res_timedelta = t_res_timedelta.astype(get_timedelta_types()[timedelta_unit])    

    ##------------------------------------------------------------------------.
    # Initialize 
    model.to(device)  
    model.eval()
    
    t_i = time.time()
    
    list_ds = []
    with torch.set_grad_enabled(False):
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
                                          asyncronous_GPU_transfer = asyncronous_GPU_transfer,
                                          training_mode=False)
                ##------------------------------------------------------------.
                # Forward pass and store output for stacking into next AR iterations
                dict_Y_predicted[i] = model(torch_X)
                ##------------------------------------------------------------.
                # Remove unnecessary variables 
                # --> TODO: pass to CPU? Add remove_unused from GPU? 
                # torch.cuda.synchronize()
                del torch_X
                # print("{}: {:.2f} MB".format(i, torch.cuda.memory_allocated()/1000/1000)) 
                
            ##----------------------------------------------------------------.
            # Retrieve forecast informations 
            dim_info = batch_dict['dim_info'] 
            forecast_time_info = batch_dict['forecast_time_info']
            
            forecast_reference_time = forecast_time_info["forecast_reference_time"] - t_res_timedelta
            
            ##----------------------------------------------------------------.
            ### Select needed leadtime 
            # TODO c 
            # - Args to choose which leadtime to select (when multiple are availables)
            # - Select Y to stack (along time dimension)
            # [forecast_time, leadtime, node, feature]
            # dict_forecast_leadtime_idx = forecast_time_info["dict_forecast_leadtime_idx"]
            # dict_Y_forecasted = dict_Y_predicted  # just keep required value in dim 1, already ordered
            
            ##----------------------------------------------------------------.
            # Create forecast tensors
            Y_forecasts = torch.cat(list(dict_Y_predicted.values()), dim=dim_info['time'])  

            ##----------------------------------------------------------------.
            # Create numpy array 
            # .detach() should not be necessary if grad_disabled 
            if Y_forecasts.is_cuda: 
                Y_forecasts = Y_forecasts.cpu().numpy()
            else:
                Y_forecasts = Y_forecasts.numpy()
                
            ##----------------------------------------------------------------.
            # Remove unused tensors 
            del dict_Y_predicted
       
            ##----------------------------------------------------------------.
            ### Create xarray Dataset of forecasts
            # - Retrieve coords 
            # - TODO currently works only if predict on timestep at each forecast cycle !
            leadtime_idx = np.arange(Y_forecasts.shape[1]) # TODO generalize position
            leadtime = (leadtime_idx + 1) * t_res_timedelta # start at 'forecast_cycle' hours

            # - Create xarray DataArray 
            da=xr.DataArray(Y_forecasts,           
                            dims=['forecast_reference_time', 'leadtime', 'node', 'feature'],
                            coords={'leadtime': leadtime, 
                                    'forecast_reference_time': forecast_reference_time, 
                                    'node': nodes, 
                                    'feature': features})
            # - Transform to dataset (to save to zarr)
            ds = da.to_dataset(dim='feature')
            
            ##----------------------------------------------------------------.
            # Retransform data to original dimensions           
            if scaler_inverse is not None: 
                # - Apply scaler 
                # --> scaler.inverse_transform(ds).compute() works only for GlobalScalers
                # --> Need to create the time dimension to apply correctly TemporalScalers
                ds['time'] = ds['leadtime'] + ds['forecast_reference_time']
                ds = ds.set_coords('time')
                l_rescaled_ds = []
                for i in range(len(ds['forecast_reference_time'])):
                    tmp_ds = ds.isel(forecast_reference_time=i).swap_dims({"leadtime": "time"})
                    l_rescaled_ds.append(scaler_inverse.inverse_transform(tmp_ds).swap_dims({"time": "leadtime"}).drop('time'))
                ds = xr.concat(l_rescaled_ds, dim='forecast_reference_time')
            ##----------------------------------------------------------------.
            # Rounding (if required)
            if rounding is not None: 
                if isinstance(rounding, int):
                    ds = ds.round(decimals=rounding)   
                elif isinstance(rounding, dict):
                    for var, decimal in rounding.items():
                        if decimal is not None: 
                            ds[var] = ds[var].round(decimal)                
                else: 
                    raise NotImplementedError("'rounding' should be int, dict or None.")
                
            ##----------------------------------------------------------------.
            # If zarr_fpath not provided, keep predictions in memory 
            if zarr_fpath is None:
                list_ds.append(ds)
                
            # Else, write forecast to zarr store  
            else:
                ##--------------------------------------------.
                # Chunk the dataset
                for var, chunk in chunks.items():
                    ds[var] = ds[var].chunk(chunk)  
                ##--------------------------------------------.
                # Specify compressor 
                for var, comp in compressor.items(): 
                    ds[var].encoding['compressor'] = comp
                ##--------------------------------------------. 
                # Write / Append data to zarr store 
                if not os.path.exists(zarr_fpath):
                    ds.to_zarr(zarr_fpath, mode='w') # Create
                else:                        
                    ds.to_zarr(zarr_fpath, append_dim='forecast_reference_time') # Append
                                
    ##-------------------------------------------------------------------------.
    if zarr_fpath is not None:
        ds_forecasts = xr.open_zarr(zarr_fpath, chunks="auto")
    else: 
        ds_forecasts = xr.merge(list_ds)
        
    ##------------------------------------------------------------------------.    
    print("- Elapsed time for forecast generation: {:.2f} minutes".format((time.time()-t_i)/60))
    ##------------------------------------------------------------------------.    
    return ds_forecasts  
 
#----------------------------------------------------------------------------.
def AutoregressiveSWAGPredictions(model, exp_dir, 
                                # Data
                                da_training_dynamic,
                                da_test_dynamic = None,
                                da_static = None,              
                                da_training_bc = None,         
                                da_test_bc = None, 
                                # Scaler options
                                scaler = None,
                                scaler_transform = True,  # transform_input ???
                                scaler_inverse = True,    # backtransform_predictions ????
                                # Dataloader options
                                batch_size = 64, 
                                num_workers = 0,
                                prefetch_factor = 2, 
                                prefetch_in_GPU = False,  
                                pin_memory = False,
                                asyncronous_GPU_transfer = True,
                                device = 'cpu',
                                numeric_precision = "float32", 
                                # Autoregressive settings  
                                input_k = [-3,-2,-1], 
                                output_k = [0],
                                forecast_cycle = 1,                           
                                AR_iterations = 50, 
                                stack_most_recent_prediction = True,
                                # SWAG settings
                                no_cov_mat=False,
                                sampling_scale = 0.1,
                                nb_samples = 10,
                                # Save options  
                                rounding = None,
                                compressor = "auto",
                                chunks = "auto",
                                timedelta_unit='hour'):
    
    sampling_scale_str = str(sampling_scale).replace(".", "")
    
    for i in range(1, nb_samples+1): 
        print(f"- Sample {i}")
        forecast_zarr_fpath = os.path.join(exp_dir, f"model_predictions/spatial_chunks/test_pred_{sampling_scale_str}_temp{i}.zarr")
        with torch.no_grad():
            model.sample(sampling_scale, cov=(no_cov_mat))

        bn_update(model,
                # Data
                da_dynamic = da_training_dynamic,
                da_static = da_static,              
                da_bc = da_training_bc,          
                scaler = scaler, 
                # Dataloader options
                device = device,
                batch_size = batch_size,  # number of forecasts per batch
                num_workers = num_workers, 
                # tune_num_workers = False, 
                prefetch_factor = prefetch_factor, 
                prefetch_in_GPU = prefetch_in_GPU,  
                pin_memory = pin_memory,
                asyncronous_GPU_transfer = asyncronous_GPU_transfer,
                numeric_precision = numeric_precision, 
                # Autoregressive settings  
                input_k = input_k, 
                output_k = output_k, 
                forecast_cycle = forecast_cycle,                         
                AR_iterations = AR_iterations, 
                stack_most_recent_prediction = stack_most_recent_prediction
                )


        ds_forecasts = AutoregressivePredictions(model = model, 
                                                # Data
                                                da_dynamic = da_test_dynamic,
                                                da_static = da_static,              
                                                da_bc = da_test_bc, 
                                                scaler = scaler,
                                                scaler_transform = scaler_transform,  # transform_input ???
                                                scaler_inverse = scaler_inverse,    # backtransform_predictions ????
                                                # Dataloader options
                                                device = device,
                                                batch_size = 50,  # number of forecasts per batch
                                                num_workers = num_workers, 
                                                # tune_num_workers = False, 
                                                prefetch_factor = prefetch_factor, 
                                                prefetch_in_GPU = prefetch_in_GPU,  
                                                pin_memory = pin_memory,
                                                asyncronous_GPU_transfer = asyncronous_GPU_transfer,
                                                numeric_precision = numeric_precision, 
                                                # Autoregressive settings
                                                input_k = input_k, 
                                                output_k = output_k, 
                                                forecast_cycle = forecast_cycle,                         
                                                stack_most_recent_prediction = stack_most_recent_prediction, 
                                                AR_iterations = 20,        # How many time to autoregressive iterate
                                                # Save options 
                                                zarr_fpath = forecast_zarr_fpath,  # None --> do not write to disk
                                                rounding = rounding,             # Default None. Accept also a dictionary 
                                                compressor = compressor,      # Accept also a dictionary per variable
                                                chunks = chunks,          
                                                timedelta_unit=timedelta_unit)

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
    df_median.to_zarr(forecast_zarr_fpath, mode='w') # Create
    df_median = xr.open_zarr(forecast_zarr_fpath, chunks="auto")
    
    del ds_ensemble

    return df_median

#----------------------------------------------------------------------------.
def reshape_forecasts_for_verification(ds):
    """Process a Dataset with forecasts in the format required for verification."""
    l_reshaped_ds = []
    for i in range(len(ds['leadtime'])):
        tmp_ds = ds.isel(leadtime=i)
        tmp_ds['forecast_reference_time'] = tmp_ds['forecast_reference_time'] + tmp_ds['leadtime']
        tmp_ds = tmp_ds.rename({'forecast_reference_time': 'time'})    
        l_reshaped_ds.append(tmp_ds)
    ds = xr.concat(l_reshaped_ds, dim='leadtime', join='outer')
    return ds

def rechunk_forecasts_for_verification(ds, target_store, chunks="auto", max_mem = '1GB'):
    """
    Rechunk forecast Dataset in the format required for verification.
    
    Make data contiguous over the time dimension, and chunked over space.
    The forecasted time (referred as dimension 'time') is computed by 
    summing the leadtime to the forecast_reference_time. 

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with dimensions 'forecast_reference_time' and 'leadtime'.
    target_store : TYPE
        Filepath of the zarr store where to save the new Dataset.
    chunks : str, optional
        Option for custom chunks of the new Dataset. The default is "auto".
        The default is chunked pixel-wise and per leadtime, contiguous over time.
    max_mem : str, optional
        The amount of memory (in bytes) that workers are allowed to use.
        The default is '1GB'.

    Returns
    -------
    ds_verification : xarray.Dataset
        Dataset for verification (with 'time' and 'leadtime' dimensions.

    """
    # Define temp store for rechunking
    temp_store = os.path.join(os.path.dirname(target_store), "tmp_store.zarr")
    # Define intermediate store for rechunked data
    intermediate_store = os.path.join(os.path.dirname(target_store), "rechunked_store.zarr")
    ##------------------------------------------------------------------------.
    # Default chunking
    default_chunks = {'node': 1,
                      'forecast_reference_time': -1,
                      'leadtime': 1}
    # Check chunking
    variable_names = list(ds.data_vars.keys())
    chunks = check_chunks(chunks=chunks, default_chunks=default_chunks, variable_names=variable_names) 
    ##------------------------------------------------------------------------.
    # Rechunk Dataset (on disk)
    rechunk_Dataset(ds=ds, chunks=chunks, 
                    target_store=intermediate_store, temp_store=temp_store, 
                    max_mem = max_mem)
    ##------------------------------------------------------------------------.
    # Load rechunked dataset (contiguous over forecast referece time, chunked over space)
    ds = xr.open_zarr(intermediate_store, chunks="auto")
    ##------------------------------------------------------------------------.
    # Reshape 
    ds_verification = reshape_forecasts_for_verification(ds)
    ##------------------------------------------------------------------------.
    # Remove 'chunks' key in encoding (bug in xarray-dask-zarr)
    for var in variable_names:
        ds_verification[var].encoding.pop('chunks')
    
    ##------------------------------------------------------------------------.
    # Write to disk 
    ds_verification.to_zarr(target_store)
    ##------------------------------------------------------------------------.
    # Remove rechunked store 
    shutil.rmtree(intermediate_store)
    ##------------------------------------------------------------------------.
    # Load the Dataset for verification
    ds_verification = xr.open_zarr(target_store)
    ##------------------------------------------------------------------------.
    # Return the Dataset for verification
    return ds_verification
