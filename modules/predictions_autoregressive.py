#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 19:49:16 2021

@author: ghiggi
"""
import os
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
from modules.utils_io import check_AR_DataArrays
from modules.utils_torch import check_torch_device

# conda install -c conda-forge zarr
# conda install -c conda-forge cfgrib

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

##----------------------------------------------------------------------------.
def is_numcodecs(compressor):
    """Check is a numcodec compressor."""
    if type(compressor).__module__.find("numcodecs") == -1:
        return False
    else:
        return True

##----------------------------------------------------------------------------.
def check_compressor(compressor, variable_names, default_compressor = None):
    """Check compressor validity for zarr writing.
    
    compressor = None --> No compression.
    compressor = "auto" --> Use default_compressor if specified. Otherwise no compression is applied.
    compressor = {..} --> If compressor dictionary is specified, check that is valid.
    compressor = numcodecs class --> Create a dictionary for the specified compressor for each variable_name
    """
    ##------------------------------------------------------------------------.
    # Check variable_names type 
    if not isinstance(variable_names, (list, str)):
        raise TypeError("'variable_names' must be a string or a list")
    if isinstance(variable_names, str):
        variable_names = [variable_names]
    if not all([isinstance(s,str) for s in variable_names]):
        raise ValueError("Specify all variable names as string within the 'variable_names' list")
    # Check compressor type 
    if not (isinstance(compressor, (str, dict, type(None))) or is_numcodecs(compressor)):
        raise TypeError("'compressor' must be a dictionary, numcodecs compressor, 'auto' string or None.")
    if not (isinstance(default_compressor, type(None)) or is_numcodecs(default_compressor)):
        raise TypeError("'default_compressor' must be a numcodecs compressor or None.")
    ##------------------------------------------------------------------------.
    # If a string --> Apply default compressor (if specified)
    if isinstance(compressor, str): 
        if compressor == "auto" and default_compressor is not None:
            compressor = default_compressor
        else: 
            raise ValueError("If 'compressor' is specified as string, must be 'auto'")    
    ##------------------------------------------------------------------------.        
    # If a dictionary, check valid keys and valid compressor
    if isinstance(compressor, dict):
        if not np.all(np.isin(list(compressor.keys()), variable_names)):
            raise ValueError("The 'compressor' dictionary must contain the keys {}".format(variable_names))
        if not all([is_numcodecs(cmp) or isinstance(cmp, type(None)) for cmp in compressor.values()]):
            raise ValueError("The compressors specified in the 'compressor' dictionary must be numcodecs (or None).")
    ##------------------------------------------------------------------------.
    # If a unique compressor, create a dictionary with the same compressor for all variables
    if is_numcodecs(compressor):
        compressor = {var: compressor for var in variable_names}
    ##------------------------------------------------------------------------.    
    return compressor

##----------------------------------------------------------------------------.
def check_chunks(chunks, default_chunks = None):
    """Check chunks validity.
    
    chunks = None --> No chunking --> Contiguous.
    chunks = "auto" --> Use default_chunks is specified, otherwise default xarray chunks .
    chunks = {..} --> If default_chunks is specified, check that keys are the same.
    """
    if not isinstance(chunks, (str, dict, type(None))):
        raise TypeError("'chunks' must be a dictionary, 'auto' or None")
    if isinstance(chunks, str): 
        if chunks == "auto" and default_chunks is not None:
            chunks = default_chunks
        else: 
            raise ValueError("If 'chunks' is specified as string, must be 'auto'")
    # If a dictionary, check valid keys and values  
    if isinstance(chunks, dict):
        if default_chunks is not None:
            if not np.all(np.isin(list(chunks.keys()), list(default_chunks.keys()))):
                raise ValueError("The 'chunks' dictionary must contain the keys {}".format(list(default_chunks.keys())))
        if not all([isinstance(v, int) for v in chunks.values()]):
            raise ValueError("The 'chunks' values of the dictionary must be integers.")
    return chunks 

def check_rounding(rounding, variable_names):
    """Check rounding validity.
    
    rounding = None --> No rounding.
    rounding = int --> All variables will be round to the specified decimals.
    rounding = dict --> Specify specific rounding for each variable
    """
    ##------------------------------------------------------------------------.
    # Check variable_names type 
    if not isinstance(variable_names, (list, str)):
        raise TypeError("'variable_names' must be a string or a list")
    if isinstance(variable_names, str):
        variable_names = [variable_names]
    if not all([isinstance(s,str) for s in variable_names]):
        raise ValueError("Specify all variable names as string within the 'variable_names' list")
    # Check rounding type 
    if not isinstance(rounding, (int, dict, type(None))):
        raise TypeError("'rounding' must be a dictionary, integer or None.")
    ##------------------------------------------------------------------------.   
    # If a dictionary, check valid keys and valid compressor
    if isinstance(rounding, dict):
        if not np.all(np.isin(list(rounding.keys()), variable_names)):
            raise ValueError("The 'rounding' dictionary must contain the keys {}".format(variable_names))
        if not all([isinstance(v, (int, type(None))) for v in rounding.values()]):
            raise ValueError("The rounding decimals specified in the 'rounding' dictionary must be integers (or None).")
        if any([v < 0 for v in rounding.values() if v is not None]):
            raise ValueError("The rounding decimals specified in the 'rounding' dictionary must be positive integers (or None).") 
    ##------------------------------------------------------------------------.
    # If a unique compressor, create a dictionary with the same compressor for all variables
    if isinstance(rounding, int):
        if rounding < 0: 
            raise ValueError("'rounding' decimal value must be larger than 0")
    ##------------------------------------------------------------------------.    
    return rounding

#----------------------------------------------------------------------------.
def AutoregressivePredictions(model, 
                              # Data
                              da_dynamic,
                              da_static = None,              
                              da_bc = None, 
                              scaler = None,
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
    # Work only with GlobalScaler currently 
    # Work only if output_k are not replicated and already time-ordered  !
    ##------------------------------------------------------------------------.
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
        ##--------------------------------------------------------------------.
        # Check chunking 
        default_chunks = {'node': -1,
                          'forecast_reference_time': 1,
                          'leadtime': -1}
        chunks = check_chunks(chunks=chunks, default_chunks=default_chunks)       
        ##--------------------------------------------------------------------.
        # Check compressor (used as encoding for writing to zarr)
        default_compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
        compressor = check_compressor(compressor = compressor,  
                                      default_compressor = default_compressor,
                                      variable_names = da_dynamic['feature'].values.tolist())

        # Check rounding 
        rounding = check_rounding(rounding = rounding,
                                  variable_names = da_dynamic['feature'].values.tolist())

    ##------------------------------------------------------------------------.            
    ### Check timedelta_unit
    timedelta_unit = check_timedelta_unit(timedelta_unit=timedelta_unit)

    ##------------------------------------------------------------------------. 
    ### Create training Autoregressive Dataset and DataLoader    
    dataset = AutoregressiveDataset(da_dynamic = da_dynamic,  
                                    da_bc = da_bc,
                                    da_static = da_static,
                                    # Autoregressive settings  
                                    input_k = input_k,
                                    output_k = output_k,
                                    forecast_cycle = forecast_cycle,                           
                                    AR_iterations = AR_iterations, 
                                    max_AR_iterations = AR_iterations,
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
    
    t_res_timedelta = np.diff(da_dynamic.time.values)[0]    
    t_res_timedelta.astype(get_timedelta_types()[timedelta_unit])    
    ##------------------------------------------------------------------------.
    ### 4 DEBUGGING
    # d_iter = iter(dataloader)
    # batch_dict = next(d_iter)

    ##-------------------------------------------------------------------------,
    # Initialize 
    model.to(device)  
    model.eval()
    
    t_i = time.time()
    
    list_ds = []
    with torch.set_grad_enabled(False):
        ##--------------------------------------------------------------------.     
        # Iterate along training batches       
        for batch_dict in dataloader:            
            ##----------------------------------------------------------------.      
            ### Perform autoregressive loop
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
                
            ##----------------------------------------------------------------.
            # Retrieve forecast informations 
            dim_info = batch_dict['dim_info'] 
            forecast_time_info = batch_dict['forecast_time_info']
            
            forecast_reference_time = forecast_time_info["forecast_reference_time"]
            
            ##----------------------------------------------------------------.
            ### Select needed leadtime 
            # TODO  
            # - Args to choose which leadtime to select (when multiple are availables)
            # - Select Y to stack (along time dimension)
            # [forecast_time, leadtime, node, feature]
            dict_forecast_leadtime_idx = forecast_time_info["dict_forecast_leadtime_idx"]
            dict_Y_forecasted = dict_Y_predicted  # just keep required value in dim 1, already ordered
            
            ##----------------------------------------------------------------.
            # Create forecast tensors
            Y_forecasts = torch.cat(list(dict_Y_forecasted.values()), dim=dim_info['time'])  
            
            ##----------------------------------------------------------------.
            # Create numpy array 
            # .detach() should not be necessary if grad_disabled 
            if Y_forecasts.is_cuda is True: 
                tensor_forecasts = Y_forecasts.cpu().numpy()
            else:
                tensor_forecasts = Y_forecasts.numpy()
                
            ##----------------------------------------------------------------.
            # Remove unused tensors 
            del dict_Y_predicted
            del Y_forecasts
            
            ##----------------------------------------------------------------.
            ### Create xarray Dataset of forecasts
            # - Retrieve coords 
            leadtime_idx = np.arange(4)
            leadtime = leadtime_idx * t_res_timedelta   
            # - Create xarray DataArray 
            da=xr.DataArray(tensor_forecasts,           
                            dims=['forecast_reference_time', 'leadtime', 'node', 'feature'],
                            coords={'leadtime': leadtime, 
                                    'forecast_reference_time': forecast_reference_time, 
                                    'node': nodes, 
                                    'feature': features})
            # - Transform to dataset (to save to zarr)
            ds = da.to_dataset(dim='feature')
            
            ##-----------------------------------------------------------------.
            # Retransform data to original dimensions
            # TODO:
            # - Currenly only GlobalScalers works 
            # --> Scalers based on time ... with forecast ds time is not a dimension
            # --> How to implement?  
            
            if scaler is not None: 
                ds = scaler.inverse_transform(ds).compute()
                
            ##-----------------------------------------------------------------.
            # Rounding (if required)
            if rounding is not None: 
                if isinstance(rounding, int):
                    ds = ds.round(decimals=rounding)   
                elif isinstance(rounding, dict):
                    for var, decimal in rounding.items():
                        if decimal is not None: 
                            ds[var] = ds[var].round(decimal)                
                else: 
                    raise NotImplementedError("'rounding should be int, dict or None.")
                
            ##----------------------------------------------------------------.
            # If zarr_fpath not provided, keep predictions in memory 
            if zarr_fpath is not None:
                list_ds.append(ds)
                
            # Else, write forecast to zarr store  
            else:
                # Specify chunking 
                ds.chunk(chunks)  
                
                # Write / Append data to zarr store 
                if not os.path.exists(zarr_fpath):
                    ds.to_zarr(zarr_fpath, encoding=compressor, mode='w')                             # Create
                else:                        
                    ds.to_zarr(zarr_fpath, encoding=compressor, append_dim='forecast_reference_time') # Append
                                
    ##-------------------------------------------------------------------------.
    if zarr_fpath is not None:
        ds_forecasts = xr.open_zarr(zarr_fpath, chunks="auto")
    else: 
        ds_forecasts = xr.merge(list_ds)
        
    ##-------------------------------------------------------------------------.    
    print("- Forecast generation: {:.0f}s".format(time.time()-t_i))
    ##-------------------------------------------------------------------------.    
    return ds_forecasts  
 
##----------------------------------------------------------------------------.
# Conversion to 'time' as core dim
# TODO
# 'time' = 'forecast_reference_time' + 'leadtime'
# time = ds.forecast_reference_time + ds.leadtime