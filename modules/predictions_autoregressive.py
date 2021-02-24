#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 19:49:16 2021

@author: ghiggi
"""
import os
import shutil
import time
import torch
import zarr
import dask 
import numpy as np
import xarray as xr
from rechunker import rechunk
from dask.diagnostics import ProgressBar

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
from modules.utils_torch import check_pin_memory
from modules.utils_torch import check_asyncronous_GPU_transfer
from modules.utils_torch import check_prefetch_in_GPU
from modules.utils_torch import check_prefetch_factor
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
    if not all([isinstance(s, str) for s in variable_names]):
        raise TypeError("Specify all variable names as string within the 'variable_names' list.")
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
            raise ValueError("If 'compressor' is specified as string, must be 'auto'.")    
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
def check_chunks(chunks, variable_names, default_chunks = None):
    """Check chunks validity.
    
    chunks = None --> No chunking --> Contiguous.
    chunks = "auto" --> Use default_chunks is specified, otherwise default xarray chunks .
    chunks = {..} --> If default_chunks is specified, check that keys are the same.
    """
    # Check variable_names and chunks types
    if not isinstance(chunks, (str, dict, type(None))):
        raise TypeError("'chunks' must be a dictionary, 'auto' or None.")
    if isinstance(variable_names, str):
        variable_names = [variable_names]
    if not all([isinstance(s, str) for s in variable_names]):
        raise TypeError("Specify all variable names as string within the 'variable_names' list.")
    ##------------------------------------------------------------------------.
    # If a string --> Auto --> Apply default_chunks (if specified)  
    if isinstance(chunks, str): 
        if chunks == "auto" and default_chunks is not None:
            chunks = default_chunks
        elif chunks == "auto" and default_chunks is None:
            chunks = None
        else: 
            raise ValueError("If 'chunks' is specified as string, must be 'auto'.")
    ##------------------------------------------------------------------------.
    # If a dictionary, check valid keys and values  
    if isinstance(chunks, dict):
        # If a chunk specific for each variable is specified (keys are variable_names)
        if np.all(np.isin(list(chunks.keys()), variable_names)):
            if not np.all(np.isin(variable_names, list(chunks.keys()))):
                raise ValueError("If you specify specific chunks for each variable, please specify it for all variables.")
            # - Check that the chunk for each dimension is specified
            for key in chunks.keys():
                if default_chunks is not None:
                    if not np.all(np.isin(list(chunks[key].keys()), list(default_chunks.keys()))):
                        raise ValueError("The 'chunks' dictionary of {} must contain the keys {}".format(key, list(default_chunks.keys())))
                # - Check that the chunk value are integers
                if not all([isinstance(v, int) for v in chunks[key].values()]):
                    raise ValueError("The 'chunks' values of the {} dictionary must be integers.".format(key))
        # If a common chunk is specified for all variable_names (chunks keys are not variable_names)
        elif np.all(np.isin(list(chunks.keys()), variable_names, invert=True)):
            # - Check that the chunk for each dimension is specified
            if default_chunks is not None:
                if not np.all(np.isin(list(chunks.keys()), list(default_chunks.keys()))):
                    raise ValueError("The 'chunks' dictionary must contain the keys {}".format(list(default_chunks.keys())))
            # - Check that the chunk value are integers
            if not all([isinstance(v, int) for v in chunks.values()]):
                raise ValueError("The 'chunks' values of the dictionary must be integers.")
            # - Specify chunks for each variable
            chunks = {var: chunks for var in variable_names}
        else: 
            raise ValueError("This chunks option has not been implemented.")
    ##------------------------------------------------------------------------.    
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
        raise TypeError("'variable_names' must be a string or a list.")
    if isinstance(variable_names, str):
        variable_names = [variable_names]
    if not all([isinstance(s,str) for s in variable_names]):
        raise ValueError("Specify all variable names as string within the 'variable_names' list.")
    # Check rounding type 
    if not isinstance(rounding, (int, dict, type(None))):
        raise TypeError("'rounding' must be a dictionary, integer or None.")
    ##------------------------------------------------------------------------.   
    # If a dictionary, check valid keys and valid compressor
    if isinstance(rounding, dict):
        if not np.all(np.isin(list(rounding.keys()), variable_names)):
            raise ValueError("The 'rounding' dictionary must contain the keys {}.".format(variable_names))
        if not all([isinstance(v, (int, type(None))) for v in rounding.values()]):
            raise ValueError("The rounding decimals specified in the 'rounding' dictionary must be integers (or None).")
        if any([v < 0 for v in rounding.values() if v is not None]):
            raise ValueError("The rounding decimals specified in the 'rounding' dictionary must be positive integers (or None).") 
    ##------------------------------------------------------------------------.
    # If a unique compressor, create a dictionary with the same compressor for all variables
    if isinstance(rounding, int):
        if rounding < 0: 
            raise ValueError("'rounding' decimal value must be larger than 0.")
    ##------------------------------------------------------------------------.    
    return rounding

#-----------------------------------------------------------------------------.
def rechunk_Dataset(ds, chunks, target_store, temp_store, max_mem = '1GB'):
    """
    Rechunk on disk a xarray Dataset read lazily from a zarr store.

    Parameters
    ----------
    ds : xarray.Dataset
        A Dataset opened with open_zarr().
    chunks : dict
        Custom chunks of the new Dataset.
        If not specified for each Dataset variable, implicitly assumed.
    target_store : str
        Filepath of the zarr store where to save the new Dataset.
    temp_store : str
        Filepath of a zarr store where to save temporary data.
        This store is removed at the end of the rechunking operation. 
    max_mem : str, optional
        The amount of memory (in bytes) that workers are allowed to use.
        The default is '1GB'.

    Returns
    -------
    None.

    """
    ##------------------------------------------------------------------------.
    # Retrieve variables
    variable_names = list(ds.data_vars.keys())
    # Check chunks 
    target_chunks = check_chunks(chunks=chunks, default_chunks=None, variable_names=variable_names) 
    ##------------------------------------------------------------------------.
    # Change chunk value '-1' to length of the dimension 
    # - rechunk and zarr do not currently support -1 specification used by dask and xarray 
    dict_dims = dict(ds.dims)
    for var in target_chunks.keys():
        if target_chunks[var] is not None: 
            for k, v in target_chunks[var].items():
                if v == -1: 
                    target_chunks[var][k] = dict_dims[k]   
                    
    ##------------------------------------------------------------------------.
    # Plan rechunking                
    r = rechunk(ds, 
                target_chunks=target_chunks, 
                max_mem=max_mem,
                target_store=target_store, temp_store=temp_store)
    
    ##------------------------------------------------------------------------.
    # Execute rechunking
    with ProgressBar():
        r.execute()
        
    ##------------------------------------------------------------------------.    
    # Remove temporary store 
    shutil.rmtree(temp_store)
    ##------------------------------------------------------------------------.
 
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
    """AutoregressivePredictions."""
    ##------------------------------------------------------------------------.
    # Work only if output_k are not replicated and are already time-ordered  !
    ##------------------------------------------------------------------------.
    ## Checks arguments 
    device = check_torch_device(device)
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
    
    t_res_timedelta = np.diff(da_dynamic.time.values)[0]    
    t_res_timedelta.astype(get_timedelta_types()[timedelta_unit])    

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
            
            forecast_reference_time = forecast_time_info["forecast_reference_time"]
            
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
            leadtime_idx = np.arange(Y_forecasts.shape[1])
            leadtime = leadtime_idx * t_res_timedelta   
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
    print("- Forecast generation: {:.0f}s".format(time.time()-t_i))
    ##------------------------------------------------------------------------.    
    return ds_forecasts  
 
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
