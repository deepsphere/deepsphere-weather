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
import dask
import numpy as np
import xarray as xr
 
from modules.dataloader_autoregressive import remove_unused_Y
from modules.dataloader_autoregressive import get_aligned_ar_batch
from modules.dataloader_autoregressive import AutoregressiveDataset
from modules.dataloader_autoregressive import AutoregressiveDataLoader
from modules.utils_autoregressive import check_ar_settings
from modules.utils_autoregressive import check_input_k
from modules.utils_autoregressive import check_output_k 
from modules.utils_io import _get_feature_order, check_timesteps_format, check_no_duplicate_timesteps
from modules.utils_zarr import check_chunks
from modules.utils_zarr import check_rounding
from modules.utils_zarr import rechunk_Dataset
from modules.utils_zarr import write_zarr
from modules.utils_torch import check_device
from modules.utils_torch import check_pin_memory
from modules.utils_torch import check_asyncronous_gpu_transfer
from modules.utils_torch import check_prefetch_in_gpu
from modules.utils_torch import check_prefetch_factor

from modules.utils_swag import bn_update

##----------------------------------------------------------------------------.
# conda install -c conda-forge zarr
# conda install -c conda-forge cfgrib
# conda install -c conda-forge rechunker

#----------------------------------------------------------------------------.
###############
### Checks ####
###############
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
#########################
### Prediction utils ####
#########################
def get_dict_Y_pred_selection(dim_info,
                              dict_forecast_rel_idx_Y,
                              keep_first_prediction = True): 
    # dict_forecast_rel_idx_Y = get_dict_Y(ar_iterations = ar_iterations,
    #                                      forecast_cycle = forecast_cycle, 
    #                                      output_k = output_k)                                      
    # Retrieve the time dimension index in the predicted Y tensors
    time_dim = dim_info['time']
    # Retrieve AR iterations 
    ar_iterations = max(list(dict_forecast_rel_idx_Y.keys()))
    # Initialize a general subset indexing
    all_subset_indexing = [slice(None) for i in range(len(dim_info))]
    # Retrieve all output k 
    all_output_k = np.unique(np.stack(list(dict_forecast_rel_idx_Y.values())).flatten())
    # For each output k, search which AR iterations predict such output k
    # - {output_k: [ar_iteration with output_k]}
    dict_k_occurence = {k: [] for k in all_output_k}
    for ar_iteration, leadtimes in dict_forecast_rel_idx_Y.items():
        for leadtime in leadtimes:
            dict_k_occurence[leadtime].append(ar_iteration)
    # For each output k, choose if keep the first or last prediction
    if keep_first_prediction:
        dict_k_selection = {leadtime: min(dict_k_occurence[leadtime]) for leadtime in dict_k_occurence.keys()}
    else: 
        dict_k_selection = {leadtime: max(dict_k_occurence[leadtime]) for leadtime in dict_k_occurence.keys()}
    # Build {ar_iteration: [(leadtime, subset_indexing), (...,...)]}
    dict_Y_pred_selection = {ar_iteration: [] for ar_iteration in range(ar_iterations + 1)}
    for leadtime, ar_iteration in dict_k_selection.items():
        # Retrieve tuple (leadtime, Y_tensor_indexing)
        leadtime_slice_idx = np.argwhere(dict_forecast_rel_idx_Y[ar_iteration] == leadtime)[0][0]                 
        subset_indexing = all_subset_indexing.copy()
        subset_indexing[time_dim] = leadtime_slice_idx
        dict_Y_pred_selection[ar_iteration].append((leadtime, subset_indexing))
    return dict_Y_pred_selection 

def create_ds_forecast(dict_Y_predicted_per_leadtime, 
                       forecast_reference_times,
                       leadtimes,
                       data_dynamic, 
                       dim_info_dynamic):
    """Create the forecast xarray Dataset stacking the tensors in dict_Y_predicted_per_leadtime.""" 
    # Stack forecast leadtimes 
    list_to_stack = [] 
    available_leadtimes = list(dict_Y_predicted_per_leadtime.keys()) 
    for leadtime in available_leadtimes:
        # Append the tensor slice to the list 
        list_to_stack.append(dict_Y_predicted_per_leadtime[leadtime])
        # - Remove tensor from dictionary 
        del dict_Y_predicted_per_leadtime[leadtime]
    Y_forecasts = np.stack(list_to_stack, axis=dim_info_dynamic['time'])  
    ##----------------------------------------------------------------.
    ### Create xarray Dataset of forecasts
    # - Retrieve ancient optional dimensions (to add)
    dims = list(data_dynamic.dims)
    dims_optional = np.array(dims)[np.isin(dims, ['time','feature'], invert=True)].tolist()
    # - Retrieve features 
    features = _get_feature_order(data_dynamic)
    # - Create DataArray
    forecast_dims = ['forecast_reference_time', 'leadtime'] + dims_optional + ['feature']
    da = xr.DataArray(Y_forecasts,           
                      dims = forecast_dims,
                      coords = {'leadtime': leadtimes, 
                                'forecast_reference_time': forecast_reference_times, 
                                'feature': features})
    # - Transform to dataset (to save to zarr)
    ds = da.to_dataset(dim='feature')
    ## - Add ancient coordinates 
    coords = list(data_dynamic.coords.keys()) 
    dict_coords = {coord: data_dynamic[coord] for coord in coords}
    _ = dict_coords.pop("time", None)
    _ = dict_coords.pop("feature", None)
    for k, v in dict_coords.items():
        ds[k] = v
        ds = ds.set_coords(k)

    ##----------------------------------------------------------------.
    # Return the forecast xr.Dataset
    return ds

def rescale_forecasts(ds, scaler, reconcat=True):
    """Apply the scaler inverse transform to the forecast xarray Dataset."""
    # - Apply scaler 
    # --> scaler.inverse_transform(ds).compute() works only for GlobalScalers
    # --> Need to create the time dimension to apply correctly TemporalScalers
    ds['time'] = ds['leadtime'] + ds['forecast_reference_time']
    ds = ds.set_coords('time')
    l_rescaled_ds = []
    # - Iterate over each forecast 
    for i in range(len(ds['forecast_reference_time'])):
        tmp_ds = ds.isel(forecast_reference_time=i).swap_dims({"leadtime": "time"})
        l_rescaled_ds.append(scaler.inverse_transform(tmp_ds).swap_dims({"time": "leadtime"}).drop('time'))
    if reconcat is True:               
        ds = xr.concat(l_rescaled_ds, dim='forecast_reference_time')
        return ds 
    else: 
        return l_rescaled_ds


def rescale_forecasts_and_write_zarr(ds, scaler, zarr_fpath,
                                     chunks = None, default_chunks = None, 
                                     compressor = None, default_compressor = None,
                                     rounding = None, 
                                     consolidated = True, 
                                     append = True,
                                     append_dim = 'forecast_reference_time', 
                                     show_progress = False):
    """Apply the scaler inverse transform to the forecast Dataset and write it to Zarr."""
    # It apply the scaler to each single forecast_reference_time and write it directly to disk.
    ds['time'] = ds['leadtime'] + ds['forecast_reference_time']
    ds = ds.set_coords('time')
    # - Iterate over each forecast 
    l_ds = []
    for i in range(len(ds['forecast_reference_time'])):
        ds_tmp = ds.isel(forecast_reference_time=i).swap_dims({"leadtime": "time"})
        ds_tmp = scaler.inverse_transform(ds_tmp).swap_dims({"time": "leadtime"}).drop('time').expand_dims('forecast_reference_time') 
        ## Writing each separate forecast_reference_time is much slow
        # write_zarr(zarr_fpath = zarr_fpath, 
        #            ds = ds_tmp,
        #            chunks = chunks, default_chunks = default_chunks, 
        #            compressor = compressor, default_compressor = default_compressor,
        #            rounding = rounding, 
        #            consolidated = consolidated, 
        #            append = append,
        #            append_dim = append_dim, 
        #            show_progress = show_progress)
        l_ds.append(ds_tmp)
    ds = xr.concat(l_ds, dim="forecast_reference_time")
    write_zarr(zarr_fpath = zarr_fpath, 
               ds = ds,
               chunks = chunks, default_chunks = default_chunks, 
               compressor = compressor, default_compressor = default_compressor,
               rounding = rounding, 
               consolidated = consolidated, 
               append = append,
               append_dim = append_dim, 
               show_progress = show_progress)    
    return None

#-----------------------------------------------------------------------------.
############################
### Prediction Wrappers ####
############################
def AutoregressivePredictions(model, 
                              # Data
                              data_dynamic,
                              data_static = None,              
                              data_bc = None, 
                              bc_generator = None, 
                              # AR_batching_function
                              ar_batch_fun = get_aligned_ar_batch,
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
                              # Save options 
                              zarr_fpath = None, 
                              rounding = None,
                              compressor = "auto",
                              chunks = "auto"):
    """Wrapper to generate weather forecasts following CDS Common Data Model (CDM).
    
    CDS coordinate             dtype               Synonims
    -------------------------------------------------------------------------
    - realization              int64  
    - forecast_reference_time  datetime64[ns]      (base time)
    - leadtime                 timedelta64[ns]    
    - lat                      float64
    - lon                      float64
    - time                     datetime64[ns]      (forecasted_time/valid_time)
    
    To convert to ECMWF Common Data Model use the following code:
    import cf2cdm
    cf2cdm.translate_coords(ds_forecasts, cf2cdm.ECMWF)
 
    Terminology
    - Forecasts reference time: The time of the analysis from which the forecast was made
    - (Validity) Time: The time represented by the forecast
    - Leadtime: The time interval between the forecast reference time and the (validity) time. 
    
    Coordinates notes:
    - output_k = 0 correspond to the first forecast leadtime 
    - leadtime = 0 is not forecasted. It correspond to the analysis forecast_reference_time 
    - In the ECMWF CMD, forecast_reference_time is termed 'time', 'time' termed 'valid_time'!
    
    Prediction settings 
    - ar_blocks = None (or ar_blocks = ar_iterations + 1) run all ar_iterations in a single run.
    - ar_blocks < ar_iterations + 1:  run ar_iterations per ar_block of ar_iteration
    """
    # Possible speed up: rescale only after all batch have been processed ... 
    ##------------------------------------------------------------------------.
    with dask.config.set(scheduler='synchronous'):
        ## Checks arguments 
        device = check_device(device)
        pin_memory = check_pin_memory(pin_memory=pin_memory, num_workers=num_workers, device=device)  
        asyncronous_gpu_transfer = check_asyncronous_gpu_transfer(asyncronous_gpu_transfer=asyncronous_gpu_transfer, device=device) 
        prefetch_in_gpu = check_prefetch_in_gpu(prefetch_in_gpu=prefetch_in_gpu, num_workers=num_workers, device=device) 
        prefetch_factor = check_prefetch_factor(prefetch_factor=prefetch_factor, num_workers=num_workers)
        ##------------------------------------------------------------------------.
        # Check that autoregressive settings are valid 
        # - input_k and output_k must be numpy arrays hereafter ! 
        input_k = check_input_k(input_k=input_k, ar_iterations=ar_iterations)   
        output_k = check_output_k(output_k = output_k)
        check_ar_settings(input_k = input_k,
                          output_k = output_k,
                          forecast_cycle = forecast_cycle,                           
                          ar_iterations = ar_iterations, 
                          stack_most_recent_prediction = stack_most_recent_prediction)
        ar_iterations = int(ar_iterations)
        ##------------------------------------------------------------------------.   
        ### Retrieve feature info of the forecast 
        features = _get_feature_order(data_dynamic)
        
        ##------------------------------------------------------------------------.
        # Check Zarr settings 
        WRITE_TO_ZARR = zarr_fpath is not None 
        if WRITE_TO_ZARR:
            # - If zarr fpath provided, create the required folder  
            if not os.path.exists(os.path.dirname(zarr_fpath)):
                os.makedirs(os.path.dirname(zarr_fpath))
            if os.path.exists(zarr_fpath):
                raise ValueError("An {} store already exists.")
            # - Set default chunks and compressors 
            # ---> -1 to all optional dimensions (i..e nodes, lat, lon, ens, plevels,...)
            dims = list(data_dynamic.dims)
            dims_optional = np.array(dims)[np.isin(dims, ['time','feature'], invert=True)].tolist()
            default_chunks = {dim : -1 for dim in dims_optional}
            default_chunks['forecast_reference_time'] = 1
            default_chunks['leadtime'] = 1
            default_compressor = zarr.Blosc(cname="zstd", clevel=0, shuffle=2)
            # - Check rounding settings
            rounding = check_rounding(rounding = rounding,
                                      variable_names = features)
        ##------------------------------------------------------------------------.
        # Check ar_blocks 
        if not isinstance(ar_blocks, (int, float, type(None))):
            raise TypeError("'ar_blocks' must be int or None.")
        if isinstance(ar_blocks, float):
            ar_blocks = int(ar_blocks)
        if not WRITE_TO_ZARR and isinstance(ar_blocks, int):
            raise ValueError("If 'zarr_fpath' not specified, 'ar_blocks' must be None.")
        if ar_blocks is None: 
            ar_blocks = ar_iterations + 1
        if ar_blocks > ar_iterations + 1:
            raise ValueError("'ar_blocks' must be equal or smaller to 'ar_iterations'")
        PREDICT_AR_BLOCKS = ar_blocks != (ar_iterations + 1)
    
        ##------------------------------------------------------------------------. 
        ### Define DataLoader subset_timesteps 
        subset_timesteps = None 
        if forecast_reference_times is not None:
            # Check forecast_reference_times
            forecast_reference_times = check_timesteps_format(forecast_reference_times)
            if len(forecast_reference_times) == 0: 
                raise ValueError("If you don't want to specify specific 'forecast_reference_times', set it to None")
            check_no_duplicate_timesteps(forecast_reference_times, var_name='forecast_reference_times')
            # Ensure the temporal order of forecast_reference_times  
            forecast_reference_times.sort() 
            # Define subset_timesteps (aka idx_k=0 aka first forecasted timestep)
            t_res_timedelta = np.diff(data_dynamic.time.values)[0] 
            subset_timesteps = forecast_reference_times + -1*max(input_k)*t_res_timedelta
            # Redefine batch_size if larger than the number of forecast to generate
            # --> And set num_workers to 0 (only 1 batch to load ...)
            if batch_size >= len(forecast_reference_times):
                batch_size = len(forecast_reference_times)
                num_workers = 0
                
        ##------------------------------------------------------------------------.                                 
        ### Create training Autoregressive Dataset and DataLoader    
        dataset = AutoregressiveDataset(data_dynamic = data_dynamic,  
                                        data_bc = data_bc,
                                        data_static = data_static,
                                        bc_generator = bc_generator, 
                                        scaler = scaler_transform, 
                                        # Dataset options 
                                        subset_timesteps = subset_timesteps, 
                                        training_mode = False, 
                                        # Autoregressive settings  
                                        input_k = input_k,
                                        output_k = output_k,
                                        forecast_cycle = forecast_cycle,                           
                                        ar_iterations = ar_iterations, 
                                        stack_most_recent_prediction = stack_most_recent_prediction, 
                                        # GPU settings 
                                        device = device)
        dataloader = AutoregressiveDataLoader(dataset = dataset, 
                                              batch_size = batch_size, 
                                              drop_last_batch = False, 
                                              shuffle = False,
                                              num_workers = num_workers,
                                              prefetch_factor = prefetch_factor, 
                                              prefetch_in_gpu = prefetch_in_gpu,  
                                              pin_memory = pin_memory,
                                              asyncronous_gpu_transfer = asyncronous_gpu_transfer, 
                                              device = device)
        ##------------------------------------------------------------------------.
        # Retrieve custom ar_batch_fun fuction
        ar_batch_fun = dataset.ar_batch_fun 
    
        assert features == dataset.feature_order['dynamic']
        ### Start forecasting
        # - Initialize 
        t_i = time.time()
        model.to(device) 
        # - Set dropout and batch normalization layers to evaluation mode 
        model.eval()
        list_ds = []
        FIRST_PREDICTION = True
        with torch.set_grad_enabled(False):
            ##--------------------------------------------------------------------.     
            # Iterate along batches     
            dataloader_iter = iter(dataloader)
            num_batches = len(dataloader_iter)
            batch_indices = range(num_batches)            
            for batch_count in batch_indices: 
                batch_dict = next(dataloader_iter)
                t_gen = time.time()
                ##----------------------------------------------------------------.
                ### Retrieve forecast informations 
                dim_info_dynamic = batch_dict['dim_info']['dynamic']
                feature_order_dynamic = batch_dict['feature_order']['dynamic']
                forecast_time_info = batch_dict['forecast_time_info']  
                forecast_reference_times = forecast_time_info["forecast_reference_time"] 
                dict_forecast_leadtime = forecast_time_info["dict_forecast_leadtime"]
                dict_forecast_rel_idx_Y = forecast_time_info["dict_forecast_rel_idx_Y"]
                leadtimes = np.unique(np.stack(list(dict_forecast_leadtime.values())).flatten())
                assert features == feature_order_dynamic
                ##----------------------------------------------------------------.
                ### Retrieve dictionary providing at each AR iteration 
                #   the tensor slice indexing to obtain a "regular" forecasts
                if FIRST_PREDICTION: 
                    dict_Y_pred_selection = get_dict_Y_pred_selection(dim_info = dim_info_dynamic,
                                                                      dict_forecast_rel_idx_Y = dict_forecast_rel_idx_Y,
                                                                      keep_first_prediction = keep_first_prediction)
                    FIRST_PREDICTION = False 
                ##----------------------------------------------------------------.      
                ### Perform autoregressive forecasting
                dict_Y_predicted = {}
                dict_Y_predicted_per_leadtime = {}
                ar_counter_per_block = 0  
                previous_block_ar_iteration = 0 
                for ar_iteration in range(ar_iterations+1):
                    # Retrieve X and Y for current AR iteration
                    # - Torch Y stays in CPU with training_mode=False
                    torch_X, _ = ar_batch_fun(ar_iteration = ar_iteration, 
                                              batch_dict = batch_dict, 
                                              dict_Y_predicted = dict_Y_predicted,
                                              device = device, 
                                              asyncronous_gpu_transfer = asyncronous_gpu_transfer)
                                             
                    ##------------------------------------------------------------.
                    # Forward pass and store output for stacking into next AR iterations
                    dict_Y_predicted[ar_iteration] = model(torch_X)
                    ##------------------------------------------------------------.
                    # Select required tensor slices (along time dimension) for final forecast
                    if len(dict_Y_pred_selection[ar_iteration]) > 0: 
                        for leadtime, subset_indexing in dict_Y_pred_selection[ar_iteration]:
                            dict_Y_predicted_per_leadtime[leadtime] = dict_Y_predicted[ar_iteration][subset_indexing].cpu().numpy()
                    ##------------------------------------------------------------.
                    # Remove unnecessary variables on GPU 
                    remove_unused_Y(ar_iteration = ar_iteration, 
                                    dict_Y_predicted = dict_Y_predicted,
                                    dict_Y_to_remove = batch_dict['dict_Y_to_remove']) 
                    del torch_X
                    ##------------------------------------------------------------.
                    # The following code can be used to verify that no leak of memory occurs 
                    # torch.cuda.synchronize()
                    # print("{}: {:.2f} MB".format(ar_iteration, torch.cuda.memory_allocated()/1000/1000)) 
                    
                    ##------------------------------------------------------------.
                    # Create and save a forecast Dataset after each ar_block ar_iterations 
                    ar_counter_per_block += 1 
                    if ar_counter_per_block == ar_blocks:
                        block_slice = slice(previous_block_ar_iteration, ar_iteration+1)
                        ds = create_ds_forecast(dict_Y_predicted_per_leadtime = dict_Y_predicted_per_leadtime,
                                                leadtimes = leadtimes[block_slice],
                                                forecast_reference_times = forecast_reference_times,
                                                data_dynamic = data_dynamic, 
                                                dim_info_dynamic = dim_info_dynamic) 
                                            
                        # Reset ar_counter_per_block
                        ar_counter_per_block = 0 
                        previous_block_ar_iteration = ar_iteration + 1
                        # --------------------------------------------------------.
                        # If predicting blocks of ar_iterations 
                        # - Write AR blocks temporary to disk (and append progressively)
                        if PREDICT_AR_BLOCKS: # (WRITE_TO_ZARR=True implicit)
                            tmp_ar_block_zarr_fpath = os.path.join(os.path.dirname(zarr_fpath), "tmp_ar_blocks.zarr")
                            write_zarr(zarr_fpath = tmp_ar_block_zarr_fpath, 
                                       ds = ds,
                                       chunks = chunks, default_chunks = default_chunks, 
                                       compressor = compressor, default_compressor = default_compressor,
                                       rounding = rounding, 
                                       consolidated = True, 
                                       append = True,
                                       append_dim = 'leadtime', 
                                       show_progress = False)   
                        # --------------------------------------------------------.        
                ##--------------------------------------.-------------------------.
                # Clean memory 
                del dict_Y_predicted    
                del dict_Y_predicted_per_leadtime  
                ##----------------------------------------------------------------.
                ### Post-processing 
                t_post = time.time() 
                # - Retransform data to original dimensions (and write to Zarr optionally)   
                if WRITE_TO_ZARR:
                    if PREDICT_AR_BLOCKS:
                        # - Read the temporary ar_blocks saved on disk 
                        ds = xr.open_zarr(tmp_ar_block_zarr_fpath)
                    if scaler_inverse is not None: 
                        # TODO: Here an error occur if chunk forecast_reference_time > 1
                        # --> Applying the inverse scaler means processing each
                        #     forecast_reference_time separately
                        # ---> A solution would be to stack all forecasts together before
                        #      write to disk ... but this would consume memory and time.
                        rescale_forecasts_and_write_zarr(ds = ds,
                                                         scaler = scaler_inverse,
                                                         zarr_fpath = zarr_fpath, 
                                                         chunks = chunks, default_chunks = default_chunks, 
                                                         compressor = compressor, default_compressor = default_compressor,
                                                         rounding = rounding,             
                                                         consolidated = True, 
                                                         append = True,
                                                         append_dim = 'forecast_reference_time', 
                                                         show_progress = False)
                    else: 
                        write_zarr(zarr_fpath = zarr_fpath, 
                                   ds = ds,
                                   chunks = chunks, default_chunks = default_chunks, 
                                   compressor = compressor, default_compressor = default_compressor,
                                   rounding = rounding, 
                                   consolidated = True, 
                                   append = True,
                                   append_dim = 'forecast_reference_time', 
                                   show_progress = False) 
                    if PREDICT_AR_BLOCKS:
                        shutil.rmtree(tmp_ar_block_zarr_fpath)
                        
                else: 
                    if scaler_inverse is not None: 
                        ds = rescale_forecasts(ds=ds, scaler=scaler_inverse, reconcat=True)
                    list_ds.append(ds)  
                #-------------------------------------------------------------------. 
                # Print prediction report 
                tmp_time_gen = round(t_post - t_gen, 1)
                tmp_time_post = round(time.time() - t_post, 1)
                tmp_time_per_forecast = round((tmp_time_gen+tmp_time_post)/batch_size, 3)
                print(" - Batch: {} / {} | Generation: {}s | Writing: {}s |"
                      "Single forecast computation: {}s ".format(batch_count, len(dataloader),
                                                                 tmp_time_gen, tmp_time_post,
                                                                 tmp_time_per_forecast))
            #---------------------------------------------------------------------.
            # Remove the dataloader and dataset to avoid deadlocks
            del batch_dict
            del dataset
            del dataloader
            del dataloader_iter
        
    ##------------------------------------------------------------------------.
    # Re-read the forecast dataset
    if WRITE_TO_ZARR:
        ds_forecasts = xr.open_zarr(zarr_fpath, chunks="auto")
    else: 
        ds_forecasts = xr.merge(list_ds)
    ##------------------------------------------------------------------------.    
    print("- Elapsed time for forecast generation: {:.2f} minutes".format((time.time()-t_i)/60))
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

def rechunk_forecasts_for_verification(ds, target_store, chunks="auto", max_mem = '1GB', force=False):
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
    ##------------------------------------------------------------------------.
    # Check target_store do not exist already
    if os.path.exists(target_store):
        if force: 
            shutil.rmtree(target_store)
        else:
            raise ValueError("A zarr store already exists at {}. If you want to overwrite, specify force=True".format(target_store))
    ##------------------------------------------------------------------------.
    # Define temp store for rechunking
    temp_store = os.path.join(os.path.dirname(target_store), "tmp_store.zarr")
    # Define intermediate store for rechunked data
    intermediate_store = os.path.join(os.path.dirname(target_store), "rechunked_store.zarr")

    ##------------------------------------------------------------------------.
    # Remove temp_store and intermediate_store is exists 
    if os.path.exists(temp_store):
        shutil.rmtree(temp_store)
    if os.path.exists(intermediate_store):
        shutil.rmtree(intermediate_store) 
    ##------------------------------------------------------------------------.
    # Default chunking
    # - Do not chunk along forecast_reference_time, chunk 1 to all other dimensions
    dims = list(ds.dims)
    dims_optional = np.array(dims)[np.isin(dims, ['time','feature'], invert=True)].tolist()
    default_chunks = {dim : 1 for dim in dims_optional}
    default_chunks['forecast_reference_time'] = -1
    default_chunks['leadtime'] = 1
    # Check chunking
    chunks = check_chunks(ds=ds, chunks=chunks, default_chunks=default_chunks) 
    ##------------------------------------------------------------------------.
    # Rechunk Dataset (on disk)
    rechunk_Dataset(ds=ds, chunks=chunks, 
                    target_store=intermediate_store, temp_store=temp_store, 
                    max_mem = max_mem,
                    force=force)
    ##------------------------------------------------------------------------.
    # Load rechunked dataset (contiguous over forecast referece time, chunked over space)
    ds = xr.open_zarr(intermediate_store, chunks="auto")
    ##------------------------------------------------------------------------.
    # Reshape 
    ds_verification = reshape_forecasts_for_verification(ds)
    ##------------------------------------------------------------------------.
    # Remove 'chunks' key in encoding (bug in xarray-dask-zarr)
    for var in list(ds_verification.data_vars.keys()):
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

#----------------------------------------------------------------------------.
#----------------------------------------------------------------------------.

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