#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 00:04:12 2021

@author: ghiggi
"""
import time
import torch
import random
import numpy as np
from functools import partial 
from tabulate import tabulate 
from torch.utils.data import Dataset, DataLoader

from modules.utils_autoregressive import get_dict_stack_info
from modules.utils_autoregressive import get_first_valid_idx
from modules.utils_autoregressive import get_last_valid_idx
from modules.utils_autoregressive import get_dict_Y
from modules.utils_autoregressive import get_dict_X_dynamic
from modules.utils_autoregressive import get_dict_X_bc    
from modules.utils_autoregressive import check_input_k
from modules.utils_autoregressive import check_output_k 
from modules.utils_io import is_dask_DataArray
from modules.utils_io import check_AR_DataArrays
from modules.utils_io import _check_timesteps
from modules.utils_io import _get_subset_timesteps_idxs
from modules.utils_torch import set_seeds
from modules.utils_torch import check_device
from modules.utils_torch import check_pin_memory
from modules.utils_torch import check_asyncronous_gpu_transfer
from modules.utils_torch import check_prefetch_in_gpu
from modules.utils_torch import check_prefetch_factor
from modules.utils_torch import get_time_function

##----------------------------------------------------------------------------.
### Assumptions 
# - Currently da_dynamic and da_bc must have the same number of timesteps 
# --> The dataloader loads the same positional indices along the time dimension
# - The code inside collate_fn is not thread parallelized, but the code could be accelerated
#   by parallelizing per data type and per forecast iterations 

#-----------------------------------------------------------------------------.  
def _cyclic(iterable):
    while True:
        for x in iterable:
            yield x
            
def cylic_iterator(iterable):
    """Make an iterable a cyclic iterator."""
    return iter(_cyclic(iterable))

def worker_init_fn(worker_id):
    """Function to initialize the seed of the DataLoader workers."""
    initial_seed = int(torch.initial_seed() % (2**32-1))
    worker_seed = initial_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    # set_seeds(worker_seed)
    return None

#-----------------------------------------------------------------------------.       
# ############################# 
### Autoregressive Dataset ####
# #############################   
class AutoregressiveDataset(Dataset):
    """Map-style autoregressive pytorch dataset."""
    
    ## -----------------------------------------------------------------------.
    def __init__(self,
                 da_dynamic, 
                 # Autoregressive settings  
                 input_k, 
                 output_k,
                 forecast_cycle,                           
                 AR_iterations, 
                 stack_most_recent_prediction, 
                 # Facultative input data
                 da_bc = None, 
                 da_static = None,
                 scaler = None, 
                 # Setting for optional time subsets
                 subset_timesteps = None, 
                 training_mode = True, 
                 # GPU settings 
                 device = 'cpu'):    
        """
        "Create the Dataset required to generate an AutoregressiveDataloader.

        Parameters
        ----------
        da_dynamic : DataArray
            DataArray with dynamic data.
        da_bc : DataArray, optional
            DataArray with boundary conditions features.
            The default is None.
        da_static : DataArray, optional
            DataArray with static features.  
            The default is None.
        scaler : xscaler 
            xscaler object to transform the DataArrays.
            The default is None.
        subset_timesteps : np.array with datetime
            Allows to restrict the timesteps that the DataLoader will load.
        training_mode : bool
            When training_mode = True (default), the dataloader loads also the ground truth Y. 
        input_k : list
            Indices representing predictors past timesteps.
        output_k : list
            Indices representing forecasted timesteps. Must include 0.
        forecast_cycle : int
            Indicates the lag between forecasts.
        AR_iterations : int
            Number of AR iterations.
        stack_most_recent_prediction : bool
            Whether to use the most recent prediction when autoregressing.
        device : torch.device, optional
            Device on which to train the model. The default is 'cpu'.
        """        
        ##--------------------------------------------------------------------.
        # Check input_k and output_k type
        input_k = check_input_k(input_k=input_k, AR_iterations=AR_iterations)   
        output_k = check_output_k(output_k=output_k)
        # Check DataArrays  
        check_AR_DataArrays(da_training_dynamic = da_dynamic,
                            da_training_bc = da_bc,
                            da_static = da_static) 
        # Checks device
        device = check_device(device)
        self.device = device
        # Check training_mode 
        if not isinstance(training_mode, bool): 
            raise TypeError("'training_mode must be either True or False.")
        self.training_mode = training_mode 
        ## -------------------------------------------------------------------.
        ### - Initialize autoregressive configs   
        self.input_k = input_k 
        self.output_k = output_k
        self.forecast_cycle = forecast_cycle
        self.AR_iterations = AR_iterations
        self.stack_most_recent_prediction = stack_most_recent_prediction
        ##--------------------------------------------------------------------.
        ### Initialize scaler 
        self.scaler = scaler 
        ##--------------------------------------------------------------------.
        ### - Define dimension positions 
        # - Sample/batch dimension is fixed to be the first ! 
        # - The others can vary
        dims_dynamic = da_dynamic.dims
        dim_info = {}
        dim_info['sample'] = 0 # Here I force batch_dim to be the first dimension (for all code)! 
        dim_info['time'] = np.argwhere(np.array(dims_dynamic) == 'time')[0][0] + 1
        dim_info['node'] = np.argwhere(np.array(dims_dynamic) == 'node')[0][0] + 1 
        dim_info['feature'] = np.argwhere(np.array(dims_dynamic) == 'feature')[0][0] + 1  
        self.dim_info = dim_info 
        ##--------------------------------------------------------------------.
        ### - Define data precision
        self.torch_dtype = torch.get_default_dtype() 
        
        ##--------------------------------------------------------------------.
        ### - Assign dynamic and bc data
        self.da_dynamic = da_dynamic 
        self.da_bc = da_bc 
        ##--------------------------------------------------------------------.
        ### - Build dictionary of DataArray availability 
        data_availability = {}
        data_availability['static'] = da_static is not None
        data_availability['bc'] = da_bc is not None
        self.data_availability = data_availability
        ##--------------------------------------------------------------------.
        ### Load static tensor into CPU (and expand over the time dimension) 
        if data_availability['static']:
            # - Apply scaler 
            if self.scaler is not None:
                da_static = self.scaler.transform(da_static, variable_dim='feature').compute()
            ##----------------------------------------------------------------.
            # - Reshape da_static to match ('node','feature') order of da_dynamic
            required_static_order = np.array(da_dynamic.dims)[np.isin(da_dynamic.dims, da_static.dims)].tolist()
            if not required_static_order == list(da_static.dims):
                print("Reshaping static DataArray to have dimension order: {}".format(required_static_order))
                da_static = da_static.transpose(*required_static_order)    
            ##----------------------------------------------------------------.
            # If da_static still lazy, load data 
            da_static = da_static.compute()
            ##----------------------------------------------------------------.
            ## Add batch and time dimension and then expand along time dimension 
            # - Define ways to unsqueeze the static tensor 
            unsqueeze_time_dim = dim_info['time'] - 1 # (without batch dim ...)
            unsqueeze_batch_dim = dim_info['sample']  
            # - Define the dimensions of the expanded tensor 
            dim_batch = dim_info['sample']  
            dim_time = dim_info['time']  
            new_dim_size = [-1 for i in range(len(da_static.dims) + 2)]
            new_dim_size[dim_batch] = 1            # Batch dimension 
            new_dim_size[dim_time] = len(input_k)  # The (predictor lag) 'time' dimension)
            # - Use a view to expand (to not allocate new memory)
            self.torch_static = torch.tensor(da_static.values, dtype=self.torch_dtype, device='cpu').unsqueeze(unsqueeze_time_dim).unsqueeze(unsqueeze_batch_dim).expand(new_dim_size)
        else: 
            self.torch_static = None
            
        ##--------------------------------------------------------------------.
        ### - Add forecast reference times idxs for tailored forecast predictions 
        # - This it restrics the idxs in update_indexes() that will be loaded
        subset_idxs = None
        if subset_timesteps is not None:
            timesteps = da_dynamic['time'].values
            subset_timesteps = np.array(subset_timesteps)
            subset_idxs = _get_subset_timesteps_idxs(timesteps, subset_timesteps, strict_match=True) 
            self.subset_timesteps = timesteps[subset_idxs]                                        
        self.subset_idxs = subset_idxs
        
        ##--------------------------------------------------------------------.
        ### - Generate indexing
        self.n_timesteps = da_dynamic.shape[0] # subset_idxs refers to range(0, n_timesteps)
        self.update_indexing()
        
        ##--------------------------------------------------------------------.    
   
    ##------------------------------------------------------------------------.
    def __len__(self):
        """Return the number of samples available."""
        return self.n_samples

    ##------------------------------------------------------------------------.
    def __getitem__(self, idx):
        """Return sample and label corresponding to an index as torch.Tensor objects."""
        # rel_idx correspond to input_k and output_k (aka leadtime_idx)
        # TODO allow DataArray and Dataset ... 
        ## -------------------------------------------------------------------.
        # Retrieve current idx of xarray  
        xr_idx_k_0 = self.idxs[idx]
        
        ## -------------------------------------------------------------------.
        ### Retrieve dynamic data 
        # - Retrieve xarray indices 
        xr_idx_dynamic_required = xr_idx_k_0 + self.rel_idx_dynamic_required  
        # - Subset the xarray Datarray (need for all autoregressive iterations)
        da_dynamic_subset = self.da_dynamic.isel(time=xr_idx_dynamic_required)
        
        ## -------------------------------------------------------------------.
        ### Load batch data
        # TODO : from here below ensure that is a DataArray 
        # - If not preloaded in CPU, load the zarr chunks  
        if is_dask_DataArray(da_dynamic_subset): 
            da_dynamic_subset = da_dynamic_subset.compute()
        # - Apply the scaler if provided 
        if self.scaler is not None:
            da_dynamic_subset = self.scaler.transform(da_dynamic_subset, variable_dim='feature').compute()
        # - Assign relative indices (onto the "rel_idx" dimension)
        da_dynamic_subset = da_dynamic_subset.assign_coords(rel_idx=('time', self.rel_idx_dynamic_required)).swap_dims({'time': 'rel_idx'}) 
        ## -------------------------------------------------------------------.
        ### Loop over leadtimes and store Numpy arrays in a dictionary(leadtime)
        # - Extract numpy array from DataArray and convert to Torch Tensor
        # - X_dynamic 
        dict_X_dynamic_data = {}
        for i in range(self.AR_iterations + 1): 
            # X_dynamic
            if self.dict_rel_idx_X_dynamic[i] is not None:
                dict_X_dynamic_data[i] = torch.as_tensor(torch.from_numpy(da_dynamic_subset.sel(rel_idx=self.dict_rel_idx_X_dynamic[i]).values), dtype=self.torch_dtype, device='cpu')
            else: 
                dict_X_dynamic_data[i] = None
        # - Y 
        if self.training_mode:
            dict_Y_data = {}
            for i in range(self.AR_iterations + 1): 
                dict_Y_data[i] = torch.as_tensor(torch.from_numpy(da_dynamic_subset.sel(rel_idx=self.dict_rel_idx_Y[i]).values), dtype=self.torch_dtype, device='cpu')
        else: 
            dict_Y_data = None
        ##--------------------------------------------------------------------.
        ## Retrieve forecast time infos     
        # - Forecast start time 
        forecast_start_time = self.da_dynamic.isel(time=xr_idx_k_0).time.values
        # - Forecast reference time 
        reference_time_idx = xr_idx_k_0 + max(self.input_k)
        forecast_reference_time = self.da_dynamic.isel(time=reference_time_idx).time.values
        # - Forecast leadtime_idx 
        dict_forecast_rel_idx_Y = self.dict_rel_idx_Y
        # - Forecasted time and forecast leadtime 
        dict_forecasted_time = {}
        dict_forecast_leadtime = {}
        for i in range(self.AR_iterations + 1): 
            dict_forecasted_time[i] = da_dynamic_subset.sel(rel_idx=self.dict_rel_idx_Y[i]).time.values 
            dict_forecast_leadtime[i] = dict_forecasted_time[i] - forecast_reference_time
        # - Create forecast_time_info dictionary 
        forecast_time_info = {'forecast_start_time': forecast_start_time,
                              'forecast_reference_time': forecast_reference_time,
                              'dict_forecast_rel_idx_Y': dict_forecast_rel_idx_Y,
                              'dict_forecast_leadtime': dict_forecast_leadtime, 
                              'dict_forecasted_time': dict_forecasted_time}
         
        ## -------------------------------------------------------------------.
        ### Retrieve boundary conditions data (if provided)
        if self.data_availability['bc']: 
            xr_idx_bc_required = xr_idx_k_0 + self.rel_idx_bc_required  
            # - Subset the xarray Datarray (need for all autoregressive iterations)
            da_bc_subset = self.da_bc.isel(time=xr_idx_bc_required)
            # - Apply scaler 
            if self.scaler is not None:
                da_bc_subset = self.scaler.transform(da_bc_subset, variable_dim='feature').compute()
            # - Assign relative indices (onto the "rel_idx" dimension)
            da_bc_subset = da_bc_subset.assign_coords(rel_idx=('time', self.rel_idx_bc_required)).swap_dims({'time': 'rel_idx'}) 
            # - If not preloaded in CPU, read from disk the zarr chunks (with dask)  
            if is_dask_DataArray(da_bc_subset): 
                da_bc_subset = da_bc_subset.compute() 
            # - Loop over leadtimes and store Numpy arrays in a dictionary(leadtime) 
            dict_X_bc_data = {}
            for i in range(self.AR_iterations + 1): 
                # Extract numpy array from DataArray and conver to Torch Tensor
                dict_X_bc_data[i] = torch.as_tensor(torch.from_numpy(da_bc_subset.sel(rel_idx=self.dict_rel_idx_X_bc[i]).values), dtype=self.torch_dtype, device='cpu')
        else: 
            dict_X_bc_data = None 
        
        ## -------------------------------------------------------------------.
        # Return the sample dictionary  
        return {'X_dynamic': dict_X_dynamic_data, 
                'X_bc': dict_X_bc_data, 
                'Y': dict_Y_data, 
                'dict_Y_to_stack': self.dict_Y_to_stack,
                'dict_Y_to_remove': self.dict_Y_to_remove,
                'AR_iterations': self.AR_iterations,
                'dim_info': self.dim_info,
                'forecast_time_info': forecast_time_info,
                'training_mode': self.training_mode, 
                'data_availability': self.data_availability
                }
    
    def update_indexing(self):
        """Update indices."""
        input_k = self.input_k 
        output_k = self.output_k
        forecast_cycle = self.forecast_cycle
        AR_iterations = self.AR_iterations
        stack_most_recent_prediction = self.stack_most_recent_prediction
        n_timesteps = self.n_timesteps
        ##--------------------------------------------------------------------.
        ## Update dictionary Y to stack and remove 
        dict_Y_to_stack, dict_Y_to_remove = get_dict_stack_info(AR_iterations = AR_iterations, 
                                                                forecast_cycle = forecast_cycle, 
                                                                input_k = input_k, 
                                                                output_k = output_k, 
                                                                stack_most_recent_prediction = stack_most_recent_prediction)
        self.dict_Y_to_stack = dict_Y_to_stack
        self.dict_Y_to_remove = dict_Y_to_remove
        ##--------------------------------------------------------------------.
        # - Update valid data range indexing 
        idx_start = get_first_valid_idx(input_k)
        idx_end = get_last_valid_idx(output_k = output_k,
                                     forecast_cycle = forecast_cycle, 
                                     AR_iterations = AR_iterations)
        
        # - Define valid idx for training (and prediction)
        subset_idxs = self.subset_idxs
        if subset_idxs is not None:
            subset_timesteps = self.subset_timesteps
            if any(subset_idxs < idx_start):
                raise ValueError("With current AR settings, the following 'subset_timesteps' are not allowed:",
                                 list(subset_timesteps[subset_idxs < idx_start]))
            if any(subset_idxs > n_timesteps - idx_end):
                raise ValueError("With current AR settings, the following 'subset_timesteps' are not allowed:",
                                 list(subset_timesteps[subset_idxs > idx_end]))            
            self.idxs = subset_idxs
        else: 
            self.idxs = np.arange(n_timesteps)[idx_start:(-1*idx_end - 1)]
       
        # - Compute the number of samples available
        self.n_samples = len(self.idxs)
        if self.n_samples == 0: 
            raise ValueError("No samples available. Maybe reduce number of AR iterations.")
        ##--------------------------------------------------------------------.
        ### - Update dictionary with indexing information for autoregressive training
        self.dict_rel_idx_Y = get_dict_Y(AR_iterations = AR_iterations,
                                         forecast_cycle = forecast_cycle, 
                                         output_k = output_k)
        self.dict_rel_idx_X_dynamic = get_dict_X_dynamic(AR_iterations = AR_iterations,
                                                         forecast_cycle = forecast_cycle, 
                                                         input_k = input_k)
        self.dict_rel_idx_X_bc = get_dict_X_bc(AR_iterations = AR_iterations,
                                               forecast_cycle = forecast_cycle,
                                               input_k = input_k)
        
        ##--------------------------------------------------------------------.
        ### - Based on the current value of AR_iterations, create a
        #     list of (relative) indices required to load data from da_dynamic and da_bc 
        #   --> This indices are updated when Dataset.update_AR_iterations() is called
        rel_idx_X_dynamic_required = np.unique(np.concatenate([x for x in self.dict_rel_idx_X_dynamic.values() if x is not None]))
        rel_idx_Y_dynamic_required = np.unique(np.concatenate([x for x in self.dict_rel_idx_Y.values() if x is not None]))
        self.rel_idx_dynamic_required = np.unique(np.concatenate((rel_idx_X_dynamic_required, rel_idx_Y_dynamic_required)))
        
        if self.da_bc is not None:
            self.rel_idx_bc_required = np.unique(np.concatenate([x for x in self.dict_rel_idx_X_bc.values() if x is not None]))
        else: 
            self.rel_idx_bc_required = None
            
        ##--------------------------------------------------------------------.
            
    def update_AR_iterations(self, new_AR_iterations):  
        """Update Dataset informations.
        
        If the number of forecast iterations changes, the function update
        the relative indices in order to retrieve only the needed amount of data 
        The changes to the Dataset implicitly affect the next DataLoader call!
        """
        if self.AR_iterations != new_AR_iterations:                
            # Update AR iterations
            self.AR_iterations = new_AR_iterations
            # Update valid idxs and rel_idx_dictionaries 
            self.update_indexing()
                
##----------------------------------------------------------------------------.    
def autoregressive_collate_fn(list_samples,  
                              torch_static=None, 
                              pin_memory = False,
                              prefetch_in_gpu = False,
                              asyncronous_gpu_transfer=True, 
                              device = 'cpu'):        
    """Stack the list of samples into batch of data."""
    # list_samples is a list of what returned by __get_item__ of AutoregressiveDataset
    # To debug: list_samples = [dataset.__getitem__(0), dataset.__getitem__(1)]
    ##------------------------------------------------------------------------.
    # Retrieve other infos
    dict_Y_to_stack = list_samples[0]['dict_Y_to_stack']
    dict_Y_to_remove = list_samples[0]['dict_Y_to_remove']
    dim_info = list_samples[0]['dim_info']
    AR_iterations = list_samples[0]['AR_iterations']
    batch_dim = dim_info['sample']
    training_mode = list_samples[0]['training_mode']
    data_availability = list_samples[0]['data_availability']

    ##------------------------------------------------------------------------.
    # Retrieve the different data (and forecast time info)
    list_X_dynamic_samples = []
    list_X_bc_samples = []
    list_Y_samples = []
    
    list_forecast_start_time = []
    list_forecast_reference_time = []
    dict_forecast_leadtime = list_samples[0]['forecast_time_info']['dict_forecast_leadtime']
    dict_forecast_rel_idx_Y = list_samples[0]['forecast_time_info']['dict_forecast_rel_idx_Y']
    for dict_samples in list_samples:
        list_X_dynamic_samples.append(dict_samples['X_dynamic'])
        list_X_bc_samples.append(dict_samples['X_bc'])
        list_Y_samples.append(dict_samples['Y'])
        # Forecast time info 
        list_forecast_start_time.append(dict_samples['forecast_time_info']['forecast_start_time'])
        list_forecast_reference_time.append(dict_samples['forecast_time_info']['forecast_reference_time'])
    
    ##------------------------------------------------------------------------. 
    # Assemble forecast_time_info   
    forecast_reference_time = np.stack(list_forecast_reference_time)
    forecast_start_time = np.stack(list_forecast_start_time)
    forecast_time_info = {"forecast_reference_time": forecast_reference_time,
                          "forecast_start_time": forecast_start_time,
                          "dict_forecast_leadtime": dict_forecast_leadtime,
                          "dict_forecast_rel_idx_Y": dict_forecast_rel_idx_Y}
    
    ##------------------------------------------------------------------------.  
    ### Batch data togethers   
    # - Process X_dynamic  
    dict_X_dynamic_batched = {}
    for i in range(AR_iterations+1):
        # X dynamic 
        list_X_dynamic_tensors = [dict_leadtime[i] for dict_leadtime in list_X_dynamic_samples if dict_leadtime[i] is not None]
        if len(list_X_dynamic_tensors) > 0: 
            if pin_memory:
                dict_X_dynamic_batched[i] = torch.stack(list_X_dynamic_tensors, dim=batch_dim).pin_memory()
            else: 
                dict_X_dynamic_batched[i] = torch.stack(list_X_dynamic_tensors, dim=batch_dim) 
            if prefetch_in_gpu:
                dict_X_dynamic_batched[i] = dict_X_dynamic_batched[i].to(device=device, non_blocking=asyncronous_gpu_transfer)  
        else: # when no X dynamic (after some AR iterations)
            dict_X_dynamic_batched[i] = None
    ##------------------------------------.        
    # - Process Y
    if training_mode:
        dict_Y_batched = {}
        for i in range(AR_iterations+1):
            if pin_memory:
                dict_Y_batched[i] = torch.stack([dict_leadtime[i] for dict_leadtime in list_Y_samples], dim=batch_dim).pin_memory()  
            else: 
                dict_Y_batched[i] = torch.stack([dict_leadtime[i] for dict_leadtime in list_Y_samples], dim=batch_dim)  
            if prefetch_in_gpu:
                dict_Y_batched[i] = dict_Y_batched[i].to(device=device, non_blocking=asyncronous_gpu_transfer)
    else:
        dict_Y_batched = None
        
    ##-------------------------------------.            
    # - Process X_bc
    if data_availability['bc']: 
        dict_X_bc_batched = {}  
        for i in range(AR_iterations+1):
            if len(list_X_bc_samples) != 0 and list_X_bc_samples[0] is not None: 
                if pin_memory:
                    dict_X_bc_batched[i] = torch.stack([dict_leadtime[i] for dict_leadtime in list_X_bc_samples], dim=batch_dim).pin_memory()
                else: 
                    dict_X_bc_batched[i] = torch.stack([dict_leadtime[i] for dict_leadtime in list_X_bc_samples], dim=batch_dim) 
                if prefetch_in_gpu:
                    if dict_X_bc_batched[i] is not None:
                        dict_X_bc_batched[i] = dict_X_bc_batched[i].to(device=device, non_blocking=asyncronous_gpu_transfer)  
            else:
                dict_X_bc_batched[i] = None
    else: 
        dict_X_bc_batched = None
    ##------------------------------------------------------------------------.
    # - Prefetch static to GPU if asked
    if prefetch_in_gpu:            
        if torch_static is not None: 
            torch_static = torch_static.to(device=device, non_blocking=asyncronous_gpu_transfer) 
    
    ##------------------------------------------------------------------------.   
    # Return dictionary of batched data 
    batch_dict = {'X_dynamic': dict_X_dynamic_batched, 
                  'X_bc': dict_X_bc_batched, 
                  'X_static': torch_static,
                  'Y': dict_Y_batched, 
                  'dim_info': dim_info, 
                  'forecast_time_info': forecast_time_info,
                  'dict_Y_to_remove': dict_Y_to_remove,
                  'dict_Y_to_stack': dict_Y_to_stack,
                  'training_mode': training_mode, 
                  'data_availability': data_availability, 
                  'prefetched_in_gpu': prefetch_in_gpu}
    
    return batch_dict
     
#-----------------------------------------------------------------------------.
# ################################
### Autoregressive DataLoader ####
# ################################
def AutoregressiveDataLoader(dataset, 
                             batch_size = 64,  
                             drop_last_batch = True,
                             shuffle = True,
                             shuffle_seed = 69, 
                             num_workers = 0,
                             pin_memory = False,
                             prefetch_in_gpu = False, 
                             prefetch_factor = 2, 
                             asyncronous_gpu_transfer = True, 
                             device = 'cpu',
                             verbose = False):
    """
    Create the DataLoader required for autoregressive model training.

    Parameters
    ----------
    dataset : AutoregressiveDataset
        An AutoregressiveDataset.        
    batch_size : int, optional
        Number of samples within a batch. The default is 64.
    drop_last_batch : bool, optional
        Wheter to drop the last batch_size (with less samples). The default is True.
    shuffle : bool, optional
        Wheter to random shuffle the samples each epoch. The default is True.
    shuffle_seed : int, optional
        Empower deterministic random shuffling.
    num_workers : 0, optional
        Number of processes that generate batches in parallel.
        0 means ONLY the main process will load batches (that can be a bottleneck).
        1 means ONLY one worker (just not the main process) will load data 
        A high enough number of workers usually assures that CPU computations 
        are efficiently managed. However, increasing num_workers increase the 
        CPU memory consumption.
        The Dataloader prefetch into the CPU prefetch_factor*num_workers batches.
        The default is 0.
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
    device: torch.device, optional 
         Only used if 'prefetch_in_gpu' = True.
         Indicates to which GPUs to transfer the data. 
         
    Returns
    -------
    dataloader : AutoregressiveDataLoader
        pytorch DataLoader for autoregressive model training.

    """
    ##------------------------------------------------------------------------.
    ## Checks 
    device = check_device(device)
    pin_memory = check_pin_memory(pin_memory=pin_memory, num_workers=num_workers, device=device)  
    asyncronous_gpu_transfer = check_asyncronous_gpu_transfer(asyncronous_gpu_transfer=asyncronous_gpu_transfer, device=device) 
    prefetch_in_gpu = check_prefetch_in_gpu(prefetch_in_gpu=prefetch_in_gpu, num_workers=num_workers, device=device) 
    prefetch_factor = check_prefetch_factor(prefetch_factor=prefetch_factor, num_workers=num_workers)
    device = check_device(device)
    
    ##------------------------------------------------------------------------. 
    # Retrieve dimension info dictiorary from Dataset 
    # - Provide information of Torch Tensor dimensions.
    dim_info = dataset.dim_info 
    batch_dim = dim_info['sample']
    ##------------------------------------------------------------------------. 
    # Retrieve static feature tensor (preloaded in GPU) (if available)
    torch_static = dataset.torch_static  
     
    ##------------------------------------------------------------------------. 
    # Expand static feature tensor (along the batch dimension) 
    if torch_static is not None:
        new_dim_size = [-1 for i in range(torch_static.dim())]
        new_dim_size[batch_dim] = batch_size
        torch_static = torch_static.expand(new_dim_size)  
    
    ##------------------------------------------------------------------------.    
    # Set seeds for deterministic random shuffling
    if shuffle:
        set_seeds(shuffle_seed) # update np.seed, random.seed and torch.seed
    ##------------------------------------------------------------------------.
    # Create pytorch Dataloader 
    # - Pass torch tensor of static data (to not reload every time)
    # - Data are eventually pinned into memory after data have been stacked into the collate_fn
    dataloader = DataLoader(dataset = dataset, 
                            batch_size = batch_size,  
                            shuffle = shuffle,
                            # sampler = None,  # Option not implemented yet 
                            drop_last = drop_last_batch, 
                            num_workers = num_workers,
                            persistent_workers = False, 
                            prefetch_factor = prefetch_factor, 
                            pin_memory = False,  
                            worker_init_fn = worker_init_fn,
                            collate_fn = partial(autoregressive_collate_fn, 
                                                 torch_static = torch_static,
                                                 pin_memory = pin_memory,
                                                 prefetch_in_gpu = prefetch_in_gpu,
                                                 asyncronous_gpu_transfer = asyncronous_gpu_transfer, 
                                                 device = device)
                            )
    return dataloader

#-----------------------------------------------------------------------------.
# #####################
### AR batch tools ####
# #####################               
def get_AR_batch(AR_iteration, 
                 batch_dict, 
                 dict_Y_predicted,
                 device = 'cpu', 
                 asyncronous_gpu_transfer = True):
    """Create X and Y Torch Tensors for a specific AR iteration."""
    i = AR_iteration
    ##------------------------------------------------------------------------.
    # Get dimension info 
    dim_info = batch_dict['dim_info']
    n_dims = len(dim_info)   
    time_dim = dim_info['time']  
    feature_dim = dim_info['feature']  
    # batch_dim = dim_info['sample']  
    # node_dim = dim_info['node']  
 
    ##------------------------------------------------------------------------.
    ## Get dictionary with batched data for all forecast iterations 
    torch_static = batch_dict['X_static']   # Can be None if no static data specified 
    dict_X_dynamic_batched = batch_dict['X_dynamic']
    dict_X_bc_batched = batch_dict['X_bc']  # Can be None if no bc data specified 
    dict_Y_batched = batch_dict['Y']        # Can be None if training_mode = False 
    prefetched_in_gpu = batch_dict["prefetched_in_gpu"]
    training_mode = batch_dict['training_mode']
    # data_availability = batch_dict['data_availability'] 
    ##------------------------------------------------------------------------.
    # Check if static and bc data are available 
    static_is_available = torch_static is not None
    if dict_X_bc_batched is None: 
        bc_is_available = False 
    else: 
        bc_is_available = dict_X_bc_batched[i] is not None
   
    ##------------------------------------------------------------------------.
    # Retrieve info and data for current iteration 
    list_tuple_idx_to_stack = batch_dict['dict_Y_to_stack'][i]
    
    ##------------------------------------------------------------------------.
    ### Prepare torch Tensor available in GPU 
    # - X_dynamic 
    if dict_X_dynamic_batched[i] is not None:
        if not prefetched_in_gpu:
            torch_X = dict_X_dynamic_batched[i].to(device=device, non_blocking=asyncronous_gpu_transfer)
        else: 
            torch_X = dict_X_dynamic_batched[i]
    else:
        torch_X = None
    ##--------------------------------------------.
    # - X_static
    if torch_static is not None: 
        if not prefetched_in_gpu:
            torch_static = torch_static.to(device=device, non_blocking=asyncronous_gpu_transfer)
    ##--------------------------------------------.
    # - X_bc 
    if bc_is_available:
        if not prefetched_in_gpu:
            torch_X_bc = dict_X_bc_batched[i].to(device=device, non_blocking=asyncronous_gpu_transfer)
        else: 
            torch_X_bc = dict_X_bc_batched[i]
    ##--------------------------------------------.
    # - Y 
    if training_mode:
        if not prefetched_in_gpu:
            torch_Y = dict_Y_batched[i].to(device=device, non_blocking=asyncronous_gpu_transfer)
        else:
            torch_Y = dict_Y_batched[i] 
    else: # prediction mode 
        torch_Y = None         
    ##-------------------------------------------------------------------------.
    # Stack together data required for the current forecast 
    # --> Previous predictions to stack are already in the GPU
    # --> Data need to be already in the GPU (if device is not cpu)!
    if list_tuple_idx_to_stack is not None:
        # The loop below allow to generalize the stacking to whatever time_dim position  
        # list_Y_to_stack = [dict_Y_predicted[ldt][:,idx,...] for ldt, idx in list_tuple_idx_to_stack]
        # --> Maybe one day: torch.isel({dim_name: idx}) à la xarray
        general_index = [slice(None) for i in range(n_dims)]
        list_Y_to_stack = []
        for ldt, idx in list_tuple_idx_to_stack:
            custom_idx = general_index.copy()
            custom_idx[time_dim] = idx
            list_Y_to_stack.append(dict_Y_predicted[ldt][custom_idx])  
 
        torch_X_to_stack = torch.stack(list_Y_to_stack, dim=time_dim)  
        if torch_X is not None:
            torch_X = torch.cat((torch_X, torch_X_to_stack), dim=time_dim)
        else:
            torch_X = torch_X_to_stack
            
    # - Add boundary conditions data (if available)
    if bc_is_available:
        torch_X = torch.cat((torch_X_bc, torch_X), dim=feature_dim) 
       
    # - Combine with the static tensor (which is constantly in the GPU)
    # --> In the batch dimension, match the number of samples of torch X
    if static_is_available: 
        batch_size = torch_X.shape[0]
        torch_X = torch.cat((torch_static[0:batch_size,...], torch_X), dim=feature_dim)
        
    ##------------------------------------------------------------------------.
    # - Remove unused torch Tensors 
    del dict_X_dynamic_batched[i]   
    if training_mode:
        del dict_Y_batched[i]            
    if bc_is_available:
        del torch_X_bc
        del dict_X_bc_batched[i]    
    if static_is_available: 
        del torch_static
    ##------------------------------------------------------------------------.    
    return (torch_X, torch_Y)

##----------------------------------------------------------------------------.
def remove_unused_Y(AR_iteration, dict_Y_predicted, dict_Y_to_remove):
    """Remove unused Y predictions of past AR iterations."""
    list_idx_Y_to_remove = dict_Y_to_remove[AR_iteration]
    if list_idx_Y_to_remove is not None:
        for ldt in list_idx_Y_to_remove: 
            del dict_Y_predicted[ldt]
    return None
    
#-----------------------------------------------------------------------------.
######################
#### Timing utils ####
######################
def timing_AR_DataLoader(dataset,
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
                         sleeping_time = 0.5, 
                         n_repetitions = 10,
                         verbose = True):
    """
    Time execution and memory consumption of AR DataLoader.

    Parameters
    ----------
    dataset : AutoregressiveDataLoader
        AutoregressiveDataLoader
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
        Wheter to random shuffle the samples each epoch. The default is True.
    shuffle_seed : int, optional
        Empower deterministic random shuffling.
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
    sleeping_time : float 
        Sleeping time in seconds after batch creation between AR iterations.
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
    ##------------------------------------------------------------------------.  
    # Retrieve informations 
    AR_iterations = dataset.AR_iterations
    device = dataset.device                      
    # Retrieve function to get time 
    get_time = get_time_function(device)
    ##------------------------------------------------------------------------.
    # Initialize list 
    Dataloader_timing = []  
    AR_batch_timing = []  
    Total_timing = []
    ##------------------------------------------------------------------------.
    # Initialize DataLoader 
    dataloader = AutoregressiveDataLoader(dataset = dataset,                                                   
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
    dataloader_iter = iter(dataloader)
    ##------------------------------------------------------------------------.
    # Repeat training n_repetitions
    for count in range(n_repetitions):  
        # Measure background memory used
        if device.type != 'cpu':
            background_memory = torch.cuda.memory_allocated()/1000/1000
        ##----------------------------------------------------------------. 
        # Retrieve batch
        t_i = get_time()
        training_batch_dict = next(dataloader_iter)
        Dataloader_timing.append(get_time() - t_i)
        # Perform AR iterations 
        dict_training_Y_predicted = {}
        ##----------------------------------------------------------------.
        # Initialize stuff for AR loop timing 
        tmp_AR_data_removal_timing = 0
        tmp_AR_batch_timing = 0 
        batch_memory_size = 0 
        for i in range(AR_iterations+1):
            # Retrieve X and Y for current AR iteration   
            t_i = get_time()
            torch_X, torch_Y = get_AR_batch(AR_iteration = i, 
                                            batch_dict = training_batch_dict, 
                                            dict_Y_predicted = dict_training_Y_predicted,
                                            device = device, 
                                            asyncronous_gpu_transfer = asyncronous_gpu_transfer)
            tmp_AR_batch_timing = tmp_AR_batch_timing + (get_time() - t_i)
            ##------------------------------------------------------------.                                
            # Measure batch size in MB 
            if device.type != 'cpu' and i == 0:
                batch_memory_size = torch.cuda.memory_allocated()/1000/1000 - background_memory
            ##------------------------------------------------------------.
            # Spend some custom time here
            time.sleep(sleeping_time)
            dict_training_Y_predicted[i] = torch_Y.detach().clone()
            ##------------------------------------------------------------.
            # Remove unnecessary stored Y predictions 
            t_i = get_time()
            remove_unused_Y(AR_iteration = i, 
                            dict_Y_predicted = dict_training_Y_predicted,
                            dict_Y_to_remove = training_batch_dict['dict_Y_to_remove'])
            del torch_X, torch_Y
            if i == AR_iterations:
                del dict_training_Y_predicted
            tmp_AR_data_removal_timing = tmp_AR_data_removal_timing + (get_time()- t_i)
                
        ##----------------------------------------------------------------.
        # Summarize timing 
        AR_batch_timing.append(tmp_AR_batch_timing - tmp_AR_data_removal_timing)

        ##----------------------------------------------------------------.
        # - Total time elapsed
        Total_timing.append(AR_batch_timing[-1] + Dataloader_timing[-1])

    ##------------------------------------------------------------------------.
    # Create timing info dictionary 
    timing_info = {'Run': list(range(n_repetitions)), 
                   'Total': Total_timing, 
                   'Dataloader': Dataloader_timing,
                   'AR Batch': AR_batch_timing,
                   }
    ##------------------------------------------------------------------------. 
    memory_info = {'Batch': batch_memory_size}
    
    ##-------------------------------------------------------------------------. 
    # Create timing table 
    if verbose:
        table = []
        headers = ['Run', 'Total', 'Dataloader','AR Batch', 'Delete', ' ', ' ', ' ']
        for count in range(n_repetitions):
            table.append([count,    
                         round(Total_timing[count], 4),
                         round(Dataloader_timing[count], 4),
                         round(AR_batch_timing[count], 4),
                          ])
        print(tabulate(table, headers=headers))   
    ##------------------------------------------------------------------------.               
    return timing_info, memory_info
