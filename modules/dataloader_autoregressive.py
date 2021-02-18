#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 00:04:12 2021

@author: ghiggi
"""
import torch
import numpy as np
from functools import partial 
from torch.utils.data import Dataset, DataLoader

from modules.utils_autoregressive import get_dict_stack_info
from modules.utils_autoregressive import get_first_valid_idx
from modules.utils_autoregressive import get_last_valid_idx
from modules.utils_autoregressive import get_dict_Y
from modules.utils_autoregressive import get_dict_X_dynamic
from modules.utils_autoregressive import get_dict_X_bc    
from modules.utils_autoregressive import check_input_k
from modules.utils_autoregressive import check_output_k 
from modules.utils_autoregressive import check_AR_settings
from modules.utils_io import is_dask_DataArray
from modules.utils_io import check_AR_DataArrays
from modules.utils_torch import get_torch_dtype
from modules.utils_torch import check_torch_device
##----------------------------------------------------------------------------.
# TODO DataLoader Options    
# - sampler                    # Provide this option? To generalize outside batch samples?  
# - worker_init_fn             # To initialize dask scheduler? To set RNG?
##----------------------------------------------------------------------------.
### Possible speedups
# collate_fn
# - Check if collate_fn is applied not in parallel out of the multiprocess loop 
# - Code in collate_fn can be parallelized per data type and per forecast iterations)

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
                 # GPU settings 
                 device = 'cpu',
                 # Precision settings
                 numeric_precision = 'float32'):    
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
        numeric_precision : str, optional
            Numeric precision for model training. The default is 'float32'.

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
        device = check_torch_device(device)
        self.device = device
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
        torch_dtype = get_torch_dtype(numeric_precision)
        self.torch_dtype = torch_dtype
        
        ##--------------------------------------------------------------------.
        ### - Assign dynamic and bc data
        self.da_dynamic = da_dynamic 
        self.da_bc = da_bc 
        
        ##--------------------------------------------------------------------.
        ### Load static tensor into GPU (and expand over the time dimension) 
        if da_static is not None:
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
            self.torch_static = torch.tensor(da_static.values, dtype=torch_dtype).unsqueeze(unsqueeze_time_dim).unsqueeze(unsqueeze_batch_dim).expand(new_dim_size)
        else: 
            self.torch_static = None
            
        ##--------------------------------------------------------------------.
        ### - Generate indexing
        self.n_timesteps = da_dynamic.shape[0]
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
        # TODO c
        # - Use dask.delayed ??  Data are loaded on the workers, or loaded 
        #   on the client and sended to the worker after???
        # https://examples.dask.org/machine-learning/torch-prediction.html
        # https://towardsdatascience.com/computer-vision-at-scale-with-dask-and-pytorch-a18e17fc5bad
        ## -------------------------------------------------------------------.
        # Retrieve current idx of xarray  
        xr_idx_k_0 = self.idxs[idx]
        
        ## -------------------------------------------------------------------.
        ### Retrieve dynamic data 
        # - Retrieve xarray indices 
        xr_idx_dynamic_required = xr_idx_k_0 + self.rel_idx_dynamic_required  
        # - Subset the xarray Datarray (need for all autoregressive iterations)
        da_dynamic_subset = self.da_dynamic.isel(time=xr_idx_dynamic_required)
        # - Apply the scaler if provided 
        if self.scaler is not None:
            da_dynamic_subset = self.scaler.transform(da_dynamic_subset, variable_dim='feature').compute()
        # - Assign relative indices (onto the "rel_idx" dimension)
        da_dynamic_subset = da_dynamic_subset.assign_coords(rel_idx=('time', self.rel_idx_dynamic_required)).swap_dims({'time': 'rel_idx'}) 
        # - If not preloaded in CPU, load the zarr chunks (with dask)  
        if is_dask_DataArray(da_dynamic_subset): 
            da_dynamic_subset = da_dynamic_subset.compute()
        # - Loop over leadtimes and store Numpy arrays in a dictionary(leadtime)
        dict_X_dynamic_data = {}
        dict_Y_data = {}
        for i in range(self.AR_iterations + 1): 
            # Extract numpy array from DataArray and conver to Torch Tensor
            dict_Y_data[i] = torch.as_tensor(torch.from_numpy(da_dynamic_subset.sel(rel_idx=self.dict_rel_idx_Y[i]).values), dtype=self.torch_dtype)
            if self.dict_rel_idx_X_dynamic[i] is not None:
                dict_X_dynamic_data[i] = torch.as_tensor(torch.from_numpy(da_dynamic_subset.sel(rel_idx=self.dict_rel_idx_X_dynamic[i]).values), dtype=self.torch_dtype)
            else: 
                dict_X_dynamic_data[i] = None
        ##--------------------------------------------------------------------.
        ## Retrieve forecast time infos     
        # - Forecast reference time 
        forecast_reference_time = self.da_dynamic.isel(time=xr_idx_k_0).time.values
        # - Forecast leadtime_idx 
        dict_forecast_leadtime_idx = self.dict_rel_idx_Y
        # - Forecasted time
        dict_forecasted_time = {}
        for i in range(self.AR_iterations + 1): 
            dict_forecasted_time[i] = da_dynamic_subset.sel(rel_idx=self.dict_rel_idx_Y[i]).time.values 
        # - Create forecast_time_info dictionary 
        forecast_time_info = {'forecast_reference_time': forecast_reference_time,
                              'dict_forecast_leadtime_idx': dict_forecast_leadtime_idx,
                              'dict_forecasted_time': dict_forecasted_time}
         
        ## -------------------------------------------------------------------.
        ### Retrieve boundary conditions data (if provided)
        if self.da_bc is not None: 
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
                dict_X_bc_data[i] = torch.as_tensor(torch.from_numpy(da_bc_subset.sel(rel_idx=self.dict_rel_idx_X_bc[i]).values), dtype=self.torch_dtype)
        else: 
            dict_X_bc_data = None 
        
        ## -------------------------------------------------------------------.
        # Return the sample dictionary  
        return {'X_dynamic': dict_X_dynamic_data, 'X_bc': dict_X_bc_data, 
                'Y': dict_Y_data, 
                'dict_Y_to_stack': self.dict_Y_to_stack,
                'dict_Y_to_remove': self.dict_Y_to_remove,
                'dim_info': self.dim_info,
                'forecast_time_info': forecast_time_info}
    
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
        self.idxs = np.arange(n_timesteps)[idx_start:(-1*idx_end -1)]
        self.idx_start = idx_start
        self.idx_end = idx_end
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
                              prefetch_in_GPU = False,
                              asyncronous_GPU_transfer=True, 
                              device = 'cpu'):        
    """Stack the list of samples into batch of data."""
    # list_samples is a list of what returned by __get_item__ of AutoregressiveDataset
    ##------------------------------------------------------------------------.
    # Retrieve other infos
    dict_Y_to_stack = list_samples[0]['dict_Y_to_stack']
    dict_Y_to_remove = list_samples[0]['dict_Y_to_remove']
    dim_info = list_samples[0]['dim_info']
    batch_dim = dim_info['sample']
    ##------------------------------------------------------------------------.
    # Retrieve the different data (and forecast time info)
    list_X_dynamic_samples = []
    list_X_bc_samples = []
    list_Y_samples = []
    
    list_forecast_reference_time = []
    list_dict_forecasted_time = []
    dict_forecast_leadtime_idx = list_samples[0]['forecast_time_info']['dict_forecast_leadtime_idx']

    for dict_samples in list_samples:
        list_X_dynamic_samples.append(dict_samples['X_dynamic'])
        list_X_bc_samples.append(dict_samples['X_bc'])
        list_Y_samples.append(dict_samples['Y'])
        # Forecast time info 
        list_forecast_reference_time.append(dict_samples['forecast_time_info']['forecast_reference_time'])
        list_dict_forecasted_time.append(dict_samples['forecast_time_info']['dict_forecasted_time'])
    ##------------------------------------------------------------------------.
    # Retrieve the number of autoregressive iterations 
    AR_iterations = len(list_X_dynamic_samples[0]) - 1
    
    ##------------------------------------------------------------------------.    
    ### Batch data togethers   
    # - Process X_dynamic and Y
    dict_X_dynamic_batched = {}
    dict_Y_batched = {}
    for i in range(AR_iterations+1):
        if pin_memory:
            # Y
            dict_Y_batched[i] = torch.stack([dict_leadtime[i] for dict_leadtime in list_Y_samples], dim=batch_dim).pin_memory()  
            # X dynamic 
            list_X_dynamic_tensors = [dict_leadtime[i] for dict_leadtime in list_X_dynamic_samples if dict_leadtime[i] is not None]
            if len(list_X_dynamic_tensors) > 0: 
                dict_X_dynamic_batched[i] = torch.stack(list_X_dynamic_tensors, dim=batch_dim).pin_memory()
            else: # when no X dynamic (after some AR iterations)
                dict_X_dynamic_batched[i] = None
        else: 
            # Y
            dict_Y_batched[i] = torch.stack([dict_leadtime[i] for dict_leadtime in list_Y_samples], dim=batch_dim)    
            # X dynamic 
            list_X_dynamic_tensors = [dict_leadtime[i] for dict_leadtime in list_X_dynamic_samples if dict_leadtime[i] is not None]
            if len(list_X_dynamic_tensors) > 0: 
                dict_X_dynamic_batched[i] = torch.stack(list_X_dynamic_tensors, dim=batch_dim) 
            else: # when no X dynamic (after some AR iterations)
                dict_X_dynamic_batched[i] = None
                
    # - Process X_bc
    dict_X_bc_batched = {}  
    for i in range(AR_iterations+1):
        if list_X_bc_samples[0][0] is not None: 
            if pin_memory:
                dict_X_bc_batched[i] = torch.stack([dict_leadtime[i] for dict_leadtime in list_X_bc_samples], dim=batch_dim).pin_memory()
            else: 
                dict_X_bc_batched[i] = torch.stack([dict_leadtime[i] for dict_leadtime in list_X_bc_samples], dim=batch_dim) 
        else:
            dict_X_bc_batched[i] = None   
    ##------------------------------------------------------------------------. 
    # Assemble forecast_time_info 
    # dict_forecast_leadtime_idx
    # list_dict_forecasted_time 
    forecast_reference_time = np.stack(list_forecast_reference_time)
    forecast_time_info = {"forecast_reference_time": forecast_reference_time,
                          "dict_forecast_leadtime_idx": dict_forecast_leadtime_idx}
 
    ##------------------------------------------------------------------------.
    # Prefetch to GPU if asked
    if prefetch_in_GPU:
        for i in range(AR_iterations+1):
            dict_X_dynamic_batched[i] = dict_X_dynamic_batched[i].to(device=device, non_blocking=asyncronous_GPU_transfer)  
            dict_Y_batched[i] = dict_Y_batched[i].to(device=device, non_blocking=asyncronous_GPU_transfer)       
            if dict_X_bc_batched[i] is not None:
                dict_X_bc_batched[i] = dict_X_bc_batched[i].to(device=device, non_blocking=asyncronous_GPU_transfer) 
        if torch_static is not None: 
            torch_static = torch_static.to(device=device, non_blocking=asyncronous_GPU_transfer) 
    #-------------------------------------------------------------------------.   
    # Return dictionary of batched data 
    batch_dict = {'X_dynamic': dict_X_dynamic_batched, 
                  'X_bc': dict_X_bc_batched, 
                  'X_static': torch_static,
                  'Y': dict_Y_batched, 
                  'dim_info': dim_info, 
                  'forecast_time_info': forecast_time_info,
                  'dict_Y_to_remove': dict_Y_to_remove,
                  'dict_Y_to_stack': dict_Y_to_stack,
                  'prefetched_in_GPU': prefetch_in_GPU}
    
    return batch_dict
     
#-----------------------------------------------------------------------------.
# ################################
### Autoregressive DataLoader ####
# ################################
def AutoregressiveDataLoader(dataset, 
                             batch_size = 64,  
                             drop_last_batch = True,
                             random_shuffle = True,
                             num_workers = 0,
                             pin_memory = False,
                             prefetch_in_GPU = False, 
                             prefetch_factor = 2, 
                             asyncronous_GPU_transfer = True, 
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
    random_shuffle : bool, optional
        Wheter to random shuffle the samples each epoch. The default is True.
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
    device: torch.device, optional 
         Only used if 'prefetch_in_GPU' = True.
         Indicates to which GPUs to transfer the data. 
         
    Returns
    -------
    dataloader : AutoregressiveDataLoader
        pytorch DataLoader for autoregressive model training.

    """
    ##------------------------------------------------------------------------.
    ## Checks 
    device = check_torch_device(device)
    if device.type == 'cpu':
        if pin_memory:
            pin_memory = False
            if verbose:
                print("GPU is not available. 'pin_memory' set to False.")
            
        if prefetch_in_GPU: 
            prefetch_in_GPU = False
            if verbose:
                print("GPU is not available. 'prefetch_in_GPU' set to False.")
            
        if asyncronous_GPU_transfer: 
            asyncronous_GPU_transfer = False    
            if verbose:
                print("GPU is not available. 'asyncronous_GPU_transfer' set to False.")
            
    if num_workers == 0 and prefetch_factor !=2:
        prefetch_factor = 2 # bug in pytorch ... need to set to 2 
        if verbose:
            print("Since num_workers=0, no prefetching is done.")
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
    # Create pytorch Dataloader 
    # - Pass torch tensor of static data (to not reload every time)
    dataloader = DataLoader(dataset = dataset, 
                            batch_size = batch_size,  
                            shuffle = random_shuffle,
                            drop_last = drop_last_batch, 
                            num_workers = num_workers,
                            persistent_workers = False, 
                            prefetch_factor = prefetch_factor, 
                            pin_memory = False,  # pin after data have been stacked into the collate_fn
                            collate_fn = partial(autoregressive_collate_fn, 
                                                 torch_static = torch_static,
                                                 pin_memory = pin_memory,
                                                 prefetch_in_GPU = prefetch_in_GPU,
                                                 asyncronous_GPU_transfer = asyncronous_GPU_transfer, 
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
                 asyncronous_GPU_transfer = True):
    """Create X and Y Torch Tensors for a specific AR iteration."""
    i = AR_iteration
    ##------------------------------------------------------------------------.
    # Get dimension info 
    # batch_dim = batch_dict['dim_info']['sample']  
    # node_dim = batch_dict['dim_info']['node']  
    time_dim = batch_dict['dim_info']['time']  
    feature_dim = batch_dict['dim_info']['feature']  
 
    ##------------------------------------------------------------------------.
    ## Get dictionary with batched data for all forecast iterations 
    torch_static = batch_dict['X_static']
    dict_X_dynamic_batched = batch_dict['X_dynamic']
    dict_X_bc_batched = batch_dict['X_bc']
    dict_Y_batched = batch_dict['Y']
    prefetched_in_GPU = batch_dict["prefetched_in_GPU"]
    ##------------------------------------------------------------------------.
    # Check if static and bc data are available 
    static_is_available = torch_static is not None
    bc_is_available = dict_X_bc_batched[i] is not None
   
    ##------------------------------------------------------------------------.
    # Retrieve info and data for current iteration 
    list_tuple_idx_to_stack = batch_dict['dict_Y_to_stack'][i]
    
    ##------------------------------------------------------------------------.
    # Transfer into GPU (if available, or not prefetched in GPU)
    if not prefetched_in_GPU:
        # X_dynamic
        if dict_X_dynamic_batched[i] is not None:
            torch_X_dynamic = dict_X_dynamic_batched[i].to(device=device, non_blocking=asyncronous_GPU_transfer)
        else:
            torch_X_dynamic = None
        # X_static 
        if torch_static is not None: 
            torch_static = torch_static.to(device=device, non_blocking=asyncronous_GPU_transfer)
        # X_bc
        if bc_is_available:
            torch_X_bc = dict_X_bc_batched[i].to(device=device, non_blocking=asyncronous_GPU_transfer)
        # Y 
        torch_Y = dict_Y_batched[i].to(device=device, non_blocking=asyncronous_GPU_transfer)       
    else:
        torch_X_dynamic = dict_X_dynamic_batched[i]
        if bc_is_available:
            torch_X_bc = dict_X_bc_batched[i]
        torch_Y = dict_Y_batched[i] 
    ##-------------------------------------------------------------------------.
    # Stack together data required for the current forecast 
    # --> Previous predictions to stack are already in the GPU
    # --> Data need to be already in the GPU (if device is not cpu)!
    if list_tuple_idx_to_stack is not None:
        # The loop below allow to generalize the stacking to whatever time_dim position  
        # list_Y_to_stack = [dict_Y_predicted[ldt][:,idx,...] for ldt, idx in list_tuple_idx_to_stack]
        # --> Maybe one day: torch.isel({dim_name: idx}) à la xarray
        general_index = [slice(None) for i in range(len(torch_Y.shape))]
        list_Y_to_stack = []
        for ldt, idx in list_tuple_idx_to_stack:
            custom_idx = general_index.copy()
            custom_idx[time_dim] = idx
            list_Y_to_stack.append(dict_Y_predicted[ldt][custom_idx])  
            
        torch_X_to_stack = torch.stack(list_Y_to_stack, dim=time_dim)  
        if torch_X_dynamic is not None:
            torch_X_dynamic = torch.cat((torch_X_dynamic, torch_X_to_stack), dim=time_dim)
        else:
            torch_X_dynamic = torch_X_to_stack
            
    # - Add boundary conditions data (if available)
    if bc_is_available:
        torch_X = torch.cat((torch_X_bc, torch_X_dynamic), dim=feature_dim) 
    else: 
        torch_X = torch_X_dynamic
       
    # - Combine with the static tensor (which is constantly in the GPU)
    # --> In the batch dimension, match the number of samples of torch X
    if static_is_available: 
        batch_size = torch_X.shape[0]
        torch_X = torch.cat((torch_static[0:batch_size,...], torch_X), dim=feature_dim)
        
    ##------------------------------------------------------------------------.
    # - Remove unused torch Tensors 
    # del torch_X_dynamic
    del dict_X_dynamic_batched[i]  # free space
    del dict_Y_batched[i]          # free space
    if bc_is_available:
        # del torch_X_bc
        del dict_X_bc_batched[i]   # free space
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

def _cyclic(iterable):
    while True:
        for x in iterable:
            yield x
            
def cylic_iterator(iterable):
    """Make an iterable a cyclic iterator."""
    return iter(_cyclic(iterable))
            
#-----------------------------------------------------------------------------.
