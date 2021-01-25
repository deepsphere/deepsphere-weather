#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 00:04:12 2021

@author: ghiggi
"""
import time
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
from modules.utils_io import check_Datasets
from modules.utils_io import is_dask_DataArray
from modules.utils_config import get_torch_dtype

##----------------------------------------------------------------------------.
### TODO Improvements
# collate_fn
# - Check if collate_fn is applied in parallel or out of the multiprocess loop 
# - Code below can be parallelized per data type and per forecast iterations)

##----------------------------------------------------------------------------.
### Dask settings 
# - Decide if: 
#   - parallelize with pytorch.DataLoader (num_workers > 0)
#   - parallelize loading xarray data with dask
#   - both 

# dask.config.set(schedular='threads', pool=ThreadPool(5)) 
# - https://stackoverflow.com/questions/44193979/how-do-i-run-a-dask-distributed-cluster-in-a-single-thread
# - https://docs.dask.org/en/latest/scheduling.html

#-----------------------------------------------------------------------------.
############## 
### Utils ####
##############

    
#-----------------------------------------------------------------------------. 
############################### 
### Autoregressive Dataset ####
###############################   
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
                 # GPU settings 
                 device = 'cpu',
                 # Precision settings
                 numeric_precision = 'float64'):    
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
        device : str, optional
            Device on which to train the model. The default is 'cpu'.
        numeric_precision : str, optional
            Numeric precision for model training. The default is 'float64'.

        """                
        ## -------------------------------------------------------------------.
        ### - Initialize autoregressive configs   
        self.input_k = input_k 
        self.output_k = output_k
        self.forecast_cycle = forecast_cycle
        self.AR_iterations = AR_iterations
        self.stack_most_recent_prediction = stack_most_recent_prediction
        ##--------------------------------------------------------------------.
        ### - Retrieve data
        self.da_dynamic = da_dynamic 
        self.da_bc = da_bc 
        ##--------------------------------------------------------------------.
        ### - Define data precision
        torch_dtype = get_torch_dtype(numeric_precision)
        self.torch_dtype = torch_dtype
        
        ##--------------------------------------------------------------------.
        ### Load static tensor into GPU (and expand over the time dimension) 
        # - Expand by only creating a new view on the existing tensor (not allocating new memory)
        if da_static is not None:
            dim_time = 0   # static has: [node, features]
            new_dim_size = [-1 for i in range(len(da_static.shape) + 1)]
            new_dim_size[dim_time] = len(input_k)
            self.torch_static = torch.tensor(da_static.values, dtype=torch_dtype, device=device).unsqueeze(dim_time).expand(new_dim_size).unsqueeze(0) 
        else: 
            self.torch_static = None
            
        ##--------------------------------------------------------------------.
        ### - Generate valid sample indices
        n_timesteps = da_dynamic.shape[0]
        idx_start = get_first_valid_idx(input_k)
        idx_end = get_last_valid_idx(output_k = output_k,
                                     forecast_cycle = forecast_cycle, 
                                     AR_iterations = AR_iterations)
        self.idxs = np.arange(n_timesteps)[idx_start:-(idx_end)]
        self.n_samples = len(self.idxs)
        
        ##--------------------------------------------------------------------.
        ### - Define dictionary with indexing information for autoregressive training
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
        
        if da_bc is not None:
            self.rel_idx_bc_required = np.unique(np.concatenate([x for x in self.dict_rel_idx_X_bc.values() if x is not None]))
        else: 
            self.rel_idx_bc_required = None
            
        ##--------------------------------------------------------------------.    
   
    ##------------------------------------------------------------------------.
    def __len__(self):
        """Return the number of samples available."""
        return self.n_samples

    ##------------------------------------------------------------------------.
    def __getitem__(self, idx):
        """Return sample and label corresponding to an index as torch.Tensor objects."""
        # TODO:
        # - Wrap already into torch ? 
        # - Check if dask cause problems 
        # - The return tensor shapes are [ ]  ????  
        # - Use dask.delayed ? to load the data on the workers, rather than loading 
        #    it on the client and sending it to the worker
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
            dict_X_dynamic_data[i] = torch.as_tensor(torch.from_numpy(da_dynamic_subset.sel(rel_idx=self.dict_rel_idx_X_dynamic[i]).values), dtype=self.torch_dtype)
            dict_Y_data[i] = torch.as_tensor(torch.from_numpy(da_dynamic_subset.sel(rel_idx=self.dict_rel_idx_Y[i]).values), dtype=self.torch_dtype)
        
        ## -------------------------------------------------------------------.
        ### Retrieve boundary conditions data (if provided)
        if self.da_bc is not None: 
            xr_idx_bc_required = xr_idx_k_0 + self.rel_idx_bc_required  
            # - Subset the xarray Datarray (need for all autoregressive iterations)
            da_bc_subset = self.da_bc.isel(time=xr_idx_bc_required)
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
        return {'X_dynamic': dict_X_dynamic_data, 'X_bc': dict_X_bc_data, 'Y': dict_Y_data}
            
    def update_AR_iterations(self, new_AR_iterations):  
        """Update Dataset informations.
        
        If the number of forecast iterations changes, the function update
        the relative indices in order to retrieve only the needed amount of data 
        The changes to the Dataset implicitly affect the next DataLoader call!
        """
        if self.AR_iterations != new_AR_iterations:
            self.AR_iterations = new_AR_iterations
            
            # Infos for dynamic data (X and Y)
            rel_idx_X_dynamic_required = np.unique(np.concatenate([x for x in self.dict_rel_idx_X_dynamic.values() if x is not None]))
            rel_idx_Y_dynamic_required = np.unique(np.concatenate([x for x in self.dict_rel_idx_Y.values() if x is not None]))
            self.rel_idx_dynamic_required = np.unique(np.concatenate((rel_idx_X_dynamic_required, rel_idx_Y_dynamic_required)))
            
            # Infos for boundary conditions data
            if self.da_bc is not None:
                self.rel_idx_bc_required = np.unique(np.concatenate([x for x in self.dict_rel_idx_X_bc.values() if x is not None]))
            else: 
                self.rel_idx_bc_required = None    
        return None
        
##----------------------------------------------------------------------------.    
def autoregressive_collate_fn(list_samples, dict_Y_to_stack, dim_info, torch_static=None, pin_memory = False):        
    """Stack the list of samples into batch of data."""
    # list_samples is a list of what returned by __get_item__ of AutoregressiveDataset
    ##------------------------------------------------------------------------.
    # Retrieve the different data 
    list_X_dynamic_samples = []
    list_X_bc_samples = []
    list_Y_samples = []
    for dict_samples in list_samples:
        list_X_dynamic_samples.append(dict_samples['X_dynamic'])
        list_X_bc_samples.append(dict_samples['X_bc'])
        list_Y_samples.append(dict_samples['Y'])
        
    ##------------------------------------------------------------------------.
    # Retrieve the number of autoregressive iterations 
    AR_iterations = len(list_X_dynamic_samples[0]) - 1
    
    ##------------------------------------------------------------------------.    
    ### Batch data togethers   
    # - Process X_dynamic and Y
    dict_X_dynamic_batched = {}
    dict_Y_batched = {}
    for i in range(AR_iterations+1):
        if pin_memory is True:
            dict_X_dynamic_batched[i] = torch.stack([dict_leadtime[i] for dict_leadtime in list_X_dynamic_samples], dim=0).pin_memory()
            dict_Y_batched[i] = torch.stack([dict_leadtime[i] for dict_leadtime in list_Y_samples], dim=0).pin_memory()  
        else: 
            dict_X_dynamic_batched[i] = torch.stack([dict_leadtime[i] for dict_leadtime in list_X_dynamic_samples], dim=0)
            dict_Y_batched[i] = torch.stack([dict_leadtime[i] for dict_leadtime in list_Y_samples], dim=0)    
    
    # - Process X_bc
    dict_X_bc_batched = {}  
    for i in range(AR_iterations+1):
        if list_X_bc_samples[0][0] is not None: 
            if pin_memory is True:
                dict_X_bc_batched[i] = torch.stack([dict_leadtime[i] for dict_leadtime in list_X_bc_samples], dim=0).pin_memory()
            else: 
                dict_X_bc_batched[i] = torch.stack([dict_leadtime[i] for dict_leadtime in list_X_bc_samples], dim=0) 
        else:
            dict_X_bc_batched[i] = None   
            
    #-------------------------------------------------------------------------.   
    # Return dictionary of batched data 
    batch_dict = {'X_dynamic': dict_X_dynamic_batched, 
                  'X_bc': dict_X_bc_batched, 
                  'X_static': torch_static,
                  'Y': dict_Y_batched, 
                  'dim_info': dim_info, 
                  'dict_Y_to_stack': dict_Y_to_stack}
    return batch_dict
     
def autoregressive_collate_fn_old(list_samples, pin_memory = False):        
    """Stack the list of samples into batch of data."""
    # list_samples is a list of what returned by __get_item__ of AutoregressiveDataset
    ##------------------------------------------------------------------------.
    # Retrieve the different data 
    list_X_dynamic_samples = []
    list_X_bc_samples = []
    list_Y_samples = []
    for dict_samples in list_samples:
        list_X_dynamic_samples.append(dict_samples['X_dynamic'])
        list_X_bc_samples.append(dict_samples['X_bc'])
        list_Y_samples.append(dict_samples['Y'])
        
    ##------------------------------------------------------------------------.
    # Retrieve the number of autoregressive iterations 
    AR_iterations = len(list_X_dynamic_samples[0]) - 1
    
    ##------------------------------------------------------------------------.    
    ### Batch data togethers   
    
    # Process X_dynamic and Y
    dict_X_dynamic_batched = {}
    dict_Y_batched = {}
    for i in range(AR_iterations+1):
        if pin_memory is True:
            dict_X_dynamic_batched[i] = torch.stack([dict_leadtime[i] for dict_leadtime in list_X_dynamic_samples], dim=0).pin_memory()
            dict_Y_batched[i] = torch.stack([dict_leadtime[i] for dict_leadtime in list_Y_samples], dim=0).pin_memory()  
        else: 
            dict_X_dynamic_batched[i] = torch.stack([dict_leadtime[i] for dict_leadtime in list_X_dynamic_samples], dim=0)
            dict_Y_batched[i] = torch.stack([dict_leadtime[i] for dict_leadtime in list_Y_samples], dim=0)    
    
    # Process X_bc
    dict_X_bc_batched = {}  
    for i in range(AR_iterations+1):
        if list_X_bc_samples[0][0] is not None: 
            if pin_memory is True:
                dict_X_bc_batched[i] = torch.stack([dict_leadtime[i] for dict_leadtime in list_X_bc_samples], dim=0).pin_memory()
            else: 
                dict_X_bc_batched[i] = torch.stack([dict_leadtime[i] for dict_leadtime in list_X_bc_samples], dim=0) 
        else:
            dict_X_bc_batched[i] = None   
            
    #-------------------------------------------------------------------------.   
    # Return dictionary of batched data 
    return {'X_dynamic': dict_X_dynamic_batched, 'X_bc': dict_X_bc_batched, 'Y': dict_Y_batched}
   
#-----------------------------------------------------------------------------.
##################################
### Autoregressive DataLoader ####
##################################
def AutoregressiveDataLoader(dataset, 
                             dim_info, 
                             batch_size = 64,  
                             random_shuffle = True,
                             num_workers = 0,
                             pin_memory = False):
    """
    Create the DataLoader required for autoregressive model training.

    Parameters
    ----------
    dataset : AutoregressiveDataset
        An AutoregressiveDataset.
    dim_info : dict
        Dictiorary providing information of Torch Tensor dimensions.
    batch_size : init, optional
        Number of samples within a batch. The default is 64.
    random_shuffle : bool, optional
        Wheter to random shuffle the samples each epoch. The default is True.
    num_workers : 0, optional
        Number of processes that generate batches in parallel.
        0 means ONLY the main process will load batches (that can be a bottleneck).
        1 means ONLY one worker (just not the main process) will load data 
        A high enough number of workers usually assures that CPU computations 
        are efficiently managed. However, increasing num_workers increase the 
        CPU memory consumption.
        The Dataloader prefetch into the CPU 2*num_workers batches.
        The default is 0.
    pin_memory : bool, optional
        When True, it prefetch the batch data into the pinned memory.  
        pin_memory=True enables (asynchronous) fast data transfer to CUDA-enabled GPUs.
        Useful only if training on GPU.
        The default is False.

    Returns
    -------
    dataloader : AutoregressiveDataLoader
        pytorch DataLoader for autoregressive model training.

    """
    # --> If drop_last=False, need to expand ad-hoc the static tensor (to match the batch dimension) !  
    # check dim_info 
    ##------------------------------------------------------------------------.  
    # Retrieve static feature tensor (preloaded in GPU) (if available)
    torch_static = dataset.torch_static  
    
    ##------------------------------------------------------------------------. 
    # Expand static feature tensor (along first dimension) to match the batch size ! 
    if torch_static is not None:
        new_dim_size = [-1 for i in range(torch_static.dim())]
        new_dim_size[0] = batch_size
        torch_static = torch_static.expand(new_dim_size)  

    ##------------------------------------------------------------------------.
    # Retrieve information for autoregress/stack the predicted data  
    dict_Y_to_stack, dict_Y_to_remove = get_dict_stack_info(AR_iterations = dataset.AR_iterations, 
                                                            forecast_cycle = dataset.forecast_cycle, 
                                                            input_k = dataset.input_k, 
                                                            output_k = dataset.output_k, 
                                                            stack_most_recent_prediction = dataset.stack_most_recent_prediction)
    
    ##------------------------------------------------------------------------.
    # Create pytorch Dataloader 
    # - Pass torch tensor for static data and dict_Y_to_stack into collate_fn 
    # --> This allow to have all required information (within batch_dict) to stack the data during AR training 
    dataloader = DataLoader(dataset = dataset, 
                            batch_size = batch_size,  
                            shuffle = random_shuffle,
                            drop_last = True, 
                            num_workers = num_workers,
                            pin_memory = False,  # pin after data have been stacked into the collate_fn
                            collate_fn = partial(autoregressive_collate_fn, 
                                                 dim_info = dim_info, 
                                                 dict_Y_to_stack=dict_Y_to_stack, 
                                                 torch_static=torch_static,
                                                 pin_memory=pin_memory))
    return dataloader

#-----------------------------------------------------------------------------.
#######################
### AR batch tools ####
#######################               
def get_AR_batch(AR_iteration, 
                 batch_dict, 
                 dict_Y_predicted,
                 device = 'cpu', 
                 asyncronous_GPU_transfer = True):
    """Create X and Y Torch Tensors for a specific AR iteration."""
    i = AR_iteration
    ##------------------------------------------------------------------------.
    # Get dimension info 
    # batch_dim = batch_dict['dim_info']['batch_dim']  
    # node_dim = batch_dict['dim_info']['node']  
    time_dim = batch_dict['dim_info']['time']  
    feature_dim = batch_dict['dim_info']['feature']  

    ##------------------------------------------------------------------------.
    ## Get dictionary with batched data for all forecast iterations 
    torch_static = batch_dict['X_static']
    dict_X_dynamic_batched = batch_dict['X_dynamic']
    dict_X_bc_batched = batch_dict['X_bc']
    dict_Y_batched = batch_dict['Y']
    
    ##------------------------------------------------------------------------.
    # Check if static and bc data are available 
    static_is_available = torch_static is not None
    bc_is_available = dict_X_bc_batched[i] is not None
   
    ##------------------------------------------------------------------------.
    # Retrieve info and data for current iteration 
    list_tuple_idx_to_stack = batch_dict['dict_Y_to_stack'][i]
    
    ##------------------------------------------------------------------------.
    # Transfer into GPU (if available)
    torch_X_dynamic = dict_X_dynamic_batched[i].to(device=device, non_blocking=asyncronous_GPU_transfer)  
    if bc_is_available:
        torch_X_bc = dict_X_bc_batched[i].to(device=device, non_blocking=asyncronous_GPU_transfer) 
    torch_Y = dict_Y_batched[i].to(device=device, non_blocking=asyncronous_GPU_transfer)       
       
    #-------------------------------------------------------------------------.
    # Stack together data required for the current forecast 
    # --> Previous predictions to stack are already in the GPU
    # --> Data need to be already in the GPU (if device is not cpu)!
    if list_tuple_idx_to_stack is not None:
        torch_X_to_stack = torch.cat([dict_Y_predicted[ldt][:,idx,:,:] for ldt, idx in list_tuple_idx_to_stack], dim=feature_dim) 
        torch_X_dynamic = torch.cat(torch_X_dynamic, torch_X_to_stack, dim=time_dim)
    
    # - Add boundary conditions data (if available)
    if bc_is_available:
        torch_X = torch.cat((torch_X_bc, torch_X_dynamic), dim=feature_dim) 
    else: 
        torch_X = torch_X_dynamic.to(device)
       
    # - Combine with the static tensor (which is constantly in the GPU)
    if static_is_available: 
        torch_X = torch.cat((torch_static, torch_X), dim=feature_dim)
        
    ##------------------------------------------------------------------------.
    # - Remove unused torch Tensors 
    # del torch_X_dynamic
    del dict_X_dynamic_batched[i]  # free space
    del dict_Y_batched[i]          # free space
    if bc_is_available:
        # del torch_X_bc
        del dict_X_bc_batched[i]   # free space
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
#################
### Wrappers ####
#################
def create_AR_DataLoaders(ds_training_dynamic,
                          ds_validation_dynamic = None,
                          ds_static = None,              
                          ds_training_bc = None,         
                          ds_validation_bc = None,       
                          # Data loading options
                          preload_data_in_CPU = False, 
                          # Autoregressive settings  
                          input_k = [-3,-2,-1], 
                          output_k = [0],
                          forecast_cycle = 1,                           
                          AR_iterations = 2, 
                          stack_most_recent_prediction = True, 
                          # DataLoader options
                          training_batch_size = 128,
                          validation_batch_size = 128, 
                          random_shuffle = True,
                          num_workers = 0,
                          pin_memory = False,
                          # GPU settings 
                          device = 'cpu',
                          # Precision settings
                          numeric_precision = 'float64'):
    """DOC STRING."""    
    ##------------------------------------------------------------------------. 
    # Check Datasets are in the expected format for AR training 
    check_Datasets(ds_training_dynamic = ds_training_dynamic,
                   ds_validation_dynamic = ds_validation_dynamic,
                   ds_static = ds_static,              
                   ds_training_bc = ds_training_bc,         
                   ds_validation_bc = ds_validation_bc)   
    
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
    ### Load all data into CPU memory here if asked 
    if preload_data_in_CPU is True:
        ##  Dynamic data
        print("- Preload xarray Dataset of dynamic data into CPU memory:")
        t_i = time.time()
        ds_training_dynamic = ds_training_dynamic.compute()
        print('  --> Training Dynamic Dataset: {:.2f}s'.format(time.time() - t_i))
        if ds_validation_dynamic is not None:
            t_i = time.time()
            ds_validation_dynamic = ds_validation_dynamic.compute()
            print('  --> Validation Dynamic Dataset: {:.2f}s'.format(time.time() - t_i))
        
        ##--------------------------------------------------------------------.
        ## Boundary conditions data
        if ds_training_bc is not None: 
            print("- Preload xarray Dataset of boundary conditions data into CPU memory:")
            t_i = time.time()
            ds_training_bc = ds_training_bc.compute()
            print('  --> Training Boundary Condition Dataset: {:.2f}s'.format(time.time() - t_i))
            if ds_validation_bc is not None:
                t_i = time.time()
                ds_validation_bc = ds_validation_bc.compute()
                print('  --> Validation Boundary Condition Dataset: {:.2f}s'.format(time.time() - t_i))
            
    ##------------------------------------------------------------------------. 
    ### Conversion to DataArray and order dimensions 
    # - For dynamic and bc: ['time', 'node', 'features']
    # - For static: ['node', 'features']
    t_i = time.time()
    da_training_dynamic = ds_training_dynamic.to_array(dim='feature', name='Dynamic').transpose('time', 'node', 'feature')
    if ds_validation_dynamic is not None:
        da_validation_dynamic = ds_validation_dynamic.to_array(dim='feature', name='Dynamic').transpose('time', 'node', 'feature')
    if ds_training_bc is not None:
        da_training_bc = ds_training_bc.to_array(dim='feature', name='BC').transpose('time', 'node', 'feature')
    if ds_validation_bc is not None:
        da_validation_bc = ds_validation_bc.to_array(dim='feature', name='BC').transpose('time', 'node', 'feature')
    if ds_static is not None: 
        da_static = ds_static.to_array(dim='feature', name='Static').transpose('node','feature') 
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
    trainingDataset = AutoregressiveDataset(da_dynamic = da_training_dynamic,  
                                            da_bc = da_training_bc,
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
    print('- Creation of Training AutoregressiveDataset: {:.2f}s'.format(time.time() - t_i))
    
    t_i = time.time()
    trainingDataLoader = AutoregressiveDataLoader(dataset = trainingDataset, 
                                                  batch_size = training_batch_size,  
                                                  random_shuffle = random_shuffle,
                                                  num_workers = num_workers,
                                                  pin_memory = pin_memory,
                                                  dim_info = dim_info)
    print('- Creation of Training AutoregressiveDataLoader: {:.2f}s'.format(time.time() - t_i))
    
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
                                                  AR_iterations = AR_iterations, 
                                                  stack_most_recent_prediction = stack_most_recent_prediction, 
                                                  # GPU settings 
                                                  device = device,
                                                  # Precision settings
                                                  numeric_precision = numeric_precision)
        print('- Creation of Validation AutoregressiveDataset: {:.2f}s'.format(time.time() - t_i))
        validationDataLoader = AutoregressiveDataLoader(dataset = validationDataset, 
                                                        dim_info = dim_info,
                                                        batch_size = validation_batch_size,  
                                                        random_shuffle = random_shuffle,
                                                        num_workers = num_workers,
                                                        pin_memory = pin_memory)
        print('- Creation of Validation AutoregressiveDataLoader: {:.2f}s'.format(time.time() - t_i))
    else: 
        validationDataLoader = None
    
    ##------------------------------------------------------------------------. 
    return trainingDataLoader, validationDataLoader
