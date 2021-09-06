#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 00:04:12 2021

@author: ghiggi
"""
from re import I
import time
import warnings
import torch
import random
import inspect
import xarray as xr
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
from modules.utils_xr import xr_align_dim
from modules.utils_xr import xr_align_start_time
from modules.utils_xr import xr_has_uniform_resolution
from modules.utils_xr import xr_have_same_timesteps
from modules.utils_xr import is_dask_DataArray
from modules.utils_io import check_timesteps_format
from modules.utils_io import check_no_duplicate_timesteps
from modules.utils_io import _check_temporal_data
from modules.utils_io import _check_static_data
from modules.utils_io import _get_feature_info_dicts
from modules.utils_io import _get_feature_order_dicts
from modules.utils_io import _get_dim_info_dicts
from modules.utils_io import _get_dim_order_dicts
from modules.utils_io import _get_dim_order
from modules.utils_io import _get_dim_info
from modules.utils_io import _get_subset_timesteps_idxs
from modules.utils_torch import set_seeds
from modules.utils_torch import check_device
from modules.utils_torch import check_pin_memory
from modules.utils_torch import check_asyncronous_gpu_transfer
from modules.utils_torch import check_prefetch_in_gpu
from modules.utils_torch import check_prefetch_factor
from modules.utils_torch import get_time_function

##----------------------------------------------------------------------------.
### Main thoughts
# Per sphere-patch dataloading IS NOT IMPLEMENTED !
# - It requires information on the RNN model theoretical receptive field to decide which data to load 

## AR_DataLoader assumptions 
# - Only 1 level input data 
# - Same input-output resolution --> Data can be retrieved by single DataArray/Dataset  
# - DataArrays within a Dataset must share all the same dimension! 
# - data_dynamic and data_bc do not need to share all dimensions (but time yes for training !) 
# - Boundary condition option (data_bc or bc_generator)
# - bc_generator(data_dynamic, data_static) allows the user to:   [NOT YET IMPLEMENTED]
#   --> generate the custom bc data from DataArray dimensions (i.e. time) 
#   --> generate data_dynamic dependent bc data (i.e. surface closures, ... ) 
#   --> ...
   
## (Classical) DataLoader assumption [NOT YET IMPLEMENTED]
# - Should allow different input - output levels
# - Should allow multiscale inputs outputs 
# --> [args: input_data_dynamic={...}, output_data_dynamic, input_data_static={...}]
# --> [dim_info['dynamic']{'level1': ..., 'level'2': ...}

##----------------------------------------------------------------------------.
### Current assumptions 
# - Currently data_dynamic and data_bc must have the same number of timesteps (TO IMPROVE)
# --> The dataloader loads the same positional indices along the time dimension (TO IMPROVE)
# - We use a map-style datasets, the main process generates the indices using sampler
#   and sends them to the workers
##----------------------------------------------------------------------------.
# ############################
### DataLoader utils      ####
# ############################
def _cyclic(iterable):
    while True:
        for x in iterable:
            yield x
            
def cylic_iterator(iterable):
    """Make an iterable a cyclic iterator."""
    return iter(_cyclic(iterable))

def worker_init_fn(worker_id):
    """Function to initialize the seed of the DataLoader workers."""
    # - Eeach worker has an independent seed that is initialized to 
    #   the curent random seed + the id of the worker
    # - Need to reset the numpy random seed at the beginning of each epoch
    #   because all random seed modifications in __getitem__ are local to each worker
    initial_seed = int(torch.initial_seed() % (2**32-1))
    worker_seed = initial_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    # set_seeds(worker_seed)
    return None

#-----------------------------------------------------------------------------.
# ############################
### AR batching functions ####
# ############################   
def _check_ar_batch_fun_kwargs(ar_batch_fun):
    """ Check if the custom ar_batch_fun is properly defined."""
    if not callable(ar_batch_fun):
        raise TypeError("'ar_batch_fun' must be a function.")
    # Retrieve function arguments
    fun_kwargs = inspect.getfullargspec(ar_batch_fun).args
    # Check the function has the required kwargs 
    required_kwargs = ["ar_iteration", "batch_dict", "dict_Y_predicted", "device", "asyncronous_gpu_transfer"]
    missing_kwargs = np.array(required_kwargs)[np.isin(required_kwargs, fun_kwargs, invert=True)]
    excess_kwargs = np.array(fun_kwargs)[np.isin(fun_kwargs, required_kwargs, invert=True)]
    if len(missing_kwargs) > 0:
        raise ValueError('ar_batch_fun miss the {!r] kwargs.'.format(missing_kwargs))
    if len(excess_kwargs) > 0: 
        raise ValueError('ar_batch_fun does not require the {!r] kwargs.'.format(excess_kwargs))
    return None
           
def get_aligned_ar_batch(ar_iteration, 
                         batch_dict, 
                         dict_Y_predicted,
                         device = 'cpu', 
                         asyncronous_gpu_transfer = True):
    """ar_batch_fun that return X and Y Torch Tensors for a specific AR iteration.

    This ar_batch_fun assumes that data_dynamic and data_bc tensors have aligned  
    dimensions (same dimensions and same shape).    
    The data are stacked with the following order: [data_static, data_bc, data_dynamic]
    """
    ##------------------------------------------------------------------------.
    # Get dimension info 
    dim_info_dynamic = batch_dict['dim_info']['dynamic']
    dim_info_bc = batch_dict['dim_info']['bc']
    dim_order_dynamic = batch_dict['dim_order']['dynamic']
    dim_order_bc = batch_dict['dim_order']['bc']
    data_availability = batch_dict['data_availability']
    
    n_dims = len(dim_info_dynamic)   
    time_dim = dim_info_dynamic['time']  
    feature_dim = dim_info_dynamic['feature']  
    # batch_dim = dim_info_dynamic['sample']  
    # node_dim = dim_info_dynamic['node']  
    ##------------------------------------------------------------------------.
    # Check that if data_bc are provided, have same dimension order
    if data_availability['bc']:
        if not np.array_equal(dim_order_dynamic, dim_order_bc): 
            raise ValueError("'dynamic and bc batch data do not have the same dimension order!")
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
        bc_is_available = dict_X_bc_batched[ar_iteration] is not None
   
    ##------------------------------------------------------------------------.
    # Retrieve info and data for current iteration 
    list_tuple_idx_to_stack = batch_dict['dict_Y_to_stack'][ar_iteration]
    
    ##------------------------------------------------------------------------.
    ### Prepare torch Tensor available in GPU 
    # - X_dynamic 
    if dict_X_dynamic_batched[ar_iteration] is not None:
        if not prefetched_in_gpu:
            torch_X = dict_X_dynamic_batched[ar_iteration].to(device=device, non_blocking=asyncronous_gpu_transfer)
        else: 
            torch_X = dict_X_dynamic_batched[ar_iteration]
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
            torch_X_bc = dict_X_bc_batched[ar_iteration].to(device=device, non_blocking=asyncronous_gpu_transfer)
        else: 
            torch_X_bc = dict_X_bc_batched[ar_iteration]
    ##--------------------------------------------.
    # - Y 
    if training_mode:
        if not prefetched_in_gpu:
            torch_Y = dict_Y_batched[ar_iteration].to(device=device, non_blocking=asyncronous_gpu_transfer)
        else:
            torch_Y = dict_Y_batched[ar_iteration] 
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
        shape_dynamic = torch_X_bc.shape[:-1]
        shape_bc = torch_X.shape[:-1]
        if not np.allclose(shape_dynamic, shape_bc):
            raise ValueError("'data_dynamic' and 'data_bc' are likely disaligned."
                             "You might want to use the 'ar_batch_fun' like 'get_disaligned_ar_batch'."
                             "'Excluding the feature dimension, the bc batch has shape {},"
                             " while the dynamic batch has shape {}".format(shape_dynamic, shape_bc))
        torch_X = torch.cat((torch_X_bc, torch_X), dim=feature_dim) 
       
    # - Combine with the static tensor (which is constantly in the GPU)
    # --> In the batch dimension, match the number of samples of torch X
    if static_is_available: 
        shape_dynamic = torch_X.shape[:-1]
        shape_static = torch_X.shape[:-1]
        if not np.allclose(shape_dynamic, shape_static):
            raise ValueError("'data_dynamic' and 'data_static' are likely disaligned."
                             "You might want to use the 'ar_batch_fun' like 'get_disaligned_ar_batch'."
                             "'Excluding the feature dimension, the static batch has shape {},"
                             " while the dynamic batch has shape {}".format(shape_dynamic, shape_static))
        batch_size = torch_X.shape[0]
        torch_X = torch.cat((torch_static[0:batch_size,...], torch_X), dim=feature_dim)
        
    ##------------------------------------------------------------------------.
    # - Remove unused torch Tensors 
    del dict_X_dynamic_batched[ar_iteration]   
    if training_mode:
        del dict_Y_batched[ar_iteration]            
    if bc_is_available:
        del torch_X_bc
        del dict_X_bc_batched[ar_iteration]    
    if static_is_available: 
        del torch_static

    ##------------------------------------------------------------------------.    
    return (torch_X, torch_Y)

def get_disaligned_ar_batch(ar_iteration, 
                            batch_dict, 
                            dict_Y_predicted,
                            device = 'cpu', 
                            asyncronous_gpu_transfer = True):
    """ar_batch_fun that return a dictionary and Y Torch Tensors for a specific AR iteration.

    This ar_batch_fun deals with disaligned data_dynamic and data_bc tensors.
    It returns a dictionary with the various input and the Y dynamic torch tensor.
    """
    ##------------------------------------------------------------------------.
    # Get dimension info 
    dim_info_dynamic = batch_dict['dim_info']['dynamic']
    dim_info_bc = batch_dict['dim_info']['bc']
    dim_order_dynamic = batch_dict['dim_order']['dynamic']
    dim_order_bc = batch_dict['dim_order']['bc']
    data_availability = batch_dict['data_availability']
    
    n_dims = len(dim_info_dynamic)   
    time_dim = dim_info_dynamic['time']  
    feature_dim = dim_info_dynamic['feature']  
    # batch_dim = dim_info_dynamic['sample']  
    # node_dim = dim_info_dynamic['node']  

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
        bc_is_available = dict_X_bc_batched[ar_iteration] is not None
   
    ##------------------------------------------------------------------------.
    # Retrieve info and data for current iteration 
    list_tuple_idx_to_stack = batch_dict['dict_Y_to_stack'][ar_iteration]
    
    ##------------------------------------------------------------------------.
    ### Prepare torch Tensor available in GPU 
    # - X_dynamic 
    if dict_X_dynamic_batched[ar_iteration] is not None:
        if not prefetched_in_gpu:
            torch_X = dict_X_dynamic_batched[ar_iteration].to(device=device, non_blocking=asyncronous_gpu_transfer)
        else: 
            torch_X = dict_X_dynamic_batched[ar_iteration]
    else:
        torch_X = None
    ##--------------------------------------------.
    # - X_static
    if torch_static is not None: 
        if not prefetched_in_gpu:
            torch_static = torch_static.to(device=device, non_blocking=asyncronous_gpu_transfer)
    else: 
        torch_static = None
    ##--------------------------------------------.
    # - X_bc 
    if bc_is_available:
        if not prefetched_in_gpu:
            torch_X_bc = dict_X_bc_batched[ar_iteration].to(device=device, non_blocking=asyncronous_gpu_transfer)
        else: 
            torch_X_bc = dict_X_bc_batched[ar_iteration]
    else: 
        torch_X_bc = None
    ##--------------------------------------------.
    # - Y 
    if training_mode:
        if not prefetched_in_gpu:
            torch_Y = dict_Y_batched[ar_iteration].to(device=device, non_blocking=asyncronous_gpu_transfer)
        else:
            torch_Y = dict_Y_batched[ar_iteration] 
    else: # prediction mode 
        torch_Y = None         
    ##-------------------------------------------------------------------------.
    # Stack together the dynamic data required for the current forecast 
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
    ##-------------------------------------------------------------------------.
    ### Prepare dictionary of X torch tensors 
    dict_torch_X = {}
    dict_torch_X['dynamic'] = torch_X
    dict_torch_X['bc'] = torch_X_bc
    dict_torch_X['static'] = torch_static
    
    ##------------------------------------------------------------------------.
    # - Remove unused torch Tensors 
    del dict_X_dynamic_batched[ar_iteration]   
    if training_mode:
        del dict_Y_batched[ar_iteration]            
    if bc_is_available:
        del torch_X_bc
        del dict_X_bc_batched[ar_iteration]    
    if static_is_available: 
        del torch_static
    ##------------------------------------------------------------------------.    
    return (dict_torch_X, torch_Y)

##----------------------------------------------------------------------------.
def remove_unused_Y(ar_iteration, dict_Y_predicted, dict_Y_to_remove):
    """Remove unused Y predictions of past AR iterations."""
    list_idx_Y_to_remove = dict_Y_to_remove[ar_iteration]
    if list_idx_Y_to_remove is not None:
        for ldt in list_idx_Y_to_remove: 
            del dict_Y_predicted[ldt]
    return None

#-----------------------------------------------------------------------------.       
# ############################# 
### Autoregressive Dataset ####
# #############################   
class AutoregressiveDataset(Dataset):
    """Map-style autoregressive pytorch dataset."""
    
    ## -----------------------------------------------------------------------.
    def __init__(self,
                 data_dynamic, 
                 # Autoregressive settings  
                 input_k, 
                 output_k,
                 forecast_cycle,                           
                 ar_iterations, 
                 stack_most_recent_prediction, 
                 # Facultative input data
                 data_bc = None, 
                 data_static = None,
                 bc_generator = None, 
                 scaler = None, 
                 # Custom function to get batches across AR iterations
                 ar_batch_fun = get_aligned_ar_batch,
                 # Setting for optional time subsets
                 subset_timesteps = None, 
                 training_mode = True, 
                 # GPU settings 
                 device = 'cpu'):    
        """
        "Create the Dataset required to generate an AutoregressiveDataloader.

        Parameters
        ----------
        data_dynamic : DataArray
            DataArray with dynamic data.
        data_bc : DataArray, optional
            DataArray with boundary conditions features.
            The default is None.
        data_static : DataArray, optional
            DataArray with static features.  
            The default is None.
        bc_generator : 
            Function taking as input data_dynamic and (if available) 
            data_bc, and data_static to generate boundary conditions on the fly.
            Format: def bc_generator(data_dynamic, data_bc=None, data_static=None):
        scaler : xscaler 
            xscaler object to transform the DataArrays.
            The default is None.
        ar_batch_fun : callable 
            Custom function that batch/stack together data across AR iterations.
            The custom function must return a tuple of length 2 (X, Y), but X and Y 
            can be whatever desired objects (torch.Tensor, dict of Tensor, ...). 
            The custom function must have the following arguments: 
                def ar_batch_fun(ar_iteration, batch_dict, dict_Y_predicted,
                                 device = 'cpu', asyncronous_gpu_transfer = True)
            The default ar_batch_fun function is the pre-implemented get_aligned_ar_batch() which return 
            two torch.Tensor: one for X (input) and one four Y (output). Such function expects 
            the dynamic and bc batch data to be aligned (have same dimensions and shape)
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
        ar_iterations : int
            Number of AR iterations.
        stack_most_recent_prediction : bool
            Whether to use the most recent prediction when autoregressing.
        device : torch.device, optional
            Device on which to train the model. The default is 'cpu'.
        """      
        # - data_bc and data_dynamic are reshaped to DataArray only in __get_item__
        # - data_dynamic, data_bc, data_static must not have all dimension aligned
        # - data_dynamic and data_bc must however be aligned over the time dimension if training_mode=True
        # - how/if the 3 data_type are stacked together is specified in ar_batch_fun
        # If Dataset are provided, it assumes all variables DataArray have same dimensions and are aligned.

        ##--------------------------------------------------------------------.
        # Check input_k and output_k type
        input_k = check_input_k(input_k=input_k, ar_iterations=ar_iterations)   
        output_k = check_output_k(output_k=output_k)
        ## -------------------------------------------------------------------.
        # Check ar_batch_fun is properly defined 
        _check_ar_batch_fun_kwargs(ar_batch_fun)
        ## -------------------------------------------------------------------.
        # Check input data 
        # - Checks data_dynamic 
        if data_dynamic is None: 
            raise ValueError("'data_dynamic' cannot be None! Provide an xr.Dataset or xr.DataArray.")
        _check_temporal_data(data_dynamic, data_type = 'data_dynamic', 
                             time_dim = "time", feature_dim='feature')
        # - Checks data_static  
        _check_static_data(data_static, data_type = 'data_static', 
                           time_dim = "time", feature_dim='feature')
        ##---------------------------------------------------------------------.
        # - Check data_bc 
        if data_bc is not None and bc_generator is not None: 
            raise ValueError("Either provide 'data_bc' or 'bc_generator'.")     
        if bc_generator is not None:
            # TODO
            # data_bc = ... 
            raise NotImplementedError()
        _check_temporal_data(data_bc, data_type = 'data_bc', 
                             time_dim = "time", feature_dim='feature')
        
        ## -------------------------------------------------------------------.
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
        self.ar_iterations = ar_iterations
        self.stack_most_recent_prediction = stack_most_recent_prediction
        ##--------------------------------------------------------------------.
        ### - Define data precision
        self.torch_dtype = torch.get_default_dtype() 
        ##--------------------------------------------------------------------.
        ### - Define optional timesteps subsets
        self.subset_timesteps = check_timesteps_format(subset_timesteps)
        ##--------------------------------------------------------------------.
        ### Initialize scaler 
        self.scaler = scaler 
        ##--------------------------------------------------------------------.
        ### - Assign dynamic, bc data, bc_generator
        self.data_dynamic = data_dynamic 
        self.data_bc = data_bc 
        ### - Assign bc generator
        self.bc_generator = bc_generator
        ### - Assign ar_batch_fun function to prepare batches across AR iterations 
        self.ar_batch_fun = ar_batch_fun
        ##--------------------------------------------------------------------.
        ### - Build dictionary of data availability 
        data_availability = {}
        data_availability['static'] = data_static is not None
        data_availability['bc'] = data_bc is not None  
        data_availability['bc_generator'] = bc_generator is not None  
        self.data_availability = data_availability
        
        ##---------------------------------------------------------------------.
        # Retrieve data temporal resoltuon and check dynamic and bc data have the same  
        if not xr_has_uniform_resolution(data_dynamic, dim = "time"):
            raise ValueError("'data_dynamic' does not have an uniform 'time' resolution.")
        self.t_res_timedelta = np.diff(data_dynamic['time'].values)[0] 

        if data_availability['bc']:
            if not xr_has_uniform_resolution(data_bc, dim = "time"):
                raise ValueError("'data_bc' does not have an uniform 'time' resolution.")
            if self.t_res_timedelta != np.diff(data_bc['time'].values)[0]:
                raise ValueError("'data_dynamic' and 'data_bc' does not have the same 'time' resolution.")

        ##---------------------------------------------------------------------.
        # Check time alignment of data_dynamic and data_bc if training_mode = True
        # - If training_mode = False, check only that start_time is the same and 
        #   end_time of data_bc equal or is after data-dynamic 
        if training_mode: 
            if not xr_have_same_timesteps(data_dynamic, data_bc, time_dim = "time"):
                print("'data_dynamic' and 'data_bc' do not have the same timesteps."
                      "Data are going to be re-aligned along the 'time' dimension.")
                data_dynamic, data_bc = xr_align_dim(data_dynamic, data_bc, dim='time')
        else:
            data_dynamic, data_bc = xr_align_start_time(data_dynamic, data_bc, time_dim='time')
            if data_availability['bc']:
                if np.max(data_bc['time'].values) < np.max(data_dynamic['time'].values):
                    raise ValueError("'data_bc' must have a 'time' dimension equal or longer than 'data_dynamic'.")        

        ##--------------------------------------------------------------------.
        ### - Test bc generation and get a mock sample 
        if data_availability['bc_generator']:
           # TODO:
           # bc_sample = bc_generator(data_dynamic, data_static)
            # data_bc = bc_sample
            raise NotImplementedError
             
        ##--------------------------------------------------------------------.
        ### - Define dim_info and dim_order  
        # - Sample/batch dimension is fixed to become the first dimension ! 
        # - The others can vary  
        dim_info = _get_dim_info_dicts(data_dynamic, data_bc, data_static)
        dim_order = _get_dim_order_dicts(data_dynamic, data_bc, data_static)   
        self.dim_info = dim_info 
        self.dim_order = dim_order 

        ##--------------------------------------------------------------------.
        ### - Define feature_order and feature_info
        feature_info = _get_feature_info_dicts(data_dynamic, data_bc, data_static)
        feature_order = _get_feature_order_dicts(data_dynamic, data_bc, data_static)            
        self.feature_info = feature_info
        self.feature_order = feature_order

        ##--------------------------------------------------------------------.
        ### Load static tensor into CPU (and expand over the time dimension) 
        if data_availability['static']:
            # - If xr.Dataset, convert to DataArray
            if isinstance(data_static, xr.Dataset): 
                data_static = data_static.to_array(dim='feature')
            # - Ensure that the feature dimension is the last 
            data_static = data_static.transpose(...,'feature')
            # - If not preloaded in CPU, load the static data   
            if is_dask_DataArray(data_static): 
                data_static = data_static.compute()  
            
            # - Apply scaler 
            if self.scaler is not None:
                data_static = self.scaler.transform(data_static, variable_dim='feature')
            ##----------------------------------------------------------------.
            # - Reshape data_static to match ('node','feature') order of data_dynamic
            dims_dynamic = dim_order['dynamic'][1:]     # Remove 'sample' in the first position
  
            dims_static = _get_dim_order(data_static)[1:] # Remove 'sample' in the first position
            required_static_dim_order = np.array(dims_dynamic)[np.isin(dims_dynamic, dims_static)].tolist()
            if not required_static_dim_order == dims_static:
                print("Reshaping static DataArray to have dimension order: {}".format(required_static_dim_order))
                data_static = data_static.transpose(*required_static_dim_order)   
                self.dim_order['static'] = _get_dim_order(data_static)
                self.dim_info['static'] = _get_dim_info(data_static)
            ##----------------------------------------------------------------.
            ## Add batch and time dimension and then expand to match the lag dimension of the dynamic tensor  
            # - It's reasonable that static data wanto to be expanded to match
            #   the "lag" time dimension of the model input dynamic tensor.
            # - Here we don't expand for other possible dimensions of data_dynamic (i.e. p_levels, ens_level).
            # - The expansion in the batch_dim is done within the AR_Dataloader based on batch_size argument.

            # - Define ways to unsqueeze the static tensor 
            unsqueeze_batch_dim = dim_info['dynamic']['sample']   # currently 0 
            unsqueeze_time_dim = dim_info['dynamic']['time'] - 1  # (without batch dim ...)
            # - Define the dimensions of the expanded tensor 
            dim_batch = dim_info['dynamic']['sample']  # currently 0 
            dim_time = dim_info['dynamic']['time']  
            new_dim_size = [-1 for i in range(len(data_static.dims) + 2)]
            new_dim_size[dim_batch] = 1            # Batch dimension 
            new_dim_size[dim_time] = len(input_k)  # The (predictor lag) 'time' dimension)
            # - Use a view to expand (to not allocate new memory)
            self.torch_static = torch.tensor(data_static.values, dtype=self.torch_dtype, device='cpu').unsqueeze(unsqueeze_time_dim).unsqueeze(unsqueeze_batch_dim).expand(new_dim_size)
        else: 
            self.torch_static = None
            
        ##--------------------------------------------------------------------.
        ### - Generate indexing
        self.update_indexing()
        
        ##--------------------------------------------------------------------.    
    def update_indexing(self):
        """Update indices."""
        input_k = self.input_k 
        output_k = self.output_k
        forecast_cycle = self.forecast_cycle
        ar_iterations = self.ar_iterations
        stack_most_recent_prediction = self.stack_most_recent_prediction

        ##--------------------------------------------------------------------.
        ## Update dictionary Y to stack and remove 
        dict_Y_to_stack, dict_Y_to_remove = get_dict_stack_info(ar_iterations = ar_iterations, 
                                                                forecast_cycle = forecast_cycle, 
                                                                input_k = input_k, 
                                                                output_k = output_k,
                                                                stack_most_recent_prediction = stack_most_recent_prediction)
        self.dict_Y_to_stack = dict_Y_to_stack
        self.dict_Y_to_remove = dict_Y_to_remove

        ##--------------------------------------------------------------------.
        # - Update valid data range and idxs for training (and prediction)
        available_timesteps = self.data_dynamic['time'].values
        n_timesteps = len(available_timesteps)
        idx_start = get_first_valid_idx(input_k)
        if self.training_mode:
            idx_end = get_last_valid_idx(output_k = output_k,
                                         forecast_cycle = forecast_cycle, 
                                         ar_iterations = ar_iterations)
            idx_end = n_timesteps - 1 - idx_end 
        else: 
            idx_end = n_timesteps - 1
        self.idxs = np.arange(n_timesteps)[idx_start:idx_end+1]

        ##--------------------------------------------------------------------.
        # - Restricts self.idxs to match specific timesteps 
        # - This is useful to launch prediction for specific forecast reference times  
        # - It restrics self.idxs that can be loaded by the DataLoader
        if self.subset_timesteps is not None:
            subset_timesteps = self.subset_timesteps
            subset_timesteps.sort() # ensure the temporal order 
            check_no_duplicate_timesteps(subset_timesteps, var_name='subset_timesteps')
            # - Remove 'subset_timesteps' that are outside the time range of data_dynamic 
            subset_idxs = _get_subset_timesteps_idxs(timesteps = available_timesteps, 
                                                     subset_timesteps = subset_timesteps,
                                                     strict_match=False) 
           
            # - Retrieve subset_idxs that are not valid because of current AR settings (and 'data_dynamic')
            unvalid_idxs = subset_idxs[subset_idxs < idx_start]
            if len(unvalid_idxs) > 0:
                warnings.warn("With current 'data_dynamic' and AR settings, the following 'forecast_start_time' are "
                              "not allowed: {} \n".format(list(available_timesteps[unvalid_idxs])))
            unvalid_idxs = subset_idxs[subset_idxs > idx_end]                   
            if len(unvalid_idxs) > 0:
                warnings.warn("With current 'data_dynamic' and AR settings, the following 'forecast_start_time' are "
                              "not allowed: {} \n".format(list(available_timesteps[unvalid_idxs])))
            # - Select only valid subset_idxs
            subset_idxs = subset_idxs[subset_idxs >= idx_start]
            if len(subset_idxs) == 0: 
                raise ValueError("Because of 'data_dynamic' time coverage and current AR settings, "
                                 "the specified 'forecast_reference_times' are not valid.")
            subset_idxs = subset_idxs[subset_idxs <= idx_end]
            if len(subset_idxs) == 0: 
                raise ValueError("Because of 'data_dynamic' time coverage and current AR settings, "
                                 "the specified 'forecast_reference_times' are not valid.")
            # - Update valid idxs 
            self.idxs = subset_idxs  
            # - Update valid subset_timesteps based on 'data_dynamic' and 'ÂR settings'    
            subset_timesteps = available_timesteps[subset_idxs]
            self.subset_timesteps = subset_timesteps     
                   

        ##--------------------------------------------------------------------.
        # - Retrieve timesteps of forecast leadtime 0 available (rel_idx = 0)
        self.available_starts = available_timesteps[self.idxs]
        
        ##--------------------------------------------------------------------.
        # - Compute the number of samples available
        self.n_samples = len(self.idxs)
        if self.n_samples == 0: 
            raise ValueError("No samples available. Maybe reduce number of AR iterations.")

        ##--------------------------------------------------------------------.
        ### - Update dictionary with indexing information for autoregressive training
        self.dict_rel_idx_Y = get_dict_Y(ar_iterations = ar_iterations,
                                         forecast_cycle = forecast_cycle, 
                                         output_k = output_k)
        self.dict_rel_idx_X_dynamic = get_dict_X_dynamic(ar_iterations = ar_iterations,
                                                         forecast_cycle = forecast_cycle, 
                                                         input_k = input_k)
        self.dict_rel_idx_X_bc = get_dict_X_bc(ar_iterations = ar_iterations,
                                               forecast_cycle = forecast_cycle,
                                               input_k = input_k)

        ##---------------------------------------------------------------------.
        # Check that data_bc is enough long for the specified ar iterations
        #  in training_mode = False  
        if self.data_availability['bc'] and not self.training_mode:
            n_bc_timesteps = len(self.data_bc['time'].values)
            max_rel_idx_bc_k = np.max(list(self.dict_rel_idx_X_bc.values()))
            idxs_bc_required = self.idxs + max_rel_idx_bc_k
            valid_start_idxs = self.idxs[idxs_bc_required < n_bc_timesteps]
            unvalid_start_idxs = self.idxs[idxs_bc_required >= n_bc_timesteps]
            # If any forecast can be generated, raise an error 
            if len(valid_start_idxs) == 0: 
                raise ValueError("Because of 'data_bc' time coverage and current AR settings, "
                                 "it's not possible to generate any forecast with {} 'ar_iterations'.\n"
                                 "Try to reduce the number of 'ar_iterations'!".format(self.ar_iterations))
        
            # Restrict the idxs to possible ones
            if len(unvalid_start_idxs) > 0: 
                last_valid_ref_time = available_timesteps[valid_start_idxs[-1] + max(input_k)]
                print("'data_bc' is not long enough in the 'time' dimension "
                      "to generate all forecast specified.\n"
                      "The last forecast which can be generated has 'forecast_reference_time': {!r}".format(last_valid_ref_time))
                # - Update valid idxs and subset timesteps 
                self.idxs = valid_start_idxs
                self.subset_timesteps = available_timesteps[self.idxs]
                # - Update available timesteps of forecast leadtime 0  (rel_idx = 0)
                self.available_starts = available_timesteps[self.idxs]

        ##--------------------------------------------------------------------.
        # - Compute the number of samples available
        self.n_samples = len(self.idxs)
        if self.n_samples == 0: 
            raise ValueError("No samples available. Maybe reduce number of AR iterations.")

        ##---------------------------------------------------------------------.
        ### - Based on the current value of ar_iterations, create a
        #     list of (relative) indices required to load data from da_dynamic and da_bc 
        #   --> This indices are updated when Dataset.update_ar_iterations() is called
        rel_idx_X_dynamic_required = np.unique(np.concatenate([x for x in self.dict_rel_idx_X_dynamic.values() if x is not None]))
        if self.training_mode:
            rel_idx_Y_dynamic_required = np.unique(np.concatenate([x for x in self.dict_rel_idx_Y.values() if x is not None]))
            self.rel_idx_dynamic_required = np.unique(np.concatenate((rel_idx_X_dynamic_required, rel_idx_Y_dynamic_required)))
        else:
            self.rel_idx_dynamic_required = np.unique(rel_idx_X_dynamic_required)

        if self.data_availability['bc'] or self.data_availability['bc_generator']:
            self.rel_idx_bc_required = np.unique(np.concatenate([x for x in self.dict_rel_idx_X_bc.values() if x is not None]))
        else: 
            self.rel_idx_bc_required = None
            
        ##--------------------------------------------------------------------.
            
    def update_ar_iterations(self, new_ar_iterations):  
        """Update Dataset informations.
        
        If the number of forecast iterations changes, the function update
        the relative indices in order to retrieve only the needed amount of data 
        The changes to the Dataset implicitly affect the next DataLoader call!
        """
        if self.ar_iterations != new_ar_iterations:                
            # Update AR iterations
            self.ar_iterations = new_ar_iterations
            # Update valid idxs and rel_idx_dictionaries 
            self.update_indexing()

    ##------------------------------------------------------------------------.
    def __len__(self):
        """Return the number of samples available."""
        return self.n_samples

    ##------------------------------------------------------------------------.
    def __getitem__(self, idx):
        """Return sample and label corresponding to an index as torch.Tensor objects."""
        # Assumptions:
        # - The code currently assume that data_dynamic and data_bc are aligned over time 
        # - rel_idx correspond to input_k and output_k (aka leadtime_idx)
        ## -------------------------------------------------------------------.
        # Retrieve current idx of xarray  
        xr_idx_k_0 = self.idxs[idx] 
        ## -------------------------------------------------------------------.
        #############################
        ### Retrieve dynamic data ###
        #############################
        # - Retrieve xarray indices 
        xr_idx_dynamic_required = xr_idx_k_0 + self.rel_idx_dynamic_required  
        # - Subset the xarray object (need for all autoregressive iterations)
        data_dynamic_subset = self.data_dynamic.isel(time=xr_idx_dynamic_required)
        # - Ensure that here after is a xr.DataArray
        if isinstance(data_dynamic_subset, xr.Dataset):
            da_dynamic_subset = data_dynamic_subset.to_array(dim='feature').transpose(...,'feature') # to_array() stack along first axis
        else: 
            da_dynamic_subset = data_dynamic_subset
        ## -------------------------------------------------------------------.    
        # - If not preloaded in CPU, load the dynamic data 
        # t_i = time.time()
        if is_dask_DataArray(da_dynamic_subset): 
            da_dynamic_subset = da_dynamic_subset.compute()    
        # print(time.time() - t_i)
        ## -------------------------------------------------------------------.
        # - Apply the scaler if provided 
        if self.scaler is not None:
            # t_i = time.time()
            da_dynamic_subset = self.scaler.transform(da_dynamic_subset, variable_dim='feature')
            # print("scaler", time.time() - t_i)
        ## -------------------------------------------------------------------.    
        # - Assign relative indices (onto the "rel_idx" dimension)
        da_dynamic_subset = da_dynamic_subset.assign_coords(rel_idx=('time', self.rel_idx_dynamic_required)).swap_dims({'time': 'rel_idx'}) 
        ## -------------------------------------------------------------------.
        ### Loop over leadtimes and store Numpy arrays in a dictionary(leadtime)
        # - Extract numpy array from DataArray and convert to Torch Tensor
        # - X_dynamic 
        dict_X_dynamic_data = {}
        for i in range(self.ar_iterations + 1): 
            # X_dynamic
            if self.dict_rel_idx_X_dynamic[i] is not None:
                dict_X_dynamic_data[i] = torch.as_tensor(torch.from_numpy(da_dynamic_subset.sel(rel_idx=self.dict_rel_idx_X_dynamic[i]).values), dtype=self.torch_dtype, device='cpu')
            else: 
                dict_X_dynamic_data[i] = None
        # - Y 
        if self.training_mode:
            dict_Y_data = {}
            for i in range(self.ar_iterations + 1): 
                dict_Y_data[i] = torch.as_tensor(torch.from_numpy(da_dynamic_subset.sel(rel_idx=self.dict_rel_idx_Y[i]).values), dtype=self.torch_dtype, device='cpu')
        else: 
            dict_Y_data = None

        ##--------------------------------------------------------------------.
        ####################################
        ## Retrieve forecast time infos ####
        ####################################  
        # - Define timedelta 
        t_res_timedelta = self.t_res_timedelta  
        # - Forecast start time (time of rel_idx = 0) 
        forecast_start_time = self.data_dynamic.isel(time=xr_idx_k_0).time.values
        # - Forecast reference time 
        reference_time_idx = xr_idx_k_0 + max(self.input_k)
        forecast_reference_time = self.data_dynamic.isel(time=reference_time_idx).time.values
        # - Forecast leadtime_idx 
        dict_forecast_rel_idx_Y = self.dict_rel_idx_Y
        # - Forecasted time and forecast leadtime 
        dict_forecasted_time = {}
        dict_forecast_leadtime = {}
        for i in range(self.ar_iterations + 1): 
            dict_forecasted_time[i] = forecast_start_time + self.dict_rel_idx_Y[i]*t_res_timedelta
            dict_forecast_leadtime[i] = dict_forecasted_time[i] - forecast_reference_time
        # - Create forecast_time_info dictionary 
        forecast_time_info = {'forecast_start_time': forecast_start_time,
                              'forecast_reference_time': forecast_reference_time,
                              'dict_forecast_rel_idx_Y': dict_forecast_rel_idx_Y,
                              'dict_forecast_leadtime': dict_forecast_leadtime, 
                              'dict_forecasted_time': dict_forecasted_time}
         
        ## -------------------------------------------------------------------.
        #######################################################
        ### Retrieve boundary conditions data (if provided) ###
        #######################################################
        dict_X_bc_data = None 
        if self.data_availability['bc']: 
            xr_idx_bc_required = xr_idx_k_0 + self.rel_idx_bc_required  
            # - Subset the xarray Datarray (need for all autoregressive iterations)
            data_bc_subset = self.data_bc.isel(time=xr_idx_bc_required)
            # - Ensure that here after is a xr.DataArray
            if isinstance(data_bc_subset, xr.Dataset):
                da_bc_subset = data_bc_subset.to_array(dim='feature').transpose(...,'feature') # to_array() stack along first axis
            else: 
                da_bc_subset = data_bc_subset
            ## ---------------------------------------------------------------.    
            # - If not preloaded in CPU, load the bc data   
            if is_dask_DataArray(da_bc_subset): 
                da_bc_subset = da_bc_subset.compute()
            ## ---------------------------------------------------------------.
            # - Apply scaler 
            if self.scaler is not None:
                da_bc_subset = self.scaler.transform(da_bc_subset, variable_dim='feature')
            ## ---------------------------------------------------------------.
            # - Assign relative indices (onto the "rel_idx" dimension)
            da_bc_subset = da_bc_subset.assign_coords(rel_idx=('time', self.rel_idx_bc_required)).swap_dims({'time': 'rel_idx'}) 
            ## ---------------------------------------------------------------.
            # - Loop over leadtimes and store Numpy arrays in a dictionary(leadtime) 
            dict_X_bc_data = {}
            for i in range(self.ar_iterations + 1): 
                # Extract numpy array from DataArray and conver to Torch Tensor
                dict_X_bc_data[i] = torch.as_tensor(torch.from_numpy(da_bc_subset.sel(rel_idx=self.dict_rel_idx_X_bc[i]).values), dtype=self.torch_dtype, device='cpu')

        ## -------------------------------------------------------------------.
        #################################################################
        ### Generate boundary conditions if bc_generator is provided) ###
        #################################################################
        if self.data_availability['bc_generator']: 
            # dict_X_bc_data
             
            raise NotImplementedError

        ## -------------------------------------------------------------------.
        # Return the sample dictionary  
        return {'X_dynamic': dict_X_dynamic_data, 
                'X_bc': dict_X_bc_data, 
                'Y': dict_Y_data, 
                'dict_Y_to_stack': self.dict_Y_to_stack,
                'dict_Y_to_remove': self.dict_Y_to_remove,
                'ar_iterations': self.ar_iterations,
                'dim_info': self.dim_info,
                'dim_order': self.dim_order, 
                'feature_info': self.feature_info,
                'feature_order' : self.feature_order,
                'forecast_time_info': forecast_time_info,
                'training_mode': self.training_mode, 
                'data_availability': self.data_availability
                }
                
##----------------------------------------------------------------------------.    

def autoregressive_collate_fn(list_samples,  
                              torch_static=None, 
                              pin_memory = False,
                              prefetch_in_gpu = False,
                              asyncronous_gpu_transfer=True, 
                              device = 'cpu'):        
    """Stack the list of samples into batch of data."""
    # TODO: why this throw error if attached to AR_Dataset as a method
    # list_samples is a list of what returned by __get_item__ of AutoregressiveDataset
    # To debug: list_samples = [dataset.__getitem__(0), dataset.__getitem__(1)]
    ##------------------------------------------------------------------------.
    # Retrieve other infos
    dict_Y_to_stack = list_samples[0]['dict_Y_to_stack']
    dict_Y_to_remove = list_samples[0]['dict_Y_to_remove']
    dim_info = list_samples[0]['dim_info']
    dim_order = list_samples[0]['dim_order']
    feature_info = list_samples[0]['feature_info']
    feature_order = list_samples[0]['feature_order']
    ar_iterations = list_samples[0]['ar_iterations']
    batch_dim = dim_info['dynamic']['sample']
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
    for i in range(ar_iterations+1):
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
        for i in range(ar_iterations+1):
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
    if data_availability['bc'] or data_availability['bc_generator'] : 
        dict_X_bc_batched = {}  
        for i in range(ar_iterations+1):
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
                  'dim_order':dim_order,
                  'feature_info': feature_info,
                  'feature_order': feature_order,
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
    # Retrieve static feature tensor (preloaded in GPU) (if available)
    torch_static = dataset.torch_static  
     
    ##------------------------------------------------------------------------. 
    # Expand static feature tensor (along the batch dimension) 
    if torch_static is not None:
        dim_info = dataset.dim_info['static']
        batch_dim = dim_info['sample']
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
    ar_iterations = dataset.ar_iterations
    device = dataset.device                      
    # Retrieve function to get time 
    get_time = get_time_function(device)
    # Retrieve custom ar_batch_fun fuction
    ar_batch_fun = dataset.ar_batch_fun  
    ##------------------------------------------------------------------------.
    # Initialize list 
    dataloader_timing = []  
    ar_batch_timing = []  
    total_timing = []
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
        dataloader_timing.append(get_time() - t_i)
        # Perform AR iterations 
        dict_training_Y_predicted = {}
        ##----------------------------------------------------------------.
        # Initialize stuff for AR loop timing 
        tmp_ar_data_removal_timing = 0
        tmp_ar_batch_timing = 0 
        batch_memory_size = 0 
        for i in range(ar_iterations+1):
            # Retrieve X and Y for current AR iteration   
            t_i = get_time()
            torch_X, torch_Y = ar_batch_fun(ar_iteration = i, 
                                            batch_dict = training_batch_dict, 
                                            dict_Y_predicted = dict_training_Y_predicted,
                                            device = device, 
                                            asyncronous_gpu_transfer = asyncronous_gpu_transfer)
            tmp_ar_batch_timing = tmp_ar_batch_timing + (get_time() - t_i)
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
            remove_unused_Y(ar_iteration = i, 
                            dict_Y_predicted = dict_training_Y_predicted,
                            dict_Y_to_remove = training_batch_dict['dict_Y_to_remove'])
            del torch_X, torch_Y
            if i == ar_iterations:
                del dict_training_Y_predicted
            tmp_ar_data_removal_timing = tmp_ar_data_removal_timing + (get_time()- t_i)
                
        ##----------------------------------------------------------------.
        # Summarize timing 
        ar_batch_timing.append(tmp_ar_batch_timing - tmp_ar_data_removal_timing)

        ##----------------------------------------------------------------.
        # - Total time elapsed
        total_timing.append(ar_batch_timing[-1] + dataloader_timing[-1])

    ##------------------------------------------------------------------------.
    # Create timing info dictionary 
    timing_info = {'Run': list(range(n_repetitions)), 
                   'Total': total_timing, 
                   'Dataloader': dataloader_timing,
                   'AR Batch': ar_batch_timing,
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
                         round(total_timing[count], 4),
                         round(dataloader_timing[count], 4),
                         round(ar_batch_timing[count], 4),
                          ])
        print(tabulate(table, headers=headers))   
    ##------------------------------------------------------------------------.               
    return timing_info, memory_info
