#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 23:56:40 2021

@author: ghiggi
"""
import datetime
import numpy as np 
import xarray as xr
import warnings
from modules.utils_xr import xr_have_Dataset_vars_same_dims

#------------------------------------------------------------------------------.
#########################
### Timesteps checks ####
#########################

def check_timesteps_format(timesteps): 
    """Ensure timesteps format is numpy array of numpy.datetime64."""
    if timesteps is None:
        return None
    if not isinstance(timesteps, (str, list, np.ndarray, datetime.datetime, np.datetime64)):
        raise ValueError("The timestep(s) must be provided as a str, list, np.array of datetime or np.datetime64.")
    if isinstance(timesteps, str): 
        timesteps = np.array([timesteps], dtype='M8')
    if isinstance(timesteps, list): 
         timesteps = np.array(timesteps, dtype='M8')
    if isinstance(timesteps, np.datetime64):
        timesteps = np.array([timesteps])
    if isinstance(timesteps, datetime.datetime):
        timesteps = np.array([timesteps], dtype='M8')
    if isinstance(timesteps, np.ndarray): 
        if len(timesteps.shape) == 0:  # array('2019-01-01', dtype='datetime64) --> fail len(timesteps)
            timesteps = np.array([timesteps.tolist()],dtype="M8")
    # Ensure is datetime64
    timesteps = timesteps.astype('M8')
    return timesteps

def check_no_duplicate_timesteps(timesteps, var_name="timesteps"):
    """Check if there are missing timesteps in a list or numpy datetime64 array."""
    n_timesteps = len(timesteps) 
    n_unique_timesteps = len(np.unique(timesteps))
    if n_timesteps != n_unique_timesteps:
        raise ValueError("{!r} contains non-unique timesteps".format(var_name)) 
    return None

def _check_timesteps(timesteps):
    """Check timesteps object and return a numpy array."""
    # - Check is not None
    if timesteps is None: 
        raise ValueError("'timesteps' is None.")
    # - Ensure format np.array([...], dtype='M8')
    try: 
        timesteps = check_timesteps_format(timesteps)
    except ValueError: 
        raise ValueError("Unvalid 'timesteps' specification.")
    # - Check there are timesteps 
    if len(timesteps) == 0: 
        raise ValueError("'timesteps' is empty.")
    # - Return timesteps 
    return timesteps        

def check_no_missing_timesteps(timesteps, verbose=True):
    """Check if there are missing timesteps in a list or numpy datetime64 array."""
    timesteps = _check_timesteps(timesteps) 
    # Check if there are data
    if timesteps.size == 0: 
        raise ValueError("No data available !")
    # Check if missing timesteps 
    dt = np.diff(timesteps)
    dts, counts = np.unique(dt, return_counts=True)
    if verbose:
        print("  --> Starting at", timesteps[0])
        print("  --> Ending at", timesteps[-1])
    if (len(counts) > 1):
        print("Missing data between:")
        bad_dts = dts[counts != counts.max()] 
        for bad_dt in bad_dts:
            bad_idxs = np.where(dt == bad_dt)[0]
            bad_idxs = [b.tolist() for b in bad_idxs]
            for bad_idx in bad_idxs:
                tt_missings = timesteps[bad_idx:(bad_idx+2)]
                print("-", tt_missings[0], "and", tt_missings[1])
        raise ValueError("The process has been interrupted") 
    return 

#------------------------------------------------------------------------------.
###########################
### Data values checks ####
###########################

def check_finite_Dataset(ds):
    """Check Dataset does not contain NaN and Inf values."""
    # Check is a Dataset
    if not isinstance(ds, xr.Dataset):
        raise TypeError("'ds' must be an xarray Dataset.")
    # Check no NaN values
    ds_isnan = xr.ufuncs.isnan(ds)  
    list_vars_with_nan = []
    flag_raise_error = False
    for var in list(ds_isnan.data_vars.keys()):
        if ds_isnan[var].sum().values != 0:
            list_vars_with_nan.append(var)
            flag_raise_error = True
    if flag_raise_error: 
        raise ValueError('The variables {} contain NaN values'.format(list_vars_with_nan))
    # Check no Inf values
    ds_isinf = xr.ufuncs.isinf(ds)  
    list_vars_with_inf = []
    flag_raise_error = False
    for var in list(ds_isinf.data_vars.keys()):
        if ds_isinf[var].sum().values != 0:
            list_vars_with_inf.append(var)
            flag_raise_error = True
    if flag_raise_error: 
        raise ValueError('The variables {} contain Inf values.'.format(list_vars_with_inf))
        
#------------------------------------------------------------------------------.
############################################
### Check data format for AR dataloader ####
############################################

def _check_input_data(data, feature_dim = 'feature'):  
    if not isinstance(data, (xr.DataArray, xr.Dataset)):
        raise TypeError("Expecting xr.DataArray or xr.Dataset")
    # If Dataset, must not have feature dimension
    if isinstance(data, xr.Dataset):
        if feature_dim in data.dims:
            raise ValueError("The 'xr.Dataset' cannot contain dimension 'feature'")
    return None

def _check_has_feature_dimension(data, feature_dim="feature"): 
     if feature_dim not in data.dims:
            raise ValueError("The '{!r}' must have a dimension called 'feature'".format(type(data)))
     return None 

def _check_has_time_dimension(data, time_dim = "time"): 
     if time_dim not in data.dims:
            raise ValueError("The '{!r}' must have a dimension called 'time'".format(type(data)))
     return None 

def _check_temporal_data(data, data_type, feature_dim='feature', time_dim = 'time', verbose=False):
    if data is None: 
        return None
    # - Checks data type 
    _check_input_data(data)
    # - Check no missing timesteps 
    check_no_missing_timesteps(timesteps=data[time_dim].values, verbose=verbose)
    # - Check that "time" is a dimension 
    _check_has_time_dimension(data, time_dim = time_dim)
    # - If Dataset, check that all DataArray have same dimensions
    if isinstance(data, xr.Dataset): 
        if not xr_have_Dataset_vars_same_dims(data):
            raise ValueError("All xr.DataArray(s) within the {!r} xr.Dataset must have the same dimensions.".format(data_type)) 
    # - If Dataset, conversion to DataArray for further checks 
    if isinstance(data, xr.Dataset):
        data = data.to_array(feature_dim) 
    # - Check that 'feature' is a dimension 
    _check_has_feature_dimension(data, feature_dim = feature_dim)
    # - Put feature as the last position  The Dataset/Dataloader will 
    data = data.transpose(..., feature_dim)
    # - Retrieve DataArray dimensions (and number)
    dims = list(data.dims)
    # - Retrieve non-required dimensions 
    dims_required=[time_dim, feature_dim]
    dims_optional = np.array(dims)[np.isin(dims, dims_required, invert=True)].tolist()
    if len(dims_optional) < 1:
        raise ValueError("{!r} must have at least one additional dimension (i.e. space) other than {!r}".format(data_type, dims_required))
    return None

def _check_static_data(data, data_type, feature_dim='feature', time_dim = 'time'):
    if data is None:
        return None
    # - Checks data type 
    _check_input_data(data)
    # - Check does not have "time" dimension
    if time_dim in list(data.dims):
        raise ValueError("{!r} must not contain the 'time' dimension.".format(data_type))
    # - If Dataset, check that all DataArray have same dimensions
    if isinstance(data, xr.Dataset): 
        if not xr_have_Dataset_vars_same_dims(data):
            raise ValueError("All xr.DataArray(s) within the {!r} xr.Dataset must have the same dimensions.".format(data_type)) 
    # - If Dataset, conversion to DataArray for further checks 
    if isinstance(data, xr.Dataset):
        data = data.to_array(feature_dim) 
    # - Check that 'feature' is a dimension 
    _check_has_feature_dimension(data, feature_dim = feature_dim)
    # - Put feature as the last position  The Dataset/Dataloader will 
    data = data.transpose(...,feature_dim)
     # - Retrieve DataArray dimensions (and number)
    dims = list(data.dims)
    # - Retrieve non-required dimensions 
    dims_required = [feature_dim]
    dims_optional = np.array(dims)[np.isin(dims, dims_required, invert=True)].tolist()
    if len(dims_optional) < 1:
        raise ValueError("{!r} must have at least one additional dimension (i.e. space) other than {!r}".format(data_type, dims_required))
    return None

#------------------------------------------------------------------------------.
#############################
#### Tensor Info Getters ####
#############################
def _get_subset_timesteps_idxs(timesteps, subset_timesteps, strict_match=True):
    """Check subset_timesteps are within timesteps and return the matching indices."""
    # Check timesteps format 
    subset_timesteps = _check_timesteps(subset_timesteps)
    timesteps = _check_timesteps(timesteps)
    # Ensure same time precision for comparison 
    subset_timesteps = subset_timesteps.astype(timesteps.dtype)  
    # Retrieve idxs corresponding to subset_timesteps 
    subset_idxs = np.array([idx for idx, v in enumerate(timesteps) if v in set(subset_timesteps)])  
    # If no idxs available,means that subset_timesteps are not in the period covered by 'data_dyamic
    if subset_idxs.size == 0:
        raise ValueError("All 'forecast_reference_times' are not within the time period covered by 'data_dynamic'.")
    if len(subset_idxs) != len(subset_timesteps):
        timesteps_not_in = subset_timesteps[np.isin(subset_timesteps, timesteps, invert=True)] 
        if strict_match:
            raise ValueError("The following 'forecast_start_time(s)' are not within the time period covered by 'data_dynamic': {}".format(list(timesteps_not_in)))      
        else:
            warnings.warn("The following 'forecast_start_time(s)' are not within the time period covered by 'data_dynamic': {}".format(list(timesteps_not_in)))
    return subset_idxs

def _get_dim_order(data):
    # If None, return None
    if data is None: 
        return None
    # If DataArray retrieve dimension position 
    dims = list(data.dims) 
    if isinstance(data, xr.DataArray):
        if 'feature' not in dims:
            raise ValueError("The 'feature' dimension is required in a DataArray.")
        # If last dimension is not 'feature', move to last position (it's done in the dataset/dataloader)
        if dims[-1] != 'feature': 
            warnings.warn("The last dimension of a DataArray should be 'feature'.")
            dims = [dim for dim in dims if dim != 'feature']
            dims.append('feature')
    # If Dataset data.dims does not correspond to the dimension of within DataArrays !!!
    if isinstance(data, xr.Dataset):
        # - Here I assume that all within DataArrays have same dimension order !!!
        # --> xr_have_Dataset_vars_same_dims(data) must be performed before !
        dims = list(data[list(data.data_vars.keys())[0]].dims)
        dims = dims + ['feature']
    dim_order = ['sample'] + dims   
    return dim_order

def _get_dim_info(data):
    # If None, return None
    if data is None: 
        return None
    dim_order = _get_dim_order(data)
    dim_info = {k: i for i, k in enumerate(dim_order)}
    return dim_info

def _get_feature_order(data):
    # If None, return None
    if data is None: 
        return None
    # If DataArray or Dataset, retrieve features 
    if isinstance(data, xr.Dataset):
        feature_order = list(data.data_vars.keys())   
    elif isinstance(data, xr.DataArray):
        feature_order = data['feature'].values.tolist()    
    else:
        raise NotImplementedError
    # Ensure that all feature string are str and not np.str_ 
    feature_order = [str(f) for f in feature_order]
    # Return feature order 
    return feature_order

def _get_feature_info(data):
    # If None, return None
    if data is None: 
        return None
    # If DataArray or Dataset, retrieve features 
    feature_order = _get_feature_order(data)
    feature_info = {k: i for i, k in enumerate(feature_order)}
    return feature_info

def _get_shape_order(data):
    # If None, return None
    if data is None: 
        return None
    # If DataArray or Dataset, retrieve features 
    if isinstance(data, xr.Dataset):
        data = data.to_array('feature')
    # Ensure 'feature' is the last dimension 
    data = data.transpose(..., 'feature')
    # Retrieve shape
    shape_order = list(data.shape)
    shape_order = [None] + shape_order
    return shape_order

def _get_shape_order_dicts(data_dynamic, data_bc, data_static): 
     shape_order = {}
     shape_order['dynamic'] =_get_shape_order(data_dynamic)
     shape_order['static'] =_get_shape_order(data_static)
     shape_order['bc'] = _get_shape_order(data_bc)
     return shape_order

def _get_feature_info_dicts(data_dynamic, data_bc, data_static): 
     feature_info = {}
     feature_info['dynamic'] =_get_feature_info(data_dynamic)
     feature_info['static'] =_get_feature_info(data_static)
     feature_info['bc'] = _get_feature_info(data_bc)
     return feature_info

def _get_feature_order_dicts(data_dynamic, data_bc, data_static): 
     feature_order = {}
     feature_order['dynamic'] =_get_feature_order(data_dynamic)
     feature_order['static'] =_get_feature_order(data_static)
     feature_order['bc'] = _get_feature_order(data_bc) 
     return feature_order

def _get_dim_info_dicts(data_dynamic, data_bc, data_static): 
     dim_info = {}
     dim_info['dynamic'] =_get_dim_info(data_dynamic)
     dim_info['static'] =_get_dim_info(data_static)
     dim_info['bc'] = _get_dim_info(data_bc)
     return dim_info

def _get_dim_order_dicts(data_dynamic, data_bc, data_static): 
        dim_order = {}
        dim_order['dynamic'] =_get_dim_order(data_dynamic)
        dim_order['static'] =_get_dim_order(data_static)
        dim_order['bc'] = _get_dim_order(data_bc)
        return dim_order  

# Retrieve input-output dims 
def get_ar_model_tensor_info(ar_settings, data_dynamic,
                             data_static=None, 
                             data_bc=None, bc_generator=None):
    """Retrieve dimension information for AR DeepSphere models.""" 
    ##------------------------------------------------------------------------.
    ### Checks ar_settings
    if not isinstance(ar_settings, dict): 
        raise ValueError("Please provide 'ar_settings' as a dictionary")
    ar_settings_keys = list(ar_settings.keys())
    if not np.all(np.isin(['input_k','output_k'], ar_settings_keys)):
        raise ValueError("The 'ar_settings' dictionary must contain 'input_k' and 'output_k' keys.")
    ##------------------------------------------------------------------------.
    ### Checks either data_bc or bc_generator is provided
    if data_bc is not None and bc_generator is not None: 
        raise ValueError("Either provide 'data_bc' or 'bc_generator'.")
    ##------------------------------------------------------------------------.
    # Define name of required dimensions
    time_dim = 'time'
    feature_dim = 'feature'
    ##------------------------------------------------------------------------.
    ### Checks data_dynamic 
    if data_dynamic is None: 
        raise ValueError("'data_dynamic' cannot be None! Provide an xr.Dataset or xr.DataArray.")
    _check_temporal_data(data_dynamic, data_type = 'data_dynamic', 
                         feature_dim=feature_dim, time_dim = time_dim)
    ##------------------------------------------------------------------------.
    ### Check data_bc      
    if bc_generator is not None:
        # TODO
        # data_bc = ... 
        raise NotImplementedError()
    _check_temporal_data(data_bc, data_type = 'data_bc', 
                        feature_dim=feature_dim, time_dim = time_dim)
     
    ##------------------------------------------------------------------------.
    ### Checks data_static  
    _check_static_data(data_static, data_type = 'data_statuc', 
                       feature_dim=feature_dim, time_dim = time_dim)

    ##------------------------------------------------------------------------. 
    # Define shape of the input and output time dimensions
    input_n_time = len(ar_settings['input_k']) 
    output_n_time = len(ar_settings['output_k']) 
    
    ##------------------------------------------------------------------------. 
    ### Create informative objects 
    # - Retrieve feature and dimension position dictionaries 
    dim_info = _get_dim_info_dicts(data_dynamic, data_bc, data_static)
    dim_order = _get_dim_order_dicts(data_dynamic, data_bc, data_static)   
    feature_info = _get_feature_info_dicts(data_dynamic, data_bc, data_static)
    feature_order = _get_feature_order_dicts(data_dynamic, data_bc, data_static)   
   
    feature_dynamic = feature_order['dynamic'] if feature_order['dynamic'] is not None else []   
    feature_bc = feature_order['bc'] if feature_order['bc'] is not None else []      
    feature_static = feature_order['static'] if feature_order['static'] is not None else []   
    feature_total = feature_dynamic + feature_bc + feature_static  

    # Define total number of features 
    input_n_feature = len(feature_total)
    output_n_feature = len(feature_dynamic) 

    # - Retrieve shape info (with batch shape None) for input and output tensors 
    # - The batch dimension is marked by a None 
    shape_order = _get_shape_order_dicts(data_dynamic, data_bc, data_static)
    
    input_shape_info = {}
    for data_type in ['dynamic','bc','static']:
        if dim_order[data_type] is not None:
            input_shape_info[data_type] = {k: v for k, v in zip(dim_order[data_type], shape_order[data_type])}
            if data_type != "static":
                input_shape_info[data_type]['time'] = input_n_time
        else: 
            input_shape_info[data_type] = None
    
    output_shape_info = {}
    for data_type in ['dynamic','bc','static']:
        if data_type == "dynamic":
            output_shape_info[data_type] = {k: v for k, v in zip(dim_order[data_type], shape_order[data_type])}
            output_shape_info[data_type]['time'] = output_n_time
        else: 
            output_shape_info[data_type] = None
        
    ##------------------------------------------------------------------------. 
    # Define input-ouput tensor shape 
    input_shape = [v if k != "feature" else input_n_feature for k, v in  input_shape_info['dynamic'].items()]
    output_shape = [v if k != "feature" else output_n_feature for k, v in output_shape_info['dynamic'].items()]
    ##------------------------------------------------------------------------.
    # Create dictionary with dimension infos 
    tensor_info = {'input_shape': input_shape,
                   'output_shape': output_shape,
                   'input_n_time': input_n_time,
                   'output_n_time': output_n_time,
                   'input_n_feature': input_n_feature,
                   'output_n_feature': output_n_feature,
                   'dim_order': dim_order,
                   'dim_info': dim_info, 
                   'feature_order': feature_order,
                   'feature_info': feature_info,
                   'input_shape_info': input_shape_info,
                   'output_shape_info': output_shape_info,
                   }
    ##------------------------------------------------------------------------. 
    return tensor_info 
