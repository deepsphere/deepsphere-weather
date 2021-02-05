#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 23:56:40 2021

@author: ghiggi
"""
import numpy as np 
import xarray as xr

def is_dask_DataArray(da):
    """Check if data in the xarray DataArray are lazy loaded."""
    if da.chunks is not None:
        return True
    else: 
        return False

def check_no_missing_timesteps(timesteps, verbose=True):
    """Check if there are missing timesteps in a numpy datetime64 array."""
    dt = np.diff(timesteps)
    dts, counts = np.unique(dt, return_counts=True)
    if verbose is True:
        print("Starting at", timesteps[0])
        print("Ending at", timesteps[-1])
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

def check_DataArray_dim_names(da, dim_names, da_name):
    """Check dim_names are dimensions of the DataArray."""
    if not isinstance(da, xr.DataArray):
        raise TypeError("'da' must be an xarray DataArray")
    if not isinstance(dim_names, list):
        raise TypeError("'dim_names' must be a list")
    da_dims = list(da.dims)
    for dim_name in dim_names: 
        if dim_name not in da_dims:
            raise ValueError("The {} must contain the '{}' dimension".format(da_name, dim_name))
  
def check_DataArray_dimnames(da_dynamic = None,
                             da_bc = None, 
                             da_static = None):
    """Check the dimension names of DataArray required for AR training."""
    # Check for dimensions of the dynamic DataArray
    if da_dynamic is not None: 
        required_dim_names = np.array(['node','time','feature'])
        dim_names = list(da_dynamic.dims.keys())
        missing_dims = required_dim_names[np.isin(required_dim_names, dim_names, invert=True)]
        if len(missing_dims) > 0: 
            raise ValueError("The dynamic DataArray must have the {} dimensions".format(missing_dims))
   
    # Check for dimensions of the boundary conditions DataArray           
    if da_static is not None: 
        required_dim_names = ['node', 'feature']
        dim_names = list(da_static.dims.keys())
        missing_dims = required_dim_names[np.isin(required_dim_names, dim_names, invert=True)]
        if len(missing_dims) > 0: 
            raise ValueError("The static DataArray must have the 'node' dimension")
   
    # Check for dimension of the static DataArray      
    if da_bc is not None: 
        required_dim_names = ['node', 'time','feature'] 
        dim_names = list(da_bc.dims.keys())
        missing_dims = required_dim_names[np.isin(required_dim_names, dim_names, invert=True)]
        if len(missing_dims) > 0: 
            raise ValueError("The boundary conditions DataArray must have the {} dimensions".format(missing_dims))
   
    
def check_Dataset_dimnames(ds_dynamic = None,
                           ds_bc = None, 
                           ds_static = None):
    """Check the dimension names of Datasets required for AR training."""
    # Check for dimensions of the dynamic Dataset
    if ds_dynamic is not None: 
        required_dim_names = np.array(['node','time'])
        dim_names = list(ds_dynamic.dims.keys())
        missing_dims = required_dim_names[np.isin(required_dim_names, dim_names, invert=True)]
        if len(missing_dims) > 0: 
            raise ValueError("The dynamic Dataset must have the {} dimensions".format(missing_dims))
   
    # Check for dimensions of the boundary conditions Dataset           
    if ds_static is not None: 
        required_dim_names = np.array(['node'])
        dim_names = list(ds_static.dims.keys())
        missing_dims = required_dim_names[np.isin(required_dim_names, dim_names, invert=True)]
        if len(missing_dims) > 0: 
            raise ValueError("The static Dataset must have the 'node' dimension")
        if len(dim_names) > 1: 
            raise ValueError("The static Dataset must have just the 'node' dimension")
            
    # Check for dimension of the static Dataset      
    if ds_bc is not None: 
        required_dim_names = np.array(['node', 'time'])
        dim_names = list(ds_bc.dims.keys())
        missing_dims = required_dim_names[np.isin(required_dim_names, dim_names, invert=True)]
        if len(missing_dims) > 0: 
            raise ValueError("The boundary conditions Dataset must have the {} dimensions".format(missing_dims))

##----------------------------------------------------------------------------.
def check_Datasets(ds_training_dynamic,
                   ds_validation_dynamic = None,
                   ds_static = None,              
                   ds_training_bc = None,         
                   ds_validation_bc = None):
    """Check Datasets required for AR training."""
    # Check dimensions 
    check_Dataset_dimnames(ds_dynamic=ds_training_dynamic)
    check_Dataset_dimnames(ds_dynamic=ds_validation_dynamic)
    check_Dataset_dimnames(ds_bc=ds_training_bc)
    check_Dataset_dimnames(ds_bc=ds_validation_bc)
    check_Dataset_dimnames(ds_static=ds_static)
    ##------------------------------------------------------------------------.
    # Check that the required Datasets are provided 
    if ds_validation_dynamic is not None: 
        if ((ds_training_bc is not None) and (ds_validation_bc is None)):  
            raise ValueError("If boundary conditions data are provided for the training, must be provided also for validation!")
    ##------------------------------------------------------------------------.
    # Check no missing timesteps 
    check_no_missing_timesteps(ds_training_dynamic['time'].values)
    if ds_validation_dynamic is not None: 
        check_no_missing_timesteps(ds_validation_dynamic['time'].values, verbose=False)
    if ds_validation_dynamic is not None: 
        check_no_missing_timesteps(ds_validation_dynamic['time'].values, verbose=False)
    if ds_training_bc is not None: 
        check_no_missing_timesteps(ds_training_bc['time'].values, verbose=False)
    if ds_validation_bc is not None: 
        check_no_missing_timesteps(ds_validation_bc['time'].values, verbose=False)
    ##------------------------------------------------------------------------.
    # Check time alignement of training and validation dataset
    if ds_training_bc is not None: 
        same_timesteps = ds_training_dynamic['time'].values == ds_training_bc['time'].values
        if not all(same_timesteps):
            raise ValueError("The training dynamic Dataset and the training boundary conditions Dataset does not have the same timesteps!")
    if ((ds_validation_dynamic is not None) and (ds_validation_bc is not None)): 
        same_timesteps = ds_validation_dynamic['time'].values == ds_validation_bc['time'].values
        if not all(same_timesteps):
            raise ValueError("The validation dynamic Dataset and the validation boundary conditions Dataset does not have the same timesteps!")
    ##------------------------------------------------------------------------.

##----------------------------------------------------------------------------.  
# Retrieve input-output dims 
def get_AR_model_diminfo(da_dynamic, da_static=None, da_bc=None, AR_settings=None):
    """Retrieve dimension information for AR DeepSphere models.""" 
    # Required dimensions
    time_dim='time'
    node_dim='node'
    variable_dim='feature'
    ##------------------------------------------------------------------------.
    # Dynamic variables 
    check_DataArray_dim_names(da = da_dynamic, da_name = "dynamic DataArray",
                              dim_names=[time_dim, node_dim, variable_dim])
    dynamic_variables = da_dynamic[variable_dim].values.tolist()
    n_dynamic_variables = len(dynamic_variables)
    dims_dynamic = list(da_dynamic.dims)
    # Static variables 
    if da_static is not None:
        check_DataArray_dim_names(da = da_static, da_name = "static DataArray",
                              dim_names=[node_dim, variable_dim])
        dims_static = list(da_static.dims)
        static_variables = da_static[variable_dim].values.tolist()
        n_static_variables = len(static_variables)
    else:
        dims_static = None
        static_variables = []
        n_static_variables = 0
        
    # Boundary condition variables     
    if da_bc is not None:
        check_DataArray_dim_names(da = da_bc, da_name = "bc DataArray",
                                  dim_names=[time_dim, node_dim, variable_dim])
        dims_bc = list(da_bc.dims)
        bc_variables = da_bc[variable_dim].values.tolist()
        n_bc_variables = len(bc_variables)
    else: 
        dims_bc = None
        bc_variables = []
        n_bc_variables = 0
    ##-------------------------------------------------------------------------.
    # Check dims_bc order is the same as dims_dynamic
    if dims_bc is not None:
        if not np.array_equal(dims_dynamic, dims_bc):
            raise ValueError("Dimension order of dynamic and bc DataArrays must be equal.")
      
    ##------------------------------------------------------------------------. 
    # Define feature dimensions 
    input_feature_dim = n_static_variables + n_bc_variables + n_dynamic_variables 
    output_feature_dim = n_dynamic_variables
    input_features = static_variables + bc_variables + dynamic_variables                     
    output_features = dynamic_variables
    ##------------------------------------------------------------------------. 
    # Define number of nodes 
    input_node_dim = len(da_dynamic['node'])
    output_node_dim = len(da_dynamic['node'])
    ##------------------------------------------------------------------------. 
    # Define dimension order
    dim_order = ['sample'] + list(da_dynamic.dims) 
    ##------------------------------------------------------------------------. 
    # Define time dimension 
    if AR_settings is not None:
        input_time_dim = len(AR_settings['input_k']) 
        output_time_dim = len(AR_settings['output_k']) 
        # Create dictionary with dimension infos 
        dim_info = {'input_feature_dim': input_feature_dim,
                    'output_feature_dim': output_feature_dim,
                    'input_features': input_features,
                    'output_features': output_features,
                    'input_time_dim': input_time_dim,
                    'output_time_dim': output_time_dim,
                    'input_node_dim': input_node_dim,
                    'output_node_dim': output_node_dim,
                    'dim_order': dim_order,
                    }
    else:
        # Create dictionary with dimension infos 
        dim_info = {'input_feature_dim': input_feature_dim,
                    'output_feature_dim': output_feature_dim,
                    'input_features': input_features,
                    'output_features': output_features,
                    'input_node_dim': input_node_dim,
                    'output_node_dim': output_node_dim,
                    'dim_order': dim_order,
                    }    
    ##------------------------------------------------------------------------. 
    return dim_info

def check_DataArrays_dimensions(da_training_dynamic,
                                da_validation_dynamic = None, 
                                da_training_bc = None,
                                da_validation_bc = None, 
                                da_static = None):
    
    dim_info_training = get_AR_model_diminfo(da_dynamic = da_training_dynamic, 
                                             da_bc = da_training_bc,
                                             da_static = da_static)
    if da_validation_dynamic is not None:
        dim_info_validation = get_AR_model_diminfo(da_dynamic = da_validation_dynamic, 
                                                   da_bc = da_validation_bc,
                                                   da_static = da_static)
        if not dim_info_training == dim_info_validation:
            raise ValueError("The dimension order of training and validation DataArrays do not coincide!")
    