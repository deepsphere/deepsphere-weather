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

def check_dimnames_DataArray(da, required_dimnames, da_name):
    """Check dimnames are dimensions of the DataArray."""
    if not isinstance(da_name, str): 
        raise TypeError("'da_name' must be a string.")
    if not isinstance(da, xr.DataArray):
        raise TypeError("'da' must be an xarray DataArray")
    if not isinstance(required_dimnames, list):
        raise TypeError("'required_dimnames' must be a list")
    # Retrieve DataArray dimension names 
    da_dims = list(da.dims)
    # Identify which dimension are missing 
    missing_dims = np.array(required_dimnames)[np.isin(required_dimnames, da_dims, invert=True)]
    # If missing, raise an error
    if len(missing_dims) > 0: 
        raise ValueError("The {} must have also the '{}' dimension".format(da_name, missing_dims))

def check_dimnames_Dataset(ds, required_dimnames, ds_name):
    """Check dimnames are dimensions of the Dataset."""
    if not isinstance(ds_name, str): 
        raise TypeError("'ds_name' must be a string.")
    if not isinstance(ds, xr.Dataset):
        raise TypeError("'ds' must be an xarray Dataset")
    if not isinstance(required_dimnames, list):
        raise TypeError("'required_dimnames' must be a list")
    # Retrieve Dataset dimension names 
    ds_dims = list(ds.dims.keys())    
    # Identify which dimension are missing 
    missing_dims = np.array(required_dimnames)[np.isin(required_dimnames, ds_dims, invert=True)]
    # If missing, raise an error
    if len(missing_dims) > 0: 
        raise ValueError("The {} must have also the '{}' dimension".format(ds_name, missing_dims))
        
#-----------------------------------------------------------------------------.
def _check_AR_DataArray_dimnames(da_dynamic = None,
                                 da_bc = None, 
                                 da_static = None):
    """Check the dimension names of DataArray required for AR training and predictions."""
    # Required dimensions (names)
    time_dim='time'
    node_dim='node'
    variable_dim='feature'
    # Check for dimensions of the dynamic DataArray
    if da_dynamic is not None: 
        check_dimnames_DataArray(da = da_dynamic, da_name = "dynamic DataArray",
                                 required_dimnames=[time_dim, node_dim, variable_dim])
   
    # Check for dimensions of the boundary conditions DataArray           
    if da_static is not None: 
        check_dimnames_DataArray(da = da_static, da_name = "static DataArray",
                                 required_dimnames=[node_dim, variable_dim])
   
    # Check for dimension of the static DataArray      
    if da_bc is not None: 
        check_dimnames_DataArray(da = da_bc, da_name = "bc DataArray",
                                 required_dimnames=[time_dim, node_dim, variable_dim])

def check_AR_DataArrays(da_training_dynamic,
                        da_validation_dynamic = None, 
                        da_training_bc = None,
                        da_validation_bc = None, 
                        da_static = None,
                        verbose = False):
    """Check DataArrays required for AR training and predictions."""
    # Check dimension names 
    _check_AR_DataArray_dimnames(da_dynamic=da_training_dynamic,
                                 da_bc=da_training_bc,
                                 da_static=da_static)
    _check_AR_DataArray_dimnames(da_dynamic=da_validation_dynamic,
                                 da_bc=da_validation_bc,
                                 da_static=da_static)
    ##------------------------------------------------------------------------.
    # Check that the required DataArrays are provided 
    if da_validation_dynamic is not None: 
        if ((da_training_bc is not None) and (da_validation_bc is None)):  
            raise ValueError("If boundary conditions data are provided for the training, must be provided also for validation!")
    ##------------------------------------------------------------------------.
    # Check no missing timesteps 
    if verbose: 
        print("- Data time period")
    check_no_missing_timesteps(da_training_dynamic['time'].values, verbose=verbose)
    if da_validation_dynamic is not None: 
        if verbose: 
            print("- Validation Data time period")
        check_no_missing_timesteps(da_validation_dynamic['time'].values, verbose=verbose)
    if da_training_bc is not None: 
        check_no_missing_timesteps(da_training_bc['time'].values, verbose=verbose)
    if da_validation_bc is not None: 
        check_no_missing_timesteps(da_validation_bc['time'].values, verbose=verbose)
    ##------------------------------------------------------------------------.
    # Check time alignment of training and validation DataArray
    if da_training_bc is not None: 
        same_timesteps = da_training_dynamic['time'].values == da_training_bc['time'].values
        if not all(same_timesteps):
            raise ValueError("The training dynamic DataArray and the training boundary conditions DataArray does not have the same timesteps!")
    if ((da_validation_dynamic is not None) and (da_validation_bc is not None)): 
        same_timesteps = da_validation_dynamic['time'].values == da_validation_bc['time'].values
        if not all(same_timesteps):
            raise ValueError("The validation dynamic DataArray and the validation boundary conditions DataArray does not have the same timesteps!")
    ##------------------------------------------------------------------------.
    ## Check dimension order coincide
    if da_validation_dynamic is not None:
        dim_info_training = get_AR_model_diminfo(da_dynamic = da_training_dynamic, 
                                                 da_bc = da_training_bc,
                                                 da_static = da_static)
        dim_info_validation = get_AR_model_diminfo(da_dynamic = da_validation_dynamic, 
                                                   da_bc = da_validation_bc,
                                                   da_static = da_static)
        if not dim_info_training == dim_info_validation:
            raise ValueError("The dimension order of training and validation DataArrays do not coincide!")
            
#-----------------------------------------------------------------------------.
# #############################
### Checks for AR Datasets ####
# #############################    
def _check_AR_Dataset_dimnames(ds_dynamic = None,
                               ds_bc = None, 
                               ds_static = None):
    """Check the dimension names of Datasets required for AR training and predictions."""
    # Required dimensions (names)
    time_dim='time'
    node_dim='node'
    ##------------------------------------------------------------------------.
    # Check for dimensions of the dynamic Dataset
    if ds_dynamic is not None: 
        check_dimnames_Dataset(ds = ds_dynamic, ds_name = "dynamic Dataset",
                               required_dimnames=[time_dim, node_dim])
   
    # Check for dimensions of the static Dataset              
    if ds_static is not None: 
        check_dimnames_Dataset(ds = ds_static, ds_name = "static Dataset",
                               required_dimnames=[node_dim])
   
    # Check for dimension of the boundary conditions Dataset     
    if ds_bc is not None: 
        check_dimnames_Dataset(ds = ds_bc, ds_name = "bc Dataset",
                               required_dimnames=[time_dim, node_dim])   
 

##----------------------------------------------------------------------------.
def check_AR_Datasets(ds_training_dynamic,
                      ds_validation_dynamic = None,
                      ds_static = None,              
                      ds_training_bc = None,         
                      ds_validation_bc = None,
                      verbose=False):
    """Check Datasets required for AR training and predictions."""
    # Check dimension names 
    _check_AR_Dataset_dimnames(ds_dynamic=ds_training_dynamic,
                               ds_bc=ds_training_bc,
                               ds_static=ds_static)
    _check_AR_Dataset_dimnames(ds_dynamic=ds_validation_dynamic,
                               ds_bc=ds_validation_bc,
                               ds_static=ds_static)
    ##------------------------------------------------------------------------.
    # Check that the required Datasets are provided 
    if ds_validation_dynamic is not None: 
        if ((ds_training_bc is not None) and (ds_validation_bc is None)):  
            raise ValueError("If boundary conditions data are provided for the training, must be provided also for validation!")
    ##------------------------------------------------------------------------.
    # Check no missing timesteps 
    if verbose: 
        print("Data")
    check_no_missing_timesteps(ds_training_dynamic['time'].values, verbose=verbose)
    if ds_validation_dynamic is not None: 
        if verbose: 
            print("Validation Data")
        check_no_missing_timesteps(ds_validation_dynamic['time'].values, verbose=verbose)
    if ds_training_bc is not None: 
        check_no_missing_timesteps(ds_training_bc['time'].values, verbose=False)
    if ds_validation_bc is not None: 
        check_no_missing_timesteps(ds_validation_bc['time'].values, verbose=False)
    ##------------------------------------------------------------------------.
    # Check time alignment of training and validation dataset
    if ds_training_bc is not None: 
        same_timesteps = ds_training_dynamic['time'].values == ds_training_bc['time'].values
        if not all(same_timesteps):
            raise ValueError("The training dynamic Dataset and the training boundary conditions Dataset does not have the same timesteps!")
    if ((ds_validation_dynamic is not None) and (ds_validation_bc is not None)): 
        same_timesteps = ds_validation_dynamic['time'].values == ds_validation_bc['time'].values
        if not all(same_timesteps):
            raise ValueError("The validation dynamic Dataset and the validation boundary conditions Dataset does not have the same timesteps!")
    ##------------------------------------------------------------------------.

#-----------------------------------------------------------------------------.   
# Retrieve input-output dims 
def get_AR_model_diminfo(da_dynamic, da_static=None, da_bc=None, AR_settings=None):
    """Retrieve dimension information for AR DeepSphere models.""" 
    # Required dimensions
    time_dim='time'
    node_dim='node'
    variable_dim='feature'
    ##------------------------------------------------------------------------.
    # Dynamic variables 
    check_dimnames_DataArray(da = da_dynamic, da_name = "dynamic DataArray",
                             required_dimnames=[time_dim, node_dim, variable_dim])
    dynamic_variables = da_dynamic[variable_dim].values.tolist()
    n_dynamic_variables = len(dynamic_variables)
    dims_dynamic = list(da_dynamic.dims)
    # Static variables 
    if da_static is not None:
        check_dimnames_DataArray(da = da_static, da_name = "static DataArray",
                                 required_dimnames=[node_dim, variable_dim])
        dims_static = list(da_static.dims)
        static_variables = da_static[variable_dim].values.tolist()
        n_static_variables = len(static_variables)
    else:
        dims_static = None
        static_variables = []
        n_static_variables = 0
        
    # Boundary condition variables     
    if da_bc is not None:
        check_dimnames_DataArray(da = da_bc, da_name = "bc DataArray",
                                 required_dimnames=[time_dim, node_dim, variable_dim])
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