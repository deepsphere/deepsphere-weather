#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 23:56:40 2021

@author: ghiggi
"""
import numpy as np 

def is_dask_DataArray(da):
    """Check if data in the xarray DataArray are lazy loaded."""
    if da.chunks is not None:
        return True
    else: 
        return False
    
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
