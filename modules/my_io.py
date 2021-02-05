#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 18:46:41 2021

@author: ghiggi
"""
import time
import xarray as xr
from modules.utils_io import check_Datasets


##----------------------------------------------------------------------------.
def load_dynamic_dataset(data_dir, chunk_size="auto"):
    z500 = xr.open_mfdataset(f'{data_dir}geopotential_500/*.nc', 
                             lock = False, 
                             combine='by_coords', 
                             chunks={'time': chunk_size}).rename({'z': 'z500'})
                            
    t850 = xr.open_mfdataset(f'{data_dir}temperature_850/*.nc', 
                             lock = False,
                             combine='by_coords',  
                             chunks={'time': chunk_size}).rename({'t': 't850'})
    return xr.merge([z500, t850], compat='override')    
   
def load_bc_dataset(data_dir, chunk_size = "auto"): 
    return xr.open_mfdataset(f'{data_dir}toa_incident_solar_radiation/*.nc',
                             lock = False,
                             combine='by_coords',  
                             chunks={'time': chunk_size})

def load_static_dataset(data_dir):
    return xr.open_dataset(f'{data_dir}constants/constants_5.625deg.nc')
    
##----------------------------------------------------------------------------.
 
def readDatasets(data_dir, feature_type, chunk_size="auto"):
    if feature_type=="dynamic":
        return load_dynamic_dataset(data_dir, chunk_size)
    elif feature_type=="bc":       
        return load_bc_dataset(data_dir, chunk_size)
    elif feature_type=="static":
        return load_static_dataset(data_dir)
    else:
        raise ValueError('Specify valid feature_type')
        
    
def reformat_Datasets(ds_training_dynamic,
                      ds_validation_dynamic = None,
                      ds_static = None,              
                      ds_training_bc = None,         
                      ds_validation_bc = None,
                      preload_data_in_CPU = False):
    ##------------------------------------------------------------------------. 
    # Check Datasets are in the expected format for AR training 
    check_Datasets(ds_training_dynamic = ds_training_dynamic,
                   ds_validation_dynamic = ds_validation_dynamic,
                   ds_static = ds_static,              
                   ds_training_bc = ds_training_bc,         
                   ds_validation_bc = ds_validation_bc)   
                  
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
    else: 
        da_validation_dynamic = None
    if ds_training_bc is not None:
        da_training_bc = ds_training_bc.to_array(dim='feature', name='BC').transpose('time', 'node', 'feature')
    else:
        da_training_bc = None
    if ds_validation_bc is not None:
        da_validation_bc = ds_validation_bc.to_array(dim='feature', name='BC').transpose('time', 'node', 'feature')
    else:
        da_validation_bc = None
    if ds_static is not None: 
        da_static = ds_static.to_array(dim='feature', name='Static').transpose('node','feature') 
    else: 
        da_static = None
    print('- Conversion to xarray DataArrays: {:.2f}s'.format(time.time() - t_i))
    
    ##------------------------------------------------------------------------.
    dict_DataArrays = {'da_training_dynamic': da_training_dynamic, 
                       'da_validation_dynamic': da_validation_dynamic,
                       'da_training_bc': da_training_bc, 
                       'da_validation_bc': da_validation_bc,
                       'da_static': da_static
                       }
    return dict_DataArrays
