#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 18:46:41 2021

@author: ghiggi
"""
import xarray as xr

##----------------------------------------------------------------------------.
def load_dynamic_dataset(data_dir, chunk_size="auto"):
    z500 = xr.open_mfdataset(f'{data_dir}geopotential_500/*.nc', 
                             combine='by_coords', 
                             chunks={'time': chunk_size}).rename({'z': 'z500'})
    t850 = xr.open_mfdataset(f'{data_dir}temperature_850/*.nc', 
                             combine='by_coords',  
                             chunks={'time': chunk_size}).rename({'t': 't850'})
    return xr.merge([z500, t850], compat='override')    
   
def load_bc_dataset(data_dir, chunk_size = "auto"): 
    return xr.open_mfdataset(f'{data_dir}toa_incident_solar_radiation/*.nc',  
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