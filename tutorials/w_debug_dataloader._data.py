#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 15:57:40 2021

@author: ghiggi
"""
import os 
import xarray as xr 
from modules.my_io import reformat_Datasets

data_dir = "/home/ghiggi/Projects/DeepSphere/ToyData/Healpix_400km"

# Create zarr Dataset
ds_dynamic  = ds_dynamic.load()
ds_bc = ds_bc.load()
ds_static = ds_static.load()

ds_dynamic = ds_dynamic.chunk({'node':-1, 'time': 20})
ds_bc = ds_bc.chunk({'node':-1, 'time': 20})
ds_static = ds_static.chunk({'node':-1})

ds_dynamic.to_zarr(os.path.join(data_dir,"Dataset","dynamic.zarr"))
ds_bc.to_zarr(os.path.join(data_dir,"Dataset", "bc.zarr"))
ds_static.to_zarr(os.path.join(data_dir,"Dataset", "static.zarr"))

# Create zarr DataArrays
ds_dynamic = xr.open_zarr(os.path.join(data_dir, "Dataset", "dynamic.zarr"))
ds_bc = xr.open_zarr(os.path.join(data_dir,"Dataset", "bc.zarr"))
ds_static = xr.open_zarr(os.path.join(data_dir,"Dataset","static.zarr"))
 
dict_DataArrays = reformat_Datasets(ds_training_dynamic = ds_dynamic,
                                    ds_training_bc = ds_bc,
                                    ds_static = ds_static,              
                                    preload_data_in_CPU = False)
da_static = dict_DataArrays['da_static']
da_dynamic = dict_DataArrays['da_training_dynamic']
da_bc = dict_DataArrays['da_training_bc']

da_dynamic = da_dynamic.load()
da_bc = da_bc.load()
da_static = da_static

da_dynamic = da_dynamic.chunk({'node':-1, 'time': 20, 'feature': 1})
da_bc = da_bc.chunk({'node':-1, 'time': 20, 'feature': 1})
da_static = da_static.chunk({'node':-1, 'feature': 1})

da_dynamic = da_dynamic.to_dataset(name='Data') # to_zarr not available for DataArray
da_bc = da_bc.to_dataset(name='Data')
da_static = da_static.to_dataset(name='Data')

da_dynamic.to_zarr(os.path.join(data_dir,"DataArray","dynamic.zarr"))
da_bc.to_zarr(os.path.join(data_dir,"DataArray", "bc.zarr"))
da_static.to_zarr(os.path.join(data_dir,"DataArray", "static.zarr"))


 
