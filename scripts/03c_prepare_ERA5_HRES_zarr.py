#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 15:13:48 2021

@author: ghiggi
"""
import os
os.chdir('/home/ghiggi/Projects/deepsphere-weather')
import zarr
import glob
import numpy as np
import xarray as xr
from modules.utils_io import check_no_missing_timesteps 
from modules.my_io import pl_to_zarr  
from modules.my_io import toa_to_zarr
from modules.utils_zarr import rechunk_Dataset
# Define data directory
raw_dataset_dirpath = "/home/ghiggi/Projects/DeepSphere/data/raw/ERA5_HRES"
zarr_dataset_dirpath = "/home/ghiggi/Projects/DeepSphere/data/preprocessed/ERA5_HRES"

# raw_dataset_dirpath = "/ltenas3/DeepSphere/data/raw/ERA5_HRES"
# zarr_dataset_dirpath = "/ltenas3/DeepSphere/data/preprocessed/ERA5_HRES"

# Define spherical samplings
spherical_samplings = [ 
    # 400 km 
    'Healpix_400km', 
    'Icosahedral_400km',
    'O24',
    'Equiangular_400km',
    'Equiangular_400km_tropics',
    'Cubed_400km',
    # # 100 km 
    'Healpix_100km'
] 

spherical_samplings = ['Healpix_100km']
sampling = spherical_samplings[0]

# - Global settings 
NOVERTICAL_DIMENSION = True # --> Each pressure level treated as a feature 
STACK_VARIABLES = True     # --> Create a DataArray with all features along the "feature" dimension
start_time = '1980-01-01T07:00:00'
end_time = '2018-12-31T23:00:00'

# - Define variable dictionary 
pl_var_dict = {'var129': 'z',
               'var130': 't',
               'var133': 'q'}
toa_var_dict = {'var212': 'tisr'}
static_var_dict = {'var172': 'lsm',
                   'var43': 'slt',
                   'z': 'orog'}

# - Define chunking option for the various samplings
chunks_400km = {'node': -1,
                'time': 8760*10, # expects to be preloaded in memory since small !!!
                'feature': 1} 
chunks_100km = {'node': -1,
                'time': 72,  # TODO OPTIMIZATION
                'feature': 1} 
chunks_dict = {'Healpix_400km': chunks_400km,
               'Icosahedral_400km': chunks_400km,
               'O24': chunks_400km,
               'Equiangular_400km': chunks_400km,
               'Equiangular_400km_tropics': chunks_400km,
               'Cubed_400km': chunks_400km,
               'Healpix_100km': chunks_100km,
               } 
                         
# - Define compressor option for the various samplings 
compressor_400km = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
compressor_100km = zarr.Blosc(cname="zstd", clevel=3, shuffle=2) # TODO OPTIMIZATION
compressor_dict = {'Healpix_400km': compressor_400km,
                   'Icosahedral_400km': compressor_400km,
                   'O24': compressor_400km,
                   'Equiangular_400km': compressor_400km,
                   'Equiangular_400km_tropics': compressor_400km,
                   'Cubed_400km': compressor_400km,
                   'Healpix_100km': compressor_100km,
                   } 

#----------------------------------------------------------------------------.
# Process all data for model training  
for sampling in spherical_samplings:
    print("Preprocessing", sampling, "data")
    ##------------------------------------------------------------------------.   
    ### Define directories 
    # - Raw data 
    raw_pl_dirpath = os.path.join(raw_dataset_dirpath, sampling, "dynamic", "pressure_levels")
    raw_toa_dirpath = os.path.join(raw_dataset_dirpath, sampling, "dynamic", "boundary_conditions")
    static_dirpath = os.path.join(raw_dataset_dirpath, sampling, "static")
    
    # - Zarr data 
    dynamic_zarr_fpath = os.path.join(zarr_dataset_dirpath, sampling, "Data", "dynamic", "time_chunked", "dynamic.zarr")
    toa_zarr_fpath = os.path.join(zarr_dataset_dirpath, sampling, "Data", "bc", "time_chunked", "bc.zarr")
    static_zarr_fpath = os.path.join(zarr_dataset_dirpath, sampling, "Data","static.zarr")
    
    ##------------------------------------------------------------------------.   
    ### Pressure levels
    print("- Zarrify pressure levels data")
    # - Retrieve all raw netCDF files 
    pl_fpaths = sorted(glob.glob(raw_pl_dirpath + "/pl_*.nc"))
    # - Open all netCDF4 files
    ds = xr.open_mfdataset(pl_fpaths, chunks = "auto")
    ds = ds.sel(time=slice(start_time,end_time))
    # - Check there are not missing timesteps 
    check_no_missing_timesteps(timesteps=ds.time.values)
    # - Unstack pressure levels dimension, create a feature dimension and save to zarr
    pl_to_zarr(ds = ds, 
               zarr_fpath = dynamic_zarr_fpath, 
               var_dict = pl_var_dict, 
               unstack_plev = NOVERTICAL_DIMENSION, 
               stack_variables = STACK_VARIABLES, 
               chunks = chunks_dict[sampling], 
               compressor = compressor_dict[sampling],
               append = False)

    ##------------------------------------------------------------------------. 
    ### TOA 
    print("- Zarrify TOA data")
    # - Retrieve all raw netCDF files 
    toa_fpaths = sorted(glob.glob(raw_toa_dirpath + "/toa_*.nc"))
    # - Open all netCDF4 files
    ds = xr.open_mfdataset(toa_fpaths)
    ds = ds.sel(time=slice(start_time,end_time))
    # - Check there are not missing timesteps 
    check_no_missing_timesteps(timesteps=ds.time.values)
    # - Stack TOA into 'feature' dimension and save to zarr 
    toa_to_zarr(ds = ds, 
                zarr_fpath = toa_zarr_fpath, 
                var_dict = toa_var_dict, 
                stack_variables = STACK_VARIABLES, 
                chunks = chunks_dict[sampling], 
                compressor = compressor_dict[sampling],
                append = False)

    ##------------------------------------------------------------------------.  
    ### Static features 
    print("- Zarrify static data")
    # - Retrieve all raw netCDF files 
    static_fpaths = glob.glob(static_dirpath + "/*/*.nc", recursive=True) 
    l_ds = []
    for fpath in static_fpaths:
        tmp_ds = xr.open_dataset(fpath)  
        tmp_ds = tmp_ds.squeeze()
        tmp_ds = tmp_ds.drop_vars(['time'])  # causing problem ... 
        l_ds.append(tmp_ds) 
    ds = xr.merge(l_ds) 
    ds = ds.drop_vars(['lon_bnds','lat_bnds','lev'])
    ds = ds.drop_vars(['hyai','hybi','hyam','hybm'])
    ds = ds.rename({'ncells': 'node'})
    ds = ds.rename(static_var_dict)
    # - Stack all variables into the 'feature' dimension
    da_stacked = ds.to_stacked_array(new_dim="feature", variable_dim='variable', 
                                     sample_dims=list(ds.dims.keys()), 
                                     name = 'data')
    # - Remove MultiIndex for compatibility with netCDF and Zarr
    da_stacked = da_stacked.reset_index('feature')
    da_stacked = da_stacked.set_index(feature='variable', append=True)
    # - Remove attributes from DataArray 
    da_stacked.attrs = {}
    da_stacked.encoding = {}
    # - Reshape to Dataset
    ds = da_stacked.to_dataset() 
    # - Write to zarr     
    ds.to_zarr(static_zarr_fpath)
 
#-----------------------------------------------------------------------------.
# Rechunk dynamic and bc data across space for efficient time statistics 
for sampling in spherical_samplings:
    print("Rechunking", sampling, "data")
    ##------------------------------------------------------------------------.   
    ### Define directories 
    # - Source "time-chunked" zarr data 
    dynamic_timechunked_zarr_fpath = os.path.join(zarr_dataset_dirpath, sampling, "Data", "dynamic", "time_chunked", "dynamic.zarr")
    toa_timechunked_zarr_fpath = os.path.join(zarr_dataset_dirpath, sampling, "Data", "bc", "time_chunked", "bc.zarr")
    # - Destination "space-chunked" zarr data 
    dynamic_spacechunked_zarr_fpath = os.path.join(zarr_dataset_dirpath, sampling, "Data", "dynamic", "space_chunked", "dynamic.zarr")
    toa_spacechunked_zarr_fpath = os.path.join(zarr_dataset_dirpath, sampling, "Data", "bc", "space_chunked", "bc.zarr")
    # - Define temporary directory
    tmp_fpath = os.path.join(zarr_dataset_dirpath, "tmp")
    ##------------------------------------------------------------------------.
    ### Define chunks 
    chunks = {'node': 1,    # chunked across space (each pixel)
              'time': -1,   # unchunked across time
              'feature': -1}
    
    ##------------------------------------------------------------------------.
    ### Rechunk Zarr data
    # Open dynamic dataset and rechunk 
    ds_dynamic = xr.open_zarr(dynamic_spacechunked_zarr_fpath) 
    rechunk_Dataset(ds = ds_dynamic,
                    chunks = chunks, 
                    target_store = dynamic_spacechunked_zarr_fpath, 
                    temp_store = os.path.join(tmp_fpath, "tmp_store.zarr"), 
                    max_mem = '2GB')
    
    # Open TOA dataset and rechunk 
    ds_bc = xr.open_zarr(toa_timechunked_zarr_fpath) 
    rechunk_Dataset(ds = ds_dynamic,
                    chunks = chunks, 
                    target_store = toa_spacechunked_zarr_fpath, 
                    temp_store = os.path.join(tmp_fpath, "tmp_store.zarr"), 
                    max_mem = '2GB')
 
#-----------------------------------------------------------------------------.

 

 