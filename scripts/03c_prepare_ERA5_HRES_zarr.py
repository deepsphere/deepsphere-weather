#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 15:13:48 2021

@author: ghiggi
"""
import os
import sys
sys.path.append('../')
import time 
import zarr
import glob
import dask
import shutil
import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster
from modules.utils_io import check_no_missing_timesteps 
from modules.utils_zarr import rechunk_Dataset
from modules.utils_zarr import write_zarr
from modules.my_io import reformat_pl  
from modules.my_io import reformat_toa

def zarrify_raw_data(sampling, raw_dataset_dirpath, 
                     zarr_dataset_dirpath, 
                     dst_subdirname, 
                     compressor_dict, 
                     chunks_dict, 
                     start_time, 
                     end_time, 
                     unstack_plev = True, 
                     stack_variables = False, 
                     force=True):
    print("==================================================================")
    print("Preprocessing", sampling, "raw data")
    ##------------------------------------------------------------------------.  
    # - Define variable dictionary 
    pl_var_dict = {'var129': 'z',
                'var130': 't',
                'var133': 'q'}
    toa_var_dict = {'var212': 'tisr'}
    static_var_dict = {'var172': 'lsm',
                    'var43': 'slt',
                    'z': 'orog'}

    t_i = time.time()

    ##------------------------------------------------------------------------.   
    ### Define directories 
    # - Raw data 
    raw_pl_dirpath = os.path.join(raw_dataset_dirpath, sampling, "dynamic", "pressure_levels")
    raw_toa_dirpath = os.path.join(raw_dataset_dirpath, sampling, "dynamic", "boundary_conditions")
    static_dirpath = os.path.join(raw_dataset_dirpath, sampling, "static")
    
    # - Zarr data 
    dynamic_zarr_fpath = os.path.join(zarr_dataset_dirpath, sampling, "Data", "dynamic", dst_subdirname, "dynamic.zarr")
    toa_zarr_fpath = os.path.join(zarr_dataset_dirpath, sampling, "Data", "bc", dst_subdirname, "bc.zarr")
    static_zarr_fpath = os.path.join(zarr_dataset_dirpath, sampling, "Data","static.zarr")
    
    ##------------------------------------------------------------------------.   
    ### Pressure levels  
    print(" - Zarrify pressure levels data")
    # - If the zarr store already exists, remove or stop computation 
    if os.path.exists(dynamic_zarr_fpath):
        if force: 
            shutil.rmtree(dynamic_zarr_fpath)
        else:
            raise ValueError("{!r} already exists.".format(dynamic_zarr_fpath)) 
    # - Retrieve temporal sorted filepath of all raw netCDF files 
    pl_fpaths = sorted(glob.glob(raw_pl_dirpath + "/pl_*.nc"))

    # - Reformat each netCDF4 separetely 
    # --> Reason: 
    #     - netCDF4 is based on HDF which does not allow multi-threaded I/O
    #     - Once data of netCDF4 are in memory, perform threaded computations
    append = False
    for fpath in pl_fpaths:  
        tt = time.time()
        print("Reformatting:", fpath)
        ds = xr.open_dataset(fpath)
        # - Subset time 
        ds = ds.sel(time=slice(start_time,end_time))
        # - Zarrify if some timesteps are available
        if len(ds['time']) > 0:
            # - Check there are not missing timesteps 
            check_no_missing_timesteps(timesteps=ds.time.values)
            # - Unstack pressure levels dimension, create a feature dimension and save to zarr
            ds = reformat_pl(ds = ds, 
                            var_dict = pl_var_dict, 
                            unstack_plev = unstack_plev, 
                            stack_variables = stack_variables)
            # - Write data
            write_zarr(zarr_fpath = dynamic_zarr_fpath, 
                    ds = ds,  
                    chunks = chunks_dict[sampling], 
                    compressor = compressor_dict[sampling],
                    consolidated = True,
                    show_progress = False, 
                    append = append,
                    append_dim = "time")
            print(time.time() - tt)
            append = True

    ##------------------------------------------------------------------------. 
    ### TOA 
    print(" - Zarrify TOA data")
    # - If the zarr store already exists, remove or stop computation 
    if os.path.exists(toa_zarr_fpath):
        if force: 
            shutil.rmtree(toa_zarr_fpath)
        else:
            raise ValueError("{!r} already exists !".format(dynamic_zarr_fpath))

    # - Retrieve temporal sorted filepath of all raw netCDF files 
    toa_fpaths = sorted(glob.glob(raw_toa_dirpath + "/toa_*.nc"))

    # - Reformat each netCDF4 separately 
    # --> Reason: 
    #    - netCDF4 is based on HDF which does not allow multi-threaded I/O
    #    - Once data of netCDF4 are in memory, perform threaded computations
    append = False
    for fpath in toa_fpaths:  
        print("Reformatting:", fpath)
        ds = xr.open_dataset(fpath)
        ds = ds.sel(time=slice(start_time,end_time))
        # - Zarrify if some timesteps are available
        if len(ds['time']) > 0:
            # - Check there are not missing timesteps 
            check_no_missing_timesteps(timesteps=ds.time.values)
            # - Stack TOA into 'feature' dimension  
            ds = reformat_toa(ds = ds, 
                            var_dict = toa_var_dict, 
                            stack_variables = stack_variables)
            # - Write data to zarr
            write_zarr(zarr_fpath = toa_zarr_fpath, 
                    ds = ds,  
                    chunks = chunks_dict[sampling], 
                    compressor = compressor_dict[sampling],
                    consolidated = True,
                    show_progress = False, 
                    append = append,
                    append_dim = "time")
            append = True 
          
    ##------------------------------------------------------------------------.  
    ### Static features 
    print(" - Zarrify static data")
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
    #-----------------------------------------.
    # - Fix orography to be 0 lower bound 
    da_orog = ds['orog'].load() 
    da_orog[da_orog < 0] = 0
    ds['orog'] = da_orog
    #-----------------------------------------.
    # - Add latitude and longitude feature 
    ds['sin_latitude'] =  np.sin(ds['lat'])
    ds['sin_longitude'] = np.sin(ds['lon']/np.pi)
    ds['cos_longitude'] = np.sin(ds['lon']/np.pi)
    #-----------------------------------------.
    # - Stack all variables into the 'feature' dimension
    # da_stacked = ds.to_stacked_array(new_dim="feature", variable_dim='variable', 
    #                                  sample_dims=list(ds.dims.keys()), 
    #                                  name = 'data')
    # # - Remove MultiIndex for compatibility with netCDF and Zarr
    # da_stacked = da_stacked.reset_index('feature')
    # da_stacked = da_stacked.set_index(feature='variable', append=True)
    # # - Remove attributes from DataArray 
    # da_stacked.attrs = {}
    # da_stacked.encoding = {}
    # # - Reshape to Dataset
    # ds = da_stacked.to_dataset() 
    # - Write to zarr     
    ds.to_zarr(static_zarr_fpath)
    #-------------------------------------------------------------------------.
    # Report elapsed time 
    print("---> Elapsed time: {:.1f} minutes ".format((time.time() - t_i)/60))
    print("==================================================================")
    #-------------------------------------------------------------------------.
#-----------------------------------------------------------------------------.
 
#-----------------------------------------------------------------------------.
def rechunk_zarr_stores(zarr_dataset_dirpath, sampling, chunks, src_subdirname, dst_subdirname):
    print("==================================================================")
    print("Rechunking", sampling, "data")
    t_i = time.time()
    ##------------------------------------------------------------------------.   
    ### Define directories 
    # - Source "time-chunked" zarr data 
    dynamic_timechunked_zarr_fpath = os.path.join(zarr_dataset_dirpath, sampling, "Data", "dynamic", src_subdirname, "dynamic.zarr")
    toa_timechunked_zarr_fpath = os.path.join(zarr_dataset_dirpath, sampling, "Data", "bc", src_subdirname, "bc.zarr")
    # - Destination "space-chunked" zarr data 
    dynamic_spacechunked_zarr_fpath = os.path.join(zarr_dataset_dirpath, sampling, "Data", "dynamic", dst_subdirname, "dynamic.zarr")
    toa_spacechunked_zarr_fpath = os.path.join(zarr_dataset_dirpath, sampling, "Data", "bc", dst_subdirname, "bc.zarr")
    # - Define temporary directory
    tmp_fpath = os.path.join(zarr_dataset_dirpath, "tmp")
    ##------------------------------------------------------------------------.
    ### Rechunk Zarr data
    # Open dynamic dataset and rechunk 
    ds_dynamic = xr.open_zarr(dynamic_timechunked_zarr_fpath) 
    rechunk_Dataset(ds = ds_dynamic,
                    chunks = chunks, 
                    target_store = dynamic_spacechunked_zarr_fpath, 
                    temp_store = os.path.join(tmp_fpath, "tmp_store.zarr"), 
                    max_mem = '2GB')
    
    # Open TOA dataset and rechunk 
    ds_bc = xr.open_zarr(toa_timechunked_zarr_fpath) 
    rechunk_Dataset(ds = ds_bc,
                    chunks = chunks, 
                    target_store = toa_spacechunked_zarr_fpath, 
                    temp_store = os.path.join(tmp_fpath, "tmp_store.zarr"), 
                    max_mem = '2GB')
   #-------------------------------------------------------------------------.
    # Report elapsed time 
    print("---> Elapsed time: {:.1f} minutes ".format((time.time() - t_i)/60))
    print("==================================================================")
    #-------------------------------------------------------------------------.
#-----------------------------------------------------------------------------.
#-----------------------------------------------------------------------------.

## Define data directory
raw_dataset_dirpath = "/ltenas3/DeepSphere/data/raw/ERA5_HRES"
zarr_dataset_dirpath = "/ltenas3/DeepSphere/data/preprocessed/ERA5_HRES"

# Define spherical samplings
spherical_samplings = [ 
     # 400 km 
    'Healpix_400km', 
    'Icosahedral_400km',
    'O24',
    'Equiangular_400km',
    'Equiangular_400km_tropics',
    'Cubed_400km',
     #  100 km 
     'Healpix_100km'
] 

# - Global settings 
unstack_plev = True      # --> Each pressure level treated as a feature 
stack_variables = False  # --> If True the output is a DataArray with 'feature' dimension        
start_time = '1980-01-01T07:00:00'
end_time = '2018-12-31T23:00:00'

# - Define chunking options for the various samplings
chunks_400km = {'node': -1,
                'time': 24*7}
chunks_100km = {'node': -1,
                'time': 24*2} 

chunks_dict = {'Healpix_400km': chunks_400km,
               'Icosahedral_400km': chunks_400km,
               'O24': chunks_400km,
               'Equiangular_400km': chunks_400km,
               'Equiangular_400km_tropics': chunks_400km,
               'Cubed_400km': chunks_400km,
               'Healpix_100km': chunks_100km,
               } 
                         
# - Define compressor options for the various samplings 
compressor_400km = zarr.Blosc(cname="zstd", clevel=0, shuffle=1)
compressor_100km = zarr.Blosc(cname="lz4", clevel=0, shuffle=2)  
compressor_dict = {'Healpix_400km': compressor_400km,
                   'Icosahedral_400km': compressor_400km,
                   'O24': compressor_400km,
                   'Equiangular_400km': compressor_400km,
                   'Equiangular_400km_tropics': compressor_400km,
                   'Cubed_400km': compressor_400km,
                   'Healpix_100km': compressor_100km,
                   } 

# - Dask settings
from dask.distributed import Client
client = Client(processes=False)   

# Process all data for model training  
for sampling in spherical_samplings:
    zarrify_raw_data(sampling = sampling,
                     raw_dataset_dirpath = raw_dataset_dirpath,
                     zarr_dataset_dirpath = zarr_dataset_dirpath,
                     dst_subdirname = "time_chunked", 
                     compressor_dict = compressor_dict, 
                     chunks_dict = chunks_dict, 
                     start_time = start_time, 
                     end_time = end_time,
                     unstack_plev = unstack_plev, 
                     stack_variables = stack_variables)
    #  Rechunk dynamic and bc data across space for efficient time statistics 
    chunks = {'node': 1,    # chunked across space (each pixel)
              'time': -1}  # unchunked across time
    rechunk_zarr_stores(zarr_dataset_dirpath = zarr_dataset_dirpath,
                        sampling = sampling, 
                        chunks = chunks, 
                        src_subdirname = "time_chunked",
                        dst_subdirname = "space_chunked")
 