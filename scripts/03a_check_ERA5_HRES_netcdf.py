#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 17:12:06 2021

@author: ghiggi
"""
import os
import sys
sys.path.append('../')
import glob
import numpy as np
import xarray as xr

dataset_path = "/ltenas3/data/DeepSphere/data/raw/ERA5_HRES"

spherical_samplings = [ 
    # 400 km 
    'Healpix_400km', 
    'Icosahedral_400km',
    'O24',
    'Equiangular_400km',
    'Equiangular_400km_tropics',
    'Cubed_400km',
    # 100 km 
    'Healpix_100km'
]  

# spherical_samplings = ['Healpix_100km']
# sampling = spherical_samplings[0]

for sampling in spherical_samplings:
    print("Preprocessing", sampling, "data")
    pl_dirpath = os.path.join(dataset_path, sampling, "dynamic","pressure_levels")
    toa_dirpath = os.path.join(dataset_path, sampling, "dynamic","boundary_conditions")
    
    #-------------------------------------------------------------------------.
    ### Process all netCDF and make sure that February data do not contain March data 
    # - Retrievals from MARS sometimes lead to March data into February query
    # - Pressure levels (analysis)
    print("- Correcting February pressure levels (analysis) data")
    pl_fpaths = sorted(glob.glob(pl_dirpath + "/pl_*_02.nc"))
    for fpath in pl_fpaths:
        tmp_ds = xr.open_dataset(fpath)
        idx_February = tmp_ds['time.month']==2
        if not all(idx_February):
            print("--", fpath)
            tmp_ds = tmp_ds.sel(time = idx_February)
            tmp_ds = tmp_ds.compute()
            os.remove(fpath)
            tmp_ds.to_netcdf(fpath, mode="w")
    
    # - TOA (forecasts)
    # -- Ensure the last timestep is XXXX-03-01-T06:00:00
    print("- Correcting February TOA (forecasts) data")
    toa_fpaths = sorted(glob.glob(toa_dirpath + "/toa_*_02.nc"))
    for fpath in toa_fpaths:
        tmp_ds = xr.open_dataset(fpath)
        tmp_year = np.unique(tmp_ds['time.year'].values).tolist()[0]
        tmp_max_Date = str(tmp_year) + "-03-01T06:00:00"
        idx_February = tmp_ds['time'].values <= np.datetime64(tmp_max_Date)
        if not all(idx_February):
            print("--", fpath)
            tmp_ds = tmp_ds.sel(time=idx_February)
            tmp_ds = tmp_ds.compute()
            os.remove(fpath)
            tmp_ds.to_netcdf(fpath, mode="w")

    #-------------------------------------------------------------------------.
    ### Process ERA5 HRES analysis data of 1980-01-01 and make it start at 07:00:00 
    #  such as ERA5 HRES forecast data
    # - Pressure levels (analysis)
    print("- Correcting starting time of pressure levels (analysis) data")
    start_time = '1980-01-01T07:00:00'
    pl_1980_01_fpath = os.path.join(pl_dirpath + "/pl_1980_01.nc")
    tmp_ds = xr.open_dataset(pl_1980_01_fpath)
    if np.any(tmp_ds['time'].values < np.datetime64(start_time)):
        tmp_ds = tmp_ds.sel(time=slice(start_time, None))
        tmp_ds = tmp_ds.compute()
        os.remove(pl_1980_01_fpath)
        tmp_ds.to_netcdf(pl_1980_01_fpath, mode="w")
    
    #-------------------------------------------------------------------------.
 
#-------------------------------------------------------------------------.
# PL checks
# f = os.path.join("/ltenas3/DeepSphere/data/raw/ERA5_HRES",
#                   sampling, "dynamic/pressure_levels/pl_1980_02.nc")
# ds_tmp = xr.open_dataset(f)
# print(ds_tmp.time.values[0])
# print(ds_tmp.time.values[-1])

# TOA checks
# f = os.path.join("/ltenas3/DeepSphere/data/raw/ERA5_HRES",
#                  sampling, "dynamic/boundary_conditions/toa_1980_03.nc")
# ds_tmp = xr.open_dataset(f)
# print(ds_tmp.time.values[0])
# print(ds_tmp.time.values[-1])

 