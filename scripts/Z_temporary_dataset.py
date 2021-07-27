#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:22:15 2020

@author: ghiggi
"""
import os
import numpy as np
import xarray as xr
import dask
import glob
from copy import copy

dataset_path = "/ltenas3/DeepSphere/data/raw/ERA5_HRES"
temporary_data_dir_path = "/ltenas3/DeepSphere/data/temporary/"

# dataset_path = "/home/ghiggi/Projects/DeepSphere/data/raw/ERA5_HRES"
# temporary_data_dir_path = "/home/ghiggi/Projects/DeepSphere/data/temporary/"

spherical_samplings = [ 
    # 400 km 
    'Healpix_400km', 
    'Icosahedral_400km',
    'O24',
    'Equiangular_400km',
    'Equiangular_400km_tropics',
    'Cubed_400km',
    # # 100 km 
    # 'Healpix_100km'
] 

start_time = '1980-01-02T00:00:00'
end_time = '2018-12-31T23:00:00'

# spherical_samplings = ['Healpix_400km']
spherical_samplings = ["Random_2800"]

#### TODO move to io.py
### check all timestpes 
def check_no_missing_timesteps(timesteps):
    dt = np.diff(timesteps)
    dts, counts = np.unique(dt, return_counts=True)
    print("Starting at", timesteps[0])
    print("Ending at", timesteps[-1])
    if (len(counts) > 1):
        print("Missing data between:")
        bad_dts = dts[counts != counts.max()] 
        for bad_dt in bad_dts:
           bad_idxs = np.where(dt == bad_dt)[0]
           bad_idxs =  [b.tolist() for b in bad_idxs]
           for bad_idx in bad_idxs:
               tt_missings = timesteps[bad_idx:(bad_idx+2)]
               print("-", tt_missings[0], "and", tt_missings[1])
        raise ValueError("The process has been interrupted") 
    return 
 
for sampling in spherical_samplings:
    print("Preprocessing", sampling, "data")
    ##------------------------------------------------------------------------. 
    ### Correct February data for pressure_levels (analysis)
    print("- Correcting February pressure levels (analysis) data")
    pl_dirpath = os.path.join(dataset_path, sampling, "dynamic","pressure_levels")
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
    
    ### Correct February data for TOA (forecasts)
    print("- Correcting February TOA (forecasts) data")
    toa_dirpath = os.path.join(dataset_path, sampling, "dynamic","boundary_conditions")
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
  
    ##------------------------------------------------------------------------.   
    ### Pressure levels
    print("- Join pressure levels data")
    pl_dirpath = os.path.join(dataset_path, sampling, "dynamic","pressure_levels")
    pl_fpaths = sorted(glob.glob(pl_dirpath + "/pl_*.nc"))
    # Open all netCDF4 files
    ds = xr.open_mfdataset(pl_fpaths, chunks = "auto")
    ds = ds.sel(time=slice(start_time,end_time))
    # Check not missing timesteps 
    check_no_missing_timesteps(timesteps=ds.time.values)
    # Reformat 
    ds = ds.drop_vars(['lon_bnds','lat_bnds'])
    ds = ds.rename({'ncells': 'node'})
    ds = ds.rename({'plev': 'level'})
    ds = ds.rename({'var129': 'z'})
    ds = ds.rename({'var130': 't'})
    ds = ds.rename({'var133': 'q'})
    
    ds = ds.assign_coords(node=(np.arange(0, ds.dims['node'])))
    # Select variables 
    ds_z = ds.z.sel(level=500*100).to_dataset()
    ds_t = ds.t.sel(level=850*100).to_dataset()
    # Save netcdf for each variable
    tmp_z_dir = os.path.join(temporary_data_dir_path, sampling, "data", "geopotential_500")
    tmp_t_dir = os.path.join(temporary_data_dir_path, sampling, "data", "temperature_850") 
    if not os.path.exists(tmp_z_dir): 
        os.makedirs(tmp_z_dir)
    if not os.path.exists(tmp_t_dir): 
        os.makedirs(tmp_t_dir) 
    z_fpath = os.path.join(tmp_z_dir, "geopotential_500_5.625deg.nc")
    t_fpath = os.path.join(tmp_t_dir, "temperature_850_5.625deg.nc")
    ds_z.to_netcdf(path=z_fpath, 
                   encoding={'z': {"zlib": True, "complevel": 6}})
    ds_t.to_netcdf(path=t_fpath, 
                   encoding={'t': {"zlib": True, "complevel": 6}})
    ##------------------------------------------------------------------------. 
    ### TOA 
    print("- Join TOA data")
    toa_dirpath = os.path.join(dataset_path, sampling, "dynamic","boundary_conditions")
    toa_fpaths = sorted(glob.glob(toa_dirpath + "/toa_*.nc"))
    # Open all netCDF4 files
    ds = xr.open_mfdataset(toa_fpaths)
    ds = ds.sel(time=slice(start_time,end_time))
    # Check not missing timesteps 
    check_no_missing_timesteps(timesteps=ds.time.values)
    # Reformat 
    ds = ds.drop_vars(['lon_bnds','lat_bnds'])
    ds = ds.rename({'ncells': 'node'})
    ds = ds.rename({'var212': 'tisr'})
    ds = ds.assign_coords(node=(np.arange(0, ds.dims['node'])))
    
    # Save 
    tmp_toa_dir = os.path.join(temporary_data_dir_path, sampling, "data", "toa_incident_solar_radiation") 
    if not os.path.exists(tmp_toa_dir): 
        os.makedirs(tmp_toa_dir)
    
    toa_fpath = os.path.join(tmp_toa_dir, "toa_incident_solar_radiation_5.625deg.nc")
     
    ds.to_netcdf(path=toa_fpath, 
                 encoding={'tisr': {"zlib": True, "complevel": 6}})
    ##------------------------------------------------------------------------.  
    ### Static features 
    print("- Join static data")
    static_dirpath = os.path.join(dataset_path, sampling, "static")
    static_fpaths = glob.glob(static_dirpath + "/*/*.nc", recursive=True) 
    l_ds = []
    for fpath in static_fpaths:
        tmp_ds = xr.open_dataset(fpath)  
        tmp_ds = tmp_ds.squeeze()
        tmp_ds = tmp_ds.drop_vars(['time'])  # causing problem ... 
        l_ds.append(tmp_ds) 
    ds = xr.merge(l_ds) 
    ds = ds.drop_vars(['lon_bnds','lat_bnds'])
    ds = ds.rename({'ncells': 'node'})
    ds = ds.rename({'var172': 'lsm'})
    ds = ds.rename({'var43': 'slt'})
    ds = ds.rename({'z': 'orog'})
    ds = ds.assign_coords(node=(np.arange(0, ds.dims['node'])))
    ds = ds.drop_vars(['hyai','hybi','hyam','hybm'])
    ds = ds.squeeze()
    ds = ds.drop_vars(['lev'])
    ds['lat2d'] = ds['lat']
    ds['lon2d'] = ds['lon']
    ## Save
    tmp_constants_dir = os.path.join(temporary_data_dir_path, sampling, "data", "constants") 
    if not os.path.exists(tmp_constants_dir): 
        os.makedirs(tmp_constants_dir)
    constants_fpath = os.path.join(tmp_constants_dir, "constants_5.625deg.nc")
    
    ds.to_netcdf(path=constants_fpath)









# ####  PL checks
# f = os.path.join("/ltenas3/DeepSphere/data/raw/ERA5_HRES",
#                  sampling, "dynamic/pressure_levels/pl_1980_02.nc")
# ds_tmp = xr.open_dataset(f)
# print(ds_tmp.time.values[0])
# print(ds_tmp.time.values[-1])

# ####  TOA checks
# f = os.path.join("/ltenas3/DeepSphere/data/raw/ERA5_HRES",
#                  sampling, "dynamic/boundary_conditions/toa_1980_03.nc")
# ds_tmp = xr.open_dataset(f)
# print(ds_tmp.time.values[0])
# print(ds_tmp.time.values[-1])


# ##### GRIB TOA checks 
# # - Forecast : YYY-02-01T07:00:00 ----> YYYY-03-01T06:00:00     
# ### GRIB format  ---> Remmapping already convert to time
# ## time 
# ## step 
# ## valid_time
# f = "/ltenas3/DeepSphere/data/raw/ERA5_HRES/N320/dynamic/boundary_conditions/toa_1980_02.grib"
# ds_tmp = xr.open_dataset(f, engine="cfgrib")
# ds_tmp.valid_time.values[0,:]   
# ds_tmp.time.values[0]   #  
                                             

# #### Check temporary 
# sampling = "Healpix_400km"
# tmp_z_dir = os.path.join(temporary_data_dir_path, sampling, "data", "geopotential_500")
# z_fpath = os.path.join(tmp_z_dir, "geopotential_500_5.625deg.nc")
# ds = xr.open_mfdataset(z_fpath)
# check_no_missing_timesteps(timesteps=ds.time.values) 


# ## Check no timestep is missing at the end of February month 
# f = "/ltenas3/DeepSphere/data/temporary/Healpix_400km/data/toa_incident_solar_radiation/toa_incident_solar_radiation_5.625deg.nc"
# ds_tmp = xr.open_dataset(f)
# ds_tmp.isel(time=slice(29*2*24, 29*2*24 + 48)).time.values


#### Optimize compression 
# tmp_z_dir = os.path.join(temporary_data_dir_path, sampling, "data", "geopotential_500")
# z_fpath = os.path.join(tmp_z_dir, "geopotential_500_5.625deg.nc")
 

# ds = xr.open_mfdataset(z_fpath)
# ds = ds.compute()
# for i in range(1,6):
#     print(i)
#     tmp_filename = "geopotential_500_complevel_%s.nc"%(i)
#     tmp_filepath = os.path.join(tmp_z_dir, tmp_filename)
#     ds.to_netcdf(path = tmp_filepath, 
#                  encoding = {'z': {"zlib": True, "complevel": i}})

# import datetime 

 # start = 190
    # for i in np.arange(start, len(pl_fpaths)):
    #     print(i)
    #     ds = xr.open_mfdataset(pl_fpaths[start:(i+1)], chunks="auto")

    # # 24-25, 36-37, 60-61, 72-73, 84-85, 108-109, 120-121, 132-133
    # pl_fpaths[84]
    # ds1 = xr.open_dataset(pl_fpaths[24]) # 1982-february
    # ds2 = xr.open_dataset(pl_fpaths[25])
    # ds1.time
    # ds2.time
    # l_ds =[]
    # for fpath in pl_fpaths:
    #     tmp_ds = xr.open_dataset(fpath) #  chunks='auto')
    #     tmp_ds = tmp_ds.drop_vars(["lon","lat", "lon_bnds","lat_bnds"])
    #     l_ds.append(tmp_ds)
    # for i in range(len(l_ds)):
    #     print(i)
    #     new_ds = xr.merge(l_ds[0:(i+1)], compat = "override")