#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import os
os.chdir('/ltenas3/DeepSphere')
# os.chdir('/home/ghiggi/Projects/DeepSphere')

 
import xarray as xr
 
 
 
temporary_data_dir_path = "/ltenas3/DeepSphere/data/temporary/"
toy_data_dir = "/ltenas3/DeepSphere/data/toy_data"

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

start_time = '1980-01-01T00:00:00'
end_time = '1981-01-01T00:00:00'
spherical_samplings = ['Healpix_400km']
sampling = spherical_samplings[0]
for sampling in spherical_samplings:
    ##------------------------------------------------------------------------.   
    ## Create directories 
    
    tmp_toa_dir = os.path.join(toy_data_dir, sampling, "data", "toa_incident_solar_radiation") 
    if not os.path.exists(tmp_toa_dir): 
        os.makedirs(tmp_toa_dir)
    tmp_toa_dir = os.path.join(toy_data_dir, sampling, "data", "geopotential_500") 
    if not os.path.exists(tmp_toa_dir): 
        os.makedirs(tmp_toa_dir)
    tmp_toa_dir = os.path.join(toy_data_dir, sampling, "data", "temperature_850") 
    if not os.path.exists(tmp_toa_dir): 
        os.makedirs(tmp_toa_dir)   
    
    # Slice and save (z)
    z_fpath = os.path.join(temporary_data_dir_path, sampling, "data", "geopotential_500", "geopotential_500_5.625deg.nc")
    z_new_fpath = os.path.join(toy_data_dir, sampling, "data", "geopotential_500", "geopotential_500_5.625deg.nc")
    ds_z = xr.open_mfdataset(z_fpath, chunks = "auto") 
    ds_z = ds_z.sel(time=slice(start_time,end_time))
    ds_z.to_netcdf(path=z_new_fpath, 
                   encoding={'z': {"zlib": True, "complevel": 6}})
    
    # Slice and save (t)
    t_fpath = os.path.join(temporary_data_dir_path, sampling, "data", "temperature_850", "temperature_850_5.625deg.nc")
    t_new_fpath = os.path.join(toy_data_dir, sampling, "data", "temperature_850", "temperature_850_5.625deg.nc")
    ds_t = xr.open_mfdataset(t_fpath, chunks = "auto") 
    ds_t = ds_t.sel(time=slice(start_time,end_time))
    
    ds_t.to_netcdf(path=t_new_fpath, 
                   encoding={'t': {"zlib": True, "complevel": 6}})
    
    # Slice and save (toa)
    toa_fpath = os.path.join(temporary_data_dir_path, sampling, "data", "toa_incident_solar_radiation", "toa_incident_solar_radiation_5.625deg.nc")
    toa_new_fpath = os.path.join(toy_data_dir, sampling, "data", "toa_incident_solar_radiation", "toa_incident_solar_radiation_5.625deg.nc")
    ds = xr.open_mfdataset(toa_fpath)
    ds = ds.sel(time=slice(start_time,end_time))
    ds.to_netcdf(path=toa_new_fpath, 
                 encoding={'tisr': {"zlib": True, "complevel": 6}})
    
    
    
    



 