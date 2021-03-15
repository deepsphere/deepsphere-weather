#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 23:28:58 2021

@author: ghiggi
"""
##----------------------------------------------------------------------------.
import os
import xarray as xr
import pygsp as pg
import cartopy.crs as ccrs
import modules.xverif as xverif
data_dir = "/home/ghiggi/Projects/DeepSphere/ToyData/Healpix_400km"

ds = xr.open_zarr(os.path.join(data_dir,"Dataset","dynamic.zarr"))
# ds = ds.isel(time=slice(0,10))
ds = ds.load()

ds1 = ds.copy()
ds1 = ds + 0.15
ds1 = xr.concat((ds1,ds1), dim="leadtime")
ds1 = ds1.assign_coords({'leadtime': [1,2]})

ds = ds.chunk({'time':-1, 'node':30})
ds1 = ds1.chunk({'time':-1, 'node':300})

pred = ds1
obs = ds
  
##----------------------------------------------------------------------------.
# Compute deterministic metric (at each node, each leadtime)
ds_skill = xverif.deterministic(pred, obs, 
                                forecast_type="continuous",
                                aggregating_dim="time")

# Add information related to mesh area
ds_skill = ds_skill.sphere.add_nodes_from_pygsp(pygsp_graph=pg.graphs.SphereHealpix(subdivisions=16, k=20, nest=True))
ds_skill = ds_skill.sphere.add_SphericalVoronoiMesh(x='lon', y='lat')

# Compute skill summary statics    
ds_global_skill = xverif.global_summary(ds_skill, area_coords="area")
ds_latitudinal_skill = xverif.latitudinal_summary(ds_skill, lat_dim='lat', lon_dim='lon', lat_res=5) 
ds_longitudinal_skill = xverif.longitudinal_summary(ds_skill, lat_dim='lat', lon_dim='lon', lon_res=5) 

# Example
ds_skill['t850'].to_dataset('skill') 
ds_global_skill['t850'].to_dataset('skill')  
ds_latitudinal_skill['t850'].to_dataset('skill') 
ds_longitudinal_skill['t850'].to_dataset('skill') 

 

 