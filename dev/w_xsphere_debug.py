#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 12:14:56 2021

@author: ghiggi
"""
import os 
os.chdir('/home/ghiggi/Projects/deepsphere-weather')
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np 
from modules import xsphere

fpath = "/ltenas3/DeepSphere/data/raw/ERA5_HRES/Healpix_400km/static/land_sea_mask/land_sea_mask.nc"
var = "var172"
fpath = "/ltenas3/DeepSphere/data/raw/ERA5_HRES/Healpix_400km/static/soil_type/soil_type.nc"
var = "var43"
fpath = "/ltenas3/DeepSphere/data/raw/ERA5_HRES/Healpix_400km/static/topography/topography.nc"
var = "z"
ds = xr.open_dataset(fpath)
ds = ds.sphere.add_SphericalVoronoiMesh(x='lon', y='lat')
ds[var]= ds[var].squeeze() # Remove time_dim
crs_proj = ccrs.PlateCarree() 

fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs_proj))
p = xsphere._plot(ds[var],
                  ax=ax, 
                  transform=ccrs.Geodetic(),  
                  # Projection options
                  # subplot_kws={'projection': crs_proj},
                  # Polygon border option
                  edgecolors="white", # None to not display polygon border 
                  linewidths=0.01,
                  antialiased=True,
                  alpha = 1,
                  # Colorbar options 
                  add_colorbar = True,
                  cmap = plt.get_cmap('Spectral_r'),
                  norm=None,
                  center=None,
                  colors=None,
                  levels=None,
                  #   vmin = 48000,
                  #   vmax = 56500,
                  robust=True,
                  extend = 'both', # 'neither', 'both', 'min', 'max'
                  # Add colorbar label 
                  add_labels = True)

ax.coastlines(alpha=0.2)
plt.show()
 
# Currently:
# - _plot require ax and do not use the subplot_kws
# --> Allow ax=None, and

fpath = "/ltenas3/DeepSphere/data/preprocessed/ERA5_HRES/Healpix_100km/Data/static.zarr" 
ds = xr.open_zarr(fpath) 
ds = ds.sphere.add_SphericalVoronoiMesh(x='lon', y='lat')

crs_proj = ccrs.PlateCarree() 
vars = list(ds.data_vars.keys())
for var in vars:
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs_proj))
    p = xsphere._plot(ds[var],
                      ax = ax, 
                      transform=ccrs.Geodetic(),  
                      # Projection options
                      # subplot_kws={'projection': crs_proj},
                      # Polygon border option
                      edgecolors="white", # None to not display polygon border 
                      linewidths=0.01,
                      antialiased=True,
                      alpha = 1,
                      # Colorbar options 
                      add_colorbar = True,
                      cmap = plt.get_cmap('Spectral_r'),
                      norm=None,
                      center=None,
                      colors=None,
                      levels=None,
                      #   vmin = 48000,
                      #   vmax = 56500,
                      robust=True,
                      extend = 'both', # 'neither', 'both', 'min', 'max'
                      # Add colorbar label 
                      add_labels = True)

    ax.coastlines(alpha=0.2)
    plt.show()


fpath = "/ltenas3/DeepSphere/data/preprocessed/ERA5_HRES/Healpix_100km/Data/static.zarr" 
ds = xr.open_zarr(fpath) 
ds = ds.sphere.add_SphericalVoronoiMesh(x='lon', y='lat')

crs_proj = ccrs.PlateCarree() 
vars = list(ds.data_vars.keys())
for var in vars:
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs_proj))
    p = xsphere._plot(ds[var],
                      ax = ax, 
                      transform=ccrs.Geodetic(),  
                      # Projection options
                      # subplot_kws={'projection': crs_proj},
                      # Polygon border option
                      edgecolors="white", # None to not display polygon border 
                      linewidths=0.01,
                      antialiased=True,
                      alpha = 1,
                      # Colorbar options 
                      add_colorbar = True,
                      cmap = plt.get_cmap('Spectral_r'),
                      norm=None,
                      center=None,
                      colors=None,
                      levels=None,
                      #   vmin = 48000,
                      #   vmax = 56500,
                      robust=True,
                      extend = 'both', # 'neither', 'both', 'min', 'max'
                      # Add colorbar label 
                      add_labels = True)

    ax.coastlines(alpha=0.2)
    plt.show()

 
fpath = "/ltenas3/DeepSphere/data/preprocessed/ERA5_HRES/Healpix_100km/Data/dynamic/time_chunked/dynamic.zarr" 
fpath = "/ltenas3/DeepSphere/data/preprocessed/ERA5_HRES/Healpix_100km/Data/bc/time_chunked/bc.zarr" 

ds = xr.open_zarr(fpath) 
ds = ds.sphere.add_SphericalVoronoiMesh(x='lon', y='lat')
ds = ds['data'].to_dataset('feature')
ds = ds.isel(time = 0)
crs_proj = ccrs.PlateCarree() 
vars = list(ds.data_vars.keys())
for var in vars:
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs_proj))
    p = xsphere._plot(ds[var],
                      ax = ax, 
                      transform=ccrs.Geodetic(),  
                      # Projection options
                      # subplot_kws={'projection': crs_proj},
                      # Polygon border option
                      edgecolors="white", # None to not display polygon border 
                      linewidths=0.01,
                      antialiased=True,
                      alpha = 1,
                      # Colorbar options 
                      add_colorbar = True,
                      cmap = plt.get_cmap('Spectral_r'),
                      norm=None,
                      center=None,
                      colors=None,
                      levels=None,
                      #   vmin = 48000,
                      #   vmax = 56500,
                      robust=True,
                      extend = 'both', # 'neither', 'both', 'min', 'max'
                      # Add colorbar label 
                      add_labels = True)

    ax.coastlines(alpha=0.2)
    plt.show()


# Plot pred 
var = "t850"
var = "z500"
isel = 0
tmp_pred = ds_pred[var].isel(time=isel)
tmp_obs = ds_obs[var].isel(time=isel)

fig, ax = plt.subplots(1,1, 
                       subplot_kw={'projection': ccrs.Robinson()})
s_p = tmp_pred.sphere.plot(ax = ax,
                           edgecolors = None, 
                           antialiased = True,
                           vmin=get_var_clim(var,'state')[0],
                           vmax=get_var_clim(var,'state')[1],
                           cmap=get_var_cmap(var,'state'),
                           add_colorbar=True,
                           )
plt.show()

fig, ax1 = plt.subplots(1,1, 
                       subplot_kw={'projection': ccrs.Robinson()})
s_p = tmp_obs.sphere.plot(
                    ax = ax1,
                    edgecolors = None, 
                    antialiased = True,
                    vmin=get_var_clim(var,'state')[0],
                    vmax=get_var_clim(var,'state')[1],
                    cmap=get_var_cmap(var,'state'),
                    add_colorbar=True,
                    )
plt.show()







