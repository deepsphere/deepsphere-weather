#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 14:31:51 2021

@author: ghiggi
"""
import os
os.chdir('/home/ghiggi/Projects/weather_prediction')
import cartopy
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pygsp as pg
from modules import xsphere
from modules.my_io import readDatasets   


##-----------------------------------------------------------------------------.
# Choose one of the field below
# sampling = "Healpix_100km" # high resolution
# sampling = "Healpix_400km" # low resolution --> start with this ... faster to plot
# dataset_path = "/home/ghiggi/Projects/DeepSphere/data/temporary"
# folderpath = os.path.join(dataset_path, sampling, "geopotential_500")
# filename = "geopotential_500_5.625deg.nc"  
# # Load xarray Dataset
# filepath = os.path.join(folderpath,filename)
# ds = xr.load_dataset(filepath)
# ds = ds.drop('level')
##-----------------------------------------------------------------------------.
sampling = "Healpix_400km" # you can choose another sampling ;)
data_dir = "/data/weather_prediction/data"
exp_dir = "/data/weather_prediction/experiments"
data_sampling_dir = os.path.join(data_dir, sampling)
# - Dynamic data (i.e. pressure and surface levels variables)
ds_dynamic = readDatasets(data_dir=data_sampling_dir, feature_type='dynamic')
# - Boundary conditions data (i.e. TOA)
ds_bc = readDatasets(data_dir=data_sampling_dir, feature_type='bc')
# - Static features
ds_static = readDatasets(data_dir=data_sampling_dir, feature_type='static')

ds_dynamic = ds_dynamic.drop(["level","lat","lon"])
ds_bc = ds_bc.drop(["lat","lon"])
ds_static = ds_static.drop(["lat","lon"])

#-----------------------------------------------------------------------------.
ds = ds_dynamic
ds = ds.isel(time=[0,6,12,18]) # select 4 timesteps
ds = ds.load()

##----------------------------------------------------------------------------.
## Add nodes mesh from pygsp graph (DEBUG !!!!)
# --> TODO !!! CHECK MATCHING helpix nest... CDO REVERSE POLES? where the order change?
# --> Data are not as nest=True ? 
# ds1 = ds.sphere.add_nodes_from_pygsp(pygsp_graph=pg.graphs.SphereHealpix(subdivisions=16, k=20, nest=True))

##----------------------------------------------------------------------------.
## Infer mesh using SphericalVoronoi from node coordinates
ds = ds.sphere.add_SphericalVoronoiMesh(x='lon',y='lat')

##----------------------------------------------------------------------------.
## Prepare data for below plots
da = ds['z500']
da = ds['t850']
da_single = ds['t850'].isel(time=0)

##----------------------------------------------------------------------------.

#### Add mesh and nodes info 
# ds = ds.sphere.add_mesh(PolygonPatches_list)
# ds = ds.sphere.add_nodes(lon=lon, lat=lat, node_dim='node')

# from modules.xsphere import SphericalVoronoiMesh
# l_poly_xy, area = xsphere.SphericalVoronoiMesh(ds.lon.values, ds.lat.values)
# PolygonPatches_list = sphere.get_PolygonPatchesList(l_poly_xy)
# ds = ds.sphere.add_mesh(PolygonPatches_list)
# ds = ds.sphere.add_area(area, node_dim='node')

# ds = da.sphere.compute_mesh_area() # current: planar assumption

## Not yet implemented:
# da.sphere.add_mesh_from_bnds(x, y)
# da.sphere.plot.add_mesh_from_shp(fpath)
# da.sphere.plot.add_nodes_from_shp(fpath) 

#----------------------------------------------------------------------------.
#### Define reference coordinate reference system 
# - Plate Carrée uses the straight-line path on a longitude/latitude basis
# - Geodetic uses the shortest path in a spherical sense (the geodesic.)
# --> contour() and contourf() requires Plate Carrée
# --> plot() reequires Geodetic 

# - Define reference CRS 
crs_ref = ccrs.Geodetic()    # for plot()
crs_ref = ccrs.PlateCarree() # for contour and contourf()

# - Define plot (global) projection  
crs_proj = ccrs.PlateCarree() 
crs_proj = ccrs.Mollweide()
crs_proj = ccrs.InterruptedGoodeHomolosine()
crs_proj = ccrs.Robinson()

# - Define plot (geostationary) projection (central_longitude=0)
# crs_proj = ccrs.Geostationary(central_longitude=0,satellite_height=35786000)

# # - Define plot (side view) projection  
# crs_proj = ccrs.Orthographic()
# crs_proj = ccrs.Orthographic(central_longitude=0.0, central_latitude=90.0)  # from the North Pole 
# crs_proj = ccrs.Orthographic(central_longitude=0.0, central_latitude=-90.0) # from the South Pole 

#----------------------------------------------------------------------------.
#### Plotting methods 
# Mesh based:
# - plot()  
# - plot_mesh()  
# - plot_mesh_area()  
# - plot_mesh_order()  

# Nodes based:
# - plot_nodes()
# - contour()   
# - contourf()  

#-----------------------------------------------------------------------------.
#### Plot() Dataset 
p = ds.sphere.plot(transform=ccrs.Geodetic(),  
                   subplot_kws={'projection': crs_proj},
                   # Facets options 
                   col="time", col_wrap=2, 
                   # Polygon border option
                   edgecolors="white", # None to not display polygon border 
                   linewidths=0.01,
                   antialiased=True,
                   # Colorbar options 
                   add_colorbar = True,
                   cmap = plt.get_cmap('Spectral_r'),
                   #    vmin = 48000,
                   #    vmax = 56500,
                   robust=True,
                   extend = 'neither', # 'neither', 'both', 'min', 'max'
                   # norm=None,
                   # center=None,
                   # colors=None,
                   # levels=None,
                   # cbar_kwargs
                   # Add colorbar label 
                   add_labels = True)
# Set map options
for ax in p.axes.flat:
    ax.coastlines(alpha=0.2)
plt.show()

#### Plot() DataArray  
p = da.sphere.plot(transform=ccrs.Geodetic(),  
                   subplot_kws={'projection': crs_proj},
                   # Facets options 
                   col="time", col_wrap=2, 
                   # Polygon border option
                   edgecolors="white", # None to not display polygon border 
                   linewidths=0.01,
                   antialiased=True,
                   # Colorbar options 
                   add_colorbar = True,
                   cmap = plt.get_cmap('Spectral_r'),
                   #    vmin = 48000,
                   #    vmax = 56500,
                   robust=True,
                   extend = 'neither', # 'neither', 'both', 'min', 'max'
                   # norm=None,
                   # center=None,
                   # colors=None,
                   # levels=None,
                   # cbar_kwargs
                   # Add colorbar label 
                   add_labels = True)
# Set map options
for ax in p.axes.flat:
    ax.coastlines(alpha=0.2)
plt.show()

##----------------------------------------------------------------------------.
#### Contour() DataArray  
crs_proj1 = ccrs.PlateCarree()
crs_proj1 = crs_proj 
p = da.sphere.contour(x='lon',
                      y='lat',
                      transform=ccrs.PlateCarree(),  
                      subplot_kws={'projection': crs_proj1},
                      # Facets options 
                      col="time",
                      col_wrap=2, 
                      # Line options
                      linewidths=1,
                      linestyles='solid', # 'dashed' 
                      # Contour label options
                      add_contour_labels=True, 
                      add_contour_labels_interactively=False, 
                      contour_labels_colors="black", 
                      contour_labels_fontsize=8, 
                      contour_labels_inline=True,   
                      contour_labels_format="{:.0f}".format, 
                      # Color options 
                      add_colorbar=False,
                      # colors="black",
                      cmap=plt.get_cmap('Spectral_r'),
                      levels=10,
                      # vmin = 48000,
                      # vmax = 56500,
                      robust=False,
                      extend='neither', # 'neither', 'both', 'min', 'max'
                      norm=None,
                      center=None,
                      cbar_kwargs=None,
                      # Add colorbar label 
                      add_labels = True)
# Set map options
for ax in p.axes.flat:
    ax.coastlines(alpha=0.2)
plt.show()

##----------------------------------------------------------------------------.
#### Contourf() DataArray  
# BUGS
# - plot_type="tricontourf only works for ccrs.PlateCarree()
# - With orthographic projection, empty line .... 
# https://github.com/koldunovn/pi_mesh_python
# https://github.com/SciTools/cartopy/issues/1301
# https://github.com/SciTools/cartopy/issues/1302  
# https://github.com/koldunovn/pi_mesh_python/blob/master/triangular_mesh_cartopy.ipynb 
crs_proj1 = ccrs.PlateCarree()
crs_proj1 = crs_proj 
p = da.sphere.contourf(x='lon',
                      y='lat',
                      transform=ccrs.PlateCarree(),  
                      subplot_kws={'projection': crs_proj1},
                      # Facets options 
                      col="time",
                      col_wrap=2, 
                      # Color options 
                      add_colorbar=True,
                      cmap = plt.get_cmap('Spectral_r'),
                      #   vmin=48000,
                      #   vmax=56500,
                      robust=True,
                      extend='both', # 'neither', 'both', 'min', 'max'
                      norm=None,
                      center=None,
                      # colors="black",
                      levels=10,
                      # cbar_kwargs
                      # Add colorbar label 
                      add_labels = True)
# Set map options
for ax in p.axes.flat:
    ax.coastlines(alpha=0.2)
plt.show()

##----------------------------------------------------------------------------.
#### Complex FacetGrid 
# - Plot mesh values
fg = da.sphere.plot(subplot_kws={'projection': crs_proj},
                    # Facets options 
                    col="time", col_wrap=2, 
                    # Polygon border option
                    edgecolors="white", # None to not display polygon border 
                    linewidths=0.01,
                    antialiased=True,
                    # Colorbar options 
                    add_colorbar = True,
                    cmap = plt.get_cmap('Spectral_r'),
                    robust = True, 
                    # Customize colorbar 
                    cbar_kwargs={"orientation": "vertical",
                                 "shrink": 0.8,
                                 "aspect": 40,
                                 "label": "Geopotential [m²s²]"})

# - Add coastlines and grid lines
for ax in fg.axes.flat:
    ax.coastlines(alpha=0.2)
    ax.gridlines()
    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical','land','110m',
                                                       facecolor='black',
                                                       alpha=0.4))
    
# - Superimpose contours to each panel
fg.map_dataarray_unstructured(fg, xsphere._contour, 
                              subplot_kws={'projection': crs_proj},
                              contour_labels_format="{:.0f}".format,
                              contour_labels_format_fontsize=5, 
                              colors="k", levels=13, 
                              add_colorbar=False)

# - Display the graphic 
plt.show()
    
#-----------------------------------------------------------------------------.
#### Plot() single  
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs_proj))
p = xsphere._plot(da_single,
                  ax=ax, 
                  transform=ccrs.Geodetic(),  
                  # Projection options
                  subplot_kws={'projection': crs_proj},
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

#### Contour() single 
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs_proj))
p = xsphere._contour(da_single, 
                     x='lon',
                     y='lat',
                     transform=ccrs.PlateCarree(),  
                     ax=ax, 
                     # Line options
                     linewidths=1,
                     # Contour label options
                     add_contour_labels=True, 
                     add_contour_labels_interactively=False, 
                     contour_labels_colors="black", 
                     contour_labels_fontsize=8, 
                     contour_labels_inline=True,   
                     contour_labels_format="{:.0f} ".format, 
                     # Color options 
                     add_colorbar = False,
                     cmap = plt.get_cmap('Spectral_r'),
                     levels=10,
                    #  vmin = 48000,
                    #  vmax = 56500,
                     robust=True,
                     norm=None,
                     center=None,
                     # colors="black",
                     cbar_kwargs=None,
                     # Add colorbar label 
                     add_labels = True)
ax.coastlines(alpha=0.2)
plt.show()

#### Contourf() single
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs_proj))
p = xsphere._contourf(da_single, 
                      x='lon',
                      y='lat',
                     transform=ccrs.PlateCarree(),  
                     ax=ax, 
                     plot_type='contourf',
                     # Color options 
                     add_colorbar = True,
                     cmap = plt.get_cmap('Spectral_r'),
                     levels=10,
                     #  vmin = 48000,
                     #  vmax = 56500,
                     robust=True,
                     norm=None,
                     center=None,
                     # colors="black",
                     cbar_kwargs=None,
                     # Add colorbar label 
                     add_labels = True)
ax.coastlines(alpha=0.2)
plt.show()
#-----------------------------------------------------------------------------.

#### Plot mesh
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs_proj))
p = ds.sphere.plot_mesh(ax=ax, 
                        transform=ccrs.Geodetic(),   
                        add_background=True, 
                        edgecolors = "white", 
                        linewidths = 0.3, alpha=0.4)
plt.show()

#### Plot mesh order    
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs_proj))
p = da.sphere.plot_mesh_order(ax=ax, 
                              transform=ccrs.Geodetic(),
                              cmap=plt.get_cmap('Spectral_r'), 
                              # add_colorbar = True,
                              antialiased = True,
                              edgecolors = "white", 
                              linewidths = 0.3,
                              alpha=1)
ax.coastlines()
plt.show()

#### Plot mesh area 
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs_proj))
p = da.sphere.plot_mesh_area(ax=ax, 
                             transform=ccrs.Geodetic(),
                             cmap=plt.get_cmap('Spectral_r'), 
                             add_colorbar = True,
                             antialiased = True,
                             edgecolors = "white", 
                             linewidths = 0.3,
                             alpha=1)
ax.coastlines()
plt.show()

#### Plot nodes  
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs_proj))
p = ds.sphere.plot_nodes(ax=ax, transform=crs_ref, 
                         add_background=True, 
                         c="orange")
plt.show()

#-----------------------------------------------------------------------------.
