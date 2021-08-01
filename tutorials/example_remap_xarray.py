#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:31:40 2020

@author: ghiggi
"""
### Currently rely on CDO for interpolation weights definition 
# pip install --upgrade git+https://github.com/epfl-lts2/pygsp@sphere-graphs
# conda install -c conda-forge trimesh
# conda install -c conda-forge cdo      [version >= 1.9.8 !!!]

### CDO info 
# - CDO user guide: https://code.mpimet.mpg.de/projects/cdo/embedded/cdo.pdf
# - Source code for remapping: https://earth.bsc.es/gitlab/ces/cdo/-/blob/master/src/Remap.c

## Remapping notes
# - Currently using YAC. 
# - CDO remapping functions name change between CDO 1.9.8 and 1.9.9. 
# - REMAPYCON becomes REMAPCON in CDO 2.0.0 ? 
# REMAPCON --> SCRIP first order conservative [going to be deprecated soon?]
# REMAPYCON --> YAC first order conservative 

##-----------------------------------------------------------------------------.
import os
import numpy as np 
os.chdir('/home/ghiggi/Projects/DeepSphere')
import pygsp as pg
import healpy as hp
import xarray as xr
from uremap.remap import remap_dataset

##-----------------------------------------------------------------------------.
### Load sample data
sample_dir = "/home/ghiggi/"
ds_Healpix_64_notime = xr.open_dataset(os.path.join(sample_dir, "Healpix64_sample_notime.nc")) 
ds_Healpix_64_time = xr.open_dataset(os.path.join(sample_dir, "Healpix64_sample_2timesteps.nc")) 
ds_Healpix_16_notime = xr.open_dataset(os.path.join(sample_dir, "Healpix16_sample_notime.nc")) 
ds_Healpix_16_time = xr.open_dataset(os.path.join(sample_dir, "Healpix16_sample_2timesteps.nc")) 

### Define spherical samplings 
Healpix64 = pg.graphs.SphereHealpix(subdivisions=64, k=20) # 400 km 
Healpix16 = pg.graphs.SphereHealpix(subdivisions=16, k=20) # 400 km 
Healpix8 = pg.graphs.SphereHealpix(subdivisions=8, k=20)   
Healpix4 = pg.graphs.SphereHealpix(subdivisions=4, k=20)  
Healpix2 = pg.graphs.SphereHealpix(subdivisions=2, k=20) 

##-----------------------------------------------------------------------------.
### Remapping xarray Datasets 
# Important points: 
# - CDO returns the node dimension as 'ncells' (relevant when reading remapped Dataset from disk)
# - CDO expect the time to be the first dimension !
# --> ds = ds.transpose('time','nodes') # Set time as first dimension 
# - Conservative remapping is meant to keep the intergral of fields after the interpolation

# Output dst_ds for unstructured grids
# - lon, lat contains the destination pygsp graph nodes   
# - lon_bnds, lat_bnds contains the mesh vertices obtained using SphericalVoronoi    

ds_Healpix8 = remap_dataset(src_ds=ds_Healpix_16_time, 
                            src_graph=Healpix16, 
                            dst_graph=Healpix8,
                            method="conservative",
                            normalization="fracarea",
                            n_threads=4)
 
ds_Healpix16_back = remap_dataset(src_ds=ds_Healpix8, 
                                  src_graph=Healpix8, 
                                  dst_graph=Healpix16,
                                  method="conservative",
                                  normalization="fracarea",
                                  n_threads=4)

#----------------------------------------------------------------------------.
### Check if difference between fracarea and destarea
# - No difference expected if no missing data 
# - When there are nan, these are set to 0 (for summing).
# - fracarea, and destarea difference explained here:
#   https://github.com/JiaweiZhuang/xESMF/issues/17
ds_fracarea = remap_dataset(src_ds=ds_Healpix_16_time, 
                            src_graph=Healpix16, 
                            dst_graph=Healpix8,
                            method="conservative",
                            normalization="fracarea",
                            n_threads=4)
ds_destarea = remap_dataset(src_ds=ds_Healpix_16_time, 
                            src_graph=Healpix16, 
                            dst_graph=Healpix8,
                            method="conservative",
                            normalization="destarea",
                            n_threads=4)

np.allclose(ds_fracarea.t.values, ds_destarea.t.values) # strange

#----------------------------------------------------------------------------.
### Check conservation
# TODO 
ds_Healpix8_back = remap_dataset(src_ds=ds_Healpix16_back, 
                                 src_graph=Healpix16, 
                                 dst_graph=Healpix8,
                                 method="conservative",
                                 normalization="fracarea",
                                 n_threads=4)

ds_Healpix8.t.values
ds_Healpix8_back.t.values

##------------------------------------------------------------------------.
# - Healpix100km  (to see performance at 100 km)
ds_Healpix8_tmp = remap_dataset(src_ds=ds_Healpix_64_time, 
                                src_graph=Healpix64, 
                                dst_graph=Healpix8,
                                method="conservative",
                                normalization='fracarea',
                                n_threads=4)

##------------------------------------------------------------------------.
### If we want to benchmark all predictions to coarser common sampling
# - Following code allow to remap predictions to coarser sampling 
# remap_dataset(src_ds = ds_predictions_sampling, 
#               src_graph = pygsp_graph_src_sampling
#               dst_graph = pygsp_graph_common_coarse_sampling,  
#               method = "conservative",
#               normalization = 'fracarea', 
#               ds_fpath = fpath_remapped_coarse,
#               return_remapped_ds = False) 

##-------------------------------------------------------------------------. 
### Other remapping options for unstructured grids
# - If the src grid is structured --> bilinear and bicubic interpolation is available 
# - If the src grid is structured --> Provide src_CDO_grid_fpath instead
ds_nn = remap_dataset(src_ds=ds_Healpix_16_time, 
                      src_graph=Healpix16, 
                      dst_graph=Healpix8,
                      method="nearest_neighbors",
                      n_threads=4)
ds_idw = remap_dataset(src_ds=ds_Healpix_16_time, 
                       src_graph=Healpix16, 
                       dst_graph=Healpix8,
                       method="idw", # inverse distance weighting
                       n_threads=4)
ds_laf = remap_dataset(src_ds=ds_Healpix_16_time, 
                       src_graph=Healpix16, 
                       dst_graph=Healpix8,
                       method="largest_area_fraction",
                       n_threads=4)

#-----------------------------------------------------------------------------.
### Testing / Checks  
## - Check for CDO warning when no time dimension 
ds_Healpix8_notime = remap_dataset(src_ds=ds_Healpix_16_notime, 
                                   src_graph=Healpix16, 
                                   dst_graph=Healpix8,
                                   method="conservative",
                                   normalization = "fracarea",
                                   n_threads=4)
ds_Healpix8_time = remap_dataset(src_ds=ds_Healpix_16_time, 
                                 src_graph=Healpix16, 
                                 dst_graph=Healpix8,
                                 method="conservative",
                                 normalization = "fracarea",
                                 n_threads=4)
assert np.allclose(ds_Healpix8_notime.t.values, ds_Healpix8_time.t.isel(time=0).values)                
 
##----------------------------------------------------------------------------.
## - Check lon-lat correspond to dst_graph 
# - pygsp longitudes are between 0-360.
# - remap_dataset() output longitudes are between -180-180
dst_graph = Healpix8
dst_ds = ds_Healpix8_time
lon_graph = dst_graph.signals['lon']*180/np.pi 
lat_graph = dst_graph.signals['lat']*180/np.pi
lon_graph[lon_graph>180] = lon_graph[lon_graph>180] - 360 

assert np.allclose(dst_ds.lat.values, lat_graph)
assert np.allclose(dst_ds.lon.values, lon_graph)

##----------------------------------------------------------------------------.
## - Check output lon_bnds, lat_bnds are SphericalVoronoi mesh of dst_graph 
from uremap.remap import SphericalVoronoiMesh_from_pygsp
from uremap.remap import get_lat_lon_bnds
dst_graph = Healpix8
dst_ds = ds_Healpix8_time

lon_bnds, lat_bnds = get_lat_lon_bnds(SphericalVoronoiMesh_from_pygsp(dst_graph))
 
assert np.allclose(dst_ds.lat_bnds.values, lat_bnds)
assert np.allclose(dst_ds.lon_bnds.values, lon_bnds)

##----------------------------------------------------------------------------.
## - Check remapping works also without lon_bnds, lat_bnds info in the src 
# - lon_bnds, lat_bnds should be provided by the CDO grid  
src_ds_with_bnds = ds_Healpix_16_time
src_ds_no_bnds = src_ds_with_bnds.drop_vars(['lon_bnds','lat_bnds'])
dst_ds = remap_dataset(src_ds=src_ds_no_bnds, 
                       src_graph=Healpix16, 
                       dst_graph=Healpix8,
                       method="conservative",
                       normalization = "fracarea",
                       n_threads=4)

