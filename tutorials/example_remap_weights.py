#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 17:45:35 2020

@author: ghiggi
"""
### Currently rely on CDO for interpolation weights definition 
# pip install --upgrade git+https://github.com/epfl-lts2/pygsp@sphere-graphs
# conda install -c conda-forge trimesh
# conda install -c conda-forge cdo      [version >= 1.9.8 !!!]  

# !!! 4 Wentao --> Change uremap.remap with modules.remap in your project 
import os
import numpy as np 
os.chdir('/home/ghiggi/Projects/DeepSphere')
import pygsp as pg
import healpy as hp
from uremap.remap import compute_interpolation_weights
from uremap.remap import get_available_interp_methods

##----------------------------------------------------------------------------.
### Define spherical samplings 
Healpix16 = pg.graphs.SphereHealpix(subdivisions=16, k=20) # 400 km 
Healpix8 = pg.graphs.SphereHealpix(subdivisions=8, k=20)   
Healpix4 = pg.graphs.SphereHealpix(subdivisions=4, k=20)  
Healpix2 = pg.graphs.SphereHealpix(subdivisions=2, k=20) 

O24 = pg.graphs.SphereGaussLegendre(nlat=48, nlon='ecmwf-octahedral') # 024
O12 = pg.graphs.SphereGaussLegendre(nlat=24, nlon='ecmwf-octahedral')
O6 = pg.graphs.SphereGaussLegendre(nlat=12, nlon='ecmwf-octahedral')
O3 = pg.graphs.SphereGaussLegendre(nlat=6, nlon='ecmwf-octahedral')

Cubed24 = pg.graphs.SphereCubed(subdivisions=24)
Cubed12 = pg.graphs.SphereCubed(subdivisions=12)
Cubed6 = pg.graphs.SphereCubed(subdivisions=6)
Cubed3 = pg.graphs.SphereCubed(subdivisions=3)

Healpix100km = pg.graphs.SphereHealpix(subdivisions=64, k=20)
Healpix32 = pg.graphs.SphereHealpix(subdivisions=32, k=20)
# pg.graphs.SphereIcosahedral(subdivisions=16, k=10)
# pg.graphs.SphereEquiangular(nlat=36, nlon=72, poles=0),

##----------------------------------------------------------------------------.
# Available interpolation/remapping methods 
# - conservativ2 --> 2nd order conservative remapping 
get_available_interp_methods()

##----------------------------------------------------------------------------.
# Generate weights between whatever pygsp spherical sampling 
ds_weights = compute_interpolation_weights(src_graph = Healpix16,
                                           dst_graph = Healpix8,
                                           method = "conservative",    
                                           normalization = 'fracarea') # destarea’

ds_weights_back = compute_interpolation_weights(src_graph = Healpix8,
                                                dst_graph = Healpix16,
                                                method = "conservative",    
                                                normalization = 'fracarea') # destarea’

# Computing time from grid at 100 km to ~ 200 km 
ds_weight_high_res = compute_interpolation_weights(src_graph = Healpix100km,
                                                   dst_graph = Healpix32,
                                                   method = "conservative",    
                                                   normalization = 'fracarea')

#-----------------------------------------------------------------------------.
### TODO GG [TOMORROW]
# - Benchmark weights real data and new sampling with pygsp-pygsp to ensure no bug
# - Provide real sample data 

            
## TODO Wentao []
# Generalized pooling --> Multiplications with sparse matrices from ds_weights
# Ensure conservative remapping is done correctly  (normalization, ...)
# --> dst_grid_frac, dst_grid_area

# - 'fracarea' uses the sum of the source cell intersected areas to normalize 
#   each destination cell field value. 
#   Flux is not locally conserved.
# - 'destarea' uses the total destination cell area to normalize each destination
#   cell field value. Local flux conservation is ensured, but unreasonable flux values 
#   may result [i.e. in small patches].

## Normalization : dst_grid_frac*dst_grid_area ?
     
#-----------------------------------------------------------------------------.

## Full function 
# ds_weights = compute_interpolation_weights(src_graph = Healpix16,
#                                           dst_graph = Healpix8,
#                                           method = "conservative", 
#                                           normalization = 'fracarea',
#                                           weights_fpath = None,
#                                           src_CDO_grid_fpath = None, 
#                                           dst_CDO_grid_fpath = None, 
#                                           recreate_CDO_grids = False,
#                                           return_weights = True)

## Spherica sampling that I will provide as soon as possible 
spherical_samplings_dict = {
    # 400 km
    'Healpix_400km': pg.graphs.SphereHealpix(subdivisions=16, k=20),
    'Icosahedral_400km': pg.graphs.SphereIcosahedral(subdivisions=16, k=10),
    'O24': pg.graphs.SphereGaussLegendre(nlat=48, nlon='ecmwf-octahedral'),
    'Cubed_400km': pg.graphs.SphereCubed(subdivisions=24),
    'Equiangular_400km': pg.graphs.SphereEquiangular(nlat=36, nlon=72, poles=0),
    'Equiangular_400km_tropics': pg.graphs.SphereEquiangular(nlat=46, nlon=92, poles=0),
    # 100 km 
    'Healpix_100km': pg.graphs.SphereHealpix(subdivisions=64, k=20)
}








