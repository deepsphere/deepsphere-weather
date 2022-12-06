#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:57:14 2020

@author: ghiggi
"""
import os
import sys
import pygsp as pg

sys.path.append("../")
from xsphere.remapping import pygsp_to_CDO_grid

### Define folder paths
proj_dir = "/ltenas3/data/DeepSphere/"
# proj_dir = "/home/ghiggi/Projects/DeepSphere"
CDO_grids_dir = os.path.join(proj_dir, "grids", "CDO_grids")
# -----------------------------------------------------------------------------.
# Define Spherical Samplings
spherical_samplings_dict = {
    # 400 km
    "Healpix_400km": pg.graphs.SphereHealpix(subdivisions=16, nest=True),
    "Icosahedral_400km": pg.graphs.SphereIcosahedral(subdivisions=16),
    "O24": pg.graphs.SphereGaussLegendre(nlat=48, nlon="ecmwf-octahedral"),
    "Cubed_400km": pg.graphs.SphereCubed(subdivisions=24),
    "Equiangular_400km": pg.graphs.SphereEquiangular(nlat=36, nlon=72, poles=0),
    "Equiangular_400km_tropics": pg.graphs.SphereEquiangular(nlat=46, nlon=92, poles=0),
    # 100 km
    "Healpix_100km": pg.graphs.SphereHealpix(subdivisions=64, nest=True),
}

# spherical_sampling = "Healpix_100km"
# graph = pg.graphs.SphereHealpix(subdivisions=64, k=20, nest=True)
# -----------------------------------------------------------------------------.
# Define CDO grid for remapping
for spherical_sampling, graph in spherical_samplings_dict.items():
    # Define filename and filepath of CDO grids
    CDO_grid_fpath = os.path.join(CDO_grids_dir, spherical_sampling)
    # Write CDO grid
    pygsp_to_CDO_grid(graph=graph, CDO_grid_fpath=CDO_grid_fpath)

# -----------------------------------------------------------------------------.
