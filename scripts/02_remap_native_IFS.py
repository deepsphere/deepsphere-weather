#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 14:40:14 2020

@author: ghiggi
"""
import os
import sys

sys.path.append("../")
from modules.my_remap import remap_grib_files

### Define folder paths
proj_dir = "/ltenas3/data/DeepSphere/"
data_dir = "/ltenas3/data/DeepSphere/data/raw"

CDO_grids_dir = os.path.join(proj_dir, "grids", "CDO_grids")
CDO_grids_weights_dir = os.path.join(proj_dir, "grids", "CDO_grids_weights")

##-----------------------------------------------------------------------------.
# Define spherical samplings to remap
spherical_samplings = [
    # 400 km
    "Healpix_400km",
    # 'Icosahedral_400km',
    # 'O24',
    # 'Equiangular_400km',
    # 'Equiangular_400km_tropics',
    # 'Cubed_400km',
    ## 100 km
    # 'Healpix_100km'
]
# Define dataset to remap
datasets = ["IFS_HRES", "IFS_ENS"]

# Define variable types to remap
variable_types = ["dynamic"]

##-----------------------------------------------------------------------------.
# Remap
for sampling in spherical_samplings:
    for dataset in datasets:
        for variable_type in variable_types:
            remap_grib_files(
                data_dir=data_dir,
                CDO_grids_dir=CDO_grids_dir,
                CDO_grids_weights_dir=CDO_grids_weights_dir,
                dataset=dataset,
                sampling=sampling,
                variable_type=variable_type,
                precompute_weights=True,
                normalization="fracarea",
                compression_level=1,
                n_threads=4,
                force_remapping=False,
            )
