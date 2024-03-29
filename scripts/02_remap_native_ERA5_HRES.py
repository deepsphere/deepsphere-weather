#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:57:14 2020

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
    "Icosahedral_400km",
    "O24",
    "Equiangular_400km",
    "Equiangular_400km_tropics",
    "Cubed_400km",
    # 100 km
    "Healpix_100km",
]
# Define dataset to remap
dataset = "ERA5_HRES"

# Define variable types to remap
variable_types = ["static", "dynamic"]

##-----------------------------------------------------------------------------.
# Remap
for sampling in spherical_samplings:
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
            force_remapping=True,
        )

##----------------------------------------------------------------------------.
# Re-Remap static variables because of a bug in genlaf in CDO < 2.0
# - Bug report: https://code.mpimet.mpg.de/boards/1/topics/11137?r=11181#message-11181
variable_types = ["static"]
for sampling in spherical_samplings:
    for variable_type in variable_types:
        remap_grib_files(
            data_dir=data_dir,
            CDO_grids_dir=CDO_grids_dir,
            CDO_grids_weights_dir=CDO_grids_weights_dir,
            dataset=dataset,
            sampling=sampling,
            variable_type=variable_type,
            precompute_weights=False,  # because of CDO BUG !
            normalization="fracarea",
            compression_level=1,
            n_threads=4,
            force_remapping=False,
        )
