#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 13:57:22 2020

@author: ghiggi
"""

import os
import sys
sys.path.append('../')
import numpy as np 
import xarray as xr 
import cf2cdm
from modules.my_remap import remap_grib_files

### Define folder paths
proj_dir = "/ltenas3/DeepSphere/"
data_dir = "/ltenas3/DeepSphere/data/raw"

fpath = "/ltenas3/DeepSphere/ftp.ecmwf.int/data_EPFL/dynamic/fc_en_t850_G.grib"
fpath = "/ltenas3/DeepSphere/ftp.ecmwf.int/data_EPFL/dynamic/fc_hrs_t850_G.grib" 
ds = xr.open_dataset(fpath, engine="cfgrib")
ds.valid_time
cf2cdm.translate_coords(ds, cf2cdm.ECMWF)
cf2cdm.translate_coords(ds, cf2cdm.CDS)

# ECMWF = CDS
# time = forecast_reference_time # forecast time
# valid_time = time               # forecasted time
# level = plev
# number= realization


#  combine along time / forecast_reference_time

##-----------------------------------------------------------------------------.
# Check order nodes as atlas-mesh 
grib_lat = ds.latitude.values
grib_lon = ds.longitude.values
grib_lonlat = np.vstack((grib_lon,grib_lat)).transpose()
grib_lonlat.shape

# Read atlas-meshgen file 
filepath = "/ltenas3//DeepSphere/grids/grids_ECMWF/O1280.msh"
filepath = "/ltenas3//DeepSphere/grids/grids_ECMWF/O640.msh"
from uremap.remap import read_ECMWF_atlas_msh
node_lon, node_lat, _ = read_ECMWF_atlas_msh(filepath)
node_lonlat = np.vstack((node_lon,node_lat)).transpose()
node_lat.shape


np.allclose(grib_lonlat,node_lonlat)
 