#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:47:01 2021

@author: ghiggi
"""
 
##----------------------------------------------------------------------------.
print(source_group.tree())
source_array = source_group['air']
source_array.info
# no. bytes
# no. bytes stored 
# storage ratio = no. bytes / no. bytes stored 

##----------------------------------------------------------------------------.

# plot meshes --> xarray

# numcodecs.blosc.use_threads = False
# synchronizer =  zarr.ProcessSynchronizer


https://github.com/jweyn/DLWP-CS/blob/master/DLWP/verify.py

##----------------------------------------------------------------------------.
## ERA5 data retrieval 
https://github.com/jweyn/DLWP-CS/blob/master/DLWP/data/era5.py