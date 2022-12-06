#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 06:33:57 2021

@author: ghiggi
"""
import os   
import glob
import shutil
import numcodecs
import numpy as np
import xarray as xr 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from xforecasting.utils.zarr import (
    # get_reading_time,
    # get_reading_throughput,
    # get_storage_ratio_zarr,
    get_memory_size_chunk, 
    get_memory_size_zarr, 
    get_memory_size_dataset,
    _getlossless_compressors,
    profile_zarr_io,
    write_zarr
)
from modules.my_io import reformat_pl  

##----------------------------------------------------------------------------.
# Thoughts: 
# - Here the compressor is choosed based on optimization of a specific feature
#   This might be suboptimal, because a different feature may have other data characteristics
# - How much rounding/precision impact the compression?
# - How chunksize impact the compression? But zarr recommends chunks of 100-200 MB!

# Hourly data 
# 1 days = 24 samples
# 2 days = 48 samples   
# 3 days = 72 samples      
# 7 days = 168 samples 
# 14 days = 336 samples 
# 30 days = 720 samples 
# 120 days = 2880 samples 
# 365 days = 8760 samples 

##----------------------------------------------------------------------------.
# Define paths and settings 
data_dir = "/ltenas3/data/DeepSphere/data" 
zarr_benchmark_dir = "/ltenas3/data/DeepSphere/data/zarr_benchmark"
sampling = "Healpix_400km"
NOVERTICAL_DIMENSION = True # --> Each pressure level treated as a feature 
STACK_VARIABLES = True      # --> Create a DataArray with all features along the "feature" dimension
SHOW_PROGRESS = False

##----------------------------------------------------------------------------.
### Define chunking of a dataset [ideally between 100 and 200 MB]
# - Open dataset 
raw_pl_dirpath = os.path.join(data_dir,"raw","ERA5_HRES", sampling, "dynamic", "pressure_levels")
pl_fpaths = sorted(glob.glob(raw_pl_dirpath + "/pl_*.nc"))
pl_var_dict = {'var129': 'z',
               'var130': 't',
               'var133': 'q'}
# - Open all netCDF4 files and reshape to the desired format 
ds_orig = xr.open_mfdataset(pl_fpaths, parallel = True,
                            concat_dim = "time",
                            # decode_cf = False, 
                            chunks = "auto")
ds = ds_orig.isel(time = slice(0, 24*366*3)) # Subset 5 year of data 
ds = reformat_pl(ds = ds, 
                 var_dict = pl_var_dict, 
                 unstack_plev = NOVERTICAL_DIMENSION, 
                 stack_variables = STACK_VARIABLES)

# - Define chunk sizes 
time_chunks = [24*30*m for m in range(1,12)] + [24*365*y for y in range(1,3)]

# - Measure the chunk memory size 
chunk_size_dict = {}
for time_chunk in time_chunks: 
    chunks = {'node': -1,
              'time': time_chunk,  
              'feature': 1} 
    ds = ds.chunk(chunks)
    chunk_size_dict[time_chunk] = get_memory_size_chunk(ds['data'])

# - Plot 
chunk_sizes = [int(s) for s in chunk_size_dict.keys()]
chunk_memory = list(chunk_size_dict.values())
plt.plot(chunk_sizes , chunk_memory, ".")
plt.xlabel("Chunking in time dimension")
plt.ylabel("Chunk size [MB]")

##----------------------------------------------------------------------------.
### Write a temporary Zarr store, with a single chunk and 1 feature of the full dataset
# Select time_chunks 
chunks = {'node': -1,
          'time': 24*365*1,  
          'feature': 1} 
compressor = numcodecs.Blosc(cname="lz4", clevel=0, shuffle=2) 
ds = ds_orig.isel(time = slice(0, chunks['time']))  
ds = reformat_pl(ds = ds, 
                 var_dict = pl_var_dict, 
                 unstack_plev = NOVERTICAL_DIMENSION, 
                 stack_variables = STACK_VARIABLES)

# - Select only 1 feature
ds = ds.isel(feature=slice(0,1)) 

# - Write data to zarr
tmp_ds_chunk_fpath = os.path.join(zarr_benchmark_dir,"tmp_ds.zarr")
write_zarr(zarr_fpath = tmp_ds_chunk_fpath, 
           ds = ds,  
           chunks = chunks, 
           compressor = compressor,
           consolidated = True,
           append = False, 
           show_progress = True) 

#-----------------------------------------------------------------------------.
### Choose the best compressor for a specific feature 
# - Load dataset and select 1 feature 
ds = xr.open_zarr(tmp_ds_chunk_fpath)
ds['feature'].encoding = {} # to avoid zarr writing bug

# - Look at the Dataset size 
print(get_memory_size_zarr(tmp_ds_chunk_fpath))
print(get_memory_size_dataset(ds))

# - Define compressor options 
compressors = _getlossless_compressors(clevels=[0])
len(compressors)

# compressors = _getlossless_compressors(clevels=[0,1,3,5,9])
# len(compressors)
# compressors = {k:v for i, (k,v) in enumerate(compressors.items()) if i < 24}

# - Set number of repetitions when profiling 
n_repetitions = 5

# - Start profiling 
timing_dict = {}
for cname, compressor in compressors.items(): 
    print("- Profiling", cname)
    tmp_zarr_fpath = os.path.join(zarr_benchmark_dir, cname + ".zarr")
    timing_dict[cname] = profile_zarr_io(ds = ds,
                                         fpath = tmp_zarr_fpath,
                                         chunks = None,  # Use current chunking 
                                         compressor = compressor,
                                         # isel_dict = {}, 
                                         consolidated = True, 
                                         n_repetitions = n_repetitions)

# - Plot and choose the best compressor 
cnames = list(timing_dict.keys())
select_best = 15 
show_best = True
df_dict = {cname: timing_dict[cname]["writing"] for cname in cnames}
df = pd.DataFrame(df_dict)
index_sort = df.median().sort_values(ascending=show_best).index
df_sorted = df[index_sort].iloc[:, :select_best]
sns.boxplot(data = df_sorted, orient='h', showfliers=False)
plt.xlabel('Writing time [seconds]')
plt.show()

df_dict = {cname: timing_dict[cname]["reading"] for cname in cnames}
df = pd.DataFrame(df_dict)
index_sort = df.median().sort_values(ascending=show_best).index
df_sorted = df[index_sort].iloc[:, :select_best]
sns.boxplot(data = df_sorted, orient='h', showfliers=False)
plt.xlabel('Reading time [seconds]')
plt.show()

df_dict = {cname: timing_dict[cname]["reading_throughput"] for cname in cnames}
df = pd.DataFrame(df_dict)
index_sort = df.median().sort_values(ascending= not show_best).index
df_sorted = df[index_sort].iloc[:, :select_best]
sns.boxplot(data = df_sorted, orient='h', showfliers=False)
plt.xlabel('Reading throughput [MB/s]')
plt.show()

selected_cnames = df_sorted.columns.values
compression_ratios = [timing_dict[cname]["compression_ratio"] for cname in selected_cnames]
index_sort = np.argsort(compression_ratios)
compression_ratios = np.array(compression_ratios)[index_sort]  
sorted_cnames = np.array(selected_cnames)[index_sort]
sns.barplot(x = compression_ratios, y = sorted_cnames,  orient='h')
plt.xlabel('Compression ratio')
plt.xlim((1,None))
plt.show()

#-----------------------------------------------------------------------------.
# Remove the temporary Zarr Store 
shutil.rmtree(tmp_ds_chunk_fpath)
 
##----------------------------------------------------------------------------.
# # Compression ratio vs reading throughput  
# fig, ax = plt.subplots()
# for cname in cnames: 
#     ax.scatter(np.median(timing_dict[cname]["compression_ratio"]),
#             np.median(timing_dict[cname]["reading_throughput"]))
#     plt.xlabel('Compression ratio')    
#     plt.ylabel('Reading throughput [MB/s]')
# plt.show()


# # Reading vs reading throughput  
# fig, ax = plt.subplots()
# for cname in cnames: 
#     ax.scatter(np.median(timing_dict[cname]["reading"]),
#             np.median(timing_dict[cname]["reading_throughput"]))
#     plt.xlabel('Reading time [seconds]')    
#     plt.ylabel('Reading throughput [MB/s]')
# plt.show()


# # Writing vs reading throughput  
# fig, ax = plt.subplots()
# for cname in cnames: 
#     ax.scatter(np.median(timing_dict[cname]["writing"]),
#             np.median(timing_dict[cname]["reading_throughput"]))
#     plt.xlabel('Writing time [seconds]')    
#     plt.ylabel('Reading throughput [MB/s]')
#     plt.legend(labels=cnames)
# plt.show() 
 
 

 






