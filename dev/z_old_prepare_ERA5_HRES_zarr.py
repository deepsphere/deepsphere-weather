#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 15:13:48 2021

@author: ghiggi
"""
import os
import sys

sys.path.append("../")
import time
import zarr
import glob
import dask
import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster
from modules.utils_io import check_no_missing_timesteps
from modules.utils_zarr import rechunk_Dataset
from modules.utils_zarr import write_zarr
from modules.my_io import reformat_pl
from modules.my_io import reformat_toa


## Define data directory
# raw_dataset_dirpath = "/home/ghiggi/Projects/DeepSphere/data/raw/ERA5_HRES"
# zarr_dataset_dirpath = "/home/ghiggi/Projects/DeepSphere/data/preprocessed/ERA5_HRES"

raw_dataset_dirpath = "/ltenas3/DeepSphere/data/raw/ERA5_HRES"
zarr_dataset_dirpath = "/ltenas3/DeepSphere/data/preprocessed/ERA5_HRES"

# Define spherical samplings
spherical_samplings = [
    # 400 km
    # 'Healpix_400km',
    "Icosahedral_400km",
    "O24",
    "Equiangular_400km",
    "Equiangular_400km_tropics",
    "Cubed_400km",
    # # 100 km
    # 'Healpix_100km'
]

# - Global settings
NOVERTICAL_DIMENSION = True  # --> Each pressure level treated as a feature
STACK_VARIABLES = (
    True  # --> Create a DataArray with all features along the "feature" dimension
)
SHOW_PROGRESS = True  # --> Report progress
start_time = "1980-01-01T07:00:00"
end_time = "2018-12-31T23:00:00"

# - Dask settings
# cluster = LocalCluster(n_workers = 30,        # n processes
#                         threads_per_worker=1, # n_workers*threads_per_worker
#                         processes=True, memory_limit='132GB')
# client = Client(cluster)

# - Define variable dictionary
pl_var_dict = {"var129": "z", "var130": "t", "var133": "q"}
toa_var_dict = {"var212": "tisr"}
static_var_dict = {"var172": "lsm", "var43": "slt", "z": "orog"}

# - Define chunking option for the various samplings
chunks_400km = {
    "node": -1,
    "time": 24 * 365 * 1,  # we assume to load the 400 km in memory
    "feature": 1,
}
chunks_100km = {"node": -1, "time": 36, "feature": 1}
chunks_dict = {
    "Healpix_400km": chunks_400km,
    "Icosahedral_400km": chunks_400km,
    "O24": chunks_400km,
    "Equiangular_400km": chunks_400km,
    "Equiangular_400km_tropics": chunks_400km,
    "Cubed_400km": chunks_400km,
    "Healpix_100km": chunks_100km,
}

# - Define compressor option for the various samplings
compressor_400km = zarr.Blosc(cname="zstd", clevel=0, shuffle=1)
compressor_100km = zarr.Blosc(cname="lz4", clevel=0, shuffle=2)
compressor_dict = {
    "Healpix_400km": compressor_400km,
    "Icosahedral_400km": compressor_400km,
    "O24": compressor_400km,
    "Equiangular_400km": compressor_400km,
    "Equiangular_400km_tropics": compressor_400km,
    "Cubed_400km": compressor_400km,
    "Healpix_100km": compressor_100km,
}

# ----------------------------------------------------------------------------.
# start_time = '2005-01-01T00:00:00'
# end_time = '2018-12-31T23:00:00'
# spherical_samplings = ['Healpix_100km']
# sampling = spherical_samplings[0]

# Process all data for model training
for sampling in spherical_samplings:
    print("==================================================================")
    print("Preprocessing", sampling, "raw data")
    t_i = time.time()
    ##------------------------------------------------------------------------.
    ### Define directories
    # - Raw data
    raw_pl_dirpath = os.path.join(
        raw_dataset_dirpath, sampling, "dynamic", "pressure_levels"
    )
    raw_toa_dirpath = os.path.join(
        raw_dataset_dirpath, sampling, "dynamic", "boundary_conditions"
    )
    static_dirpath = os.path.join(raw_dataset_dirpath, sampling, "static")

    # - Zarr data
    dynamic_zarr_fpath = os.path.join(
        zarr_dataset_dirpath,
        sampling,
        "Data",
        "dynamic",
        "time_chunked",
        "dynamic.zarr",
    )
    toa_zarr_fpath = os.path.join(
        zarr_dataset_dirpath, sampling, "Data", "bc", "time_chunked", "bc.zarr"
    )
    static_zarr_fpath = os.path.join(
        zarr_dataset_dirpath, sampling, "Data", "static.zarr"
    )

    ##------------------------------------------------------------------------.
    ### Pressure levels
    print(" - Zarrify pressure levels data")
    # - Retrieve all raw netCDF files
    pl_fpaths = sorted(glob.glob(raw_pl_dirpath + "/pl_*.nc"))
    # - Open all netCDF4 files
    ds = xr.open_mfdataset(
        pl_fpaths,
        parallel=True,
        concat_dim="time",
        # decode_cf = False,
        chunks="auto",
    )
    # ds = ds.decode_cf(ds)
    # - Subset time
    ds = ds.sel(time=slice(start_time, end_time))
    # - Check there are not missing timesteps
    check_no_missing_timesteps(timesteps=ds.time.values)
    # - Unstack pressure levels dimension, create a feature dimension and save to zarr
    if sampling in ["Healpix_100km"]:
        # --> Perform unstacking year per year block
        block_size = 24 * 30 * 2  # TODO depending on sampling
        n_blocks = int(len(ds.time) / block_size + 1)
        append = False
        if os.path.exists(dynamic_zarr_fpath):
            append = True
        for i in range(n_blocks):
            print(i, "/", n_blocks)
            slice_start = block_size * i
            slice_end = block_size * (i + 1)
            if i == n_blocks - 1:
                slice_end = None
            tmp_ds = ds.isel(time=slice(slice_start, slice_end))
            ds = reformat_pl(
                ds=tmp_ds,
                var_dict=pl_var_dict,
                unstack_plev=NOVERTICAL_DIMENSION,
                stack_variables=STACK_VARIABLES,
            )
            # - Write data
            write_zarr(
                zarr_fpath=dynamic_zarr_fpath,
                ds=ds,
                chunks=chunks_dict[sampling],
                compressor=compressor_dict[sampling],
                consolidated=True,
                show_progress=SHOW_PROGRESS,
                append=append,
                append_dim="time",
            )
            append = True
    else:
        ds = reformat_pl(
            ds=ds,
            var_dict=pl_var_dict,
            unstack_plev=NOVERTICAL_DIMENSION,
            stack_variables=STACK_VARIABLES,
        )
        # - Write data to zarr
        write_zarr(
            zarr_fpath=dynamic_zarr_fpath,
            ds=ds,
            chunks=chunks_dict[sampling],
            compressor=compressor_dict[sampling],
            consolidated=True,
            append=False,
            show_progress=SHOW_PROGRESS,
        )
    ##------------------------------------------------------------------------.
    ### TOA
    print(" - Zarrify TOA data")
    # - Retrieve all raw netCDF files
    toa_fpaths = sorted(glob.glob(raw_toa_dirpath + "/toa_*.nc"))
    # - Open all netCDF4 files
    ds = xr.open_mfdataset(toa_fpaths)
    ds = ds.sel(time=slice(start_time, end_time))
    # - Check there are not missing timesteps
    check_no_missing_timesteps(timesteps=ds.time.values)
    # - Stack TOA into 'feature' dimension
    ds = reformat_toa(ds=ds, var_dict=toa_var_dict, stack_variables=STACK_VARIABLES)
    # - Write data to zarr
    write_zarr(
        zarr_fpath=toa_zarr_fpath,
        ds=ds,
        chunks=chunks_dict[sampling],
        compressor=compressor_dict[sampling],
        consolidated=True,
        show_progress=SHOW_PROGRESS,
        append=False,
    )

    ##------------------------------------------------------------------------.
    ### Static features
    print(" - Zarrify static data")
    # - Retrieve all raw netCDF files
    static_fpaths = glob.glob(static_dirpath + "/*/*.nc", recursive=True)
    l_ds = []
    for fpath in static_fpaths:
        tmp_ds = xr.open_dataset(fpath)
        tmp_ds = tmp_ds.squeeze()
        tmp_ds = tmp_ds.drop_vars(["time"])  # causing problem ...
        l_ds.append(tmp_ds)
    ds = xr.merge(l_ds)
    ds = ds.drop_vars(["lon_bnds", "lat_bnds", "lev"])
    ds = ds.drop_vars(["hyai", "hybi", "hyam", "hybm"])
    ds = ds.rename({"ncells": "node"})
    ds = ds.rename(static_var_dict)
    # -----------------------------------------.
    # - Fix orography to be 0 lower bound
    da_orog = ds["orog"].load()
    da_orog[da_orog < 0] = 0
    ds["orog"] = da_orog
    # -----------------------------------------.
    # - Add latitude and longitude feature
    ds["sin_latitude"] = np.sin(ds["lat"])
    ds["sin_longitude"] = np.sin(ds["lon"] / np.pi)
    ds["cos_longitude"] = np.sin(ds["lon"] / np.pi)
    # -----------------------------------------.
    # - Stack all variables into the 'feature' dimension
    # da_stacked = ds.to_stacked_array(new_dim="feature", variable_dim='variable',
    #                                  sample_dims=list(ds.dims.keys()),
    #                                  name = 'data')
    # # - Remove MultiIndex for compatibility with netCDF and Zarr
    # da_stacked = da_stacked.reset_index('feature')
    # da_stacked = da_stacked.set_index(feature='variable', append=True)
    # # - Remove attributes from DataArray
    # da_stacked.attrs = {}
    # da_stacked.encoding = {}
    # # - Reshape to Dataset
    # ds = da_stacked.to_dataset()
    # - Write to zarr
    ds.to_zarr(static_zarr_fpath)
    # -------------------------------------------------------------------------.
    # Report elapsed time
    print("---> Elapsed time: {:.1f} minutes ".format((time.time() - t_i) / 60))
    print("==================================================================")
    # -------------------------------------------------------------------------.
# -----------------------------------------------------------------------------.

# -----------------------------------------------------------------------------.
# Rechunk dynamic and bc data across space for efficient time statistics
spherical_samplings = [
    "Healpix_100km",
    # 400 km
    "Healpix_400km",
    "Icosahedral_400km",
    "O24",
    "Equiangular_400km",
    "Equiangular_400km_tropics",
    "Cubed_400km",
    # # 100 km
    # 'Healpix_100km'
]
for sampling in spherical_samplings:
    print("==================================================================")
    print("Rechunking", sampling, "data over space")
    t_i = time.time()
    ##------------------------------------------------------------------------.
    ### Define directories
    # - Source "time-chunked" zarr data
    dynamic_timechunked_zarr_fpath = os.path.join(
        zarr_dataset_dirpath,
        sampling,
        "Data",
        "dynamic",
        "time_chunked",
        "dynamic.zarr",
    )
    toa_timechunked_zarr_fpath = os.path.join(
        zarr_dataset_dirpath, sampling, "Data", "bc", "time_chunked", "bc.zarr"
    )
    # - Destination "space-chunked" zarr data
    dynamic_spacechunked_zarr_fpath = os.path.join(
        zarr_dataset_dirpath,
        sampling,
        "Data",
        "dynamic",
        "space_chunked",
        "dynamic.zarr",
    )
    toa_spacechunked_zarr_fpath = os.path.join(
        zarr_dataset_dirpath, sampling, "Data", "bc", "space_chunked", "bc.zarr"
    )
    # - Define temporary directory
    tmp_fpath = os.path.join(zarr_dataset_dirpath, "tmp")
    ##------------------------------------------------------------------------.
    ### Define chunks
    chunks = {
        "node": 1,  # chunked across space (each pixel)
        "time": -1,  # unchunked across time
        "feature": 1,
    }

    ##------------------------------------------------------------------------.
    ### Rechunk Zarr data
    # Open dynamic dataset and rechunk
    ds_dynamic = xr.open_zarr(dynamic_timechunked_zarr_fpath)
    ds_dynamic["feature"] = ds_dynamic["feature"].astype(str)
    rechunk_Dataset(
        ds=ds_dynamic,
        chunks=chunks,
        target_store=dynamic_spacechunked_zarr_fpath,
        temp_store=os.path.join(tmp_fpath, "tmp_store.zarr"),
        max_mem="2GB",
    )

    # Open TOA dataset and rechunk
    ds_bc = xr.open_zarr(toa_timechunked_zarr_fpath)
    ds_bc["feature"] = ds_bc["feature"].astype(str)
    rechunk_Dataset(
        ds=ds_bc,
        chunks=chunks,
        target_store=toa_spacechunked_zarr_fpath,
        temp_store=os.path.join(tmp_fpath, "tmp_store.zarr"),
        max_mem="2GB",
    )
    # -------------------------------------------------------------------------.
    # Report elapsed time
    print("---> Elapsed time: {:.1f} minutes ".format((time.time() - t_i) / 60))
    print("==================================================================")
    # -------------------------------------------------------------------------.
# -----------------------------------------------------------------------------.
