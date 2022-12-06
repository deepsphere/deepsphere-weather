#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 19:16:10 2021

@author: ghiggi
"""
import os
import sys

sys.path.append("../")
import time
import numpy as np
import dask
import xarray as xr
import xverif
import xsphere  #  it load the xarray sphere accessor !
from xscaler import LoadClimatology

##----------------------------------------------------------------------------.
#### Compute climatology and persistence skills
def compute_verif(sampling_name, base_data_dir, test_period, list_climatologies):

    ### Define the sampling-specific folder
    data_dir = os.path.join(base_data_dir, sampling_name)
    ##------------------------------------------------------------------------.
    ### Load dynamic data
    print("- Loading Dataset", sep="")
    dynamic_fpath = os.path.join(
        data_dir, "Data", "dynamic", "time_chunked", "dynamic.zarr"
    )
    data_dynamic = xr.open_zarr(dynamic_fpath, chunks="auto")

    # - Retrieve data for test period
    data_dynamic = data_dynamic.sel(time=slice(test_period[0], test_period[-1]))

    # - Load data into memory
    t_i = time.time()
    data_dynamic = data_dynamic.load()
    print(" (Elapsed time: {:.0f}s)".format((time.time() - t_i)), sep="\n")
    ##------------------------------------------------------------------------.
    #### Compute climatology forecast skils
    print(
        "- Computing climatology forecasts skills for {} sampling".format(sampling_name)
    )
    for clim_name in list_climatologies:
        print("  - Computing {} forecast skill".format(clim_name))
        # - Load Climatology
        tmp_clim_fpath = os.path.join(
            data_dir, "Climatology", clim_name + "_dynamic.nc"
        )
        tmp_clim = LoadClimatology(tmp_clim_fpath)
        # - Create climatology forecasts
        ds_clim_forecast = tmp_clim.forecast(data_dynamic["time"].values)
        # - Rechunk dataset
        data_dynamic = data_dynamic.chunk({"time": -1, "node": 1})
        ds_clim_forecast = ds_clim_forecast.chunk({"time": -1, "node": 1})
        # - Compute deterministic spatial skills
        ds_skill = xverif.deterministic(
            pred=ds_clim_forecast,
            obs=data_dynamic,
            forecast_type="continuous",
            aggregating_dim="time",
        )
        # - Save spatial skills
        if not os.path.exists(os.path.join(data_dir, "Benchmarks")):
            os.makedirs(os.path.join(data_dir, "Benchmarks"))
        ds_skill.to_netcdf(
            os.path.join(data_dir, "Benchmarks", clim_name + "_Spatial_Skills.nc")
        )
        # - Compute deterministic global skills
        ds_skill = ds_skill.sphere.add_SphericalVoronoiMesh(x="lon", y="lat")
        ds_global_skill = xverif.global_summary(ds_skill, area_coords="area")
        ds_global_skill.to_netcdf(
            os.path.join(data_dir, "Benchmarks", clim_name + "_Global_Skills.nc")
        )

    ##------------------------------------------------------------------------.
    #### Compute persistence forecast skils
    print(
        "- Computing persistence forecasts skills for {} sampling".format(sampling_name)
    )
    # - Rechunk dataset
    data_dynamic = data_dynamic.chunk({"time": -1, "node": 1})
    # - Define leadtimes
    forecast_cycle = 6
    AR_iterations = 40
    leadtimes = np.arange(1, AR_iterations) * np.timedelta64(forecast_cycle, "h")
    # - Compute persistence forecast at each leadtime
    list_skills = []
    for leadtime in leadtimes:
        t_i = time.time()
        print("   - Leadtime {}".format(leadtime))
        lagged_ds = data_dynamic.copy()
        lagged_ds["time"] = lagged_ds["time"] + leadtime
        ds_skill = xverif.deterministic(
            pred=lagged_ds,
            obs=data_dynamic,
            forecast_type="continuous",
            aggregating_dim="time",
        )
        ds_skill = ds_skill.assign_coords({"leadtime": np.array(leadtime)})
        ds_skill = ds_skill.expand_dims("leadtime")
        list_skills.append(ds_skill)
    print("   - Saving persistence skill to disk")
    # - Combine peristence forecast skill at all leadtimes
    ds_persistence_skill = xr.merge(list_skills)
    ds_persistence_skill.to_netcdf(
        os.path.join(data_dir, "Benchmarks", "Persistence_Spatial_Skills.nc")
    )
    # - Compute peristence forecast deterministic global skills
    ds_persistence_skill = ds_persistence_skill.sphere.add_SphericalVoronoiMesh(
        x="lon", y="lat"
    )
    ds_global_skill = xverif.global_summary(ds_persistence_skill, area_coords="area")
    ds_global_skill.to_netcdf(
        os.path.join(data_dir, "Benchmarks", "Persistence_Global_Skills.nc")
    )
    ##------------------------------------------------------------------------.


##----------------------------------------------------------------------------.

# -----------------------------------------------------------------------------.
if __name__ == "__main__":
    ##------------------------------------------------------------------------.
    ### Set dask configs
    # - By default, Xarray and dask.array use thee multi-threaded scheduler (dask.config.set(scheduler='threads')
    # - 'num_workers' defaults to the number of cores
    # - dask.config.set(scheduler='threads') # Uses a ThreadPoolExecutor in the local process
    # - dask.config.set(scheduler='processes') # Uses a ProcessPoolExecutor to spread work between processes
    from dask.distributed import Client

    client = Client(processes=False)
    # - Set array.chunk-size default
    dask.config.set({"array.chunk-size": "1024 MiB"})
    # - Avoid to split large dask chunks
    dask.config.set(**{"array.slicing.split_large_chunks": False})

    ##------------------------------------------------------------------------.
    ### Define settings
    # - Define data directory
    base_data_dir = "/ltenas3/data/DeepSphere/data/preprocessed/ERA5_HRES/"
    # base_data_dir = "/ltenas3/DeepSphere/data/preprocessed/ERA5_HRES/"

    # - Define samplings
    sampling_name_list = [  # 'Cubed_400km',
        "O24",
        "Healpix_400km",
        "Equiangular_400km",
        "Equiangular_400km_tropics",
        "Icosahedral_400km",
        # 'Healpix_100km'
    ]

    # - Define test period
    test_period = np.array(["2017-01-01T00:00", "2018-12-31T23:00"], dtype="M8[m]")

    # - Define climatology forecasts to analyze
    list_climatologies = ["MonthlyClimatology", "WeeklyClimatology"]
    ##------------------------------------------------------------------------.
    # Launch computations
    for sampling_name in sampling_name_list:
        print("==================================================================")
        print("Computing verification benchmarks for", sampling_name, "data")
        t_i = time.time()
        # ----------------------------------------------------------------------.
        compute_verif(
            sampling_name=sampling_name,
            base_data_dir=base_data_dir,
            test_period=test_period,
            list_climatologies=list_climatologies,
        )
        # ----------------------------------------------------------------------.
        # Report elapsed time
        print("---> Elapsed time: {:.1f} minutes ".format((time.time() - t_i) / 60))
        print("==================================================================")
