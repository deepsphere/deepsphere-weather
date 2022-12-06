import os

os.chdir("/home/ghiggi/Projects/deepsphere-weather")
import sys

sys.path.append("../")
import warnings
import time
import dask
import argparse

# import torch
# import pickle
# import pygsp as pg
# import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sampling_name = "Cubed_400km"
old_data_dir = "/data/weather_prediction/data"
new_data_dir = "/ltenas3/DeepSphere/data/preprocessed_ds/ERA5_HRES"

new_data_sampling_dir = os.path.join(new_data_dir, sampling_name)
old_data_sampling_dir = os.path.join(old_data_dir, sampling_name)

new_data_dynamic = xr.open_zarr(
    os.path.join(
        new_data_sampling_dir, "Data", "dynamic", "time_chunked", "dynamic.zarr"
    )
)[["z500", "t850"]]
new_data_bc = xr.open_zarr(
    os.path.join(new_data_sampling_dir, "Data", "bc", "time_chunked", "bc.zarr")
)

old_data_dynamic_z500 = xr.open_dataset(
    os.path.join(
        old_data_sampling_dir,
        "data",
        "geopotential_500",
        "geopotential_500_5.625deg.nc",
    )
)
old_data_dynamic_t850 = xr.open_dataset(
    os.path.join(
        old_data_sampling_dir, "data", "temperature_850", "temperature_850_5.625deg.nc"
    )
)
old_data_bc = xr.open_dataset(
    os.path.join(
        old_data_sampling_dir,
        "data",
        "toa_incident_solar_radiation",
        "toa_incident_solar_radiation_5.625deg.nc",
    )
)

## Compare dynamic data
old_data_dynamic_z500 = old_data_dynamic_z500.sel(time=slice("1980-01-01T07:00", None))
new_data_dynamic_z500 = new_data_dynamic["z500"]

old_data_dynamic_t850 = old_data_dynamic_t850.sel(time=slice("1980-01-01T07:00", None))
new_data_dynamic_t850 = new_data_dynamic["t850"]

old_x = old_data_dynamic_z500.isel(time=slice(0, 200))["z"].values
old_y = new_data_dynamic_z500.isel(time=slice(0, 200)).values
old_x - old_y
np.allclose(old_x, old_y, rtol=0.0001)

old_x = old_data_dynamic_t850.isel(time=slice(0, 200))["t"].values
old_y = new_data_dynamic_t850.isel(time=slice(0, 200)).values
old_x - old_y
np.allclose(old_x, old_y, rtol=0.0001)

old_x = old_data_bc["tisr"].isel(time=slice(0, 200)).values
old_y = new_data_bc["tisr"].isel(time=slice(0, 200)).values
old_x - old_y
np.allclose(old_x, old_y, rtol=0.01)


### Compare climatologies
clim_name = "MonthlyClimatology_dynamic.nc"
clim_name = "WeeklyClimatology_dynamic.nc"
old_clim = xr.open_dataset(
    os.path.join(old_data_sampling_dir, "Climatology", clim_name)
)
new_clim = xr.open_dataset(
    os.path.join(new_data_sampling_dir, "Climatology", clim_name)
)

old_x = old_clim["Mean"].values
old_y = new_clim["Mean"].sel(variable=["z500", "t850"]).values
old_x - old_y
np.allclose(old_x, old_y, rtol=0.01)

### Compare scalers
scaler_name = "MonthlyStdAnomalyScaler_dynamic.nc"
scaler_name = "GlobalStandardScaler_dynamic.nc"

old_scaler = xr.open_dataset(
    os.path.join(old_data_sampling_dir, "Scalers", scaler_name)
)
new_scaler = xr.open_dataset(
    os.path.join(new_data_sampling_dir, "Scalers", scaler_name)
)

old_x = old_scaler["mean_"].values
old_y = new_scaler["mean_"].sel(variable=["z500", "t850"]).values
old_x - old_y
np.allclose(old_x, old_y, rtol=0.001)

scaler_name = "GlobalMinMaxScaler_dynamic.nc"
old_scaler = xr.open_dataset(
    os.path.join(old_data_sampling_dir, "Scalers", scaler_name)
)
new_scaler = xr.open_dataset(
    os.path.join(new_data_sampling_dir, "Scalers", scaler_name)
)
old_x = old_scaler["max_"].values
old_y = new_scaler["max_"].sel(variable=["z500", "t850"]).values
old_x - old_y
np.allclose(old_x, old_y, rtol=0.001)

scaler_name = "GlobalMinMaxScaler_bc.nc"
old_scaler = xr.open_dataset(
    os.path.join(old_data_sampling_dir, "Scalers", scaler_name)
)
new_scaler = xr.open_dataset(
    os.path.join(new_data_sampling_dir, "Scalers", scaler_name)
)
old_x = old_scaler["max_"].values
old_y = new_scaler["max_"].values
old_x - old_y
np.allclose(old_x, old_y, rtol=0.0001)

scaler_name = "GlobalStandardScaler_bc.nc"
old_scaler = xr.open_dataset(
    os.path.join(old_data_sampling_dir, "Scalers", scaler_name)
)
new_scaler = xr.open_dataset(
    os.path.join(new_data_sampling_dir, "Scalers", scaler_name)
)
old_x = old_scaler["std_"].values
old_y = new_scaler["std_"].values
old_x - old_y
np.allclose(old_x, old_y, rtol=0.001)

scaler_name = "MonthlyStdAnomalyScaler_bc.nc"
old_scaler = xr.open_dataset(
    os.path.join(old_data_sampling_dir, "Scalers", scaler_name)
)
new_scaler = xr.open_dataset(
    os.path.join(new_data_sampling_dir, "Scalers", scaler_name)
)
old_x = old_scaler["mean_"].values
old_y = new_scaler["mean_"].values
old_x - old_y
np.allclose(old_x, old_y, rtol=0.001)
