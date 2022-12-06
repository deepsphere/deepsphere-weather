#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 13:12:02 2021

@author: ghiggi
"""
import os

# os.chdir("/home/ghiggi/Projects/deepsphere-weather")
import sys

sys.path.append("../")

import xarray as xr
import modules.xsphere  # required for xarray 'sphere' accessor
from xscaler import LoadScaler, SequentialScaler, LoadAnomaly, HovmollerDiagram
from xforecasting.utils.xr import xr_common_vars

from modules.my_plotting import get_var_clim
from modules.my_plotting import get_var_cmap
from modules.my_plotting import create_gif_forecast_evolution

# scp -r ghiggi@ltesrv7.epfl.ch:/data/weather_prediction/experiments_GG/new_old_archi/gifs /home/ghiggi/

##------------------------------------------------------------------------------.
# Define dir and fpaths
data_dir = "/ltenas3/DeepSphere/data/preprocessed_ds/ERA5_HRES"
exp_dir = "/data/weather_prediction/experiments_GG/new_old_archi"
model_name = (
    "OLD_fine_tuned-RNN-AR6-UNetSpherical-Healpix_400km-Graph_knn-k20-MaxAreaPooling"
)
sampling_name = "Healpix_400km"
model_dir = os.path.join(exp_dir, model_name)
data_sampling_dir = os.path.join(data_dir, sampling_name)
long_forecast_zarr_fpath = os.path.join(
    model_dir, "model_predictions", "long_simulation", "2year_sim.zarr"
)
# long_forecast_zarr_fpath = os.path.join(model_dir, "model_predictions", "forecast_chunked", "test_forecasts.zarr")

##-----------------------------------------------------------------------------.
### Open observation and forecast data
ds_forecasts = xr.open_zarr(long_forecast_zarr_fpath)
ds_dynamic = xr.open_zarr(
    os.path.join(data_sampling_dir, "Data", "dynamic", "time_chunked", "dynamic.zarr")
)

# Add mesh
ds_forecasts = ds_forecasts.sphere.add_SphericalVoronoiMesh(x="lon", y="lat")
ds_dynamic = ds_dynamic.sphere.add_SphericalVoronoiMesh(x="lon", y="lat")

# Ensure forecast and obs have same variables
variables = xr_common_vars(ds_forecasts, ds_dynamic)
ds_dynamic = ds_dynamic[variables]
ds_forecasts = ds_forecasts[variables]

# Load anomaly scalers
monthly_std_anomaly_scaler = LoadAnomaly(
    os.path.join(data_sampling_dir, "Scalers", "MonthlyStdAnomalyScaler_dynamic.nc")
)

# Animation:  left side: map over time,  right side: howmoller diagram

# Select forecast
# ds_forecast = ds_forecasts.sel(forecast_reference_time=forecast_reference_time)
ds_forecast = ds_forecasts.isel(forecast_reference_time=0)
ds_forecast["time"] = (
    ds_forecast["forecast_reference_time"].values + ds_forecast["leadtime"]
)
ds_forecast = ds_forecast.set_coords("time").swap_dims({"leadtime": "time"})

# Align dimensions and load data
ds_pred, ds_obs = xr.align(ds_forecast, ds_dynamic)
ds_pred = ds_pred.compute()
ds_obs = ds_obs.compute()

# Compute anomalies
ds_obs_anom = monthly_std_anomaly_scaler.transform(ds_obs)
ds_pred_anom = monthly_std_anomaly_scaler.transform(ds_pred)

# Define gif directory
gif_dir = "/data/weather_prediction/experiments_GG/new_old_archi/gifs"

# Define variables to plot
variables = ["z500", "t850"]
fps = 30
var = "t850"

import time

t_i = time.time()

##----------------------------------------------------------------------------.
### Some thoughts
# -  With multiprocessing it probably takes more time to copy the data each time to
#    each process, rather than computing the figure synchronously in the main process
# - With threads, the GIL prevent gaining performance right?
# - Some notes
# - https://discourse.pangeo.io/t/using-dask-to-parallelize-plotting/1179/13
# - Try pathos multiprocessing
# - Create the figure structure in parallel
# - https://stackoverflow.com/questions/15525431/how-to-fix-the-python-multiprocessing-matplotlib-savefig-issue
# - Save in parallel
# 50 workers - 100 img: 128 s , 138s (when passing the object)
# synchronous - 100 img 60 s
# 25 threads - 100 img: 123 s
# for loop - 100 img: 64 s

# mpl.use("agg") # non-interactive backend which save to disk
# mpl.rcParams['figure.max_open_warning'] = 60

##----------------------------------------------------------------------------.
### Plot pred vs obs  state
for var in variables:
    dict_data = {
        "Observed "
        + var: {
            "data": ds_obs[var],
            "clim": get_var_clim(var=var, arg="state"),
            "cmap": get_var_cmap(var=var, arg="state"),
        },
        "Predicted "
        + var: {
            "data": ds_pred[var],
            "clim": get_var_clim(var=var, arg="state"),
            "cmap": get_var_cmap(var=var, arg="state"),
        },
    }
    gif_fpath = os.path.join(gif_dir, var + "_state_obs_vs_pred.gif")
    create_gif_forecast_evolution(
        gif_fpath=gif_fpath,
        dict_data=dict_data,
        # Plot options
        antialiased=False,
        edgecolors=None,
        # GIF options
        fps=fps,
        create_gif=False,
    )

##----------------------------------------------------------------------------.
### Plot pred vs obs anom
for var in variables:
    dict_data = {
        "Observed "
        + var
        + " std. anomalies": {
            "data": ds_obs_anom[var],
            "clim": (-3, 3),
            "cmap": get_var_cmap(var=var, arg="anom"),
        },
        "Predicted "
        + var
        + "std. anomalies": {
            "data": ds_pred_anom[var],
            "clim": (-3, 3),
            "cmap": get_var_cmap(var=var, arg="anom"),
        },
    }
    gif_fpath = os.path.join(gif_dir, var + "_anom_obs_vs_pred.gif")
    create_gif_forecast_evolution(
        gif_fpath=gif_fpath,
        dict_data=dict_data,
        # Plot options
        antialiased=False,
        edgecolors=None,
        # GIF options
        fps=fps,
        create_gif=False,
    )

##----------------------------------------------------------------------------.
### Plot all variables of obs and pred
# Obs state
dict_data = {}
for var in variables:
    dict_data[var] = {
        "data": ds_obs[var],
        "clim": get_var_clim(var=var, arg="state"),
        "cmap": get_var_cmap(var=var, arg="state"),
    }

gif_fpath = os.path.join(gif_dir, "obs_state.gif")
create_gif_forecast_evolution(
    gif_fpath=gif_fpath,
    dict_data=dict_data,
    # Plot options
    antialiased=False,
    edgecolors=None,
    # GIF options
    fps=fps,
    create_gif=False,
)

##------------------------------------------
# Pred state
dict_data = {}
for var in variables:
    dict_data[var] = {
        "data": ds_pred[var],
        "clim": get_var_clim(var=var, arg="state"),
        "cmap": get_var_cmap(var=var, arg="state"),
    }

gif_fpath = os.path.join(gif_dir, "pred_state.gif")
create_gif_forecast_evolution(
    gif_fpath=gif_fpath,
    dict_data=dict_data,
    # Plot options
    antialiased=False,
    edgecolors=None,
    # GIF options
    fps=fps,
    create_gif=False,
)

##------------------------------------------
# Obs anom
dict_data = {}
for var in variables:
    dict_data[var] = {
        "data": ds_obs_anom[var],
        "clim": (-3, 3),
        "cmap": get_var_cmap(var=var, arg="anom"),
    }


gif_fpath = os.path.join(gif_dir, "obs_anom.gif")
create_gif_forecast_evolution(
    gif_fpath=gif_fpath,
    dict_data=dict_data,
    # Plot options
    antialiased=False,
    edgecolors=None,
    # GIF options
    fps=fps,
    create_gif=False,
)

##------------------------------------------
# Pred anom
dict_data = {}
for var in variables:
    dict_data[var] = {
        "data": ds_pred_anom[var],
        "clim": (-3, 3),
        "cmap": get_var_cmap(var=var, arg="anom"),
    }


gif_fpath = os.path.join(gif_dir, "pred_anom.gif")
create_gif_forecast_evolution(
    gif_fpath=gif_fpath,
    dict_data=dict_data,
    # Plot options
    antialiased=False,
    edgecolors=None,
    # GIF options
    fps=fps,
    create_gif=False,
)

##----------------------------------------------------------------------------.
### For each var, plot state and anom in same plot
# Obs
for var in variables:
    dict_data = {
        "Observed "
        + var: {
            "data": ds_obs[var],
            "clim": get_var_clim(var=var, arg="state"),
            "cmap": get_var_cmap(var=var, arg="state"),
        },
        "Observed "
        + var
        + " std. anomalies": {
            "data": ds_obs_anom[var],
            "clim": (-3, 3),
            "cmap": get_var_cmap(var=var, arg="anom"),
        },
    }
    gif_fpath = os.path.join(gif_dir, var + "_obs_state_vs_anom.gif")
    create_gif_forecast_evolution(
        gif_fpath=gif_fpath,
        dict_data=dict_data,
        # Plot options
        antialiased=False,
        edgecolors=None,
        # GIF options
        fps=fps,
        create_gif=False,
    )

##------------------------------------------
# Pred
for var in variables:
    dict_data = {
        "Predicted "
        + var: {
            "data": ds_pred[var],
            "clim": get_var_clim(var=var, arg="state"),
            "cmap": get_var_cmap(var=var, arg="state"),
        },
        "Predicted "
        + var
        + " std. anomalies": {
            "data": ds_pred_anom[var],
            "clim": (-3, 3),
            "cmap": get_var_cmap(var=var, arg="anom"),
        },
    }
    gif_fpath = os.path.join(gif_dir, var + "_pred_state_vs_anom.gif")
    create_gif_forecast_evolution(
        gif_fpath=gif_fpath,
        dict_data=dict_data,
        # Plot options
        antialiased=False,
        edgecolors=None,
        # GIF options
        fps=fps,
        create_gif=False,
    )

# ----------------------------------------------------------------------------.

print(time.time() - t_i)
