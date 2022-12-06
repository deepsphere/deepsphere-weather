import os

os.chdir("/home/ghiggi/Projects/deepsphere-weather")
import sys

sys.path.append("../")
import shutil
import argparse
import dask
import glob
import time
import torch
import zarr
import numpy as np
import xarray as xr

## DeepSphere-Weather
from modules.utils_config import read_config_file
from modules.utils_config import get_model_settings
from modules.utils_config import get_training_settings
from modules.utils_config import get_ar_settings
from modules.utils_config import get_dataloader_settings
from modules.utils_config import check_same_dict
from modules.utils_config import get_pytorch_model
from modules.utils_config import set_pytorch_settings
from modules.utils_config import load_pretrained_model
from modules.utils_config import print_tensor_info
from modules.utils_io import get_ar_model_tensor_info
from modules.predictions_autoregressive import AutoregressivePredictions

## Functions within AutoregressivePredictions
from modules.dataloader_autoregressive import remove_unused_Y
from modules.dataloader_autoregressive import get_aligned_ar_batch
from modules.dataloader_autoregressive import AutoregressiveDataset
from modules.dataloader_autoregressive import AutoregressiveDataLoader
from modules.utils_autoregressive import check_ar_settings
from modules.utils_autoregressive import check_input_k
from modules.utils_autoregressive import check_output_k
from modules.utils_io import _get_feature_order
from modules.utils_zarr import check_chunks
from modules.utils_zarr import check_rounding
from modules.utils_zarr import rechunk_Dataset
from modules.utils_zarr import write_zarr
from modules.utils_torch import check_device
from modules.utils_torch import check_pin_memory
from modules.utils_torch import check_asyncronous_gpu_transfer
from modules.utils_torch import check_prefetch_in_gpu
from modules.utils_torch import check_prefetch_factor
from modules.utils_swag import bn_update

## Project specific functions
import modules.my_models_graph as my_architectures

## Side-project utils (maybe migrating to separate packages in future)
from modules.xscaler import LoadScaler
from modules.xscaler import SequentialScaler

# -------------------------------------------------------------------------.

data_dir = "/ltenas3/DeepSphere/data/preprocessed_ds/ERA5_HRES"
model_dir = "/data/weather_prediction/experiments_GG/new/RNN-AR6-UNetSpherical-Healpix_400km-Graph_knn-k20-MaxAreaPooli/"

# -------------------------------------------------------------------------.
# Read config file
cfg_path = os.path.join(model_dir, "config.json")
cfg = read_config_file(fpath=cfg_path)
# Some special options to adjust for prediction
cfg["dataloader_settings"]["autotune_num_workers"] = False
cfg["training_settings"]["gpu_training"] = True  # to run prediction in GPU if possible
##------------------------------------------------------------------------.
### Retrieve experiment-specific configuration settings
model_settings = get_model_settings(cfg)
ar_settings = get_ar_settings(cfg)
training_settings = get_training_settings(cfg)
dataloader_settings = get_dataloader_settings(cfg)
dataloader_settings["num_workers"] = 10

##------------------------------------------------------------------------.
#### Load Zarr Datasets
data_sampling_dir = os.path.join(data_dir, cfg["model_settings"]["sampling_name"])

data_dynamic = xr.open_zarr(
    os.path.join(data_sampling_dir, "Data", "dynamic", "time_chunked", "dynamic.zarr")
)
data_bc = xr.open_zarr(
    os.path.join(data_sampling_dir, "Data", "bc", "time_chunked", "bc.zarr")
)
data_static = xr.open_zarr(os.path.join(data_sampling_dir, "Data", "static.zarr"))

# - Select dynamic features
# data_dynamic = data_dynamic[['z500','t850']]

##------------------------------------------------------------------------.
### Prepare static data
# - Keep land-surface mask as it is

# - Keep sin of latitude and remove longitude information
data_static = data_static.drop(["sin_longitude", "cos_longitude"])

# - Scale orography between 0 and 1 (is already left 0 bounded)
data_static["orog"] = data_static["orog"] / data_static["orog"].max()

# - One Hot Encode soil type
# ds_slt_OHE = xscaler.OneHotEnconding(data_static['slt'])
# data_static = xr.merge([data_static, ds_slt_OHE])
# data_static = data_static.drop('slt')

# - Load static data
data_static = data_static.load()

##------------------------------------------------------------------------.
#### Define scaler to apply on the fly within DataLoader
# - Load scalers
dynamic_scaler = LoadScaler(
    os.path.join(data_sampling_dir, "Scalers", "GlobalStandardScaler_dynamic.nc")
)
bc_scaler = LoadScaler(
    os.path.join(data_sampling_dir, "Scalers", "GlobalStandardScaler_bc.nc")
)
# # - Create single scaler
scaler = SequentialScaler(dynamic_scaler, bc_scaler)

##------------------------------------------------------------------------.
### Define pyTorch settings (before PyTorch model definition)
# - Here inside is eventually set the seed for fixing model weights initialization
# - Here inside the training precision is set (currently only float32 works)
device = set_pytorch_settings(training_settings)

##------------------------------------------------------------------------.
## Retrieve dimension info of input-output Torch Tensors
tensor_info = get_ar_model_tensor_info(
    ar_settings=ar_settings,
    data_dynamic=data_dynamic,
    data_static=data_static,
    data_bc=data_bc,
)
print_tensor_info(tensor_info)

# Check that tensor_info match between model training and now
check_same_dict(model_settings["tensor_info"], tensor_info)

##------------------------------------------------------------------------.
### Define the model architecture
model = get_pytorch_model(module=my_architectures, model_settings=model_settings)

###-----------------------------------------------------------------------.
## Load a pre-trained model
load_pretrained_model(model=model, model_dir=model_dir)

###-----------------------------------------------------------------------.
### Transfer model to the device (i.e. GPU)
model = model.to(device)

###-----------------------------------------------------------------------.
## AutoregressivePredictions arguments
forecast_reference_times = np.datetime64("2016-12-26T23:00:00.000000000")
forecast_reference_times1 = np.datetime64("2016-06-26T23:00:00.000000000")
forecast_reference_times = [forecast_reference_times, forecast_reference_times1]
ar_iterations = 2 * 365 * 4
ar_iterations = 20
batch_size = 32
ar_blocks = None
forecast_zarr_fpath = None
num_workers = 10  # dataloader_settings['num_workers']

bc_generator = None
ar_batch_fun = get_aligned_ar_batch
scaler_transform = scaler
scaler_inverse = scaler
# Dataloader options
device = device
batch_size = batch_size  # number of forecasts per batch

prefetch_factor = dataloader_settings["prefetch_factor"]
prefetch_in_gpu = dataloader_settings["prefetch_in_gpu"]
pin_memory = dataloader_settings["pin_memory"]
asyncronous_gpu_transfer = dataloader_settings["asyncronous_gpu_transfer"]
# Autoregressive settings
input_k = ar_settings["input_k"]
output_k = ar_settings["output_k"]
forecast_cycle = ar_settings["forecast_cycle"]
stack_most_recent_prediction = ar_settings["stack_most_recent_prediction"]
# Prediction options
forecast_reference_times = forecast_reference_times
ar_blocks = ar_blocks
ar_iterations = ar_iterations  # How many time to autoregressive iterate
keep_first_prediction = True
# Save options
zarr_fpath = forecast_zarr_fpath  # None --> do not write to disk
rounding = 2  # Default None. Accept also a dictionary
compressor = "auto"  # Accept also a dictionary per variable
chunks = "auto"

# 1 Valid timestep : OK
forecast_reference_times = np.datetime64("2018-12-26T23:00:00.000000000")

### 2 (valid) timesteps --> OK
forecast_reference_times1 = np.datetime64("2018-12-26T22:00:00.000000000")
forecast_reference_times2 = np.datetime64("2018-12-26T23:00:00.000000000")
forecast_reference_times = [forecast_reference_times2, forecast_reference_times1]

## One valid, one unvalid
forecast_reference_times1 = np.datetime64("2018-12-26T23:00:00.000000000")
forecast_reference_times2 = np.datetime64("2018-12-27T00:00:00.000000000")
forecast_reference_times = [forecast_reference_times2, forecast_reference_times1]

## 1 Unvalid (future) timestep  --> OK: raise correct error
forecast_reference_times = np.datetime64("2018-12-27T00:00:00.000000000")

## 1 Unvalid timestep (past) --> OK: raise correct error
forecast_reference_times = np.datetime64("1980-01-01T07:00:00.000000000")

forecast_reference_times = np.datetime64("1970-01-01T07:00:00.000000000")

## 2 unvalid (future) timesteps  --> OK: raise correct error
forecast_reference_times1 = np.datetime64("2018-12-27T00:00:00.000000000")
forecast_reference_times2 = np.datetime64("2018-12-27T01:00:00.000000000")
forecast_reference_times = [forecast_reference_times2, forecast_reference_times1]

## 2 unvalid (past) timesteps  --> OK: raise correct error
forecast_reference_times1 = np.datetime64("1980-01-01T07:00:00.000000000")
forecast_reference_times2 = np.datetime64("1980-01-01T06:00:00.000000000")
forecast_reference_times = [forecast_reference_times2, forecast_reference_times1]

# ----
### No duplicate (unvalid) timesteps --> OK raise correct error
forecast_reference_times1 = np.datetime64("2018-12-27T00:00:00.000000000")
forecast_reference_times2 = np.datetime64("2018-12-27T00:00:00.000000000")
forecast_reference_times = [forecast_reference_times2, forecast_reference_times1]

### No duplicate (valid) timesteps --> OK raise correct error
forecast_reference_times1 = np.datetime64("2018-12-26T23:00:00.000000000")
forecast_reference_times2 = np.datetime64("2018-12-26T23:00:00.000000000")
forecast_reference_times = [forecast_reference_times2, forecast_reference_times1]

## Empty list --> OK raise correct error
forecast_reference_times = []
# ----

## AutoregressivePredictions arguments
forecast_reference_times = np.datetime64("2016-12-26T23:00:00.000000000")
forecast_reference_times1 = np.datetime64("2016-06-26T23:00:00.000000000")
forecast_reference_times = [forecast_reference_times, forecast_reference_times1]
ar_iterations = 2 * 365 * 4

dask.config.set(scheduler="synchronous")
ds_forecasts = AutoregressivePredictions(
    model=model,
    # Data
    data_dynamic=data_dynamic,
    data_static=data_static,
    data_bc=data_bc,
    scaler_transform=scaler,
    scaler_inverse=scaler,
    # Dataloader options
    device=device,
    batch_size=batch_size,  # number of forecasts per batch
    num_workers=dataloader_settings["num_workers"],
    prefetch_factor=dataloader_settings["prefetch_factor"],
    prefetch_in_gpu=dataloader_settings["prefetch_in_gpu"],
    pin_memory=dataloader_settings["pin_memory"],
    asyncronous_gpu_transfer=dataloader_settings["asyncronous_gpu_transfer"],
    # Autoregressive settings
    input_k=ar_settings["input_k"],
    output_k=ar_settings["output_k"],
    forecast_cycle=ar_settings["forecast_cycle"],
    stack_most_recent_prediction=ar_settings["stack_most_recent_prediction"],
    # Prediction options
    forecast_reference_times=forecast_reference_times,
    ar_blocks=ar_blocks,
    ar_iterations=ar_iterations,  # How many time to autoregressive iterate
    # Save options
    zarr_fpath=forecast_zarr_fpath,  # None --> do not write to disk
    rounding=2,  # Default None. Accept also a dictionary
    compressor="auto",  # Accept also a dictionary per variable
    chunks="auto",
)
print(ds_forecasts)

ds_forecasts.to_zarr("/ltenas3/DeepSphere/tmp/2ysim.zarr")


###-----------------------------------------------------------------------.
## DEBUG Code within AutoregressivePredictions
##------------------------------------------------------------------------.
## Checks arguments
device = check_device(device)
pin_memory = check_pin_memory(
    pin_memory=pin_memory, num_workers=num_workers, device=device
)
asyncronous_gpu_transfer = check_asyncronous_gpu_transfer(
    asyncronous_gpu_transfer=asyncronous_gpu_transfer, device=device
)
prefetch_in_gpu = check_prefetch_in_gpu(
    prefetch_in_gpu=prefetch_in_gpu, num_workers=num_workers, device=device
)
prefetch_factor = check_prefetch_factor(
    prefetch_factor=prefetch_factor, num_workers=num_workers
)
##------------------------------------------------------------------------.
# Check that autoregressive settings are valid
# - input_k and output_k must be numpy arrays hereafter !
input_k = check_input_k(input_k=input_k, ar_iterations=ar_iterations)
output_k = check_output_k(output_k=output_k)
check_ar_settings(
    input_k=input_k,
    output_k=output_k,
    forecast_cycle=forecast_cycle,
    ar_iterations=ar_iterations,
    stack_most_recent_prediction=stack_most_recent_prediction,
)
ar_iterations = int(ar_iterations)
##------------------------------------------------------------------------.
### Retrieve feature info of the forecast
features = _get_feature_order(data_dynamic)

##------------------------------------------------------------------------.
# Check Zarr settings
WRITE_TO_ZARR = zarr_fpath is not None
if WRITE_TO_ZARR:
    # - If zarr fpath provided, create the required folder
    if not os.path.exists(os.path.dirname(zarr_fpath)):
        os.makedirs(os.path.dirname(zarr_fpath))
    # - Set default chunks and compressors
    # ---> -1 to all optional dimensions (i..e nodes, lat, lon, ens, plevels,...)
    dims = list(data_dynamic.dims)
    dims_optional = np.array(dims)[
        np.isin(dims, ["time", "feature"], invert=True)
    ].tolist()
    default_chunks = {dim: -1 for dim in dims_optional}
    default_chunks["forecast_reference_time"] = 1
    default_chunks["leadtime"] = 1
    default_compressor = zarr.Blosc(cname="zstd", clevel=0, shuffle=2)
    # - Check rounding settings
    rounding = check_rounding(rounding=rounding, variable_names=features)
##------------------------------------------------------------------------.
# Check ar_blocks
if not isinstance(ar_blocks, (int, float, type(None))):
    raise TypeError("'ar_blocks' must be int or None.")
if isinstance(ar_blocks, float):
    ar_blocks = int(ar_blocks)
if not WRITE_TO_ZARR and isinstance(ar_blocks, int):
    raise ValueError("If 'zarr_fpath' not specified, 'ar_blocks' must be None.")
if ar_blocks is None:
    ar_blocks = ar_iterations + 1
if ar_blocks > ar_iterations + 1:
    raise ValueError("'ar_blocks' must be equal or smaller to 'ar_iterations'")
PREDICT_ar_BLOCKS = ar_blocks != (ar_iterations + 1)

##------------------------------------------------------------------------.
### Define DataLoader subset_timesteps
forecast_reference_times = check_timesteps_format(forecast_reference_times)
check_no_duplicate_timesteps(
    forecast_reference_times, var_name="forecast_reference_times"
)
forecast_reference_times.sort()  # ensure the temporal order
subset_timesteps = None
if forecast_reference_times is not None:
    if len(forecast_reference_times) == 0:
        raise ValueError(
            "If you don't want to specify specific 'forecast_reference_times', set it to None"
        )
    t_res_timedelta = np.diff(data_dynamic.time.values)[0]
    subset_timesteps = forecast_reference_times + -1 * max(input_k) * t_res_timedelta

##------------------------------------------------------------------------.
### Create training Autoregressive Dataset and DataLoader
dataset = AutoregressiveDataset(
    data_dynamic=data_dynamic,
    data_bc=data_bc,
    data_static=data_static,
    bc_generator=bc_generator,
    scaler=scaler_transform,
    # Dataset options
    subset_timesteps=subset_timesteps,
    training_mode=False,
    # Autoregressive settings
    input_k=input_k,
    output_k=output_k,
    forecast_cycle=forecast_cycle,
    ar_iterations=ar_iterations,
    stack_most_recent_prediction=stack_most_recent_prediction,
    # GPU settings
    device=device,
)

dataset[0]
self = dataset
self.subset_timesteps
self.idxs
len(self)
