#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 13:12:02 2021

@author: ghiggi
"""
import os
os.chdir("/home/ghiggi/Projects/deepsphere-weather")
import dask
import numpy as np
import cartopy.crs as ccrs
import xarray as xr
import matplotlib.pyplot as plt

## DeepSphere-Earth
from modules.utils_config import read_config_file
from modules.utils_config import get_model_settings
from modules.utils_config import get_training_settings
from modules.utils_config import get_ar_settings
from modules.utils_config import get_dataloader_settings
# from modules.utils_config import get_pytorch_model
from modules.utils_config import set_pytorch_settings
from modules.utils_config import load_pretrained_model
from modules.utils_models import get_pygsp_graph
from modules.utils_io import get_ar_model_diminfo
from modules.predictions_autoregressive import AutoregressivePredictions

## Project specific functions
from modules.xscaler import LoadScaler
from modules.xscaler import SequentialScaler
from modules.xscaler import LoadAnomaly
import modules.xsphere  # required for xarray 'sphere' accessor 
 

from modules.my_io import readDatasets   
from modules.my_io import reformat_Datasets
import modules.my_models_graph as my_architectures

# For plotting 
import matplotlib
matplotlib.use('cairo') # Cairo
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["savefig.facecolor"] = "white" # (1,1,1,0)
matplotlib.rcParams["savefig.edgecolor"] = 'none'

#-----------------------------------------------------------------------------.
## To cfg add: 
# model_settings['dim_info']
# scaler information?
# numeric precision to model_settings?  
#-----------------------------------------------------------------------------.

model_name = "RNN-UNetSpherical-healpix-16-k20-MaxAreaPooling-float32-AR6-LinearStep_weight_corrected"
# model_name = "RNN-UNetDiffSpherical-healpix-16-k20-MaxAreaPooling-float32-AR6-LinearStep"
# model_name = "Anom-RNN-UNetSpherical-healpix-16-k20-MaxAreaPooling-float32-AR6-LinearStep" 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_dir = "/data/weather_prediction/data"
exp_dir = "/data/weather_prediction/experiments_GG"
model_dir = os.path.join(exp_dir, model_name)
cfg_path = os.path.join(model_dir, 'config.json')

# Read config path 
cfg = read_config_file(fpath=cfg_path)
model_fpath = os.path.join(model_dir, "model_weights", "model.h5")

# Print model settings 
# print_model_description(cfg)

# Some special stuff you might want to adjust 
cfg['dataloader_settings']["prefetch_in_gpu"] = False  
cfg['dataloader_settings']["prefetch_factor"] = 2      
cfg['dataloader_settings']["num_workers"] = 8
cfg['dataloader_settings']["autotune_num_workers"] = False
cfg['dataloader_settings']["pin_memory"] = False
cfg['dataloader_settings']["asyncronous_gpu_transfer"] = True
   
##------------------------------------------------------------------------.
### Retrieve experiment-specific configuration settings   
model_settings = get_model_settings(cfg)   
ar_settings = get_ar_settings(cfg)
training_settings = get_training_settings(cfg) 
dataloader_settings = get_dataloader_settings(cfg) 

##------------------------------------------------------------------------.
#### Load netCDF4 Datasets
data_sampling_dir = os.path.join(data_dir, cfg['model_settings']["sampling_name"])

# - Dynamic data (i.e. pressure and surface levels variables)
ds_dynamic = readDatasets(data_dir=data_sampling_dir, feature_type='dynamic')
# - Boundary conditions data (i.e. TOA)
ds_bc = readDatasets(data_dir=data_sampling_dir, feature_type='bc')
# - Static features
ds_static = readDatasets(data_dir=data_sampling_dir, feature_type='static')

ds_dynamic = ds_dynamic.drop(["level"])
ds_dynamic, ds_bc = xr.align(ds_dynamic, ds_bc)
##-----------------------------------------------------------------------------.
#### Define scaler to apply on the fly within DataLoader 
# - Load scalers
dynamic_scaler = LoadScaler(os.path.join(data_sampling_dir, "Scalers", "GlobalStandardScaler_dynamic.nc"))
bc_scaler = LoadScaler(os.path.join(data_sampling_dir, "Scalers", "GlobalStandardScaler_bc.nc"))
static_scaler = LoadScaler(os.path.join(data_sampling_dir, "Scalers", "GlobalStandardScaler_static.nc"))
# # - Create single scaler 
scaler = SequentialScaler(dynamic_scaler, bc_scaler, static_scaler)

##-----------------------------------------------------------------------------.
dict_DataArrays = reformat_Datasets(ds_training_dynamic = ds_dynamic,
                                         ds_static = ds_static,  
                                         ds_training_bc = ds_bc,            
                                         preload_data_in_CPU = True)

da_static = dict_DataArrays['da_static']
da_dynamic = dict_DataArrays['da_training_dynamic']
da_bc = dict_DataArrays['da_training_bc']

##------------------------------------------------------------------------.
### Define pyTorch settings 
device = set_pytorch_settings(training_settings)

##------------------------------------------------------------------------.
## Retrieve dimension info of input-output Torch Tensors
dim_info = get_ar_model_diminfo(ar_settings=ar_settings,
                                da_dynamic=da_dynamic, 
                                da_static=da_static, 
                                da_bc=da_bc)

model_settings['dim_info'] = dim_info
# print_dim_info(dim_info)  



##------------------------------------------------------------------------.
### Define the model architecture   
# TODO (@Wentao ... )  
# - wrap below in a function --> utils_config ? 
# - model = get_pytorch_model(module, model_settings = model_settings) 
DeepSphereModelClass = getattr(my_architectures, model_settings['architecture_name'])
# - Retrieve required model arguments
model_keys = ['dim_info', 'sampling', 'resolution',
              'knn', 'kernel_size_conv',
              'pool_method', 'kernel_size_pooling']
model_args = {k: model_settings[k] for k in model_keys}
model_args['numeric_precision'] = training_settings['numeric_precision']
# - Define DeepSphere model 
model = DeepSphereModelClass(**model_args)              

###-----------------------------------------------------------------------.
## Load a pre-trained model  
load_pretrained_model(model = model, 
                      exp_dir = exp_dir, 
                      model_name = model_settings['model_name'])
    
###-----------------------------------------------------------------------.
### Transfer model to the device (i.e. GPU)
model = model.to(device)

##------------------------------------------------------------------------.
forecast_zarr_fpath = os.path.join(exp_dir, "1_year_sims/test_pred.zarr")
forecast_reference_times = np.array(['2013-12-31T18:00'], dtype='M8[m]')  
dask.config.set(scheduler='synchronous')
# tmp fix 
pygsp_graph = get_pygsp_graph(sampling = model_settings['sampling'], 
                              resolution = model_settings['resolution'],
                              knn = model_settings['knn'])
da_dynamic = da_dynamic.sphere.add_nodes_from_pygsp(pygsp_graph=pygsp_graph)

n_year_sims = 4
zarr_fpath = forecast_zarr_fpath
ar_blocks = 24/6*30
ar_iterations = 24/6*n_year_sims
batch_size = 1 
num_workers = 8
da_dynamic = da_dynamic
da_static = da_static     
da_bc = da_bc
scaler_transform = scaler
scaler_inverse = scaler
                                         
keep_first_prediction = True
input_k = ar_settings['input_k']
output_k = ar_settings['output_k']
forecast_cycle = ar_settings['forecast_cycle']      
# Dataloader options
device = device
prefetch_factor = dataloader_settings['prefetch_factor'] 
prefetch_in_gpu = dataloader_settings['prefetch_in_gpu']  
pin_memory = dataloader_settings['pin_memory']
asyncronous_gpu_transfer = dataloader_settings['asyncronous_gpu_transfer']
numeric_precision = training_settings['numeric_precision']  # to be read from configs 
# Autoregressive settings                     
stack_most_recent_prediction = ar_settings['stack_most_recent_prediction']    
# Save options 
rounding = 2             # Default None. Accept also a dictionary 
compressor = "auto"      # Accept also a dictionary per variable
chunks = "auto"        

shutil.rmtree(forecast_zarr_fpath)


ds_forecasts = AutoregressivePredictions(model = model, 
                                          # Data
                                          da_dynamic = da_dynamic,
                                          da_static = da_static,              
                                          da_bc = da_bc, 
                                          scaler_transform = scaler,
                                          scaler_inverse = scaler,
                                          # Dataloader options
                                          device = device,
                                          batch_size = 1,  # number of forecasts per batch
                                          num_workers = dataloader_settings['num_workers'], 
                                          prefetch_factor = dataloader_settings['prefetch_factor'], 
                                          prefetch_in_gpu = dataloader_settings['prefetch_in_gpu'],  
                                          pin_memory = dataloader_settings['pin_memory'],
                                          asyncronous_gpu_transfer = dataloader_settings['asyncronous_gpu_transfer'],
                                          # Autoregressive settings
                                          input_k = ar_settings['input_k'], 
                                          output_k = ar_settings['output_k'], 
                                          forecast_cycle = ar_settings['forecast_cycle'],                         
                                          stack_most_recent_prediction = ar_settings['stack_most_recent_prediction'], 
                                          # Prediction options 
                                          forecast_reference_times = forecast_reference_times, 
                                          ar_blocks = 24/6*30,
                                          ar_iterations = 24/6*365,  # How many time to autoregressive iterate
                                          # Save options 
                                          zarr_fpath = forecast_zarr_fpath,  # None --> do not write to disk
                                          rounding = 2,             # Default None. Accept also a dictionary 
                                          compressor = "auto",      # Accept also a dictionary per variable
                                          chunks = "auto")



pygsp_graph = get_pygsp_graph(sampling = model_settings['sampling'], 
                              resolution = model_settings['resolution'],
                              knn = model_settings['knn'])
ds_dynamic = ds_dynamic.sphere.add_nodes_from_pygsp(pygsp_graph=pygsp_graph)

forecast_zarr_fpath = os.path.join(exp_dir, "1_year_sims/test_pred.zarr")

from modules.xscaler import LoadAnomaly
from modules.xscaler import HovmollerDiagram
hourly_weekly_anomaly_scaler = LoadAnomaly(os.path.join(data_sampling_dir, "Scalers", "WeeklyHourlyStdAnomalyScaler_dynamic.nc"))


ds_forecasts = xr.open_zarr(forecast_zarr_fpath)
ds_forecasts = ds_forecasts.isel(forecast_reference_time=0)
ds_forecasts['time'] = ds_forecasts['forecast_reference_time'].values + ds_forecasts['leadtime']
ds_forecasts = ds_forecasts.set_coords('time').swap_dims({"leadtime": "time"}) 

ds_pred, ds_obs = xr.align(ds_forecasts, ds_dynamic)
ds_pred = ds_pred.load()
ds_obs = ds_obs.load()

# ds_obs = hourly_weekly_anomaly_scaler.transform(ds_obs)
# ds_pred = hourly_weekly_anomaly_scaler.transform(ds_pred)

ds_err = ds_pred - ds_obs 

# %matplotlib inline

time_groups = None 
time_groups = {"hour": 1 }      
time_groups = ["dayofyear"]
time_groups = ["weekofyear"]
time_groups = ["month"]
hovmoller_pred = HovmollerDiagram(ds_pred, 
                                  time_dim = "time", 
                                  time_groups = time_groups,
                                  spatial_dim = "lat", bin_width = 5,
                                  time_average_before_binning = True)
hovmoller_obs = HovmollerDiagram(ds_obs, 
                             time_dim = "time", 
                             time_groups = time_groups,
                             spatial_dim = "lat", bin_width = 5,
                             time_average_before_binning = True)

hovmoller_err = HovmollerDiagram(ds_err, 
                                 time_dim = "time", 
                                 time_groups = time_groups,
                                 spatial_dim = "lat", bin_width = 5,
                                 time_average_before_binning = True)

hovmoller_diff = hovmoller_pred - hovmoller_obs
diff_hov = hovmoller_diff - hovmoller_err

fig, ax = plt.subplots()
hovmoller_obs['z500'].plot(ax=ax)
ax.set_ylabel("Latitude")
ax.set_title("Observed")
plt.show()

fig, ax = plt.subplots()
hovmoller_pred['z500'].plot(ax=ax)
ax.set_ylabel("Latitude")
ax.set_title("Predicted")
plt.show()

fig, ax = plt.subplots()
hovmoller_diff['z500'].plot(ax=ax)
ax.set_ylabel("Latitude")
plt.show()

fig, ax = plt.subplots()
hovmoller_err['z500'].plot(ax=ax)
ax.set_ylabel("Latitude")
plt.show()

fig, ax = plt.subplots()
diff_hov['z500'].plot(ax=ax)
ax.set_ylabel("Latitude")
plt.show()

spatial_dim = "lat"
time_dim = "time"
bin_edges=None
bin_width=5
time_groups=time_groups
time_average_before_binning=True
variable_dim=None

# fig, ax = plt.subplots()
# hovmoller_err['z500'].plot(x="lon_bins", y="time", ax=ax)
# ax.set_xlabel("Longitude")
# plt.show()

# Check time_average_before_binning when having >1 year data 
hovmoller_pred1 = HovmollerDiagram(ds, 
                                  time_dim = "time", 
                                  time_groups = time_groups,
                                  spatial_dim = "lat", bin_width = 5,
                                  time_average_before_binning = False)
d = hovmoller_pred - hovmoller_pred1
fig, ax = plt.subplots()
d['z500'].plot(ax=ax)
ax.set_ylabel("Latitude")
plt.show() 
 # plot 
 # contour 
 # conoturf 

## - Hovmoller 
# Diurnal cycle (over 1 week) 
# Annual simulation (original t_res) 
# Annual simulation (daily mean) 

# Animation:  left side: map over time,  right side: howmoller diagram 

hovmoller_pred = HovmollerDiagram(da, 
                                  time_dim = "time", 
                                  time_groups = "month",
                                  spatial_dim = "lat", bin_width = 5,
                                  time_average_before_binning = True) # xarray bug ... 

hovmoller_pred = HovmollerDiagram(da, 
                                  time_dim = "time", 
                                  time_groups = time_groups,
                                  spatial_dim = "lat", bin_width = 5,
                                  time_average_before_binning = True,
                                  variable_dim = "variable")