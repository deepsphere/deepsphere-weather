#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 13:12:02 2021

@author: ghiggi
"""
import os
os.chdir("/home/ghiggi/Projects/deepsphere-weather")
import dask
import matplotlib
import xarray as xr

import xsphere  # required for xarray 'sphere' accessor 
from xscaler import LoadScaler, SequentialScaler, LoadAnomaly
from xforecasting.utils.io import get_ar_model_tensor_info
from xforecasting import AutoregressivePredictions

## DeepSphere-Weather
import modules.my_models_graph_old as my_architectures
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
from modules.my_plotting import create_hovmoller_plots

# For plotting 
# matplotlib.use('cairo') # Cairo
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["savefig.facecolor"] = "white" # (1,1,1,0)
matplotlib.rcParams["savefig.edgecolor"] = 'none'

##------------------------------------------------------------------------------.
# Define settings 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_dir = "/ltenas3/DeepSphere/data/preprocessed_ds/ERA5_HRES"
exp_dir = "/data/weather_prediction/experiments_GG/new_old_archi"
model_name = "OLD_fine_tuned-RNN-AR6-UNetSpherical-Healpix_400km-Graph_knn-k20-MaxAreaPooling"
model_dir = os.path.join(exp_dir, model_name)

batch_size = 4
n_year_sims = 2
ar_blocks_days = 366*2   
forecast_reference_times = ['1992-07-22T00:00:00','2000-01-01T18:00:00','2003-04-01T00:00:00', '2015-01-01T05:00:00']
long_forecast_zarr_fpath = None
long_forecast_zarr_fpath = os.path.join(model_dir, "model_predictions", "long_simulation", "2year_sim.zarr")

# Read config path 
cfg_path = os.path.join(model_dir, 'config.json')
cfg = read_config_file(fpath=cfg_path)

data_sampling_dir = os.path.join(data_dir, cfg['model_settings']["sampling_name"])

##-----------------------------------------------------------------------------.
# Some special stuff you might want to adjust 
cfg['dataloader_settings']["prefetch_factor"] = 2      
cfg['dataloader_settings']["num_workers"] = 8
cfg['dataloader_settings']["autotune_num_workers"] = False
cfg['training_settings']['gpu_training'] = True
cfg['dataloader_settings']["pin_memory"] = False
cfg['dataloader_settings']["asyncronous_gpu_transfer"] = True

##-----------------------------------------------------------------------------.
# Define model weights fpath 
model_fpath = os.path.join(model_dir, "model_weights", "model.h5")

##------------------------------------------------------------------------.
### Retrieve experiment-specific configuration settings   
model_settings = get_model_settings(cfg)   
ar_settings = get_ar_settings(cfg)
training_settings = get_training_settings(cfg) 
dataloader_settings = get_dataloader_settings(cfg) 

##------------------------------------------------------------------------.
#### Load Zarr Datasets
ds_dynamic = xr.open_zarr(os.path.join(data_sampling_dir, "Data","dynamic", "time_chunked", "dynamic.zarr")) 
ds_bc = xr.open_zarr(os.path.join(data_sampling_dir, "Data","bc", "time_chunked", "bc.zarr")) 
ds_static = xr.open_zarr(os.path.join(data_sampling_dir, "Data", "static.zarr")) 

# - Select dynamic features 
ds_dynamic = ds_dynamic[['z500','t850']]    

##------------------------------------------------------------------------.
### Prepare static data 
# - Keep land-surface mask as it is 

# - Keep sin of latitude and remove longitude information 
ds_static = ds_static.drop(["sin_longitude","cos_longitude"])

# - Scale orography between 0 and 1 (is already left 0 bounded)
ds_static['orog'] = ds_static['orog']/ds_static['orog'].max()

# - One Hot Encode soil type 
# ds_slt_OHE = xscaler.OneHotEnconding(data_static['slt'])
# ds_static = xr.merge([ds_static, ds_slt_OHE])
# ds_static = ds_static.drop('slt')

# - Load static data 
ds_static = ds_static.load()

##------------------------------------------------------------------------.
#### Define scaler to apply on the fly within DataLoader 
# - Load scalers
dynamic_scaler = LoadScaler(os.path.join(data_sampling_dir, "Scalers", "GlobalStandardScaler_dynamic.nc"))
bc_scaler = LoadScaler(os.path.join(data_sampling_dir, "Scalers", "GlobalStandardScaler_bc.nc"))
# - Create single scaler 
scaler = SequentialScaler(dynamic_scaler, bc_scaler)

##------------------------------------------------------------------------.
### Define pyTorch settings (before PyTorch model definition)
# - Here inside is eventually set the seed for fixing model weights initialization
# - Here inside the training precision is set (currently only float32 works)
device = set_pytorch_settings(training_settings)

##------------------------------------------------------------------------.
## Retrieve dimension info of input-output Torch Tensors
tensor_info = get_ar_model_tensor_info(ar_settings = ar_settings,
                                       data_dynamic = ds_dynamic, 
                                       data_static = ds_static, 
                                       data_bc = ds_bc)
print_tensor_info(tensor_info)         

# Check that tensor_info match between model training and now 
check_same_dict(model_settings['tensor_info'], tensor_info)

##------------------------------------------------------------------------.
### Define the model architecture  
model = get_pytorch_model(module = my_architectures,
                            model_settings = model_settings)            

###-----------------------------------------------------------------------.
## Load a pre-trained model  
load_pretrained_model(model = model, 
                        model_dir = model_dir)

###-----------------------------------------------------------------------.
### Transfer model to the device (i.e. GPU)
model = model.to(device)

##------------------------------------------------------------------------.
# Run predictions  
dask.config.set(scheduler='synchronous')
forecast_cycle = ar_settings['forecast_cycle']
ar_iterations = 24/forecast_cycle*365*n_year_sims
ar_blocks = None  
ds_long_forecasts = AutoregressivePredictions( model = model, 
                                        # Data
                                        data_dynamic = ds_dynamic,
                                        data_static = ds_static,              
                                        data_bc = ds_bc, 
                                        scaler_transform = scaler,
                                        scaler_inverse = scaler,
                                        # Dataloader options
                                        device = device,
                                        batch_size = batch_size,  # number of forecasts per batch
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
                                        ar_blocks = ar_blocks,
                                        ar_iterations = ar_iterations,  # How many time to autoregressive iterate
                                        # Save options 
                                        zarr_fpath = long_forecast_zarr_fpath, # None --> do not write to disk
                                        rounding = 2,             # Default None. Accept also a dictionary 
                                        compressor = "auto",      # Accept also a dictionary per variable
                                        chunks = "auto")


print(ds_long_forecasts)

##------------------------------------------------------------------------------.
##------------------------------------------------------------------------------.
################################
#### Create hovmoller plots ####
################################
# data_sampling_dir = os.path.join(data_dir, cfg['model_settings']["sampling_name"])
# ds_dynamic = xr.open_zarr(os.path.join(data_sampling_dir, "Data","dynamic", "time_chunked", "dynamic.zarr"))  
# ds_long_forecasts = xr.open_zarr(long_forecast_zarr_fpath)

#-------------------------------------------------------------------------------
# - Load anomaly scalers
monthly_std_anomaly_scaler = LoadAnomaly(os.path.join(data_sampling_dir, "Scalers", "MonthlyStdAnomalyScaler_dynamic.nc"))
# - Create directory where to save figures
# os.makedirs(os.path.join(model_dir, "figs/hovmoller_plots"), exist_ok=True)
# - Create figures 
for i in range(len(forecast_reference_times)):
    # Select 1 forecast 
    ds_forecast = ds_long_forecasts.isel(forecast_reference_time=i)
    # Plot variable 'State' Hovmoller 
    fig = create_hovmoller_plots(ds_obs = ds_dynamic, 
                                 ds_pred = ds_forecast, 
                                 scaler = None,
                                 arg = "state",
                                 time_groups = None)
    fig.savefig(os.path.join(model_dir, "figs/hovmoller_plots", "state_sim" + '{:01}.png'.format(i)))
    # Plot variable 'standard anomalies' Hovmoller 
    fig = create_hovmoller_plots(ds_obs = ds_dynamic, 
                                 ds_pred = ds_forecast, 
                                 scaler = monthly_std_anomaly_scaler,
                                 arg = "anom",
                                 time_groups = None)
    fig.savefig(os.path.join(model_dir, "figs/hovmoller_plots", "anom_sim" + '{:01}.png'.format(i)))
    
#-------------------------------------------------------------------------------
# # Select 1 forecast 
# ds_forecast = ds_forecasts.isel(forecast_reference_time=1)
# # Plot 
# fig = create_hovmoller_plots(ds_obs = ds_dynamic, 
#                              ds_pred = ds_forecast, 
#                              scaler=None,
#                              arg="state",
#                              time_groups=None)
# plt.show()

# fig = create_hovmoller_plots(ds_obs = ds_dynamic, 
#                              ds_pred = ds_forecast, 
#                              scaler=anomaly_scaler,
#                              arg="anom",
#                              time_groups=None)
# plt.show()
                
# Save figure 
# fig.savefig("/home/Projects/deepsphere-weather/figs/hovmoller_long_sim.png")

