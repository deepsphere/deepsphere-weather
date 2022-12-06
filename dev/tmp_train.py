import os
# os.chdir('/home/ghiggi/Projects/deepsphere-weather')
import sys
sys.path.append('../')
import shutil
import argparse
import dask
import numpy as np
import xarray as xr

import os
# os.chdir('/home/ghiggi/Projects/deepsphere-weather')
import sys
sys.path.append('../')
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
import cartopy.crs as ccrs

from torch import optim
from torchinfo import summary

## DeepSphere-Weather modules
from modules.utils_config import read_config_file
from modules.utils_config import write_config_file
from modules.utils_config import get_model_settings
from modules.utils_config import get_training_settings
from modules.utils_config import get_ar_settings
from modules.utils_config import get_dataloader_settings
from modules.utils_config import get_pytorch_model
from modules.utils_config import get_model_name
from modules.utils_config import set_pytorch_settings
from modules.utils_config import load_pretrained_model
from modules.utils_config import create_experiment_directories
from modules.utils_config import print_model_description
from modules.utils_config import print_tensor_info

from modules.utils_models import get_pygsp_graph
from modules.utils_io import get_ar_model_tensor_info
from modules.training_autoregressive import AutoregressiveTraining
from modules.predictions_autoregressive import AutoregressivePredictions
from modules.predictions_autoregressive import rechunk_forecasts_for_verification
from modules.utils_torch import summarize_model
from modules.AR_Scheduler import AR_Scheduler
from modules.early_stopping import EarlyStopping
from modules.loss import WeightedMSELoss, AreaWeights

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

## Project specific functions
import modules.my_models_graph as my_architectures

## Side-project utils (maybe migrating to separate packages in future)
import modules.xsphere  # required for xarray 'sphere' accessor 
import modules.xverif as xverif

# import modules.xscaler as xscaler  
from modules.xscaler import LoadScaler
from modules.xscaler import SequentialScaler
from modules.xscaler import LoadAnomaly

# - Plotting functions
from modules.my_plotting import plot_skill_maps
from modules.my_plotting import plot_global_skill
from modules.my_plotting import plot_global_skills
from modules.my_plotting import plot_skills_distribution
from modules.my_plotting import create_gif_forecast_error
from modules.my_plotting import create_gif_forecast_anom_error

# For plotting 
import matplotlib
matplotlib.use('cairo') # Cairo
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["savefig.facecolor"] = "white" # (1,1,1,0)
matplotlib.rcParams["savefig.edgecolor"] = 'none'

# Disable warnings
# warnings.filterwarnings("ignore")

## DeepSphere-Weather

data_dir = "/ltenas3/DeepSphere/data/preprocessed_ds/ERA5_HRES"  
model_dir = "/data/weather_prediction/experiments_GG/new/RNN-AR6-UNetSpherical-Healpix_400km-Graph_knn-k20-MaxAreaPooli/"
 
#-------------------------------------------------------------------------.
# Read config file 
cfg_path = os.path.join(model_dir, 'config.json')
cfg = read_config_file(fpath=cfg_path)
# Some special options to adjust for prediction
cfg['dataloader_settings']["autotune_num_workers"] = False
cfg['dataloader_settings']["num_workers"] = 11
cfg['training_settings']['gpu_training'] = True  # to run prediction in GPU if possible 
##------------------------------------------------------------------------.
### Retrieve experiment-specific configuration settings   
model_settings = get_model_settings(cfg)   
ar_settings = get_ar_settings(cfg)
training_settings = get_training_settings(cfg) 
dataloader_settings = get_dataloader_settings(cfg) 

##------------------------------------------------------------------------.
#### Load Zarr Datasets
data_sampling_dir = os.path.join(data_dir, cfg['model_settings']["sampling_name"])

data_dynamic = xr.open_zarr(os.path.join(data_sampling_dir, "Data","dynamic", "time_chunked", "dynamic.zarr")) 
data_bc = xr.open_zarr(os.path.join(data_sampling_dir, "Data","bc", "time_chunked", "bc.zarr")) 
data_static = xr.open_zarr(os.path.join(data_sampling_dir, "Data", "static.zarr")) 

# - Select dynamic features 
data_dynamic = data_dynamic[['z500','t850']]    
data_dynamic['lat'] = data_dynamic['lat'].load()
data_dynamic['lon'] = data_dynamic['lon'].load()
##------------------------------------------------------------------------.
### Prepare static data 
# - Keep land-surface mask as it is 

# - Keep sin of latitude and remove longitude information 
data_static = data_static.drop(["sin_longitude","cos_longitude"])

# - Scale orography between 0 and 1 (is already left 0 bounded)
data_static['orog'] = data_static['orog']/data_static['orog'].max()

# - One Hot Encode soil type 
# ds_slt_OHE = xscaler.OneHotEnconding(data_static['slt'])
# data_static = xr.merge([data_static, ds_slt_OHE])
# data_static = data_static.drop('slt')

# - Load static data 
data_static = data_static.load()

##-----------------------------------------------------------------------------.
#### Split data into train, test and validation set 
# - Defining time split for training 
training_years = np.array(['1980-01-01T07:00','2014-12-31T23:00'], dtype='M8[m]')  
validation_years = np.array(['2015-01-01T00:00','2016-12-31T23:00'], dtype='M8[m]')    
test_years = np.array(['2017-01-01T00:00','2018-12-31T23:00'], dtype='M8[m]')   

# - Split data sets 
t_i = time.time()
training_data_dynamic = data_dynamic.sel(time=slice(training_years[0], training_years[-1]))
training_data_bc = data_bc.sel(time=slice(training_years[0], training_years[-1]))
    
validation_data_dynamic = data_dynamic.sel(time=slice(validation_years[0], validation_years[-1]))
validation_data_bc = data_bc.sel(time=slice(validation_years[0], validation_years[-1]))

test_data_dynamic = data_dynamic.sel(time=slice(test_years[0], test_years[-1]))
test_data_bc = data_bc.sel(time=slice(test_years[0], test_years[-1]))

print('- Splitting data into train, validation and test sets: {:.2f}s'.format(time.time() - t_i))


##------------------------------------------------------------------------.
#### Define scaler to apply on the fly within DataLoader 
# - Load scalers
dynamic_scaler = LoadScaler(os.path.join(data_sampling_dir, "Scalers", "GlobalStandardScaler_dynamic.nc"))
bc_scaler = LoadScaler(os.path.join(data_sampling_dir, "Scalers", "GlobalStandardScaler_bc.nc"))
    # # - Create single scaler 
scaler = SequentialScaler(dynamic_scaler, bc_scaler)

##------------------------------------------------------------------------.
### Define pyTorch settings (before PyTorch model definition)
# - Here inside is eventually set the seed for fixing model weights initialization
# - Here inside the training precision is set (currently only float32 works)
device = set_pytorch_settings(training_settings)

##------------------------------------------------------------------------.
## Retrieve dimension info of input-output Torch Tensors
tensor_info = get_ar_model_tensor_info(ar_settings = ar_settings,
                                        data_dynamic = data_dynamic, 
                                        data_static = data_static, 
                                        data_bc = data_bc)
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

##-------------------------------------------------------------------------.
### - Create predictions 
exp_dir = model_dir 
print("========================================================================================")
print("- Running predictions")
forecast_zarr_fpath = os.path.join(exp_dir, "model_predictions/forecast_chunked/test_forecasts.zarr")
dask.config.set(scheduler='synchronous') # This is very important otherwise the dataloader hang
ds_forecasts = AutoregressivePredictions(model = model, 
                                        # Data
                                        data_dynamic = test_data_dynamic,
                                        data_static = data_static,              
                                        data_bc = test_data_bc, 
                                        scaler_transform = scaler,
                                        scaler_inverse = scaler,
                                        # Dataloader options
                                        device = device,
                                        batch_size = 50,  # number of forecasts per batch
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
                                        ar_iterations = 20,        # How many time to autoregressive iterate
                                        # Save options 
                                        zarr_fpath = forecast_zarr_fpath,  # None --> do not write to disk
                                        rounding = 2,             # Default None. Accept also a dictionary 
                                        compressor = "auto",      # Accept also a dictionary per variable
                                        chunks = "auto")
##------------------------------------------------------------------------.
### Reshape forecast Dataset for verification
# - For efficient verification, data must be contiguous in time, 
#   but chunked over space (and leadtime) 
dask.config.set(scheduler='threads')
## Reshape from 'forecast_reference_time'-'leadtime' to 'time (aka) forecasted_time'-'leadtime'  
# - Rechunk Dataset over space on disk and then reshape the for verification
print("========================================================================================")
print("- Reshape test set predictions for verification")
t_i = time.time()
verification_zarr_fpath = os.path.join(exp_dir, "model_predictions/space_chunked/test_forecasts.zarr")
ds_verification_format = rechunk_forecasts_for_verification(ds=ds_forecasts, 
                                                            chunks="auto", 
                                                            target_store=verification_zarr_fpath,
                                                            max_mem = '2GB')
    
##------------------------------------------------------------------------.
### - Run deterministic verification 
print("========================================================================================")
print("- Run deterministic verification")
# dask.config.set(scheduler='processes')
# - Compute skills
ds_obs = xr.open_zarr(os.path.join(data_sampling_dir, "Data","dynamic", "space_chunked", "dynamic.zarr")) 
ds_skill = xverif.deterministic(pred = ds_verification_format,
                                obs = ds_obs, 
                                forecast_type="continuous",
                                aggregating_dim='time')
# - Save sptial skills 
ds_skill.to_netcdf(os.path.join(exp_dir, "model_skills/deterministic_spatial_skill.nc"))

##------------------------------------------------------------------------.
### - Create verification summary plots and maps
print("========================================================================================")
print("- Create verification summary plots and maps")
# - Add mesh information 
ds_skill = ds_skill.sphere.add_SphericalVoronoiMesh(x='lon', y='lat')

# - Compute global and latitudinal skill summary statistics    
ds_global_skill = xverif.global_summary(ds_skill, area_coords="area")
# ds_latitudinal_skill = xverif.latitudinal_summary(ds_skill, lat_dim='lat', lon_dim='lon', lat_res=5) 
# ds_longitudinal_skill = xverif.longitudinal_summary(ds_skill, lat_dim='lat', lon_dim='lon', lon_res=5) 

# - Save global skills
ds_global_skill.to_netcdf(os.path.join(exp_dir, "model_skills/deterministic_global_skill.nc"))

# - Create spatial maps 
plot_skill_maps(ds_skill = ds_skill,  
                figs_dir = os.path.join(exp_dir, "figs/skills/SpatialSkill"),
                crs_proj = ccrs.Robinson(),
                skills = ['BIAS','RMSE','rSD', 'pearson_R2', 'error_CoV'],
                # skills = ['percBIAS','percMAE','rSD', 'pearson_R2', 'KGE'],
                suffix="",
                prefix="")

# - Create skill vs. leadtime plots 
plot_global_skill(ds_global_skill).savefig(os.path.join(exp_dir, "figs/skills/RMSE_skill.png"))
plot_global_skills(ds_global_skill).savefig(os.path.join(exp_dir, "figs/skills/skills_global.png"))
plot_skills_distribution(ds_skill).savefig(os.path.join(exp_dir, "figs/skills/skills_distribution.png"))
    
##------------------------------------------------------------------------.
### - Create animations 
print("========================================================================================")
print("- Create forecast error animations")
t_i = time.time()
# - Add information related to mesh area
ds_forecasts = ds_forecasts.sphere.add_SphericalVoronoiMesh(x='lon', y='lat')
ds_obs = test_data_dynamic 
ds_obs = ds_obs.sphere.add_SphericalVoronoiMesh(x='lon', y='lat')

# - Plot GIF for different months (variable states)
for month in [1,4,7,10]:
    idx_month = np.argmax(ds_forecasts['forecast_reference_time'].dt.month.values == month)
    ds_forecast = ds_forecasts.isel(forecast_reference_time = idx_month)
    create_gif_forecast_error(gif_fpath = os.path.join(exp_dir, "figs/forecast_states", "M" + '{:02}'.format(month) + ".gif"),
                                ds_forecast = ds_forecast,
                                ds_obs = ds_obs,
                                aspect_cbar = 40,
                                antialiased = False,
                                edgecolors = None)
# - Plot GIF for different months (variable anomalies)
hourly_weekly_anomaly_scaler = LoadAnomaly(os.path.join(data_sampling_dir, "Scalers", "MonthlyStdAnomalyScaler_dynamic.nc"))
for month in [1,4,7,10]:
    idx_month = np.argmax(ds_forecasts['forecast_reference_time'].dt.month.values == month)
    ds_forecast = ds_forecasts.isel(forecast_reference_time = idx_month)
    create_gif_forecast_anom_error(gif_fpath = os.path.join(exp_dir, "figs/forecast_anom", "M" + '{:02}'.format(month) + ".gif"),
                                    ds_forecast = ds_forecast,
                                    ds_obs = ds_obs,
                                    scaler = hourly_weekly_anomaly_scaler,
                                    anom_title = "Hourly-Weekly Std. Anomaly",
                                    aspect_cbar = 40,
                                    antialiased = True,
                                    edgecolors = None)