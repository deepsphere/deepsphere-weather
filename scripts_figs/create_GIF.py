import os
import sys
sys.path.append('../')    
import numpy as np
import pygsp as pg
import matplotlib.pyplot as plt 
from modules.my_io import readDatasets  
from modules.utils_torch import get_time_function
from modules.utils_config import read_config_file
from modules.utils_config import get_model_settings
from modules.utils_config import get_training_settings
from modules.utils_config import get_AR_settings
from modules.utils_config import get_dataloader_settings
from modules.utils_models import get_pygsp_graph
from modules.remap import SphericalVoronoiMeshArea_from_pygsp
import modules.xsphere  
from modules.xscaler import LoadAnomaly
# Plotting options
import matplotlib
matplotlib.use('cairo') # Cairo
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["savefig.facecolor"] = "white" # (1,1,1,0)
matplotlib.rcParams["savefig.edgecolor"] = 'none'
matplotlib.rcParams["figure.dpi"] = 300

##-----------------------------------------------------------------------------.
# Define directories
base_dir = "/data/weather_prediction"
figs_dir = os.path.join(base_dir, "figs")
data_dir = os.path.join(base_dir, "data")  


##-----------------------------------------------------------------------------.
### Model config 
cfg_path = '/home/ghiggi/Projects/weather_prediction/configs/UNetSpherical/Healpix_400km/MaxAreaPool-k20.json'
exp_dir = os.path.join(base_dir, "experiments_GG","RNN-UNetSpherical-healpix-16-k20-MaxAreaPooling-float32-AR6-LinearStep_weight_corrected")

### Read experiment configuration settings 
cfg = read_config_file(fpath=cfg_path)
model_settings = get_model_settings(cfg)   
AR_settings = get_AR_settings(cfg)
training_settings = get_training_settings(cfg) 
dataloader_settings = get_dataloader_settings(cfg) 

### Load data 
data_sampling_dir = os.path.join(data_dir, cfg['model_settings']["sampling_name"])
ds_dynamic = readDatasets(data_dir=data_sampling_dir, feature_type='dynamic')

test_years = np.array(['2017-01-01T00:00','2018-12-31T23:00'], dtype='M8[m]')   
ds_test_dynamic = ds_dynamic.sel(time=slice(test_years[0], test_years[-1]))
ds_obs_test = ds_test_dynamic

forecast_zarr_fpath = os.path.join(exp_dir, "model_predictions/spatial_chunks/test_pred.zarr")
ds_forecasts = xr.open_zarr(forecast_zarr_fpath)

# Get mesh grap
pygsp_graph = get_pygsp_graph(sampling = model_settings['sampling'], 
                              resolution = model_settings['resolution'],
                              knn = model_settings['knn'])

# - Add information related to mesh area
ds_forecasts = ds_forecasts.sphere.add_nodes_from_pygsp(pygsp_graph=pygsp_graph)
ds_forecasts = ds_forecasts.sphere.add_SphericalVoronoiMesh(x='lon', y='lat')
ds_obs_test = ds_obs_test.chunk({'time': 100,'node': -1})
ds_obs = ds_obs_test.sphere.add_nodes_from_pygsp(pygsp_graph=pygsp_graph)
ds_obs = ds_obs.sphere.add_SphericalVoronoiMesh(x='lon', y='lat')

ds_forecast = ds_forecasts.isel(forecast_reference_time = 0)
aspect_cbar = 40
antialiased = False
edgecolors = None
tmp_dir = os.path.join(base_dir, "figs_tmp")
GIF_fpath = os.path.join(base_dir, "figs_tmp","GIF1.gif")

hourly_weekly_anomaly_scaler = LoadAnomaly(os.path.join(data_sampling_dir, "Scalers", "WeeklyHourlyStdAnomalyScaler_dynamic.nc"))


create_GIF_forecast_error(GIF_fpath = os.path.join(base_dir, "figs_tmp", "Forecast_State_Error.gif"),
                          ds_forecast = ds_forecast,
                          ds_obs = ds_obs,
                          fps = 4,
                          aspect_cbar = 40,
                          antialiased = False,
                          edgecolors = None)

create_GIF_forecast_anom_error(GIF_fpath = os.path.join(base_dir, "figs_tmp", "Forecast_Anom_Error.gif"),
                               ds_forecast = ds_forecast,
                               ds_obs = ds_obs,
                               scaler = hourly_weekly_anomaly_scaler,
                               anom_title = "Hourly-Weekly Std. Anomaly",
                               fps = 4,
                               aspect_cbar = 40,
                               antialiased = False,
                               edgecolors = None)
