import os
# os.chdir('/home/ghiggi/Projects/deepsphere-weather')
import sys
sys.path.append('../')    

import xarray as xr
import xsphere   # required for xarray 'sphere' accessor
from xscaler import LoadAnomaly 

from modules.utils_config import read_config_file
from modules.my_plotting import create_gif_forecast_error
from modules.my_plotting import create_gif_forecast_anom_error

# Plotting options
import matplotlib
matplotlib.use('cairo') # Cairo
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["savefig.facecolor"] = "white" # (1,1,1,0)
matplotlib.rcParams["savefig.edgecolor"] = 'none'
matplotlib.rcParams["figure.dpi"] = 200

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12 
matplotlib.rcParams['font.size'] = SMALL_SIZE  # controls default text sizes
      
matplotlib.rcParams['axes.titlesize'] = SMALL_SIZE  # fontsize of the axes title
matplotlib.rcParams['axes.labelsize'] = SMALL_SIZE     # fontsize of the x and y labels
matplotlib.rcParams['xtick.labelsize'] = SMALL_SIZE    # fontsize of the tick labels
matplotlib.rcParams['ytick.labelsize'] = SMALL_SIZE    # fontsize of the tick labels
matplotlib.rcParams['legend.fontsize'] = SMALL_SIZE    # legend fontsize
matplotlib.rcParams['figure.titlesize'] = MEDIUM_SIZE  # fontsize of the figure title


##-----------------------------------------------------------------------------.
# Define directories
base_dir = "/data/weather_prediction"
figs_dir = os.path.join(base_dir, "figs")

data_dir = "/ltenas3/DeepSphere/data/preprocessed_ds/ERA5_HRES"  
model_name = "OLD_fine_tuned2_without_batchnorm_-RNN-AR6-UNetSpherical-Healpix_400km-Graph_knn-k20-MaxAreaPooling"
model_dir = os.path.join("/data/weather_prediction/experiments_GG/new_old_archi", model_name)   
 
#-------------------------------------------------------------------------.
# Read config file 
cfg_path = os.path.join(model_dir, 'config.json')
cfg = read_config_file(fpath=cfg_path)

##------------------------------------------------------------------------.
#### Load Zarr Datasets
data_sampling_dir = os.path.join(data_dir, cfg['model_settings']["sampling_name"])

ds_dynamic = xr.open_zarr(os.path.join(data_sampling_dir, "Data","dynamic", "time_chunked", "dynamic.zarr")) 
ds_bc = xr.open_zarr(os.path.join(data_sampling_dir, "Data","bc", "time_chunked", "bc.zarr")) 
ds_static = xr.open_zarr(os.path.join(data_sampling_dir, "Data", "static.zarr")) 

# - Select dynamic features 
ds_dynamic = ds_dynamic[['z500','t850']]    

# - Load lat and lon coordinates
ds_dynamic['lat'] = ds_dynamic['lat'].load()
ds_dynamic['lon'] = ds_dynamic['lon'].load()

###-----------------------------------------------------------------------.
### Load forecasts 
forecast_zarr_fpath = os.path.join(model_dir, "model_predictions/forecast_chunked/test_forecasts.zarr")
ds_forecasts = xr.open_zarr(forecast_zarr_fpath)

# - Add information related to mesh area
ds_forecasts = ds_forecasts.sphere.add_SphericalVoronoiMesh(x='lon', y='lat')
ds_obs = ds_dynamic 
ds_obs = ds_obs.sphere.add_SphericalVoronoiMesh(x='lon', y='lat')

###-----------------------------------------------------------------------.
# Plot anims 
ds_forecast = ds_forecasts.isel(forecast_reference_time = 0)

create_gif_forecast_error(gif_fpath = os.path.join(figs_dir, "Forecast_State_Error.gif"),
                          ds_forecast = ds_forecast,
                          ds_obs = ds_obs,
                          fps = 4,
                          aspect_cbar = 40,
                          antialiased = False,
                          edgecolors = None)

hourly_weekly_anomaly_scaler = LoadAnomaly(os.path.join(data_sampling_dir, "Scalers", "WeeklyHourlyStdAnomalyScaler_dynamic.nc"))
create_gif_forecast_anom_error(gif_fpath = os.path.join(figs_dir, "Forecast_Anom_Error.gif"),
                               ds_forecast = ds_forecast,
                               ds_obs = ds_obs,
                               scaler = hourly_weekly_anomaly_scaler,
                               anom_title = "Hourly-Weekly Std. Anomaly",
                               fps = 4,
                               aspect_cbar = 40,
                               antialiased = False,
                               edgecolors = None)
