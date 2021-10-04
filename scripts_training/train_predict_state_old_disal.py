#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 13:41:16 2021

@author: ghiggi
"""
import os
# os.chdir('/home/ghiggi/Projects/deepsphere-weather')
import sys
sys.path.append('../')
import warnings
import time
import dask
import argparse
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

from modules.utils_io import get_ar_model_tensor_info
from modules.training_autoregressive import AutoregressiveTraining
from modules.predictions_autoregressive import AutoregressivePredictions
from modules.predictions_autoregressive import rechunk_forecasts_for_verification
from modules.utils_torch import summarize_model
from modules.AR_Scheduler import AR_Scheduler
from modules.early_stopping import EarlyStopping
from modules.loss import WeightedMSELoss, AreaWeights

## Project specific functions
# import modules.my_models_graph as my_architectures
import modules.my_models_graph_old as my_architectures

## Side-project utils (maybe migrating to separate packages in future)
import modules.xsphere  # required for xarray 'sphere' accessor 
import modules.xverif as xverif
from modules.xscaler import LoadScaler
from modules.xscaler import SequentialScaler
from modules.xscaler import LoadAnomaly

# - Plotting functions
from modules.my_plotting import plot_skill_maps
from modules.my_plotting import plot_global_skill
from modules.my_plotting import plot_global_skills
from modules.my_plotting import plot_skills_distribution
from modules.my_plotting import create_hovmoller_plots
from modules.my_plotting import create_gif_forecast_error
from modules.my_plotting import create_gif_forecast_anom_error

# For plotting 
import matplotlib
matplotlib.use('cairo') # Cairo
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["savefig.facecolor"] = "white" # (1,1,1,0)
matplotlib.rcParams["savefig.edgecolor"] = 'none'

# Disable warnings
warnings.filterwarnings("ignore")
    
#-----------------------------------------------------------------------------.
def main(cfg_path, exp_dir, data_dir, force=False):
    """General function for training DeepSphere-Weather models."""
    ##------------------------------------------------------------------------.
    t_start = time.time()
    ### Read experiment configuration settings 
    cfg = read_config_file(fpath=cfg_path)

    ##------------------------------------------------------------------------.
    ### Retrieve experiment-specific configuration settings   
    model_settings = get_model_settings(cfg)   
    ar_settings = get_ar_settings(cfg)
    training_settings = get_training_settings(cfg) 
    dataloader_settings = get_dataloader_settings(cfg) 

    ##------------------------------------------------------------------------.
    # TODO REMOVE 
    model_settings["model_name_prefix"] = ''
    model_settings["architecture_name"] = "ResNetSpherical"
    
    training_settings['seed_model_weights'] = 30 # 20 the previous   
    training_settings['seed_random_shuffling'] = 15 # 15 the previous     
 
    model_settings['knn'] = 20
    model_settings['bias'] = False
    model_settings['batch_norm'] = True
    model_settings['batch_norm_before_activation'] = True
    model_settings['activation'] = True
    model_settings['activation_fun'] = 'relu'

    training_settings['learning_rate'] = 0.001 # 0.007 was working   
    training_settings['scoring_interval'] = 10
    training_settings['training_batch_size'] = 16
    training_settings['validation_batch_size'] = 16
    dataloader_settings['prefetch_factor'] = 6
    dataloader_settings['pin_memory'] = False
    dataloader_settings['num_workers'] = 12
    dataloader_settings['random_shuffling'] = True
    dataloader_settings['autotune_num_workers'] = False

    training_settings['deterministic_training'] = False
    training_settings['benchmark_cuDNN'] = True
    
    ##------------------------------------------------------------------------.
    ### Update experiment-specific configuration settings   
    cfg["model_settings"] = model_settings  
    cfg["ar_settings"] = ar_settings
    cfg["training_settings"] = training_settings
    cfg["dataloader_settings"] = dataloader_settings
    
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

    ##------------------------------------------------------------------------.
    ### Prepare static data 
    # - Keep land-surface mask as it is 

    # - Keep sin of latitude and remove longitude information 
    ds_static = ds_static.drop(["sin_longitude","cos_longitude"])

    # - Scale orography between 0 and 1 (is already left 0 bounded)
    ds_static['orog'] = ds_static['orog']/ds_static['orog'].max()

    # - One Hot Encode soil type 
    # ds_slt_OHE = xscaler.OneHotEnconding(ds_static['slt'])
    # ds_static = xr.merge([ds_static, ds_slt_OHE])
    # ds_static = ds_static.drop('slt')

    # - Load static data 
    ds_static = ds_static.load()
 
    #------------------------------------------------------------------------.
    ### Define scaler to apply on the fly within DataLoader 
    # - Load scalers
    dynamic_scaler = LoadScaler(os.path.join(data_sampling_dir, "Scalers", "GlobalStandardScaler_dynamic.nc"))
    bc_scaler = LoadScaler(os.path.join(data_sampling_dir, "Scalers", "GlobalStandardScaler_bc.nc"))
    # # - Create single scaler object
    scaler = SequentialScaler(dynamic_scaler, bc_scaler)

    ##-----------------------------------------------------------------------------.
    #### Split data into train, test and validation set 
    # - Defining time split for training 
    training_years = np.array(['1980-01-01T07:00','2014-12-31T23:00'], dtype='M8[m]')  
    validation_years = np.array(['2015-01-01T00:00','2016-12-31T23:00'], dtype='M8[m]')    
    test_years = np.array(['2017-01-01T00:00','2018-12-31T23:00'], dtype='M8[m]')   

    # - Split data sets 
    t_i = time.time()
    training_ds_dynamic = ds_dynamic.sel(time=slice(training_years[0], training_years[-1]))
    training_ds_bc = ds_bc.sel(time=slice(training_years[0], training_years[-1]))
        
    validation_ds_dynamic = ds_dynamic.sel(time=slice(validation_years[0], validation_years[-1]))
    validation_ds_bc = ds_bc.sel(time=slice(validation_years[0], validation_years[-1]))

    test_ds_dynamic = ds_dynamic.sel(time=slice(test_years[0], test_years[-1]))
    test_ds_bc = ds_bc.sel(time=slice(test_years[0], test_years[-1]))

    print('- Splitting data into train, validation and test sets: {:.2f}s'.format(time.time() - t_i))
    
    ##------------------------------------------------------------------------.
    ### Define pyTorch settings (before PyTorch model definition)
    # - Here inside is eventually set the seed for fixing model weights initialization
    # - Here inside the training precision is set (currently only float32 works)
    device = set_pytorch_settings(training_settings)

    ##------------------------------------------------------------------------.
    ## Retrieve dimension info of input-output Torch Tensors
    tensor_info = get_ar_model_tensor_info(ar_settings = ar_settings,
                                           data_dynamic = training_ds_dynamic, 
                                           data_static = ds_static, 
                                           data_bc = training_ds_bc)
    print_tensor_info(tensor_info)         
    # - Add dim info to cfg file 
    model_settings['tensor_info'] = tensor_info
    cfg['model_settings']['tensor_info'] = tensor_info 

    ##------------------------------------------------------------------------.
    # Print model settings 
    print_model_description(cfg)
    
    ##------------------------------------------------------------------------.
    ### Define the model architecture  
    model = get_pytorch_model(module = my_architectures,
                              model_settings = model_settings)
    ###-----------------------------------------------------------------------.
    ## If requested, load a pre-trained model for fine-tuning
    if model_settings['pretrained_model_name'] is not None:
        pretrained_model_dir = os.path.join(exp_dir, model_settings['model_name'])
        load_pretrained_model(model = model, 
                              model_dir = pretrained_model_dir)
    ###-----------------------------------------------------------------------.
    ### Transfer model to the device (i.e. GPU)
    model = model.to(device)
    
    ###-----------------------------------------------------------------------.
    ### Summarize the model 
    input_shape = tensor_info['input_shape'].copy()
    input_shape[0] = training_settings["training_batch_size"]
    print(summary(model, input_shape, col_names = ["input_size", "output_size","num_params"]))
 
    _ = summarize_model(model=model, 
                        input_size = tuple(tensor_info['input_shape'][1:]),  
                        batch_size = training_settings["training_batch_size"], 
                        device=device)

    ###-----------------------------------------------------------------------.
    # DataParallel training option on multiple GPUs
    # if training_settings['dataparallel_training'] is True:
    #     if torch.cuda.device_count() > 1 and len(training_settings['gpu_devices_ids']) > 1:
    #         model = nn.DataParallel(model, device_ids=[i for i in training_settings['gpu_devices_ids']])
        
    ###-----------------------------------------------------------------------.
    ## Generate the (new) model name and its directories 
    if model_settings['model_name'] is not None:
        model_name = model_settings['model_name']
    else: 
        model_name = get_model_name(cfg)
        model_settings['model_name'] = model_name
        cfg['model_settings']["model_name_prefix"] = None
        cfg['model_settings']["model_name_suffix"] = None
    
    model_dir = create_experiment_directories(exp_dir = exp_dir,      
                                              model_name = model_name,
                                              force=force) # force=True will delete existing directory
    
    ##------------------------------------------------------------------------.
    # Define model weights filepath 
    model_fpath = os.path.join(model_dir, "model_weights", "model.h5")
    
    ##------------------------------------------------------------------------.
    # Write config file in the experiment directory 
    write_config_file(cfg = cfg,
                      fpath = os.path.join(model_dir, 'config.json'))
       
    ##------------------------------------------------------------------------.
    ### - Define custom loss function  
    # - Compute area weights
    weights = AreaWeights(model.graphs[0])
    
    # - Define weighted loss 
    criterion = WeightedMSELoss(weights=weights)
    
    ##------------------------------------------------------------------------.
    ### - Define optimizer 
    optimizer = optim.Adam(model.parameters(),    
                           lr=training_settings['learning_rate'], 
                           eps=1e-7,    
                           weight_decay=0, amsgrad=False)
    
    ##------------------------------------------------------------------------.
    ## - Define AR_Weights_Scheduler 
    # - For RNN: growth and decay works well (fix the first)
    if training_settings["ar_training_strategy"] == "RNN":
        ar_scheduler = AR_Scheduler(method = "LinearStep",
                                    factor = 0.0005,
                                    fixed_ar_weights = [0],
                                    initial_ar_absolute_weights = [1,1]) 
    # - FOR AR : Do not decay weights once they growthed
    elif training_settings["ar_training_strategy"] == "AR":                                
        ar_scheduler = AR_Scheduler(method = "LinearStep",
                                    factor = 0.0005,
                                    fixed_ar_weights = np.arange(0, ar_settings['ar_iterations']),
                                    initial_ar_absolute_weights = [1, 1])   
    else:
        raise NotImplementedError("'ar_training_strategy' must be either 'AR' or 'RNN'.")

    ##------------------------------------------------------------------------.
    ### - Define Early Stopping 
    # - Used also to update ar_scheduler (aka increase AR iterations) if 'ar_iterations' not reached.
    patience = 3000 # when 'scoring_interval' = 10  --> minimum 4000 batches    
    minimum_iterations = 5000      # 1000
    minimum_improvement = 0.0001   # could try to use 0.0001
    stopping_metric = 'validation_total_loss' # training_total_loss                                                     
    mode = "min" # MSE best when low  
    early_stopping = EarlyStopping(patience = patience,
                                   minimum_improvement = minimum_improvement,
                                   minimum_iterations = minimum_iterations,
                                   stopping_metric = stopping_metric,                                                         
                                   mode = mode)  
    
    ##------------------------------------------------------------------------.
    ### - Define LR_Scheduler 
    lr_scheduler = None
    
    ##------------------------------------------------------------------------.
    ### - Train the model 
    dask.config.set(scheduler='synchronous') # This is very important otherwise the dataloader hang
    ar_training_info = AutoregressiveTraining( model = model,
                                               model_fpath = model_fpath,  
                                               # Loss settings 
                                               criterion = criterion,
                                               optimizer = optimizer,  
                                               lr_scheduler = lr_scheduler, 
                                               ar_scheduler = ar_scheduler,                                
                                               early_stopping = early_stopping,
                                               # Data
                                               data_static = ds_static,   
                                               training_data_dynamic = training_ds_dynamic,
                                               training_data_bc = training_ds_bc, 
                                               validation_data_dynamic = validation_ds_dynamic,
                                               validation_data_bc = validation_ds_bc,  
                                               scaler = scaler, 
                                               # Dataloader settings
                                               num_workers = dataloader_settings['num_workers'],  # dataloader_settings['num_workers'], 
                                               autotune_num_workers = dataloader_settings['autotune_num_workers'], 
                                               prefetch_factor = dataloader_settings['prefetch_factor'],  
                                               prefetch_in_gpu = dataloader_settings['prefetch_in_gpu'], 
                                               drop_last_batch = dataloader_settings['drop_last_batch'],     
                                               shuffle = dataloader_settings['random_shuffling'], 
                                               shuffle_seed = training_settings['seed_random_shuffling'],
                                               pin_memory = dataloader_settings['pin_memory'], 
                                               asyncronous_gpu_transfer = dataloader_settings['asyncronous_gpu_transfer'], 
                                               # Autoregressive settings  
                                               input_k = ar_settings['input_k'], 
                                               output_k = ar_settings['output_k'], 
                                               forecast_cycle = ar_settings['forecast_cycle'],                         
                                               ar_iterations = ar_settings['ar_iterations'], 
                                               stack_most_recent_prediction = ar_settings['stack_most_recent_prediction'], 
                                               # Training settings 
                                               ar_training_strategy = training_settings["ar_training_strategy"],
                                               training_batch_size = training_settings['training_batch_size'], 
                                               validation_batch_size = training_settings['validation_batch_size'],   
                                               epochs = training_settings['epochs'], 
                                               scoring_interval = training_settings['scoring_interval'], 
                                               save_model_each_epoch = training_settings['save_model_each_epoch'], 
                                               # GPU settings 
                                               device = device)
    
    ##------------------------------------------------------------------------.
    ## Load AR TrainingInfo
    # with open(os.path.join(os.path.dirname(model_fpath), "AR_TrainingInfo.pickle"), 'rb') as handle:
    #    ar_training_info = pickle.load(handle)
  
    ##------------------------------------------------------------------------.
    ### Create plots related to training evolution  
    print("========================================================================================")
    print("- Creating plots to investigate training evolution")
    ar_training_info.plots(model_dir=model_dir, ylim=(0,0.06))  # TODO: 0.03
    
    ##-------------------------------------------------------------------------.
    ##########################################
    ### - Run predictions for the test set ###
    ##########################################  
    print("========================================================================================")
    print("- Running predictions on the test set")
    forecast_zarr_fpath = os.path.join(model_dir, "model_predictions/forecast_chunked/test_forecasts.zarr")
    dask.config.set(scheduler='synchronous') # This is very important otherwise the dataloader hang
    ds_forecasts = AutoregressivePredictions(model = model, 
                                             # Data
                                             data_dynamic = test_ds_dynamic,        
                                             data_bc = test_ds_bc, 
                                             data_static = ds_static,  
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

    ##-------------------------------------------------------------------------.                                                                                
    ##########################################
    ### - Run multi-year simulations       ###
    ########################################## 
    print("========================================================================================")
    print("- Running some multi-year simulations")
    # - Define multi-years simulations settings
    n_year_sims = 2
    forecast_cycle = ar_settings['forecast_cycle']
    ar_iterations = int(24/forecast_cycle*365*n_year_sims)
    ar_blocks = None # Do all predictions in one-run
    forecast_reference_times = ['1992-07-22T00:00:00','2015-12-31T18:00:00','2016-04-01T10:00:00']
    batch_size = len(forecast_reference_times)
    long_forecast_zarr_fpath = os.path.join(model_dir, "model_predictions", "long_simulation", "2year_sim.zarr")
    # - Run long-term simulations
    dask.config.set(scheduler='synchronous')
    ds_long_forecasts = AutoregressivePredictions(model = model, 
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

    ##------------------------------------------------------------------------.
    #########################################
    ### - Run deterministic verification ####
    #########################################
    ### Reshape forecast Dataset for verification
    # - For efficient verification, data must be contiguous in time, but chunked over space (and leadtime) 
    # - It also neeed to swap from 'forecast_reference_time' to the (forecasted) 'time' dimension 
    #   The (forecasted) 'time'dimension is calculed as the 'forecast_reference_time'+'leadtime'  
    print("========================================================================================")
    print("- Rechunk and reshape test set forecasts for verification")
    dask.config.set(scheduler='threads')
    t_i = time.time()
    verification_zarr_fpath = os.path.join(model_dir, "model_predictions/space_chunked/test_forecasts.zarr")
    ds_verification_format = rechunk_forecasts_for_verification(ds=ds_forecasts, 
                                                                chunks="auto", 
                                                                target_store=verification_zarr_fpath,
                                                                max_mem = '1GB')
    print("   ---> Elapsed time: {:.1f} minutes ".format((time.time() - t_i)/60)) 

    ##------------------------------------------------------------------------.
    ### Run deterministic verification
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
    ds_skill.to_netcdf(os.path.join(model_dir, "model_skills/deterministic_spatial_skill.nc"))
    
    ##------------------------------------------------------------------------.
    ####################################################
    ### - Create verification summary plots and maps ###
    ####################################################
    print("========================================================================================")
    print("- Create verification summary plots and maps")
    # - Add mesh information 
    ds_skill = ds_skill.sphere.add_SphericalVoronoiMesh(x='lon', y='lat')
    
    # - Compute global and latitudinal skill summary statistics    
    ds_global_skill = xverif.global_summary(ds_skill, area_coords="area")
    # ds_latitudinal_skill = xverif.latitudinal_summary(ds_skill, lat_dim='lat', lon_dim='lon', lat_res=5) 
    # ds_longitudinal_skill = xverif.longitudinal_summary(ds_skill, lat_dim='lat', lon_dim='lon', lon_res=5) 
    
    # - Save global skills
    ds_global_skill.to_netcdf(os.path.join(model_dir, "model_skills/deterministic_global_skill.nc"))

    # - Create spatial maps 
    plot_skill_maps(ds_skill = ds_skill,  
                    figs_dir = os.path.join(model_dir, "figs/skills/SpatialSkill"),
                    crs_proj = ccrs.Robinson(),
                    skills = ['BIAS','RMSE','rSD', 'pearson_R2', 'error_CoV'],
                    # skills = ['percBIAS','percMAE','rSD', 'pearson_R2', 'KGE'],
                    suffix="",
                    prefix="")

    # - Create skill vs. leadtime plots 
    plot_global_skill(ds_global_skill).savefig(os.path.join(model_dir, "figs/skills/RMSE_skill.png"))
    plot_global_skills(ds_global_skill).savefig(os.path.join(model_dir, "figs/skills/skills_global.png"))
    plot_skills_distribution(ds_skill).savefig(os.path.join(model_dir, "figs/skills/skills_distribution.png"))
        
    ##------------------------------------------------------------------------.
    ############################
    ### - Create animations ####
    ############################
    print("========================================================================================")
    print("- Create forecast error animations")
    t_i = time.time()
    # - Add information related to mesh area
    ds_forecasts = ds_forecasts.sphere.add_SphericalVoronoiMesh(x='lon', y='lat')
    ds_obs = ds_dynamic 
    ds_obs = ds_obs.sphere.add_SphericalVoronoiMesh(x='lon', y='lat')

    # - Plot GIF for different months (variable states)
    for month in [1,4,7,10]:
        idx_month = np.argmax(ds_forecasts['forecast_reference_time'].dt.month.values == month)
        ds_forecast = ds_forecasts.isel(forecast_reference_time = idx_month)
        create_gif_forecast_error(gif_fpath = os.path.join(model_dir, "figs/forecast_states", "M" + '{:02}'.format(month) + ".gif"),
                                  ds_forecast = ds_forecast,
                                  ds_obs = ds_obs,
                                  aspect_cbar = 40,
                                  antialiased = False,
                                  edgecolors = None)
        
    # - Plot GIF for different months (variable anomalies)
    # hourly_weekly_anomaly_scaler = LoadAnomaly(os.path.join(data_sampling_dir, "Scalers", "WeeklyHourlyStdAnomalyScaler_dynamic.nc"))
    # for month in [1,4,7,10]:
    #     idx_month = np.argmax(ds_forecasts['forecast_reference_time'].dt.month.values == month)
    #     ds_forecast = ds_forecasts.isel(forecast_reference_time = idx_month)
    #     create_gif_forecast_anom_error(gif_fpath = os.path.join(model_dir, "figs/forecast_anom", "M" + '{:02}'.format(month) + ".gif"),
    #                                    ds_forecast = ds_forecast,
    #                                    ds_obs = ds_obs,
    #                                    scaler = hourly_weekly_anomaly_scaler,
    #                                    anom_title = "Hourly-Weekly Std. Anomaly",
    #                                    aspect_cbar = 40,
    #                                    antialiased = True,
    #                                    edgecolors = None)
                                   
    print("   ---> Elapsed time: {:.1f} minutes ".format((time.time() - t_i)/60))

    ##-------------------------------------------------------------------------. 
    ###########################################################
    ### - Create Hovmoller plot of multi-years simulations ####
    ###########################################################
    print("========================================================================================")
    print("- Create Hovmoller plots of multi-years simulations")
    t_i = time.time()
    # - Load anomaly scalers
    monthly_std_anomaly_scaler = LoadAnomaly(os.path.join(data_sampling_dir, "Scalers", "MonthlyStdAnomalyScaler_dynamic.nc"))
    # - Create directory where to save figures
    os.makedirs(os.path.join(model_dir, "figs/hovmoller_plots"))
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
                                   
    print("   ---> Elapsed time: {:.1f} minutes ".format((time.time() - t_i)/60))

    ##-------------------------------------------------------------------------.                            
    print("========================================================================================")
    print("- Model training and verification terminated. Elapsed time: {:.1f} hours ".format((time.time() - t_start)/60/60))  
    print("========================================================================================")
    ##-------------------------------------------------------------------------.

if __name__ == '__main__':
    default_data_dir = "/data/deepsphere-weather/data/preprocessed/ERA5_HRES" 
    default_exp_dir = "/data/deepsphere-weather/experiments"
    default_config = '/home/ghiggi/Projects/deepsphere-weather/configs/UNetSpherical/Healpix_400km/MaxAreaPool-Graph_knn.json'
      
    parser = argparse.ArgumentParser(description='Training a numerical weather prediction model emulator')
    parser.add_argument('--config_file', type=str, default=default_config)
    parser.add_argument('--data_dir', type=str, default=default_data_dir)
    parser.add_argument('--exp_dir', type=str, default=default_exp_dir)
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--force', type=str, default='True')                    
    
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] ="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda  
    if args.force == 'True':
        force = True
    else: 
        force = False
        
    main(cfg_path = args.config_file, 
         exp_dir =  args.exp_dir,
         data_dir = args.data_dir,
         force = force)