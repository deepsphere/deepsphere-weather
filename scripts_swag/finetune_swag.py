#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 20:04:40 2021

@author: ghiggi
"""
import os
# os.chdir('/home/ghiggi/Projects/deepsphere-weather')
import sys
# sys.path.append('../')
import warnings
import time
import dask
import argparse
import numpy as np
import cartopy.crs as ccrs
import xarray as xr
from torch import optim
from torchinfo import summary

import xverif 
import xsphere  # required for xarray 'sphere' accessor 
from xscaler import LoadScaler, SequentialScaler, LoadAnomaly
from xforecasting import (
    AutoregressiveTraining,
    # AutoregressivePredictions,
    rechunk_forecasts_for_verification,
    EarlyStopping, 
    AR_Scheduler,
)
from xforecasting.utils.io import get_ar_model_tensor_info
from xforecasting.utils.torch import summarize_model

## DeepSphere-Weather modules
import modules.my_models_graph as my_architectures
from modules.loss import WeightedMSELoss, AreaWeights
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
from modules.utils_config import print_dim_info
from modules.utils_models import get_pygsp_graph

## SWAG utils
from modules.utils_config import get_SWAG_settings
from modules.utils_config import get_pytorch_SWAG_model
from modules.utils_config import load_pretrained_ar_scheduler # TODO maybe unecessary
from modules.predictions_autoregressive import AutoregressiveSWAGPredictions

# - Plotting functions
from modules.my_plotting import plot_skill_maps
from modules.my_plotting import plot_global_skill
from modules.my_plotting import plot_global_skills
from modules.my_plotting import plot_skills_distribution
from modules.my_plotting import create_gif_forecast_error
from modules.my_plotting import create_gif_forecast_anom_error

# For plotting 
import matplotlib
# matplotlib.use('cairo') # Cairo
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["savefig.facecolor"] = "white" # (1,1,1,0)
matplotlib.rcParams["savefig.edgecolor"] = 'none'

# Disable warnings
warnings.filterwarnings("ignore")

# # - exp_swag.py 
#-----------------------------------------------------------------------------.
def main(cfg_path, exp_dir, data_dir, train=True, pred=True):
    """General function for training DeepSphere4Earth models."""
    ##------------------------------------------------------------------------.
    t_start = time.time()
    ### Read experiment configuration settings 
    cfg = read_config_file(fpath=cfg_path)
    
    # Some special stuff you might want to adjust 
    cfg['dataloader_settings']["prefetch_in_gpu"] = False  
    cfg['dataloader_settings']["prefetch_factor"] = 2      
    cfg['dataloader_settings']["num_workers"] = 8
    cfg['dataloader_settings']["autotune_num_workers"] = False
    cfg['dataloader_settings']["pin_memory"] = False
    cfg['dataloader_settings']["asyncronous_gpu_transfer"] = True
    # cfg['model_settings']["architecture_name"] = 'UNetSpherical'
    # cfg['model_settings']["model_name_suffix"] = "LinearStep"
    cfg['model_settings']['pretrained_model_name'] = 'RNN-UNetSpherical-icosahedral-16-k20-MaxAreaPooling-float32-AR6-LinearStep'
    cfg['model_settings']['model_name'] = 'RNN-UNetSpherical-icosahedral-16-k20-MaxAreaPooling-float32-AR6-LinearStep-SWAG-LR0007'
    cfg['training_settings']["ar_training_strategy"] = "RNN" # "RNN" # "AR" # "RNN"
    cfg['training_settings']['epochs'] = 4
    cfg['training_settings']['learning_rate'] = 0.007
    cfg['ar_settings']["ar_iterations"] = 6
    cfg['SWAG_settings']["sampling_scale"] = 0.0
    cfg['SWAG_settings']["nb_samples"] = 1
    cfg['SWAG_settings']["target_learning_rate"] = 0.001
    ##------------------------------------------------------------------------.
    ### Retrieve experiment-specific configuration settings   
    model_settings = get_model_settings(cfg)   
    ar_settings = get_ar_settings(cfg)
    training_settings = get_training_settings(cfg) 
    dataloader_settings = get_dataloader_settings(cfg) 
    SWAG_settings = get_SWAG_settings(cfg)

    sampling_scale = str(SWAG_settings["sampling_scale"]).replace(".", "")
    
    ##------------------------------------------------------------------------.
    #### Load Datasets
    data_sampling_dir = os.path.join(data_dir, cfg['model_settings']["sampling_name"])

    da_dynamic = xr.open_zarr(os.path.join(data_sampling_dir, "Data","dynamic", "time_chunked", "dynamic.zarr"))["data"]
    da_bc = xr.open_zarr(os.path.join(data_sampling_dir, "Data","bc", "time_chunked", "bc.zarr"))["data"]
    ds_static = xr.open_zarr(os.path.join(data_sampling_dir, "Data", "static.zarr")) 
    # - Align Datasets (currently required)
    # ds_dynamic, ds_bc = xr.align(ds_dynamic, ds_bc) 
    # - Select dynamic features 
    da_dynamic = da_dynamic.sel(feature=["z500", "t850"])
    
    ##------------------------------------------------------------------------.
    # - Prepare static data 
    # - Keep land-surface mask as it is 
    # - Keep sin of latitude and remove longitude information 
    ds_static = ds_static.drop(["sin_longitude","cos_longitude"])
    # - Scale orography between 0 and 1 (is already left 0 bounded)
    ds_static['orog'] = ds_static['orog']/ds_static['orog'].max()
    # - One Hot Encode soil type 
    # ds_slt_OHE = xscaler.OneHotEnconding(ds_static['slt'])
    # ds_static = xr.merge([ds_static, ds_slt_OHE])
    # ds_static = ds_static.drop('slt')
    # - Convert to DataArray 
    da_static = ds_static.to_array(dim="feature")    
    
    ##------------------------------------------------------------------------.
    #### Define scaler to apply on the fly within DataLoader 
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
    da_training_dynamic = da_dynamic.sel(time=slice(training_years[0], training_years[-1]))
    da_training_bc = da_bc.sel(time=slice(training_years[0], training_years[-1]))
      
    da_validation_dynamic = da_dynamic.sel(time=slice(validation_years[0], validation_years[-1]))
    da_validation_bc = da_bc.sel(time=slice(validation_years[0], validation_years[-1]))
    
    da_test_dynamic = da_dynamic.sel(time=slice(test_years[0], test_years[-1]))
    da_test_bc = da_bc.sel(time=slice(test_years[0], test_years[-1]))
    
    print('- Splitting data into train, validation and test sets: {:.2f}s'.format(time.time() - t_i))
        
    ##------------------------------------------------------------------------.
    ### Define pyTorch settings 
    device = set_pytorch_settings(training_settings)
    
    ##------------------------------------------------------------------------.
    ## Retrieve dimension info of input-output Torch Tensors
    dim_info = get_ar_model_tensor_info(ar_settings=ar_settings,
                                    da_dynamic=da_training_dynamic, 
                                    da_static=da_static, 
                                    da_bc=da_training_bc)
    
    # - Add dim info to cfg file 
    model_settings['dim_info'] = dim_info
    cfg['model_settings']['dim_info'] = dim_info
    
    ##------------------------------------------------------------------------.
    # Print model settings 
    print_model_description(cfg)
    print_dim_info(dim_info)  

    ##------------------------------------------------------------------------. 
    ### Define the model architecture  
    model = get_pytorch_model(module = my_architectures,
                              model_settings = model_settings)
    # - Define SWAG model
    swag_model = get_pytorch_SWAG_model(module = my_architectures, 
                                        model_settings = model_settings, 
                                        swag_settings = SWAG_settings)
    ###-----------------------------------------------------------------------.
    ## If requested, load a pre-trained model for fine-tuning
    if model_settings['pretrained_model_name'] is not None:
        model_dir = os.path.join(exp_dir, model_settings['model_name'])
        load_pretrained_model(model = model, 
                              model_dir = model_dir)
        
        # - Collect the model weights in SWAG model
        swag_model.collect_model(model)
        
    ###-----------------------------------------------------------------------.
    ### Transfer model to the device (i.e. GPU)
    model = model.to(device)
    swag_model = swag_model.to(device)
    
    ###-----------------------------------------------------------------------.
    ### Summarize the model 
    input_size=(training_settings["training_batch_size"], *dim_info['input_shape'])
    summary(model, input_size, col_names = ["input_size", "output_size","num_params"])
    
    _ = summarize_model(model=model, 
                        input_size=dim_info['input_shape'],  
                        batch_size=training_settings["training_batch_size"], 
                        device=device)

    ###-----------------------------------------------------------------------.
    # DataParallel training option on multiple GPUs
    # if training_settings['DataParallel_training'] is True:
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
    
    experiments_dir = exp_dir
    exp_dir = create_experiment_directories(exp_dir = exp_dir,      
                                            model_name = model_name,
                                            force=True)
 
    ##------------------------------------------------------------------------.
    # Define model weights filepath 
    model_fpath = os.path.join(exp_dir, "model_weights", "model_swag.h5")
    
    ##------------------------------------------------------------------------.
    # Write config file in the experiment directory 
    write_config_file(cfg = cfg,
                      fpath = os.path.join(exp_dir, 'config.json'))
       
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
    # - Load pretrained AR_Weights_Scheduler
    # TODO: this is inside ar_training_info.AR_scheduler now ! 
    if model_settings['pretrained_model_name'] is not None:
        ar_scheduler = load_pretrained_ar_scheduler(exp_dir = experiments_dir,
                                                    model_name = model_settings['pretrained_model_name'])
    else:
        # - For RNN: growth and decay works well (fix the first)
        if training_settings["AR_training_strategy"] == "RNN":
            ar_scheduler = AR_Scheduler(method = "LinearStep",
                                        factor = 0.0005,
                                        fixed_ar_weights = [0],

                                        initial_ar_absolute_weights = [1,1]) 
        # - FOR AR : Do not decay weights once they growthed
        elif training_settings["AR_training_strategy"] == "AR":                                
            ar_scheduler = AR_Scheduler(method = "LinearStep",
                                        factor = 0.0005,
                                        fixed_ar_weights = np.arange(0, ar_settings['ar_iterations']),
                                        initial_ar_absolute_weights = [1, 1])   
        else:
            raise NotImplementedError("'AR_training_strategy' must be either 'AR' or 'RNN'.")

    ##------------------------------------------------------------------------.
    ### - Define Early Stopping 
    # - Used also to update AR_scheduler (increase AR iterations) if 'ar_iterations' not reached.
    patience = 500
    minimum_iterations = 500
    minimum_improvement = 0 # 0 to not stop 
    stopping_metric = 'training_total_loss'   # training_total_loss                                                     
    mode = "min" # MSE best when low  
    early_stopping = EarlyStopping(patience = patience,
                                   minimum_improvement = minimum_improvement,
                                   minimum_iterations = minimum_iterations,
                                   stopping_metric = stopping_metric,                                                         
                                   mode = mode)  
                                   
    ##------------------------------------------------------------------------.
    ### - Defining LR_Scheduler 
    lr_scheduler = None
    ##------------------------------------------------------------------------.
    ### - Train the model 
    if train:
        ### - Create predictions 
        print("========================================================================================")
        print("- SWAG training")
        dask.config.set(scheduler='synchronous')
        ar_training_info = AutoregressiveTraining(model = model,
                                                model_fpath = model_fpath,  
                                                # Loss settings 
                                                criterion = criterion,
                                                optimizer = optimizer,  
                                                lr_scheduler = lr_scheduler, 
                                                ar_scheduler = ar_scheduler,                                
                                                early_stopping = early_stopping,
                                                # Data
                                                training_data_dynamic = da_training_dynamic,
                                                validation_data_dynamic = da_validation_dynamic,
                                                da_static = da_static,              
                                                training_data_bc = da_training_bc,         
                                                validation_data = da_validation_bc,  
                                                scaler = scaler, 
                                                # Dataloader settings
                                                num_workers = dataloader_settings['num_workers'],  # dataloader_settings['num_workers'], 
                                                autotune_num_workers = dataloader_settings['autotune_num_workers'], 
                                                prefetch_factor = dataloader_settings['prefetch_factor'],  
                                                prefetch_in_gpu = dataloader_settings['prefetch_in_gpu'], 
                                                drop_last_batch = dataloader_settings['drop_last_batch'],     
                                                random_shuffle = dataloader_settings['random_shuffle'], 
                                                pin_memory = dataloader_settings['pin_memory'], 
                                                asyncronous_gpu_transfer = dataloader_settings['asyncronous_gpu_transfer'], 
                                                # Autoregressive settings  
                                                input_k = ar_settings['input_k'], 
                                                output_k = ar_settings['output_k'], 
                                                forecast_cycle = ar_settings['forecast_cycle'],                         
                                                ar_iterations = ar_settings['ar_iterations'], 
                                                stack_most_recent_prediction = ar_settings['stack_most_recent_prediction'], 
                                                # Training settings 
                                                ar_training_strategy = training_settings["AR_training_strategy"],
                                                training_batch_size = training_settings['training_batch_size'], 
                                                validation_batch_size = training_settings['validation_batch_size'],   
                                                epochs = training_settings['epochs'], 
                                                scoring_interval = training_settings['scoring_interval'], 
                                                save_model_each_epoch = training_settings['save_model_each_epoch'],
                                                # SWAG settings
                                                swag = True,
                                                swag_model = swag_model,
                                                swag_freq = SWAG_settings['swag_freq'],
                                                swa_start = SWAG_settings['swa_start'], 
                                                # GPU settings 
                                                device = device)
        ##------------------------------------------------------------------------.    
        # # Load AR TrainingInfo
        # with open(os.path.join(exp_dir,"ar_training_info/AR_TrainingInfo.pickle"), 'rb') as handle:
        #    ar_training_info = pickle.load(handle)
       
        ##------------------------------------------------------------------------.
        ### Create plots related to training evolution  
        print("========================================================================================")
        print("- Creating plots to investigate training evolution")
        ar_training_info.plots(exp_dir=exp_dir, ylim=(0,0.06)) 
        
        ##-------------------------------------------------------------------------.
        
    if pred:
        ### - Create predictions 
        print("========================================================================================")
        print("- Running predictions")
        forecast_zarr_fpath = os.path.join(exp_dir, f"model_predictions/spatial_chunks/test_pred_{sampling_scale}_median.zarr")
        dask.config.set(scheduler='synchronous')
        ds_forecasts = AutoregressiveSWAGPredictions(model = swag_model, 
                                                    exp_dir = exp_dir, 
                                                    # Data
                                                    training_data_dynamic = da_training_dynamic,
                                                    test_data_dynamic = da_test_dynamic,
                                                    data_static = da_static,              
                                                    training_data_bc = da_training_bc,         
                                                    test_data_bc = da_test_bc, 
                                                    # Scaler options
                                                    scaler_transform = scaler,
                                                    scaler_inverse = scaler,
                                                    # Dataloader options
                                                    device = device,
                                                    batch_size = training_settings['training_batch_size'],  # number of forecasts per batch
                                                    num_workers = dataloader_settings['num_workers'], 
                                                    prefetch_factor = dataloader_settings['prefetch_factor'], 
                                                    prefetch_in_gpu = dataloader_settings['prefetch_in_gpu'],  
                                                    pin_memory = dataloader_settings['pin_memory'],
                                                    asyncronous_gpu_transfer = dataloader_settings['asyncronous_gpu_transfer'],
                                                    # Autoregressive settings  
                                                    input_k = ar_settings['input_k'], 
                                                    output_k = ar_settings['output_k'], 
                                                    forecast_cycle = ar_settings['forecast_cycle'],                         
                                                    ar_iterations = 20, 
                                                    stack_most_recent_prediction = ar_settings['stack_most_recent_prediction'],
                                                    # SWAG settings
                                                    no_cov_mat=SWAG_settings['no_cov_mat'],
                                                    sampling_scale = SWAG_settings["sampling_scale"],
                                                    nb_samples = SWAG_settings["nb_samples"],
                                                    # Save options  
                                                    rounding = 2,             # Default None. Accept also a dictionary 
                                                    compressor = "auto",      # Accept also a dictionary per variable
                                                    chunks = "auto",          
                                                    timedelta_unit='hour')
    else:
        ds_forecasts = xr.open_zarr(forecast_zarr_fpath, chunks="auto")

    ##------------------------------------------------------------------------.
    ### Reshape forecast Dataset for verification
    # - For efficient verification, data must be contiguous in time, but chunked over space (and leadtime) 
    # dask.config.set(pool=ThreadPool(4))
    dask.config.set(scheduler='processes')
    
    ## Reshape from 'forecast_reference_time'-'leadtime' to 'time (aka) forecasted_time'-'leadtime'  
    # - Rechunk Dataset over space on disk and then reshape the for verification
    print("========================================================================================")
    print("- Reshape test set predictions for verification")
    verification_zarr_fpath = os.path.join(exp_dir, f"model_predictions/temporal_chunks/test_pred_{sampling_scale}.zarr")
    ds_verification_format = rechunk_forecasts_for_verification(ds=ds_forecasts, 
                                                                chunks="auto", 
                                                                target_store=verification_zarr_fpath,
                                                                max_mem = '1GB')
     
    ##------------------------------------------------------------------------.
    ### - Run deterministic verification 
    print("========================================================================================")
    print("- Run deterministic verification")
    # dask.config.set(scheduler='processes')
    # - Compute skills
    ds_obs_test = da_test_dynamic.to_dataset('feature')
    ds_obs_test = ds_obs_test.chunk({'time': -1,'node': 1})
    ds_skill = xverif.deterministic(pred = ds_verification_format,
                                    obs = ds_obs_test, 
                                    forecast_type="continuous",
                                    aggregating_dim='time')
    # - Save sptial skills 
    ds_skill.to_netcdf(os.path.join(exp_dir, f"model_skills/deterministic_spatial_skill_{sampling_scale}.nc"))
    
    ##------------------------------------------------------------------------.
    ### - Create verification summary plots and maps
    print("========================================================================================")
    print("- Create verification summary plots and maps")
    # - Add mesh information 
    # ---> TODO: To generalize based on cfg sampling !!!
    pygsp_graph = get_pygsp_graph(sampling = model_settings['sampling'], 
                                  resolution = model_settings['resolution'],
                                  knn = model_settings['knn'])
    ds_skill = ds_skill.sphere.add_nodes_from_pygsp(pygsp_graph=pygsp_graph)
    ds_skill = ds_skill.sphere.add_SphericalVoronoiMesh(x='lon', y='lat')
    
    # - Compute global and latitudinal skill summary statistics    
    ds_global_skill = xverif.global_summary(ds_skill, area_coords="area")
    # ds_latitudinal_skill = xverif.latitudinal_summary(ds_skill, lat_dim='lat', lon_dim='lon', lat_res=5) 
    # ds_longitudinal_skill = xverif.longitudinal_summary(ds_skill, lat_dim='lat', lon_dim='lon', lon_res=5) 
    
    # - Save global skills
    ds_global_skill.to_netcdf(os.path.join(exp_dir, f"model_skills/deterministic_global_skill_{sampling_scale}.nc"))

    # - Create spatial maps 
    plot_skill_maps(ds_skill = ds_skill,  
                    figs_dir = os.path.join(exp_dir, f"figs/sampling_{sampling_scale}/skills/SpatialSkill"),
                    crs_proj = ccrs.Robinson(),
                    skills = ['BIAS','RMSE','rSD', 'pearson_R2', 'error_CoV'],
                    # skills = ['percBIAS','percMAE','rSD', 'pearson_R2', 'KGE'],
                    suffix="",
                    prefix="")

    # - Create skill vs. leadtime plots 
    plot_global_skill(ds_global_skill).savefig(os.path.join(exp_dir, f"figs/sampling_{sampling_scale}/skills/RMSE_skill.png"))
    plot_global_skills(ds_global_skill).savefig(os.path.join(exp_dir, f"figs/sampling_{sampling_scale}/skills/skills_global.png"))
    plot_skills_distribution(ds_skill).savefig(os.path.join(exp_dir, f"figs/sampling_{sampling_scale}/skills/skills_distribution.png"))
        
    ##------------------------------------------------------------------------.
    ### - Create animations 
    print("========================================================================================")
    print("- Create forecast error animations")
    t_i = time.time()
    # - Add information related to mesh area
    ds_forecasts = ds_forecasts.sphere.add_nodes_from_pygsp(pygsp_graph=pygsp_graph)
    ds_forecasts = ds_forecasts.sphere.add_SphericalVoronoiMesh(x='lon', y='lat')
    ds_obs_test = ds_obs_test.chunk({'time': 100,'node': -1})
    ds_obs = ds_obs_test.sphere.add_nodes_from_pygsp(pygsp_graph=pygsp_graph)
    ds_obs = ds_obs.sphere.add_SphericalVoronoiMesh(x='lon', y='lat')
    # - Plot gif for different months (variable states)
    for month in range(1,13):
        idx_month = np.argmax(ds_forecasts['forecast_reference_time'].dt.month.values == month)
        ds_forecast = ds_forecasts.isel(forecast_reference_time = idx_month)
        create_gif_forecast_error(gif_fpath = os.path.join(exp_dir, f"figs/sampling_{sampling_scale}/forecast_states", "M" + '{:02}'.format(month) + ".gif"),
                                  ds_forecast = ds_forecast,
                                  ds_obs = ds_obs,
                                  aspect_cbar = 40,
                                  antialiased = False,
                                  edgecolors = None)
    # - Plot gif for different months (variable anomalies)
    hourly_weekly_anomaly_scaler = LoadAnomaly(os.path.join(data_sampling_dir, "Scalers", "WeeklyHourlyStdAnomalyScaler_dynamic.nc"))
    for month in range(1,13):
        idx_month = np.argmax(ds_forecasts['forecast_reference_time'].dt.month.values == month)
        ds_forecast = ds_forecasts.isel(forecast_reference_time = idx_month)
        create_gif_forecast_anom_error(gif_fpath = os.path.join(exp_dir, f"figs/sampling_{sampling_scale}/forecast_anom", "M" + '{:02}'.format(month) + ".gif"),
                                       ds_forecast = ds_forecast,
                                       ds_obs = ds_obs,
                                       scaler = hourly_weekly_anomaly_scaler,
                                       anom_title = "Hourly-Weekly Std. Anomaly",
                                       aspect_cbar = 40,
                                       antialiased = True,
                                       edgecolors = None)
    ##-------------------------------------------------------------------------.                                     
    print("   ---> Elapsed time: {:.1f} minutes ".format((time.time() - t_i)/60))
    ##-------------------------------------------------------------------------.                            
    print("========================================================================================")
    print("- Model training and verification terminated. Elapsed time: {:.1f} hours ".format((time.time() - t_start)/60/60))  
    print("========================================================================================")
    ##-------------------------------------------------------------------------.

if __name__ == '__main__':
    # default_config = 'configs/UNetSpherical/Healpix_400km/SWAG.json' 
    default_config = 'configs/UNetSpherical/Icosahedral_400km/SWAG.json'

    parser = argparse.ArgumentParser(description='Training weather prediction model')
    parser.add_argument('--config_file', type=str, default=default_config)
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--pred', action='store_true')

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    data_dir = "/mnt/scratch/students/wefeng/data/"
    exp_dir = "/mnt/scratch/students/haddad/experiments"

    main(args.config_file, exp_dir, data_dir, train=args.train, pred=args.pred)