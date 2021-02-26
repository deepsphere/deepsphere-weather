#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 20:04:40 2021

@author: ghiggi
"""
import os
import warnings
import time
import dask
import argparse
import torch
import pickle
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from torch import nn, optim

## DeepSphere-Earth
from modules.utils_config import get_default_settings
from modules.utils_config import read_config_file
from modules.utils_config import write_config_file
from modules.utils_config import get_model_settings
from modules.utils_config import get_training_settings
from modules.utils_config import get_AR_settings
from modules.utils_config import get_dataloader_settings
# from modules.utils_config import get_pytorch_model
from modules.utils_config import get_model_name
from modules.utils_config import set_pytorch_settings
from modules.utils_config import load_pretrained_model
from modules.utils_config import create_experiment_directories
from modules.utils_config import print_model_description
from modules.utils_config import print_dim_info
from modules.utils_io import get_AR_model_diminfo
from modules.utils_io import check_AR_DataArrays 
from modules.training_autoregressive import AutoregressiveTraining
from modules.predictions_autoregressive import AutoregressivePredictions
from modules.predictions_autoregressive import rechunk_forecasts_for_verification
from modules.predictions_autoregressive import reshape_forecasts_for_verification
from modules.utils_torch import summarize_model
from modules.AR_Scheduler import AR_Scheduler
from modules.early_stopping import EarlyStopping

## Project specific functions
from modules.xscaler import LoadScaler
from modules.xscaler import SequentialScaler
 
from modules.my_io import readDatasets   
from modules.my_io import reformat_Datasets
import modules.my_models_graph as my_architectures
from modules.loss import WeightedMSELoss, compute_error_weight

## Disable warnings
warnings.filterwarnings("ignore")
##----------------------------------------------------------------------------.
## Assumptions 
# - Assume data with same temporal resolution 
# - Assume data without missing timesteps 
# - Do not expect data with different scales 

##----------------------------------------------------------------------------.
## Tensor dimension order !!!
# - For autoregressive: 
# ['sample','time', 'node', ..., 'feature']
# - For classical 
# ['sample','node', ... , 'feature]
# ['sample','time', 'node','feature']

# - Dimension order should be generalized (to a cfg setting) ?
# - ['time', 'node', 'level', 'ensemble', 'feature'] 

# Feature dimension data order = [static, bc, dynamic]
##----------------------------------------------------------------------------.
### TODO
# - kernel_size vs kernel_size_pooling  ???
# - ratio? periodic?
# - numeric_precision: currently work only for float32 !!!

# Torch precision 
# --> Set torch.set_default_tensor_type() 

# Set RNG in worker_init_fn of DataLoader for reproducible ... 
##----------------------------------------------------------------------------.
#### Set CUDA GPU options
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]=  
# os.environ["CUDA_LAUNCH_BLOCKING"] = 1

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

data_dir = "/data/weather_prediction/data"
exp_dir = "/data/weather_prediction/experiments"
cfg_path = "/home/ghiggi/Projects/weather_prediction/configs/UNetSpherical/Healpix_400km/LearnPool-k20.json"
#-----------------------------------------------------------------------------.

"""General function for training DeepSphere4Earth models."""
##------------------------------------------------------------------------.
### Read experiment configuration settings 
cfg = read_config_file(fpath=cfg_path)

# Some special stuff you might want to adjust 
cfg['dataloader_settings']["prefetch_in_GPU"] = False# True? To test 
cfg['dataloader_settings']["prefetch_factor"] = 2     # Maybe increase if only prefetch on CPU? 
cfg['dataloader_settings']["num_workers"] = 8
cfg['dataloader_settings']["autotune_num_workers"] = False
cfg['dataloader_settings']["pin_memory"] = False
cfg['dataloader_settings']["asyncronous_GPU_transfer"] = True

##------------------------------------------------------------------------.
### Retrieve experiment-specific configuration settings   
model_settings = get_model_settings(cfg)   
AR_settings = get_AR_settings(cfg)
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

ds_dynamic = ds_dynamic.drop(["level","lat","lon"])
ds_bc = ds_bc.drop(["lat","lon"])
ds_static = ds_static.drop(["lat","lon"])

##-----------------------------------------------------------------------------.
#### Define scaler to apply on the fly within DataLoader 
# - Load scalers
dynamic_scaler = LoadScaler(os.path.join(data_sampling_dir, "Scalers", "GlobalStandardScaler_dynamic.nc"))
bc_scaler = LoadScaler(os.path.join(data_sampling_dir, "Scalers", "GlobalStandardScaler_bc.nc"))
static_scaler = LoadScaler(os.path.join(data_sampling_dir, "Scalers", "GlobalStandardScaler_static.nc"))
# # - Create single scaler 
scaler = SequentialScaler(dynamic_scaler, bc_scaler, static_scaler)

##-----------------------------------------------------------------------------.
#### Split data into train, test and validation set 
# - Defining time split for training 
training_years = np.array(['2010-01-01T07:00','2014-12-31T23:00'], dtype='M8[m]')  
validation_years = np.array(['2015-01-01T00:00','2016-12-31T23:00'], dtype='M8[m]')    
test_years = np.array(['2017-01-01T00:00','2018-12-31T23:00'], dtype='M8[m]')   

# - Split data sets 
t_i = time.time()
ds_training_dynamic = ds_dynamic.sel(time=slice(training_years[0], training_years[-1]))
ds_training_bc = ds_bc.sel(time=slice(training_years[0], training_years[-1]))
    
ds_validation_dynamic = ds_dynamic.sel(time=slice(validation_years[0], validation_years[-1]))
ds_validation_bc = ds_bc.sel(time=slice(validation_years[0], validation_years[-1]))

ds_test_dynamic = ds_dynamic.sel(time=slice(test_years[0], test_years[-1]))
ds_test_bc = ds_bc.sel(time=slice(test_years[0], test_years[-1]))

print('- Splitting data into train, validation and test sets: {:.2f}s'.format(time.time() - t_i))

##-----------------------------------------------------------------------------.
### Dataset to DataArray conversion 
# TODO this will be remove when Zarr source data are DataArrays 
dict_DataArrays = reformat_Datasets(ds_training_dynamic = ds_training_dynamic,
                                    ds_validation_dynamic = ds_validation_dynamic,
                                    ds_static = ds_static,              
                                    ds_training_bc = ds_training_bc,         
                                    ds_validation_bc = ds_validation_bc,
                                    preload_data_in_CPU = True)

da_static = dict_DataArrays['da_static']
da_training_dynamic = dict_DataArrays['da_training_dynamic']
da_validation_dynamic = dict_DataArrays['da_validation_dynamic']
da_training_bc = dict_DataArrays['da_training_bc']
da_validation_bc = dict_DataArrays['da_validation_bc']

check_AR_DataArrays(da_training_dynamic = da_training_dynamic,
                    da_training_bc = da_training_bc,  
                    da_static = da_static,
                    da_validation_dynamic = da_validation_dynamic,
                    da_validation_bc = da_validation_bc,
                    verbose=True)    

dict_test_DataArrays = reformat_Datasets(ds_training_dynamic = ds_test_dynamic,
                                            ds_static = ds_static,  
                                            ds_training_bc = ds_test_bc,            
                                            preload_data_in_CPU = True)

da_static = dict_test_DataArrays['da_static']
da_test_dynamic = dict_test_DataArrays['da_training_dynamic']
da_test_bc = dict_test_DataArrays['da_training_bc']

##------------------------------------------------------------------------.
### Define pyTorch settings 
device = set_pytorch_settings(training_settings)

##------------------------------------------------------------------------.
## Retrieve dimension info of input-output Torch Tensors
dim_info = get_AR_model_diminfo(AR_settings=AR_settings,
                                da_dynamic=da_training_dynamic, 
                                da_static=da_static, 
                                da_bc=da_training_bc)

model_settings['dim_info'] = dim_info

print_dim_info(dim_info)
    
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
## If requested, load a pre-trained model for fine-tuning
if model_settings['pretrained_model_name'] is not None:
    load_pretrained_model(model = model, exp_dir=exp_dir, model_name = model_settings['pretrained_model_name'])
    
###-----------------------------------------------------------------------.
### Transfer model to GPU 
model = model.to(device)

### Summarize the model 
profiling_info = summarize_model(model=model, 
                                    input_size=dim_info['input_shape'],  
                                    batch_size=training_settings["training_batch_size"], 
                                    device=device)
###-----------------------------------------------------------------------.
# DataParallel training option on multiple GPUs
# if training_settings['DataParallel_training'] is True:
#     if torch.cuda.device_count() > 1 and len(training_settings['GPU_devices_ids']) > 1:
#         model = nn.DataParallel(model, device_ids=[i for i in training_settings['GPU_devices_ids']])
    
###-----------------------------------------------------------------------.
## Generate the (new) model name and its directories 
if model_settings['model_name'] is not None:
    model_name = model_settings['model_name']
else: 
    cfg['model_settings']["model_name_prefix"] = None
    cfg['model_settings']["model_name_suffix"] = None
    model_name = get_model_name(cfg)
    model_settings['model_name'] = model_name

exp_dir = create_experiment_directories(exp_dir = exp_dir,      
                                        model_name = model_name,
                                        force=True) 

##------------------------------------------------------------------------.
# Define model weights filepath 
# TODO: (@Yasser, Wentao) (better name convention?)
model_fpath = os.path.join(exp_dir, "model_weights", "model.h5")

##------------------------------------------------------------------------.
# Write config file in the experiment directory 
write_config_file(cfg = cfg,
                    fpath = os.path.join(exp_dir, 'config.json'))

##------------------------------------------------------------------------.
# Print model settings 
print_model_description(cfg)

##------------------------------------------------------------------------.
### - Define custom loss function 
# TODO (@Wentao) 
# - --> variable weights 
# - --> spatial masking 
# - --> area_weights   
weights = compute_error_weight(model.graphs[0])
criterion = WeightedMSELoss(weights=weights)

##------------------------------------------------------------------------.
### - Define optimizer 
optimizer = optim.Adam(model.parameters(),    
                        lr=training_settings['learning_rate'], 
                        eps=1e-7,    
                        weight_decay=0, amsgrad=False)
    
##------------------------------------------------------------------------.
### - Define AR_Weights_Scheduler 
AR_scheduler = AR_Scheduler(method = "LinearStep",
                            factor = 0.0005,
                            initial_AR_weights = [1])   

### - Define Early Stopping 
# - Used also to update AR_scheduler (increase AR iterations) if 'AR_iterations' not reached.
patience = 1 #200
minimum_iterations = 300 # 10000
# patience = 100
# minimum_iterations = 5000
minimum_improvement = 0.001 # 0 to not stop 
stopping_metric = 'validation_total_loss'   # training_total_loss                                                     
mode = "min" # MSE best when low  
early_stopping = EarlyStopping(patience = patience,
                                minimum_improvement = minimum_improvement,
                                minimum_iterations = minimum_iterations,
                                stopping_metric = stopping_metric,                                                         
                                mode = mode)  
##------------------------------------------------------------------------.
### - Defining LR_Scheduler 
# TODO (@Yasser)
# - https://www.kaggle.com/isbhargav/guide-to-pytorch-learning-rate-scheduling
# LR_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1, last_epoch=-1)
# LR_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80,100,150], gamma=0.1)
# LR_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
# LR_scheduler = optim.lr_scheduler.ReduceLROnPlateau
LR_scheduler = None
##------------------------------------------------------------------------.
### - Train the model 
dask.config.set(scheduler='synchronous')
training_info = AutoregressiveTraining(model = model,
                                        model_fpath = model_fpath,  
                                        # Loss settings 
                                        criterion = criterion,
                                        optimizer = optimizer,  
                                        LR_scheduler = LR_scheduler, 
                                        AR_scheduler = AR_scheduler,                                
                                        early_stopping = early_stopping,
                                        # Data
                                        da_training_dynamic = da_training_dynamic,
                                        da_validation_dynamic = da_validation_dynamic,
                                        da_static = da_static,              
                                        da_training_bc = da_training_bc,         
                                        da_validation_bc = da_validation_bc,  
                                        scaler = scaler, 
                                        # Dataloader settings
                                        num_workers = dataloader_settings['num_workers'],  # dataloader_settings['num_workers'], 
                                        autotune_num_workers = dataloader_settings['autotune_num_workers'], 
                                        prefetch_factor = dataloader_settings['prefetch_factor'],  
                                        prefetch_in_GPU = dataloader_settings['prefetch_in_GPU'], 
                                        drop_last_batch = dataloader_settings['drop_last_batch'],     
                                        random_shuffle = dataloader_settings['random_shuffle'], 
                                        pin_memory = dataloader_settings['pin_memory'], 
                                        asyncronous_GPU_transfer = dataloader_settings['asyncronous_GPU_transfer'], 
                                        # Autoregressive settings  
                                        input_k = AR_settings['input_k'], 
                                        output_k = AR_settings['output_k'], 
                                        forecast_cycle = AR_settings['forecast_cycle'],                         
                                        AR_iterations = AR_settings['AR_iterations'], 
                                        stack_most_recent_prediction = AR_settings['stack_most_recent_prediction'], 
                                        # Training settings 
                                        numeric_precision = training_settings['numeric_precision'], 
                                        training_batch_size = training_settings['training_batch_size'], 
                                        validation_batch_size = training_settings['validation_batch_size'],   
                                        epochs = training_settings['epochs'], 
                                        scoring_interval = training_settings['scoring_interval'], 
                                        save_model_each_epoch = training_settings['save_model_each_epoch'], 
                                        # GPU settings 
                                        device = device)

# Save AR TrainingInfo
with open(os.path.join(exp_dir,"training_info/AR_TrainingInfo.pickle"), 'wb') as handle:
    pickle.dump(training_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

exp_dir = "/data/weather_prediction/experiments_GG/UNetSpherical-healpix-16-k20-InterpPooling-float32-AR6"
# # Load AR TrainingInfo
with open(os.path.join(exp_dir,"training_info/AR_TrainingInfo.pickle"), 'rb') as handle:
    training_info = pickle.load(handle)
    
##------------------------------------------------------------------------.
### Create plots related to training evolution  
## - Plot the loss at all AR iterations (in one figure)
fig, ax = plt.subplots()
for AR_iteration in range(training_info.AR_iterations+1):
    ax = training_info.plot_loss_per_AR_iteration(AR_iteration = AR_iteration, 
                                                    ax = ax,
                                                    linestyle="solid",
                                                    linewidth=0.3, 
                                                    plot_validation = False, 
                                                    plot_labels = True,
                                                    plot_legend = False,
                                                    add_AR_weights_updates=False)
leg = ax.legend(labels=list(range(training_info.AR_iterations + 1)), 
                title="AR iteration", 
                loc='upper right')
# - Make legend line more thick
for line in leg.get_lines():
    line.set_linewidth(2)

plt.gca().set_prop_cycle(None) # Reset color cycling 
for AR_iteration in range(training_info.AR_iterations+1):
    ax = training_info.plot_loss_per_AR_iteration(AR_iteration = AR_iteration, 
                                                    ax = ax,
                                                    linestyle="dashed",
                                                    linewidth=0.3,
                                                    plot_training = False, 
                                                    plot_labels = False,
                                                    plot_legend = False,
                                                    add_AR_weights_updates=False)  
# - Add vertical line when AR iteration is added
iterations_of_AR_updates = training_info.iterations_of_AR_updates()
if len(iterations_of_AR_updates) > 0: 
    [ax.axvline(x=x, color=(0, 0, 0, 0.90), linewidth=0.1) for x in iterations_of_AR_updates]
# - Save figure
fig.savefig(os.path.join(exp_dir, "figs/training_info/Loss_at_all_AR_iterations.png"))    
##------------------------------------------------------------------------.   
## - Plot the loss at each AR iteration (in separate figures)
for AR_iteration in range(training_info.AR_iterations+1):
    fname = os.path.join(exp_dir, "figs/training_info/Loss_at_AR_{}.png".format(AR_iteration)) 
    training_info.plot_loss_per_AR_iteration(AR_iteration = AR_iteration, 
                                                title="Loss evolution at AR iteration {}".format(AR_iteration)).savefig(fname) 

##------------------------------------------------------------------------.
## - Plot total loss 
training_info.plot_total_loss().savefig(os.path.join(exp_dir, "figs/training_info/Total_Loss.png"))

##------------------------------------------------------------------------.
## - Plot AR weights  
training_info.plot_AR_weights(normalized=True).savefig(os.path.join(exp_dir, "figs/training_info/AR_Normalized_Weights.png"))
training_info.plot_AR_weights(normalized=False).savefig(os.path.join(exp_dir, "figs/training_info/AR_Absolute_Weights.png"))   

##-------------------------------------------------------------------------.
### - Create predictions 
forecast_zarr_fpath = os.path.join(exp_dir, "model_predictions/spatial_chunks/test_pred.zarr")
dask.config.set(scheduler='synchronous')
ds_forecasts = AutoregressivePredictions(model = model, 
                                            # Data
                                            da_dynamic = da_test_dynamic,
                                            da_static = da_static,              
                                            da_bc = da_test_bc, 
                                            scaler = scaler,
                                            # Dataloader options
                                            device = device,
                                            batch_size = training_settings["validation_batch_size"],  # number of forecasts per batch
                                            num_workers = dataloader_settings['num_workers'], 
                                        #  tune_num_workers = False, 
                                            prefetch_factor = dataloader_settings['prefetch_factor'], 
                                            prefetch_in_GPU = dataloader_settings['prefetch_in_GPU'],  
                                            pin_memory = dataloader_settings['pin_memory'],
                                            asyncronous_GPU_transfer = dataloader_settings['asyncronous_GPU_transfer'],
                                            numeric_precision = training_settings['numeric_precision'], 
                                            # Autoregressive settings
                                            input_k = AR_settings['input_k'], 
                                            output_k = AR_settings['output_k'], 
                                            forecast_cycle = AR_settings['forecast_cycle'],                         
                                            stack_most_recent_prediction = AR_settings['stack_most_recent_prediction'], 
                                            AR_iterations = 20,        # How many time to autoregressive iterate
                                            # Save options 
                                            zarr_fpath = forecast_zarr_fpath,  # None --> do not write to disk
                                            rounding = 2,             # Default None. Accept also a dictionary 
                                            compressor = "auto",      # Accept also a dictionary per variable
                                            chunks = "auto",          
                                            timedelta_unit='hour')
##------------------------------------------------------------------------.
### Reshape forecast Dataset for verification
# - For efficient verification, data must be contiguous in time, but chunked over space (and leadtime) 
# dask.config.set(pool=ThreadPool(4))
dask.config.set(scheduler='processes')
## Reshape on the fly (might work with the current size of data)
# --> To test ...  
# ds_verification = reshape_forecasts_for_verification(ds_forecasts)

## Reshape from 'forecast_reference_time'-'leadtime' to 'time (aka) forecasted_time'-'leadtime'  
# - Rechunk Dataset over space on disk and then reshape the for verification
verification_zarr_fpath = os.path.join(exp_dir, "model_predictions/temporal_chunks/test_pred.zarr")
ds_verification = rechunk_forecasts_for_verification(ds=ds_forecasts, 
                                                        chunks="auto", 
                                                        target_store=verification_zarr_fpath,
                                                        max_mem = '1GB')
    
##------------------------------------------------------------------------.
from modules import xsphere
import cartopy.crs as ccrs

forecast_zarr_fpath = os.path.join(exp_dir, "model_predictions/spatial_chunks/test_pred.zarr")
verification_zarr_fpath = os.path.join(exp_dir, "model_predictions/temporal_chunks/test_pred.zarr")

ds = xr.open_zarr(forecast_zarr_fpath)
ds = ds.isel(forecast_reference_time=0, leadtime=[0,10, 15, 20]) # select 4 timesteps
ds = ds.load()

import pygsp as pg
ds = ds.sphere.add_nodes_from_pygsp(pygsp_graph=pg.graphs.SphereHealpix(subdivisions=16, k=20, nest=True))
ds = ds.sphere.add_SphericalVoronoiMesh(x='lon',y='lat')
##----------------------------------------------------------------------------.
## Prepare data for below plots
da = ds['z500']
da = ds['t850']
 

crs_proj = ccrs.Robinson()

#### Plot() Dataset 
p = da.sphere.plot(transform=ccrs.Geodetic(),  
                   subplot_kws={'projection': crs_proj},
                   # Facets options 
                   col="leadtime", col_wrap=2, 
                   # Polygon border option
                   edgecolors="white", # None to not display polygon border 
                   linewidths=0.01,
                   antialiased=True,
                   # Colorbar options 
                   add_colorbar = True,
                   cmap = plt.get_cmap('Spectral_r'),
                   #    vmin = 48000,
                   #    vmax = 56500,
                   robust=True,
                   extend = 'neither', # 'neither', 'both', 'min', 'max'
                   # norm=None,
                   # center=None,
                   # colors=None,
                   # levels=None,
                   # cbar_kwargs
                   # Add colorbar label 
                   add_labels = True)
# Set map options
for ax in p.axes.flat:
    ax.coastlines(alpha=0.2)
plt.show()