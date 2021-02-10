#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 18:43:17 2021

@author: ghiggi
"""
import os
import warnings
import time
import torch
import pickle 
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from torch import nn, optim

os.chdir("/home/ghiggi/Projects/weather_prediction/")

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
from modules.utils_config import pytorch_settings
from modules.utils_config import load_pretrained_model
from modules.utils_config import create_experiment_directories
from modules.utils_config import print_model_description
from modules.utils_config import print_dim_info
from modules.utils_io import get_AR_model_diminfo
from modules.utils_io import check_AR_DataArrays 
from modules.training_autoregressive import AutoregressiveTraining
from modules.predictions_autoregressive import AutoregressivePredictions

from modules.AR_Scheduler import AR_Scheduler
from modules.early_stopping import EarlyStopping

## Project specific functions
from modules.xscaler import GlobalStandardScaler  # TemporalStandardScaler
from modules.xscaler import LoadScaler
 
from modules.my_io import readDatasets   
from modules.my_io import reformat_Datasets
import modules.architectures as my_architectures
from modules.loss import WeightedMSELoss, compute_error_weight

## Disable warnings
warnings.filterwarnings("ignore")

##----------------------------------------------------------------------------.
## Assumptions 
# - Assume data with same temporal resolution 
# - Assume data without missing timesteps 
# - Do not expect data with different scales 
#-----------------------------------------------------------------------------.
## Tensor dimension order !!!
# - For autoregressive: 
# ['sample','time', 'node', ..., 'feature']
# - For classical 
# ['sample','node', ... , 'feature]
# ['sample','time', 'node','feature']

# - Dimension order should be generalized (to a cfg setting) ?
# - ['time', 'node', 'level', 'ensemble', 'feature'] 

# Feature dimension data order = [static, bc, dynamic]
#-----------------------------------------------------------------------------.
### TODO
# - kernel_size vs kernel_size_pooling 
# - numeric_precision: currently work only for float32 !!!

# Torch precision 
# --> Set torch.set_default_tensor_type() 

# dataloader_settings : remove preload in CPU 
# Check input data start same timestep

## Code for measuring execution time as function of n. AR iterations 

# import pdb
# pdb.set_trace()

# - Example applications
# https://github.com/deepsphere/deepsphere-pytorch/blob/master/scripts/config.example.yml
# https://github.com/deepsphere/deepsphere-pytorch/blob/master/scripts/temporality/run_ar_tc.py
# https://github.com/deepsphere/deepsphere-pytorch/blob/master/scripts/run_ar_tc.py
#-----------------------------------------------------------------------------.
#######################
# Pytorch Settings ####
#######################
# data_dir = "/data/weather_prediction/ToyData/Healpix_400km/data/"
# data_dir = "/home/ghiggi/Projects/DeepSphere/ToyData/Healpix_400km/data/" # to change to scratch/... 
# # - Dynamic data (i.e. pressure and surface levels variables)
# ds_dynamic = readDatasets(data_dir=data_dir, feature_type='dynamic')
# # - Boundary conditions data (i.e. TOA)
# ds_bc = readDatasets(data_dir=data_dir, feature_type='bc')
# # - Static features
# ds_static = readDatasets(data_dir=data_dir, feature_type='static')

# ds_dynamic = ds_dynamic.drop(["level","lat","lon"])
# ds_bc = ds_bc.drop(["lat","lon"])
# ds_static = ds_static.drop(["lat","lon"])

#-----------------------------------------------------------------------------.
### Lazy Loading of Datasets 
data_dir = "/home/ghiggi/Projects/DeepSphere/ToyData/Healpix_400km"
t_i = time.time()
# - Dynamic data (i.e. pressure and surface levels variables)
ds_dynamic = xr.open_zarr(os.path.join(data_dir,"Dataset","dynamic.zarr"))
# - Boundary conditions data (i.e. TOA)
ds_bc = xr.open_zarr(os.path.join(data_dir,"Dataset", "bc.zarr"))
# - Static features
ds_static = xr.open_zarr(os.path.join(data_dir,"Dataset", "static.zarr"))
print('- Open the Zarr Store (Lazily): {:.2f}s'.format(time.time() - t_i))

# da_dynamic = xr.open_zarr(os.path.join(data_dir, "DataArray", "dynamic.zarr"))['Data']
# da_bc = xr.open_zarr(os.path.join(data_dir,"DataArray", "bc.zarr"))['Data']
# da_static = xr.open_zarr(os.path.join(data_dir,"DataArray","static.zarr"))['Data']

# TO DEBUG
ds_training_dynamic = ds_dynamic
ds_training_bc = ds_bc
ds_validation_dynamic = ds_dynamic
ds_validation_bc = ds_bc
device = 'cpu'

cfg = get_default_settings()
cfg['model_settings']['architecture_name'] = "UNetSpherical"
cfg['model_settings']['architecture_fpath'] = "/home/ghiggi/Projects/DeepSphere/modules/architectures.py"
cfg['model_settings']['sampling'] = "Healpix"
cfg['model_settings']['resolution'] = 16
cfg['model_settings']['model_dir'] = "/home/ghiggi/Projects/DeepSphere/models"

# Current experiment (6h deltat)
cfg['AR_settings']['input_k'] = [-3, -2, -1]
cfg['AR_settings']['output_k'] = [0]
cfg['AR_settings']['forecast_cycle'] = 1
cfg['AR_settings']['AR_iterations'] = 3

# cfg['AR_settings']['input_k'] = [-18, -12, -6]
# cfg['AR_settings']['output_k'] = [0]
# cfg['AR_settings']['forecast_cycle'] = 6
# cfg['AR_settings']['AR_iterations'] = 10

cfg['training_settings']["scoring_interval"] = 5
cfg['training_settings']["training_batch_size"] = 16
cfg['training_settings']["validation_batch_size"] = 16
cfg['training_settings']["epochs"] = 5

cfg['training_settings']['numeric_precision'] = "float32"

cfg['dataloader_settings']["preload_data_in_CPU"] = True
cfg['dataloader_settings']["prefetch_in_GPU"] = False
cfg['dataloader_settings']["prefetch_factor"] = 2
cfg['dataloader_settings']["pin_memory"] = True
cfg['dataloader_settings']["asyncronous_GPU_transfer"] = True
cfg['dataloader_settings']["num_workers"] = 0  
cfg['dataloader_settings']["drop_last_batch"] = False  
        
 
#-----------------------------------------------------------------------------.
### Scale data with xscaler 
dynamic_scaler = GlobalStandardScaler(data=ds_dynamic)
dynamic_scaler.fit()

# bc_scaler = GlobalStandardScaler(data=ds_bc)
# bc_scaler.fit()

# static_scaler = GlobalStandardScaler(data=ds_static)
# static_scaler.fit()

# dynamic_scaler.save("/home/ghiggi/dynamic_scaler.nc")
# dynamic_scaler = LoadScaler("/home/ghiggi/dynamic_scaler.nc")

# ds_dynamic_std = dynamic_scaler.transform(ds_dynamic) # delayed computation
# ds_bc_std = bc_scaler.transform(ds_bc)           # delayed computation
# ds_static_std = static_scaler.transform(ds_static)   # delayed computation

#ds_dynamic1 = dynamic_scaler.inverse_transform(ds_dynamic_std.compute()).compute()
#xr.testing.assert_equal(ds_dynamic, ds_dynamic1) 

#-----------------------------------------------------------------------------.
# ### Split data into train, test and validation set 
# # - Defining time split for training 
# training_years = np.array(['1980-01-01T07:00','2013-12-31T23:00'], dtype='M8[m]')  
# validation_years = np.array(['2014-01-01T00:00','2015-12-31T23:00'], dtype='M8[m]')    
# test_years = np.array(['2016-01-01T00:00','2018-12-31T23:00'], dtype='M8[m]')   
# # - Split data sets 
# t_i = time.time()
# ds_training_dynamic = ds_dynamic_std.sel(time=slice(training_years[0], training_years[-1]))
# ds_training_bc = ds_bc_std.sel(time=slice(training_years[0], training_years[-1]))
  
# ds_validation_dynamic = ds_dynamic_std.sel(time=slice(validation_years[0], validation_years[-1]))
# ds_validation_bc = ds_bc_std.sel(time=slice(validation_years[0], validation_years[-1]))
# print('- Splitting data into train and validation set: {:.2f}s'.format(time.time() - t_i))

# ds_test_dynamic = ds_dynamic_std.sel(time=slice(test_years[0], test_years[-1]))
# ds_test_bc = ds_bc_std.sel(time=slice(test_years[0], test_years[-1]))

#-----------------------------------------------------------------------------.
### Dataset to DataArray conversion 
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

# to debug
da_test_dynamic = da_training_dynamic
da_test_static = da_static             
da_bc = da_test_bc = da_training_bc
                                         
check_AR_DataArrays(da_training_dynamic = da_training_dynamic,
                    da_validation_dynamic = da_validation_dynamic,
                    da_static = da_static,              
                    da_training_bc = da_training_bc,         
                    da_validation_bc = da_validation_bc,
                    verbose=True)                         

scaler = None
#-----------------------------------------------------------------------------.

### GPU options 
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="2,4"

##############  
#### Main ####
############## 
# def main(cfg_path):
# """General function for training DeepSphere4Earth models."""
# TODO: add missing input arguments    

##------------------------------------------------------------------------.
### Read experiment configuration settings 
# cfg = read_config_file(fpath=cfg_path)

##------------------------------------------------------------------------.
### Retrieve experiment-specific configuration settings   
model_settings = get_model_settings(cfg)   
AR_settings = get_AR_settings(cfg)
training_settings = get_training_settings(cfg) 
dataloader_settings = get_dataloader_settings(cfg) 

##------------------------------------------------------------------------.
### Define pyTorch settings 
device = pytorch_settings(training_settings)

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
# model = get_pytorch_model(model_settings = model_settings)
DeepSphereModelClass = getattr(my_architectures, model_settings['architecture_name'])
# - Retrieve required model arguments
model_keys = ['dim_info','resolution', 'conv_type', 'kernel_size', 'sampling',
              'knn', 'pool_method', 'kernel_size_pooling', 'periodic', 'ratio']
model_args = {k: model_settings[k] for k in model_keys}
# - Define DeepSphere model 
model = DeepSphereModelClass(**model_args)              
 
###-----------------------------------------------------------------------.
## If requested, load a pre-trained model for fine-tuning
# TODO: To better define required settings (exp_dir, and weights_fpath?)
if model_settings['pretrained_model_name'] is not None:
    load_pretrained_model(model = model, model_settings = model_settings)
    
###-----------------------------------------------------------------------.
### Transfer model to GPU 
model = model.to(device)

###-----------------------------------------------------------------------.
# DataParallel training option on multiple GPUs
if training_settings['DataParallel_training'] is True:
    if torch.cuda.device_count() > 1 and len(training_settings['GPU_devices_ids']) > 1:
        model = nn.DataParallel(model, device_ids=[i for i in training_settings['GPU_devices_ids']])
    
###-----------------------------------------------------------------------.
## Generate the (new) model name and its directories 
if model_settings['model_name'] is not None:
    model_name = model_settings['model_name']
else: 
    cfg['model_settings']["model_name_prefix"] = None
    cfg['model_settings']["model_name_suffix"] = None
    model_name = get_model_name(cfg)
    model_settings['model_name'] = model_name

exp_dir = create_experiment_directories(model_dir = model_settings['model_dir'],      
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
weights = compute_error_weight(model.sphere_graph)
criterion = WeightedMSELoss(weights=weights)
criterion.to(device) 

##------------------------------------------------------------------------.
### - Define optimizer 
optimizer = optim.Adam(model.parameters(),    
                       lr=training_settings['learning_rate'], 
                       eps=1e-7,    
                       weight_decay=0, amsgrad=False)
  
##------------------------------------------------------------------------.
### - Define AR_Weights_Scheduler 
AR_scheduler = AR_Scheduler(method = "LinearDecay",
                            factor = 0.025)    

### - Define Early Stopping 
# - Used also to update AR_scheduler (increase AR iterations) if 'AR_iterations' not reached.
patience = 10
minimum_improvement = 0.01 # 0 to not stop 
stopping_metric = 'validation_total_loss'   # training_total_loss                                                     
mode = "min" # MSE best when low  
early_stopping = EarlyStopping(patience = patience,
                               minimum_improvement = minimum_improvement,
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
                                       num_workers = dataloader_settings['num_workers'], 
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

# # Load AR TrainingInfo
# with open(os.path.join(exp_dir,"training_info/AR_TrainingInfo.pickle"), 'rb') as handle:
#     training_info = pickle.load(handle)
    
##------------------------------------------------------------------------.
### Create plots related to training evolution  
## - Plot the loss at all AR iterations (in one figure)
fig, ax = plt.subplots()
for AR_iteration in range(training_info.AR_iterations+1):
    ax = training_info.plot_loss_per_AR_iteration(AR_iteration = AR_iteration, 
                                                  ax = ax,
                                                  linestyle="solid",
                                                  plot_validation = False, 
                                                  plot_labels = True,
                                                  plot_legend = False)
ax.legend(labels=list(range(training_info.AR_iterations + 1)), title="AR iteration", loc='upper right')
plt.gca().set_prop_cycle(None) # Reset color cycling 
for AR_iteration in range(training_info.AR_iterations+1):
    ax = training_info.plot_loss_per_AR_iteration(AR_iteration = AR_iteration, 
                                                  ax = ax,
                                                  linestyle="dashed",
                                                  plot_training = False, 
                                                  plot_labels = False,
                                                  plot_legend = False)  
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

#-------------------------------------------------------------------------.
### - Create predictions 
zarr_fpath = os.path.join(exp_dir, "model_predictions/spatial_chunks/test_pred.zarr")
# zarr_fpath = os.path.join(exp_dir, "/model_predictions/temporal_chunks/test_pred.zarr")
ds_forecasts = AutoregressivePredictions(model = model, 
                                         # Data
                                         da_dynamic = da_test_dynamic,
                                         da_static = da_test_static,              
                                         da_bc = da_test_bc, 
                                         scaler = dynamic_scaler,
                                         # Dataloader options
                                         device = 'cpu',
                                         batch_size = 20,  # number of forecasts per batch
                                         num_workers = 0, 
                                         prefetch_factor = 2, 
                                         prefetch_in_GPU = False,  
                                         pin_memory = True,
                                         asyncronous_GPU_transfer = True,
                                         numeric_precision = training_settings['numeric_precision'], 
                                         # Autoregressive settings
                                         input_k = AR_settings['input_k'], 
                                         output_k = AR_settings['output_k'], 
                                         forecast_cycle = AR_settings['forecast_cycle'],                         
                                         stack_most_recent_prediction = AR_settings['stack_most_recent_prediction'], 
                                         AR_iterations = 5,        # How many time to autoregressive iterate
                                         # Save options 
                                         zarr_fpath = zarr_fpath,  # None --> do not write to disk
                                         rounding = 2,             # Default None. Accept also a dictionary 
                                         compressor = "auto",      # Accept also a dictionary per variable
                                         chunks = "auto",          
                                         timedelta_unit='hour')
    
    ##------------------------------------------------------------------------.
    ### - Run deterministic verification 
    
    ##------------------------------------------------------------------------.
    ### - Create verification summaries 
    
    ##------------------------------------------------------------------------.
    ### - Create verification maps 
    
    ##------------------------------------------------------------------------.
    ### - Create animations 
    
    #-------------------------------------------------------------------------.
