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
from modules.predictions_autoregressive import rechunk_forecasts_for_verification
from modules.predictions_autoregressive import reshape_forecasts_for_verification

from modules.utils_torch import profile_model
from modules.utils_torch import summarize_model

from modules.AR_Scheduler import AR_Scheduler
from modules.early_stopping import EarlyStopping

## Project specific functions
from modules.xscaler import GlobalStandardScaler  # TemporalStandardScaler
from modules.xscaler import LoadScaler
 
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
##-----------------------------------------------------------------------------.
## Tensor dimension order !!!
# - For autoregressive: 
# ['sample','time', 'node', ..., 'feature']
# - For classical 
# ['sample','node', ... , 'feature]
# ['sample','time', 'node','feature']

# - Dimension order should be generalized (to a cfg setting) ?
# - ['time', 'node', 'level', 'ensemble', 'feature'] 

# Feature dimension data order = [static, bc, dynamic]
##-----------------------------------------------------------------------------.
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
# #####################
# Pytorch Settings ####
# #####################
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

##-----------------------------------------------------------------------------.
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
cfg['model_settings']['sampling'] = "Healpix"
cfg['model_settings']['resolution'] = 16


# Current experiment (6h deltat)
cfg['AR_settings']['input_k'] = [-18, -12, -6]
cfg['AR_settings']['output_k'] = [0]
cfg['AR_settings']['forecast_cycle'] = 6
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

cfg['dataloader_settings']["prefetch_in_GPU"] = False
cfg['dataloader_settings']["prefetch_factor"] = 2
cfg['dataloader_settings']["pin_memory"] = True
cfg['dataloader_settings']["asyncronous_GPU_transfer"] = True
cfg['dataloader_settings']["num_workers"] = 1    # os.cpu_count()  
cfg['dataloader_settings']["drop_last_batch"] = False  

##-----------------------------------------------------------------------------.
# ### Scale data with xscaler 
# dynamic_scaler = GlobalStandardScaler(data=ds_dynamic)
# dynamic_scaler.fit()
# # dynamic_scaler.save("/home/ghiggi/dynamic_scaler.nc")

# bc_scaler = GlobalStandardScaler(data=ds_bc)
# bc_scaler.fit()
# # bc_scaler.save("/home/ghiggi/bc_scaler.nc")

# static_scaler = GlobalStandardScaler(data=ds_static)
# static_scaler.fit()
# # static_scaler.save("/home/ghiggi/static_scaler.nc")

# ## Load scaler 
# # dynamic_scaler = LoadScaler("/home/ghiggi/dynamic_scaler.nc")
# # bc_scaler = LoadScaler("/home/ghiggi/bc_scaler.nc")
# # static_scaler = LoadScaler("/home/ghiggi/static_scaler.nc")

# scaler = SequentialScaler(dynamic_scaler, bc_scaler, static_scaler)

# ds_dynamic_std = dynamic_scaler.transform(ds_dynamic) # delayed computation
# ds_bc_std = bc_scaler.transform(ds_bc)           # delayed computation
# ds_static_std = static_scaler.transform(ds_static)   # delayed computation

#ds_dynamic1 = dynamic_scaler.inverse_transform(ds_dynamic_std.compute()).compute()
#xr.testing.assert_equal(ds_dynamic, ds_dynamic1) 

##-----------------------------------------------------------------------------.
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

##-----------------------------------------------------------------------------.
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
##-----------------------------------------------------------------------------.

### GPU options 
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="2,4"
# os.environ["CUDA_LAUNCH_BLOCKING"] = 1

# ############  
#### Main ####
# ############ 
# def main(cfg_path):
# """General function for training DeepSphere4Earth models."""
# TODO: add missing input arguments    

exp_dir = "/home/ghiggi/Projects/DeepSphere/models" # "/data/weather_prediction/experiments"
architecture_fpath = "/home/ghiggi/Projects/weather_prediction/modules/models.py"

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
### Define pyTorch settings and get pytorch device 
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
model_settings['architecture_name'] = "UNetDiffSpherical"
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
summarize_model(model=model, 
                input_size=dim_info['input_shape'],  
                batch_size=training_settings["training_batch_size"], 
                device=device)

### Pytorch profiler 
prof = profile_model(model=model,
                     input_size=dim_info['input_shape'], 
                     batch_size=training_settings["training_batch_size"], 
                     device=device)

# print(prof.key_averages(group_by_input_shape=False).table(sort_by="cpu_time", row_limit=10))
# print(prof.key_averages(group_by_input_shape=False).table(sort_by="cpu_time_total", row_limit=10))
# print(prof.key_averages(group_by_input_shape=False).table(sort_by="cpu_memory_usage", row_limit=10))
# print(prof.key_averages(group_by_input_shape=False).table(sort_by="self_cpu_memory_usage", row_limit=10))

# print(prof.key_averages(group_by_input_shape=False).table(sort_by="cuda_time", row_limit=10))
# print(prof.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", row_limit=10))
# print(prof.key_averages(group_by_input_shape=False).table(sort_by="cuda_memory_usage", row_limit=10))
# print(prof.key_averages(group_by_input_shape=False).table(sort_by="self_cuda_memory_usage", row_limit=10))
  

 





    
    
 



  