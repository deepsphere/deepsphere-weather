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
from torch import nn, optim

os.chdir("/home/ghiggi/Projects/DeepSphere/")

## DeepSphere-Earth
from modules.utils_config import get_default_settings
from modules.utils_config import read_config_file
from modules.utils_config import write_config_file
from modules.utils_config import get_model_settings
from modules.utils_config import get_training_settings
from modules.utils_config import get_AR_settings
from modules.utils_config import get_dataloader_settings
from modules.utils_config import get_AR_model_dimension_info
from modules.utils_config import get_pytorch_model
from modules.utils_config import get_model_name
from modules.utils_config import pytorch_settings
from modules.utils_config import load_pretrained_model
from modules.utils_config import create_experiment_directories
from modules.utils_config import print_model_description
from modules.training_autoregressive import AutoregressiveTraining
from modules.AR_Scheduler import AR_Scheduler
# from modules.utils_training import EarlyStopping
## Project specific functions
from modules.scalers import temporary_data_scaling
from modules.my_io import readDatasets   
# import modules.architectures as my_architectures
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
# - Loss function  
# - Loss weights / mask ... 
# - LR_scheduler ?
# - kernel_size vs kernel_size_pooling 
# - conv_type: graph --> DeepSphere 

# https://github.com/deepsphere/deepsphere-pytorch/blob/master/scripts/config.example.yml
# https://github.com/deepsphere/deepsphere-pytorch/blob/master/scripts/temporality/run_ar_tc.py
# https://github.com/deepsphere/deepsphere-pytorch/blob/master/scripts/run_ar_tc.py

# Torch precision 
# --> Set torch.set_default_tensor_type() 
#-----------------------------------------------------------------------------.
#######################
# Pytorch Settings ####
#######################

data_dir = "/home/ghiggi/Projects/DeepSphere/ToyData/Healpix_400km/data/" # to change to scratch/... 
#-----------------------------------------------------------------------------.
### Lazy Loading of Datasets 
# TODO 
# - Check that "auto" xarray adapt to nc/zarr chunking 
# - da_static standardization of categorical variables ? 

t_i = time.time()
# - Dynamic data (i.e. pressure and surface levels variables)
ds_dynamic = readDatasets(data_dir=data_dir, feature_type='dynamic')
# - Boundary conditions data (i.e. TOA)
ds_bc = readDatasets(data_dir=data_dir, feature_type='bc')
# - Static features
ds_static = readDatasets(data_dir=data_dir, feature_type='static')
print('- Open the Zarr Store (Lazily): {:.2f}s'.format(time.time() - t_i))

ds_dynamic = ds_dynamic.drop(["level","lat","lon"])
ds_bc = ds_bc.drop(["lat","lon"])
ds_static = ds_static.drop(["lat","lon"])

ds_dynamic = ds_dynamic.isel(time=slice(7,None))  # because bc start at 1980-01-01T07:00:00.000000000

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

model_settings = get_model_settings(cfg)   
AR_settings = get_AR_settings(cfg)
training_settings = get_training_settings(cfg) 
dataloader_settings = get_dataloader_settings(cfg) 

# Current experiment (6h deltat)
AR_settings['input_k'] = [-18, -12,  -6]
AR_settings['output_k'] = [0]
AR_settings['forecast_cycle'] = 6
AR_settings['AR_iterations'] = 10

#-----------------------------------------------------------------------------.
### Apply custom scalers 
# TODO (@GG)
# - scaler = None: Should be provided when data are already standardized
# - scaler = <scaler object to fit> 
# - scaler =  <filepath of a saved scaler object>
# --> Need to way x-scaler.py is ready
# --> scaler.transform(...) 
# - If data already in memory ... standardize on the fly 
# - If data lazy loaded ... delayed standardization 
# - If data lazy loaded (load_memory is False) but need scaler fit --> Stop --> Create before 

# # Meanwhile ....  # Do not remove "blank" line here below !
# da_static, da_training_dynamic, da_validation_dynamic, da_training_bc, da_validation_bc = temporary_data_scaling(da_static = da_static, 
#                                                                                                                  da_training_dynamic = da_training_dynamic, 
#                                                                                                                  da_validation_dynamic = da_validation_dynamic, 
#                                                                                                                  da_training_bc = da_training_bc,
#                                                                                                                  da_validation_bc = da_validation_bc)
# ! Do not remove "blank" line here above !

#-----------------------------------------------------------------------------.
### Split data into train, test and validation set 
### - Defining time split for training 
# training_years = (1980, 2013)
# validation_years = (2015, 2016) 
# test_years = (2017, 2018) 

# t_i = time.time()
# ds_training_dynamic = ds_dynamic.sel(time=slice(*training_years))
# ds_training_bc = ds_bc.sel(time=slice(*training_years))
  
# ds_validation_dynamic = ds_dynamic.sel(time=slice(*validation_years))
# ds_validation_bc = ds_bc.sel(time=slice(*validation_years))
# print('- Splitting data into train and validation set: {:.2f}s'.format(time.time() - t_i))

#-----------------------------------------------------------------------------.
##############  
#### Main ####
############## 
def main(cfg_path):
    """General function for training DeepSphere4Earth models."""
    # TODO: add missing input arguments    

    ##------------------------------------------------------------------------.
    ### Read experiment configuration settings 
    cfg = read_config_file(fpath=cfg_path)
    
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
    dim_info = get_AR_model_dimension_info(AR_settings=AR_settings,
                                           ds_dynamic=ds_dynamic, 
                                           ds_static=ds_static, 
                                           ds_bc=ds_bc)
    model_settings['dim_info'] = dim_info
    print_model_description(dim_info)
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
    ## Generate the (new) model name and its directories 
    else: 
        if model_settings['model_name'] is not None:
            model_name = model_settings['model_name']
        else: 
            model_name = get_model_name(cfg, prefix=None, suffix=None)
            model_settings['model_name'] = model_name
    
    exp_dir = create_experiment_directories(model_dir = model_settings['model_dir'],      
                                            model_name = model_name) 
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
    stopping_rounds = 10 
    stopping_metric = "validation_total_loss"   
    stopping_patience = 20   # number of allowed scores without seeing improvements 
    minimum_increase = 0.001  # 0 to not stop 
    stopping_tolerance = 0    # 0 to not stop 
    early_stopping = EarlyStopping(stopping_patience = stopping_patience, 
                                   stopping_rounds = stopping_rounds,
                                   stopping_tolerance = stopping_tolerance,
                                   minimum_increase = minimum_increase,
                                   stopping_metric = stopping_metric,                                                         
                                   mode = "min") # MSE best when low 
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
    # Train the model 
    training_info = AutoregressiveTraining(model = model,
                                           model_fpath = model_fpath,  
                                           # Loss settings 
                                           criterion = criterion,
                                           optimizer = optimizer,  
                                           LR_scheduler = LR_scheduler, 
                                           AR_scheduler = AR_scheduler,                                
                                           early_stopping = early_stopping,
                                           # Data
                                           ds_training_dynamic = ds_training_dynamic,
                                           ds_validation_dynamic = ds_validation_dynamic,
                                           ds_static = ds_static,              
                                           ds_training_bc = ds_training_bc,         
                                           ds_validation_bc = ds_validation_bc,       
                                           # Dataloader settings
                                           preload_data_in_CPU = dataloader_settings['preload_data_in_CPU'], 
                                           prefetch_in_GPU = dataloader_settings['prefetch_in_GPU'],   
                                           random_shuffle = dataloader_settings['random_shuffle'], 
                                           num_workers = dataloader_settings['num_workers'], 
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
    
    #-------------------------------------------------------------------------.
    # Create plots related to training evolution                     
    
    #-------------------------------------------------------------------------.
    # Create predictions 
    
    #-------------------------------------------------------------------------.
    # Run deterministic verification 
    
    #-------------------------------------------------------------------------.
    # Create verification summaries 
    
    #-------------------------------------------------------------------------.
    # Create verification maps 
    
    #-------------------------------------------------------------------------.
    # Create animations 
    
    #-------------------------------------------------------------------------.