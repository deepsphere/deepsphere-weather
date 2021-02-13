#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:31:25 2021

@author: ghiggi
"""
import time
import torch
import numpy as np
import pdb
from torch import optim
from functools import partial 
from torch.utils.data import Dataset, DataLoader

from modules.utils_config import get_default_settings
from modules.utils_config import get_model_settings
from modules.utils_config import get_training_settings
from modules.utils_config import get_AR_settings
from modules.utils_config import get_dataloader_settings

from modules.utils_autoregressive import get_dict_stack_info
from modules.utils_autoregressive import get_first_valid_idx
from modules.utils_autoregressive import get_last_valid_idx
from modules.utils_autoregressive import get_dict_Y
from modules.utils_autoregressive import get_dict_X_dynamic
from modules.utils_autoregressive import get_dict_X_bc    
from modules.utils_autoregressive import check_input_k
from modules.utils_autoregressive import check_output_k 
from modules.utils_autoregressive import check_AR_settings
 
from modules.utils_io import check_AR_Datasets
from modules.utils_io import is_dask_DataArray
from modules.utils_torch import get_torch_dtype
from modules.utils_torch import check_torch_device

from modules.dataloader_autoregressive import AutoregressiveDataset
from modules.dataloader_autoregressive import AutoregressiveDataLoader
from modules.dataloader_autoregressive import get_AR_batch
from modules.dataloader_autoregressive import remove_unused_Y
from modules.dataloader_autoregressive import cylic_iterator

from modules.utils_io import check_AR_DataArrays
from modules.utils_training import AR_TrainingInfo



cfg = get_default_settings()
cfg['model_settings']['architecture_name'] = "UNetSpherical"
cfg['model_settings']['architecture_fpath'] = "/home/ghiggi/Projects/DeepSphere/modules/architectures.py"
cfg['model_settings']['sampling'] = "Healpix"
cfg['model_settings']['resolution'] = 16
cfg['model_settings']['exp_dir'] = "/home/ghiggi/Projects/DeepSphere/models"

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

cfg['dataloader_settings']["prefetch_in_GPU"] = False
cfg['dataloader_settings']["prefetch_factor"] = 2
cfg['dataloader_settings']["pin_memory"] = True
cfg['dataloader_settings']["asyncronous_GPU_transfer"] = True
cfg['dataloader_settings']["num_workers"] = 0  
cfg['dataloader_settings']["drop_last_batch"] = False  

model_settings = get_model_settings(cfg)   
AR_settings = get_AR_settings(cfg)
training_settings = get_training_settings(cfg) 
dataloader_settings = get_dataloader_settings(cfg) 


num_workers = dataloader_settings['num_workers'] 
prefetch_factor = dataloader_settings['prefetch_factor']   
prefetch_in_GPU = dataloader_settings['prefetch_in_GPU'] 
drop_last_batch = dataloader_settings['drop_last_batch']     
random_shuffle = dataloader_settings['random_shuffle']
pin_memory = dataloader_settings['pin_memory']
asyncronous_GPU_transfer = dataloader_settings['asyncronous_GPU_transfer']
# Autoregressive settings  
input_k = AR_settings['input_k']
output_k = AR_settings['output_k']
forecast_cycle = AR_settings['forecast_cycle']
AR_iterations = AR_settings['AR_iterations']
stack_most_recent_prediction = AR_settings['stack_most_recent_prediction']
# Training settings 
numeric_precision = training_settings['numeric_precision']
training_batch_size = training_settings['training_batch_size']
validation_batch_size = training_settings['validation_batch_size']
epochs = training_settings['epochs']
scoring_interval = training_settings['scoring_interval']
save_model_each_epoch = training_settings['save_model_each_epoch']

autotune_num_workers = True, 
shuffle = random_shuffle
rounding = 2
batch_size = 30
chunks = "auto" 
compressor = "auto"
timedelta_unit = 'hour'
device = 'cpu'
num_workers = 1 
training_num_workers = 2
validation_num_workers = 2


# zarr_fpath = os.path.join(exp_dir, "model_predictions/spatial_chunks/test_pred.zarr")
 
# preload_data_in_CPU = True
# random_shuffle = True
# num_workers = 0
# pin_memory = False
# asyncronous_GPU_transfer = True
# # Autoregressive settings  
# input_k = [-3,-2,-1]
# output_k = [0]
# forecast_cycle = 1                         
# AR_iterations = 2
# stack_most_recent_prediction = True
# # Training settings 
# learning_rate = 0.001 
# batch_size = 128
# epochs = 10
# training_info = None
# numeric_precision = "float32"
# # GPU settings 
# device = 'cpu'

# num_workers = 0

