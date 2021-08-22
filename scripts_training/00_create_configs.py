#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 17:28:57 2021

@author: ghiggi
"""
import os
import sys
os.chdir("/home/ghiggi/Projects/deepsphere-weather/configs")
sys.path.append('../')

from modules.utils_config import get_default_settings
from modules.utils_config import read_config_file
from modules.utils_config import write_config_file
##----------------------------------------------------------------------------.
# Get default settings 
cfg = get_default_settings()

# Current experiment (6h deltat)
cfg['AR_settings']['input_k'] = [-18, -12, -6]
cfg['AR_settings']['output_k'] = [0]
cfg['AR_settings']['forecast_cycle'] = 6
cfg['AR_settings']['AR_iterations'] = 6
cfg['AR_settings']['stack_most_recent_prediction'] = True

## Training settings 
cfg['training_settings']["training_batch_size"] = 16
cfg['training_settings']["validation_batch_size"] = 16
cfg['training_settings']["epochs"] = 15
cfg['training_settings']['numeric_precision'] = "float32"
cfg['training_settings']["learning_rate"] = 0.007
cfg['training_settings']["scoring_interval"] = 30
cfg['training_settings']["save_model_each_epoch"] = False

# Reproducibility options
cfg['training_settings']['deterministic_training'] = True
cfg['training_settings']['seed_model_weights'] = 10
cfg['training_settings']['seed_random_shuffling'] = 15

# GPU options
cfg['training_settings']['gpu_training'] = True 
cfg['training_settings']['benchmark_cudnn'] = True  
cfg['training_settings']['dataparallel_training'] = False
cfg['training_settings']['gpu_devices_ids'] = [0]
                         
## Dataloader settings 
cfg['dataloader_settings']["random_shuffling"] = True
cfg['dataloader_settings']["prefetch_in_gpu"] = False
cfg['dataloader_settings']["prefetch_factor"] = 2
cfg['dataloader_settings']["num_workers"] = 8 
cfg['dataloader_settings']["pin_memory"] = False
cfg['dataloader_settings']["asyncronous_gpu_transfer"] = True
cfg['dataloader_settings']["autotune_num_workers"] = False 
cfg['dataloader_settings']["drop_last_batch"] = False  

##----------------------------------------------------------------------------.
### Create general configs for various samplings 
# - Define samplings specifics ('sampling_name': {'sampling': ..., 'resolution: ...}}
dict_samplings = {'Healpix_400km': {'sampling': 'healpix', 
                                    'resolution': 16,
                                    'knn': 20},                     
                  'Equiangular_400km': {'sampling': 'equiangular',
                                        'resolution': [36,72]},         
                  'Equiangular_400km_tropics': {'sampling': 'equiangular',
                                                'resolution': [46,92]},    
                  'Icosahedral_400km': {'sampling': 'icosahedral', 
                                        'resolution': 16,
                                        'knn': 10},                      
                  'O24': {'sampling': 'gauss', 
                          'resolution': 48},  
                  'Cubed_400km': {'sampling': 'cubed', 
                                  'resolution': 24},
                  'Healpix_100km': {'sampling': 'healpix', 
                                    'resolution': 64,
                                    'knn': 20},  
                  }

# - Architecture options 
kernel_size_conv = 3    # default is 3
kernel_size_pooling = 4 # default is 4
 
architecture_names = ["UNetSpherical"]  
knn_list = [20]
gtype_list = ['knn', 'mesh']
pool_methods = ['Max', 'Avg', 'MaxArea', 'MaxVal', 'Interp', 'Learn']

# - Config folder 
config_path = "/home/ghiggi/Projects/deepsphere-weather/configs"

for architecture_name in architecture_names:
    for gtype in gtype_list: 
        for pool_method in pool_methods:
            for sampling_name in dict_samplings.keys():
                custom_cfg = cfg 
                sampling = dict_samplings[sampling_name]['sampling'].lower() 
                custom_cfg['model_settings']['sampling_name'] = sampling_name    
                custom_cfg['model_settings']['sampling'] = dict_samplings[sampling_name]['sampling']  
                custom_cfg['model_settings']['resolution'] = dict_samplings[sampling_name]['resolution']
                if sampling in ['healpix', 'icosahedral']:
                    custom_cfg['model_settings']['knn'] = dict_samplings[sampling_name]['knn']    
                custom_cfg['model_settings']['gtype'] = gtype 
                custom_cfg['model_settings']['architecture_name'] = architecture_name
                custom_cfg['model_settings']['pool_method'] = pool_method
                custom_cfg['model_settings']['kernel_size_conv'] = kernel_size_conv
                custom_cfg['model_settings']['kernel_size_pooling'] = kernel_size_pooling
    
                if pool_method.lower() in ['max','avg']:
                    if sampling not in ['healpix','equiangular']:
                        continue  # do not save config 
                # Create config directory
                tmp_dir = os.path.join(config_path, architecture_name, sampling_name)
                if not os.path.exists(tmp_dir):
                    os.makedirs(tmp_dir)
                # Write config file 
                tmp_config_name = "-".join([pool_method + "Pool", 
                                            # "knn" + str(knn),
                                            'Graph_' + gtype]) + ".json"
                write_config_file(custom_cfg, fpath=os.path.join(tmp_dir, tmp_config_name))
