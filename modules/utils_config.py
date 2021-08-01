#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 13:23:20 2021

@author: ghiggi
"""
import os
import sys
import json
import torch
import pickle
import shutil
import numpy as np

from modules.utils_torch import set_pytorch_deterministic
from modules.utils_torch import set_pytorch_numeric_precision
#-----------------------------------------------------------------------------.
########################
### Default settings ###
########################

def get_default_model_settings():
    """Return some default settings for a DeepSphere model."""
    model_settings = {"pretrained_model_name": None,
                      "model_name_prefix": None, 
                      "model_name": None, 
                      "model_name_suffix": None, 
                      "knn": 20, 
                      "pool_method": "Max",
                      "kernel_size_conv": 3, 
                      "kernel_size_pooling": 4,
                      }
    return model_settings

def get_default_training_settings():
    """Return some default settings for training the model."""
    training_settings = {"epochs": 10,
                         "AR_training_strategy": "AR",
                         "learning_rate": 0.001,
                         "training_batch_size": 32,
                         "validation_batch_size": 32,
                         "scoring_interval": 10,
                         "save_model_each_epoch": False, 
                         "numeric_precision": "float64",
                         "deterministic_training": False, 
                         "deterministic_training_seed": 100,
                         "benchmark_cuDNN": True,
                         "GPU_training": True, 
                         "GPU_devices_ids": [0], 
                         "DataParallel_training": False, 
                         }
    return training_settings

def get_default_AR_settings():
    """Return some default settings for the autoregressive model."""
    AR_settings = {"input_k": [-3,-2,-1], 
                   "output_k": [0],
                   "forecast_cycle": 1,                           
                   "AR_iterations": 2, 
                   "stack_most_recent_prediction": True,
                   }
    return AR_settings

def get_default_dataloader_settings():
    """Return some default settings for the DataLoader."""
    dataloader_settings = {"random_shuffle": True,
                           "drop_last_batch": True, 
                           "prefetch_in_GPU": False, 
                           "prefetch_factor": 2,
                           "pin_memory": False,  
                           "asyncronous_GPU_transfer": True, 
                           "num_workers": 0,
                           "autotune_num_workers": False, 
                           }  
    return dataloader_settings

def get_default_SWAG_settings():
    """Return some default settings for the SWAG model."""
    dataloader_settings = {"SWAG": False,
                           "target_learning_rate": 0.007,
                           "no_cov_mat": False,
                           "max_num_models": 40,
                           "swag_freq": 10,
                           "swa_start": 0,
                           "sampling_scale": 0.1,
                           "nb_samples": 10
                           }  
    return dataloader_settings

def get_default_settings():
    """Return the default config settings."""
    AR_settings = get_default_AR_settings()
    training_settings = get_default_training_settings()
    model_settings = get_default_model_settings()   
    dataloader_settings = get_default_dataloader_settings()
    default_settings = {"model_settings": model_settings, 
                        "dataloader_settings": dataloader_settings,
                        "training_settings": training_settings, 
                        "AR_settings": AR_settings,
                        }
    return default_settings

#-----------------------------------------------------------------------------.
######################## 
### I/O config file ####
########################
def read_config_file(fpath):
    """Create a dictionary of settings based on the json config file."""
    with open(fpath) as input_file:
        cfg = json.load(input_file)   
    return cfg

def write_config_file(cfg, fpath):
    """Write a json config file from the python dictionary config file."""
    with open(fpath, "w") as output_file:  
        json.dump(cfg, output_file, indent=4) 

#-----------------------------------------------------------------------------.
#############################
### Check config file keys ##   
#############################  
def get_model_settings(cfg): 
    """Return model settings from the config file."""
    # Initialize model settings 
    model_settings = {}
    default_model_settings = get_default_model_settings()
    
    mandatory_keys = ['architecture_name', 'sampling', 'resolution', "sampling_name"]
    optional_keys = list(default_model_settings.keys())
    available_keys = mandatory_keys + optional_keys
  
    # Check that only correct keys are specified 
    cfg_keys = np.array(list(cfg['model_settings'].keys())) 
    invalid_keys = cfg_keys[np.isin(cfg_keys, available_keys, invert=True)]
    if len(invalid_keys) > 0: 
        for key in invalid_keys: 
            print("'{}' is an unvalid model setting key!".format(key))
        raise ValueError('Specify only correct model setting keys in the config file!')        
    
    # Retrieve mandatory model settings  
    model_settings["architecture_name"] = cfg['model_settings'].get("architecture_name", None)
    model_settings["sampling"] = cfg['model_settings'].get("sampling", None)
    model_settings["resolution"] = cfg['model_settings'].get("resolution", None)
    model_settings["sampling_name"] = cfg['model_settings'].get("sampling", None)
   
    # Stop if some mandatory keys are missing 
    flag_error = False 
    for key in mandatory_keys: 
        if model_settings[key] is None:
            flag_error = True
            print("'{}' is a mandatory key that must be specified in the model settings section of the config file.".format(key))
    if flag_error:
        raise ValueError('Specify the mandatory model settings keys in the config file!')    
    
    # Retrieve optional model settings  
    for key in optional_keys:
        model_settings[key] = cfg['model_settings'].get(key, default_model_settings[key])

    # Return model settings 
    return model_settings
 
def get_training_settings(cfg):
    """Return training settings from the config file."""
    # Initialize training settings
    training_settings = {}
    default_training_settings = get_default_training_settings()  
    available_keys = list(default_training_settings.keys())
    
    # Check that only correct keys are specified 
    cfg_keys = np.array(list(cfg['training_settings'].keys())) 
    invalid_keys = cfg_keys[np.isin(cfg_keys, available_keys, invert=True)]
    if len(invalid_keys) > 0: 
        for key in invalid_keys: 
            print("'{}' is an unvalid training setting key!".format(key))
        raise ValueError('Specify only correct training setting keys in the config file!')        
    
    # Retrieve optional training settings  
    for key in available_keys:
        training_settings[key] = cfg['training_settings'].get(key, default_training_settings[key])
    
    # Special checks 
    if not isinstance(training_settings['GPU_devices_ids'], list):
        training_settings['GPU_devices_ids'] = [training_settings['GPU_devices_ids']]
        
    if not training_settings['GPU_training']:    
        if training_settings['DataParallel_training']:
            print("DataParallel training is available only on GPUs!")
            training_settings['DataParallel_training'] = False
    
    # Return training settings 
    return training_settings

def get_dataloader_settings(cfg):
    """Return dataloader settings from the config file."""
    # Initialize dataloader settings
    dataloader_settings = {}
    default_dataloader_settings = get_default_dataloader_settings()  
    available_keys = list(default_dataloader_settings.keys())
    
    # Check that only correct keys are specified 
    cfg_keys = np.array(list(cfg['dataloader_settings'].keys())) 
    invalid_keys = cfg_keys[np.isin(cfg_keys, available_keys, invert=True)]
    if len(invalid_keys) > 0: 
        for key in invalid_keys: 
            print("'{}' is an unvalid dataloader setting key!".format(key))
        raise ValueError('Specify only correct dataloader setting keys in the config file!')        
    
    # Retrieve optional dataloader settings  
    for key in available_keys:
        dataloader_settings[key] = cfg['dataloader_settings'].get(key, default_dataloader_settings[key])
    
    # Return dataloader settings 
    return dataloader_settings

def get_AR_settings(cfg):
    """Return AR settings from the config file."""
    # Initialize AR settings
    AR_settings = {}
    default_AR_settings = get_default_AR_settings()  
    available_keys = list(default_AR_settings.keys())
    
    # Check that only correct keys are specified 
    cfg_keys = np.array(list(cfg['AR_settings'].keys())) 
    invalid_keys = cfg_keys[np.isin(cfg_keys, available_keys, invert=True)]
    if len(invalid_keys) > 0: 
        for key in invalid_keys: 
            print("'{}' is an unvalid AR setting key!".format(key))
        raise ValueError('Specify only correct AR setting keys in the config file!')        
    
    # Retrieve optional AR settings  
    for key in available_keys:
        AR_settings[key] = cfg['AR_settings'].get(key, default_AR_settings[key])
    
    # Ensure input_k and output_k are list 
    if not isinstance(AR_settings['input_k'], list):
        AR_settings['input_k'] = [AR_settings['input_k']]
    if not isinstance(AR_settings['output_k'], list):    
        AR_settings['output_k'] = [AR_settings['output_k']]    
        
    # Return AR settings 
    return AR_settings

def get_SWAG_settings(cfg):
    """Return SWAG settings from the config file."""
    # Initialize AR settings
    SWAG_settings = {}
    default_SWAG_settings = get_default_SWAG_settings()  
    available_keys = list(default_SWAG_settings.keys())
    
    # Check that only correct keys are specified 
    cfg_keys = np.array(list(cfg['SWAG_settings'].keys())) 
    invalid_keys = cfg_keys[np.isin(cfg_keys, available_keys, invert=True)]
    if len(invalid_keys) > 0: 
        for key in invalid_keys: 
            print("'{}' is an unvalid SWAG setting key!".format(key))
        raise ValueError('Specify only correct SWAG setting keys in the config file!')        
    
    # Retrieve optional AR settings  
    for key in available_keys:
        SWAG_settings[key] = cfg['SWAG_settings'].get(key, default_SWAG_settings[key])   
        
    # Return AR settings 
    return SWAG_settings

#-----------------------------------------------------------------------------.
#################################
### Checks config key values ####
#################################
def check_numeric_precision(numeric_precision): 
    """Check numeric precision argument."""
    if not isinstance(numeric_precision, str):
        raise TypeError("Specify 'numeric_precision' as a string")
    if not [numeric_precision] in ['float64', 'float32','float16','bfloat16']:
        raise ValueError("Valid 'numeric precision' are: 'float64', 'float32','float16' and 'bfloat16'")

# TODO: others     
        
#-----------------------------------------------------------------------------.
########################
### Model definition ###
########################
# def get_pytorch_model(module_with_custom_models, 
#                       model_settings, 
#                       training_settings):
#  

# def get_pytorch_model(module_with_custom_models, model_settings, training_settings):
#     """Define a DeepSphere model based on general architecture structure.
       
#     The architecture structure must be define in a custom python file.
    
#     Model settings dictionary must contain two mandatory keys:
#     'architecture_fpath' and 'architecture_name'.
    
#     The mandatory key 'architecture_fpath' must indicate the filepath of 
#     the python file where the architecture is defined.
#     The mandatory key 'architecture_name' represent the class name of
#     the DeepSphere architecture to use.
    
#     """
#     # Retrieve main model info 
#     sampling = model_settings['sampling']
#     architecture_name = model_settings['architecture_name']
#     print('- Defining model {} for {} sampling.'.format(architecture_name, sampling))
    
#     ##------------------------------------------------------------------------.
#     # Retrieve file paths
#     MODULE_PATH = os.path.dirname(architecture_fpath)
#     # MODULE_INIT_PATH = os.path.join(MODULE_PATH, "__init__.py")
#     MODULE_NAME = os.path.basename(architecture_fpath).split(".")[0]
    
#     # MODULE_PATH = "/home/ghiggi/Projects/DeepSphere/modules/"
#     # MODULE_INIT_PATH = "/home/ghiggi/Projects/DeepSphere/modules/__init__.py" 
#     # MODULE_NAME = "architectures"   
    
#     ##------------------------------------------------------------------------.
#     # Import custom architecture.py
#     sys.path.append(MODULE_PATH)
#     module = __import__(MODULE_NAME)
#     DeepSphereModelClass = getattr(module, architecture_name)
    
#     ##------------------------------------------------------------------------.
#     # Import custom architecture.py  
#     # spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_INIT_PATH)
#     # module = importlib.util.module_from_spec(spec)
#     # sys.modules[spec.name] = module  # bind relative imports 
#     # spec.loader.exec_module(module)
#     # getattr(module, "UNetSpherical")
    
#     # Import custom architecture.py  
#     # import modules.architectures as module
    
#     # DeepSphereModelClass = getattr(module_with_custom_models, model_settings['architecture_name'])
#     ##------------------------------------------------------------------------.
#     # Retrieve required model arguments
#     model_keys = ['dim_info', 'sampling', 'resolution',
#                   'knn', 'kernel_size_conv',
#                   'pool_method', 'kernel_size_pooling']
#     model_args = {k: model_settings[k] for k in model_keys}
#     model_args['numeric_precision'] = training_settings['numeric_precision']
#     # - Define DeepSphere model 
#     model = DeepSphereModelClass(**model_args)       
#     return model 

##----------------------------------------------------------------------------.

def load_pretrained_model(model, exp_dir, model_name):
    """Load a pre-trained pytorch model."""
    model_fpath = os.path.join(exp_dir, model_name, 'model_weights', "model.h5")
    state = torch.load(model_fpath)
    model.load_state_dict(state, strict=False)

def load_pretrained_ar_scheduler(exp_dir, model_name):
    """Load a pre-trained AR scheduler."""
    training_info_fpath = os.path.join(exp_dir, model_name, 'training_info', 'AR_TrainingInfo.pickle')
    with open(training_info_fpath, 'rb') as f:
        training_info = pickle.load(f)
    ar_scheduler = pickle.loads(training_info.AR_scheduler)

    return ar_scheduler

#-----------------------------------------------------------------------------.
######################### 
### Pytorch settings ####
#########################
def set_pytorch_settings(training_settings):
    """Set training options with pytorch."""
    # Retrieve pytorch settings options
    deterministic_training = training_settings['deterministic_training'] 
    deterministic_training_seed = training_settings['deterministic_training_seed'] 
    benchmark_cuDNN = training_settings['benchmark_cuDNN'] 
    GPU_training = training_settings['GPU_training'] 
    GPU_devices_ids = training_settings['GPU_devices_ids']     
    numeric_precision = training_settings['numeric_precision']
    ##------------------------------------------------------------------------.
    # Set options for deterministic training 
    if deterministic_training:
        set_pytorch_deterministic(seed=deterministic_training_seed)
    
    ##------------------------------------------------------------------------.
    # If requested, autotunes to the best cuDNN kernel (for performing convolutions)
    # --> Find the best algorithm to use with the available hardware.
    # --> Usually leads to faster runtime. 
    if benchmark_cuDNN:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    else: 
        torch.backends.cudnn.benchmark = False
    
    ##------------------------------------------------------------------------.
    # Return the device to make the pytorch architecture working both on CPU and GPU
    if GPU_training:
        if torch.cuda.is_available():
            device = torch.device(GPU_devices_ids[0])
        else:
            print("- GPU is not available. Switching to CPU !")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')   

    #------------------------------------------------------------------------.
    # Set numeric precision 
    # set_pytorch_numeric_precision(numeric_precision=numeric_precision, device=device)

    #------------------------------------------------------------------------.
    # Return the torch device 
    return device

#-----------------------------------------------------------------------------.
############################
### Experiment structure ###
############################   
def get_model_name(cfg):
    """Create a model name based on the config settings."""
    ##------------------------------------------------------------------------.
    # Retrieve model_name, suffix and prefix 
    model_name = cfg['model_settings']['model_name']
    model_name_prefix = cfg['model_settings']["model_name_prefix"]
    model_name_suffix = cfg['model_settings']["model_name_suffix"]
    
    ##------------------------------------------------------------------------.
    # Define model name based on config settings if not specified
    if (model_name is None): 
        # Retrieve important "discriminatory" settings 
        architecture_name = cfg['model_settings']["architecture_name"]
        sampling = cfg['model_settings']["sampling"]
        resolution = cfg['model_settings']["resolution"]
        knn = cfg['model_settings']["knn"]
        pool_method = cfg['model_settings']["pool_method"]
        AR_training_strategy = cfg['training_settings']["AR_training_strategy"]
        numeric_precision = cfg['training_settings']["numeric_precision"]
        AR_iterations = cfg['AR_settings']["AR_iterations"]
        # Create model name 
        model_name = "-".join([AR_training_strategy,
                               architecture_name,
                               sampling,
                               str(resolution),
                               "k" + str(knn),
                               pool_method + "Pooling",
                               numeric_precision,
                               "AR" + str(AR_iterations),
                               ])
        
    ##------------------------------------------------------------------------.
    # Add prefix and suffix if specified 
    if model_name_prefix is not None: 
        model_name = "-".join([model_name_prefix, model_name])
    if model_name_suffix is not None: 
        model_name = "-".join([model_name, model_name_suffix])
        
    ##------------------------------------------------------------------------.
    # Update cfg model_name 
    cfg['model_settings']['model_name'] = model_name
    
    ##------------------------------------------------------------------------.
    # Return model name 
    return model_name 
 

def create_experiment_directories(exp_dir, model_name, force=False): 
    """Create the required directory for a specific DeepSphere model."""
    # Check if the experiment directory already exists 
    exp_dir = os.path.join(exp_dir, model_name)
    if os.path.exists(exp_dir):
        if force: 
            shutil.rmtree(exp_dir)
        else:
            raise ValueError("The directory {} already exists.\
                             force=True in create_experiment_directories() will delete content of the existing directory.\
                             Please delete such directory manually or: \
                                 - specify 'model_name' in model_settings \
                                 - specify 'model_prefix' and/or 'model_suffix' in model_settings".format(exp_dir))
    
    ##------------------------------------------------------------------------.
    # Define standard directories 
    model_weights_dir = os.path.join(exp_dir, "model_weights")
    figures_dir = os.path.join(exp_dir, "figs")
    figs_skills_dir = os.path.join(figures_dir, "skills")
    figs_training_info_dir = os.path.join(figures_dir, "training_info")
    model_predictions_dir = os.path.join(exp_dir, "model_predictions")
    spatial_chunks_dir = os.path.join(model_predictions_dir, "spatial_chunks")
    temporal_chunks_dir = os.path.join(model_predictions_dir, "temporal_chunks")
    model_skills_dir = os.path.join(exp_dir, "model_skills")
    training_info_dir = os.path.join(exp_dir, "training_info")
    
    ##------------------------------------------------------------------------.
    # Create directories     
    os.makedirs(model_weights_dir, exist_ok=False)
    os.makedirs(figs_skills_dir, exist_ok=False) 
    os.makedirs(figs_training_info_dir, exist_ok=False) 
    os.makedirs(model_skills_dir, exist_ok=False) 
    os.makedirs(training_info_dir, exist_ok=False) 
    os.makedirs(spatial_chunks_dir, exist_ok=False) 
    os.makedirs(temporal_chunks_dir, exist_ok=False) 
   
    ##------------------------------------------------------------------------.
    # Return the experiment directory
    return exp_dir      

##------------------------------------------------------------------------.
#########################
### Print model info ####
#########################
def pretty_printing(d, indent=0, indent_factor=2):
    """Pretty pritting of nested dictionaries."""
    for key, value in d.items():
        print((' '*indent*indent_factor) + "- " + str(key) + ":", end="")
        if isinstance(value, dict):
            print(end="\n")
            pretty_printing(value, indent=indent+1, indent_factor=indent_factor)
        else:
            print(' ' + str(value), end="\n")

# def pretty_printing(d, indent=0):
#     """Pretty pritting of nested dictionaries."""
#     for key, value in d.items():
#         print('\t' * indent + str(key))
#         if isinstance(value, dict):
#             pretty_printing(value, indent+1)
#         else:
#             print('\t' * (indent+1) + str(value))

def print_dim_info(dim_info):
    """Pretty printing of tensor dimension information."""
    print("- Input-Output Tensors characteristics:")
    pretty_printing(dim_info, indent=1, indent_factor=2)  
    
def print_model_description(cfg, dim_info=None):
    """Pretty printing of experiment settings."""
    print("- Experiment settings:")
    if dim_info is not None: 
        cfg['dim_info'] = dim_info 
    pretty_printing(cfg, indent=1, indent_factor=2)
    