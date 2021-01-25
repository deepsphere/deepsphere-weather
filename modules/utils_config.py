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
import random
import importlib
import importlib.util
import numpy as np
#-----------------------------------------------------------------------------.
## Key names 
# AR_iterations --> AR_iterations ?

# Keys values:
# conv_type --> Graph  Capital
# pool_method --> Max  Capital

# cfg = get_default_settings()
# cfg['model_settings']['architecture_name'] = "SphericalUNet"
# cfg['model_settings']['sampling'] = "Healpix"
# cfg['model_settings']['resolution'] = "16"
# model_name = get_model_name(cfg)
   
########################
### Default settings ###
########################

def get_default_model_settings():
    """Return some default settings for a DeepSphere model."""
    model_settings = {"pretrained_model_name": None,
                      #pretrained_model_weights_fpath: ??? 
                      "model_name_prefix": None, 
                      "model_name": None, 
                      "model_name_suffix": None, 
                      "knn": 20, 
                      "conv_type": "graph",
                      "pool_method": "max",
                      "ratio": None,
                      "periodic": None,
                      "kernel_size": 3, 
                      "kernel_size_pooling": 4,
                      }
    return model_settings

def get_default_training_settings():
    """Return some default settings for training the model."""
    training_settings = {"epochs": 10,
                         "learning_rate": 0.001,
                         "training_batch_size": 32,
                         "validation_batch_size": 32,
                         "scoring_interval": 10,
                         "save_model_each_epoch": False, 
                         "numeric_precision": "float64",
                         "deterministic_training": False, 
                         "deterministic_training_seed": 100,
                         "benchmark_cuDNN": True,
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
                           "pin_memory": False,   
                           "preload_data_in_CPU": False,  
                           "asyncronous_GPU_transfer": True, 
                           "num_workers": 0,
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
        json.dump(cfg, output_file) 

#-----------------------------------------------------------------------------.
#############################
### Check config file keys ##   
#############################  
def get_model_settings(cfg): 
    """Return model settings from the config file."""
    # Initialize model settings 
    model_settings = {}
    default_model_settings = get_default_model_settings()
    
    mandatory_keys = ['model_dir', 'architecture_name', 'architecture_fpath', 'sampling', 'resolution']
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
    model_settings["model_dir"] = cfg['model_settings'].get("model_dir", None)
    model_settings["architecture_name"] = cfg['model_settings'].get("architecture_name", None)
    model_settings["architecture_fpath"] = cfg['model_settings'].get("architecture_fpath", None)
    model_settings["sampling"] = cfg['model_settings'].get("sampling", None)
    model_settings["resolution"] = cfg['model_settings'].get("resolution", None)
  
    # Stop if some mandatory keys are missing 
    flag_error = False 
    for key in mandatory_keys: 
        if model_settings[key] is None:
            flag_error = True
            print("'{}' is a mandatory key that must be specified in the model settings section of the config file.".format(key))
    if (flag_error is True):
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
# Retrieve input-output dims 
def get_AR_model_dimension_info(AR_settings, ds_dynamic, ds_static=None, ds_bc=None):
    """Retrieve dimension information for AR DeepSphere models."""
    # Dynamic variables 
    dynamic_variables = list(ds_dynamic.data_vars.keys())
    n_dynamic_variables = len(dynamic_variables)
    # Static variables 
    if ds_static is not None:
        static_variables = list(ds_static.data_vars.keys()) 
        n_static_variables = len(static_variables)
    else:
        static_variables = []
        n_static_variables = 0
    # Boundary condition variables     
    if ds_bc is not None:
        bc_variables = list(ds_bc.data_vars.keys())  
        n_bc_variables = len(bc_variables)
    else: 
        bc_variables = []
        n_bc_variables = 0
    ##------------------------------------------------------------------------. 
    # Define feature dimensions 
    input_feature_dim = n_static_variables + n_bc_variables + n_dynamic_variables 
    output_feature_dim = n_dynamic_variables
    input_features = static_variables + bc_variables + dynamic_variables                     
    output_features = dynamic_variables
    ##------------------------------------------------------------------------. 
    # Define time dimension 
    input_time_dim = len(AR_settings['input_k']) 
    output_time_dim = len(AR_settings['output_k']) 
    ##------------------------------------------------------------------------. 
    # Define number of nodes 
    input_node_dim = len(ds_dynamic['node'])
    output_node_dim = len(ds_dynamic['node'])
    ##------------------------------------------------------------------------. 
    # Define dimension order
    dim_order = ['sample', 'time', 'node', 'feature'] 
    ##------------------------------------------------------------------------. 
    # Create dictionary with dimension infos 
    dim_info = {'input_feature_dim': input_feature_dim,
                'output_feature_dim': output_feature_dim,
                'input_features': input_features,
                'output_features': output_features,
                'input_time_dim': input_time_dim,
                'output_time_dim': output_time_dim,
                'input_node_dim': input_node_dim,
                'output_node_dim': output_node_dim,
                'dim_order': dim_order,
                }
    ##------------------------------------------------------------------------. 
    return dim_info

def get_pytorch_model(model_settings):
    """Define a DeepSphere model based on general architecture structure.
       
    The architecture structure must be define in a custom python file.
    
    Model settings dictionary must contain two mandatory keys:
    'architecture_fpath' and 'architecture_name'.
    
    The mandatory key 'architecture_fpath' must indicate the filepath of 
    the python file where the architecture is defined.
    The mandatory key 'architecture_name' represent the class name of
    the DeepSphere architecture to use.
    
    """
    # Retrieve main model info 
    sampling = model_settings['sampling']
    architecture_name = model_settings['architecture_name']
    architecture_fpath = model_settings['architecture_fpath']  
    print('- Defining model {} for {} sampling.'.format(architecture_name, sampling))
    
    ##------------------------------------------------------------------------.
    # Retrieve file paths
    MODULE_PATH = os.path.dirname(architecture_fpath)
    MODULE_INIT_PATH = os.path.join(MODULE_PATH, "__init__.py")
    MODULE_NAME = os.path.basename(architecture_fpath).split(".")[0]
    
    # MODULE_PATH = "/home/ghiggi/Projects/DeepSphere/modules/"
    # MODULE_INIT_PATH = "/home/ghiggi/Projects/DeepSphere/modules/__init__.py" 
    # MODULE_NAME = "architectures"   
    
    ##------------------------------------------------------------------------.
    # Import custom architecture.py
    sys.path.append(MODULE_PATH)
    module = __import__(MODULE_NAME)
    DeepSphereModelClass = getattr(module, architecture_name)
    
    ##------------------------------------------------------------------------.
    # Import custom architecture.py  
    # spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_INIT_PATH)
    # module = importlib.util.module_from_spec(spec)
    # sys.modules[spec.name] = module  # bind relative imports 
    # spec.loader.exec_module(module)
    # getattr(module, "UNetSpherical")
    
    # Import custom architecture.py  
    # import modules.architectures as module
    
    ##------------------------------------------------------------------------.
    # Retrieve required model arguments
    model_keys = ['dim_info','resolution', 'conv_type', 'kernel_size', 'sampling',
                  'knn', 'pool_method', 'kernel_size_pooling', 'periodic', 'ratio']
    model_args = {k: model_settings[k] for k in model_keys}
    
    ##------------------------------------------------------------------------.
    # Define DeepSphere model 
    model = DeepSphereModelClass(**model_args)  
    
    ##------------------------------------------------------------------------.
    return model 

def load_pretrained_model(model, model_settings):
    """Load a pre-trained pytorch model."""
    # TODO: add pretrained_model_weights_fpath key ? 
    # --> or change pretrained_model_name to pretrained_model_weights_fpath ? 
    model_fname = model_settings['pretrained_model_name'] + '.h5'
    model_fpath = os.path.join(model_settings['model_dir'], 'models_weights', model_fname)
    state = torch.load(model_fpath)
    model.load_state_dict(state, strict=False)

#-----------------------------------------------------------------------------.
######################### 
### Pytorch settings ####
#########################
def set_pytorch_deterministic(seed=100):
    """Set seeds for deterministic training with pytorch."""
    # TODO
    # - https://pytorch.org/docs/stable/generated/torch.set_deterministic.html#torch.set_deterministic
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    return 

def get_torch_dtype(numeric_precision): 
    """Provide torch dtype based on numeric precision string."""
    dtypes = {'float64': torch.float64,
              'float32': torch.float32,
              'float16': torch.float16,
              'bfloat16': torch.bfloat16
              }
    return dtypes[numeric_precision]

# TODO set_numeric_precision 

def pytorch_settings(training_settings):
    """Set training options with pytorch."""
    # Retrieve pytorch settings options
    deterministic_training = training_settings['deterministic_training'] 
    deterministic_training_seed = training_settings['deterministic_training_seed'] 
    benchmark_cuDNN = training_settings['benchmark_cuDNN'] 
    
    ##------------------------------------------------------------------------.
    # Set options for deterministic training 
    if deterministic_training is True:
        set_pytorch_deterministic(seed=deterministic_training_seed)
    
    ##------------------------------------------------------------------------.
    # If requested, enable the inbuilt cudnn auto-tuner 
    # --> Find the best algorithm to use with the available hardware.
    # --> Usually leads to faster runtime.
    if benchmark_cuDNN is True:
        torch.backends.cudnn.benchmark = True
    else: 
        torch.backends.cudnn.benchmark = False
    
    ##------------------------------------------------------------------------.
    # Return the device to make the pytorch architecture working both on CPU and GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = 'cpu'
    
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
        conv_type = cfg['model_settings']["conv_type"]
        pool_method = cfg['model_settings']["pool_method"]
        numeric_precision = cfg['training_settings']["numeric_precision"]
        AR_iterations = cfg['AR_settings']["AR_iterations"]
        # Create model name 
        model_name = "-".join([architecture_name,
                               sampling,
                               str(resolution),
                               "k" + str(knn),
                               "Conv" + conv_type, 
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
 

def create_experiment_directories(model_dir, model_name): 
    """Create the required directory for a specific DeepSphere model."""
    # Check if the experiment directory already exists 
    exp_dir = os.path.join(model_dir, model_name)
    if os.path.exists(exp_dir):
        raise ValueError("The directory {} already exists.\
                         Please delete such directory manually or: \
                         - specify 'model_name' in model_settings \
                         - specify 'model_prefix' and/or 'model_suffix' in model_settings".format(exp_dir))
    
    ##------------------------------------------------------------------------.
    # Define standard directories 
    model_weights_dir = os.path.join(exp_dir, "model_weights")
    figures_dir = os.path.join(exp_dir, "figures")
    model_predictions_dir = os.path.join(exp_dir, "model_predictions")
    spatial_chunks_dir = os.path.join(model_predictions_dir, "spatial_chunks")
    temporal_chunks_dir = os.path.join(model_predictions_dir, "temporal_chunks")
    model_skills_dir = os.path.join(exp_dir, "model_skills")
    training_info_dir = os.path.join(exp_dir, "training_info")
    
    ##------------------------------------------------------------------------.
    # Create directories     
    os.makedirs(model_weights_dir, exist_ok=False)
    os.makedirs(figures_dir, exist_ok=False) 
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
def pretty_printing(d, indent=0):
    """Pretty pritting of nested dictionaries."""
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty_printing(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))
         
def print_model_description(cfg, dim_info=None):
    """Pretty printing of model settings."""
    if dim_info is not None: 
        cfg['dim_info'] = dim_info 
    pretty_printing(cfg)
    