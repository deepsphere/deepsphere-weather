#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 20:01:27 2021

@author: ghiggi
"""
##----------------------------------------------------------------------------.
import os
import sys
sys.path.append('../')
import glob
import shutil 
import tempfile
import subprocess
from modules.utils_config import read_config_file
from modules.utils_config import write_config_file
##----------------------------------------------------------------------------.
# Define fpaths

# data_dir = "/data/weather_prediction/data"
# data_dir = "/ltenas3/DeepSphere/data/preprocessed/ERA5_HRES"
# exp_dir = "/data/weather_prediction/experiments_GG"

data_dir = "/home/ghiggi/Projects/DeepSphere/data/toy_data/ERA5_HRES/"
exp_dir = "/home/ghiggi/Projects/DeepSphere/data/experiments/exp_DeepEnsemble"

project_dir = "/home/ghiggi/Projects/deepsphere-weather"
tmp_dirpath = tempfile.mkdtemp()
cuda = '0'
force = True # If True, if model folder already exist, is replaced.

# Define python routine 
model_name_prefix = "State"
python_training_routine = os.path.join(project_dir,"scripts_training","train_predict_state.py")
python_verification_routine = os.path.join(project_dir,"scripts_training","verify_DeepEnsemble.py")
#-----------------------------------------------------------------------------.
# Define config file path  
cfg_path = os.path.join(project_dir, "configs",
                        "UNetSpherical",
                        "Healpix_400km",
                        "MaxAreaPool-Graph_knn.json")

#-----------------------------------------------------------------------------.
# Define model seed 
model_seed = 10
shuffle_seed = 50
#-----------------------------------------------------------------------------.
### DeepEnsemble (model + data random)
model_suffix =  "WeightInitRandom-ShuffleRandom"
DeepEnsemble_dir = os.path.join(exp_dir,"DeepEnsemble_" + model_suffix)
remove_model_dirs = False
n_models = 15
# - Training 
for i in range(1, n_models+1): 
    # Read general config
    cfg = read_config_file(fpath=cfg_path)
    # Optional configs 
    cfg['training_settings']["deterministic_training"] = True
    cfg['dataloader_settings']["random_shuffling"] = True
    cfg['training_settings']["seed_random_shuffling"] = shuffle_seed*10*i  
    cfg['training_settings']["model_seed"] = shuffle_seed*20*i  
    
    # Define custom prefix and suffix of model name 
    # - Default: RNN-AR6-UNetSpherical-Healpix_400km-Graph_knn-k20-MaxAreaPooling
    cfg['model_settings']["model_name_suffix"] = model_suffix + "-Model_" + str(i)
    cfg['model_settings']["model_name_prefix"] = model_name_prefix  
    
    # Save config path (temporary)
    tmp_cfg_path = os.path.join(tmp_dirpath, "tmp_config.json")
    write_config_file(cfg, tmp_cfg_path)
    
    # Define command 
    cmd = " ".join(["python", python_training_routine,
                    '--config_file', tmp_cfg_path,
                    '--cuda', cuda,
                    '--data_dir', data_dir,
                    '--exp_dir', exp_dir,
                    '--force', str(force)])                  
    subprocess.run(cmd, shell=True)
    os.remove(tmp_cfg_path) 

# - Verification 
model_dirs = glob.glob(os.path.join(exp_dir, "*"))
model_dirs = " ".join(model_dirs)
cmd = " ".join(["python", python_verification_routine,
                '--data_dir', data_dir,
                '--DeepEnsemble_dir', DeepEnsemble_dir,
                '--model_dirs', model_dirs,
                '--remove_model_dirs', str(remove_model_dirs),
                '--force', str(force)])                  
 

##----------------------------------------------------------------------------.
# Remove temporary directory 
shutil.rmtree(tmp_dirpath)
