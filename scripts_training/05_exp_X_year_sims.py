#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 19:17:46 2021

@author: ghiggi
"""
##----------------------------------------------------------------------------.
import os
import sys
sys.path.append('../')
import shutil 
import tempfile
import subprocess
from modules.utils_config import read_config_file
from modules.utils_config import write_config_file
##----------------------------------------------------------------------------.
# Define fpaths
data_dir = "/ltenas3/DeepSphere/data/preprocessed/ERA5_HRES"
exp_dir = "/data/weather_prediction/experiments/XXXX"

# Define model name 
model_name = "RNN-UNetSpherical-healpix-16-k20-MaxAreaPooling-float32-AR6-LinearStep_weight_corrected"

# Retrieve model directory 
model_dir = os.path.join(exp_dir, model_name)

# Define python routine 
project_dir = "/home/ghiggi/Projects/deepsphere-weather"
model_name_prefix = "State"
python_routine = os.path.join(project_dir, 
                              "scripts_training",
                              "predict_state.py")

#-----------------------------------------------------------------------------.
# Prediction options
n_years = 5
forecast_cycle = 6
AR_iterations = int(n_years*365*24/forecast_cycle)
cuda = '0'
forecast_reference_times = ['2013-12-31T18:00'] # ['2013-12-31T19:00']
batch_size = 32
AR_blocks = 1000
zarr_fname = "5years_pred.zarr"
force_zarr = False # force to write a new zarr if already existing
force_gpu = True
dst_dirpath = None # to save prediction within <model_dir>/model_predictions/... 

#-----------------------------------------------------------------------------.
   
# Define command 
cmd = " ".join(["python", python_routine,
                 '--data_dir', data_dir,
                 "--model_dir", model_dir,
                 "--forecast_reference_times", *forecast_reference_times,
                 "--AR_iterations", str(AR_iterations), 
                 "--batch_size", str(batch_size), 
                 "--AR_blocks", str(AR_blocks), 
                 # "--dst_dirpath", dst_dirpath,   
                 "--zarr_fname", zarr_fname, 
                 "--force_zarr", str(force_zarr),
                 "--force_gpu", str(force_gpu),
                 "--cuda", cuda])
             
subprocess.run(cmd, shell=True)

 

