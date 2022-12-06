#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 20:01:27 2021

@author: ghiggi
"""
##----------------------------------------------------------------------------.
import os
import sys

sys.path.append("../")
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
exp_dir = "/home/ghiggi/Projects/DeepSphere/data/experiments/exp_samplings"

project_dir = "/home/ghiggi/Projects/deepsphere-weather"
tmp_dirpath = tempfile.mkdtemp()
cuda = "0"
force = True  # If True, if model folder already exist, is replaced.

# Define python routine
model_name_prefix = "State"
python_routine = os.path.join(project_dir, "scripts_training", "train_predict_state.py")

# -----------------------------------------------------------------------------.
### Run trainings using graph convolutions
# Define samplings
samplings = [
    "Healpix_400km",
    "Icosahedral_400km",
    "O24",
    "Equiangular_400km",
    "Equiangular_400km_tropics",
    "Cubed_400km",
]

# Define pooling method
pool_methods = ["MaxArea"]  # ['Interp', 'MaxVal', 'MaxArea', 'Learn']
# Define graph type
graph_types = ["knn", "voronoi"]
# Define model seed and data sets
model_seed = 8  # Model seed is same across model (change only between n_models)
shuffle_seed = 10  # Provide same data to all samplings (change only between n_models)
# -----------------------------------------------------------------------------.
# Define number of models
n_models = 2

for i in range(1, n_models + 1):
    for graph_type in graph_types:
        for sampling in samplings:
            for pool_method in pool_methods:
                # Define config file path
                cfg_path = os.path.join(
                    project_dir,
                    "configs",
                    "UNetSpherical",
                    sampling,
                    pool_method + "Pool-Graph_" + graph_type + ".json",
                )

                # Read general config
                cfg = read_config_file(fpath=cfg_path)
                # Optional configs
                cfg["training_settings"]["deterministic_training"] = True
                cfg["dataloader_settings"]["random_shuffling"] = True
                cfg["training_settings"]["seed_random_shuffling"] = (
                    shuffle_seed * 10 * i
                )
                cfg["training_settings"]["seed_model_weights"] = model_seed * 10 * i

                # Define custom prefix and suffix of model name
                # - Default: RNN-AR6-UNetSpherical-Healpix_400km-Graph_knn-k20-MaxAreaPooling
                cfg["model_settings"]["model_name_suffix"] = "Model_" + str(i)
                cfg["model_settings"]["model_name_prefix"] = model_name_prefix

                # Save config path (temporary)
                tmp_cfg_path = os.path.join(tmp_dirpath, "tmp_config.json")
                write_config_file(cfg, tmp_cfg_path)

                # Define command
                cmd = " ".join(
                    [
                        "python",
                        python_routine,
                        "--config_file",
                        tmp_cfg_path,
                        "--cuda",
                        cuda,
                        "--data_dir",
                        data_dir,
                        "--exp_dir",
                        exp_dir,
                        "--force",
                        str(force),
                    ]
                )
                subprocess.run(cmd, shell=True)
                os.remove(tmp_cfg_path)

##----------------------------------------------------------------------------.
### Run training with classical conv2D image on Equiangular
# - Define pooling method
pool_methods = "Max"  # ["Max","Avg"]
# - Define other options
sampling = ["Equiangular_400km"]  # 'Equiangular_400km_tropics'
periodic_paddings = [True, False]

# - Define number of models
n_models = 2
for i in range(1, n_models + 1):
    for sampling in samplings:
        for pool_method in pool_methods:
            for periodic_padding in periodic_paddings:
                # Define config file path
                cfg_path = os.path.join(
                    project_dir,
                    "configs",
                    "UNetSpherical",
                    sampling,
                    pool_method + "Pool-Graph_" + "knn" + ".json",
                )

                # Read general config
                cfg = read_config_file(fpath=cfg_path)
                # Optional configs
                cfg["training_settings"]["deterministic_training"] = True
                cfg["dataloader_settings"]["random_shuffling"] = True
                cfg["training_settings"]["seed_random_shuffling"] = (
                    shuffle_seed * 10 * i
                )
                cfg["training_settings"]["seed_model_weights"] = model_seed * 10 * i

                cfg["model_settings"]["conv_type"] = "image"
                cfg["model_settings"]["periodic_padding"] = periodic_padding

                # Define custom prefix and suffix of model name
                # - Default: RNN-AR6-UNetSpherical-Equiangular_400km-ConvImage-MaxPooling
                if periodic_padding:
                    cfg["model_settings"][
                        "model_name_suffix"
                    ] = "PeriodicPadding-Model_ +" + str(i)
                else:
                    cfg["model_settings"]["model_name_suffix"] = "Model_ +" + str(i)
                cfg["model_settings"]["model_name_prefix"] = model_name_prefix

                # Save config path (temporary)
                tmp_cfg_path = os.path.join(tmp_dirpath, "tmp_config.json")
                write_config_file(cfg, tmp_cfg_path)

                # Define command
                cmd = " ".join(
                    [
                        "python",
                        python_routine,
                        "--config_file",
                        tmp_cfg_path,
                        "--cuda",
                        cuda,
                        "--data_dir",
                        data_dir,
                        "--exp_dir",
                        exp_dir,
                        "--force",
                        str(force),
                    ]
                )
                subprocess.run(cmd, shell=True)
                os.remove(tmp_cfg_path)

##----------------------------------------------------------------------------.

# Remove temporary directory
shutil.rmtree(tmp_dirpath)
