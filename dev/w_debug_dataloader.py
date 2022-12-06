#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 13:41:16 2021

@author: ghiggi
"""
import os

os.chdir("/home/ghiggi/Projects/deepsphere-weather")
import sys

sys.path.append("../")
import warnings
import time
import dask
import argparse
import torch
import pickle
import pygsp as pg
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs

from tabulate import tabulate
from torch import optim
from torchinfo import summary

## DeepSphere-Weather modules
from modules.utils_config import read_config_file
from modules.utils_config import write_config_file
from modules.utils_config import get_model_settings
from modules.utils_config import get_training_settings
from modules.utils_config import get_ar_settings
from modules.utils_config import get_dataloader_settings
from modules.utils_config import get_pytorch_model
from modules.utils_config import get_model_name
from modules.utils_config import set_pytorch_settings
from modules.utils_config import load_pretrained_model
from modules.utils_config import create_experiment_directories
from modules.utils_config import print_model_description
from modules.utils_config import print_tensor_info

from modules.utils_models import get_pygsp_graph
from modules.utils_io import get_ar_model_tensor_info
from modules.training_autoregressive import AutoregressiveTraining
from modules.predictions_autoregressive import AutoregressivePredictions
from modules.predictions_autoregressive import rechunk_forecasts_for_verification
from modules.utils_torch import summarize_model
from modules.AR_Scheduler import AR_Scheduler
from modules.early_stopping import EarlyStopping
from modules.loss import WeightedMSELoss, AreaWeights

from modules.dataloader_autoregressive import AutoregressiveDataset
from modules.dataloader_autoregressive import AutoregressiveDataLoader
from modules.dataloader_autoregressive import get_aligned_ar_batch
from modules.dataloader_autoregressive import remove_unused_Y
from modules.dataloader_autoregressive import cylic_iterator
from modules.utils_autoregressive import check_ar_settings
from modules.utils_autoregressive import check_input_k
from modules.utils_autoregressive import check_output_k
from modules.utils_training import AR_TrainingInfo
from modules.utils_torch import check_device
from modules.utils_torch import check_pin_memory
from modules.utils_torch import check_asyncronous_gpu_transfer
from modules.utils_torch import check_prefetch_in_gpu
from modules.utils_torch import check_prefetch_factor
from modules.utils_torch import check_ar_training_strategy
from modules.utils_torch import get_time_function
from modules.loss import reshape_tensors_4_loss

from modules.utils_swag import bn_update_with_loader

## Project specific functions
import modules.my_models_graph as my_architectures

## Side-project utils (maybe migrating to separate packages in future)
import modules.xsphere  # required for xarray 'sphere' accessor
import modules.xverif as xverif

# import modules.xscaler as xscaler
from modules.xscaler import LoadScaler
from modules.xscaler import SequentialScaler
from modules.xscaler import LoadAnomaly


# - Plotting functions
from modules.my_plotting import plot_skill_maps
from modules.my_plotting import plot_global_skill
from modules.my_plotting import plot_global_skills
from modules.my_plotting import plot_skills_distribution
from modules.my_plotting import create_gif_forecast_error
from modules.my_plotting import create_gif_forecast_anom_error

# For plotting
import matplotlib

matplotlib.use("cairo")  # Cairo
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["savefig.facecolor"] = "white"  # (1,1,1,0)
matplotlib.rcParams["savefig.edgecolor"] = "none"

# Disable warnings
warnings.filterwarnings("ignore")

##----------------------------------------------------------------------------.
# data_dir = "/home/ghiggi/Projects/DeepSphere/data/toy_data/ERA5_HRES/"
# exp_dir =  "/home/ghiggi/Projects/DeepSphere/data/experiments_GG"

data_dir = "/ltenas3/DeepSphere/data/preprocessed_ds/ERA5_HRES"
exp_dir = "/data/weather_prediction/experiments_GG/new"


cfg_path = "/home/ghiggi/Projects/deepsphere-weather/configs/UNetSpherical/Cubed_400km/MaxAreaPool-Graph_knn.json"
data_sampling_dir = os.path.join(data_dir, "Cubed_400km")
force = True

cfg = read_config_file(fpath=cfg_path)

# cfg['training_settings']["training_batch_size"] = 2
cfg["dataloader_settings"]["random_shuffling"] = True
cfg["training_settings"]["seed_random_shuffling"] = 10
cfg["training_settings"]["seed_model_weights"] = 10
cfg["training_settings"]["deterministic_training"] = True

# 'State-RNN-AR6-UNetSpherical-Healpix_400km-Graph_knn-k20-MaxAreaPooling
cfg["model_settings"]["model_name_suffix"] = ""  # Model_1, Model_2
cfg["model_settings"]["model_name_prefix"] = "State"  # StateIncrement, Anomaly
cfg["model_settings"]["architecture_name"] = "UNetSpherical"

# -----------------------------------------------------------------------------.

### Read experiment configuration settings
cfg = read_config_file(fpath=cfg_path)

##------------------------------------------------------------------------.
### Retrieve experiment-specific configuration settings
model_settings = get_model_settings(cfg)
ar_settings = get_ar_settings(cfg)
training_settings = get_training_settings(cfg)
dataloader_settings = get_dataloader_settings(cfg)

##------------------------------------------------------------------------.
#### Load Datasets
data_sampling_dir = os.path.join(data_dir, cfg["model_settings"]["sampling_name"])

data_dynamic = xr.open_zarr(
    os.path.join(data_sampling_dir, "Data", "dynamic", "time_chunked", "dynamic.zarr")
)
data_bc = xr.open_zarr(
    os.path.join(data_sampling_dir, "Data", "bc", "time_chunked", "bc.zarr")
)
ds_static = xr.open_zarr(os.path.join(data_sampling_dir, "Data", "static.zarr"))

# - Select dynamic features
data_dynamic = data_dynamic[["z500", "t850"]]

##------------------------------------------------------------------------.
# Preload data in memory
# t_i = time.time()
# data_dynamic = data_dynamic.compute()
# data_bc = data_bc.compute()
# ds_static = ds_static.compute()
# print('- Preload data in memory: {:.2f} minutes'.format((time.time() - t_i)/60))

##------------------------------------------------------------------------.
# - Prepare static data
# - Keep land-surface mask as it is
# - Keep sin of latitude and remove longitude information
ds_static = ds_static.drop(["sin_longitude", "cos_longitude"])
# - Scale orography between 0 and 1 (is already left 0 bounded)
ds_static["orog"] = ds_static["orog"] / ds_static["orog"].max()
# - One Hot Encode soil type
# ds_slt_OHE = xscaler.OneHotEnconding(ds_static['slt'])
# ds_static = xr.merge([ds_static, ds_slt_OHE])
# ds_static = ds_static.drop('slt')

data_static = ds_static

##------------------------------------------------------------------------.
#### Define scaler to apply on the fly within DataLoader
# - Load scalers
dynamic_scaler = LoadScaler(
    os.path.join(data_sampling_dir, "Scalers", "GlobalStandardScaler_dynamic.nc")
)
bc_scaler = LoadScaler(
    os.path.join(data_sampling_dir, "Scalers", "GlobalStandardScaler_bc.nc")
)
# # - Create single scaler object
scaler = SequentialScaler(dynamic_scaler, bc_scaler)

##-----------------------------------------------------------------------------.
#### Split data into train, test and validation set
# - Defining time split for training
training_years = np.array(["1980-01-01T07:00", "2014-12-31T23:00"], dtype="M8[m]")
validation_years = np.array(["2015-01-01T00:00", "2016-12-31T23:00"], dtype="M8[m]")
test_years = np.array(["2017-01-01T00:00", "2018-12-31T23:00"], dtype="M8[m]")

# - Split data sets
t_i = time.time()
training_data_dynamic = data_dynamic.sel(
    time=slice(training_years[0], training_years[-1])
)
training_data_bc = data_bc.sel(time=slice(training_years[0], training_years[-1]))

validation_data_dynamic = data_dynamic.sel(
    time=slice(validation_years[0], validation_years[-1])
)
validation_data_bc = data_bc.sel(time=slice(validation_years[0], validation_years[-1]))

test_data_dynamic = data_dynamic.sel(time=slice(test_years[0], test_years[-1]))
test_data_bc = data_bc.sel(time=slice(test_years[0], test_years[-1]))

print(
    "- Splitting data into train, validation and test sets: {:.2f}s".format(
        time.time() - t_i
    )
)

##------------------------------------------------------------------------.
### Define pyTorch settings (before PyTorch model definition)
# - Here inside is eventually set the seed for fixing model weights initialization
# - Here inside the training precision is set (currently only float32 works)
device = set_pytorch_settings(training_settings)

##------------------------------------------------------------------------.
## Retrieve dimension info of input-output Torch Tensors
tensor_info = get_ar_model_tensor_info(
    ar_settings=ar_settings,
    data_dynamic=training_data_dynamic,
    data_static=data_static,
    data_bc=training_data_bc,
)
print_tensor_info(tensor_info)
# - Add dim info to cfg file
model_settings["tensor_info"] = tensor_info
cfg["model_settings"]["tensor_info"] = tensor_info

##------------------------------------------------------------------------.
# Print model settings
print_model_description(cfg)

##------------------------------------------------------------------------.
### Define the model architecture
model = get_pytorch_model(module=my_architectures, model_settings=model_settings)
###-----------------------------------------------------------------------.
## If requested, load a pre-trained model for fine-tuning
if model_settings["pretrained_model_name"] is not None:
    model_dir = os.path.join(exp_dir, model_settings["model_name"])
    load_pretrained_model(model=model, model_dir=model_dir)
###-----------------------------------------------------------------------.
### Transfer model to the device (i.e. GPU)
model = model.to(device)

###-----------------------------------------------------------------------.
### Summarize the model
input_shape = tensor_info["input_shape"]
input_shape[0] = training_settings["training_batch_size"]
print(
    summary(model, input_shape, col_names=["input_size", "output_size", "num_params"])
)

_ = summarize_model(
    model=model,
    input_size=tuple(tensor_info["input_shape"][1:]),
    batch_size=training_settings["training_batch_size"],
    device=device,
)

###-----------------------------------------------------------------------.
# DataParallel training option on multiple GPUs
# if training_settings['dataparallel_training'] is True:
#     if torch.cuda.device_count() > 1 and len(training_settings['gpu_devices_ids']) > 1:
#         model = nn.DataParallel(model, device_ids=[i for i in training_settings['gpu_devices_ids']])

###-----------------------------------------------------------------------.
## Generate the (new) model name and its directories
if model_settings["model_name"] is not None:
    model_name = model_settings["model_name"]
else:
    model_name = get_model_name(cfg)
    model_settings["model_name"] = model_name
    cfg["model_settings"]["model_name_prefix"] = None
    cfg["model_settings"]["model_name_suffix"] = None

exp_dir = create_experiment_directories(
    exp_dir=exp_dir, model_name=model_name, force=force
)  # force=True will delete existing directory

##------------------------------------------------------------------------.
# Define model weights filepath
model_fpath = os.path.join(exp_dir, "model_weights", "model.h5")

##------------------------------------------------------------------------.
# Write config file in the experiment directory
write_config_file(cfg=cfg, fpath=os.path.join(exp_dir, "config.json"))

##------------------------------------------------------------------------.
### - Define custom loss function
# - Compute area weights
weights = AreaWeights(model.graphs[0])

# - Define weighted loss
criterion = WeightedMSELoss(weights=weights)

##------------------------------------------------------------------------.
### - Define optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=training_settings["learning_rate"],
    eps=1e-7,
    weight_decay=0,
    amsgrad=False,
)

##------------------------------------------------------------------------.
## - Define AR_Weights_Scheduler
# - For RNN: growth and decay works well (fix the first)
if training_settings["ar_training_strategy"] == "RNN":
    ar_scheduler = AR_Scheduler(
        method="LinearStep",
        factor=0.0005,
        fixed_ar_weights=[0],
        initial_ar_absolute_weights=[1, 1],
    )
# - FOR AR : Do not decay weights once they growthed
elif training_settings["ar_training_strategy"] == "AR":
    ar_scheduler = AR_Scheduler(
        method="LinearStep",
        factor=0.0005,
        fixed_ar_weights=np.arange(0, ar_settings["ar_iterations"]),
        initial_ar_absolute_weights=[1, 1],
    )
else:
    raise NotImplementedError("'ar_training_strategy' must be either 'AR' or 'RNN'.")

##------------------------------------------------------------------------.
### - Define Early Stopping
# - Used also to update ar_scheduler (aka increase AR iterations) if 'ar_iterations' not reached.
patience = 500
minimum_iterations = 500
minimum_improvement = 0.001  # 0 to not stop
stopping_metric = "validation_total_loss"  # training_total_loss
mode = "min"  # MSE best when low
early_stopping = EarlyStopping(
    patience=patience,
    minimum_improvement=minimum_improvement,
    minimum_iterations=minimum_iterations,
    stopping_metric=stopping_metric,
    mode=mode,
)

##------------------------------------------------------------------------.
### - Define LR_Scheduler
lr_scheduler = None

##------------------------------------------------------------------------.
############################
### - Debug AR_Training ####
############################
dask.config.set(
    scheduler="synchronous"
)  # This is very important otherwise the dataloader hang
model = model
model_fpath = model_fpath
# Loss settings
criterion = criterion
optimizer = optimizer
lr_scheduler = lr_scheduler
ar_scheduler = ar_scheduler
early_stopping = early_stopping
# Data
data_static = data_static
training_data_dynamic = training_data_dynamic
training_data_bc = training_data_bc
validation_data_dynamic = validation_data_dynamic
validation_data_bc = validation_data_bc
scaler = scaler
bc_generator = None
ar_batch_fun = get_aligned_ar_batch
# Dataloader settings
num_workers = dataloader_settings["num_workers"]
autotune_num_workers = dataloader_settings["autotune_num_workers"]
prefetch_factor = dataloader_settings["prefetch_factor"]
prefetch_in_gpu = dataloader_settings["prefetch_in_gpu"]
drop_last_batch = dataloader_settings["drop_last_batch"]
shuffle = dataloader_settings["random_shuffling"]
shuffle_seed = training_settings["seed_random_shuffling"]
pin_memory = dataloader_settings["pin_memory"]
asyncronous_gpu_transfer = dataloader_settings["asyncronous_gpu_transfer"]
# Autoregressive settings
input_k = ar_settings["input_k"]
output_k = ar_settings["output_k"]
forecast_cycle = ar_settings["forecast_cycle"]
ar_iterations = ar_settings["ar_iterations"]
stack_most_recent_prediction = ar_settings["stack_most_recent_prediction"]
# Training settings
ar_training_strategy = training_settings["ar_training_strategy"]
training_batch_size = training_settings["training_batch_size"]
validation_batch_size = training_settings["validation_batch_size"]
epochs = training_settings["epochs"]
scoring_interval = training_settings["scoring_interval"]
save_model_each_epoch = training_settings["save_model_each_epoch"]
# GPU settings
device = device
# SWAG settings
swag = False
swag_model = None
swag_freq = 10
swa_start = 8


training_num_workers = 8
validation_num_workers = 8
epoch = 1
ar_iteration = 2
##------------------------------------------------------------------------.
##############################
### - Debug AR_dataloader ####
##############################
##------------------------------------------------------------------------.
## Checks arguments
device = check_device(device)
pin_memory = check_pin_memory(
    pin_memory=pin_memory, num_workers=num_workers, device=device
)
asyncronous_gpu_transfer = check_asyncronous_gpu_transfer(
    asyncronous_gpu_transfer=asyncronous_gpu_transfer, device=device
)
prefetch_in_gpu = check_prefetch_in_gpu(
    prefetch_in_gpu=prefetch_in_gpu, num_workers=num_workers, device=device
)
prefetch_factor = check_prefetch_factor(
    prefetch_factor=prefetch_factor, num_workers=num_workers
)
ar_training_strategy = check_ar_training_strategy(ar_training_strategy)
# Check ar_scheduler
if len(ar_scheduler.ar_weights) > ar_iterations + 1:
    raise ValueError(
        "The AR scheduler has {} AR weights, but ar_iterations is specified to be {}".format(
            len(ar_scheduler.ar_weights), ar_iterations
        )
    )
##------------------------------------------------------------------------.
# Check that autoregressive settings are valid
# - input_k and output_k must be numpy arrays hereafter !
print("- Defining AR settings:")
input_k = check_input_k(input_k=input_k, ar_iterations=ar_iterations)
output_k = check_output_k(output_k=output_k)
check_ar_settings(
    input_k=input_k,
    output_k=output_k,
    forecast_cycle=forecast_cycle,
    ar_iterations=ar_iterations,
    stack_most_recent_prediction=stack_most_recent_prediction,
)
##------------------------------------------------------------------------.
## Check early stopping
if validation_data_dynamic is None:
    if early_stopping is not None:
        if early_stopping.stopping_metric == "total_validation_loss":
            print(
                "Validation dataset is not provided."
                "Stopping metric of early_stopping set to 'total_training_loss'"
            )
            early_stopping.stopping_metric = "total_training_loss"

##------------------------------------------------------------------------.
ar_training_info = AR_TrainingInfo(
    ar_iterations=ar_iterations, epochs=epochs, ar_scheduler=ar_scheduler
)
##------------------------------------------------------------------------.
## Decide wheter to tune num_workers
num_workers_list = [num_workers]
##------------------------------------------------------------------------.
# Ensure criterion and model are on device
model.to(device)
criterion.to(device)
##------------------------------------------------------------------------.
# Zeros gradients
optimizer.zero_grad()
##------------------------------------------------------------------------.
### Create Datasets
t_i = time.time()
trainingDataset = AutoregressiveDataset(
    data_dynamic=training_data_dynamic,
    data_bc=training_data_bc,
    data_static=data_static,
    bc_generator=bc_generator,
    scaler=scaler,
    # Custom AR batching function
    ar_batch_fun=ar_batch_fun,
    training_mode=True,
    # Autoregressive settings
    input_k=input_k,
    output_k=output_k,
    forecast_cycle=forecast_cycle,
    ar_iterations=ar_scheduler.current_ar_iterations,
    stack_most_recent_prediction=stack_most_recent_prediction,
    # GPU settings
    device=device,
)
trainingDataLoader = AutoregressiveDataLoader(
    dataset=trainingDataset,
    batch_size=16,
    drop_last_batch=drop_last_batch,
    shuffle=False,
    shuffle_seed=shuffle_seed,
    num_workers=num_workers,
    prefetch_factor=prefetch_factor,
    prefetch_in_gpu=prefetch_in_gpu,
    pin_memory=pin_memory,
    asyncronous_gpu_transfer=asyncronous_gpu_transfer,
    device=device,
)

ar_training_info.new_epoch()
##--------------------------------------------------------------------.
# Iterate along training batches
trainingDataLoader_iter = iter(trainingDataLoader)

##----------------------------------------------------------------.
# Retrieve the training batch
dask.config.set(
    scheduler="synchronous"
)  # This is very important otherwise the dataloader hang
training_batch_dict = next(trainingDataLoader_iter)
training_batch_dict


##----------------------------------------------------------------.
training_batch_dict["forecast_time_info"]

##----------------------------------------------------------------.
### Check dataloader load the correct data
feature_idx = 1
sample = 10
time_Y = training_batch_dict["forecast_time_info"]["forecast_start_time"][sample]
feature_order = training_batch_dict["feature_order"]["dynamic"]


x_dynamic_torch = training_batch_dict["X_dynamic"][0][sample, ..., feature_idx].numpy()
Y_dynamic_torch = (
    training_batch_dict["Y"][0][sample, ..., feature_idx].numpy().squeeze()
)

x_feature = (
    data_dynamic.to_array("feature")
    .isel(feature=slice(feature_idx, feature_idx + 1))
    .sel(time=slice(time_Y, time_Y + 1))
    .load()
)
x_feature = x_feature.transpose(..., "feature")
x_std_feature = (
    scaler.transform(x_feature, variable_dim="feature").compute().values.squeeze()
)

np.allclose(x_std_feature, Y_dynamic_torch)


##----------------------------------------------------------------------------.
# Check dataset works
trainingDataset[1]
trainingDataset[8764]

trainingDataset.__len__()
trainingDataLoader.dataset.__len__()
trainingDataLoader.sampler.num_samples

dir(trainingDataLoader)

##----------------------------------------------------------------------------.
# Define iterator dataloader
d_iter = iter(trainingDataLoader)
t_i = time.time()
training_batch_dict = next(d_iter)
time.time() - t_i


##----------------------------------------------------------------------------.
# Check training dataloader
# for batch_dict in trainingDataLoader:
#     print(".")
#     # print(len(batch_dict['Y']))

# # Check valid dataloader
# a = next(validationDataLoader_iter)

##----------------------------------------------------------------------------.
## Check update AR iterations works
trainingDataset.update_AR_iterations(0)
trainingDataLoader_iter = cylic_iterator(trainingDataLoader)
for i in range(trainingDataset.AR_iterations, AR_iterations + 1):
    trainingDataset.AR_iterations
    training_batch_dict = next(trainingDataLoader_iter)
    print(training_batch_dict["Y"].keys())
    if i < AR_iterations:
        trainingDataset.update_AR_iterations(trainingDataset.AR_iterations + 1)
        del trainingDataLoader
        trainingDataLoader = AutoregressiveDataLoader(
            dataset=trainingDataset,
            batch_size=batch_size,
            drop_last_batch=drop_last_batch,
            random_shuffle=random_shuffle,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            prefetch_in_GPU=prefetch_in_GPU,
            pin_memory=pin_memory,
            asyncronous_GPU_transfer=asyncronous_GPU_transfer,
            device=device,
        )
        # print(trainingDataset.__len__())
        # print(trainingDataLoader.sampler.num_samples)
        trainingDataLoader_iter = iter(trainingDataLoader)


##----------------------------------------------------------------------------.
## Check in AR loop
trainingDataset.update_AR_iterations(0)
update_every = 5
count = 0
flag_print = True
flag_break = False

# Check batching and stacking works
print(trainingDataset.AR_iterations)
trainingDataLoader_iter = iter(trainingDataLoader)
for batch_count in range(len(trainingDataLoader_iter)):
    training_batch_dict = next(trainingDataLoader_iter)
    print(".", end="")
    count = count + 1
    # training_batch_dict = next(d_iter)
    if flag_print is True:
        print(end="\n")
        print(training_batch_dict["Y"].keys())
        flag_print = False

    ##----------------------------------------------------------------.
    # Perform autoregressive training loop
    # - The number of AR iterations is determined by AR_scheduler.AR_weights
    # - If AR_weights are all zero after N forecast iteration:
    #   --> Load data just for F forecast iteration
    #   --> Autoregress model predictions just N times to save computing time
    dict_training_Y_predicted = {}
    dict_training_loss_per_leadtime = {}
    for i in range(trainingDataset.AR_iterations + 1):
        # Retrieve X and Y for current AR iteration
        torch_X, torch_Y = get_AR_batch(
            AR_iteration=i,
            batch_dict=training_batch_dict,
            dict_Y_predicted=dict_training_Y_predicted,
            device=device,
            asyncronous_GPU_transfer=asyncronous_GPU_transfer,
        )

        # if i != trainingDataset.AR_iterations:
        #     print(end="\n")
        #     print(training_batch_dict['Y'].keys())
        ##------------------------------------------------------------.
        # Forward pass and store output for stacking into next AR iterations
        dict_training_Y_predicted[i] = torch_Y
    if count == update_every:
        if trainingDataset.AR_iterations < AR_iterations:
            print(end="\n")
            print("Update AR")
            trainingDataset.update_AR_iterations(trainingDataset.AR_iterations + 1)
            del trainingDataLoader
            trainingDataLoader = AutoregressiveDataLoader(
                dataset=trainingDataset,
                batch_size=batch_size,
                drop_last_batch=drop_last_batch,
                random_shuffle=random_shuffle,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                prefetch_in_GPU=prefetch_in_GPU,
                pin_memory=pin_memory,
                asyncronous_GPU_transfer=asyncronous_GPU_transfer,
                device=device,
            )
            trainingDataLoader_iter = iter(trainingDataLoader)
            count = 0
            flag_print = True
        else:
            flag_break = True
    if flag_break:
        break
