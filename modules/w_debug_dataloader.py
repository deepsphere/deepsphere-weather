#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 17:14:20 2021

@author: ghiggi
"""

##----------------------------------------------------------------.
import os
import dask
import xarray as xr
from modules.dataloader_autoregressive import AutoregressiveDataset
from modules.dataloader_autoregressive import AutoregressiveDataLoader
from modules.dataloader_autoregressive import get_AR_batch
from modules.utils_autoregressive import check_input_k
from modules.utils_autoregressive import check_output_k 


dask.config.set(scheduler='synchronous')

##----------------------------------------------------------------.
## Optimize dataloading (benchmark !)
# - DataArray vs Dataset to_array()  (time the overhead)
# - Scaler in _get_item              (time the overhead) 
##----------------------------------------------------------------.
# - Num_workers and prefetch_factor speed up a lot  
# - Change dtype only when converting to Torch Tensor ... --> Enable disk savings and fast I/O

##----------------------------------------------------------------.
data_dir = "/home/ghiggi/Projects/DeepSphere/ToyData/Healpix_400km"
# DataArray
da_dynamic = xr.open_zarr(os.path.join(data_dir, "DataArray", "dynamic.zarr"))['Data']
da_bc = xr.open_zarr(os.path.join(data_dir,"DataArray", "bc.zarr"))['Data']
da_static = xr.open_zarr(os.path.join(data_dir,"DataArray","static.zarr"))['Data']

# da_dynamic = da_dynamic.isel(time=slice(0,100))
# da_bc = da_bc.isel(time=slice(0,100))

# Dataset 
# from modules.my_io import reformat_Datasets
# ds_dynamic = xr.open_zarr(os.path.join(data_dir, "Dataset", "dynamic.zarr"))
# ds_bc = xr.open_zarr(os.path.join(data_dir,"Dataset", "bc.zarr"))
# ds_static = xr.open_zarr(os.path.join(data_dir,"Dataset","static.zarr"))
# dict_DataArrays = reformat_Datasets(ds_training_dynamic = ds_dynamic,
#                                     ds_training_bc = ds_bc,
#                                     ds_static = ds_static,              
#                                     preload_data_in_CPU = False)
# da_static = dict_DataArrays['da_static']
# da_dynamic = dict_DataArrays['da_training_dynamic']
# da_bc = dict_DataArrays['da_training_bc']

#-----------------------------------------------------------------------------.

scaler = None

scaler = GlobalStandardScaler(data=da_dynamic)
scaler.fit()



input_k = check_input_k(input_k=input_k, AR_iterations=AR_iterations)   
output_k = check_output_k(output_k=output_k)
num_workers = 4   # if > 0, when changing AR iterations in Dataset ... do not update
prefetch_factor = 5
random_shuffle = True 

drop_last_batch = False
AR_iterations = 10
batch_size = 30

trainingDataset = AutoregressiveDataset(da_dynamic = da_dynamic,  
                                        da_bc = da_bc,
                                        da_static = da_static,
                                        scaler = scaler,
                                        # Autoregressive settings  
                                        input_k = input_k,
                                        output_k = output_k,
                                        forecast_cycle = forecast_cycle,                           
                                        AR_iterations = 0,
                                        max_AR_iterations = AR_iterations,
                                        stack_most_recent_prediction = stack_most_recent_prediction, 
                                        # GPU settings 
                                        device = device,
                                        # Precision settings
                                        numeric_precision = numeric_precision)
trainingDataLoader = AutoregressiveDataLoader(dataset = trainingDataset,                                                   
                                              batch_size = batch_size,  
                                              drop_last_batch = drop_last_batch,
                                              random_shuffle = random_shuffle,
                                              num_workers = num_workers,
                                              prefetch_factor = prefetch_factor, 
                                              prefetch_in_GPU = prefetch_in_GPU,  
                                              pin_memory = pin_memory,
                                              asyncronous_GPU_transfer = asyncronous_GPU_transfer, 
                                              device = device)
##----------------------------------------------------------------------------.
# Shape 
da_dynamic.shape
# Check dataset works 
trainingDataset[1] 
trainingDataset[8764]

trainingDataset.__len__()
trainingDataLoader.dataset.__len__()
trainingDataLoader.sampler.num_samples

dir(trainingDataLoader)

## Shared Array 
# https://discuss.pytorch.org/t/dataloader-resets-dataset-state/27960/17

# For map-style datasets, the main process generates the indices using sampler
#  and sends them to the workers
 

## persistent_workers
# https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/5
# - Default = False 
# - If num_workers=0, no worker processes, so persistent_workers has no effects 
# When False
# - Every epoch (when the code hits the line 'for sample in dataloader') a
#   new set of workers is created  
    
# When False

# - Might improve performances as creating the workers is expensive 
# - The dataloader will have some persistent state even when it is not used 
#   (which can use some RAM depending on your dataset)

# dynamic dataloader --> changing within loop 
# enumerate(dataloader)

# Manipulating the data on the fly inside a DataLoader loop might not work,
#  if you are using multiple workers. Changes wouldn’t be reflected across all worker threads.
# Forced to use num_workers=0 or use some shared memory approach.
# The Dataset is copied to the workers in case you are using multiple workers.
# This also means that you cannot modify the Dataset inplace anymore

# If you are using persistent_workers=False in the DataLoader (the default setup), 
# you should be able to manipulate the underlying Dataset after each epoch even with num_workers>0

# https://discuss.pytorch.org/t/dataloader-resets-dataset-state/27960/19
# https://discuss.pytorch.org/t/communicating-with-dataloader-workers/11473

#-----------------------------------------------------------------------------.
### Set seed for batches 
# - Eeach worker has an independent seed that is initialized to 
#   the curent random seed + the id of the worker
# - Need to reset the numpy random seed at the beginning of each epoch
#    because all random seed modifications in __getitem__ are local to each worker

#np.random.seed(1) # reset seed
# def seed_init_fn(worker_id):
#    seed = np.random.get_state()[1][0] + worker_id
#    np.random.seed(seed)
#    random.seed(seed)
#    torch.manual_seed(seed)
#    return
## worker_init_fn = seed_init_fn --> always yield same data 

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
for i in range(trainingDataset.AR_iterations, AR_iterations+1):
    trainingDataset.AR_iterations 
    training_batch_dict = next(trainingDataLoader_iter)
    print(training_batch_dict['Y'].keys())
    if i < AR_iterations:
        trainingDataset.update_AR_iterations(trainingDataset.AR_iterations + 1) 
        del trainingDataLoader
        trainingDataLoader = AutoregressiveDataLoader(dataset = trainingDataset,                                                   
                                                      batch_size = batch_size,  
                                                      drop_last_batch = drop_last_batch,
                                                      random_shuffle = random_shuffle,
                                                      num_workers = num_workers,
                                                      prefetch_factor = prefetch_factor, 
                                                      prefetch_in_GPU = prefetch_in_GPU,  
                                                      pin_memory = pin_memory,
                                                      asyncronous_GPU_transfer = asyncronous_GPU_transfer, 
                                                      device = device)
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
        print(training_batch_dict['Y'].keys())
        flag_print = False

    ##----------------------------------------------------------------.      
    # Perform autoregressive training loop
    # - The number of AR iterations is determined by AR_scheduler.AR_weights 
    # - If AR_weights are all zero after N forecast iteration:
    #   --> Load data just for F forecast iteration 
    #   --> Autoregress model predictions just N times to save computing time
    dict_training_Y_predicted = {}
    dict_training_loss_per_leadtime = {}
    for i in range(trainingDataset.AR_iterations+1):
        # Retrieve X and Y for current AR iteration
        torch_X, torch_Y = get_AR_batch(AR_iteration = i, 
                                        batch_dict = training_batch_dict, 
                                        dict_Y_predicted = dict_training_Y_predicted,
                                        device = device, 
                                        asyncronous_GPU_transfer = asyncronous_GPU_transfer)
        
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
            trainingDataLoader = AutoregressiveDataLoader(dataset = trainingDataset,                                                   
                                                          batch_size = batch_size,  
                                                          drop_last_batch = drop_last_batch,
                                                          random_shuffle = random_shuffle,
                                                          num_workers = num_workers,
                                                          prefetch_factor = prefetch_factor, 
                                                          prefetch_in_GPU = prefetch_in_GPU,  
                                                          pin_memory = pin_memory,
                                                          asyncronous_GPU_transfer = asyncronous_GPU_transfer, 
                                                          device = device)
            trainingDataLoader_iter = iter(trainingDataLoader)
            count = 0
            flag_print = True
        else:
            flag_break = True
    if flag_break:
        break 

##----------------------------------------------------------------------------.
### Choose dask config for faster lazy loading of xarray  
# https://docs.dask.org/en/latest/scheduling.html 
# https://stackoverflow.com/questions/44193979/how-do-i-run-a-dask-distributed-cluster-in-a-single-thread 
from multiprocessing.pool import ThreadPool
import dask
 
dask.config.set(num_workers=4)
    
dask.config.set(pool=ThreadPool(4))

dask.config.set(scheduler='threads') # 40 secs

dask.config.set(scheduler='processes') # SLOW ! (144 secs)

dask.config.set(scheduler='single-threaded') # (40 secs)

dask.config.set(scheduler='synchronous') # single-threaded synchronous scheduler executes all computations in the local thread with no parallelism at all. 

# Define iterator dataloader 
d_iter = iter(trainingDataLoader)
t_i = time.time()
training_batch_dict = next(d_iter)
time.time() - t_i

# --> dask.config.set(scheduler='synchronous'): fast loading 

# zarr.ThreadSynchronizer()
# zarr.ProcessSynchronizer()

# https://discuss.pytorch.org/t/problems-using-dataloader-for-dask-xarray-netcdf-data/108270
# https://discuss.pytorch.org/t/deadlock-with-dataloader-and-xarray-dask/9387

# The threaded scheduler executes computations with a local multiprocessing.pool.ThreadPool.
# It is lightweight and requires no setup. 
# It introduces very little task overhead (around 50us per task) and,
#  because everything occurs in the same process, it incurs no costs to transfer data between tasks.

##----------------------------------------------------------------------------.