#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 17:14:20 2021

@author: ghiggi
"""
## Check with lazy xarray 
dict_DataArrays = reformat_Datasets(ds_training_dynamic = ds_training_dynamic,
                                    ds_validation_dynamic = ds_validation_dynamic,
                                    ds_static = ds_static,              
                                    ds_training_bc = ds_training_bc,         
                                    ds_validation_bc = ds_validation_bc,
                                    preload_data_in_CPU = False)

da_static = dict_DataArrays['da_static']
da_training_dynamic = dict_DataArrays['da_training_dynamic']
da_validation_dynamic = dict_DataArrays['da_validation_dynamic']
da_training_bc = dict_DataArrays['da_training_bc']
da_validation_bc = dict_DataArrays['da_validation_bc']



input_k = check_input_k(input_k=input_k, AR_iterations=AR_iterations)   
output_k = check_output_k(output_k=output_k)
num_workers = 0
AR_iterations = 9



trainingDataset = AutoregressiveDataset(da_dynamic = da_training_dynamic,  
                                        da_bc = da_training_bc,
                                        da_static = da_static,
                                        # Autoregressive settings  
                                        input_k = input_k,
                                        output_k = output_k,
                                        forecast_cycle = forecast_cycle,                           
                                        AR_iterations = AR_iterations,
                                        max_AR_iterations = AR_iterations,
                                        stack_most_recent_prediction = stack_most_recent_prediction, 
                                        # GPU settings 
                                        device = device,
                                        # Precision settings
                                        numeric_precision = numeric_precision)
trainingDataLoader = AutoregressiveDataLoader(dataset = trainingDataset,                                                   
                                              batch_size = 1,  
                                              drop_last_batch = drop_last_batch,
                                              random_shuffle = random_shuffle,
                                              num_workers = num_workers,
                                              prefetch_factor = prefetch_factor, 
                                              prefetch_in_GPU = prefetch_in_GPU,  
                                              pin_memory = pin_memory,
                                              asyncronous_GPU_transfer = asyncronous_GPU_transfer, 
                                              device = device)

# Dataset works 
da_training_dynamic.shape

trainingDataset[1] 

# Define iterator dataloader 
d_iter = iter(trainingDataLoader)
t_i = time.time()
training_batch_dict = next(d_iter)
time.time() - t_i

# trainingDataset.update_AR_iterations(5)

## Check update AR iterations works

training_batch_dict = next(d_iter)
print(len(training_batch_dict['Y']))
trainingDataset.update_AR_iterations(2)
training_batch_dict = next(d_iter)
print(len(training_batch_dict['Y']))
trainingDataset.update_AR_iterations(5)
training_batch_dict = next(d_iter)
print(len(training_batch_dict['Y']))
trainingDataset.update_AR_iterations(2)
training_batch_dict = next(d_iter)
print(len(training_batch_dict['Y']))
trainingDataset.update_AR_iterations(8)
training_batch_dict = next(d_iter)
print(len(training_batch_dict['Y']))

# Check batching and stacking works 
AR_iterations = trainingDataset.AR_iterations 
for j in range(10):   
    
    training_batch_dict = next(d_iter)
    print(".", end="")
    ##----------------------------------------------------------------.      
    # Perform autoregressive training loop
    # - The number of AR iterations is determined by AR_scheduler.AR_weights 
    # - If AR_weights are all zero after N forecast iteration:
    #   --> Load data just for F forecast iteration 
    #   --> Autoregress model predictions just N times to save computing time
    dict_training_Y_predicted = {}
    dict_training_loss_per_leadtime = {}
    for i in range(AR_iterations+1):
        print(" ", i)
        # Retrieve X and Y for current AR iteration
        torch_X, torch_Y = get_AR_batch(AR_iteration = i, 
                                        batch_dict = training_batch_dict, 
                                        dict_Y_predicted = dict_training_Y_predicted,
                                        device = device, 
                                        asyncronous_GPU_transfer = asyncronous_GPU_transfer)
        print(torch_X.shape)
        ##------------------------------------------------------------.
        # Forward pass and store output for stacking into next AR iterations
        dict_training_Y_predicted[i] = torch_Y
        
        
    

# pdb.set_trace()

for i in trainingDataLoader:
    print(len(i['Y']))

# Check valid dataloader
a = next(validationDataLoader_iter)


### Optimize for loading with xarray dask (lazy)

from multiprocessing.pool import ThreadPool
import dask
 

dask.config.set(num_workers=4):
    
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

# https://discuss.pytorch.org/t/problems-using-dataloader-for-dask-xarray-netcdf-data/108270
# https://discuss.pytorch.org/t/deadlock-with-dataloader-and-xarray-dask/9387

# The threaded scheduler executes computations with a local multiprocessing.pool.ThreadPool.
# It is lightweight and requires no setup. 
# It introduces very little task overhead (around 50us per task) and,
#  because everything occurs in the same process, it incurs no costs to transfer data between tasks.
