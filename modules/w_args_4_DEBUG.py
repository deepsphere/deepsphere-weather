#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:31:25 2021

@author: ghiggi
"""

preload_data_in_CPU = dataloader_settings['preload_data_in_CPU']
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

preload_data_in_CPU = True
random_shuffle = True
num_workers = 0
pin_memory = False
asyncronous_GPU_transfer = True
# Autoregressive settings  
input_k = [-3,-2,-1]
output_k = [0]
forecast_cycle = 1                         
AR_iterations = 2
stack_most_recent_prediction = True
# Training settings 
learning_rate = 0.001 
batch_size = 128
epochs = 10
training_info = None
numeric_precision = "float64"
# GPU settings 
device = 'cpu'