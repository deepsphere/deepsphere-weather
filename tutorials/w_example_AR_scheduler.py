#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 17:33:38 2021

@author: ghiggi
"""
import os 
import numpy as np
import matplotlib.pyplot as plt
os.chdir("/home/ghiggi/Projects/weather_prediction/")
from modules.AR_Scheduler import AR_Scheduler
from modules.AR_Scheduler import plot_AR_scheduler

plot_absolute_AR_weights = True
plot_normalized_AR_weights = True
smooth_growth = True 
n_updates=4
update_every=1000
                      
## Constant 
ar_scheduler = AR_Scheduler(method = "Constant") # OK 
plot_AR_scheduler(ar_scheduler,
                  n_updates = n_updates, 
                  update_every = update_every, 
                  plot_absolute_AR_weights = plot_absolute_AR_weights,
                  plot_normalized_AR_weights = plot_normalized_AR_weights)

## DiracDelta
ar_scheduler = AR_Scheduler(method = "DiracDelta") # OK
plot_AR_scheduler(ar_scheduler,
                  n_updates = n_updates, 
                  update_every = update_every, 
                  plot_absolute_AR_weights = plot_absolute_AR_weights,
                  plot_normalized_AR_weights = plot_normalized_AR_weights)

##----------------------------------------------------------------------------.
## HalfStep
ar_scheduler = AR_Scheduler(method = "HalfStep",
                            smooth_growth = smooth_growth,    
                            factor = 0.02, 
                            step_interval = 10)    # buggy... 
plot_AR_scheduler(ar_scheduler,
                  n_updates = n_updates, 
                  update_every = 100, 
                  plot_absolute_AR_weights = plot_absolute_AR_weights,
                  plot_normalized_AR_weights = plot_normalized_AR_weights)

# StepwiseStep
ar_scheduler = AR_Scheduler(method = "StepwiseStep",
                            smooth_growth = smooth_growth,     
                            factor = 0.2,
                            step_interval = 20)
plot_AR_scheduler(ar_scheduler,
                  n_updates = n_updates, 
                  update_every = 100, 
                  plot_absolute_AR_weights = plot_absolute_AR_weights,
                  plot_normalized_AR_weights = plot_normalized_AR_weights)

##----------------------------------------------------------------------------.
## Linear Step
ar_scheduler = AR_Scheduler(method = "LinearStep", 
                            smooth_growth = smooth_growth,
                            factor = 0.0005)
plot_AR_scheduler(ar_scheduler, 
                  n_updates = n_updates, 
                  update_every = 3000,  # --> min_iteration in Early stopping ... 
                  plot_absolute_AR_weights = plot_absolute_AR_weights,
                  plot_normalized_AR_weights = plot_normalized_AR_weights)

## Exponential Step
ar_scheduler = AR_Scheduler(method = "ExponentialStep",
                            smooth_growth = smooth_growth, 
                            factor = 0.01)           
plot_AR_scheduler(ar_scheduler, 
                  n_updates = n_updates, 
                  update_every = update_every, 
                  plot_absolute_AR_weights = plot_absolute_AR_weights,
                  plot_normalized_AR_weights = plot_normalized_AR_weights)

##----------------------------------------------------------------------------.
## Multiple starting weights 
ar_scheduler = AR_Scheduler(method = "LinearStep",
                            smooth_growth = smooth_growth, 
                            factor = 0.001,
                            fixed_AR_weights = [0,1,4,6],
                            initial_AR_absolute_weights=[1,1,0,0,0.0001]) 
for i in range(0,10): 
    # print(ar_scheduler.AR_weights)
    print(ar_scheduler.AR_absolute_weights)
    ar_scheduler.step()

ar_scheduler = AR_Scheduler(method = "LinearStep",
                            smooth_growth = smooth_growth, 
                            factor = 0.001,
                            fixed_AR_weights = [0,1,4,6],
                            initial_AR_absolute_weights=[1,1]) 
          
plot_AR_scheduler(ar_scheduler, 
                  n_updates = 4, 
                  update_every = 1500, 
                  plot_absolute_AR_weights = plot_absolute_AR_weights,
                  plot_normalized_AR_weights = plot_normalized_AR_weights)

ar_scheduler = AR_Scheduler(method = "LinearStep",
                            smooth_growth = smooth_growth, 
                            factor = 0.001,
                            fixed_AR_weights = [0,1,2,3,4,5,6],
                            initial_AR_absolute_weights=[1]) 
          
plot_AR_scheduler(ar_scheduler, 
                  n_updates = 4, 
                  update_every = 1500, 
                  plot_absolute_AR_weights = plot_absolute_AR_weights,
                  plot_normalized_AR_weights = plot_normalized_AR_weights)

##----------------------------------------------------------------------------.

### AR methods methods
AR_scheduler = AR_Scheduler(method = "LinearStep",
                            factor = 0.1)
AR_scheduler.step()   # Step previous AR weights
AR_scheduler.update() # add new AR weight
AR_scheduler.AR_weights # AR weights for the loss function 