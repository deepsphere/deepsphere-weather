#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 17:33:38 2021

@author: ghiggi
"""
import os 
import numpy as np
import matplotlib.pyplot as plt
os.chdir("/home/ghiggi/Projects/DeepSphere/")
from modules.AR_Scheduler import AR_Scheduler
from modules.AR_Scheduler import plot_AR_scheduler

plot_absolute_AR_weights = False
plot_normalized_AR_weights = True

## Constant 
AR_scheduler = AR_Scheduler(method = "Constant") # OK 
plot_AR_scheduler(AR_scheduler,
                  plot_absolute_AR_weights = plot_absolute_AR_weights,
                  plot_normalized_AR_weights = plot_normalized_AR_weights)

## DiracDelta
AR_scheduler = AR_Scheduler(method = "DiracDelta") # OK
plot_AR_scheduler(AR_scheduler,
                  plot_absolute_AR_weights = plot_absolute_AR_weights,
                  plot_normalized_AR_weights = plot_normalized_AR_weights)

## HalfDecay
AR_scheduler = AR_Scheduler(method = "HalfDecay",
                            step_interval = 1)
plot_AR_scheduler(AR_scheduler,
                  update_every=15,
                  plot_absolute_AR_weights = plot_absolute_AR_weights,
                  plot_normalized_AR_weights = plot_normalized_AR_weights)

# StepwiseDecay
AR_scheduler = AR_Scheduler(method = "StepwiseDecay",
                            factor = 0.2,
                            step_interval = 5)
plot_AR_scheduler(AR_scheduler,
                  update_every=15,
                  plot_absolute_AR_weights = plot_absolute_AR_weights,
                  plot_normalized_AR_weights = plot_normalized_AR_weights)

## Linear Decay
AR_scheduler = AR_Scheduler(method = "LinearDecay",
                            factor = 0.025)
plot_AR_scheduler(AR_scheduler, 
                  update_every=15,
                  plot_absolute_AR_weights = plot_absolute_AR_weights,
                  plot_normalized_AR_weights = plot_normalized_AR_weights)

## Exponential Decay
AR_scheduler = AR_Scheduler(method = "ExponentialDecay",
                            factor = 0.3)
plot_AR_scheduler(AR_scheduler, 
                  update_every=15,
                  plot_absolute_AR_weights = plot_absolute_AR_weights,
                  plot_normalized_AR_weights = plot_normalized_AR_weights)


# AR_scheduler.AR_absolute_weights
# AR_scheduler.step()
# AR_scheduler.AR_absolute_weights
# AR_scheduler.update()
# AR_scheduler.AR_absolute_weights
# AR_scheduler.step()
# AR_scheduler.AR_absolute_weights
# AR_scheduler.step()

##----------------------------------------------------------------------------.
AR_scheduler = AR_Scheduler(method = "LinearDecay",
                            factor = 0.1)
AR_scheduler.step()   # decay previous AR weights
AR_scheduler.update() # add new AR weight

AR_scheduler.AR_weights # AR weights for the loss function 