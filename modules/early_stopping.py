#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 20:00:18 2021

@author: ghiggi
"""
import numpy as np
import torch

 

a = np.arange(40)    

sub = a[-((stopping_rounds)+stopping_rounds):]
    

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    
    Inspired by:
        https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
        https://github.com/pytorch/ignite/blob/master/ignite/handlers/early_stopping.py
        
    """
    def __init__(self, 
                 stopping_patience = 10,
                 stopping_rounds = stopping_rounds,
                 stopping_tolerance = stopping_tolerance,
                 minimum_improvement = minimum_improvement,
                 stopping_metric = stopping_metric,                                                         
                 mode = "min"):
                 
                 patience=7,  
        """
        # stopping_rounds: Stops training when the (scored) loss function (or specified metric) 
        #  doesn’t improve after x stopping_rounds based on a simple moving average. 
        #  The metric is computed on the validation data (if provided),
        #   otherwise, training data is used. 
        # stopping_tolerance = 1e-3
        
        #  - the moving average for last (stopping_round + 1) stopping rounds is calculated  
        #  - the first moving average is reference value 
        #  - the other "stopping_round" 3 moving averages to compare
        
        # The model will stop if the ratio between the best moving average and 
        # reference moving average
        # - (for metrics that low is better)  is more or equal 'stopping_tolerance'
        # - (for metrics that large is better) is less or equal 'stopping_tolerance'
        
        ## Alternative : 
        # stopping patience: the number of scoring events with no improvement before stop training
        # delta / minimum_improvement: Minimum change in the monitored quantity
           to qualify as an improvement
   
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        
    def __call__(self, training_info):
        # avg_validation_loss = np.mean(self.validation_total_loss[-self.epoch_iteration:])
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return True
