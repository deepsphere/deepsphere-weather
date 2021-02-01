#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 20:00:18 2021

@author: ghiggi
"""
import numpy as np
import torch
    

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    
    Inspired by:
        https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
        https://github.com/pytorch/ignite/blob/master/ignite/handlers/early_stopping.py
        
    """
    def __init__(self, 
                 patience = 10,
                 minimum_improvement = 0.01,
                 stopping_metric = 'validation_total_loss',                                                         
                 mode = "min"):  
        """
        # stopping patience: the number of scoring events with no improvement before stop training
        # minimum_improvement: Minimum change in the monitored quantity to qualify as an improvement
        # mode : whether to look for a minimal or maximal validation loss
   
        """
        if patience < 1:
            raise ValueError("Stopping patience should be superior or equal to 1.")
        
        if mode not in ['min', 'max']:
            raise ValueError("Mode has to be either min or max.")

        self.patience = patience
        self.stopping_metric = stopping_metric
        self.minimum_improvement = minimum_improvement
        self.mode = mode

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        
    def __call__(self, training_info):
        score = getattr(training_info, self.stopping_metric)[-1]

        if self.best_score is None:
            self.best_score = score

        elif (score > self.best_score - self.minimum_improvement and self.mode=='min') or \
            (score < self.best_score + self.minimum_improvement and self.mode=='max'):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop
    
    
