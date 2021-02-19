#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 20:00:18 2021

@author: ghiggi
"""        
  
class EarlyStopping:
    """Provide functionality for early stopping network training."""
  
    def __init__(self, 
                 patience = 10,
                 minimum_improvement = 0.001,
                 minimum_iterations = 500,
                 stopping_metric = 'training_total_loss',                                                         
                 mode = "min"):  
        """
        Initiate an EarlyStopping object.
        
        It allow to stops training if the stopping_metric doesn't improve after a
        given number of scoring rounds.

        Parameters
        ----------
        patience : int, optional
            The number of scoring events with no improvement before stop training. 
            The default is 10.
        minimum_improvement : float, optional
            Minimum change in the monitored quantity to qualify as an improvement
            The default is 0.001.
        minimum_iterations : float, optional
            Minimum number of training iterations before turning on 
            early stopping.  
            The default is 1000.    
        stopping_metric : str, optional
            Either 'training_total_loss' or 'validation_total_loss'.
            The default is 'training_total_loss'.
        mode : str, optional
            Whether to look for a minimal ('min') or maximal ('max') validation loss.
            The default is 'min'.

        """
        if not isinstance(stopping_metric, str):
            raise TypeError("'stopping_metric' must be a string.")  
        if not isinstance(patience, int):
            raise TypeError("'patience' must be a positive integer larger than 1.")
        if patience < 1:
            raise ValueError("'patience' must be a positive integer larger than 1.")
        if not isinstance(minimum_improvement, (int, float)):
            raise TypeError("'minimum_improvement' must be a value (int or float).")
        if isinstance(minimum_iterations, type(None)):
            minimum_iterations = 0 
        if not isinstance(minimum_iterations, int):
            raise TypeError("'minimum_iterations' must be an integer larger than 0.")
        if minimum_iterations < 0:
            raise ValueError("'minimum_iterations' must be an integer larger than 0.")
        if not isinstance(mode, str):  
            raise ValueError("'mode' has to be either 'min' or 'max' string.") 
        if mode not in ['min', 'max']:
            raise ValueError("'mode' has to be either 'min' or 'max' string.")
        ##--------------------------------------------------------------------.
        # Initialize early stopping             
        self.patience = patience
        self.stopping_metric = stopping_metric
        self.minimum_improvement = minimum_improvement
        self.minimum_iterations = minimum_iterations
        self.mode = mode

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, training_info):
        """Call to verify if training must stop."""
        stopping_metric = self.stopping_metric
        score = getattr(training_info, stopping_metric)[-1]
        iteration = getattr(training_info, "iteration")
        
        if iteration > self.minimum_iterations:
            if self.best_score is None:
                self.best_score = score
    
            elif ((score > (self.best_score - self.minimum_improvement)) and (self.mode=='min')) or \
                 ((score < (self.best_score + self.minimum_improvement)) and (self.mode=='max')):
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        
        return self.early_stop
    
    def reset(self):
        """Reset the counter."""
        self.counter = 0