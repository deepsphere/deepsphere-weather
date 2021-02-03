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
        stopping_metric : str, optional
            Either 'training_total_loss' or 'validation_total_loss'.
            The default is 'training_total_loss'.
        mode : str, optional
            Whether to look for a minimal ('min') or maximal ('max') validation loss.
            The default is 'min'.

        """
        if not isinstance(patience, int):
            raise ValueError("'patience' requires a positive integer larger than 1")
        if patience < 1:
            raise ValueError("'patience' requires a positive integer larger than 1")
        if not isinstance(mode, str):  
            raise ValueError("'mode' has to be either 'min' or 'max' string") 
        if mode not in ['min', 'max']:
            raise ValueError("'mode' has to be either 'min' or 'max' string")

        self.patience = patience
        self.stopping_metric = stopping_metric
        self.minimum_improvement = minimum_improvement
        self.mode = mode

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, training_info):
        """Call to verify if training must stop."""
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