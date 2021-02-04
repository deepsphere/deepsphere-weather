#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 00:07:05 2021

@author: ghiggi
"""
import torch
import numpy as np
import random 
import time
##----------------------------------------------------------------------------.
############################# 
### Training Info object ####
#############################
class TrainingInfo():
    """Training info Object."""
    
    def __init__(self, AR_iterations, epochs):
        # TODO
        # - loss per variable 
        # Initialize training info 
        self.current_epoch = 0
        self.n_epochs = epochs  
        self.AR_iterations = AR_iterations
        #---------------------------------------------------------------------.
        # Initialize iteration counts
        self.iteration = 0        # to keep track of the total number of forward-backward pass
        self.current_epoch_iteration = 0  # to keep track of iteration within the epoc
        self.score_interval = 0   # to decide when to score 
        #---------------------------------------------------------------------.
        # - Initialize dictionary to save the loss at different leadtimes
        # --> Used to analyze the impact of autoregressive weights updates
        training_loss_per_leadtime = {}
        validation_loss_per_leadtime = {}
        for i in range(AR_iterations + 1):
            training_loss_per_leadtime[i] = {}
            training_loss_per_leadtime[i]['iteration'] = []
            training_loss_per_leadtime[i]['loss'] = []
        for i in range(AR_iterations + 1):
            validation_loss_per_leadtime[i] = {}
            validation_loss_per_leadtime[i]['iteration'] = []
            validation_loss_per_leadtime[i]['loss'] = []
        self.training_loss_per_leadtime = training_loss_per_leadtime
        self.validation_loss_per_leadtime = validation_loss_per_leadtime 
        
        #---------------------------------------------------------------------.
        # - Initialize list for total loss    
        self.iteration_list = []
        self.training_total_loss = []
        self.validation_total_loss = [] 
        
        #---------------------------------------------------------------------.
        # - Initialize list for learning rate
        self.learning_rate_list = []
        
        #---------------------------------------------------------------------. 
        # - Initialize dictionary for AR weights per leadtime 
        AR_weights_per_leadtime = {}
        for i in range(AR_iterations + 1):
            AR_weights_per_leadtime[i] = {}
            AR_weights_per_leadtime[i]['iteration'] = []
            AR_weights_per_leadtime[i]['AR_absolute_weights'] = []
            AR_weights_per_leadtime[i]['AR_weights'] = []
        self.AR_weights_per_leadtime = AR_weights_per_leadtime
        
        #---------------------------------------------------------------------. 

    def step(self): 
        """Update iteration count."""
        self.iteration = self.iteration + 1
        self.current_epoch_iteration = self.current_epoch_iteration + 1
        self.score_interval = self.score_interval + 1
        
    def update_training_stats(self, total_loss,
                              dict_loss_per_leadtime,
                              AR_scheduler, LR_scheduler=None):
        """Update training info statistics."""
        # Retrieve current number of AR iterations 
        current_AR_iterations = len(dict_loss_per_leadtime)
        
        # Update the iteration_list recording when the update occurs
        self.iteration_list.append(self.iteration)
              
        # Update training_total_loss 
        self.training_total_loss.append(total_loss.item())
        
        # Update learning rate 
        if LR_scheduler is not None:
            self.learning_rate_list.append(LR_scheduler.get_lr())
        
        # Update training_loss_per_leadtime
        for i in range(current_AR_iterations+1):
            self.training_loss_per_leadtime[i]['iteration'].append(self.iteration)   
            self.training_loss_per_leadtime[i]['loss'].append(dict_loss_per_leadtime[i].item())
     
        # Update AR weights 
        for i in range(current_AR_iterations+1):
            self.AR_weights_per_leadtime[i]['iteration'].append(self.iteration)   
            self.AR_weights_per_leadtime[i]['AR_absolute_weights'].append(AR_scheduler.AR_absolute_weights[i])
            self.AR_weights_per_leadtime[i]['AR_weights'].append(AR_scheduler.AR_weights[i])
        
    def update_validation_stats(self, total_loss, dict_loss_per_leadtime):
        """Update validation loss statistics."""
        # Retrieve current number of AR iterations 
        current_AR_iterations = len(dict_loss_per_leadtime)
                      
        # Update validation_total_loss 
        self.validation_total_loss.append(total_loss.item())
        
        # Update validation_loss_per_leadtime
        for i in range(current_AR_iterations+1):
            self.validation_loss_per_leadtime[i]['iteration'].append(self.iteration)   
            self.validation_loss_per_leadtime[i]['loss'].append(dict_loss_per_leadtime[i].item()) 
    
    def reset_score_interval(self): 
        """Reset score_interval count."""
        # Reset score_interval  
        self.score_interval = 0 
  
    def new_epoch(self):
        """Update training_info at the beginning of an epoch."""
        self.current_epoch_iteration = 0
        self.current_epoch = self.current_epoch + 1
        self.current_epoch_time_start = time.time()
        print('Starting training epoch : {}'.format(self.current_epoch), end="")
        
    def print_epoch_info(self):
        """Print training info at the end of an epoch."""
        avg_training_loss = np.mean(self.training_total_loss[-self.epoch_iteration:])
        # If validation data are provided
        if len(self.validation_total_loss) != 0:
            avg_validation_loss = np.mean(self.validation_total_loss[-self.epoch_iteration:])
        
            print('Epoch: {epoch:3d}/{n_epoch:3d} - \
                   Training loss: {training_total_loss:.3f} - \
                   Validation Loss: {validation_total_loss:.5f} - \
                   Time: {elapsed_time:2f}'.format(epoch = self.current_epoch, 
                                                   n_epoch = self.n_epochs,
                                                   training_total_loss=avg_training_loss, 
                                                   validation_total_loss=avg_validation_loss,   
                                                   elapsed_time = time.time() - self.current_epoch_time_start))
        # If only training data are available 
        else: 
            print('Epoch: {epoch:3d}/{n_epoch:3d} - \
                   Training loss: {training_total_loss:.3f} - \
                   Time: {elapsed_time:2f}'.format(epoch = self.current_epoch, 
                                                   n_epoch = self.n_epochs,
                                                   training_total_loss = avg_training_loss,   
                                                   elapsed_time = time.time() - self.current_epoch_time_start))
            
##----------------------------------------------------------------------------.         
####################### 
### Early stopping ####
#######################    