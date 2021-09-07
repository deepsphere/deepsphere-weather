#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 00:07:05 2021

@author: ghiggi
"""
import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
## TODO:
## - Smooth loss by averaging over X values (rollapply in utils_numpy)
# TODO Loss plot 
# - Add xaxis epochs (as function of iterations) --> self.start_epochs_iterations
# - Add log axis, add exponential notation

# - Add epoch on top x axis  --> using training_info.start_epochs_iterations
#   axis_2 = axis_1.twinx()
#   https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/secondary_axis.html
##----------------------------------------------------------------------------.
### Utils for plotting training info statistics 

def plot_loss(iterations,
              training_loss = None, 
              validation_loss = None, 
              title=None,
              plot_labels = True,
              plot_legend = True, 
              linestyle = "solid",
              linewidth = 0.1,
              xlim=None, ylim=None, 
              ax=None):
    """Plot the loss evolution as function of training iterations."""
    ##------------------------------------------------------------------------.
    # Checks 
    if not isinstance(plot_labels, bool):
        raise TypeError("'plot_labels' must be True or False")
    if not isinstance(plot_legend, bool):
        raise TypeError("'plot_legend' must be True or False")
    ##------------------------------------------------------------------------. 
    # Check not empty list 
    if training_loss is not None: 
        if len(training_loss) == 0: 
            training_loss = None 
    
    if validation_loss is not None: 
        if len(validation_loss) == 0: 
            validation_loss = None
    
    if training_loss is None and validation_loss is None:
        raise ValueError("At least one between 'training_loss' and 'validation_loss' must provided.")
    ##------------------------------------------------------------------------.
    ## Check correct length and define labels 
    labels = []
    if training_loss is not None:
        labels.append('Training Loss')
        if len(iterations) != len(training_loss):
            raise ValueError("'iterations' and 'training_loss' must have same length.")
            
    if validation_loss is not None:
        labels.append('Validation Loss')
        if len(iterations) != len(validation_loss):
            raise ValueError("'iterations' and 'validation_loss' must have same length.")   
            
    ##------------------------------------------------------------------------.
    # Create figure if ax not provided
    if ax is None: 
        fig, ax = plt.subplots()
    ##------------------------------------------------------------------------.
    # Plot loss evolution 
    if training_loss is not None:
        ax.plot(iterations, training_loss, linestyle=linestyle, linewidth=linewidth)
    if validation_loss is not None:
        ax.plot(iterations, validation_loss, linestyle=linestyle, linewidth=linewidth)
    ##------------------------------------------------------------------------.
    # Plot legend 
    if plot_legend: 
        leg = ax.legend(labels=labels, loc='upper right')
        # Make legend line more thick
        for line in leg.get_lines():
            line.set_linewidth(2)

    ##------------------------------------------------------------------------.
    # Plot labels 
    if plot_labels: 
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")  
    ##------------------------------------------------------------------------.
    # Optional limits settings
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ##------------------------------------------------------------------------.
    # Plot title (if specified)
    if title is not None: 
        ax.set_title(title)    
    ##------------------------------------------------------------------------.
    # Return ax 
    return ax 
    ##------------------------------------------------------------------------.
    
#-----------------------------------------------------------------------------.
# ########################### 
### Training Info object ####
# ###########################
class AR_TrainingInfo():
    """Training info Object."""
    
    def __init__(self, ar_iterations, epochs, ar_scheduler):
        # TODO
        # - loss per variable 
        # Initialize training info 
        self.epoch = 0
        self.n_epochs = epochs  
        self.ar_iterations = ar_iterations
        self.validation_stats = False
        ##--------------------------------------------------------------------.
        # Initialize iteration counts
        self.iteration = 0        # track the total number of forward-backward pass
        self.epoch_iteration = 0  # track the iteration number within the epoch
        self.iteration_from_last_ar_update = 0 # track the iteration number from the last AR weight update (for early stopping)
        self.iteration_from_last_scoring = 0  # count the iteration from last scoring  
        self.start_epochs_iterations = [] # list the iterations numbers when a new epoch start
        # Score interval to be inferred when reset_counter() is called 
        self.score_interval = None   # to decide when to score 
        ##--------------------------------------------------------------------.
        # - Initialize dictionary to save the loss at different leadtimes
        # --> Used to analyze the impact of autoregressive weights updates
        training_loss_per_ar_iteration = {}
        validation_loss_per_ar_iteration = {}
        for i in range(ar_iterations + 1):
            training_loss_per_ar_iteration[i] = {}
            training_loss_per_ar_iteration[i]['iteration'] = []
            training_loss_per_ar_iteration[i]['loss'] = []
        for i in range(ar_iterations + 1):
            validation_loss_per_ar_iteration[i] = {}
            validation_loss_per_ar_iteration[i]['iteration'] = []
            validation_loss_per_ar_iteration[i]['loss'] = []
        self.training_loss_per_ar_iteration = training_loss_per_ar_iteration
        self.validation_loss_per_ar_iteration = validation_loss_per_ar_iteration 
        
        ##--------------------------------------------------------------------.
        # - Initialize list for total loss    
        self.iteration_list = []
        self.start_epoch_scoring_idx = 0 # to select score within an epoch
        self.training_total_loss = []
        self.validation_total_loss = [] 
        
        ##--------------------------------------------------------------------.
        # - Initialize list for learning rate
        self.learning_rate_list = []
        
        ##--------------------------------------------------------------------. 
        # - Initialize dictionary for AR weights per leadtime 
        ar_weights_per_ar_iteration = {}
        for i in range(ar_iterations + 1):
            ar_weights_per_ar_iteration[i] = {}
            ar_weights_per_ar_iteration[i]['iteration'] = []
            ar_weights_per_ar_iteration[i]['ar_absolute_weights'] = []
            ar_weights_per_ar_iteration[i]['ar_weights'] = []
        self.ar_weights_per_ar_iteration = ar_weights_per_ar_iteration
        ##--------------------------------------------------------------------. 
        # - Initialize the AR scheduler pickle
        self.ar_scheduler = pickle.dumps(ar_scheduler)
        
    ##------------------------------------------------------------------------.
    def step(self): 
        """Update iteration count."""
        self.iteration = self.iteration + 1
        self.epoch_iteration = self.epoch_iteration + 1

        self.iteration_from_last_ar_update = self.iteration_from_last_ar_update + 1
        self.iteration_from_last_scoring = self.iteration_from_last_scoring + 1
    
    ##------------------------------------------------------------------------.
    def new_epoch(self):
        """Update training_info at the beginning of an epoch."""
        self.start_epoch_iteration = self.iteration
        self.start_epochs_iterations.append(self.iteration)
        self.start_epoch_scoring_idx = len(self.iteration_list)
        self.epoch_iteration = 0
        self.epoch = self.epoch + 1
        self.epoch_time_start = time.time()
        print("")
        print('- Starting training epoch : {}'.format(self.epoch))    
        
    ##------------------------------------------------------------------------.    
    def update_training_stats(self, total_loss,
                              dict_loss_per_ar_iteration,
                              ar_scheduler, lr_scheduler=None):
        """Update training info statistics."""
        # Retrieve current number of AR iterations 
        current_ar_iterations = len(dict_loss_per_ar_iteration) - 1        
        self.ar_iterations = current_ar_iterations
        # Update the iteration_list recording when the updaitte occurs
        self.iteration_list.append(self.iteration)
              
        # Update training_total_loss 
        self.training_total_loss.append(total_loss.item())
        
        # Update learning rate 
        if lr_scheduler is not None:
            self.learning_rate_list.append(lr_scheduler.get_lr())
        
        # Update training_loss_per_ar_iteration
        for i in range(current_ar_iterations+1):
            self.training_loss_per_ar_iteration[i]['iteration'].append(self.iteration)   
            self.training_loss_per_ar_iteration[i]['loss'].append(dict_loss_per_ar_iteration[i].item())
     
        # Update AR weights 
        for i in range(current_ar_iterations+1):
            self.ar_weights_per_ar_iteration[i]['iteration'].append(self.iteration)   
            self.ar_weights_per_ar_iteration[i]['ar_absolute_weights'].append(ar_scheduler.ar_absolute_weights[i])
            self.ar_weights_per_ar_iteration[i]['ar_weights'].append(ar_scheduler.ar_weights[i])
        
        # Pickle AR scheduler
        self.ar_scheduler = pickle.dumps(ar_scheduler)
   
    ##------------------------------------------------------------------------.
    def update_validation_stats(self, total_loss, dict_loss_per_ar_iteration):
        """Update validation loss statistics."""
        self.validation_stats = True
        # Retrieve current number of AR iterations 
        current_ar_iterations = len(dict_loss_per_ar_iteration) - 1
                      
        # Update validation_total_loss 
        self.validation_total_loss.append(total_loss.item())
        
        # Update validation_loss_per_ar_iteration
        for i in range(current_ar_iterations+1):
            self.validation_loss_per_ar_iteration[i]['iteration'].append(self.iteration)   
            self.validation_loss_per_ar_iteration[i]['loss'].append(dict_loss_per_ar_iteration[i].item()) 
            
    ##------------------------------------------------------------------------.
    def reset_counter(self): 
        """Reset iteration counter from last scoring."""
        # Infer the score interval 
        self.score_interval = self.iteration_from_last_scoring
        # Reset iteration counter from last scoring 
        self.iteration_from_last_scoring = 0
     
    def reset_iteration_from_last_ar_update(self):
        """Reset counter of iteration from last AR weight update."""
        # Reset counter of iteration_from_last_ar_update 
        self.iteration_from_last_ar_update = 0
        
    ##------------------------------------------------------------------------.
    def print_epoch_info(self):
        """Print training info at the end of an epoch."""
        avg_training_loss = np.mean(self.training_total_loss[self.start_epoch_scoring_idx:])
        print(" ")
        print("- Epoch: {epoch:3d}/{n_epoch:3d}".format(epoch = self.epoch, n_epoch = self.n_epochs))
        print("- Training loss: {training_total_loss:.3f}".format(training_total_loss=avg_training_loss))
        # If validation data are provided
        if len(self.validation_total_loss) != 0:
            avg_validation_loss = np.mean(self.validation_total_loss[self.start_epoch_scoring_idx:])   
            print("- Validation Loss: {validation_total_loss:.3f}".format(validation_total_loss=avg_validation_loss))
        print("- Elapsed time: {elapsed_time:.2f} minutes".format(elapsed_time = (time.time() - self.epoch_time_start)/60))                                                  

    ##------------------------------------------------------------------------.
    def iterations_of_ar_updates(self):
        """Return iterations at which the number of AR iterations has been increased."""
        iter_ar_update = [self.ar_weights_per_ar_iteration[i]['iteration'][0] for i in range(self.ar_iterations+1)]
        # Remove first scoring iteration 
        iter_ar_update = np.array(iter_ar_update)
        iter_ar_update = iter_ar_update[iter_ar_update > self.score_interval].tolist()
        return iter_ar_update
    
    ##------------------------------------------------------------------------.
    def plot_loss_per_ar_iteration(self, 
                                   ar_iteration,
                                   title=None,
                                   plot_training = True, 
                                   plot_validation = True, 
                                   plot_labels = True,
                                   plot_legend = True, 
                                   add_ar_weights_updates = True, 
                                   linestyle = "solid", 
                                   linewidth = 0.1, 
                                   xlim=None, ylim=None, 
                                   ax=None):
        """
        Plot the loss of a specific AR iteration.

        Parameters
        ----------
        ar_iteration : int
            Number of the autoregressive iteration
        title : str, optional
            Title of the plot. The default is None.
        plot_training : bool, optional
            Whether to plot the training loss. The default is True.
        plot_validation : bool, optional
            Whether to plot the validation loss. The default is True.
        plot_labels : bool, optional
            Whether to plot labels. The default is True.
        plot_legend : TYPE, optional
             Whether to plot the legend. The default is True.
        linestyle : str, optional
             The type of line to draw. The default is solid (solid line).     
        xlim : tuple, optional
            x axis limits provided as (min,max) tuple. The default is None.
        ylim : TYPE, optional
            y axis limits provided as (min,max) tuple. The default is None.
        ax : TYPE, optional
            Matplotlib axis. The default is None.

        Returns
        -------
        If ax argument is not None, return ax 
        If ax argument is None, return a matplotlib.Figure 
        
        """
        ##--------------------------------------------------------------------.
        # Check plot_training and plot_validation
        if not isinstance(plot_training, bool): 
            raise TypeError("'plot_training' must be either True or False.")
        if not isinstance(plot_validation, bool): 
            raise TypeError("'plot_validation' must be either True or False.")
        if ((not plot_validation) and (not plot_training)): 
            raise ValueError("At least one between 'plot_training' and 'plot_validation' must be True.")
        ##--------------------------------------------------------------------.
        # Check ar_iteration
        if not isinstance(ar_iteration, int):
            raise TypeError("'ar_iteration' must be a positive integer.")
        if ar_iteration < 0: 
            raise ValueError("'ar_iteration' must be a positive integer.") 
        if ar_iteration > self.ar_iterations:
            raise ValueError("The  maximum 'ar_iteration' is {}.".format(self.ar_iterations))
        ##--------------------------------------------------------------------.
        # Create figure if ax not provided
        flag_ax_provided = True
        if ax is None: 
            flag_ax_provided = False
            fig, ax = plt.subplots()
        ##--------------------------------------------------------------------. 
        # Retrieve loss data to plot 
        iterations = self.training_loss_per_ar_iteration[ar_iteration]['iteration']
        training_loss =  self.training_loss_per_ar_iteration[ar_iteration]['loss']
        validation_loss = self.validation_loss_per_ar_iteration[ar_iteration]['loss']
        if not plot_training: 
            training_loss = None 
        if not plot_validation: 
            validation_loss = None
        ##--------------------------------------------------------------------. 
        # Plot the loss 
        ax = plot_loss(iterations = iterations, 
                       training_loss = training_loss,  
                       validation_loss = validation_loss,  
                       title=title, 
                       plot_labels = plot_labels,
                       plot_legend = plot_legend, 
                       linestyle = linestyle, 
                       linewidth = linewidth,
                       xlim=xlim, ylim=ylim, 
                       ax=ax)
        ##--------------------------------------------------------------------.
        ## Add vertical line when AR iteration is added
        if add_ar_weights_updates:
            iterations_of_ar_updates = self.iterations_of_ar_updates()
            if len(iterations_of_ar_updates) > 0: 
                [ax.axvline(x=x, color=(0, 0, 0, 0.90), linewidth=0.1) for x in iterations_of_ar_updates]
        ##--------------------------------------------------------------------.
        # If ax not provided to the function, return fig 
        if not flag_ax_provided:
            return fig 
        # Otherwise return ax 
        else: 
            return ax
        ##--------------------------------------------------------------------. 

    def plot_total_loss(self,
                        title="Total Loss Evolution",
                        plot_labels = True,
                        plot_legend = True, 
                        add_ar_weights_updates = True, 
                        linestyle = "solid", 
                        linewidth = 0.1, 
                        xlim=None, ylim=None, 
                        ax=None):
        """
        Plot the training total loss evolution.

        Parameters
        ----------
        title : str, optional
            Title of the plot. The default is None.
        plot_labels : bool, optional
            Whether to plot labels. The default is True.
        plot_legend : TYPE, optional
             Whether to plot the legend. The default is True.
        linestyle : str, optional
             The type of line to draw. The default is solid (solid line).     
        xlim : tuple, optional
            x axis limits provided as (min,max) tuple. The default is None.
        ylim : TYPE, optional
            y axis limits provided as (min,max) tuple. The default is None.
        ax : TYPE, optional
            Matplotlib axis. The default is None.

        Returns
        -------
        If ax argument is not None, return ax 
        If ax argument is None, return a matplotlib.Figure 
        
        """
        ##--------------------------------------------------------------------.
        # Create figure if ax not provided
        flag_ax_provided = True
        if ax is None: 
            flag_ax_provided = False
            fig, ax = plt.subplots()
        ##--------------------------------------------------------------------.    
        # Plot the loss 
        ax = plot_loss(iterations = self.iteration_list, 
                       training_loss = self.training_total_loss,  
                       validation_loss = self.validation_total_loss,
                       title=title, 
                       plot_labels = plot_labels,
                       plot_legend = plot_legend, 
                       linestyle = linestyle, 
                       linewidth = linewidth,
                       xlim=xlim, ylim=ylim, 
                       ax=ax)
        ##--------------------------------------------------------------------.  
        ## Add vertical line when AR iteration is added
        if add_ar_weights_updates:
            iterations_of_ar_updates = self.iterations_of_ar_updates()
            if len(iterations_of_ar_updates) > 0: 
                [ax.axvline(x=x, color=(0, 0, 0, 0.90), linewidth=0.1) for x in iterations_of_ar_updates]
        ##--------------------------------------------------------------------.
        # If ax not provided to the function, return fig 
        if not flag_ax_provided:
            return fig 
        # Otherwise return ax 
        else: 
            return ax
        ##--------------------------------------------------------------------.

    def plot_ar_weights(self, normalized=True, xlim=None, ylim=(0,1.05), ax=None):
        """
        Plot the autoregressive weights evolution.

        Parameters
        ----------
        normalized : bool, optional
            Whether to plot normalized or absolute AR weights. The default is True.
        xlim : tuple, optional
            x axis limits provided as (min,max) tuple. The default is None.
        ylim : TYPE, optional
            y axis limits provided as (min,max) tuple. The default is None.
        ax : TYPE, optional
            Matplotlib axis. The default is None.

        Returns
        -------
        If ax argument is not None, return ax 
        If ax argument is None, return a matplotlib.Figure 
        
        """
        ##--------------------------------------------------------------------.
        # Checks 
        if not isinstance(normalized, bool):
            raise TypeError("'normalized' must be either True or False.")
        ##--------------------------------------------------------------------.
        if ax is None: 
            flag_ax_provided = False
            fig, ax = plt.subplots()
        ##--------------------------------------------------------------------.
        # Set cycling colors and line style
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        custom_cycler = cycler(linestyle=['-', '--', ':', '-.','-', '--', ':', '-.','-', '--'],
                               color=colors)
        ax.set_prop_cycle(custom_cycler)
        ##--------------------------------------------------------------------.
        # Plot absolute AR weights 
        if not normalized:    
            for i in range(self.ar_iterations + 1):
                ax.plot(self.ar_weights_per_ar_iteration[i]['iteration'],
                        self.ar_weights_per_ar_iteration[i]['ar_absolute_weights'],
                        antialiased = True)
            ax.set_ylabel("Absolute weights")
            ax.set_title("Absolute AR weights")  
        ##--------------------------------------------------------------------.
        # Plot normalized AR weights     
        if normalized: 
            for i in range(self.ar_iterations + 1):
                plt.plot(self.ar_weights_per_ar_iteration[i]['iteration'],
                         self.ar_weights_per_ar_iteration[i]['ar_weights'], 
                         antialiased = True)
            ax.set_ylabel("Normalized weights")
            ax.set_title("Normalized AR weights")      
        ##--------------------------------------------------------------------.
        ## Add vertical line when AR iteration is added
        iterations_of_ar_updates = self.iterations_of_ar_updates()
        if len(iterations_of_ar_updates) > 0: 
            [ax.axvline(x=x, color=(0, 0, 0, 0.90), linewidth=0.1) for x in iterations_of_ar_updates]
        ##--------------------------------------------------------------------.
        ## Add xlabel 
        ax.set_xlabel("Iteration")
        ## Add legend 
        ax.legend(labels=list(range(self.ar_iterations + 1)), title="AR iteration", loc='lower left')
        ##--------------------------------------------------------------------.
        # Optional limits settings
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ##--------------------------------------------------------------------.
        # If ax not provided to the function, return fig 
        if not flag_ax_provided:
            return fig 
        # Otherwise return ax 
        else: 
            return ax
        ##--------------------------------------------------------------------.
        
    def plots(self, model_dir=None, ylim=(0,0.06)):
        ##--------------------------------------------------------------------.   
        ### - Plot the loss at all AR iterations (in one figure)
        fig, ax = plt.subplots()
        for ar_iteration in range(self.ar_iterations+1):
            ax = self.plot_loss_per_ar_iteration(ar_iteration = ar_iteration, 
                                                 ax = ax,
                                                 linestyle="solid",
                                                 linewidth=0.3,
                                                 ylim = ylim, 
                                                 plot_validation = False, 
                                                 plot_labels = True,
                                                 plot_legend = False,
                                                 add_ar_weights_updates=False)
        leg = ax.legend(labels=list(range(self.ar_iterations + 1)), 
                        title="AR iteration", 
                        loc='upper right')
        # - Make legend line more thick
        for line in leg.get_lines():
            line.set_linewidth(2)
        # - Reset color cycling 
        plt.gca().set_prop_cycle(None) 
        for ar_iteration in range(self.ar_iterations+1):
            ax = self.plot_loss_per_ar_iteration(ar_iteration = ar_iteration, 
                                                 ax = ax,
                                                 linestyle="dashed",
                                                 linewidth=0.3,
                                                 ylim = ylim, 
                                                 plot_training = False, 
                                                 plot_labels = False,
                                                 plot_legend = False,
                                                 add_ar_weights_updates=False)  
        # - Add vertical line when AR iteration is added
        iterations_of_ar_updates = self.iterations_of_ar_updates()
        if len(iterations_of_ar_updates) > 0: 
            [ax.axvline(x=x, color=(0, 0, 0, 0.90), linewidth=0.1) for x in iterations_of_ar_updates]
        # - Add title 
        ax.set_title("Loss evolution at each AR iteration")
        # - Save figure
        if model_dir is not None:
            fig.savefig(os.path.join(model_dir, "figs/training_info/Loss_at_all_ar_iterations.png")) 
        else:
            plt.show()
        ##--------------------------------------------------------------------.   
        ### - Plot the loss at each AR iteration (in separate figures)
        for ar_iteration in range(self.ar_iterations+1):
            fig = self.plot_loss_per_ar_iteration(ar_iteration = ar_iteration,
                                                  linewidth=0.6,
                                                  ylim = ylim,
                                                  title="Loss evolution at AR iteration {}".format(ar_iteration))
            if model_dir is not None:
                fname = os.path.join(model_dir, "figs/training_info/Loss_at_ar_{}.png".format(ar_iteration)) 
                fig.savefig(fname) 
            else:
                plt.show()
        ##--------------------------------------------------------------------.
        ### - Plot total loss 
        fig = self.plot_total_loss(ylim = ylim, linewidth=0.6)
        if model_dir is not None:
            fig.savefig(os.path.join(model_dir, "figs/training_info/Total_Loss.png"))
        else:
            plt.show()
        
        ##--------------------------------------------------------------------.
        ### - Plot AR weights normalized 
        fig = self.plot_ar_weights(normalized=True)
        if model_dir is not None:
            fig.savefig(os.path.join(model_dir, "figs/training_info/ar_Normalized_Weights.png"))
        else:
            plt.show()
        ### - Plot absolute AR weights  
        fig = self.plot_ar_weights(normalized=False)
        if model_dir is not None:
            fig.savefig(os.path.join(model_dir, "figs/training_info/ar_Absolute_Weights.png")) 
        else:
            plt.show()
        ##--------------------------------------------------------------------.
##----------------------------------------------------------------------------.
