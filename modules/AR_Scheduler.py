#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 11:31:21 2021

@author: ghiggi
"""
import numpy as np 
import matplotlib.pyplot as plt

##----------------------------------------------------------------------------.
# No weight update functions
def _ConstantStep(self): 
    return self.AR_weights 

def _DiracDeltaStep(self):
    return self.AR_weights  

##----------------------------------------------------------------------------.
## Intermittent weight update functions 
def _StepwiseDecayStep(self):                 
    if self.temporary_step_count >= self.step_interval: 
        weights = self.AR_absolute_weights[:-1]
        weights = weights - self.factor 
        weights[weights < 0] = 0
        self.AR_absolute_weights[:-1] = weights 
        # Reset temporary_step_count 
        self.temporary_step_count = 0    
 
def _HalfDecayStep(self):
    if self.temporary_step_count >= self.step_interval: 
        weights = self.AR_absolute_weights[:-1]
        weights = weights/2  
        self.AR_absolute_weights[:-1] = weights 
        # Reset temporary_step_count 
        self.temporary_step_count = 0    

##----------------------------------------------------------------------------.
### Continous weight update functions
def _LinearDecayStep(self): 
    initial_weights = self.AR_absolute_initial_weights[:-1]
    weights = initial_weights - self.factor*self.global_step_count_arr
    weights[weights < 0] = 0
    self.AR_absolute_weights[:-1] = weights 

def _ExponentialDecayStep(self):
    initial_weights = self.AR_absolute_initial_weights[:-1]
    weights = initial_weights * np.exp(-self.factor*self.global_step_count_arr)
    self.AR_absolute_weights[:-1] = weights 
 
#----------------------------------------------------------------------------.    
class AR_Scheduler():
    """Autoregressive (AR) weights scheduler."""
    
    def __init__(self, 
                 method = "DiracDelta", 
                 factor = None,
                 step_interval = None, 
                 initial_AR_weights = [1]):
        """Autoregressive (AR) weights scheduler.

        Parameters
        ----------
        method : str, optional
            Available methods: 'Constant','DiracDelta','StepwiseDecay','HalfDecay','LinearDecay','ExponentialDecay'
            The default method is "DiracDelta".
            
            Methods explanation:
                
            Constant: Add an AR weight (with absolute value 1) when .update() is called.
            
            DiracDelta: Add an AR weight when .update() is called and 
                        reset to 0 the others AR weights. 
                         
            StepwiseDecay: When a new AR weight is added with .update(), it start to substract  
                          'factor' from the others AR weights every 'step_interval' .step() calls.
            
            HalfDecay: When a new AR weight is added with .update(), it start to half   
                       the others AR weights every 'step_interval' .step() calls.
            
            LinearDecay : When a new AR weight is added with .update(), it start to  
                          decrease linearly (with slope '-factor') the others 
                          AR weights every .step() call.    
                          
            ExponentialDecay: When a new AR weight is added with .update(), it start to  
                              decrease exponentially (with decay rate '-factor') 
                              the others AR weights every .step() call.    
            
        factor : float, optional
            Argument required by the following methods: 'StepwiseDecay','HalfDecay','LinearDecay','ExponentialDecay'.
            Regulate the decay of AR weights when .step() is called.

        step_interval : int, optional
            Argument required by the following methods: 'StepwiseDecay','HalfDecay'.
            Specify the frequency with which the AR weights are updated with methods 'StepwiseDecay' and 'HalfDecay'.
            Step_interval = 1 cause weight update at every .step() call. 
            
        initial_AR_weights : list, optional
            Specify the initial (normalized) AR weights. By default initial_AR_weights is set to [1].
            The initial_AR_weights must sum up to 1!
            
        """
        # 'StepwiseDecay' and 'HalfDecay' factor is applied to the AR_absolute weights (not the normalized AR_weights)
        # 'LinearDecay','ExponentialDecay' is applied from the initial AR_absolute_weights
        # TODO: 
        # - Implement a min_AR_weight_option? (instead of decaying to 0) 
        #---------------------------------------------------------------------.
        # Check valid method 
        valid_method = ['Constant','DiracDelta','StepwiseDecay','HalfDecay','LinearDecay','ExponentialDecay']
        if method not in valid_method:
            raise ValueError("Provide a valid 'method'.")    
        # Check initial_AR_weights
        if isinstance(initial_AR_weights, list):
            initial_AR_weights = np.array(initial_AR_weights)
        if not isinstance(initial_AR_weights, np.ndarray):
            raise TypeError("Specify 'initial_AR_weights' with a list or a numpy array.")
        if np.sum(initial_AR_weights) != 1:
            raise ValueError("'initial_AR_weights' must sum up to 1.")
        # Check that any intial_AR_weights is negative 
        if any(initial_AR_weights < 0):
            raise ValueError("'initial_AR_weights' must not contain negative weights.")
        # Check that the last AR weight is not zero !
        if initial_AR_weights[-1] == 0:
            raise ValueError("The last weight of 'initial_AR_weights' must not be 0.")
        ##--------------------------------------------------------------------.
        # Check that factor and step_interval are not negative 
        if factor is not None: 
            if factor < 0:
                raise ValueError("Provide a factor between 0 and 1.")
        if step_interval is not None: 
            if step_interval <= 0: 
                raise ValueError("'step_interval' must be an integer value equal or larger than 1.")    
        #---------------------------------------------------------------------.
        # Check required arguments are specified 
        if method in ['StepwiseDecay','HalfDecay']:
            if step_interval is None: 
                raise ValueError("'{}' method requires specification of the 'step_interval' argument".format(method))
        if method in ['StepwiseDecay','LinearDecay','ExponentialDecay']:
            if factor is None:
                raise ValueError("'{}' method requires specification of the 'factor' argument".format(method))
        #---------------------------------------------------------------------. 
        current_AR_iterations = np.max(np.where(initial_AR_weights > 0)) 
        # Retrieve absolute AR weights  
        self.AR_absolute_weights = initial_AR_weights*1/initial_AR_weights[-1]
        self.AR_absolute_initial_weights = self.AR_absolute_weights # for 'LinearDecay' and 'ExponentialDecay'
        # Set AR_weights (normalized AR weights)
        self.AR_weights = initial_AR_weights
        # Count the number of AR iteration (at start)
        self.current_AR_iterations = current_AR_iterations
        # Add method arguments
        self.method = method
        self.step_interval = step_interval
        self.factor = factor
        # Initialize temporary step counter 
        # - For 'StepwiseDecay' and 'HalfDecay' method --> step_interval
        self.temporary_step_count = 0 
        # - Initialize global step counter 
        # - For 'LinearDecay' and 'ExponentialDecay'
        # - Do not include the last weight 
        if current_AR_iterations == 0:   
            self.global_step_count_arr = np.array([])
        else: 
            self.global_step_count_arr = np.zeros(current_AR_iterations)
        ##--------------------------------------------------------------------.
        ### Define the update_weights function 
        fun_dict = {'Constant': _ConstantStep,
                    'DiracDelta': _DiracDeltaStep,
                    'StepwiseDecay': _StepwiseDecayStep,
                    'HalfDecay': _HalfDecayStep,
                    'LinearDecay': _LinearDecayStep,
                    'ExponentialDecay': _ExponentialDecayStep,
                    }          
        self.update_weights = fun_dict[method]
        ##--------------------------------------------------------------------.
   
    def step(self):
        """Update AR weights."""
        # Update step count 
        self.temporary_step_count = self.temporary_step_count + 1    # for 'StepwiseDecay' and 'HalfDecay'
        self.global_step_count_arr = self.global_step_count_arr + 1  # for 'LinearDecay' and 'ExponentialDecay'
        if self.current_AR_iterations > 0:
            # Decrease weights of AR iterations (except the last)
            self.update_weights(self)
            # Retrieve AR weights (summing up to 1)
            self.AR_weights = np.array(self.AR_absolute_weights)/np.sum(self.AR_absolute_weights)
    
    def update(self):
        """Add an AR_absolute_weight with value 1."""
        # Add a new AR weight with (absolute) value 1
        self.AR_absolute_weights = np.append(self.AR_absolute_weights, 1)
        self.AR_absolute_initial_weights = np.append(self.AR_absolute_initial_weights, 1)
        # If DiracDelta weight update method is choosen, set to 0 the other weights 
        if self.method == "DiracDelta":
            self.AR_absolute_weights[:-1] = 0 
        # Update normalization of AR weights
        self.AR_weights = np.array(self.AR_absolute_weights)/np.sum(self.AR_absolute_weights)
        # Update the step count array (--> For LinearDecay and ExponentialDecay)
        self.global_step_count_arr = np.append(self.global_step_count_arr, 0)
        # Update the number of AR iterations
        self.current_AR_iterations = np.max(np.where(self.AR_weights > 0)) 
 
##----------------------------------------------------------------------------.   
def plot_AR_scheduler(AR_scheduler, n_updates=4, update_every=15, plot_absolute_AR_weights=True, plot_normalized_AR_weights=True):
    dict_AR_weights = {}
    dict_AR_absolute_weights = {}
    count = 0
    for i in range(n_updates):
        for j in range(update_every):
            dict_AR_weights[count] = AR_scheduler.AR_weights.copy() # Very important to copy!
            dict_AR_absolute_weights[count] = AR_scheduler.AR_absolute_weights.copy()  # Very important to copy!
            AR_scheduler.step()         
            count = count + 1
        AR_scheduler.update()
    
    method = AR_scheduler.method
    AR_iterations = AR_scheduler.AR_iterations
    iteration_arr = np.arange(len(dict_AR_absolute_weights))
    
    ### Reformat AR weights information (per leadtime)
    dict_AR_weights_leadtimes = {}
    for ld in range(AR_iterations):
        dict_AR_weights_leadtimes[ld] = {}
    for ld in range(AR_iterations):
        dict_AR_weights_leadtimes[ld]["iter"] = iteration_arr     
        dict_AR_weights_leadtimes[ld]["AR_weights"] = np.array([])  
        dict_AR_weights_leadtimes[ld]["AR_absolute_weights"] = np.array([])
        
    for arr in dict_AR_absolute_weights.values():
        tmp_max_AR_iterations = arr.shape[0] - 1
        for ld in range(AR_iterations):
            if ld <= tmp_max_AR_iterations:
                dict_AR_weights_leadtimes[ld]['AR_absolute_weights'] = np.append(dict_AR_weights_leadtimes[ld]['AR_absolute_weights'], arr[ld])                
            else: 
                dict_AR_weights_leadtimes[ld]['AR_absolute_weights'] = np.append(dict_AR_weights_leadtimes[ld]['AR_absolute_weights'], 0)                
    
    for arr in dict_AR_weights.values():
        tmp_max_AR_iterations = arr.shape[0] - 1
        for ld in range(AR_iterations):
            if ld <= tmp_max_AR_iterations:
                dict_AR_weights_leadtimes[ld]['AR_weights'] = np.append(dict_AR_weights_leadtimes[ld]['AR_weights'], arr[ld])                
            else: 
                dict_AR_weights_leadtimes[ld]['AR_weights'] = np.append(dict_AR_weights_leadtimes[ld]['AR_weights'], 0)                
     
    ### Visualize AR weights 
    if plot_absolute_AR_weights:    
        for ld in range(AR_iterations):
            plt.plot(dict_AR_weights_leadtimes[ld]['iter'],
                     dict_AR_weights_leadtimes[ld]['AR_absolute_weights'],
                     marker='.')
        plt.title("Absolute AR weights ({})".format(method))  
        plt.show()
    if plot_normalized_AR_weights:        
        for ld in range(AR_iterations):
            plt.plot(dict_AR_weights_leadtimes[ld]['iter'],
                     dict_AR_weights_leadtimes[ld]['AR_weights'], 
                     marker='.')
        plt.title("Normalized AR weights  ({})".format(method))      
        plt.show()

##----------------------------------------------------------------------------.