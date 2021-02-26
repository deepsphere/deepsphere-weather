#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 11:31:21 2021

@author: ghiggi
"""
import numpy as np 
from cycler import cycler
import matplotlib.pyplot as plt

##----------------------------------------------------------------------------.
### Check AR weights 
def check_AR_weights(AR_weights):
    """Check AR weights validity."""
    if isinstance(AR_weights, (int, float)):
        AR_weights = [AR_weights]
    if isinstance(AR_weights, list):
        AR_weights = np.array(AR_weights)
    if not isinstance(AR_weights, np.ndarray):
        raise TypeError("Specify AR weights with a list or a numpy array.")
    # Check that any intial_AR_weights is negative 
    if any(AR_weights < 0):
        raise ValueError("AR weights must not contain negative weights.")
    # Check that the last AR weight is not zero !
    if AR_weights[-1] == 0:
        raise ValueError("The last weight of AR_weights must not be 0.")
    return AR_weights

#----------------------------------------------------------------------------.
# No AR weights update when .step()  
def _ConstantStep(self): 
    return self.AR_weights 

def _DiracDeltaStep(self):
    return self.AR_weights  

##----------------------------------------------------------------------------.
## Discrete weight update functions 
def _StepwiseDecayStep(self):                 
    weights = self.AR_absolute_weights[:-1]
    weights = weights - self.factor 
    weights[weights < 0] = 0
    self.AR_absolute_weights[:-1] = weights 
 
def _StepwiseGrowthStep(self):                 
    weight = self.AR_absolute_weights[-1]
    weight = weight + self.factor
    if weight > 1:
        weight = 1 
    self.AR_absolute_weights[-1] = weight

def _StepwiseStep(self):
    if self.temporary_step_count >= self.step_interval: 
        _StepwiseDecayStep(self)
        if self.smooth_growth:
            _StepwiseGrowthStep(self)
        # Reset temporary_step_count 
        self.temporary_step_count = 0  
                      
def _HalfDecayStep(self):
    weights = self.AR_absolute_weights[:-1]
    weights = weights/2  
    self.AR_absolute_weights[:-1] = weights 

def _HalfGrowthStep(self):
    weight = self.AR_absolute_weights[-1]
    if weight == 0: 
        weight = self.factor 
    weight = weight*2
    if weight > 1:
        weight = 1 
    self.AR_absolute_weights[-1] = weight 
        
def _HalfStep(self):
    if self.temporary_step_count >= self.step_interval: 
        _HalfDecayStep(self)
        if self.smooth_growth:
            _HalfGrowthStep(self)
        # Reset temporary_step_count 
        self.temporary_step_count = 0  
        
##----------------------------------------------------------------------------.
### Continous weight update functions
def _LinearDecayStep(self): 
    initial_weights = self.AR_absolute_initial_weights[:-1]
    weights = initial_weights - self.factor*self.global_step_count_arr[:-1]
    weights[weights < 0] = 0
    self.AR_absolute_weights[:-1] = weights 

def _LinearGrowthStep(self): 
    initial_weight = self.AR_absolute_initial_weights[-1]
    weight = initial_weight + self.factor*self.global_step_count_arr[-1]
    if weight > 1:
        weight = 1
    self.AR_absolute_weights[-1] = weight 

def _LinearStep(self):
    _LinearDecayStep(self)
    if self.smooth_growth:
        _LinearGrowthStep(self)
    
def _ExponentialDecayStep(self):
    initial_weights = self.AR_absolute_initial_weights[:-1]
    weights = initial_weights * np.exp(-self.factor*self.global_step_count_arr[:-1])
    self.AR_absolute_weights[:-1] = weights 

def _ExponentialGrowthStep(self):
    weight = self.factor * np.exp(self.factor*self.global_step_count_arr[-1])
    if weight > 1:
        weight = 1
    self.AR_absolute_weights[-1] = weight 

def _ExponentialStep(self):
    _ExponentialDecayStep(self)
    if self.smooth_growth:
        _ExponentialGrowthStep(self)
    
#-----------------------------------------------------------------------------.    
class AR_Scheduler():
    """Autoregressive (AR) weights scheduler."""
    
    def __init__(self, 
                 method = "LinearStep", 
                 factor = 0.001,
                 step_interval = None, 
                 smooth_growth = True, 
                 fixed_AR_weights = None,
                 initial_AR_absolute_weights = None,
                 initial_AR_weights = None):
        """Autoregressive (AR) weights scheduler.

        Parameters
        ----------
        smooth_growth : bool, optional
            Wheter to set the new AR weight to 0 and growth it smoothly to avoid
            training destabilization.
            Do not apply to 'Constant' and 'DiracDelta' methods.
            The default is True.
        method : str, optional
            Available methods: 'Constant','DiracDelta','StepwiseDecay','HalfDecay','LinearDecay','ExponentialDecay'
            The default method is "DiracDelta".
            
            Methods explanation:
                
            Constant: Add an AR weight (with absolute value 1) when .update() is called.
            
            DiracDelta: Add an AR weight when .update() is called and 
                        reset to 0 the others AR weights. 
                         
            StepwiseStep: When a new AR weight is added with .update(), it start to substract  
                          'factor' from the others AR absolute weights every 'step_interval' .step() calls.
                          If smooth_growth=True, the new AR weight growth by step from 0 every 'step_interval' .step() calls.) 
            
            HalfStep: When a new AR weight is added with .update(), it start to half   
                      the others AR absolute weights every 'step_interval' .step() calls.
                      If smooth_growth=True, the new AR weight growth by doubling from factor every 'step_interval' .step() calls.
            
            LinearStep : When a new AR weight is added with .update(), it start to  
                         decrease linearly (with slope '-factor') the others 
                         AR absolute weights every .step() call.    
                         If smooth_growth=True, the new AR weight growth linearly 
                         starting from 0.  
                          
            ExponentialStep: When a new AR weight is added with .update(), it start to  
                             decrease exponentially (with decay rate '-factor') 
                             the others AR absolute weights every .step() call. 
                             If smooth_growth=True, the new AR weight growth exponentially 
                             starting from 'factor'.
        factor : float, optional
            Argument required by the following methods: 'StepwiseStep','HalfStep','LinearStep','ExponentialStep'.
            Regulate the decay and growth of AR absolute weights when .step() is called.
            For HalfStep and ExponentialStep, is also used as first value for the new AR_weight when smooth_growth=True.
        step_interval : int, optional
            Argument required by the following methods: 'StepwiseStep','HalfStep'.
            Specify the frequency with which the AR weights are updated with methods 'StepwiseStep' and 'HalfStep'.
            Step_interval = 1 cause weight update at every .step() call. 
        fixed_AR_weights : list, optional
            List of AR iterations for which the value AR weights must not be 
            modified by the step functions.
            The default is None. No AR weights is fixed.
        initial_AR_abolute_weights : list, optional
            Specify the initial absolute AR weights. 
            They will be rescaled to have 1 has largest value.
            If specified, initial_AR_weights must not be specified. 
            The default is AR_weights = [1].
        initial_AR_weights : list, optional
            Specify the initial normalized AR weights. (must sum up to 1).
            If specified, initial_AR_abolute_weights must not be specified. 
            The default is AR_weights = [1].
            
        """
        # 'StepwiseDecay' and 'HalfDecay' factor is applied to the AR_absolute weights (not the normalized AR_weights)
        # 'LinearDecay','ExponentialDecay' is applied from the initial AR_absolute_weights
        # TODO: 
        # - Implement a min_AR_weight_option? (instead of decaying to 0) 
        # - Increasing-Decreasing Decay  ... "
        # Check smooth_growth
        ##--------------------------------------------------------------------.
        if not isinstance(smooth_growth, bool):
            raise TypeError("'smooth_growth' must be either True or False.")
        ##--------------------------------------------------------------------.
        # Check valid method 
        valid_method = ['Constant','DiracDelta','StepwiseStep','HalfStep','LinearStep','ExponentialStep']
        if method not in valid_method:
            raise ValueError("Provide a valid 'method'.")   
        ##--------------------------------------------------------------------.    
        # Check fixed_AR_weights
        if not isinstance(fixed_AR_weights, (type(None), np.ndarray, list)):
            raise TypeError("'fixed_AR_weights' must be specified as list.")
        if isinstance(fixed_AR_weights, list):
            fixed_AR_weights = np.array(fixed_AR_weights)
        if fixed_AR_weights is not None: 
            if len(fixed_AR_weights) == 0:
                fixed_AR_weights = None
        ##---------------------------------------------------------------------.
        # Check initial_AR_weights and initial_AR_absolute_weights are not both specified.
        if initial_AR_weights is not None and initial_AR_absolute_weights is not None:
            raise ValueError("Specify either 'initial_AR_weights' or 'initial_AR_absolute_weights'.")
        
        # Set default AR_weights if not specified
        if initial_AR_weights is None and initial_AR_absolute_weights is None:
            initial_AR_weights = [1]
       
        # Check initial_AR_weights
        if initial_AR_weights is not None: 
            # Check AR weights validity 
            initial_AR_weights = check_AR_weights(initial_AR_weights)
            # Check AR_weights sum up to 1 
            if np.sum(initial_AR_weights) != 1:
                raise ValueError("'initial_AR_weights' must sum up to 1.")
            # Compute AR absolute weights
            # - Force the largest values to be 1
            initial_AR_absolute_weights = initial_AR_weights/initial_AR_weights.max()
       
        # Check initial_AR_absolute_weights
        elif initial_AR_absolute_weights is not None:    
            # Check AR weights validity 
            initial_AR_absolute_weights = check_AR_weights(initial_AR_absolute_weights)
            # - Force the maximum values to be 1
            initial_AR_absolute_weights = initial_AR_absolute_weights/initial_AR_absolute_weights.max()
            # Compute the normalized AR weights 
            initial_AR_weights = initial_AR_absolute_weights/initial_AR_absolute_weights.sum()
        else: 
            raise NotImplementedError("This option has been not considered.")
            
        ##--------------------------------------------------------------------.
        # Check that factor and step_interval are not negative 
        if factor is not None: 
            if factor < 0:
                raise ValueError("Provide a factor between 0 and 1.")
        if step_interval is not None: 
            if step_interval <= 0: 
                raise ValueError("'step_interval' must be an integer value equal or larger than 1.")    
        ##---------------------------------------------------------------------.
        # Check required method arguments are specified 
        if method in ['StepwiseStep','HalfStep']:
            if step_interval is None: 
                raise ValueError("'{}' method requires specification of the 'step_interval' argument".format(method))
        if method in ['HalfStep','StepwiseStep','LinearStep','ExponentialStep']:
            if factor is None:
                raise ValueError("'{}' method requires specification of the 'factor' argument".format(method))
        if method in ['Constant', 'DiracDelta']:
            smooth_growth = False
        ##---------------------------------------------------------------------. 
        # Count the number of AR iteration (at start)
        current_AR_iterations = len(initial_AR_weights) - 1
        self.current_AR_iterations = current_AR_iterations
        
        # Set absolute AR weights  
        self.AR_absolute_weights = initial_AR_absolute_weights
        # Set AR_weights (normalized AR weights)
        self.AR_weights = initial_AR_weights
        # Set initial AR absolute weights (for fixed weights) and 'LinearDecay' and 'ExponentialDecay'
        self.AR_absolute_initial_weights = self.AR_absolute_weights.copy()  
        
        ##--------------------------------------------------------------------.
        # Add method arguments
        self.method = method
        self.step_interval = step_interval
        self.factor = factor
        self.smooth_growth = smooth_growth
        self.fixed_AR_weights = fixed_AR_weights
        ##--------------------------------------------------------------------.
        # Initialize temporary step counter 
        # - For 'StepwiseDecay' and 'HalfDecay' method --> step_interval
        self.temporary_step_count = 0 
        
        ##--------------------------------------------------------------------.
        # - Initialize global step counter 
        # - For 'LinearDecay' and 'ExponentialDecay'
        self.global_step_count_arr = np.zeros(current_AR_iterations+1)
            
        ##--------------------------------------------------------------------.
        ### Define the update_weights function 
        fun_dict = {'Constant': _ConstantStep,
                    'DiracDelta': _DiracDeltaStep,
                    'StepwiseStep': _StepwiseStep,
                    'HalfStep': _HalfStep,
                    'LinearStep': _LinearStep,
                    'ExponentialStep': _ExponentialStep,
                    }          
        self.update_weights = fun_dict[method]
        
        ##--------------------------------------------------------------------.
   
    def step(self):
        """Update AR weights."""
        # Update step count 
        self.temporary_step_count = self.temporary_step_count + 1    # for 'StepwiseDecay' and 'HalfDecay'
        self.global_step_count_arr = self.global_step_count_arr + 1  # for 'LinearDecay' and 'ExponentialDecay'
        ##---------------------------------------------------------------------.
        if self.current_AR_iterations > 0:
            # - Update weights 
            self.update_weights(self)
            # - Refix the value of fixed AR weights
            if self.fixed_AR_weights is not None:
                tmp_fixed_AR_weights = self.fixed_AR_weights[self.fixed_AR_weights < self.current_AR_iterations]
                self.AR_absolute_weights[tmp_fixed_AR_weights] = self.AR_absolute_initial_weights[tmp_fixed_AR_weights]
            ##---------------------------------------------------------------------.
            # Retrieve normalized AR weights (summing up to 1)
            self.AR_weights = np.array(self.AR_absolute_weights)/np.sum(self.AR_absolute_weights)
    
    def update(self):
        """Add an AR_absolute_weight with value 1."""
        # Update the number of AR iterations
        self.current_AR_iterations = self.current_AR_iterations + 1
        # Add a new AR weight 
        if not self.smooth_growth: # ... with (absolute) value 1
            self.AR_absolute_weights = np.append(self.AR_absolute_weights, 1)
            self.AR_absolute_initial_weights = np.append(self.AR_absolute_initial_weights, 1)
        else: # start at 0 (or factor for ExponentialStep, HalfStep)
            # Update current last weight value (for ExponentialStep and LInearStep)
            self.AR_absolute_initial_weights[-1] = self.AR_absolute_weights[-1] 
            # Add new weight 
            self.AR_absolute_initial_weights = np.append(self.AR_absolute_initial_weights, 0)  
            self.AR_absolute_weights = np.append(self.AR_absolute_weights, 0)
         
        ##---------------------------------------------------------------------.
        # If DiracDelta weight update method is choosen, set to 0 the other weights 
        if self.method == "DiracDelta":
            self.AR_absolute_weights[:-1] = 0 
        ##---------------------------------------------------------------------.    
        # Update normalization of AR weights
        self.AR_weights = np.array(self.AR_absolute_weights)/np.sum(self.AR_absolute_weights)
        ##---------------------------------------------------------------------.
        # Update the step count array (--> For LinearDecay and ExponentialDecay)
        self.global_step_count_arr[-1] = 0 # Reset the last (because will start to decay)
        self.global_step_count_arr = np.append(self.global_step_count_arr, 0)
      
#----------------------------------------------------------------------------.   
def plot_AR_scheduler(ar_scheduler, 
                      n_updates=4, 
                      update_every=15, 
                      plot_absolute_AR_weights=True,
                      plot_normalized_AR_weights=True):
    
    n_initial_AR_weights = len(ar_scheduler.AR_weights)
    n_final_AR_weights = n_initial_AR_weights + n_updates
    ### Initialize dictionary
    AR_weights_per_AR_iteration = {}
    for i in range(n_final_AR_weights + 1):
        AR_weights_per_AR_iteration[i] = {}
        AR_weights_per_AR_iteration[i]['iteration'] = []
        AR_weights_per_AR_iteration[i]['AR_absolute_weights'] = []
        AR_weights_per_AR_iteration[i]['AR_weights'] = []
    
    # Simulate AR weights step() and update()
    iteration = 0
    for u in range(n_updates+1):
        for i in range(update_every+1):
            current_AR_iterations = len(ar_scheduler.AR_weights) - 1
            for AR_iteration in range(current_AR_iterations+1):
                AR_weights_per_AR_iteration[AR_iteration]['iteration'].append(iteration)   
                AR_weights_per_AR_iteration[AR_iteration]['AR_absolute_weights'].append(ar_scheduler.AR_absolute_weights[AR_iteration])
                AR_weights_per_AR_iteration[AR_iteration]['AR_weights'].append(ar_scheduler.AR_weights[AR_iteration])
            ar_scheduler.step()
            iteration = iteration + 1 
        ar_scheduler.update()
    ##------------------------------------------------------------------------.
    ### Visualize AR weights 
    method = ar_scheduler.method
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    custom_cycler = cycler(linestyle=['-', '--', ':', '-.','-', '--', ':', '-.','-', '--'],
                           color=colors)
    if plot_absolute_AR_weights:    
        fig, ax = plt.subplots()
        ax.set_prop_cycle(custom_cycler)
        for AR_iteration in range(n_final_AR_weights+1):
            plt.plot(AR_weights_per_AR_iteration[AR_iteration]['iteration'],
                     AR_weights_per_AR_iteration[AR_iteration]['AR_absolute_weights'],
                     antialiased = True)
        ax.set_xlabel("Iteration")
        plt.title("Absolute AR weights ({})".format(method)) 
        ax.legend(labels=list(range(n_final_AR_weights+1)), loc='upper right') 
        plt.show()
    if plot_normalized_AR_weights:   
        fig, ax = plt.subplots()
        ax.set_prop_cycle(custom_cycler)
        for AR_iteration in range(n_final_AR_weights+1):
            plt.plot(AR_weights_per_AR_iteration[AR_iteration]['iteration'],
                     AR_weights_per_AR_iteration[AR_iteration]['AR_weights'], 
                     antialiased = True)
        ax.set_xlabel("Iteration")
        plt.title("Normalized AR weights  ({})".format(method))  
        ax.legend(labels=list(range(n_final_AR_weights+1)), loc='upper right') 
        plt.show()

##----------------------------------------------------------------------------.
