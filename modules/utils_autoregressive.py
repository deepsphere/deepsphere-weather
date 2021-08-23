#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 17:00:54 2021

@author: ghiggi
"""
import numpy as np
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------.
######################
### Miscellaneous ####
######################
def is_sorted_array(arr, increasing=True):
    """Check the array is sorted."""
    # If only 1 
    if len(arr) == 0:
        return True
    # If multiple values
    if increasing:
        return np.all(np.diff(arr) >= 0)
    else:
        return np.all(np.diff(arr) <= 0)
    
def is_number(x):
    """Check is a number."""
    if isinstance(x, (int, float)):
        return True
    else: 
        return False
   
def is_natural_number(x): 
    """Check is a natural number."""
    if not is_number(x):
        return False
    else: 
        if isinstance(x, int):
            return True
        elif isinstance(x, float):
            return x.is_integer()
        else: 
            raise ValueError("Error. Not covered all number types")

#-----------------------------------------------------------------------------.
############################### 
### First and last indices ####
############################### 
def get_first_valid_idx(input_k):
    """Provide the first available index for training, once accounted the past timesteps required."""
    past_idxs = input_k[input_k < 0]
    if past_idxs.size == 0: # empty 
        past_idxs = 0  
    else: 
        past_idxs = abs(min(past_idxs)) 
    return past_idxs

def get_last_valid_idx(output_k, forecast_cycle, ar_iterations):
    """Provide the last available index for training, once accounted for all forecasted timesteps."""
    future_idxs = output_k[output_k >= 0]
    if future_idxs.size == 0: # empty 
        future_idxs = 0  
    else: 
        future_idxs = abs(max(future_idxs)) 
    return ar_iterations*forecast_cycle + future_idxs

#-----------------------------------------------------------------------------.
##############################################
### Indexing dictionaries for AR training ####
############################################## 
# - X and Y dictionary are relative to idx_start=0 
def get_idx_lag(idx_start, ar_iteration, forecast_cycle, input_k):
    """Provide the indices  of past predictors."""
    return idx_start + (forecast_cycle*ar_iteration) + input_k

def get_idx_forecast(idx_start, ar_iteration, forecast_cycle, output_k):
    """Provide the indices of the forecasts."""
    return idx_start + (forecast_cycle*ar_iteration) + output_k

def get_dict_Y(ar_iterations, forecast_cycle, output_k):
    """Provide information to load the labels required by an AR model."""
    dict_Y = {}
    for i in range(ar_iterations+1):
        dict_Y[i] = get_idx_forecast(idx_start=0, ar_iteration=i, forecast_cycle=forecast_cycle, output_k=output_k)
    return dict_Y 
       
def get_dict_X_dynamic(ar_iterations, forecast_cycle, input_k):
    """Provide information to load the dynamic data required by an AR model."""
    dict_X_past = {}
    for i in range(ar_iterations+1):
        idxs = get_idx_lag(idx_start=0, ar_iteration=i, forecast_cycle=forecast_cycle, input_k=input_k)
        idxs_past_data = idxs[idxs < 0]
        # Indexes of past data  
        if (idxs_past_data.size == 0):
            dict_X_past[i] = None 
        else: 
            dict_X_past[i] = idxs_past_data 
    return dict_X_past

def get_dict_X_bc(ar_iterations, forecast_cycle, input_k):
    """Provide information to load the boundary conditions data required by an AR model."""
    dict_X_bc = {}
    for i in range(ar_iterations+1):
        dict_X_bc[i] = get_idx_lag(idx_start=0, ar_iteration=i, forecast_cycle=forecast_cycle, input_k=input_k)
    return dict_X_bc 
         
def get_dict_stack_info(ar_iterations, forecast_cycle, input_k, output_k, stack_most_recent_prediction = True):
    """Provide the information required to stack the iterative predictions of an AR model."""
    input_k = check_input_k(input_k, ar_iterations)
    output_k = check_output_k(output_k)
    # Compute index of Y labels 
    dict_Y = {}
    for i in range(ar_iterations+1):
        dict_Y[i] = get_idx_forecast(idx_start=0, ar_iteration=i, forecast_cycle=forecast_cycle, output_k=output_k)
    # Compute index of future X that need to be stacked from previous predicted Y
    dict_X_future = {}
    for i in range(ar_iterations+1):
        idxs = get_idx_lag(idx_start=0, ar_iteration=i, forecast_cycle=forecast_cycle, input_k=input_k)
        idxs_future_data = idxs[idxs >= 0]  
        if (idxs_future_data.size == 0):
            dict_X_future[i] = None 
        else: 
            dict_X_future[i] = idxs_future_data 
    ##------------------------------------------------------------------------.
    # - Define an index to choose which of the previous available prediction to take  
    if stack_most_recent_prediction: 
        idx_to_select = -1 # take the last  available prediction performed 
    else: 
        idx_to_select = 0  # take the first available prediction performed
    ##------------------------------------------------------------------------.    
    # - Retrieve leadtime and time index of Y that need to be stacked to X
    dict_Y_to_stack = {}
    try:
        for i, idx_X_future in dict_X_future.items():
            if idx_X_future is None:
                dict_Y_to_stack[i] = None 
            else: 
                l_tuple_idx = []
                # Search for each X_future_index, the leadtime and tensor index from which to take the data 
                for idx in idx_X_future: 
                    # Search only in Y data already predicted -> range(i) and not range(i+1)  
                    # - Return a list of tuples [(L0, idx) of possible solutions 
                    l_solutions = [(j, np.argwhere(dict_Y[j] == idx).tolist()[0][0]) for j in range(i) if idx in dict_Y[j] ]    
                    # - Select first or last based on 'stack_most_recent_prediction' option
                    l_tuple_idx.append(l_solutions[idx_to_select])
                # Add to the dictionary 
                dict_Y_to_stack[i] = l_tuple_idx 
    except IndexError:
        raise ValueError("Review the AR settings. AR training is not possible with the current configuration!")    
    ##------------------------------------------------------------------------.
    # - Construct dictionary specifying when to remove predicted Y (from GPU)
    dict_Y_to_remove = {}
    idx_arr_removed = np.array([]) # Initialize 
    # Start looping from the last forecast iteration (to the first)
    for i in range(ar_iterations+1)[::-1]: 
        l_tuple = dict_Y_to_stack[i]
        # Skip leadtime when no Y data to stack 
        if l_tuple is None: 
            dict_Y_to_remove[i] = None
        else: 
            # Retrieve required leadtime at given forecast iteration
            leadtime_arr = np.array([tuple2[0] for tuple2 in l_tuple])
            # Select the one can be deleted 
            leadtime_arr = leadtime_arr[np.isin(leadtime_arr, idx_arr_removed, invert=True)]
            if leadtime_arr.size == 0:
                dict_Y_to_remove[i] = None
            else: 
                dict_Y_to_remove[i] = leadtime_arr.tolist()
                # Update idx_arr_removed
                idx_arr_removed = np.append(idx_arr_removed, leadtime_arr)      
    return (dict_Y_to_stack, dict_Y_to_remove)
             
#-----------------------------------------------------------------------------.
###############
### Checks ####
###############
def check_input_k(input_k, ar_iterations):  
    """Check validity of 'input_k' argument."""
    if isinstance(input_k, list):  
        input_k = np.array(input_k)  
    # Check indexing is sorted increasingly
    if not is_sorted_array(input_k, increasing=True): 
        raise ValueError("Provide input_k sorted increasingly")
    # Checks for forecasting mode 
    if ar_iterations > 0:
        if np.any(input_k == 0): 
            raise ValueError("input_k contains values equal to 0. Past timesteps must be specified with negative values")    
        if np.all(input_k > 0): 
            raise ValueError("input_k requires negative values in 'forecasting' mode")  
    return input_k    
    
def check_output_k(output_k):   
    """Check validity of 'output_k' argument."""
    if isinstance(output_k, list):  
        output_k = np.array(output_k)
    # Check indexing is sorted increasingly
    if not is_sorted_array(output_k, increasing=True): 
        raise ValueError("Provide output_k sorted increasingly")
    # Check output_k contains at least a 0     
    if not np.any(output_k == 0): 
        raise ValueError("output_k must start with a 0 value. 0 indicates the 'current' timestep to predict.")    
    return output_k    

def check_ar_iterations(ar_iterations):
    """Check validity of 'ar_iterations' argument."""
    if not is_number(ar_iterations):
        raise TypeError("'ar_iterations' must be a single integer number")
    if not is_natural_number(ar_iterations):
        raise ValueError("'ar_iterations' must be a positive integer value")
    if ar_iterations < 0:
        raise ValueError("'ar_iterations' must be a positive integer value")       
    if ar_iterations >= 1:
        print(' - Autoregressive training with %d AR iterations --> Specified.'% ar_iterations)
    ar_iterations = int(ar_iterations)
    return ar_iterations 

def check_forecast_cycle(forecast_cycle, ar_iterations):
    """Check validity of 'forecast_cycle' argument."""
    if not is_number(ar_iterations):
        raise TypeError("'ar_iterations' must be a single integer number")
    if not is_natural_number(forecast_cycle):
        raise ValueError("'forecast_cycle' must be a positive integer value")
    if forecast_cycle < 1:
        raise ValueError("'forecast_cycle' must be equal or longer than 1")  
    if forecast_cycle >= 1:
        print(' - Forecast cycle of %d --> Specified.'% forecast_cycle)    
    forecast_cycle = int(forecast_cycle)
    return forecast_cycle   

def check_ar_settings(input_k, output_k, forecast_cycle, ar_iterations, stack_most_recent_prediction):
    """Check that AR settings arguments are valid."""
    input_k = check_input_k(input_k=input_k, ar_iterations=ar_iterations)   
    output_k = check_output_k(output_k=output_k)
    ar_iterations = check_ar_iterations(ar_iterations=ar_iterations)
    forecast_cycle = check_forecast_cycle(forecast_cycle=forecast_cycle, ar_iterations=ar_iterations) 
    ##------------------------------------------------------------------------.
    # Check autoregressive training is feasible
    if ar_iterations >= 1:
        dict_Y_to_stack, _ = get_dict_stack_info(ar_iterations=ar_iterations,
                                                 forecast_cycle=forecast_cycle, 
                                                 input_k=input_k, output_k=output_k, 
                                                 stack_most_recent_prediction = stack_most_recent_prediction)
    # if ar_iterations >= 1:
    #     idxs_lag_0 = get_idx_lag(idx_start=0, ar_iteration=0, forecast_cycle=forecast_cycle, input_k=input_k)
    #     idxs_forecasted_0 = get_idx_forecast(idx_start=0, ar_iteration=0, forecast_cycle=forecast_cycle, output_k=output_k)   
    #     idxs_lag_1 = get_idx_lag(idx_start=0, ar_iteration=1, forecast_cycle=forecast_cycle, input_k=input_k)
    #     # idxs_forecasted_1 = get_idx_forecast(idx_start=0, ar_iteration=1, forecast_cycle=forecast_cycle, output_k=output_k) 
    #     idxs_available = np.concatenate((idxs_lag_0, idxs_forecasted_0))
    #     if np.any([v not in idxs_available for v in idxs_lag_1]):
    #         raise ValueError("Review the autoregressive settings. Autoregressive training is not allowed with the current configuration!")
    ##------------------------------------------------------------------------.        

#-----------------------------------------------------------------------------.
##############################
### Plot AR configuration ####
##############################
def _arr_window_info(input_k, output_k, forecast_cycle, ar_iterations, past_margin=0, future_margin=0):  
    """Retrieve information of the data temporal window required for 1 training loop."""
    past_idxs = past_margin + get_first_valid_idx(input_k=input_k)    
    future_idxs = future_margin + get_last_valid_idx(output_k=output_k, forecast_cycle=forecast_cycle, ar_iterations=ar_iterations)
    idx_t_0 = past_idxs
    width = idx_t_0 + future_idxs + 1
    height = ar_iterations + 1    
    return idx_t_0, width, height 

def plot_ar_settings(input_k, output_k, forecast_cycle, 
                     ar_iterations, stack_most_recent_prediction=True,
                     past_margin=0, future_margin=0, hatch=True):
    """Plot the model AR configuration."""
    ##------------------------------------------------------------------------.
    # Create forecast temporal data window
    idx_start, width, height = _arr_window_info(input_k=input_k, 
                                                output_k=output_k, 
                                                forecast_cycle=forecast_cycle,
                                                ar_iterations=ar_iterations,
                                                past_margin=past_margin,
                                                future_margin=future_margin) 
    arr = np.zeros(shape = (height, width))
    ##------------------------------------------------------------------------.
    # Create hatching array (only for forecasting mode)
    if ((ar_iterations >= 1) and hatch):
        hatch_arr = np.zeros(shape = (height, width))
        dict_Y_to_stack, _ = get_dict_stack_info(ar_iterations=ar_iterations,
                                                 forecast_cycle=forecast_cycle, 
                                                 input_k=input_k, output_k=output_k, 
                                                 stack_most_recent_prediction = stack_most_recent_prediction)
        dict_Y = get_dict_Y(ar_iterations=ar_iterations, forecast_cycle=forecast_cycle, output_k=output_k)
        for i in range(height):  
            if dict_Y_to_stack[i] is not None: 
                for tpl in dict_Y_to_stack[i]:
                    hatch_arr[tpl[0], dict_Y[tpl[0]][tpl[1]] + idx_start] = 1
    ##------------------------------------------------------------------------.
    # Simulate data selection
    idxs_forecasted = None # Just to avoid error remark in if in loop below
    for i in range(height):  
        idxs_lag = get_idx_lag(idx_start=idx_start, ar_iteration=i, forecast_cycle=forecast_cycle, input_k=input_k)
        idxs_forecasted = get_idx_forecast(idx_start=idx_start, ar_iteration=i, forecast_cycle=forecast_cycle, output_k=output_k)
        arr[i, idxs_lag] = 1
        arr[i, idxs_forecasted] = 2
        
    ##------------------------------------------------------------------------.
    # Create figure 
    fig, ax = plt.subplots()
    ax.imshow(arr, aspect="auto")
    # - Add hatching (if forecasting mode) 
    if ((ar_iterations >= 1) and hatch):
        hatch_arr = np.ma.masked_less(hatch_arr, 1)
        ax.pcolor(np.arange(width+1)-.5, np.arange(height+1)-.5, hatch_arr, 
                  hatch='//', alpha=0.)
    # - Add grid 
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
        ax.set_xticks(np.arange(width+1)-.5, minor=True)
        ax.set_yticks(np.arange(height+1)-.5, minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", length=0, bottom=False, left=False)
    # - Plot starting time 
    plt.axvline(x=idx_start - 0.5, ymin=0, ymax=ar_iterations+1, c="red")
       
    # - Set axis ticks
    ax.set_xticks(np.arange(width))  
    ax.set_yticks(np.arange(height))
    # - Set axis labels 
    ax.set_xticklabels(np.arange(width) - idx_start)
    ax.set_yticklabels(np.arange(height))
    # - Set secondary x axis 
    ax2 = ax.twiny()
    ax2.axis(ax.axis())
    time = np.arange(width) - idx_start 
    time[time >= 0] = time[time >= 0]   # Current timestep: leadtime 0
    idx_valid = time >= 0
    # time[time >= 0] = time[time >= 0] + 1  # Current timestep: leadtime 1
    # idx_valid = time => 1
    ax2.set_xticks(np.arange(width)[idx_valid])  
    ax2.set_xticklabels(time[idx_valid])
    # -Set labels 
    ax.set_xlabel("K")
    ax.set_ylabel("Forecast Iterations")
    ax2.set_xlabel("Forecast Lead Time")
    ##------------------------------------------------------------------------.
