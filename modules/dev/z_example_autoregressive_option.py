# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 20:39:58 2021

@author: ghiggi
"""

#-----------------------------------------------------------------------------.
import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------.
###################
## Get indices ####
###################
def get_idx_lag(idx, forecast_iteration, forecast_cycle, input_k):
    return idx + (forecast_cycle*forecast_iteration) + input_k

def get_idx_forecast(idx, forecast_iteration, forecast_cycle, output_k):
    return idx + (forecast_cycle*forecast_iteration) + output_k

def get_idx_to_stack(input_k, output_k, forecast_cycle): 
    idxs_lag_0 = get_idx_lag(idx=0, forecast_iteration=0, forecast_cycle=forecast_cycle, input_k=input_k)
    idxs_forecasted_0 = get_idx_forecast(idx=0, forecast_iteration=0, forecast_cycle=forecast_cycle, output_k=output_k)   
    idxs_lag_1 = get_idx_lag(idx=0, forecast_iteration=1, forecast_cycle=forecast_cycle, input_k=input_k)
    idxs_forecasted_1 = get_idx_forecast(idx=0, forecast_iteration=1, forecast_cycle=forecast_cycle, output_k=output_k)   
    # print(idxs_lag_0)
    # print(idxs_forecasted_0)
    # print(idxs_lag_1)
    # print(idxs_forecasted_1)
    idx_to_stack = np.array([i for i, v in enumerate(idxs_forecasted_0) if v in idxs_lag_1])
    return idx_to_stack   
    
#-----------------------------------------------------------------------------.
#####################
## Miscellaneous ####
#####################
def is_sorted_array(arr, increasing=True):
    # If only 1 
    if len(arr) == 0:
        return True
    # If multiple values
    if increasing:
        return np.all(np.diff(arr) >= 0)
    else:
        return np.all(np.diff(arr) <= 0)
    
def is_number(x):
    if isinstance(x, (int, float)):
        return True
    else: 
        return False
   
def is_natural_number(x): 
    if not is_number(x):
        return False
    else: 
        if isinstance(x, int):
            return True
        elif isinstance(x, float):
            return x.is_integer()
        else: 
            raise ValueError("Error. Not covered all number types")

def _arr_window_info(input_k, output_k, forecast_cycle, forecast_iterations, past_margin=0, future_margin=0):  
    """ Retrieve information of the data temporal window required for 1 training sample."""
    # Retrieve data window information 
    past_idxs = input_k[input_k < 0]
    future_idxs = output_k[output_k >= 0]
    if past_idxs.size == 0: # empty 
        past_idxs = 0  
    else: 
        past_idxs = abs(min(past_idxs)) 
    past_idxs = past_idxs + past_margin
    if future_idxs.size == 0: # empty 
        future_idxs = 0  
    else: 
        future_idxs = abs(min(future_idxs)) 
    future_idxs = future_idxs + future_margin
    idx_start = past_idxs
    width = past_idxs + future_idxs + forecast_iterations*forecast_cycle + max(output_k) # + 1
    height = forecast_iterations    
    return idx_start, width, height 

#-----------------------------------------------------------------------------.
#############
# Checks ####
#############
def check_input_k(input_k, forecast_iterations):    
    if isinstance(input_k, list): # TODO switch to numpy.array
        input_k = np.array(input_k)  
    # Check indexing is sorted increasingly
    if not is_sorted_array(input_k, increasing=True): 
        raise ValueError("Provide input_k sorted increasingly")
    # Checks for forecasting mode 
    if forecast_iterations > 0:
        if np.any(input_k == 0): 
            raise ValueError("input_k contains values equal to 0. Past timesteps must be specified with negative values")    
        if np.all(input_k > 0): 
            raise ValueError("input_k requires negative values in 'forecasting' mode")  
    return input_k    
    
def check_output_k(output_k):   
    if isinstance(output_k, list): # TODO switch to numpy.array
        output_k = np.array(output_k)
    # Check indexing is sorted increasingly
    if not is_sorted_array(output_k, increasing=True): 
        raise ValueError("Provide output_k sorted increasingly")
    # Check output_k contains at least a 0     
    if not np.any(output_k == 0): 
        raise ValueError("output_k must start with a 0 value. 0 indicates the 'current' timestep to predict.")    
    return output_k    

def check_forecast_iterations(forecast_iterations):
    if not is_number(forecast_iterations):
        raise TypeError("'forecast_iterations' must be a single integer number")
    if not is_natural_number(forecast_iterations):
        raise ValueError("'forecast_iterations' must be a positive integer value")
    if forecast_iterations < 0:
        raise ValueError("'forecast_iterations' must be a positive integer value")       
    if (forecast_iterations >= 1):
        print(' - Autoregressive training with %d iterations --> Specified.'% forecast_iterations)
    return None 

def check_forecast_cycle(forecast_cycle, forecast_iterations):
    if not is_number(forecast_iterations):
        raise TypeError("'forecast_iterations' must be a single integer number")
    if not is_natural_number(forecast_cycle):
        raise ValueError("'forecast_cycle' must be a positive integer value")
    if forecast_cycle < 1:
        raise ValueError("'forecast_cycle' must be equal or longer than 1")  
    if forecast_iterations >= 1:
        print(' - Forecast cycle of %d --> Specified'% forecast_cycle)     
    return None   

def check_model_validity(input_k, output_k, forecast_cycle, forecast_iterations):
    input_k = check_input_k(input_k=input_k, forecast_iterations=forecast_iterations)   
    output_k = check_output_k(output_k=output_k)
    check_forecast_iterations(forecast_iterations=forecast_iterations)
    check_forecast_cycle(forecast_cycle=forecast_cycle, forecast_iterations=forecast_iterations) 
    ##---------------------------------------------------
    # Check feasibility for autoregressive training
    if forecast_iterations >= 1:
        idxs_lag_0 = get_idx_lag(idx=0, forecast_iteration=0, forecast_cycle=forecast_cycle, input_k=input_k)
        idxs_forecasted_0 = get_idx_forecast(idx=0, forecast_iteration=0, forecast_cycle=forecast_cycle, output_k=output_k)   
        idxs_lag_1 = get_idx_lag(idx=0, forecast_iteration=1, forecast_cycle=forecast_cycle, input_k=input_k)
        idxs_forecasted_1 = get_idx_forecast(idx=0, forecast_iteration=1, forecast_cycle=forecast_cycle, output_k=output_k) 
        idxs_available = np.concatenate((idxs_lag_0, idxs_forecasted_0))
        if np.any([v not in idxs_available for v in idxs_lag_1]):
            raise ValueError("Review the autoregressive settings. Autoregressive training is not allowed with the current configuration!")
    ##---------------------------------------------------        
    return (input_k, output_k)

#-----------------------------------------------------------------------------.
#####################
### Plot indexes ####
#####################  
def plot_indexing(input_k, output_k, forecast_cycle, forecast_iterations,
                  past_margin=2, future_margin=2):
    ##---------------------------------------------------
    # Create forecast temporal data window
    idx_start, width, height = _arr_window_info(input_k=input_k, 
                                                output_k=output_k, 
                                                forecast_cycle=forecast_cycle,
                                                forecast_iterations=forecast_iterations, 
                                                past_margin=past_margin,
                                                future_margin=future_margin) 
    arr = np.zeros(shape = (height, width))
    ##---------------------------------------------------
    # Create hatching array (only for forecasting mode)
    if (forecast_iterations >= 1):
        hatch_arr = np.zeros(shape = (height, width))
        idx_to_hatch = get_idx_to_stack(input_k=input_k, output_k=output_k, forecast_cycle=forecast_cycle)
    
    ##---------------------------------------------------
    # Simulate data selection
    idxs_forecasted = None # Just to avoid error remark in if in loop below
    for i in range(forecast_iterations):
        # Index hatching to stack predictions in autoregressive mode
        if (i >= 1):
            hatch_arr[i-1, idxs_forecasted[idx_to_hatch]] = 1       
        idxs_lag = get_idx_lag(idx=idx_start, forecast_iteration=i, forecast_cycle=forecast_cycle, input_k=input_k)
        idxs_forecasted = get_idx_forecast(idx=idx_start, forecast_iteration=i, forecast_cycle=forecast_cycle, output_k=output_k)
        arr[i, idxs_lag] = 1
        arr[i, idxs_forecasted] = 2
        
    ##---------------------------------------------------
    # Create figure 
    fig, ax = plt.subplots()
    ax.imshow(arr, aspect="auto")
    # - Add hatching (if forecasting mode) 
    if (forecast_iterations >= 1):
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
    plt.axvline(x=idx_start - 0.5, ymin=0, ymax=forecast_iterations, c="red")
       
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

##----------------------------------------------------------------------------.

# Example - Multi-Temporal Output 
input_k = np.array([-7,-5,-3,-1])
output_k = np.array([0, 1, 3, 7])
forecast_iterations = 6
forecast_cycle = 2 
check_model_validity(input_k=input_k, output_k=output_k, forecast_cycle=forecast_cycle, forecast_iterations=forecast_iterations)
plot_indexing(input_k=input_k, output_k=output_k, forecast_cycle=forecast_cycle, forecast_iterations = forecast_iterations)

# Example - Classical Autoregressive 
input_k = np.array([-9, -6,-3])
output_k = np.array([0])
forecast_iterations = 6
forecast_cycle = 3  

check_model_validity(input_k=input_k, output_k=output_k, forecast_cycle=forecast_cycle, forecast_iterations=forecast_iterations)
plot_indexing(input_k=input_k, output_k=output_k, forecast_cycle=forecast_cycle, forecast_iterations = forecast_iterations)
 
# Fixed delta t case  
delta_t = 3
n_input_lags = 3    
n_output_timesteps = 1
forecast_cycle = delta_t
forecast_iterations = 4
input_k = np.cumsum(-1*np.repeat(delta_t, repeats=n_input_lags))[::-1]
output_k = np.cumsum(np.concatenate((np.array([0]), np.repeat(delta_t, repeats=n_output_timesteps-1))))

check_model_validity(input_k=input_k, output_k=output_k, forecast_cycle=forecast_cycle, forecast_iterations=forecast_iterations)
plot_indexing(input_k=input_k, output_k=output_k, forecast_cycle=forecast_cycle, forecast_iterations = forecast_iterations)

# Check validity (of something do not work)
input_k = np.array([-7,-5,-3,-1])
output_k = np.array([0, 1, 3, 7])
forecast_iterations = 6
forecast_cycle = 5 

check_model_validity(input_k=input_k, output_k=output_k, forecast_cycle=forecast_cycle, forecast_iterations=forecast_iterations)
plot_indexing(input_k, output_k, forecast_cycle, forecast_iterations)

#-----------------------------------------------------------------------------.
# Check validity (of something that should be feasible)
# - TODO: DEBUG ! --> Need to keep in memory over multiple iterations (set a limit...)

input_k = np.array([-7,-5,-3,-1])
output_k = np.array([0, 1, 3, 7])
forecast_iterations = 6
forecast_cycle = 1  

check_model_validity(input_k=input_k, output_k=output_k, forecast_cycle=forecast_cycle, forecast_iterations=forecast_iterations)
plot_indexing(input_k, output_k, forecast_cycle, forecast_iterations)

#-----------------------------------------------------------------------------.
# Autoregressive parameters: 
input_k
output_k
forecast_cycle # forecast cycle determines selection during autoregressive training
forecast_iterations 

#-----------------------------------------------------------------------------.
### TODO indexing:
# iterator (compute all indices...)
# --> Save data of past timesteps 

# dynamic_get_item
# boundary_conditions_get_item
# static_get_item 

# get_first_valid_idx()
# get_last_valid_idx()  
# get_stack_idx() as function of forecast_iteration 

# idx_x, idx_y, idx_x_stack = get_indices(iteration=i,..., forecast_iterations, ) 
# idx_x ...empty after i iterations
# concat(data[idx_x], predicted[idx_x_stack])

# check_get_item__ when array of idxs is also len=1 ! 

### Implemented 
# get_idx_forecast(idx=idx, forecast_iteration =i, forecast_cycle=forecast_cycle, output_k=output_k)
# get_idx_lag(idx=idx, forecast_iteration = i, forecast_cycle=forecast_cycle, input_k=input_k)
# get_stack_idx(input_k=input_k, output_k=output_k, forecast_cycle=forecast_cycle) 

##----------------------------------------------------------------------------.
### Old stuffs

# idx_step = self.delta_t * k
# len_total = len_sqce + len_output

# total_length = n_idx * len_sqce

# idx_start = n_idx * (len_sqce + k - 1)
# idx_end = n_idx * (len_sqce + k + 1)

# ['sample','time', 'node','features']

# def __getitem__(self, idx):
# """ Returns the input predictors (X) and a list of output labels (y) for autoregressive training
#     Idx : a torch.Tensor objects
    
#     The returned tensor shapes are: [n_vertex, len_sqce, n_features]
# """
# batch[0] --> (batch_size, num_nodes, n_features*len_sq)

            
# # Create indexes
# self.idxs = [[[[sample_idx + delta_t*k for k in range(len_sqce)], sample_idx + delta_t * len_sqce], 
#               [sample_idx + delta_t * len_sqce, sample_idx + delta_t * (len_sqce+1)]] 
#              for sample_idx in range(self.n_samples)]
        
# random.shuffle(training_ds.idxs)
# idxs = training_ds.idxs    

# idx_data = idx # --> idxs 

# n_idx = len(idx)      # n_batch_size 
# len_output = self.len_output # self.out_features

# len_sqce = self.len_sqce
# nodes = self.nodes    # n_nodes

# idx_step = self.delta_t * k
# total_length = n_idx * len_sqce
# len_total = len_sqce + len_output


# idx_start = n_idx * (len_sqce + k - 1)
# idx_end = n_idx * (len_sqce + k + 1)


# idx_full = np.concatenate(np.array([[idx_data + delta_t * k] for k in range(len_total)])).reshape(-1) # reshape(-1,1))

# dat = self.data.isel(time=idx_full).values

# X = (torch.tensor(dat[:total_length, :, :], dtype=torch.float).reshape(total_length, nodes, -1),)

# y = [torch.tensor(dat[idx_start:idx_end, :, :], dtype=torch.float).reshape(total_length, nodes, -1) for k in range(len_output)]
 
# https://github.com/ownzonefeng/weather_prediction/blob/ModulesDev/modules/full_pipeline_multiple_steps.py
# https://github.com/ownzonefeng/weather_prediction/blob/ModulesDev/modules/full_pipeline.py

