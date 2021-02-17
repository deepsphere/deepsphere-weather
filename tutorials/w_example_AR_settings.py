# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 20:39:58 2021

@author: ghiggi
"""
##----------------------------------------------------------------------------.
import numpy as np

from modules.utils_autoregressive import check_AR_settings
from modules.utils_autoregressive import plot_AR_settings
from modules.utils_autoregressive import get_first_valid_idx
from modules.utils_autoregressive import get_last_valid_idx
from modules.utils_autoregressive import get_dict_Y
from modules.utils_autoregressive import get_dict_X_dynamic
from modules.utils_autoregressive import get_dict_X_bc
from modules.utils_autoregressive import get_dict_stack_info

##----------------------------------------------------------------------------. 
### Example - Classical Autoregressive 
input_k = np.array([-9, -6,-3])
output_k = np.array([0])
AR_iterations = 6
forecast_cycle = 3  

check_AR_settings(input_k=input_k, output_k=output_k, forecast_cycle=forecast_cycle,
                  AR_iterations=AR_iterations, stack_most_recent_prediction=True)
plot_AR_settings(input_k=input_k, output_k=output_k, forecast_cycle=forecast_cycle, AR_iterations = AR_iterations)
 
print(get_first_valid_idx(input_k))
print(get_last_valid_idx(output_k=output_k,
                         forecast_cycle=forecast_cycle, 
                         AR_iterations=AR_iterations))

##----------------------------------------------------------------------------.
### Example - Fixed DELTA T   
delta_t = 6
n_input_lags = 3    
n_output_timesteps = 1
forecast_cycle = delta_t
AR_iterations = 4
input_k = np.cumsum(-1*np.repeat(delta_t, repeats=n_input_lags))[::-1]
output_k = np.cumsum(np.concatenate((np.array([0]), np.repeat(delta_t, repeats=n_output_timesteps-1))))

check_AR_settings(input_k=input_k, output_k=output_k, forecast_cycle=forecast_cycle, 
                  AR_iterations=AR_iterations, stack_most_recent_prediction=True)
plot_AR_settings(input_k=input_k, output_k=output_k, forecast_cycle=forecast_cycle, AR_iterations = AR_iterations)

print(get_first_valid_idx(input_k))
print(get_last_valid_idx(output_k=output_k,
                         forecast_cycle=forecast_cycle, 
                         AR_iterations=AR_iterations))

##----------------------------------------------------------------------------.
### Example - Multi-Temporal Output 
input_k = np.array([-7,-5,-3,-1])
output_k = np.array([0, 1, 3, 7])
AR_iterations = 6
forecast_cycle = 2 

check_AR_settings(input_k=input_k, output_k=output_k, forecast_cycle=forecast_cycle,
                  AR_iterations=AR_iterations, stack_most_recent_prediction=True)
plot_AR_settings(input_k=input_k, output_k=output_k, forecast_cycle=forecast_cycle, 
                 AR_iterations = AR_iterations, stack_most_recent_prediction = True)
plot_AR_settings(input_k=input_k, output_k=output_k, forecast_cycle=forecast_cycle, 
                 AR_iterations = AR_iterations, stack_most_recent_prediction = False)

##----------------------------------------------------------------------------.
### Example - Chess pattern 
input_k = np.array([-8,-6,-4,-2])
output_k = np.array([0, 2, 4, 7])
AR_iterations = 6
forecast_cycle = 1  

check_AR_settings(input_k=input_k, output_k=output_k, forecast_cycle=forecast_cycle,
                  AR_iterations=AR_iterations, stack_most_recent_prediction=True)
plot_AR_settings(input_k, output_k, forecast_cycle, AR_iterations, stack_most_recent_prediction=True)
plot_AR_settings(input_k, output_k, forecast_cycle, AR_iterations, stack_most_recent_prediction=False)

#-----------------------------------------------------------------------------.
### Example dictionary for data retrieval and stacking
AR_iterations=5
forecast_cycle=3
output_k = np.array([0,3])
input_k = np.array([-9, -6, -3])

plot_AR_settings(input_k=input_k,
                 output_k=output_k, 
                 forecast_cycle=forecast_cycle, 
                 AR_iterations = AR_iterations,
                 stack_most_recent_prediction = False)

plot_AR_settings(input_k=input_k,
                 output_k=output_k, 
                 forecast_cycle=forecast_cycle, 
                 AR_iterations = AR_iterations,
                 stack_most_recent_prediction = True)

dict_Y = get_dict_Y(AR_iterations=AR_iterations, 
                    forecast_cycle=forecast_cycle,
                    output_k=output_k)
dict_X_dynamic = get_dict_X_dynamic(AR_iterations=AR_iterations, 
                                    forecast_cycle=forecast_cycle, 
                                    input_k=input_k)
dict_X_bc = get_dict_X_bc(AR_iterations = AR_iterations,
                          forecast_cycle = forecast_cycle,
                          input_k = input_k)
dict_Y_to_stack, dict_Y_to_remove = get_dict_stack_info(AR_iterations = AR_iterations, 
                                                        forecast_cycle = forecast_cycle, 
                                                        input_k = input_k,
                                                        output_k = output_k,
                                                        stack_most_recent_prediction = False)

#----------------------------------------------------------------------------.
### Check validity of something do not work
input_k = np.array([-7,-5,-3,-1])
output_k = np.array([0, 1, 3, 7])
AR_iterations = 2
forecast_cycle = 5 

check_AR_settings(input_k=input_k, output_k=output_k, forecast_cycle=forecast_cycle,
                  AR_iterations=AR_iterations, stack_most_recent_prediction=True)
plot_AR_settings(input_k, output_k, forecast_cycle, AR_iterations, hatch=True)

# - Use Hatch=False allow to still visualize the AR setting
plot_AR_settings(input_k, output_k, forecast_cycle, AR_iterations, hatch=False)

#-----------------------------------------------------------------------------.
