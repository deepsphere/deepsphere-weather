#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 13:50:17 2021

@author: ghiggi
"""
from tabulate import tabulate 

 ## Code for measuring execution time as function of n. AR iterations 

# import pdb
# pdb.set_trace()
     
#############################
### Autotune num_workers ####
#############################
num_workers_steps = 1
min_num_workers = 0
max_num_workers = 6
n_repetitions = 10

dict_Y_to_remove

num_workers_list = list(range(0, max_num_workers, num_workers_steps))
best_num_workers, table = tune_num_workers(Dataset = 
                                               num_workers_list = num_workers_list
                                               n_repetitions = n_repetitions,
                                               verbose = False)


 

     
