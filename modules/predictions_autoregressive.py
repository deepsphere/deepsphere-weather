#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 19:49:16 2021

@author: ghiggi
"""
import torch
import numpy as np
import time

# ## Predictions with Dask cluster 
# https://examples.dask.org/machine-learning/torch-prediction.html

def create_iterative_predictions_healpix_temp(model, device, dg, constants):
    
    out_feat = dg.out_features
    
    train_std =  dg.std.values[:2]
    train_mean = dg.mean.values[:2]
    
    delta_t = dg.delta_t
    len_sqce = dg.len_sqce
    max_lead_time = dg.max_lead_time
    nodes = dg.nodes
    initial_lead_time = delta_t * len_sqce
    total_feat = 7
    num_constants = constants.shape[1]
    
    
    batch_size = 10
    # Lead times
    lead_times = np.arange(delta_t, max_lead_time + delta_t, delta_t)
    
    
    # Actual times
    start = np.datetime64(dg.years[0], 'h') + np.timedelta64(initial_lead_time, 'h')
    stop = start + np.timedelta64(dg.n_samples, 'h')
    times = np.arange(start, stop)
    
    # Variables
    var_dict_out = {var: None for var in ['z', 't']}
    
    # Constants
    constants_expanded = constants.expand(batch_size, nodes, num_constants)
    constants1 = constants_expanded.to(device)
    idxs = dg.idxs
    
    
    dataloader = dg
    predictions = []
    model.eval()
    next_batch_ = []
    for num_lead, lead in enumerate(lead_times):
        
        print('\rProcessing lead time {} out of {}'.format(num_lead+1, len(lead_times)), end="")
        outputs = []
        
        batch_idx = 0
        batch_size = 10
        
        time1 = time.time()
        
        for i in range(0, len(idxs) - max_lead_time - len_sqce, batch_size):
            idx_pred = i
            i_next = i+batch_size
            i += num_lead*delta_t
            i_next += num_lead*delta_t

            if len(idxs[i:i_next]) < batch_size:
                constants_expanded = constants.expand(len(idxs[i:i_next]), nodes, num_constants)
                constants1 = constants_expanded.to(device)
            
            
            batch, _ = dg[idxs[i:i_next]]
            
                # Transfer to GPU
            batch_size = batch[0].shape[0]//2
                                
            if num_lead == 0:
                inputs = torch.cat((batch[0][:batch_size, :,:], \
                                    constants_expanded,batch[0][batch_size:, :,:] ,constants_expanded), dim=2).to(device)
                next_batch_.append(inputs.detach().cpu().clone())
                
            else:
                inputs = torch.cat((
                    next_batch_[batch_idx][:, :, total_feat:].to(device),\
                    torch.tensor(old_outputs[batch_idx]).to(device),\
                    batch[0][batch_size:, :,-1].view(-1, nodes, 1).to(device), \
                                    constants1), dim=2).to(device)
                
                next_batch_[batch_idx] = inputs.detach().cpu().clone() #store z, t, toa
                
            output = model(inputs)
            
            outputs.append(output.detach().cpu().clone().numpy()[:, :, :out_feat])
            batch_idx += 1
        
        
        old_outputs = outputs.copy()
        preds = np.concatenate(outputs)
        
        predictions.append(np.concatenate(outputs) * train_std + train_mean)

        time2 = time.time()
        
    predictions = np.array(predictions)
    
    return predictions, lead_times, times


  
