#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 18:42:12 2021

@author: ghiggi
"""
import time
import random
import torch
import numpy as np
import xarray as xr 
 
from modules.loss import update_w

##----------------------------------------------------------------------------.
###########################
#### Training function ####
###########################
def train(model, 
          training_info,   
          trainingDataLoader,
          validationDataLoader,  
          weights_loss, criterion,    # Loss stuffs 
          optimizer,                  # Optimizer options
          batch_size, epochs,         # Training options 
          # This will change      
          len_sqce_output, # TODO: ... 
          constants,  # TODO: to be removed 
          device):

    ##--------------------------------------------------------------------------.  
    # Initialize parameters and storage of results
    train_losses = []
    val_losses = []
    
    n_samples = trainingDataLoader.n_samples
    n_samples_val = validationDataLoader.n_samples
    num_nodes = trainingDataLoader.nodes
    out_features = trainingDataLoader.out_features
    
    # self.input_dynamic_features = self.da_dynamic.shape[-1]
    # self.input_bc_features = self.da_bc.shape[-1] 
    
    constants = trainingDataLoader.tensor_static
    
    num_constants = constants.shape[-1]  # depending on ds_static.transpose(1, 0) in training.py
    num_in_features = trainingDataLoader.shape[-1] + num_constants

    # Expand constants to match batch size
    constants_expanded = constants.expand(batch_size, num_nodes, num_constants)
    constants1 = constants_expanded.to(device)
    idxs_val = validationDataLoader.idxs
    
    ##------------------------------------------------------------------------.
    # produce only predictions of time-steps that have a positive contribution to the loss, 
    # otherwise, if the loss is 0 and they are not required for the prediction of future time-steps, 
    # avoid computation to save time and memory
    required_output = np.max(np.where(weights_loss > 0)) + 1 

    # save weight modifications along training phase
    weight_variations = [(weights_loss, 0, 0)]
    count_upd = 0
    
    # save loss for different time-ahead predictions to asses effect of weight variations
    train_loss_steps = {}
    for step_ahead in range(len_sqce_output):
        train_loss_steps['t{}'.format(step_ahead)] = []

    test_loss_steps = {}
    for step_ahead in range(len_sqce_output):
        test_loss_steps['t{}'.format(step_ahead)] = []

    threshold = 1e-4


    # iterate along epochs
    for epoch in range(epochs):

        print('\rEpoch : {}'.format(epoch), end="")

        val_loss = 0
        train_loss = 0

        model.train()

        random.shuffle(trainingDataLoader.idxs)
        idxs = trainingDataLoader.idxs

        batch_idx = 0
        train_loss_it = []
        
        # iterate along batches 
        for i in range(0, n_samples - batch_size, batch_size):
            i_next = min(i + batch_size, n_samples)

            # addapt constants size if necessary
            if len(idxs[i:i_next]) < batch_size:
                constants_expanded = constants.expand(len(idxs[i:i_next]), num_nodes, num_constants)
                constants1 = constants_expanded.to(device)

            batch, labels = trainingDataLoader[idxs[i:i_next]]

            # Transfer to GPU
            batch_size = batch[0].shape[0] // 2

            batch1 = torch.cat((batch[0][:batch_size, :, :], \
                                constants_expanded, batch[0][batch_size:, :, :], constants_expanded), dim=2).to(device)

            #  generate predictions multiple steps ahead sequentially
            output = model(batch1)

            label1 = labels[0].to(device)
            l0 = criterion(output, label1[batch_size:, :, :out_features])
            loss_ahead = weights_loss[0] * l0
            train_loss_steps['t0'].append(l0.item())

            #tbatch1 = time.time()
            for step_ahead in range(1, required_output):
                # input t-2
                inp_t2 = batch1[:, :, num_in_features:]

                #  toa at t-1
                toa_delta = labels[step_ahead][:batch_size, :, -1].view(-1, num_nodes, 1).to(device)
                batch1 = torch.cat((inp_t2, output, toa_delta, constants1), dim=2)

                output = model(batch1)
                label1 = labels[step_ahead].to(device)
                
                # evaluate loss 
                l0 = criterion(output, label1[batch_size:, :, :out_features])
                loss_ahead += weights_loss[step_ahead] * l0
                train_loss_steps['t{}'.format(step_ahead)].append(l0.item())

            #tbatch2 = time.time()
            optimizer.zero_grad()
            loss_ahead.backward()

            optimizer.step()

            #tbatch3 = time.time()

            train_loss += loss_ahead.item() * batch_size
            train_loss_it.append(train_loss / (batch_size * (batch_idx + 1)))
            
            # update weights 
            if len(train_loss_it) > 5:
                # allow weight updates if loss does not change after a certain number of epochs (count_upd)
                if (np.std(train_loss_it[-10:]) < threshold) and count_upd > 2e2:
                    weights_loss = update_w(weights_loss)
                    required_output = np.max(np.where(weights_loss > 0)) + 1 # update based on weights 
                    count_upd = 0
                    # print('New weights ', weights_loss, ' Epoch {} Iter {}'.format(epoch, i))
                    weight_variations.append((weights_loss, epoch, len(train_loss_steps['t0'])))
                    threshold /= 10
                else:
                    count_upd += 1

            if batch_idx % 50 == 0:
                print('\rBatch idx: {}; Loss: {:.3f} - Other {:.5f} - {}' \
                        .format(batch_idx, train_loss / (batch_size * (batch_idx + 1)), np.std(train_loss_it[-10:]),
                                count_upd),
                        end="")
            batch_idx += 1

        train_loss = train_loss / n_samples
        train_losses.append(train_loss)

        model.eval()
        constants1 = constants_expanded.to(device)
        with torch.set_grad_enabled(False):
            index = 0

            for i in range(0, n_samples_val - batch_size, batch_size):
                i_next = min(i + batch_size, n_samples_val)

                if len(idxs_val[i:i_next]) < batch_size:
                    constants_expanded = constants.expand(len(idxs_val[i:i_next]), num_nodes, num_constants)
                    constants1 = constants_expanded.to(device)

                batch, labels = trainingDataLoader[idxs[i:i_next]]

                # Transfer to GPU
                batch_size = batch[0].shape[0] // 2
                batch1 = torch.cat((batch[0][:batch_size, :, :], \
                                    constants_expanded, batch[0][batch_size:, :, :], constants_expanded), dim=2).to(
                    device)

                #  generate predictions multiple steps ahead sequentially
                
                output = model(batch1)

                label1 = labels[0].to(device)
                l0 = criterion(output, label1[batch_size:, :, :out_features]).item()
                loss_ahead = weights_loss[0] * l0
                test_loss_steps['t0'].append(l0)

                for step_ahead in range(1, required_output):
                    # input t-2
                    inp_t2 = batch1[:, :, num_in_features:]

                    #  toa at t-1
                    toa_delta = labels[step_ahead][:batch_size, :, -1].view(-1, num_nodes, 1).to(device)
                    batch1 = torch.cat((inp_t2, output, toa_delta, constants1), dim=2)

                    output = model(batch1)
                    
                    label1 = labels[step_ahead].to(device)
                    l0 = criterion(output, label1[batch_size:, :, :out_features]).item()
                    loss_ahead += weights_loss[step_ahead] * l0
                    test_loss_steps['t{}'.format(step_ahead)].append(l0)

                val_loss += loss_ahead * batch_size
                index = index + batch_size

        val_loss = val_loss / n_samples_val
        val_losses.append(val_loss)

        # Print stuff
        print('Epoch: {e:3d}/{n_e:3d}  - loss: {l:.3f}  - val_loss: {v_l:.5f}  - time: {t:2f}'
                .format(e=epoch + 1, n_e=epochs, l=train_loss, v_l=val_loss, t=-1))
        
    ##------------------------------------------------------------------------.
    # Update training info 
    training_info['training_loss'].append(train_losses)
    training_info['validation_loss'].append(val_losses)
    training_info['training_loss_steps'].append(train_loss_steps)
    training_info['test_loss_steps'].append(test_loss_steps)
    training_info['weight_variations'].append(weight_variations)
    training_info['weights_loss'] = weights_loss # TODO: append ?)
    training_info['criterion'] = criterion 
    training_info['optimizer'] = optimizer
    return training_info
