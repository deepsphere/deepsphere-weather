import xarray as xr
import numpy as np
import time
import os
import healpy as hp
import random
import matplotlib.pyplot as plt
from matplotlib import cm, colors

import torch
from torch import nn, optim
from torch.utils.data import Dataset

from modules.test import (compute_rmse_healpix, compute_relBIAS, compute_rSD, compute_R2, compute_anomalies, 
                          compute_relMAE, compute_ACC)

def load_data_split(input_dir, train_years, val_years, test_years, chunk_size, standardized=None):

    z500 = xr.open_mfdataset(f'{input_dir}geopotential_500/*.nc', combine='by_coords', \
                                 chunks={'time': chunk_size}).rename({'z': 'z500'})
    t850 = xr.open_mfdataset(f'{input_dir}temperature_850/*.nc', combine='by_coords', \
                             chunks={'time': chunk_size}).rename({'t': 't850'})
    rad = xr.open_mfdataset(f'{input_dir}toa_incident_solar_radiation/*.nc', combine='by_coords', \
                            chunks={'time': chunk_size})

    z500 = z500.isel(time=slice(7, None))
    t850 = t850.isel(time=slice(7, None))

    ds = xr.merge([z500, t850, rad], compat='override')

    ds_train = ds.sel(time=slice(*train_years))
    ds_valid = ds.sel(time=slice(*val_years))
    ds_test = ds.sel(time=slice(*test_years))


    return ds_train, ds_valid, ds_test

class WeatherBenchDatasetXarrayHealpixTemp(Dataset):
    
    """ Dataset used for graph models (1D), where data is loaded from stored numpy arrays.
    
    Parameters
    ----------
    ds : xarray Dataset
        Dataset containing the input data
    out_features : int
        Number of output features
    delta_t : int
        Temporal spacing between samples in temporal sequence (in hours)
    len_sqce : int
        Length of the input and output (predicted) sequences
    years : tuple(str)
        Years used to split the data
    nodes : float
        Number of nodes each sample has
    max_lead_time : int
        Maximum lead time (in case of iterative predictions) in hours
    load : bool
        If true, load dataset to RAM
    mean : np.ndarray of shape 2
        Mean to use for data normalization. If None, mean is computed from data
    std : np.ndarray of shape 2
        std to use for data normalization. If None, mean is computed from data
    """
        
    def __init__(self, ds, out_features, delta_t, len_sqce, years, nodes, nb_timesteps, 
                 max_lead_time=None, load=True, mean=None, std=None, requires_st=None):
        
        
        self.delta_t = delta_t
        self.len_sqce = len_sqce
        self.years = years
        
        self.nodes = nodes
        self.out_features = out_features
        self.max_lead_time = max_lead_time
        self.nb_timesteps = nb_timesteps
        
        self.data = ds.to_array(dim='level', name='Dataset').transpose('time', 'node', 'level')
        self.in_features = self.data.shape[-1]
        
        self.mean = self.data.mean(('time', 'node')).compute() if mean is None else mean
        self.std = self.data.std(('time', 'node')).compute() if std is None else std
        
        eps = 0.001 #add to std to avoid division by 0
        
        # Count total number of samples
        total_samples = self.data.shape[0]        
        
        if max_lead_time is None:
            self.n_samples = total_samples - (len_sqce+1) * delta_t
        else:
            self.n_samples = total_samples - (len_sqce+1) * delta_t - max_lead_time
        
        # Normalize

        if requires_st:
            self.data = self.data.groupby('time.month') - self.mean.to_array(dim='level')
            self.data = self.data.groupby('time.month') / self.std.to_array(dim='level')
            self.data.compute()
        else:
            self.data = (self.data - self.mean.to_array(dim='level')) / (
                        self.std.to_array(dim='level') + eps)
        self.data.persist()
        
        self.idxs = np.array(range(self.n_samples))
        
        print('Loading data to RAM...')
        t = time.time()
        self.data.load()
        print('Time: {:.2f}s'.format(time.time() - t))
        
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """ Returns sample and label corresponding to an index as torch.Tensor objects
            The return tensor shapes are (for the sample and the label): [n_vertex, len_sqce, n_features]
            
        """
        idx_data = idx#self.idxs[idx]
        #1,0,2
        
        #batch[0] --> (batch_size, num_nodes, n_features*len_sq)
        idx_full = np.concatenate(np.array([[idx_data+self.delta_t*k] for k in range(self.len_sqce+2)]).reshape(-1,1)) # ex: len_sqce=2 --> we need 0,1,2,3
        #idx_full = np.concatenate([idx_data+delta_t,  idx_data + delta_t * len_sqce, idx_data + delta_t * (len_sqce+1)])
        dat = self.data.isel(time=idx_full).values
        
        
        X = (
            torch.tensor(dat[:len(idx)*self.len_sqce,:,:] , \
                         dtype=torch.float).reshape(len(idx)*self.len_sqce, self.nodes, -1),
        )
        
        y = (torch.tensor(dat[len(idx):len(idx)*(self.len_sqce+1),:,:],\
                         dtype=torch.float).reshape(len(idx)*self.len_sqce, self.nodes, -1),\
             torch.tensor(dat[len(idx)*(self.len_sqce):,:,:self.out_features],\
                         dtype=torch.float).reshape(len(idx)*self.len_sqce, self.nodes, -1)
        
        )
        return X, y


class WeatherBenchDatasetXarrayHealpixTempMultiple(Dataset):
    """ Dataset used for graph models (1D), where data is loaded from stored numpy arrays.

    Parameters
    ----------
    ds : xarray Dataset
        Dataset containing the input data
    out_features : int
        Number of output features
    delta_t : int
        Temporal spacing between samples in temporal sequence (in hours)
    len_sqce : int
        Length of the input and output (predicted) sequences
    years : tuple(str)
        Years used to split the data
    nodes : float
        Number of nodes each sample has
    max_lead_time : int
        Maximum lead time (in case of iterative predictions) in hours
    load : bool
        If true, load dataset to RAM
    mean : np.ndarray of shape 2
        Mean to use for data normalization. If None, mean is computed from data
    std : np.ndarray of shape 2
        std to use for data normalization. If None, mean is computed from data
    """

    def __init__(self, ds, out_features, delta_t, len_sqce_input, len_sqce_output, years, nodes, nb_timesteps,
                 max_lead_time=None, load=True, mean=None, std=None, requires_st=None):

        self.delta_t = delta_t
        self.len_sqce = len_sqce_input
        self.len_output = len_sqce_output
        self.years = years

        self.nodes = nodes
        self.out_features = out_features
        self.max_lead_time = max_lead_time
        self.nb_timesteps = nb_timesteps

        self.data = ds.to_array(dim='level', name='Dataset').transpose('time', 'node', 'level')
        self.in_features = self.data.shape[-1]

        self.mean = self.data.mean(('time', 'node')).compute() if mean is None else mean
        self.std = self.data.std(('time', 'node')).compute() if std is None else std

        eps = 0.001  # add to std to avoid division by 0

        # Count total number of samples
        total_samples = self.data.shape[0]

        if max_lead_time is None:
            self.n_samples = total_samples - (len_sqce_output + 1) * delta_t
        else:
            self.n_samples = total_samples - (len_sqce_output + 1) * delta_t - max_lead_time

        # Normalize

        if requires_st:
            self.data = self.data.groupby('time.month') - self.mean.to_array(dim='level')
            self.data = self.data.groupby('time.month') / self.std.to_array(dim='level')
            self.data.compute()
        else:
            self.data = (self.data - self.mean.to_array(dim='level')) / (
                    self.std.to_array(dim='level') + eps)
        self.data.persist()

        self.idxs = np.array(range(self.n_samples))

        print('Loading data to RAM...')
        t = time.time()
        self.data.load()
        print('Time: {:.2f}s'.format(time.time() - t))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """ Returns sample and label corresponding to an index as torch.Tensor objects
            The return tensor shapes are (for the sample and the label): [n_vertex, len_sqce, n_features]

        """
        idx_data = idx
        idx_full = np.concatenate(
            np.array([[idx_data + self.delta_t * k] for k in range(self.len_sqce + self.len_output)])).reshape(-1)
        dat = self.data.isel(time=idx_full).values

        X = (
            torch.tensor(dat[:len(idx) * self.len_sqce, :, :], \
                         dtype=torch.float).reshape(len(idx) * self.len_sqce, self.nodes, -1),
        )

        y = [

            torch.tensor(dat[len(idx) * (self.len_sqce + k - 1):len(idx) * (self.len_sqce + k + 1), :, :], \
                         dtype=torch.float).reshape(len(idx) * self.len_sqce, self.nodes, -1)
            for k in range(self.len_output)
        ]
        return X, y


def train_model_2steps_error(model, device, training_ds, constants, batch_size, max_error, lr, validation_ds):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-7, weight_decay=0, amsgrad=False)

    train_losses = []
    val_losses = []
    n_samples = training_ds.n_samples
    n_samples_val = validation_ds.n_samples
    num_nodes = training_ds.nodes
    num_constants = constants.shape[1]
    out_features = training_ds.out_features

    constants_expanded = constants.expand(batch_size, num_nodes, num_constants)
    constants1 = constants_expanded.to(device)
    idxs_val = validation_ds.idxs
    error_val = 10000
    epoch = 0
    while error_val > max_error:

        print('\rEpoch : {}'.format(epoch), end="")

        time1 = time.time()

        val_loss = 0
        train_loss = 0

        model.train()

        random.shuffle(training_ds.idxs)
        idxs = training_ds.idxs

        batch_idx = 0
        train_loss_it = []
        times_it = []
        t0 = time.time()
        for i in range(0, n_samples - batch_size, batch_size):
            i_next = min(i + batch_size, n_samples)

            if len(idxs[i:i_next]) < batch_size:
                constants_expanded = constants.expand(len(idxs[i:i_next]), num_nodes, num_constants)
                constants1 = constants_expanded.to(device)

            batch, labels = training_ds[idxs[i:i_next]]

            # Transfer to GPU
            batch_size = batch[0].shape[0] // 2
            batch1 = torch.cat((batch[0][:batch_size, :, :], \
                                constants_expanded, batch[0][batch_size:, :, :], constants_expanded), dim=2).to(device)
            label1 = labels[0].to(device)
            label2 = labels[1].to(device)

            # t3 = time.time()
            batch_size = batch1.shape[0]

            # Model
            output1 = model(batch1)
            toa_delta = batch[0][batch_size:, :, -1].view(-1, num_nodes, 1).to(device)

            batch2 = torch.cat((output1, toa_delta, constants1, \
                                label1[batch_size:, :, :], constants1), dim=2)

            output2 = model(batch2)
            loss = criterion(output1, label1[batch_size:, :, :out_features]) + criterion(output2, label2[batch_size:, :,
                                                                                                  :out_features])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #return batch1, batch2, output1, output2
            train_loss = train_loss + loss.item() * batch_size
            train_loss_it.append(train_loss / (batch_size * (batch_idx + 1)))
            times_it.append(time.time() - t0)
            t0 = time.time()

            if batch_idx % 50 == 0:
                print('\rBatch idx: {}; Loss: {:.3f}'.format(batch_idx, train_loss / (batch_size * (batch_idx + 1))),
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

                # t1 = time.time()
                batch, labels = validation_ds[idxs_val[i:i_next]]
                # Transfer to GPU
                batch_size = batch[0].shape[0] // 2

                batch1 = torch.cat((batch[0][:batch_size, :, :], \
                                    constants_expanded, batch[0][batch_size:, :, :], constants_expanded), dim=2).to(
                    device)
                label1 = labels[0].to(device)
                label2 = labels[1].to(device)

                batch_size = batch1.shape[0]

                output1 = model(batch1)
                toa_delta = batch[0][batch_size:, :, -1].view(-1, num_nodes, 1).to(device)
                batch2 = torch.cat((output1, toa_delta, constants1, \
                                    label1[batch_size:, :, :], constants1), dim=2)
                output2 = model(batch2)

                val_loss = val_loss + (criterion(output1, label1[batch_size:, :, :out_features]).item()
                                       + criterion(output2, label2[batch_size:, :, :out_features]).item()) * batch_size
                index = index + batch_size

        val_loss = val_loss / n_samples_val
        val_losses.append(val_loss)

        error_val = val_loss
        epoch += 1

        time2 = time.time()

        # Print stuff
        print('Epoch: {e:3d}  - loss: {l:.3f}  - val_loss: {v_l:.5f}  - time: {t:2f}'
              .format(e=epoch + 1, l=train_loss, v_l=val_loss, t=time2 - time1))

    return train_losses, val_losses, train_loss_it, times_it
    
    
def train_model_2steps(model, device, training_ds, constants, batch_size, epochs, lr, validation_ds, model_name):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-7, weight_decay=0, amsgrad=False)
    
    train_losses = []
    val_losses = []
    n_samples = training_ds.n_samples
    n_samples_val = validation_ds.n_samples
    num_nodes = training_ds.nodes
    num_constants = constants.shape[1]
    out_features = training_ds.out_features
    
    constants_expanded = constants.expand(batch_size, num_nodes, num_constants)
    constants1 = constants_expanded.to(device)
    idxs_val = validation_ds.idxs

    train_loss_steps = {}
    for step_ahead in range(2):
        train_loss_steps['t{}'.format(step_ahead)] = []

    for epoch in range(epochs):
        
        print('\rEpoch : {}'.format(epoch), end="")
        
        time1 = time.time()
        
        val_loss = 0
        train_loss = 0
        
        model.train()  
        
        random.shuffle(training_ds.idxs)
        idxs = training_ds.idxs
        
        batch_idx = 0
        train_loss_it = []
        times_it = []
        t0 = time.time()
        for i in range(0, n_samples - batch_size, batch_size):
            i_next = min(i + batch_size, n_samples)
            
            if len(idxs[i:i_next]) < batch_size:
                constants_expanded = constants.expand(len(idxs[i:i_next]), num_nodes, num_constants)
                constants1 = constants_expanded.to(device)
        
            batch, labels = training_ds[idxs[i:i_next]]
            
            # Transfer to GPU
            batch_size = batch[0].shape[0]//2
            batch1 = torch.cat((batch[0][:batch_size, :,:], \
                                constants_expanded,batch[0][batch_size:, :,:] ,constants_expanded), dim=2).to(device)
            label1 = labels[0].to(device)
            label2 = labels[1].to(device)
            
            #t3 = time.time()
            batch_size = batch1.shape[0]
            
            # Model
            output1 = model(batch1)  
            toa_delta = batch[0][batch_size:, :,-1].view(-1, num_nodes, 1).to(device)
            batch2 = torch.cat((output1, toa_delta, constants1, \
                               label1[batch_size:, :,:], constants1), dim=2)
            
            output2 = model(batch2)

            l0 = criterion(output1, label1[batch_size:,:,:out_features])
            l1 = criterion(output2, label2[batch_size:,:,:out_features])
            loss =  l0 + l1

            train_loss_steps['t0'].append(l0.item())
            train_loss_steps['t1'].append(l1.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss = train_loss + loss.item() * batch_size
            train_loss_it.append(train_loss/(batch_size*(batch_idx+1)))
            times_it.append(time.time()-t0)
            t0 = time.time()

            #if train_loss/(batch_size*(batch_idx+1)) > 50 and batch_idx > 40:
            #    return batch1, batch2, label1, label2, output1, output2
            if batch_idx%5 == 0:
                print('\rBatch idx: {}; Loss: {:.3f};\tL0: {:.4f}\t L1: {:.4f}'.format(batch_idx, train_loss/(batch_size*(batch_idx+1)), l0.item(), l1.item()), end="")
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


                #t1 = time.time()
                batch, labels = validation_ds[idxs_val[i:i_next]]
                # Transfer to GPU
                batch_size = batch[0].shape[0]//2
            
                batch1 = torch.cat((batch[0][:batch_size, :,:], \
                                    constants_expanded,batch[0][batch_size:, :,:] ,constants_expanded), dim=2).to(device)
                label1 = labels[0].to(device)
                label2 = labels[1].to(device)

                batch_size = batch1.shape[0]

                output1 = model(batch1)  
                toa_delta = batch[0][batch_size:, :,-1].view(-1, num_nodes, 1).to(device)
                batch2 = torch.cat((output1, toa_delta, constants1, \
                                   label1[batch_size:, :,:], constants1), dim=2)
                output2 = model(batch2)
                
                val_loss = val_loss + (criterion(output1, label1[batch_size:,:,:out_features]).item() 
                                       + criterion(output2, label2[batch_size:,:,:out_features]).item()) * batch_size
                index = index + batch_size
                
        val_loss = val_loss / n_samples_val
        val_losses.append(val_loss)
        
        time2 = time.time()
        
        # Print stuff
        print('Epoch: {e:3d}/{n_e:3d}  - loss: {l:.3f}  - val_loss: {v_l:.5f}  - time: {t:2f}'
              .format(e=epoch+1, n_e=epochs, l=train_loss, v_l=val_loss, t=time2-time1))

        torch.save(model.state_dict(), model_name[:-3] + '_epoch{}'.format(epoch) + '.h5')
        
    return train_losses, val_losses, train_loss_it, times_it, train_loss_steps



def train_model_multiple_steps(model, device, training_ds, constants, batch_size, epochs, lr, validation_ds):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-7, weight_decay=0, amsgrad=False)

    train_losses = []
    val_losses = []
    n_samples = training_ds.n_samples
    n_samples_val = validation_ds.n_samples
    num_nodes = training_ds.nodes
    num_constants = constants.shape[1]
    out_features = training_ds.out_features

    constants_expanded = constants.expand(batch_size, num_nodes, num_constants)
    constants1 = constants_expanded.to(device)
    idxs_val = validation_ds.idxs

    for epoch in range(epochs):

        print('\rEpoch : {}'.format(epoch), end="")

        time1 = time.time()

        val_loss = 0
        train_loss = 0

        model.train()

        random.shuffle(training_ds.idxs)
        idxs = training_ds.idxs

        batch_idx = 0
        train_loss_it = []
        times_it = []
        t0 = time.time()
        for i in range(0, n_samples - batch_size, batch_size):
            i_next = min(i + batch_size, n_samples)

            if len(idxs[i:i_next]) < batch_size:
                constants_expanded = constants.expand(len(idxs[i:i_next]), num_nodes, num_constants)
                constants1 = constants_expanded.to(device)

            batch, labels = training_ds[idxs[i:i_next]]

            # Transfer to GPU
            batch_size = batch[0].shape[0] // 2
            batch1 = torch.cat((batch[0][:batch_size, :, :], \
                                constants_expanded, batch[0][batch_size:, :, :], constants_expanded), dim=2).to(device)
            label1 = labels[0].to(device)
            label2 = labels[1].to(device)

            # t3 = time.time()
            batch_size = batch1.shape[0]

            # Model
            output1 = model(batch1)
            toa_delta = batch[0][batch_size:, :, -1].view(-1, num_nodes, 1).to(device)
            batch2 = torch.cat((output1, toa_delta, constants1, \
                                label1[batch_size:, :, :], constants1), dim=2)

            output2 = model(batch2)
            loss = criterion(output1, label1[batch_size:, :, :out_features]) + criterion(output2, label2[batch_size:, :,
                                                                                                  :out_features])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = train_loss + loss.item() * batch_size
            train_loss_it.append(train_loss / (batch_size * (batch_idx + 1)))
            times_it.append(time.time() - t0)
            t0 = time.time()

            if batch_idx % 50 == 0:
                print('\rBatch idx: {}; Loss: {:.3f}'.format(batch_idx, train_loss / (batch_size * (batch_idx + 1))),
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

                # t1 = time.time()
                batch, labels = validation_ds[idxs_val[i:i_next]]
                # Transfer to GPU
                batch_size = batch[0].shape[0] // 2

                batch1 = torch.cat((batch[0][:batch_size, :, :], \
                                    constants_expanded, batch[0][batch_size:, :, :], constants_expanded), dim=2).to(
                    device)
                label1 = labels[0].to(device)
                label2 = labels[1].to(device)

                batch_size = batch1.shape[0]

                output1 = model(batch1)
                toa_delta = batch[0][batch_size:, :, -1].view(-1, num_nodes, 1).to(device)
                batch2 = torch.cat((output1, toa_delta, constants1, \
                                    label1[batch_size:, :, :], constants1), dim=2)
                output2 = model(batch2)

                val_loss = val_loss + (criterion(output1, label1[batch_size:, :, :out_features]).item()
                                       + criterion(output2, label2[batch_size:, :, :out_features]).item()) * batch_size
                index = index + batch_size

        val_loss = val_loss / n_samples_val
        val_losses.append(val_loss)

        time2 = time.time()

        # Print stuff
        print('Epoch: {e:3d}/{n_e:3d}  - loss: {l:.3f}  - val_loss: {v_l:.5f}  - time: {t:2f}'
              .format(e=epoch + 1, n_e=epochs, l=train_loss, v_l=val_loss, t=time2 - time1))

    return train_losses, val_losses, train_loss_it, times_it


def create_iterative_predictions_healpix_temp(model, device, dg, constants):
    
    out_feat = dg.out_features
    
    train_std =  dg.std[['z500','t850']].to_array().values #dg.std.values[:out_feat]
    train_mean = dg.mean[['z500','t850']].to_array().values #dg.mean.values[:out_feat]
    
    delta_t = dg.delta_t
    len_sqce = dg.len_sqce
    max_lead_time = dg.max_lead_time
    initial_lead_time = delta_t * len_sqce
    nodes = dg.nodes
    nside = int(np.sqrt(nodes/12))
    n_samples = dg.n_samples
    in_feat = 2#dg.in_features
    total_feat = 7
    num_constants = constants.shape[1]
    
    
    batch_size = 10
    # Lead times
    lead_times = np.arange(delta_t, max_lead_time + delta_t, delta_t)
    
    # Lat lon coordinates
    out_lon, out_lat = hp.pix2ang(nside, np.arange(nodes), lonlat=True)
    
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
    
    return predictions, lead_times, times, nodes, out_lat, out_lon

def _inner(x, y):
    result = np.matmul(x[..., np.newaxis, :], y[..., :, np.newaxis])
    return result[..., 0, 0]


def inner_product(x, y, dim):
    return xr.apply_ufunc(_inner, x, y, input_core_dims=[[dim], [dim]])


def compute_errors(pred, obs):

    print('loading data')
    t1 = time.time()
    pred.load()
    obs.load()
    t2 = time.time()
    print('time to load data: ', t2 - t1)
    dims = ('time')

    obs_mean = obs.mean(dims)
    t3 = time.time()
    print('time mean obs ', t3 - t2)
    pred_mean = pred.mean(dims)
    t4 = time.time()
    print('time mean pred ', t4 - t3)

    obs_std = obs.std(dims)
    t5 = time.time()
    print('time std obs ', t5 - t4)
    pred_std = pred.std(dims)
    t6 = time.time()
    print('time std pred ', t6 - t5)

    cov = inner_product(pred - pred_mean, obs - obs_mean, dim=dims) / pred.count(dims)
    t7 = time.time()
    print('time cov ', t7 - t6)
    corr_map = (cov / (pred_std * obs_std)) ** 2

    t8 = time.time()
    print('time corre_map ', t8 - t7)
    error = pred - obs
    rbias_map = error.mean(dims) / obs_mean
    t9 = time.time()
    print('time rbias map ', t9 - t8)
    rsd_map = pred_std / obs_std

    t10= time.time()
    print('time rsd map ', t10 - t9)

    rmse = np.sqrt(((error) ** 2).mean(dims))
    rmse_map = rmse.drop('lat').drop('lon').load()

    t11 = time.time()
    print('time rmse map ', t11 - t10)
    """
    t0 = time.time()
    corr_map = compute_R2(pred, obs, dims=('time'))
    t1 = time.time()
    print('time corr_map  ', t1 - t0)
    rbias_map = compute_relBIAS(pred, obs, dims=('time'))
    t2 = time.time()
    print('time rbias_map ', t2 - t1)
    rsd_map = compute_rSD(pred, obs, dims=('time'))
    t3 = time.time()
    print('time rsd_map ', t3 - t2)
    rmse_map = compute_rmse_healpix(pred, obs, dims=('time'))
    t4 = time.time()
    print('time rmse map ', t4 - t3)
    """
    obs_rmse = obs.mean(('node', 'time')) #np.sqrt(((obs)**2).mean(('node', 'time')))
    rmse_map_norm = rmse_map / obs_rmse
    
    return corr_map, rbias_map, rsd_map, rmse_map, obs_rmse, rmse_map_norm


def plot_climatology(figname, preds, vals, ticks, lat_labels, month_labels):
    
    diff_z = preds['pred_z'] / preds['obs_z']
    diff_t = preds['pred_t'] / preds['obs_t']
    
    f, axs = plt.subplots(3, 2, figsize=(17, 15))

    # Plots
    vmin_z = vals['vmin_z']
    vmax_z = vals['vmax_z']
    vmin_t = vals['vmin_t']
    vmax_t = vals['vmax_t']
    vmin_sd = vals['vmin_sd']
    vmax_sd = vals['vmax_sd']

    cax1 = axs[0, 0].pcolormesh(preds['pred_z'], cmap='Spectral_r', vmin=vmin_z, vmax=vmax_z, 
                                shading='gouraud')
    axs[0, 0].axhline(y=15, linestyle='--', color='black')
    cax2 = axs[0, 1].pcolormesh(preds['pred_t'], cmap='Spectral_r', vmin=vmin_t, vmax=vmax_t, 
                                shading='gouraud')
    axs[0, 1].axhline(y=15, linestyle='--', color='black')
    cax3 = axs[1, 0].pcolormesh(preds['obs_z'], cmap='Spectral_r', vmin=vmin_z, vmax=vmax_z, 
                                shading='gouraud')
    axs[1, 0].axhline(y=15, linestyle='--', color='black')
    cax4 = axs[1, 1].pcolormesh(preds['obs_t'], cmap='Spectral_r', vmin=vmin_t, vmax=vmax_t, 
                                shading='gouraud')
    axs[1, 1].axhline(y=15, linestyle='--', color='black')
    cax5 = axs[2, 0].pcolormesh(diff_z, cmap='RdBu_r', vmin=vmin_sd, vmax=vmax_sd, shading='gouraud')
    axs[2, 0].axhline(y=15, linestyle='--', color='black')
    cax6 = axs[2, 1].pcolormesh(diff_t, cmap='RdBu_r', vmin=vmin_sd, vmax=vmax_sd, shading='gouraud')
    axs[2, 1].axhline(y=15, linestyle='--', color='black')


    # Axes
    axs[0, 0].set_title('Z500$_{\mathrm{\mathsf{predicted}}}$', fontsize=22)
    axs[0, 1].set_title('T850$_{\mathrm{\mathsf{predicted}}}$', fontsize=22)
    axs[1, 0].set_title('Z500$_{\mathrm{\mathsf{observed}}}$', fontsize=22)
    axs[1, 1].set_title('T850$_{\mathrm{\mathsf{observed}}}$', fontsize=22)
    axs[2, 0].set_title('Z500$_{\mathrm{\mathsf{predicted}}}$ / Z500$_{\mathrm{\mathsf{observed}}}$', fontsize=22)
    axs[2, 1].set_title('T850$_{\mathrm{\mathsf{predicted}}}$ / T850$_{\mathrm{\mathsf{observed}}}$', fontsize=22)
    axs[2, 0].set_xticks(np.arange(12))
    axs[2, 0].set_xticklabels(month_labels, fontsize=16)
    axs[2, 1].set_xticks(np.arange(12))
    axs[2, 1].set_xticklabels(month_labels, fontsize=16)
    axs[0, 0].set_yticks(ticks)
    axs[0, 0].set_yticklabels(lat_labels, fontsize=16)
    axs[1, 0].set_yticks(ticks)
    axs[1, 0].set_yticklabels(lat_labels, fontsize=16)
    axs[2, 0].set_yticks(ticks)
    axs[2, 0].set_yticklabels(lat_labels, fontsize=16)
    axs[2, 0].set_xlabel('Month', fontsize=20)
    axs[2, 1].set_xlabel('Month', fontsize=20)
    axs[0, 0].set_ylabel('Latitude [$^\circ$]', fontsize=20)
    axs[1, 0].set_ylabel('Latitude [$^\circ$]', fontsize=20)
    axs[2, 0].set_ylabel('Latitude [$^\circ$]', fontsize=20)

    axs[0, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axs[1, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axs[0, 1].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, 
                          labelbottom=False, labelleft=False)
    axs[1, 1].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, 
                          labelbottom=False, labelleft=False)
    axs[2, 1].tick_params(axis='y', which='both',left=False, right=False, labelleft=False)

    # Colorbars
    cb1 = f.colorbar(cax1, ax=axs[0, 0], norm=colors.Normalize(vmin=vmin_z,vmax=vmax_z),
               fraction=0.02, aspect=40, pad=0.05)
    cb2 = f.colorbar(cax2, ax=axs[0, 1], norm=colors.Normalize(vmin=vmin_t, vmax=vmax_t),
               fraction=0.02, aspect=40, pad=0.05)
    cb3 = f.colorbar(cax3, ax=axs[1, 0], norm=colors.Normalize(vmin=vmin_z,vmax=vmax_z),
               fraction=0.02, aspect=40, pad=0.05)
    cb4 = f.colorbar(cax4, ax=axs[1, 1], norm=colors.Normalize(vmin=vmin_t, vmax=vmax_t),
               fraction=0.02, aspect=40, pad=0.05)
    cb5 = f.colorbar(cax5, ax=axs[2, 0], norm=colors.Normalize(vmin=vmin_sd, vmax=vmax_sd),
               fraction=0.02, aspect=40, pad=0.05)
    cb6 = f.colorbar(cax6, ax=axs[2, 1], norm=colors.Normalize(vmin=vmin_sd, vmax=vmax_sd),
               fraction=0.02, aspect=40, pad=0.05)

    cb1.set_label(label='[$m^2 s^{-2}$]', size=18)
    cb1.ax.tick_params(labelsize=16)
    cb2.set_label(label='[K]', size=18)
    cb2.ax.tick_params(labelsize=16)
    cb3.set_label(label='[$m^2 s^{-2}$]', size=18)
    cb3.ax.tick_params(labelsize=16)
    cb4.set_label(label='[K]', size=18)
    cb4.ax.tick_params(labelsize=16)
    cb5.ax.tick_params(labelsize=16)
    cb6.ax.tick_params(labelsize=16)

    plt.tight_layout()

    figname += '.png'
    plt.savefig(figname , bbox_inches='tight')

    plt.show()