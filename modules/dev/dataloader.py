#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 00:04:12 2021

@author: ghiggi
"""
import time
import torch
import numpy as np
import xarray as xr 
from torch.utils.data import Dataset

from io import readDatasets  
##----------------------------------------------------------------------------.
### TODO
# da_static.transpose(1, 0) using xarray labels 
# da_static standardization using xarray 
# da_static standardization of categorical variables ? 
# --> Adapt training.py to da_bc and da_dynamic (instead of current data stacked)
# --> Line 133 to modify  

### Improvements
# - Add scalers options from x-scaler.py
# - Check that "auto" xarray adapt to nc/zarr chunking 
# - Provide more CV options whitin create_DataLoaders()

### Questions? 
# - Is it nodes necessary? 

#-----------------------------------------------------------------------------.
def temporary_data_scaling(da_static, 
                           da_training_dynamic, 
                           da_validation_dynamic, 
                           da_training_bc,
                           da_validation_bc):
    # Meanwhile ...
    scaler_dynamic = None 
    scaler_bc = None   
    scaler_static = 1 
    eps = 0.0001  # add to std to avoid division by 0
    ##------------------------------------------------------------------------. 
    ### - Scale static data here if requested (because not lazy loaded)
    if scaler_static is not None: 
        t_i = time.time()
        # TODO: This should be then cleaned when x-scaler.py is ready
        # - Would also be better to xarray Dataset still at this step ! 
        # - Scaling better to be done in xarray (dask) rather than torch (for speed) 

        constants_tensor = torch.tensor(da_static.values, dtype=torch.float)
        # standardize
        constants_tensor_mean = torch.mean(constants_tensor, dim=1, keepdim=True)
        constants_tensor_std = torch.std(constants_tensor, dim=1, keepdim=True)
        constants_tensor = (constants_tensor - constants_tensor_mean) / (constants_tensor_std + 1e-6) 
        print('- Scaling static data: {:.2f}s'.format(time.time() - t_i))
    ##------------------------------------------------------------------------.
    ### Compute dynamic data scaling statistics (on training set)
    if (scaler_dynamic is not None):
        raise ValueError("This is currently not implemented")
    else:
        t_i = time.time()
        dynamic_mean = da_training_dynamic.mean(('time', 'node')).compute()
        dynamic_std = da_training_dynamic.std(('time', 'node')).compute()
        print('- Scaling statics computation for dynamic data: {:.2f}s'.format(time.time() - t_i))
    ##------------------------------------------------------------------------. 
    # - Compute Boundary conditions data scaling statistics  (on training set)
    if (scaler_bc is not None):
        raise ValueError("This is currently not implemented")
    else:
        t_i = time.time()
        bc_mean = da_training_bc.mean(('time', 'node')).compute()
        bc_std = da_training_bc.std(('time', 'node')).compute()
        print('- Scaling statics computation for boundary conditions data: {:.2f}s'.format(time.time() - t_i))
    ##------------------------------------------------------------------------. 
    # - Standardize dynamic data 
    t_i = time.time()
    da_training_dynamic = (da_training_dynamic- dynamic_mean) / (dynamic_std + eps)
    da_validation_dynamic = (da_validation_dynamic- dynamic_mean) / (dynamic_std + eps)
    print('- Scaling of dynamic data: {:.2f}s'.format(time.time() - t_i))
    ##------------------------------------------------------------------------.
    # - Standardize boundary conditions data 
    t_i = time.time()
    da_training_bc = (da_training_bc - bc_mean) / (bc_std + eps)
    da_validation_bc = (da_validation_bc - bc_mean) / (bc_std + eps)
    print('- Scaling of boundary conditions data: {:.2f}s'.format(time.time() - t_i))   
    ##-------------------------------------------------------------------------
    return (constants_tensor, da_training_dynamic, da_validation_dynamic, da_training_bc, da_validation_bc)
    
#-----------------------------------------------------------------------------.    
class DataLoader(Dataset):
    """ Dataset used for graph models (1D), where data is loaded from stored numpy arrays.
    Parameters
    ----------
    ds : xarray Dataset
        Dataset containing the input data
    out_features : int
        Number of output features
    delta_t : int
        Temporal spacing between samples in temporal sequence (in hours)
    len_sqce_input : int
        Length of the input and output (predicted) sequences
    nodes : float
        Number of nodes each sample has
    max_lead_time : int
        Maximum lead time (in case of iterative predictions) in hours
    load : bool
        If true, load dataset to RAM
    """
    ## -------------------------------------------------------------------.
    def __init__(self,
                 da_dynamic,
                 da_bc,
                 da_static, # TODO: this is currently a torch.tensor already
                 # Autoregressive params (TODO MODIFY)
                 out_features, 
                 delta_t, 
                 len_sqce_input,
                 len_sqce_output,                           
                 max_lead_time, 
                 # Is it necessary ? 
                 nodes):                    
        ## -------------------------------------------------------------------.
        ### - Initialize autoregressive configs  (TODO: to change)
        self.delta_t = delta_t
        self.len_sqce_input = len_sqce_input
        self.len_sqce_output = len_sqce_output

        self.nodes = nodes
        self.out_features = out_features
        self.max_lead_time = max_lead_time
        
        ## -------------------------------------------------------------------.
        ### - Retrieve data
        self.da_dynamic = da_dynamic 
        self.da_bc = da_bc 
        # - TODO: If provided as xarray ... could be transposed by labels 
        self.tensor_static = da_static.transpose(1, 0) # why (1, 0)
        
        ## -------------------------------------------------------------------.
        ### - Count total number of samples
        # - TODO: modify here based on new indices 
        # - TODO: compute valid idx list: get_first_valid_idx(), get_last_valid_idx() 
             
        total_samples = self.da_dynamic.shape[0]
        if max_lead_time is None:
            self.n_samples = total_samples - (len_sqce_output + 1) * delta_t
        else:
            self.n_samples = total_samples - (len_sqce_output + 1) * delta_t - max_lead_time
            
        ##--------------------------------------------------------------------.
        ### - Generate sample indices
        # - TODO: compute valid idx list: get_first_valid_idx(), get_last_valid_idx()
        self.idxs = np.arange(self.n_samples)
    
        ##--------------------------------------------------------------------.
        # For current backward compatibility with training.py
        self.data = xr.merge(self.da_dynamic, self.da_bc) # TODO: Check if rise an error  
 
    ##------------------------------------------------------------------------.
    def __len__(self):
        return self.n_samples

    ##------------------------------------------------------------------------.
    def __getitem__(self, idx):
        """ Returns sample and label corresponding to an index as torch.Tensor objects
            The return tensor shapes are (for the sample and the label): [n_vertex, len_sqce_input, n_features]
        """
        # TODO: This is still the old (wrong) version 
        idx_data = idx
        idx_full = np.concatenate(
            np.array([[idx_data + self.delta_t * k] for k in range(self.len_sqce_input + self.len_sqce_output)])).reshape(-1)
        dat = self.data.isel(time=idx_full).values

        X = (
            torch.tensor(dat[:len(idx) * self.len_sqce_input, :, :],  
                         dtype=torch.float).reshape(len(idx) * self.len_sqce_input, self.nodes, -1),
        )

        y = [

            torch.tensor(dat[len(idx) * (self.len_sqce_input + k - 1):len(idx) * (self.len_sqce_input + k + 1), :, :],  
                         dtype=torch.float).reshape(len(idx) * self.len_sqce_input, self.nodes, -1)
            for k in range(self.len_sqce_output)
        ]
        return X, y

#-----------------------------------------------------------------------------.
def create_DataLoaders(data_dir, 
                       training_years,
                       validation_years,
                       chunk_size,
                       load_in_memory, 
                       # Data scalers 
                       scaler_dynamic,
                       scaler_bc,
                       scaler_static,
                       # Autoregressive params 
                       out_features, 
                       delta_t, 
                       len_sqce_input,
                       len_sqce_output,
                       max_lead_time,
                       # Graph setting
                       nodes):
    ##------------------------------------------------------------------------.
    ### Lazy Loading of Datasets 
    t_i = time.time()
    # - Dynamic data (i.e. pressure and surface levels variables)
    ds_dynamic = readDatasets(data_dir=data_dir, feature_type='dynamic', chunk_size=chunk_size)
    # - Boundary conditions data (i.e. TOA)
    ds_bc = readDatasets(data_dir=data_dir, feature_type='bc', chunk_size=chunk_size)
    # - Static features
    ds_static = readDatasets(data_dir=data_dir, feature_type='static', chunk_size=chunk_size)
    print('- Lazy Data Reading: {:.2f}s'.format(time.time() - t_i))
    
    ##------------------------------------------------------------------------. 
    ### Check data alignment and no missing timestep
    # TODO !!!
    # --> Functions are on ltenas3 server 
    
    ##------------------------------------------------------------------------.
    ### Split data into train, test and validation set 
    t_i = time.time()
    ds_training_dynamic = ds_dynamic.sel(time=slice(*training_years))
    ds_training_bc = ds_bc.sel(time=slice(*training_years))
      
    ds_validation_dynamic = ds_dynamic.sel(time=slice(*validation_years))
    ds_validation_bc = ds_bc.sel(time=slice(*validation_years))
    print('- Splitting data into train and validation set: {:.2f}s'.format(time.time() - t_i))
    
    ##------------------------------------------------------------------------.
    ### Load data in memory here if asked 
    if (load_in_memory is True):
        # Dynamic data
        print("- All training data are being loaded into memory:")
        t_i = time.time()
        ds_training_dynamic = ds_training_dynamic.load()
        print('  --> Training Dynamic Dataset: {:.2f}s'.format(time.time() - t_i))
        t_i = time.time()
        ds_training_bc = ds_training_bc.load()
        print('  --> Training Boundary Condition Dataset: {:.2f}s'.format(time.time() - t_i))
        
        ##--------------------------------------------------------------------.
        # Boundary conditions data
        print("- All validation data are being loaded into memory:")
        t_i = time.time()
        ds_validation_dynamic = ds_validation_dynamic.load()
        print('  --> Validation Dynamic Dataset: {:.2f}s'.format(time.time() - t_i))
        t_i = time.time()
        ds_validation_bc = ds_validation_bc.load()
        print('  --> Validation Boundary Condition Dataset: {:.2f}s'.format(time.time() - t_i))
    
    ##------------------------------------------------------------------------. 
    ### Conversion to DataArray and order dimensions 
    # - TODO: Check if it take times, and bc need dim extension 
    t_i = time.time()
    da_training_dynamic = ds_training_dynamic.to_array(dim='level', name='Dataset').transpose('time', 'node', 'level')
    da_validation_dynamic = ds_validation_dynamic.to_array(dim='level', name='Dataset').transpose('time', 'node', 'level')
    da_training_bc = ds_training_bc.to_array(dim='level', name='Dataset').transpose('time', 'node', 'level')
    da_validation_bc = ds_validation_bc.to_array(dim='level', name='Dataset').transpose('time', 'node', 'level')
    # TODO
    da_static = ds_static.to_array() # TODO ERROR --> To correct ! 
    # da_static.transpose(1, 0) # why (1, 0) done in DataLoader ... This must be cleaned out !!!
    print('- Conversion to DataArrays: {:.2f}s'.format(time.time() - t_i))
    # - TODO: Dimension order should be generalized (to a cfg setting) ?
    # - TODO: ['time', node, level, 'ensemble', 'feature'] <--> Discuss on order is more convenient 
   
    ##------------------------------------------------------------------------.
    ### Data Scaling   
    # - scaler = None: Should be provided when data are already standardized
    # - scaler = <scaler object to fit> 
    # - scaler =  <filepath of a saved scaler object>
    # --> Need to way x-scaler.py is ready
    # --> scaler.transform(...) 
    # - If data already in memory ... standardize on the fly 
    # - If data lazy loaded ... delayed standardization 
    # - If data lazy loaded (load_memory is False) but need scaler fit --> Stop --> Create before 
    
    # Meanwhile ....  # Do not remove "blank" line here below !
    da_static, da_training_dynamic, da_validation_dynamic, da_training_bc, da_validation_bc = temporary_data_scaling(da_static = da_static, 
                                                                                                                     da_training_dynamic = da_training_dynamic, 
                                                                                                                     da_validation_dynamic = da_validation_dynamic, 
                                                                                                                     da_training_bc = da_training_bc,
                                                                                                                     da_validation_bc = da_validation_bc)
    # ! Do not remove "blank" line here above !
    ##------------------------------------------------------------------------. 
    ### Create DataLoaders  
    print('- Starting DataLoaders creations:')
    # - Training DataLoader
    t_ii = time.time()
    trainingDataLoader = DataLoader(da_dynamic = da_training_dynamic,
                                    da_bc = da_training_bc,
                                    da_static = da_static, 
                                    # Autoregressive options 
                                    out_features=out_features, 
                                    delta_t=delta_t,
                                    len_sqce_input=len_sqce_input, 
                                    len_sqce_output=len_sqce_output, 
                                    max_lead_time=max_lead_time,
                                    nodes=nodes)
    print('- Training DataLoader creation: {:.2f}s'.format(time.time() - t_ii))
    
    ##------------------------------------------------------------------------.
    # - Validation DataLoader
    t_ii = time.time()
    validationDataLoader = DataLoader(da_dynamic = da_validation_dynamic,
                                      da_bc = da_validation_bc,
                                      da_static = da_static,  
                                      # Autoregressive options 
                                      out_features=out_features, 
                                      delta_t=delta_t,
                                      len_sqce_input=len_sqce_input, 
                                      len_sqce_output=len_sqce_output, 
                                      max_lead_time=max_lead_time,
                                      nodes=nodes)
    
    print('- Validation DataLoader creation: {:.2f}s'.format(time.time() - t_ii))
    ##------------------------------------------------------------------------. 
    return trainingDataLoader, validationDataLoader
