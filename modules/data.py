import xarray as xr
import numpy as np

import torch
from torch import nn, optim 
from torch.utils.data import Dataset


# Test data
def load_test_data(path, var, years=slice('2017', '2018')):
    """
    Args:
        path: Path to nc files
        var: variable. Geopotential = 'z', Temperature = 't'
        years: slice for time window
    Returns:
        dataset: Concatenated dataset for 2017 and 2018
    """
    assert var in ['z', 't'], 'Test data only for Z500 and T850'
    ds = xr.open_mfdataset(f'{path}/*.nc', combine='by_coords')[var]
    try:
        ds = ds.sel(level=500 if var == 'z' else 850).drop('level')
    except ValueError:
        pass
    return ds.sel(time=years)


# Datasets
class Weather_Dataset(Dataset):
    
    def __init__(self, datadir, variables, level, years, lead_time, load=True, mean=None, std=None):
        
        var_dict = {var: levels for var in variables}
        
        ds = []
        for var, level in var_dict.items():
            ds.append(xr.open_mfdataset(f'{datadir}{var}_{level}/*.nc', combine='by_coords'))
            
        ds = xr.merge(ds, compat='override')
        ds = ds.sel(time=slice(*years))
        
        self.ds = ds
        self.var_dict = var_dict
        self.lead_time = lead_time
        self.features = len(var_dict)
        
        data = []
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
        for var, levels in var_dict.items():
            try:
                data.append(ds[var].sel(level=levels))
            except ValueError:
                data.append(ds[var].expand_dims({'level': generic_level}, 1))

        self.data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')
        self.mean = self.data.mean(('time', 'lat', 'lon')).compute() if mean is None else mean
        self.std = self.data.std('time').mean(('lat', 'lon')).compute() if std is None else std
        
        # Normalize
        self.data = (self.data - self.mean) / self.std
        
        # Stack
        self.data = self.data.stack(nodes=('lat', 'lon')).transpose('time', 'nodes', 'level')
        self.nodes = len(ds['nodes'])
        
        self.n_samples = self.data.isel(time=slice(0, -lead_time)).shape[0]
        self.idxs = np.arange(self.n_samples)
        self.init_time = self.data.isel(time=slice(None, -lead_time)).time
        self.valid_time = self.data.isel(time=slice(lead_time, None)).time
        
        if load: print('Loading data into RAM'); self.data.load()
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        idxs = self.idxs[idx]
        X = torch.Tensor(self.data.isel(time=idxs).values).view((self.nodes, self.features))
        y = torch.Tensor(self.data.isel(time=idxs + self.lead_time).values).view((self.nodes, self.features))

        return X, y


class Dataset_WeatherBench_1D(Dataset):
    
    def __init__(self, ds, var_dict, lead_time, load=True, mean=None, std=None):
        
        self.ds = ds
        self.var_dict = var_dict
        self.lead_time = lead_time
        self.features = len(var_dict)
    
        data = []
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
        for var, levels in var_dict.items():
            try:
                data.append(ds[var].sel(level=levels))
            except ValueError:
                data.append(ds[var].expand_dims({'level': generic_level}, 1))

        self.data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')
        self.mean = self.data.mean(('time', 'lat', 'lon')).compute() if mean is None else mean
        self.std = self.data.std('time').mean(('lat', 'lon')).compute() if std is None else std
        
        # Normalize
        self.data = (self.data - self.mean) / self.std
        
        # Stack
        self.data = self.data.stack(nodes=('lat', 'lon')).transpose('time', 'nodes', 'level')
        self.nodes = len(self.data['nodes'])
        
        self.n_samples = self.data.isel(time=slice(0, -lead_time)).shape[0]
        self.idxs = np.arange(self.n_samples)
        self.init_time = self.data.isel(time=slice(None, -lead_time)).time
        self.valid_time = self.data.isel(time=slice(lead_time, None)).time
        
        if load: print('Loading data into RAM'); self.data.load()
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        idxs = self.idxs[idx]
        X = torch.Tensor(self.data.isel(time=idxs).values).view((self.nodes, self.features))
        y = torch.Tensor(self.data.isel(time=idxs + self.lead_time).values).view((self.nodes, self.features))

        return X, y


class Dataset_WeatherBench_2D(Dataset):
    
    def __init__(self, ds, var_dict, lead_time, load=True, mean=None, std=None):
        
        self.ds = ds
        self.var_dict = var_dict
        self.lead_time = lead_time
        self.lat = len(ds['lat'])
        self.lon = len(ds['lon'])
        self.features = len(var_dict)
    
        data = []
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
        for var, levels in var_dict.items():
            try:
                data.append(ds[var].sel(level=levels))
            except ValueError:
                data.append(ds[var].expand_dims({'level': generic_level}, 1))

        self.data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')
        self.mean = self.data.mean(('time', 'lat', 'lon')).compute() if mean is None else mean
        self.std = self.data.std('time').mean(('lat', 'lon')).compute() if std is None else std
        
        # Normalize
        self.data = (self.data - self.mean) / self.std
        
        self.n_samples = self.data.isel(time=slice(0, -lead_time)).shape[0]
        self.idxs = np.arange(self.n_samples)
        self.init_time = self.data.isel(time=slice(None, -lead_time)).time
        self.valid_time = self.data.isel(time=slice(lead_time, None)).time
        
        if load: print('Loading data into RAM'); self.data.load()
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        idxs = self.idxs[idx]
        X = torch.Tensor(self.data.isel(time=idxs).values).view((self.features, self.lat, self.lon))
        y = torch.Tensor(self.data.isel(time=idxs + self.lead_time).values).view((self.features, 
                                                                                  self.lat, self.lon))

        return X, y