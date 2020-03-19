import xarray as xr
import numpy as np
import os

import torch
from torch import nn, optim 
from torch.utils.data import Dataset


# Test data
def load_test_data(path, delta_t, years=slice('2017', '2018')):
    zpath = path + '/geopotential_500'
    tpath = path + '/temperature_850'
    
    z = xr.open_mfdataset(zpath+'/*.nc', combine='by_coords')['z']
    t = xr.open_mfdataset(tpath+'/*.nc', combine='by_coords')['t']

    try:
        z = z.drop('level')
    except ValueError:
        pass

    try:
        t = t.drop('level')
    except ValueError:
        pass

    dataset = xr.merge([z, t], compat='override')

    return dataset.sel(time=years).isel(time=slice(delta_t, None))


# Datasets
class WeatherDataset(Dataset):
    
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
        
        self.n_samples = self.data.isel(time=slice(0, -lead_time)).shape[0]
        self.idxs = np.arange(self.n_samples)
        self.init_time = self.data.isel(time=slice(None, -lead_time)).time
        self.valid_time = self.data.isel(time=slice(lead_time, None)).time
        
        if load: print('Loading data into RAM'); self.data.load()
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        idxs = self.idxs[idx]
        X = torch.Tensor(self.data.isel(time=idxs).values)
        y = torch.Tensor(self.data.isel(time=idxs + self.lead_time).values)

        return X, y
    
class WeatherBenchDataset1dNumpy(Dataset):
    
    def __init__(self, data_path, lead_time, lat, lon, var_dict, valid_time, mean=None, std=None):
        
        self.lead_time = lead_time
        self.lat = lat
        self.lon = lon
        self.var_dict = var_dict
        self.valid_time = valid_time
    
        self.features = len(self.var_dict)
        self.latitudes = len(self.lat)
        self.longitudes = len(self.lon)
        
        self.mean = np.load(data_path + 'mean.npy') if mean is None else mean
        self.std = np.load(data_path + 'std.npy') if std is None else std

        
        total_samples = len(os.listdir(data_path)) - 2
        self.n_samples = total_samples - self.lead_time
        
        self.datafiles = [(data_path+str(id)+'.npy', data_path+str(id+lead_time)+'.npy')
                           for id in list(range(self.n_samples))]
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """ Returns sample and label corresponding to an index as torch.Tensor objects
            The return tensor shapes are (for the sample and the label): [n_vertex, n_features]
        """
        
        '''X = self.transform(np.load(self.datafiles[idx]))
        y = self.transform(np.load(self.datafiles[idx+delta_t]))'''
        
        X = torch.Tensor((np.load(self.datafiles[idx][0])-self.mean)/self.std)
        y = torch.Tensor((np.load(self.datafiles[idx][1])-self.mean)/self.std)
        
        return X, y


class WeatherBenchDataset1dXarray(Dataset):
    
    def __init__(self, ds, var_dict, lead_time, load=True, mean=None, std=None):
        
        self.ds = ds
        self.var_dict = var_dict
        self.lead_time = lead_time

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
        self.init_time = self.data.isel(time=slice(None, -lead_time)).time
        self.valid_time = self.data.isel(time=slice(lead_time, None)).time
        self.idxs = np.arange(self.n_samples)
        
        self.lat = self.data.lat
        self.lon = self.data.lon
        
        self.features = len(self.data.level)
        self.latitudes = len(self.lat)
        self.longitudes = len(self.lon)
        
        # Stack
        self.data = self.data.stack(nodes=('lat', 'lon')).transpose('time', 'nodes', 'level')
        
        # For some weird reason calling .load() earlier messes up the mean and std computations
        if load: 
            print('Loading data into RAM')
            self.data.load()
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        idxs = self.idxs[idx]
        X = torch.Tensor(self.data.isel(time=idxs).values)
        y = torch.Tensor(self.data.isel(time=idxs + self.lead_time).values)

        return X, y

    
class WeatherBenchDataset2d(Dataset):
    
    def __init__(self, ds, var_dict, lead_time, load=True, mean=None, std=None):

        self.ds = ds
        self.var_dict = var_dict
        self.lead_time = lead_time

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
        self.init_time = self.data.isel(time=slice(None, -lead_time)).time
        self.valid_time = self.data.isel(time=slice(lead_time, None)).time
        self.idxs = np.arange(self.n_samples)

        # For some weird reason calling .load() earlier messes up the mean and std computations
        if load: print('Loading data into RAM'); self.data.load()
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        idxs = self.idxs[idx]
        X = self.data.isel(time=idxs).values
        y = self.data.isel(time=idxs + self.lead_time).values
        
        X = torch.Tensor(X).permute(2, 0, 1)
        y = torch.Tensor(y).permute(2, 0, 1)

        return X, y