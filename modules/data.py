import xarray as xr
import numpy as np
import os

import torch
from torch import nn, optim 
from torch.utils.data import Dataset


# Data preprocessing
def preprocess_data(datadir, train_years, val_years, test_years):
    """ Splits data into train, validation and test sets, computes and saves the mean and standard deviation, normalizes the data, reshapes it to make 1D and saves each sample as a numpy array.
    
    Parameters
    ----------
    datadir : string
        Path to data
    train_years : slice(str)
        Years used to select the training set
    val_years : slice(str)
        Years used to select the validation set
    test_years : slice(str)
        Years used to select the test set
    """
    
    time_slices = {'train': train_years, 'val': val_years, 'test': test_years}
    
    
    zpath = datadir + 'geopotential_500/'
    tpath = datadir + 'temperature_850/'
    
    z = xr.open_mfdataset(zpath+'/*.nc', combine='by_coords')['z'].assign_coords(level=1)
    t = xr.open_mfdataset(tpath+'/*.nc', combine='by_coords')['t'].assign_coords(level=1)

    ratio = len(z.coords['lon'])/len(z.coords['lat'])

    data = xr.concat([z, t], 'level').stack(v=('lat', 'lon')).transpose('time', 'v', 'level').drop('level')
    
    data_paths = []
    for set_name in ['train']:
    
        # Create directory
        out_path = DATA_DIR + set_name + "/"
        Path(out_path).mkdir(parents=True, exist_ok=True)
        data_paths.append(out_path)
        
        # Select relevant years
        dataset = data.sel(time=time_slices[set_name])

        # Compute mean and std
        mean = data.mean(('time', 'v')).compute()
        std = data.std('time').mean(('v')).compute()
        np.save(out_path + 'mean.npy', mean.values)
        np.save(out_path + 'std.npy', std.values)
    
        # Save individual arrays
        for i, array in enumerate(dataset):
            np.save(out_path + str(i) + '.npy', array.values)



# Datasets
class WeatherBenchDataset1dNumpy(Dataset):
    """ Dataset used for graph models (1D), where data is loaded from stored numpy arrays.
    
    Parameters
    ----------
    data_path : str
        Path to data folder
    lead_time : int
        Prediction interval (in hours)
    lat :  
    lon : 
    var_dict : dict
        Dictionary where the keys are the relevant variables and the values are pressure levels at which the variables are considered 
    valid_time :  
    mean : np.ndarray of shape 2
        Mean to use for data normalization. If None, mean is computed from data
    std : np.ndarray of shape 2
        std to use for data normalization. If None, mean is computed from data
    """
    
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
    """ Dataset used for graph models (1D), where data is loaded from an xarray Dataset
    
    Parameters
    ----------
    ds : xarray Dataset
        Dataset containing the input data
    lead_time : int
        Prediction interval (in hours)
    var_dict : dict
        Dictionary where the keys are the relevant variables and the values are pressure levels at which the variables are
        considered 
    load : bool
        If true, load dataset to RAM
    mean : np.ndarray of shape 2
        Mean to use for data normalization. If None, mean is computed from data
    std : np.ndarray of shape 2
        std to use for data normalization. If None, mean is computed from data
    """
    
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
    """ Dataset used for image models (2D), where data is loaded from an xarray Dataset
    
    Parameters
    ----------
    ds : xarray Dataset
        Dataset containing the input data
    lead_time : int
        Prediction interval (in hours)
    var_dict : dict
        Dictionary where the keys are the relevant variables and the values are pressure levels at which the variables are considered
    load : bool
        If true, load dataset to RAM
    mean : np.ndarray of shape 2
        Mean to use for data normalization. If None, mean is computed from data
    std : np.ndarray of shape 2
        std to use for data normalization. If None, mean is computed from data
    """
    
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