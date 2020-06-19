import xarray as xr
import numpy as np
import healpy as hp
import os
from pathlib import Path
from scipy import interpolate

import torch
from torch import nn, optim 
from torch.utils.data import Dataset


# Data preprocessing
def preprocess_equiangular(in_data, out_data, train_years, val_years, test_years):
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
    
    z = xr.open_mfdataset(in_data + 'geopotential_500/*.nc', combine='by_coords')['z'].assign_coords(level=1)
    t = xr.open_mfdataset(in_data + 'temperature_850/*.nc', combine='by_coords')['t'].assign_coords(level=1)
    data = xr.concat([z, t], 'level').stack(v=('lat', 'lon')).transpose('time', 'v', 'level').drop('level')
   
    for set_name in ['train']:
    
        # Create directory
        out_path =  out_data + set_name + "/"
        Path(out_path).mkdir(parents=True, exist_ok=True)
        
        # Select relevant years
        dataset = data.sel(time=time_slices[set_name])

        # Compute mean and std
        mean = data.mean().compute()
        std = data.std('time').mean().compute()
        np.save(out_path + 'mean.npy', mean.values)
        np.save(out_path + 'std.npy', std.values)
    
        # Save individual arrays
        for i, array in enumerate(dataset):
            np.save(out_path + str(i) + '.npy', array.values)
            
            
def preprocess_healpix(in_data, out_data, train_years, val_years, test_years, nside, interpolation_kind='linear'):
    """ Splits data into train, validation and test sets, computes and saves the mean and standard deviation, normalizes the data, regrids all samples from a dataset from equiangular to HEALpix grid and saves each sample as a numpy array.
    
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
    nside : int
        Number of sides dividing the HEALpix cells
    interpolaton_kind : string
        Interpolation method. Options are {‘linear’, ‘cubic’, ‘quintic’}
        
    Returns 
    -------
    hp_ds : xr.Dataset of dimensions time x num_vars x n_pixels
        Regridded dataset
    """
    
    time_slices = {'train': train_years, 'val': val_years, 'test': test_years}
    
    z = xr.open_mfdataset(in_data + 'geopotential_500/*.nc', combine='by_coords')['z'].assign_coords(level=1)
    t = xr.open_mfdataset(in_data + 'temperature_850/*.nc', combine='by_coords')['t'].assign_coords(level=1)
    data = xr.concat([z, t], 'level').drop('level')
    
    # Input grid
    lat = data.lat.values
    lon = data.lon.values
    
    # Output grid
    n_pixels = hp.nside2npix(nside)
    out_lon, out_lat = hp.pix2ang(nside, np.arange(n_pixels), lonlat=True, nest=True)

    for set_name in ['train', 'val', 'test']:
    
        # Create directory
        out_path = out_data + 'healpix/' + set_name + "/"
        Path(out_path).mkdir(parents=True, exist_ok=True)
        
        # Select relevant years
        dataset = data.sel(time=time_slices[set_name])

        # Compute mean and std
        mean = data.mean(('time', 'lat', 'lon')).compute()
        std = data.std('time').mean(('lat', 'lon')).compute()
        np.save(out_path + 'mean.npy', mean.values)
        np.save(out_path + 'std.npy', std.values)

        # Save individual arrays
        for t in range(dataset.shape[1]):
            sample = dataset.isel(time=t)
            hp_sample = np.empty((n_pixels, sample.shape[0]))
            for i, signal in enumerate(sample.values):
                f = interpolate.interp2d(lat, lon, np.rot90(signal), kind=interpolation_kind)
                hp_sample[:, i] = np.array([f(lat, lon) for lat, lon in zip(out_lat, out_lon)]).flatten()
            
            np.save(out_path + str(t) + '.npy', hp_sample)

# Interpolation
def hp_to_equiangular(sample, res, method='linear'):
    
    # Input grid
    n_pixels = sample.dims['node']
    nside = int(np.sqrt(n_pixels/12))
    hp_lon, hp_lat = hp.pix2ang(nside, np.arange(n_pixels), lonlat=True, nest=True)
    points = np.array([hp_lon, hp_lat]).transpose((1, 0))
    
    # Output grid
    equi_lat = np.arange(-90 + res/2, 90, res)
    equi_lon = np.arange(0, 360, res)
    grid_x, grid_y = np.meshgrid(equi_lon, equi_lat)
    
    eq_lon = grid_x[0, :]
    eq_lat = grid_y[:, 0]
    
    interpolated = []
    for var in sample.data_vars:
        signal = sample[var]
        grid_z = interpolate.griddata(points, signal.values, (grid_x, grid_y), method=method)


        nans = np.isnan(grid_z)
        notnans = np.logical_not(nans)
        grid_z[nans] = interpolate.griddata((grid_x[notnans], grid_y[notnans]), grid_z[notnans], 
                                            (grid_x[nans], grid_y[nans]), method='nearest')

        interpolated.append(xr.DataArray(
                grid_z, 
                dims=['lat', 'lon'],
                coords={'lat': eq_lat, 'lon': eq_lon},
                name=var
            ))     
    interpolated = xr.merge(interpolated)
    
    return interpolated

# Datasets
class WeatherBenchDatasetXarrayHealpix(Dataset):
    """ Dataset used for graph models (1D), where data is loaded from an xarray Dataset
    
    Parameters
    ----------
    ds : xarray Dataset
        Dataset containing the input data
    lead_time : int
        Prediction interval (in hours)
    out_features : int
        Number of output features
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
    
    def __init__(self, ds, out_features, lead_time, years, nodes, nb_timesteps, max_lead_time=None, 
                 load=True, mean=None, std=None):
        
        self.lead_time = lead_time
        self.years = years
        self.nodes = nodes
        self.out_features = out_features
        self.max_lead_time = max_lead_time
        self.nb_timesteps = nb_timesteps
        
        self.data = ds.to_array(dim='level', name='Dataset').transpose('time', 'node', 'level')
        self.in_features = self.data.shape[-1]
        
        self.mean = self.data.mean(('time', 'node')).compute() if mean is None else mean
        self.std = self.data.std(('time', 'node')).compute() if std is None else std
        
        # Normalize
        self.data = (self.data - self.mean) / self.std
        if max_lead_time is None:
            self.n_samples = self.data.isel(time=slice(0, -self.nb_timesteps*lead_time)).shape[0]
        else:
            self.n_samples = self.data.isel(time=slice(0, -self.nb_timesteps*lead_time)).shape[0] - max_lead_time
        self.idxs = np.arange(self.n_samples)
        
        if load: 
            print('Loading data into RAM')
            self.data.load()
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        idxs = self.idxs[idx]
        
        X = (torch.Tensor(self.data.isel(time=idxs).values),
             torch.Tensor(self.data.isel(time=idxs + self.lead_time).values[:, self.out_features:]))
        
        y = (torch.Tensor(self.data.isel(time=idxs + self.lead_time).values[:, :self.out_features]), 
             torch.Tensor(self.data.isel(time=idxs + 2*self.lead_time).values[:, :self.out_features]))

        return X, y
    
    
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
                 max_lead_time=None, load=True, mean=None, std=None):
        
        
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
        
        # Normalize
        self.data = (self.data - self.mean) / self.std
        
        # Count total number of samples
        total_samples = self.data.shape[0]        
        if max_lead_time is None:
            self.n_samples = total_samples - (len_sqce+1) * delta_t
        else:
            self.n_samples = total_samples - (len_sqce+1) * delta_t - max_lead_time
            
        
        # Create indexes
        self.idxs = [[[[sample_idx + delta_t*k for k in range(len_sqce)], sample_idx + delta_t * len_sqce], 
                      [sample_idx + delta_t * len_sqce, sample_idx + delta_t * (len_sqce+1)]] 
                     for sample_idx in range(self.n_samples)]
        
        if load: 
            print('Loading data into RAM')
            self.data.load()
            
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """ Returns sample and label corresponding to an index as torch.Tensor objects
            The return tensor shapes are (for the sample and the label): [n_vertex, len_sqce, n_features]
        """
        
        X = (torch.tensor(self.data.isel(time=self.idxs[idx][0][0]).values).float().permute(1, 0, 2), 
             torch.tensor(self.data.isel(time=self.idxs[idx][0][1]).values[:, self.out_features:]).float())
        
        y = (torch.Tensor(self.data.isel(time=self.idxs[idx][1][0]).values[:, :self.out_features]).float(), 
             torch.Tensor(self.data.isel(time=self.idxs[idx][1][1]).values[:, :self.out_features]).float())
        
        return X, y 
    
    
class WeatherBenchDatasetIterative(Dataset):
    """ Dataset used for graph models (1D), where data is loaded from stored numpy arrays.
    
    Parameters
    ----------
    """
    
    def __init__(self, data):
        
        self.data = data
        self.n_samples = data.shape[0]
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """ Returns sample and label corresponding to an index as torch.Tensor objects
            The return tensor shapes are (for the sample and the label): [n_vertex, n_features]
        """
        return [torch.Tensor(self.data[idx, :, :])], 0

    
class WeatherBenchDataset1dNumpy(Dataset):
    """ Dataset used for graph models (1D), where data is loaded from stored numpy arrays.
    
    Parameters
    ----------
    data_path : str
        Path to data folder
    lead_time : int
        Prediction interval (in hours)
    var_dict : dict
        Dictionary where the keys are the relevant variables and the values are pressure levels at which the variables are considered 
    years : tuple(str)
        Years used to split the data
    res : float
        Spatial resolution
    mean : np.ndarray of shape 2
        Mean to use for data normalization. If None, mean is computed from data
    std : np.ndarray of shape 2
        std to use for data normalization. If None, mean is computed from data
    """
    
    def __init__(self, data_path, lead_time, var_dict, years, res, mean=None, std=None):
        
        self.lead_time = lead_time
        self.years = years
        self.res = res
        self.var_dict = var_dict
    
        self.features = len(self.var_dict)
        
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
    var_dict_in : dict
        Dictionary where the keys are the input variables and the values are pressure levels at which the variables 
        are considered
    var_dict_out : dict
        Dictionary where the keys are the output variables and the values are pressure levels at which the variables 
        are considered
    years : tuple(str)
        Years used to split the data
    res : float
        Spatial resolution
    load : bool
        If true, load dataset to RAM
    mean : np.ndarray of shape 2
        Mean to use for data normalization. If None, mean is computed from data
    std : np.ndarray of shape 2
        std to use for data normalization. If None, mean is computed from data
    """
    
    def __init__(self, ds, var_dict_in, var_dict_out, lead_time, years, res, load=True, mean=None, std=None):
        
        self.ds = ds
        self.var_dict_in = var_dict_in
        self.var_dict_out = var_dict_out
        self.lead_time = lead_time
        self.years = years
        self.res = res
        self.out_features = len(var_dict_out)
        self.in_features = len(var_dict_in)

        data = []
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
        for var, levels in var_dict_in.items():
            try:
                data.append(ds[var].sel(level=levels))
            except ValueError:
                data.append(ds[var].expand_dims({'level': generic_level}, 1))

        self.data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')
        self.mean = self.data.mean(('time', 'lat', 'lon')).compute() if mean is None else mean
        
        if std is None:
            
            const_std = self.data.std(('time', 'lat', 'lon')).compute()
            std_ = self.data.std(('time')).mean(('lat', 'lon')).compute()
            std_[std_ == 0] = const_std[std_==0]
            self.std = std_
        else:
            self.std = std
        
        # Normalize
        self.data = (self.data - self.mean) / self.std
        self.n_samples = self.data.isel(time=slice(0, -lead_time)).shape[0]
        self.idxs = np.arange(self.n_samples)
        
        # Stack
        self.data = self.data.stack(nodes=('lat', 'lon')).transpose('time', 'nodes', 'level')
        
        if load: 
            print('Loading data into RAM')
            self.data.load()
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        idxs = self.idxs[idx]
        X = torch.Tensor(self.data.isel(time=idxs).values)
        y = torch.Tensor(self.data.isel(time=idxs + self.lead_time).values[:, :self.out_features])

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