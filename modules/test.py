import xarray as xr
import numpy as np
import datetime
import healpy as hp
import time
from torch.utils.data import Dataset, DataLoader

from modules.data import WeatherBenchDatasetIterative

# Utils
def _inner(x, y):
        result = np.matmul(x[..., np.newaxis, :], y[..., :, np.newaxis])
        return result[..., 0, 0]

def inner_product(x, y, dim):
    return xr.apply_ufunc(_inner, x, y, input_core_dims=[[dim], [dim]])


# generate observation files
def create_iterative_observations_healpix(ds, lead_time, max_lead_time, nb_timesteps, test_years, nodes):
    """
    Assumptions
    lead_time : min time to next prediction (6h)
    max_lead_time: lastest time of predictions (72h)
    nb_timesteps: 
    nodes --> number of nodes? 
    """
    
    lead_times = np.arange(lead_time, max_lead_time + lead_time, lead_time)

    data = ds.to_array(dim='level', name='Dataset').transpose('time', 'node', 'level')
    n_samples = data.isel(time=slice(0, -nb_timesteps*lead_time)).shape[0] - max_lead_time

    obs_list = []

    for lead in lead_times:
        obs_list.append(data.isel(time=slice(lead, lead + n_samples)).isel(level=slice(0, 2)).values)

    observations_numpy = np.array(obs_list)

    # Lat lon coordinates
    nside = int(np.sqrt(nodes/12))
    out_lon, out_lat = hp.pix2ang(nside, np.arange(nodes), lonlat=True)

    # Actual times
    start = np.datetime64(test_years[0], 'h') + np.timedelta64(lead_time, 'h')
    stop = start + np.timedelta64(n_samples, 'h')
    times = np.arange(start, stop)

    # Variables
    var_dict_out = {var: None for var in ['z', 't']}

    das = [];
    lev_idx = 0
    for var, levels in var_dict_out.items():
        if levels is None:            
            das.append(xr.DataArray(
                observations_numpy[:, :, :, lev_idx],
                dims=['lead_time', 'time', 'node'],
                coords={'lead_time': lead_times, 'time': times, 'node': np.arange(nodes)},
                name=var
            ))
            lev_idx += 1

        else:
            nlevs = len(levels)
            das.append(xr.DataArray(
                observations_numpy[:, :, :, lev_idx:lev_idx+nlevs],
                dims=['lead_time', 'time', 'node', 'level'],
                coords={'lead_time': lead_times, 'time': valid_time, 'node': nodes, 'level': nlevs},
                name=var
            ))
            lev_idx += nlevs
    observation_ds = xr.merge(das)
    observation_ds = observation_ds.assign_coords({'lat': out_lat, 'lon': out_lon})
    return observation_ds


# Predict
def create_iterative_predictions_healpix(model, device, dg):
    batch_size = dg.batch_size
    
    delta_t = dg.dataset.lead_time
    max_lead_time = dg.dataset.max_lead_time
    initial_lead_time = delta_t
    nodes = dg.dataset.nodes
    nside = int(np.sqrt(nodes/12))
    n_samples = dg.dataset.n_samples
    in_feat = dg.dataset.in_features
    out_feat = dg.dataset.out_features
    data_vars = dg.dataset.mean.level.values.tolist()[:out_feat]
    
    train_std =  dg.dataset.std.values[:out_feat]
    train_mean = dg.dataset.mean.values[:out_feat]
    
    # Lead times
    lead_times = np.arange(delta_t, max_lead_time + delta_t, delta_t)
    
    # Lat lon coordinates
    out_lon, out_lat = hp.pix2ang(nside, np.arange(nodes), lonlat=True)
    
    # Actual times
    start = np.datetime64(dg.dataset.years[0], 'h') + np.timedelta64(initial_lead_time, 'h')
    stop = start + np.timedelta64(dg.dataset.n_samples, 'h')
    times = np.arange(start, stop)
    
    # Variables
    var_dict_out = {var: None for var in data_vars}
    
    # Radiation
    constants = np.array(dg.dataset.data.isel(level=slice(out_feat, None)).values)
    
    dataloader = dg
    predictions = []
    model.eval()
    for lead in lead_times:
        outputs = []
        state = []
        states = np.empty((n_samples, nodes, in_feat))
        
        time1 = time.time()

        for i, (sample, _) in enumerate(dataloader):
            inputs = sample[0].to(device)
            output = model(inputs)

            outputs.append(output.detach().cpu().clone().numpy()[:, :, :out_feat])
            state.append(output.detach().cpu().clone().numpy())
            
        preds = np.concatenate(outputs)
        states[:, :, :out_feat] = np.concatenate(state)
        states[:, :, out_feat:] = constants[lead:n_samples+lead, :]

        predictions.append(preds * train_std + train_mean)

        new_set = WeatherBenchDatasetIterative(states)
        dataloader = DataLoader(new_set, batch_size=batch_size, shuffle=False, num_workers=10)
        
        time2 = time.time()
        
    predictions = np.array(predictions)
    
    das = [];
    lev_idx = 0
    for var in data_vars:       
        das.append(xr.DataArray(
            predictions[:, :, :, lev_idx],
            dims=['lead_time', 'time', 'node'],
            coords={'lead_time': lead_times, 'time': times, 'node': np.arange(nodes)},
            name=var
        ))
        lev_idx += 1
            
    prediction_ds = xr.merge(das)
    prediction_ds = prediction_ds.assign_coords({'lat': out_lat, 'lon': out_lon})
    return prediction_ds


def create_predictions(model, device, dg):
    """Create direct predictions for models using 1D signals (eg GCNN)
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained model
    device
        GPU / CPU where model is running
    dg : DataLoader
        Test data
        
    Returns
    -------
    predictions : xr.Dataset
        Model predictions
    """
    
    lats = np.arange(-90+dg.dataset.res/2, 90+dg.dataset.res/2, dg.dataset.res)
    lons = np.arange(0, 360, dg.dataset.res)
    lat = xr.DataArray(lats, coords=[lats], dims=['lat'], name='lat')
    lon = xr.DataArray(lons, coords=[lons], dims=['lon'], name='lon')

    start = np.datetime64(dg.dataset.years[0], 'h') + np.timedelta64(dg.dataset.lead_time, 'h')
    stop = np.datetime64(str(int(dg.dataset.years[1])+1), 'h')
    times = np.arange(start, stop)
    valid_time = xr.DataArray(times, coords=[times], dims='time', name='time', attrs={'long_name': 'time'})
    
    out_features = dg.dataset.out_features
    
    output_dim = (len(lats), len(lons), dg.dataset.out_features)
    outputs = []
    
    for i, (sample, _) in enumerate(dg):
        sample = sample.to(device)
        output = model(sample).detach().cpu().clone().numpy().reshape((-1, *output_dim))
        outputs.append(output)
    preds = np.concatenate(outputs)
    
    # Unnormalize
    preds = preds * dg.dataset.std.values[:out_features] + dg.dataset.mean.values[:out_features]
    das = []
    lev_idx = 0
    for var, levels in dg.dataset.var_dict_out.items():
        if levels is None:
            das.append(xr.DataArray(
                preds[:, :, :, lev_idx],
                dims=['time', 'lat', 'lon'],
                coords={'time': valid_time, 'lat': lat, 'lon': lon},
                name=var
            ))
            lev_idx += 1
        else:
            nlevs = len(levels)
            das.append(xr.DataArray(
                preds[:, :, :, lev_idx:lev_idx+nlevs],
                dims=['time', 'lat', 'lon', 'level'],
                coords={'time': valid_time, 'lat': lat, 'lon': lon, 'level': levels},
                name=var
            ))
            lev_idx += nlevs
    return xr.merge(das)

def create_predictions_2D(model, dg, mean, std, device):
    """Create direct predictions for models using 2D signals (images)
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained model
    dg : DataLoader
        Test data
    mean : np.ndarray 
        Training set mean
    std : np.ndarray 
        Training set std
        
    Returns
    -------
    predictions : xr.Dataset
        Model predictions
    """
    
    outputs = []
    for i, (sample, _) in enumerate(dg):
        sample = sample.to(device)
        output = model(sample).detach().cpu().clone().permute(0, 2, 3, 1).numpy()
        outputs.append(output)
    preds = np.concatenate(outputs)
    
    # Unnormalize
    preds = preds * std.values + mean.values
    das = []
    lev_idx = 0
    for var, levels in dg.dataset.var_dict.items():
        if levels is None:
            das.append(xr.DataArray(
                preds[:, :, :, lev_idx],
                dims=['time', 'lat', 'lon'],
                coords={'time': dg.dataset.valid_time, 'lat': dg.dataset.data.lat, 'lon': dg.dataset.data.lon},
                name=var
            ))
            lev_idx += 1
        else:
            nlevs = len(levels)
            das.append(xr.DataArray(
                preds[:, :, :, lev_idx:lev_idx+nlevs],
                dims=['time', 'lat', 'lon', 'level'],
                coords={'time': dg.dataset.valid_time, 'lat': dg.dataset.data.lat, 'lon': dg.dataset.data.lon, 'level': levels},
                name=var
            ))
            lev_idx += nlevs
    return xr.merge(das)


def compute_anomalies(ds, mean):
    """ Computes anomalies by removing relevant average to data

    Parameters
    ----------
    ds : xr.Dataset
        Dataset from whoch to compute the anomalies
    mean : string
        Which mean to remove to the data. Options are {"monthly", "weekly"}
    
    Returns
    -------
    anomalies : xr.Dataset
        Demeaned dataset 
    """

    assert mean in ["monthly", "weekly"], "Parameter mean should be either 'monthly' or 'weekly'"
    
    if mean is "monthly":
        anomalies = ds.groupby('time.month') - ds.groupby('time.month').mean()
    else: 
        anomalies = ds.groupby('time.week') - ds.groupby('time.week').mean()
        
    return anomalies

# Metrics for healpix iterative predictions
def compute_rmse_healpix(pred, obs, dims=('node', 'time')):
    error = pred - obs
    
    rmse = np.sqrt(((error)**2).mean(dims))
    return rmse.drop('lat').drop('lon').load()


def compute_relBIAS_map_healpix(pred, obs):
    """ Compute the relative bias from two xr.DataArrays given some dimensions
    
    Parameters
    ----------
    pred : xr.DataArray
        Forecast. Time coordinate must be validation time.
    obs_mean : xr.DataArray
        Mean of observations across "time" and "lead_time" dimensions
    
    Returns
    -------
    relBIAS : xr.DataArray
        Relative bias map
    """
    error = pred - obs
    rbias = error.mean(('time')).load() / obs.mean(('time')).load()
    return rbias


def compute_weighted_rmse(da_fc, da_true, dims=xr.ALL_DIMS):
    """ Compute the root mean squared error (RMSE) with latitude weighting from two xr.DataArrays.
    
    Parameters
    ----------
    da_fc : xr.DataArray
        Forecast
    da_true : xr.DataArray
        Labels
    dims (str): 
        Dimensions over which to compute the metric
    
    Returns
    -------
    rmse : xr.DataArray
        Latitude weighted root mean squared error 
    """
    error = da_fc - da_true
    weights_lat = np.cos(np.deg2rad(error.lat))
    weights_lat /= weights_lat.mean()
    rmse = np.sqrt(((error)**2 * weights_lat).mean(dims))
    return rmse


def compute_rmse_equiangular(da_fc, da_true):
    """ Compute the root mean squared error (RMSE) from two xr.DataArrays with equiangular sampling 
    where each pixel is weighted by the proportion of spherical area it represents.
    
    Parameters
    ----------
    da_fc : xr.DataArray
        Forecast
    da_true : xr.DataArray
        Labels
    
    Returns
    -------
    rmse : xr.DataArray
        Latitude weighted root mean squared error 
    """
    error = da_fc - da_true
    
    resolution = pred.lon.values[1] - pred.lon.values[0]
    delta_lon = np.deg2rad(resolution)
    n_samples = len(da_fc.time)
    
    weights_lat = np.cos(np.deg2rad(pred.lat - resolution/2 + 90)) - np.cos(np.deg2rad(pred.lat + resolution/2 + 90))
    rmse = np.sqrt(((error)**2 * weights_lat * delta_lon /(4*np.pi)).mean('time').sum())
    return rmse


def compute_relBIAS(da_fc, da_true, dims='time'):
    """ Compute the relative bias from two xr.DataArrays given some dimensions
    
    Parameters
    ----------
    da_fc : xr.DataArray
        Forecast. Time coordinate must be validation time.
    da_true : xr.DataArray
        Labels
    dims (str): 
        Dimensions over which to compute the metric
    
    Returns
    -------
    relBIAS : xr.DataArray
        Relative bias
    """
    error = da_fc - da_true
    rbias = error.mean(dims) / da_true.mean(dims)
    return rbias

def compute_relMSE(da_fc, da_true, dims='time'):
    """ Compute the relative mean squared error (MSE) from two xr.DataArrays given some dimensions
    
    Parameters
    ----------
    da_fc : xr.DataArray
        Forecast. Time coordinate must be validation time.
    da_true : xr.DataArray
        Labels
    dims (str): 
        Dimensions over which to compute the metric
    
    Returns
    -------
    relMSE : xr.DataArray
        Relative MSE
    """
    assert dims in [xr.ALL_DIMS, 'time'], "Relative std must be computed either over all dimensions or only over time"
    
    error = da_fc - da_true
    rel_mse = (error**2).mean(dims) / (da_true**2).mean(dims)
    return rel_mse


def compute_MAE(da_fc, da_true, dims=('node', 'time')):
    error = da_fc - da_true
    mae = (abs(error)).mean(dims).load()
    return mae


def compute_relMAE(da_fc, da_true, dims='time'):
    """ Compute the relative mean absolute error (MAE) from two xr.DataArrays given some dimensions
    
    Parameters
    ----------
    da_fc : xr.DataArray
        Forecast. Time coordinate must be validation time.
    da_true : xr.DataArray
        Labels
    dims (str): 
        Dimensions over which to compute the metric
    
    Returns
    -------
    relMAE: xr.DataArray
        Relative MAE
    """
    assert dims in [xr.ALL_DIMS, 'time'], "Relative std must be computed either over all dimensions or only over time"
    
    error = da_fc - da_true
    rel_mae = (abs(error)).mean(dims) / (abs(da_true)).mean(dims)
    return rel_mae


def compute_rSD(da_fc, da_true, dims='time'):
    """ Compute the ratio of standard deviations from two xr.DataArrays given some dimensions
    
    Parameters
    ----------
    da_fc : xr.DataArray
        Forecast. Time coordinate must be validation time.
    da_true : xr.DataArray
        Labels
    dims (str): 
        Dimensions over which to compute the metric
    
    Returns
    -------
    rSD : xr.DataArray
        Ratio of stds
    """
    assert dims in [xr.ALL_DIMS, 'time'], "Relative std must be computed either over all dimensions or only over time"
    
    rsd = da_fc.std(dims) / da_true.std(dims)
    return rsd


def compute_R2(da_fc, da_true, dims='time'):
    """ Compute the squared Pearson correlation coefficient from two xr.DataArrays given some dimensions
    
    Parameters
    ----------
    da_fc : xr.DataArray
        Forecast. Time coordinate must be validation time.
    da_true : xr.DataArray
        Labels
    dims (str): 
        Dimensions over which to compute the metric
    
    Returns
    -------
    corr : xr.DataArray
        Pearson correlation coefficients
    """
    
    def covariance(x, y, dims=None):
        return inner_product(x - x.mean(dims), y - y.mean(dims), dim=dims) / x.count(dims)
    
    x = da_fc.load()
    y = da_true.load()

    return (covariance(x, y, dims) / (x.std(dims) * y.std(dims)))**2

def compute_ACC(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    """
    Compute the anomaly correlation coefficient from two xr.DataArrays
    WARNING: Does not work if datasets contain NaNs

    Parameters
    ----------
    da_fc : xr.DataArray
        Forecast. Time coordinate must be validation time.
    da_true : xr.DataArray
        Labels
    dims (str): 
        Dimensions over which to compute the metric
    
    Returns
    -------
    corr : xr.DataArray
        Anomaly correlation coefficient
    """

    clim = da_true.mean('time')
    try:
        t = np.intersect1d(da_fc.time, da_true.time)
        fa = da_fc.sel(time=t) - clim
        
    except AttributeError:
        t = da_true.time.values
        fa = da_fc - clim
    a = da_true.sel(time=t) - clim

    fa_prime = fa - fa.mean()
    a_prime = a - a.mean()
    
    acc = (np.sum(fa_prime * a_prime, axis=(1, 2)) 
           / np.sqrt(np.sum(fa_prime ** 2, axis=(1, 2)) * np.sum(a_prime ** 2, axis=(1, 2))))
    return acc


def compute_KGE(da_fc, da_true):
    """ Compute the Kling-Gupta Efficiency (KGE)
    
    Parameters
    ----------
    da_fc : xr.DataArray
        Forecast. Time coordinate must be validation time.
    da_true : xr.DataArray
        Labels
    
    Returns
    -------
    kge : xr.DataArray
        KGE
        
    References
    ----------
    Gupta, Kling, Yilmaz, Martinez, 2009, Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling
    """
        
    dims="time"
    
    cc = compute_R2(da_fc, da_true, dims=dims)
    alpha = compute_rSD(da_fc, da_true, dims=dims)
    beta = da_fc.sum(dims) / da_true.sum(dims)
    kge = 1 - np.sqrt((cc - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    return kge