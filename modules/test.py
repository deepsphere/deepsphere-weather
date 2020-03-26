import xarray as xr
import numpy as np
import datetime


# Predict
def create_predictions(model, device, dg, mean, std):
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
    
    output_dim = (dg.dataset.latitudes, dg.dataset.longitudes, dg.dataset.features)
    outputs = []
    
    for i, (sample, _) in enumerate(dg):
        sample = sample.to(device)
        output = model(sample).detach().cpu().clone().numpy().reshape((-1, *output_dim))
        outputs.append(output)
    preds = np.concatenate(outputs)
    
    # Unnormalize
    preds = preds * std + mean
    das = []
    lev_idx = 0
    for var, levels in dg.dataset.var_dict.items():
        if levels is None:
            das.append(xr.DataArray(
                preds[:, :, :, lev_idx],
                dims=['time', 'lat', 'lon'],
                coords={'time': dg.dataset.valid_time, 'lat': dg.dataset.lat, 'lon': dg.dataset.lon},
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

def create_predictions_2D(model, dg, mean, std):
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


# Metrics
def compute_weighted_rmse(da_fc, da_true, dims=xr.ALL_DIMS):
    """ Compute the room mean squared error (RMSE) with latitude weighting from two xr.DataArrays.
    
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
    rmse : xr.DataArray
        Latitude weighted root mean squared error 
    """
    error = da_fc - da_true
    weights_lat = np.cos(np.deg2rad(error.lat))
    weights_lat /= weights_lat.mean()
    rmse = np.sqrt(((error)**2 * weights_lat).mean(dims))
    return rmse


def compute_relBIAS(da_fc, da_true, dims=xr.ALL_DIMS):
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
    assert dims in [xr.ALL_DIMS, 'time'], "Relative std must be computed either over all dimensions or only over time"
    
    error = da_fc - da_true
    rbias = error.mean(dims) / da_true.mean(dims)
    return rbias

def compute_relMSE(da_fc, da_true, dims=xr.ALL_DIMS):
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


def compute_relMAE(da_fc, da_true, dims=xr.ALL_DIMS):
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


def compute_rSD(da_fc, da_true, dims=xr.ALL_DIMS):
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


def compute_temporal_correlation(da_fc, da_true, dims='time'):
    """ Compute the Pearson correlation coefficient from two xr.DataArrays given some dimensions
    
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
    
    def _inner(x, y):
        result = np.matmul(x[..., np.newaxis, :], y[..., :, np.newaxis])
        return result[..., 0, 0]

    def inner_product(x, y, dim):
        return xr.apply_ufunc(_inner, x, y, input_core_dims=[[dim], [dim]])
    
    def covariance(x, y, dims=None):
        return inner_product(x - x.mean(dims), y - y.mean(dims), dim=dims) / x.count(dims)
    
    x = da_fc.load()
    y = da_true.load()

    return (covariance(x, y, dims) / (x.std(dims) * y.std(dims)))**2


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
    
    cc = compute_temporal_correlation(da_fc, da_true, dims=dims)
    alpha = compute_rSD(da_fc, da_true, dims=dims)
    beta = da_fc.sum(dims) / da_true.sum(dims)
    kge = 1 - np.sqrt((cc - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    return kge