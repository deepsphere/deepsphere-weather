import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy

from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh


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
    rmse = np.sqrt(((error)**2 * weights_lat).mean(mean_dims))
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

def assess_model(pred, valid, path, model_description):
    """ Assess predictions comparing them to label data using several metrics
    
    Parameters
    ----------
    pred : xr.DataArray
        Forecast. Time coordinate must be validation time.
    valid : xr.DataArray
        Labels
    path : str 
        Path to which the evaluation is saved as .pdf
    model_description : str
        Plot title and filename, should distinguishly describe the model to assess
    
    Returns
    -------
    plt.plot
        Several plots showing the predictions' rightness
    """
    
    lats = pred.variables['lat'][:]
    lons = pred.variables['lon'][:]

    total_relative_bias = compute_relBIAS(pred, valid)
    total_relative_std = compute_rSD(pred, valid)
    total_w_rmse = compute_weighted_rmse(pred, valid)
    total_mse = compute_relMSE(pred, valid)
    total_mae = compute_relMAE(pred, valid)

    map_relative_bias = compute_relBIAS(pred, valid, dims='time')
    map_relative_std = compute_rSD(pred, valid, dims='time')
    map_correlation = compute_temporal_correlation(pred, valid, dims='time')
    map_w_rmse = compute_weighted_rmse(pred, valid, dims='time')
    map_rel_mse = compute_relMSE(pred, valid, dims='time')
    map_rel_mae = compute_relMAE(pred, valid, dims='time')
    map_kge = compute_KGE(pred, valid)
    
    
    proj = ccrs.PlateCarree()
    
    f, axs = plt.subplots(7, 2, figsize=(18,40), subplot_kw=dict(projection=proj))
    f.suptitle(model_description, fontsize=26, y=1.005)
    
    def plot_signal(f, signal, ax, vmin, vmax, cmap):
        cbar_shrink = 0.5
        cbar_pad = 0.03

        im = ax.contourf(lons, lats, signal, 60, transform=proj, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.coastlines()
        f.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=vmin,vmax=vmax), cmap=cmap), 
                   ax=ax, pad=cbar_pad, shrink=cbar_shrink)
    
    
    # Z500
    plot_signal(f, signal=map_relative_bias.variables['z'][:], ax=axs[0,0], vmin=-0.01, vmax=0.01, cmap='RdBu_r') # relBIAS
    plot_signal(f, signal=map_relative_std.variables['z'][:], ax=axs[1,0], vmin=0.4, vmax=1.6, cmap='RdBu_r') # rSD
    plot_signal(f, signal=map_rel_mae.variables['z'][:], ax=axs[2,0], vmin=0, vmax=0.03, cmap='Reds') # relMAE
    plot_signal(f, signal=map_correlation.variables['z'][:], ax=axs[3,0], vmin=0, vmax=1, cmap='Reds') # squared correlation
    plot_signal(f, signal=map_rel_mse.variables['z'][:], ax=axs[4,0], vmin=0, vmax=0.001, cmap='Reds') # MSE
    plot_signal(f, signal=map_w_rmse.variables['z'][:], ax=axs[5,0], vmin=0, vmax=1500, cmap='Reds') # weighted RMSE
    plot_signal(f, signal=map_kge.variables['z'][:], ax=axs[6,0], vmin=-0.2, vmax=1, cmap='Reds') # KGE
    
    # T850
    plot_signal(f, signal=map_relative_bias.variables['t'][:], ax=axs[0,1], vmin=-0.01, vmax=0.01, cmap='RdBu_r') # relBIAS
    plot_signal(f, signal=map_relative_std.variables['t'][:], ax=axs[1,1], vmin=0.4, vmax=1.6, cmap='RdBu_r') # rSD
    plot_signal(f, signal=map_rel_mae.variables['t'][:], ax=axs[2,1], vmin=0, vmax=0.03, cmap='Reds') # relMAE
    plot_signal(f, signal=map_correlation.variables['t'][:], ax=axs[3,1], vmin=0, vmax=1, cmap='Reds') # squared correlation
    plot_signal(f, signal=map_rel_mse.variables['t'][:], ax=axs[4,1], vmin=0, vmax=0.001, cmap='Reds') # MSE
    plot_signal(f, signal=map_w_rmse.variables['t'][:], ax=axs[5,1], vmin=0, vmax=8, cmap='Reds') # weighted RMSE
    plot_signal(f, signal=map_kge.variables['t'][:], ax=axs[6,1], vmin=-0.2, vmax=1, cmap='Reds') # KGE
    
    
    axs[0, 0].set_title("Z500 relBIAS map; total: {:.5f}".format(total_relative_bias.z.values), fontsize=20)
    axs[1, 0].set_title("Z500 rSD map; total: {:.5f}".format(total_relative_std.z.values), fontsize=20)
    axs[2, 0].set_title("Z500 relMAE map; total: {:.5f}".format(total_mae.z.values), fontsize=20)
    axs[3, 0].set_title("Z500 Pearsons squared correlation coefficient", fontsize=20)
    axs[4, 0].set_title("Z500 MSE map; total: {:.5f}".format(total_mse.z.values), fontsize=20)
    axs[5, 0].set_title("Z500 weighted RMSE map; total: {:.5f}".format(total_w_rmse.z.values), fontsize=20)
    axs[6, 0].set_title("Z500 KGE map", fontsize=20)
    
    
    
    axs[0, 1].set_title("T850 relBIAS map; total: {:.5f}".format(total_relative_bias.t.values), fontsize=20)
    axs[1, 1].set_title("T850 rSD map; total: {:.5f}".format(total_relative_std.t.values), fontsize=20)
    axs[2, 1].set_title("T850 relMAE map; total: {:.5f}".format(total_mae.t.values), fontsize=20)
    axs[3, 1].set_title("T850 Pearsons squared correlation coefficient map", fontsize=20)
    axs[4, 1].set_title("T850 MSE map; total: {:.5f}".format(total_mse.t.values), fontsize=20)
    axs[5, 1].set_title("T850 weighted RMSE map; total: {:.5f}".format(total_w_rmse.t.values), fontsize=20)
    axs[6, 1].set_title("T850 KGE map", fontsize=20)
    
    f.tight_layout(pad=-2)

    plt.savefig(path + model_description + ".pdf", format="pdf", bbox_inches = 'tight')
    plt.show()