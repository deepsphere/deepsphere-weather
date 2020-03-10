import xarray as xr


def compute_weighted_rmse(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    """
    Compute the RMSE with latitude weighting from two xr.DataArrays.
    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        dims (str): dimensions over which to compute the metric.
    Returns:x
        rmse: Latitude weighted root mean squared error
    """
    error = da_fc - da_true
    weights_lat = np.cos(np.deg2rad(error.lat))
    weights_lat /= weights_lat.mean()
    rmse = np.sqrt(((error)**2 * weights_lat).mean(mean_dims))
    if type(rmse) is xr.Dataset:
        rmse = rmse.rename({v: v + '_rmse' for v in rmse})
        
    else: # DataArray
        rmse.name = error.name + '_rmse' if not error.name is None else 'rmse'
    return rmse


def compute_relative_bias(da_fc, da_true, dims=xr.ALL_DIMS):
    """
    Compute the relative bias from two xr.DataArrays given some dimensions.
    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        dims (str): dimensions over which to compute the metric.
    Returns:x
        relative bias
    """
    assert dims in [xr.ALL_DIMS, 'time'], "Relative std must be computed either over all dimensions or only over time"
    
    error = da_fc - da_true
    rbias = error.mean(dims) / da_true.mean(dims)
    return rbias


def compute_relative_std(da_fc, da_true, dims=xr.ALL_DIMS):
    """
    Compute the relative standard deviation from two xr.DataArrays given some dimensions.
    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        dims (str): dimensions over which to compute the metric.
    Returns:x
        relative std
    """
    assert dims in [xr.ALL_DIMS, 'time'], "Relative std must be computed either over all dimensions or only over time"
    
    error = da_fc - da_true
    rsd = error.std(dims) / da_true.std(dims)
    return rsd


def compute_temporal_correlation(da_fc, da_true, dims='time'):
    """
    Compute the Pearson correlation coefficient from two xr.DataArrays given some dimensions.
    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        dims (str): dimensions over which to compute the metric.
    Returns:x
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

    return covariance(x, y, dims) / (x.std(dims) * y.std(dims))


def assess_model(da_fc, da_true, model_description):
    total_relative_bias = compute_relative_bias(da_fc, da_true)
    total_relative_std = compute_relative_std(da_fc, da_true)
    
    map_relative_bias = compute_relative_bias(da_fc, da_true, dims='time')
    map_relative_std = compute_relative_std(da_fc, da_true, dims='time')
    map_correlation = compute_temporal_correlation(da_fc, da_true, dims='time')
    
    f, axs = plt.subplots(1, 3, figsize=(25,5))
    f.suptitle(model_description, y=1.05)

    map_relative_bias.plot(ax=axs[0])
    map_relative_std.plot(ax=axs[1])
    map_correlation.plot(ax=axs[2])

    axs[0].set_title("Relative bias map \n Total relative bias: {:.5f}".format(total_bias.z.values))
    axs[1].set_title("Relative std map \n Total relative std: {:.5f}".format(total_std.z.values))
    axs[2].set_title("Pearsons correlation coefficient between labels and predictions\nover time")

    plt.savefig(path + model_description + ".pdf", format="pdf", bbox_inches = 'tight')
    plt.show()


