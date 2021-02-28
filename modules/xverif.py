#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 15:51:43 2021

@author: ghiggi
"""
import os
import xarray as xr 
import numpy as np 
import pandas as pd
import time 
## Weighting for equiangular 
# weights_lat = np.cos(np.deg2rad(lat))
# weights_lat /= weights_lat.sum()  
# error * weights_lat 
 
##----------------------------------------------------------------------------.
# Covariance/Correlation functions for xarray 
def _inner(x, y):
    result = np.matmul(x[..., np.newaxis, :], y[..., :, np.newaxis])
    return result[..., 0, 0]

def _xr_inner_product(x, y, dim, dask="parallelized"):
    if dim is not None:
        if isinstance(dim, str):
            dim = [dim]
        if isinstance(dim, tuple):
            dim = list()
        if len(dim) == 2: 
            # TODO reshape data to aggregate_dims x 'time'   
            raise NotImplementedError
        input_core_dims = [dim, dim] # [[x_dim, y_dim]
    else: 
        raise ValueError("Requiring a dimension...")
    return xr.apply_ufunc(_inner, x, y, 
                          input_core_dims=input_core_dims,
                          dask="parallelized",
                          output_dtypes=[float])

def _xr_covariance(x, y, aggregating_dims=None, dask="parallelized"):
    x_mean = x.mean(aggregating_dims)
    y_mean = y.mean(aggregating_dims)
    N = x.count(aggregating_dims)
    return _xr_inner_product(x - x_mean, y - y_mean, dim=aggregating_dims, dask=dask) / N
    
def _xr_pearson_correlation(x, y, aggregating_dims=None, thr=0.0000001, dask="parallelized"):
    x_std = x.std(aggregating_dims) + thr
    y_std = y.std(aggregating_dims) + thr
    return _xr_covariance(x, y, aggregating_dims=aggregating_dims, dask=dask)/(x_std*y_std)

# import bottleneck
# def _xr_rank(x, dim, dask="parallelized"): 
#     return xr.apply_ufunc(bottleneck.rankdata, x,
#                           input_core_dims=[[dim]],  
#                           dask="parallelized")

# def _xr_spearman_correlation(x, y, aggregating_dims=None, thr=0.0000001):
#     x_rank= x.rank(dim=aggregating_dims) 
#     y_rank = y.rank(dim=aggregating_dims)
#     return _xr_pearson_correlation(x_rank,y_rank, aggregating_dims=aggregating_dims, thr=thr)
##----------------------------------------------------------------------------.
def deterministic(pred, obs, 
                  forecast_type="continuous",
                  aggregating_dims=None,
                  exclude_dim=frozenset({})):
    """Compute deterministic skill metrics."""
    if not isinstance(forecast_type, str): 
        raise TypeError("'forecast_type' must be a string specifying the forecast type.")
    if forecast_type not in ["continuous", "categorical"]:
        raise ValueError("'forecast_type' must be either 'continuous' or 'categorical'.") 
    if forecast_type == 'continuous':
        t_i = time.time()
        ds_skill = _deterministic_continuous_metrics(pred = pred,
                                                 obs = obs, 
                                                 aggregating_dims=aggregating_dims,
                                                 exclude_dim=exclude_dim)
        print("- Continuous deterministic verification: {:.2f} minutes".format((time.time() - t_i)/60))
        return ds_skill
    else: 
        t_i = time.time()
        raise NotImplementedError('Categorical forecast skill metrics are not yet implemented.')
        print("- Categorical deterministic verification: {:.2f} minutes".format((time.time() - t_i)/60))
        
def _deterministic_continuous_metrics(pred, obs, 
                                      aggregating_dims = None, 
                                      exclude_dim=frozenset({}),
                                      thr=0.000001):
    """Deterministic metrics for continuous predictions forecasts.
    
    percMAE : also known as MAPE 
    pearson_R2 : also known as coefficient of determination 
    
    References
    ----------
    KGE:
         Gupta, Kling, Yilmaz, Martinez, 2009, Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling
    """
    # TODO robust with median and IQR / MAD 
    # exclude_dim: 'member'? 
    ##----------------------------------------------------------------------------.
    # - Remove NaN (in aggregating_dims (i.e. time) dimension)
    pred = pred.dropna(dim=aggregating_dims)
    obs = obs.dropna(dim=aggregating_dims)
    # - Align datasets 
    pred, obs = xr.align(pred, obs, join="inner", exclude=exclude_dim)
    ##----------------------------------------------------------------------------.
    # - Error 
    error = pred - obs
    error_abs = np.abs(error)
    error_squared = error**2
    error_perc = error_abs/(obs + thr)
    print(1)
    ##-------------------------------------------------------------------------.
    # - Mean
    from dask.diagnostics import ProgressBar
    with ProgressBar():
        pred_mean = pred.mean(aggregating_dims).compute()
    obs_mean = obs.mean(aggregating_dims).compute()
    error_mean = error.mean(aggregating_dims).compute()
    ##-------------------------------------------------------------------------. 
    # - Standard deviation
    pred_std = pred.std(aggregating_dims).compute()
    obs_std = obs.std(aggregating_dims).compute()
    error_std = error.std(aggregating_dims).compute()
    print(2)
    ##-------------------------------------------------------------------------.
    # - Coefficient of variability
    pred_CoV = pred_std / (pred_mean + thr) 
    obs_CoV = obs_std / (obs_mean + thr)
    error_CoV = error_std / (error_mean + thr)
    ##-------------------------------------------------------------------------.
    # - Magnitude metrics
    BIAS = error_mean
    MAE = np.abs(error).mean(aggregating_dims).compute()
    MSE = error_squared.mean(aggregating_dims).compute()
    RMSE = np.sqrt(MSE)

    percBIAS = error_perc.mean(aggregating_dims).compute()*100
    percMAE = np.abs(error_perc).mean(aggregating_dims).compute()*100  

    relBIAS = BIAS / (obs_mean + thr)
    relMAE = MAE / (obs_mean + thr)
    relMSE = MSE / (obs_mean + thr)
    relRMSE = RMSE / (obs_mean + thr)
    ##-------------------------------------------------------------------------.
    # - Average metrics 
    rMean = pred_mean / (obs_mean + thr)
    diffMean = pred_mean - obs_mean
    ##-------------------------------------------------------------------------.
    # - Variability metrics 
    rSD = pred_std / (obs_std + thr)
    diffSD = pred_std - obs_std  
    rCoV = pred_CoV / obs_CoV
    diff_CoV = pred_CoV - obs_CoV
    ##-------------------------------------------------------------------------.
    print(3)
    # - Correlation metrics 
    # --> TODO: only works when len(aggregating_dims) == 1
    # --> If > 1 ... reshape data of aggregating_dims into 1D dimension
    pearson_R = _xr_pearson_correlation(pred, obs, 
                                        aggregating_dims=aggregating_dims, 
                                        thr=thr).compute()
    pearson_R2 = pearson_R**2

    # spearman_R = _xr_spearman_correlation(pred, obs,
    #                                       aggregating_dims=aggregating_dims,
    #                                       thr=thr).compute()
    # spearman_R2 = spearman_R**2
    ##-------------------------------------------------------------------------.
    # - Overall skill metrics 
    print(4)
    LTM_forecast_error = ((obs_mean - obs)**2).sum(aggregating_dims) # Long-term mean as prediction
    NSE = 1 - ( error_squared.sum(aggregating_dims)/ (LTM_forecast_error + thr) )
    NSE = NSE.compute()
    KGE = 1 - ( np.sqrt((pearson_R - 1)**2 + (rSD - 1)**2 + (rMean - 1)**2) )
    ##-------------------------------------------------------------------------.
    # - Create dictionary skill 
    # If dimension is provided as a DataArray or Index
    skill_dict = {"error_CoV": error_CoV,
                    "obs_CoV": obs_CoV,
                    "pred_CoV": pred_CoV,
                    # Magnitude 
                    "BIAS": BIAS,
                    "relBIAS": relBIAS,
                    "percBIAS": percBIAS,
                    "MAE": MAE,
                    "relMAE": relMAE,
                    "percMAE": percMAE,
                    "MSE": MSE,
                    "relMSE": relMSE,
                    "RMSE": RMSE,
                    "relRMSE": relRMSE,
                    # Average
                    "rMean": rMean,
                    "diffMean": diffMean,
                    # Variability 
                    'rSD': rSD,
                    'diffSD': diffSD,
                    "rCoV": rCoV,
                    "diff_CoV": diff_CoV,
                    # Correlation 
                    "pearson_R": pearson_R,
                    "pearson_R2": pearson_R2,
                    # "spearman_R": spearman_R,
                    # "spearman_R2": spearman_R2,
                    # Overall skills
                    "NSE": NSE,
                    "KGE": KGE,
                    }
    # Create skill Dataset 
    skill_index = pd.Index(skill_dict.keys())
    ds_skill = xr.concat(skill_dict.values(), dim=skill_index, coords="all")
    ds_skill = ds_skill.rename({'concat_dim':'skill'})
    # Return the skill Dataset
    return ds_skill

def global_summary(ds, area_coords="area"):
    # Check area_coords
    area_weights = ds[area_coords]/ds[area_coords].values.sum()
    aggregating_dims = list(area_weights.dims)
    ds_weighted = ds.weighted(area_weights)
    return ds_weighted.mean(aggregating_dims)

def latitudinal_summary(ds, lat_dim='lat', lon_dim='lon', lat_res=5):
    # Check lat_dim and lon_dim 
    # Check lat_res < 90 
    # TODO: lon between -180 and 180 , lat between -90 and 90 
    aggregating_dims = list(ds[lon_dim].dims)
    bins = np.arange(-90,90+lat_res, step=lat_res)
    labels = bins[:-1] + lat_res/2
    return ds.groupby_bins(lat_dim, bins, labels=labels).mean(aggregating_dims) 
    
def longitudinal_summary(ds, lat_dim='lat', lon_dim='lon', lon_res=5):
    # Check lat_dim and lon_dim 
    # Check lon_res < 180 
    # TODO: lon between -180 and 180 , lat between -90 and 90 
    aggregating_dims = list(ds[lon_dim].dims)
    bins = np.arange(-180,180+lon_res, step=lon_res)
    labels= bins[:-1] + lon_res/2
    return ds.groupby_bins(lon_dim, bins, labels=labels).mean(aggregating_dims) 





