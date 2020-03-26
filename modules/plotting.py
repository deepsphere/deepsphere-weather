import xarray as xr
import numpy as np
import datetime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy

from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

from modules.utils import plot_signal
from modules.test import (compute_anomalies, compute_weighted_rmse, compute_relBIAS, compute_rSD, 
                          compute_temporal_correlation, compute_KGE, compute_relMSE, compute_relMAE)


def plot_anomalies(ds_input, ds_pred, ds_labels, timestep, mean, model_description, save_path):
    """ Plots the evolution of anomalies for a given prediction and its corresponding observation
    
    Parameters
    ----------
    ds_input : xr.Dataset
        Samples that are input to the model
    ds_pred : xr.Dataset
        Predictions
    ds_labels : xr.Dataset
        Observations
    timestep : int
        index of the starting timestep
    mean : str
        Whether to subtract the weekly or monthly mean. Options are {´weekly´, ´monthly´}
    """
    lead_time = ((ds_pred.isel(time=0).time.values - ds_input.isel(time=0).time.values)
                 /(1e9*3600)).astype(int)
    
    sample_in = compute_anomalies(ds_input, mean).isel(time=timestep)
    sample_pred = compute_anomalies(ds_pred, mean).isel(time=timestep)
    sample_label = compute_anomalies(ds_labels, mean).isel(time=timestep)
    
    proj = ccrs.PlateCarree()
    f, axs = plt.subplots(2, 3, figsize=(15, 5), subplot_kw=dict(projection=proj))
    f.suptitle("Anomaly displacement at t+{} hours".format(lead_time), fontsize=26, y=1.1)
    
        
    def compute_min_max(samples):
        mins = [samples[i].min(dim=xr.ALL_DIMS).values for i in range(len(samples))]
        maxs = [samples[i].max(dim=xr.ALL_DIMS).values for i in range(len(samples))]
        return min(mins), max(maxs)
    
    vmin_z, vmax_z = compute_min_max([sample_in.z, sample_pred.z, sample_label.z])
    vmin_t, vmax_t = compute_min_max([sample_in.t, sample_pred.t, sample_label.t])
    
    # Z500
    plot_signal(f, sample=sample_in, var='z', vmin=vmin_z, vmax=vmax_z, proj=proj, ax=axs[0,0])
    plot_signal(f, sample=sample_pred, var='z', vmin=vmin_z, vmax=vmax_z, proj=proj, ax=axs[0,1])
    plot_signal(f, sample=sample_label, var='z', vmin=vmin_z, vmax=vmax_z, proj=proj, ax=axs[0,2])
    
    # T850
    plot_signal(f, sample=sample_in, var='t', vmin=vmin_t, vmax=vmax_t, proj=proj, ax=axs[1,0])
    plot_signal(f, sample=sample_pred, var='t', vmin=vmin_t, vmax=vmax_t, proj=proj, ax=axs[1,1])
    plot_signal(f, sample=sample_label, var='t', vmin=vmin_t, vmax=vmax_t, proj=proj, ax=axs[1,2])
    
    axs[0, 0].set_title("Z500 t+0 hours", fontsize=18)
    axs[0, 1].set_title("Z500 t+{} hours prediction".format(lead_time), fontsize=18)
    axs[0, 2].set_title("Z500 t+{} hours observation".format(lead_time), fontsize=18)
    
    axs[1, 0].set_title("T850 t+0 hours")
    axs[1, 1].set_title("T850 t+{} hours prediction".format(lead_time), fontsize=18)
    axs[1, 2].set_title("T850 t+{} hours observation".format(lead_time), fontsize=18)
    
    f.tight_layout(pad=-2)
    filename = save_path + 'anomalies_' + model_description + ".pdf"
    plt.savefig(filename, bbox_inches = 'tight')
    
    plt.show()
    
    
    
def plot_evaluation(pred, valid, title, filename):
    """ Compute and plot relBIAS, rSD, R2 and KGE between predictions and labels and display the results for T850 and Z500
    
    Parameters
    ----------
    pred : xr.Dataset
        Predictions
    valid : xr. Dataset
        Observations
    title : str
        Plot title    
    """
    total_relative_bias = compute_relBIAS(pred, valid)
    total_relative_std = compute_rSD(pred, valid)
    
    map_relative_bias = compute_relBIAS(pred, valid, dims='time')
    map_relative_std = compute_rSD(pred, valid, dims='time')
    map_correlation = compute_temporal_correlation(pred, valid, dims='time')
    map_kge = compute_KGE(pred, valid)
    
    proj = ccrs.PlateCarree()
    f, axs = plt.subplots(4, 2, figsize=(18,18), subplot_kw=dict(projection=proj))
    f.suptitle(title, fontsize=26, y=1.05)
    
    plot_signal(f, sample=map_relative_bias, var='z', ax=axs[0,0], vmin=-0.02, vmax=0.02, proj=proj, cmap='RdBu_r') # relBIAS
    plot_signal(f, sample=map_relative_std, var='z', ax=axs[1,0], vmin=0.4, vmax=1.6, proj=proj, cmap='RdBu_r') # rSD
    plot_signal(f, sample=map_correlation, var='z', ax=axs[2,0], vmin=0, vmax=1, proj=proj, cmap='Reds') # squared correlation
    plot_signal(f, sample=map_kge, var='z', ax=axs[3,0], vmin=-0.2, vmax=1, proj=proj, cmap='Reds')
    
    plot_signal(f, sample=map_relative_bias, var='t', ax=axs[0,1], vmin=-0.02, vmax=0.02, proj=proj, cmap='RdBu_r') # relBIAS
    plot_signal(f, sample=map_relative_std, var='t', ax=axs[1,1], vmin=0.4, vmax=1.6, proj=proj, cmap='RdBu_r') # rSD
    plot_signal(f, sample=map_correlation, var='t', ax=axs[2,1], vmin=0, vmax=1, proj=proj, cmap='Reds') # squared correlation
    plot_signal(f, sample=map_kge, var='t', ax=axs[3,1], vmin=-0.2, vmax=1, proj=proj, cmap='Reds') # KGE
    
    axs[0, 0].set_title("Z500 relBIAS map; total: {:.5f}".format(total_relative_bias.z.values), fontsize=20)
    axs[1, 0].set_title("Z500 rSD map; total: {:.5f}".format(total_relative_std.z.values), fontsize=20)
    axs[2, 0].set_title("Z500 Pearsons squared correlation coefficient", fontsize=20)
    axs[3, 0].set_title("Z500 KGE map", fontsize=20)
    
    axs[0, 1].set_title("T850 relBIAS map; total: {:.5f}".format(total_relative_bias.t.values), fontsize=20)
    axs[1, 1].set_title("T850 rSD map; total: {:.5f}".format(total_relative_std.t.values), fontsize=20)
    axs[2, 1].set_title("T850 Pearsons squared correlation coefficient map", fontsize=20)
    axs[3, 1].set_title("T850 KGE map", fontsize=20)
    
    f.tight_layout(pad=-2)
    
    plt.savefig(filename, bbox_inches = 'tight')

    plt.show()
    
    
    
def assess_month(pred, valid, month, model_description, save_path):
    """ Assesses the performance of a model for a given month
    
    Parameters
    ----------
    pred : xr.Dataset
        Predictions
    valid : xr.Dataset
        Observations
    month : str
        Month to evaluate
    model_description : str
        Short description of the model, used for the filename
    save_path : str
        Path where figure is saved
    """
    months = {'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6, 'july': 7, 
             'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12}
    
    pred = pred.sel(time=pred['time.month']==months[month])
    valid = valid.sel(time=valid['time.month']==months[month])
    
    title = "Model evaluation for " + month
    filename = save_path + month + '_' + model_description + ".pdf"
    
    plot_evaluation(pred, valid, title, filename)
    
    
    
def assess_season(pred, valid, season, model_description, save_path):
    """ Assesses the performance of a model for a given season
    
    Parameters
    ----------
    pred : xr.Dataset
        Predictions
    valid : xr.Dataset
        Observations
    season : str
        Season to evaluate
    model_description : str
        Short description of the model, used for the filename
    save_path : str
        Path where figure is saved
    """
    seasons = {'winter': 'DJF', 'spring': 'MAM', 'summer': 'JJA', 'fall': 'SON'}
    
    pred = pred.sel(time=pred['time.season']==seasons[season])
    valid = valid.sel(time=valid['time.season']==seasons[season])
    
    title = "Model evaluation for " + season
    filename = save_path + season + '_' + model_description + ".pdf"
    
    plot_evaluation(pred, valid, title, filename)
    
    
def assess_seasonal_cycle(pred, valid, model_description, save_path):
    """ Assesses the performance of a model in reproducing the seasonal cycle
    
    Parameters
    ----------
    pred : xr.Dataset
        Predictions
    valid : xr.Dataset
        Observations
    model_description : str
        Short description of the model, used for the filename
    save_path : str
        Path where figure is saved
    """
    seasons = {'winter': 'DJF', 'spring': 'MAM', 'summer': 'JJA', 'fall': 'SOM'}
    
    pred = pred.groupby('time.month').mean().rename({'month':'time'})
    valid = valid.groupby('time.month').mean().rename({'month':'time'})
    
    title = "Model's seasonal cycle evaluation"
    filename = save_path + 'seasonal_cycle_' + model_description + ".pdf"
    
    plot_evaluation(pred, valid, title, filename)
    
    
def assess_daily_cycle(pred, valid, model_description, save_path):
    """ Assesses the performance of a model in reproducing the daily cycle
    
    Parameters
    ----------
    pred : xr.Dataset
        Predictions
    valid : xr.Dataset
        Observations
    model_description : str
        Short description of the model, used for the filename
    save_path : str
        Path where figure is saved
    """
    pred = pred.groupby('time.hour').mean().rename({'hour': 'time'})
    valid = valid.groupby('time.hour').mean().rename({'hour': 'time'})
    
    title = "Model's daily cycle evaluation"
    filename = save_path + 'daily_cycle_' + model_description + ".pdf"
    
    plot_evaluation(pred, valid, title, filename)


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
    
    
    # Z500
    plot_signal(f, sample=map_relative_bias, var='z', ax=axs[0,0], vmin=-0.01, vmax=0.01, cmap='RdBu_r') # relBIAS
    plot_signal(f, sample=map_relative_std, var='z', ax=axs[1,0], vmin=0.4, vmax=1.6, cmap='RdBu_r') # rSD
    plot_signal(f, sample=map_rel_mae, var='z', ax=axs[2,0], vmin=0, vmax=0.03, cmap='Reds') # relMAE
    plot_signal(f, sample=map_correlation, var='z', ax=axs[3,0], vmin=0, vmax=1, cmap='Reds') # squared correlation
    plot_signal(f, sample=map_rel_mse, var='z', ax=axs[4,0], vmin=0, vmax=0.001, cmap='Reds') # MSE
    plot_signal(f, sample=map_w_rmse, var='z', ax=axs[5,0], vmin=0, vmax=1500, cmap='Reds') # weighted RMSE
    plot_signal(f, sample=map_kge, var='z', ax=axs[6,0], vmin=-0.2, vmax=1, cmap='Reds') # KGE
    
    # T850
    plot_signal(f, sample=map_relative_bias, var='t', ax=axs[0,1], vmin=-0.01, vmax=0.01, cmap='RdBu_r') # relBIAS
    plot_signal(f, sample=map_relative_std, var='t', ax=axs[1,1], vmin=0.4, vmax=1.6, cmap='RdBu_r') # rSD
    plot_signal(f, sample=map_rel_mae, var='t', ax=axs[2,1], vmin=0, vmax=0.03, cmap='Reds') # relMAE
    plot_signal(f, sample=map_correlation, var='t', ax=axs[3,1], vmin=0, vmax=1, cmap='Reds') # squared correlation
    plot_signal(f, sample=map_rel_mse, var='t', ax=axs[4,1], vmin=0, vmax=0.001, cmap='Reds') # MSE
    plot_signal(f, sample=map_w_rmse, var='t', ax=axs[5,1], vmin=0, vmax=8, cmap='Reds') # weighted RMSE
    plot_signal(f, sample=map_kge, var='t', ax=axs[6,1], vmin=-0.2, vmax=1, cmap='Reds')
    
    
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