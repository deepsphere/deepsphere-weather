import xarray as xr
import numpy as np
import datetime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy

from matplotlib.axes import Axes
from matplotlib import cm, colors
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

from modules.test import (compute_anomalies, compute_weighted_rmse, compute_relBIAS, compute_rSD, 
                          compute_R2, compute_KGE, compute_relMSE, compute_relMAE, compute_ACC)


# Global dictionaries
skill_fcts = {"relBIAS": compute_relBIAS, "relMAE": compute_relMAE, "relMSE": compute_relMSE, "rSD": compute_rSD, 
                  "R2": compute_R2, "ACC": compute_ACC, "KGE": compute_KGE}
    
cmaps = {"relBIAS": 'RdBu_r', "relMAE": 'Reds', "relMSE": 'Reds', "rSD": 'RdBu_r', 
             "R2": 'Reds', "ACC": 'Reds', "KGE": 'Reds'}

months = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 
              9: 'September', 10: 'October', 11: 'November', 12: 'December'}

proj = ccrs.PlateCarree() 


# Private function to compute vmin, vmax
def _compute_min_max(samples):
    """ Computes the minimum and maximum of a list of xarray dataholders
    
    Parameters
    ----------
    samples : list
        List of xarray dataholders (Datasets or DataArrays)
    
    Returns
    -------
    vmin, vmax: int, int
        Minimum and maximum among the whole list of dataholders
    """
    
    mins = [samples[i].min(dim=xr.ALL_DIMS).values for i in range(len(samples))]
    maxs = [samples[i].max(dim=xr.ALL_DIMS).values for i in range(len(samples))]
    return min(mins), max(maxs)


def plot_signal(f, sample, var, ax, vmin, vmax, proj, cmap='RdBu_r', cbar_shrink=0.6, cbar_pad=0.03):
    """ Plots a weather signal drawing coastlines

    Parameters
    ----------
    f : matplotlib.pyplot.figure
        Figure container
    sample : xr.DataArray
        Sample containing signals to plot
    var : string
        Variable to plot
    ax : artopy.mpl.geoaxes
        Axes where plot is drawn
    vmin : float
        Minimum value for colorbar
    vmax : float
        Maximum value for colorbar
    proj: cartopy.crs.CRS (Coordinate Reference System)
        Geoaxes projection
    cmap : string
        Colormap
    cbar_shrink : float
        Fraction of axes describing the colorbar size
    cbar_pad : float
        Padding between plot axes and colorbar
    """
    sample = sample.roll(lon=int(len(sample.lon)/2), roll_coords=True)
    lats = sample.variables['lat'][:]
    lons = sample.variables['lon'][:]
    signal = sample.variables[var]

    im = ax.pcolormesh(lons, lats, signal, transform=proj, cmap=cmap, vmin=vmin, vmax=vmax, shading='gouraud')
    ax.coastlines()
    f.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=vmin,vmax=vmax), cmap=cmap), 
               ax=ax, pad=cbar_pad, shrink=cbar_shrink)


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
    
    
    vmin_z, vmax_z = _compute_min_max([sample_in.z, sample_pred.z, sample_label.z])
    vmin_t, vmax_t = _compute_min_max([sample_in.t, sample_pred.t, sample_label.t])
    
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
    
    axs[1, 0].set_title("T850 t+0 hours", fontsize=18)
    axs[1, 1].set_title("T850 t+{} hours prediction".format(lead_time), fontsize=18)
    axs[1, 2].set_title("T850 t+{} hours observation".format(lead_time), fontsize=18)
    
    f.tight_layout(pad=-2)
    filename = save_path + 'anomalies_' + model_description + ".pdf"
    plt.savefig(filename, bbox_inches = 'tight')
    
    plt.show()
    
    
    
def plot_evaluation(pred, valid, title, filename, acc=True):
    """ Compute and plot relBIAS, rSD, R2, relMSE, ACC and KGE between predictions and labels and display the 
    results for T850 and Z500
    
    Parameters
    ----------
    pred : xr.Dataset
        Predictions
    valid : xr. Dataset
        Observations
    title : str
        Plot title
    filename : str
        Filename to save figure
    acc : bool
        Whether or not to include ACC in evaluation
    """
    skills = ["relBIAS", "rSD", "R2", "relMSE", "ACC", "KGE"]
    if not acc:
        skills.remove("ACC")
    n_skills = len(skills)
    
    skillmaps = []
    for skill in skills:
        skillmaps.append(skill_fcts[skill](pred, valid))
    
    f, axs = plt.subplots(n_skills, 2, figsize=(18, 5*n_skills), subplot_kw=dict(projection=proj))
    axs_ = np.array(axs).reshape(-1, order='F')
    f.suptitle(title, fontsize=26, y=1.02)
    
    # Z500
    for i, ax in enumerate(axs_[:n_skills]):
        vmin, vmax = _compute_min_max([skillmaps[i].z])
        if skills[i] == "rSD":
            vmin, vmax = 1 - max(abs(1 - vmin), abs(vmax - 1)), 1 + max(abs(1 - vmin), abs(vmax - 1))
        
        plot_signal(f, sample=skillmaps[i], var='z', ax=ax, vmin=vmin, vmax=vmax, proj=proj, cmap=cmaps[skills[i]])
        ax.set_title("Z500 " + skills[i], fontsize=20)
            
    for i, ax in enumerate(axs_[n_skills:]):
        vmin, vmax = _compute_min_max([skillmaps[i].t])
        if skills[i] == "rSD":
            vmin, vmax = 1 - max(abs(1 - vmin), abs(vmax - 1)), 1 + max(abs(1 - vmin), abs(vmax - 1))
        
        plot_signal(f, sample=skillmaps[i], var='t', ax=ax, vmin=vmin, vmax=vmax, proj=proj, cmap=cmaps[skills[i]])
        ax.set_title("T850 " + skills[i], fontsize=20)
        
        
    f.tight_layout(pad=-2)
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    
    
def assess_month(pred, valid, skill_name, model_description, lead_time, save_path):
    """ Assesses the performance of a model for a given month
    
    Parameters
    ----------
    pred : xr.Dataset
        Predictions
    valid : xr.Dataset
        Observations
    skill_name : str
        Skill to evaluate
    model_description : str
        Short description of the model, used for the filename
    lead_time : int
        Forecast leading time
    save_path : str
        Path where figure is saved
    """    
    
    monthly_skill = []
    for month in range(1, 12+1):
        monthly_pred = pred.sel(time=pred['time.month']==month)
        monthly_obs = valid.sel(time=valid['time.month']==month)
        monthly_skill.append(skill_fcts[skill_name](monthly_pred, monthly_obs))
        
    vmin_z, vmax_z = _compute_min_max([skillmap['z'] for skillmap in monthly_skill])
    vmin_t, vmax_t = _compute_min_max([skillmap['t'] for skillmap in monthly_skill])
    
    if skill_name == "rSD":
        vmin_z, vmax_z = 1 - max(abs(1 - vmin_z), abs(vmax_z - 1)), 1 + max(abs(1 - vmin_z), abs(vmax_z - 1))
        vmin_t, vmax_t = 1 - max(abs(1 - vmin_t), abs(vmax_t - 1)), 1 + max(abs(1 - vmin_t), abs(vmax_t - 1))
    
    title = "Monthly evaluation of " + skill_name + " for a {} h lead time".format(lead_time)                        
    filename = save_path + "_".join(["MonthlySummary", skill_name, model_description, str(lead_time)]) + ".png"
    
    
    # Plot
    f, axs = plt.subplots(8, 3, figsize=(18, 30), subplot_kw=dict(projection=proj))
    axs = np.array(axs)
    f.suptitle(title, fontsize=26, y=1.02)

    for i, ax in enumerate(axs.reshape(-1)[:12]):
        plot_signal(f, sample=monthly_skill[i], var='z', ax=ax, vmin=vmin_z, vmax=vmax_z, proj=proj,
                    cmap=cmaps[skill_name])
        ax.set_title(months[i+1] + " Z500", fontsize=20)
    
    for i, ax in enumerate(axs.reshape(-1)[12:]):
        plot_signal(f, sample=monthly_skill[i], var='t', ax=ax, vmin=vmin_t, vmax=vmax_t, proj=proj,
                    cmap=cmaps[skill_name])
        ax.set_title(months[i+1] + " T850", fontsize=20)
    
    
    f.tight_layout(pad=-2)
    plt.savefig(filename, bbox_inches = 'tight')
    
    
def assess_season(pred, valid, skill_name, model_description, lead_time, save_path):
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
    lead_time : int
        Forecast leading time
    save_path : str
        Path where figure is saved
    """
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    
    
    seasonal_skill = []
    for season in ['DJF', 'MAM', 'JJA', 'SON']:
        seasonal_pred = pred.sel(time=pred['time.season']==season)
        seasonal_obs = valid.sel(time=valid['time.season']==season)
        seasonal_skill.append(skill_fcts[skill_name](seasonal_pred, seasonal_obs))
        
    vmin_z, vmax_z = _compute_min_max([skillmap['z'] for skillmap in seasonal_skill])
    vmin_t, vmax_t = _compute_min_max([skillmap['t'] for skillmap in seasonal_skill])
    
    if skill_name == "rSD":
        vmin_z, vmax_z = 1 - max(abs(1 - vmin_z), abs(vmax_z - 1)), 1 + max(abs(1 - vmin_z), abs(vmax_z - 1))
        vmin_t, vmax_t = 1 - max(abs(1 - vmin_t), abs(vmax_t - 1)), 1 + max(abs(1 - vmin_t), abs(vmax_t - 1))
    
    title = "Seasonal evaluation of " + skill_name + " for a {} h lead time".format(lead_time)                        
    filename = save_path + "_".join(["SeasonalSummary", skill_name, model_description, str(lead_time)]) + ".png"
    
    
    # Plot
    f, axs = plt.subplots(4, 2, figsize=(18, 18), subplot_kw=dict(projection=proj))
    axs = np.array(axs)
    f.suptitle(title, fontsize=26, y=1.05)

    for i, (ax, season) in enumerate(zip(axs.reshape(-1)[:4], seasons)):
        plot_signal(f, sample=seasonal_skill[i], var='z', ax=ax, vmin=vmin_z, vmax=vmax_z, proj=proj,
                    cmap=cmaps[skill_name])
        ax.set_title(season + " Z500", fontsize=20)
    
    for i, (ax, season) in enumerate(zip(axs.reshape(-1)[4:], seasons)):
        plot_signal(f, sample=seasonal_skill[i], var='t', ax=ax, vmin=vmin_t, vmax=vmax_t, proj=proj,
                    cmap=cmaps[skill_name])
        ax.set_title(season + " T850", fontsize=20)
    
    
    f.tight_layout(pad=-2)
    plt.savefig(filename, bbox_inches = 'tight')
    
    
def assess_seasonal_cycle(pred, valid, model_description, lead_time, save_path):
    """ Assesses the performance of a model in reproducing the seasonal cycle
    
    Parameters
    ----------
    pred : xr.Dataset
        Predictions
    valid : xr.Dataset
        Observations
    model_description : str
        Short description of the model, used for the filename
    lead_time : int
        Forecast leading time
    save_path : str
        Path where figure is saved
    """
    
    pred = pred.groupby('time.month').mean().rename({'month':'time'})
    valid = valid.groupby('time.month').mean().rename({'month':'time'})
    
    title = "Model's seasonal cycle evaluation for a {} h lead time".format(lead_time)                         
    filename = save_path + "_".join(["SeasonalCycle", model_description, str(lead_time)]) + ".png"
    
    plot_evaluation(pred, valid, title, filename, acc=False)
    
    
def assess_daily_cycle(pred, valid, model_description, lead_time, save_path):
    """ Assesses the performance of a model in reproducing the daily cycle
    
    Parameters
    ----------
    pred : xr.Dataset
        Predictions
    valid : xr.Dataset
        Observations
    model_description : str
        Short description of the model, used for the filename
    lead_time : int
        Forecast leading time
    save_path : str
        Path where figure is saved
    """
    
    pred = pred.groupby('time.hour').mean().rename({'hour': 'time'})
    valid = valid.groupby('time.hour').mean().rename({'hour': 'time'})
    
    title = "Model's daily cycle evaluation for a {} h lead time".format(lead_time)
    filename = save_path + "_".join(["DailyCycle", model_description, str(lead_time)]) + ".png"
    
    plot_evaluation(pred, valid, title, filename, acc=False)


def assess_globally(pred, valid, model_description, lead_time, save_path):
    """ Assess predictions comparing them to label data using several metrics
    
    Parameters
    ----------
    pred : xr.DataArray
        Forecast. Time coordinate must be validation time.
    valid : xr.DataArray
        Labels
    model_description : str
        Short description of the model, used for the filename
    lead_time : int
        Forecast leading time
    save_path : str
        Path where figure is saved
       
    Returns
    -------
    plt.plot
        Several plots showing the predictions' rightness
    """
    
    title = "Model's global evaluation for a {} h lead time".format(lead_time)
    filename = save_path + "_".join(["GlobalSummary", model_description, str(lead_time)]) + ".png"
    
    plot_evaluation(pred, valid, title, filename)