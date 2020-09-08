import xarray as xr
import numpy as np
import healpy as hp
import datetime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import pickle

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib import cm, colors
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

from modules.test import (compute_anomalies, compute_weighted_rmse, compute_relBIAS, compute_rSD, 
                          compute_R2, compute_KGE, compute_relMSE, compute_relMAE, compute_ACC)

#from scipy import interpolate
from modules.data import hp_to_equiangular


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

def plot_rmses(rmse, reference_rmse, lead_time, max_lead_time=120):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
    
    lead_times = np.arange(lead_time, max_lead_time + lead_time, lead_time)
    ax1.plot(lead_times, rmse.z.values, label='Spherical Weyn')
    ax1.plot(lead_times, reference_rmse.z.values, label='Actual Weyn')
    ax2.plot(lead_times, rmse.t.values, label='Spherical Weyn')
    ax2.plot(lead_times, reference_rmse.t.values, label='Actual Weyn')
    ax1.set_xlabel('Lead time (h)')
    ax1.set_xticks(lead_times)
    ax2.set_xticks(lead_times)
    ax2.set_xlabel('Lead time (h)')
    ax1.set_ylabel('RMSE')
    ax2.set_ylabel('RMSE')
    ax1.set_title('Z500')
    ax2.set_title('T850')
    ax1.legend()
    ax2.legend()
    plt.show()


def plot_signal(f, sample, var, ax, vmin, vmax, proj, cmap, colorbar, cbar_label, cbar_shrink=0.7, 
                cbar_pad=0.03, extend='neither'):
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
    colorbar : bool
        Whether or not to draw a colorbar 
    cbar_label : string
        Colorbar label
    cbar_shrink : float
        Fraction of axes describing the colorbar size
    cbar_pad : float
        Padding between plot axes and colorbar
    extend : string
        Whether or not to draw extended colorbars. Options are [ 'neither' | 'both' | 'min' | 'max' ] 
    """
    
    sample = sample.roll(lon=int(len(sample.lon)/2), roll_coords=True)
    lats = sample.variables['lat'][:]
    lons = sample.variables['lon'][:]
    signal = sample.variables[var]

    im = ax.pcolormesh(lons, lats, signal, transform=proj, cmap=cmap, vmin=vmin, vmax=vmax, shading='gouraud')
    ax.coastlines()
    
    cb = f.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=vmin,vmax=vmax), cmap=cmap), 
                    ax=ax, pad=cbar_pad, shrink=cbar_shrink, extend=extend)
    
    cb.set_label(label=cbar_label, size=18)
    cb.ax.tick_params(labelsize=16)
    
    if not colorbar:
        cb.remove()
        
        
def plot_benchmark(rmses_spherical, model_description, lead_times, input_dir, output_dir, title=True):
    """
    Plot rmse values at different lead times from different models

    :param rmses_spherical: xarray
    :param model_description: string
    :param lead_times: xarray
    :param input_dir: string
    :param output_dir: string
    :param title: string
    :return: None
    """
    
    lead_times0 = np.arange(6, lead_times[-1]+6, 6)
    
    xlabels = [str(t) if t%4 == 0 else '' for t in lead_times] if lead_times[0] < 12 else lead_times
    
    # RMSE baselines
    rmses_baselines = pickle.load(open('../data/models/baselines/'+'rmse.pkl', 'rb'))
    
    rmses_rasp_direct = rmses_baselines['CNN (direct)']
    rmses_rasp_iter = rmses_baselines['CNN (iterative)']
    rmses_climatology = rmses_baselines['Climatology']
    rmses_weekly_clim = rmses_baselines['Weekly clim.']
    rmses_persistence = rmses_baselines['Persistence']
    rmses_ifs = rmses_baselines['Operational'].sel(lead_time=lead_times)
    rmses_ifs_t42 = rmses_baselines['IFS T42'].sel(lead_time=lead_times)
    rmses_ifs_t63 = rmses_baselines['IFS T63'].sel(lead_time=slice(lead_times[0], lead_times[-1]))
    rmses_weyn = xr.open_dataset(input_dir + 'rmses_weyn.nc').rename({'z500':'z', 't850':'t'})
    
    f, axs = plt.subplots(1, 2, figsize=(17, 6))
    if title:
        f.suptitle('RMSE between forecast and observation as a function of forecast time', fontsize=24, y=1.07)

    axs[0].plot(lead_times0, rmses_persistence.z.values, label='Persistence', linestyle='--')
    axs[0].plot(lead_times0, [rmses_climatology.z.values]*len(lead_times0), label='Global climatology', linestyle='--')
    axs[0].plot(lead_times0, [rmses_weekly_clim.z.values]*len(lead_times0), label='Weekly climatology', linestyle='--')
    axs[0].plot(lead_times0, rmses_ifs.z.values, label='Operational IFS', linestyle='--')
    #axs[0].plot(lead_times0, rmses_ifs_t42.z.values, label='IFS T42', linestyle='--')
    #axs[0].plot(rmses_ifs_t63.lead_time.values, rmses_ifs_t63.z.values, label='IFS T63', linestyle='--')
    axs[0].scatter([72, 120], rmses_rasp_direct.z.values, label='Rasp 2020 (direct)', color='maroon')
    #axs[0].plot(lead_times0, rmses_rasp_iter.z.values, label='Rasp 2020 (iter)', linestyle='-')
    axs[0].plot(lead_times0, rmses_weyn.z.values[:len(lead_times0)], label='Weyn 2020', linestyle='-')
    axs[0].plot(lead_times, rmses_spherical.z.values, label='Ours', color='black', marker='o')

    axs[0].set_ylabel('RMSE [$m^2 s^{−2}$]', fontsize=18)
    axs[0].set_xlabel('Forecast time [h]', fontsize=18)
    axs[0].set_title('Z500', fontsize=22)
    axs[0].tick_params(axis='both', which='major', labelsize=16)
    axs[0].set_xticks(lead_times)
    axs[0].set_xticklabels(xlabels, fontsize=16)
    axs[0].legend(loc='upper left', fontsize=15)


    axs[1].plot(lead_times0, rmses_persistence.t.values, label='Persistence', linestyle='--')
    axs[1].plot(lead_times0, [rmses_climatology.t.values]*len(lead_times0), label='Global climatology', linestyle='--')
    axs[1].plot(lead_times0, [rmses_weekly_clim.t.values]*len(lead_times0), label='Weekly climatology', linestyle='--')
    axs[1].plot(lead_times0, rmses_ifs.t.values, label='Operational IFS', linestyle='--')
    #axs[1].plot(lead_times0, rmses_ifs_t42.t.values, label='IFS T42', linestyle='--')
    #axs[1].plot(rmses_ifs_t63.lead_time.values, rmses_ifs_t63.t.values, label='IFS T63', linestyle='--')
    axs[1].scatter([72, 120], rmses_rasp_direct.t.values, label='Rasp 2020 (direct)', color='maroon')
    #axs[1].plot(lead_times0, rmses_rasp_iter.t.values, label='Rasp 2020 (iter)', linestyle='-')
    axs[1].plot(lead_times0, rmses_weyn.t.values[:len(lead_times0)], label='Weyn 2020', linestyle='-')
    axs[1].plot(lead_times, rmses_spherical.t.values, label='Ours', color='black', marker='o')


    axs[1].set_ylabel('RMSE [K]', fontsize=18)
    axs[1].set_xlabel('Forecast time [h]', fontsize=18)
    axs[1].set_title('T850', fontsize=22)
    axs[1].set_xticks(lead_times)
    axs[1].set_xticklabels(xlabels, fontsize=16)
    axs[1].tick_params(axis='both', which='major', labelsize=16)
    axs[1].legend(loc='upper left', fontsize=15)
    
    
    filename = model_description + '_benchmark.png'
    
    plt.tight_layout()
    plt.savefig(output_dir + filename, bbox_inches='tight')

    plt.show()
    
def plot_benchmark_simple(rmses_spherical, model_description, lead_times, input_dir, output_dir, \
                          title=True, filename=None, names=[]):
    """
    Plot rmse of different models vs Weyn et al

    :param rmses_spherical: list of xarrays or xarray
    :param model_description: string
    :param lead_times: xarray
    :param input_dir: string
    :param output_dir: string
    :param title: boolean
    :param filename: string
    :param names: if rmses_spherical is a list, names should be a list of same length with the name of each model
    :return:
    """
    
    lead_times0 = np.arange(6, lead_times[-1]+6, 6)
    colors = cm.Spectral(np.linspace(0,1,len(rmses_spherical)))
    
    xlabels = [str(t) if t%4 == 0 else '' for t in lead_times] if lead_times[0] < 12 else lead_times
    
    # RMSE baselines
    
    rmses_weyn = xr.open_dataset(input_dir + 'rmses_weyn.nc').rename({'z500':'z', 't850':'t'})
    
    f, axs = plt.subplots(1, 2, figsize=(17, 6))
    if title:
        f.suptitle('RMSE between forecast and observation as a function of forecast time', fontsize=24, y=1.07)

    axs[0].plot(lead_times0, rmses_weyn.z.values[:len(lead_times0)], label='Weyn 2020', linestyle='-')
    axs[1].plot(lead_times0, rmses_weyn.t.values[:len(lead_times0)], label='Weyn 2020', linestyle='-')

    if len(names) == 0:
        axs[0].plot(lead_times, rmses_spherical.z.values, label='Ours', color='black', marker='o')
        axs[1].plot(lead_times, rmses_spherical.t.values, label='Ours', color='black', marker='o')
    else:
        for rmse, name, c in zip(rmses_spherical, names, colors):
            axs[0].plot(lead_times, rmse.z.values, label=name, color=c, marker='o')
            axs[1].plot(lead_times, rmse.t.values, label=name, color=c, marker='o')

    axs[0].set_ylabel('RMSE [$m^2 s^{−2}$]', fontsize=18)
    axs[0].set_xlabel('Forecast time [h]', fontsize=18)
    axs[0].set_title('Z500', fontsize=22)
    axs[0].tick_params(axis='both', which='major', labelsize=16)
    axs[0].set_xticks(lead_times)
    axs[0].set_xticklabels(xlabels, fontsize=16)
    axs[0].legend(loc='upper left', fontsize=15)

    axs[1].set_ylabel('RMSE [K]', fontsize=18)
    axs[1].set_xlabel('Forecast time [h]', fontsize=18)
    axs[1].set_title('T850', fontsize=22)
    axs[1].set_xticks(lead_times)
    axs[1].set_xticklabels(xlabels, fontsize=16)
    axs[1].tick_params(axis='both', which='major', labelsize=16)
    axs[1].legend(loc='upper left', fontsize=15)


    if not filename:
        filename = model_description + '_benchmark.png'
    
    plt.tight_layout()
    plt.savefig(output_dir + filename, bbox_inches='tight')

    plt.show()


def plot_benchmark_MAE(rmses_spherical, model_description, lead_times, input_dir, output_dir, title=True):
    
    lead_times0 = np.arange(6, lead_times[-1]+6, 6)
    
    xlabels = [str(t) if t%4 == 0 else '' for t in lead_times] if lead_times[0] < 12 else lead_times
    
    # RMSE baselines
    rmses_baselines = pickle.load(open(input_dir+'mae.pkl', 'rb'))
    
    rmses_rasp_direct = rmses_baselines['CNN (direct)']
    rmses_rasp_iter = rmses_baselines['CNN (iterative)']
    rmses_climatology = rmses_baselines['Climatology']
    rmses_weekly_clim = rmses_baselines['Weekly clim.']
    rmses_persistence = rmses_baselines['Persistence']
    rmses_ifs = rmses_baselines['Operational'].sel(lead_time=lead_times)
    rmses_ifs_t42 = rmses_baselines['IFS T42'].sel(lead_time=lead_times)
    rmses_ifs_t63 = rmses_baselines['IFS T63'].sel(lead_time=slice(lead_times[0], lead_times[-1]))
    rmses_weyn = xr.open_dataset(input_dir + 'rmses_weyn.nc')
    
    f, axs = plt.subplots(1, 2, figsize=(17, 6))
    if title:
        f.suptitle('RMSE between forecast and observation as a function of forecast time', fontsize=24, y=1.07)

    axs[0].plot(lead_times0, rmses_persistence.z.values, label='Persistence', linestyle='--')
    axs[0].plot(lead_times0, [rmses_climatology.z.values]*len(lead_times0), label='Global climatology', linestyle='--')
    axs[0].plot(lead_times0, [rmses_weekly_clim.z.values]*len(lead_times0), label='Weekly climatology', linestyle='--')
    axs[0].plot(lead_times0, rmses_ifs.z.values, label='Operational IFS', linestyle='--')
    #axs[0].plot(lead_times0, rmses_ifs_t42.z.values, label='IFS T42', linestyle='--')
    #axs[0].plot(rmses_ifs_t63.lead_time.values, rmses_ifs_t63.z.values, label='IFS T63', linestyle='--')
    axs[0].scatter([72, 120], rmses_rasp_direct.z.values, label='Rasp 2020 (direct)', color='maroon')
    axs[0].plot(lead_times0, rmses_rasp_iter.z.values, label='Rasp 2020 (iter)', linestyle='-')
    #axs[0].plot(lead_times0, rmses_weyn.z.values, label='Weyn 2020', linestyle='-')
    axs[0].plot(lead_times, rmses_spherical.z.values, label='Ours', color='black', marker='o')

    axs[0].set_ylabel('MAE [$m^2 s^{-2}$]', fontsize=18)
    axs[0].set_xlabel('Forecast time [h]', fontsize=18)
    axs[0].set_title('Z500', fontsize=22)
    axs[0].tick_params(axis='both', which='major', labelsize=16)
    axs[0].set_xticks(lead_times)
    axs[0].set_xticklabels(xlabels, fontsize=16)
    axs[0].legend(loc='upper left', fontsize=15)


    axs[1].plot(lead_times0, rmses_persistence.t.values, label='Persistence', linestyle='--')
    axs[1].plot(lead_times0, [rmses_climatology.t.values]*len(lead_times0), label='Global climatology', linestyle='--')
    axs[1].plot(lead_times0, [rmses_weekly_clim.t.values]*len(lead_times0), label='Weekly climatology', linestyle='--')
    axs[1].plot(lead_times0, rmses_ifs.t.values, label='Operational IFS', linestyle='--')
    #axs[1].plot(lead_times0, rmses_ifs_t42.t.values, label='IFS T42', linestyle='--')
    #axs[1].plot(rmses_ifs_t63.lead_time.values, rmses_ifs_t63.t.values, label='IFS T63', linestyle='--')
    axs[1].scatter([72, 120], rmses_rasp_direct.t.values, label='Rasp 2020 (direct)', color='maroon')
    axs[1].plot(lead_times0, rmses_rasp_iter.t.values, label='Rasp 2020 (iter)', linestyle='-')
    #axs[1].plot(lead_times0, rmses_weyn.t.values, label='Weyn 2020', linestyle='-')
    axs[1].plot(lead_times, rmses_spherical.t.values, label='Ours', color='black', marker='o')


    axs[1].set_ylabel('MAE [K]', fontsize=18)
    axs[1].set_xlabel('Forecast time [h]', fontsize=18)
    axs[1].set_title('T850', fontsize=22)
    axs[1].set_xticks(lead_times)
    axs[1].set_xticklabels(xlabels, fontsize=16)
    axs[1].tick_params(axis='both', which='major', labelsize=16)
    axs[1].legend(loc='upper left', fontsize=15)
    
    
    filename = model_description + '_mae_benchmark.png'
    
    plt.tight_layout()
    plt.savefig(output_dir + filename, bbox_inches='tight')

    plt.show()
    
    
def plot_benchmark_ACC(rmses_spherical, model_description, lead_times, input_dir, output_dir, title=True):
    
    lead_times0 = np.arange(6, lead_times[-1]+6, 6)
    
    xlabels = [str(t) if t%4 == 0 else '' for t in lead_times] if lead_times[0] < 12 else lead_times
    
    # RMSE baselines
    rmses_baselines = pickle.load(open(input_dir+'acc.pkl', 'rb'))
    
    rmses_rasp_direct = rmses_baselines['CNN (direct)']
    rmses_rasp_iter = rmses_baselines['CNN (iterative)']
    rmses_climatology = rmses_baselines['Climatology']
    rmses_weekly_clim = rmses_baselines['Weekly clim.']
    rmses_persistence = rmses_baselines['Persistence']
    rmses_ifs = rmses_baselines['Operational'].sel(lead_time=lead_times)
    rmses_ifs_t42 = rmses_baselines['IFS T42'].sel(lead_time=lead_times)
    rmses_ifs_t63 = rmses_baselines['IFS T63'].sel(lead_time=slice(lead_times[0], lead_times[-1]))
    rmses_weyn = xr.open_dataset(input_dir + 'rmses_weyn.nc')
    
    f, axs = plt.subplots(1, 2, figsize=(17, 6), sharey=True)
    if title:
        f.suptitle('RMSE between forecast and observation as a function of forecast time', fontsize=24, y=1.07)

    axs[0].plot(lead_times0, rmses_persistence.z.values, label='Persistence', linestyle='--')
    axs[0].plot(lead_times0, [rmses_climatology.z.values]*len(lead_times0), label='Global climatology', linestyle='--')
    axs[0].plot(lead_times0, [rmses_weekly_clim.z.values]*len(lead_times0), label='Weekly climatology', linestyle='--')
    axs[0].plot(lead_times0, rmses_ifs.z.values, label='Operational IFS', linestyle='--')
    #axs[0].plot(lead_times0, rmses_ifs_t42.z.values, label='IFS T42', linestyle='--')
    #axs[0].plot(rmses_ifs_t63.lead_time.values, rmses_ifs_t63.z.values, label='IFS T63', linestyle='--')
    axs[0].scatter([72, 120], rmses_rasp_direct.z.values, label='Rasp 2020 (direct)', color='maroon')
    axs[0].plot(lead_times0, rmses_rasp_iter.z.values, label='Rasp 2020 (iter)', linestyle='-')
    #axs[0].plot(lead_times0, rmses_weyn.z.values, label='Weyn 2020', linestyle='-')
    axs[0].plot(lead_times, rmses_spherical.z.values, label='Ours', color='black', marker='o')

    axs[0].set_ylabel('ACC', fontsize=18)
    axs[0].set_xlabel('Forecast time [h]', fontsize=18)
    axs[0].set_title('Z500', fontsize=22)
    axs[0].tick_params(axis='both', which='major', labelsize=16)
    axs[0].set_xticks(lead_times)
    axs[0].set_xticklabels(xlabels, fontsize=16)
    axs[0].legend(loc='lower left', fontsize=15)


    axs[1].plot(lead_times0, rmses_persistence.t.values, label='Persistence', linestyle='--')
    axs[1].plot(lead_times0, [rmses_climatology.t.values]*len(lead_times0), label='Global climatology', linestyle='--')
    axs[1].plot(lead_times0, [rmses_weekly_clim.t.values]*len(lead_times0), label='Weekly climatology', linestyle='--')
    axs[1].plot(lead_times0, rmses_ifs.t.values, label='Operational IFS', linestyle='--')
    #axs[1].plot(lead_times0, rmses_ifs_t42.t.values, label='IFS T42', linestyle='--')
    #axs[1].plot(rmses_ifs_t63.lead_time.values, rmses_ifs_t63.t.values, label='IFS T63', linestyle='--')
    axs[1].scatter([72, 120], rmses_rasp_direct.t.values, label='Rasp 2020 (direct)', color='maroon')
    axs[1].plot(lead_times0, rmses_rasp_iter.t.values, label='Rasp 2020 (iter)', linestyle='-')
    #axs[1].plot(lead_times0, rmses_weyn.t.values, label='Weyn 2020', linestyle='-')
    axs[1].plot(lead_times, rmses_spherical.t.values, label='Ours', color='black', marker='o')


    axs[1].set_xlabel('Forecast time [h]', fontsize=18)
    axs[1].set_title('T850', fontsize=22)
    axs[1].set_xticks(lead_times)
    axs[1].set_xticklabels(xlabels, fontsize=16)
    axs[1].tick_params(axis='both', which='major', labelsize=16)
    axs[1].legend(loc='lower left', fontsize=15)
    
    
    filename = model_description + '_benchmark.png'
    
    plt.tight_layout()
    plt.savefig(output_dir + filename, bbox_inches='tight')

    plt.show()
    
def plot_general_skills(rmse_map, corr_map, rbias_map, rsd_map, model_description, lead_times, 
                        output_dir,  relrmse_ylim=[0, 0.05], relbias_ylim=[-0.05, 0.05], 
                        rsd_ylim=[0, 2], r2_ylim=[0, 1], title=True):
    
    n_ticks = len(lead_times)
    
    xlabels = [str(t) if t%4 == 0 else '' for t in lead_times] if lead_times[0] < 12 else lead_times
    
    rmse = rmse_map.mean('node').compute()
    corr = corr_map.mean('node').compute()
    rbias = rbias_map.mean('node').compute()
    rsd = rsd_map.mean('node').compute()
    
    
    f, axs = plt.subplots(4, 2, figsize=(17, 20), sharex=True, sharey='row')
    
    if title:
        f.suptitle('Skill boxplots', fontsize=24, y=1.05)
    else:
        f.suptitle('   ', fontsize=24, y=1.05)

    
    cols = ['Z500', 'T850']
    rows = ['relRMSE', 'relBIAS', 'rSD', 'R2']
    
    

    colors = ['red', 'orange', 'black']
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
    labels = ['Mean', 'Median', '25% and 75% quartiles']

    for ax, col in zip(axs[0, :], cols):
        ax.set_title(col, fontsize=22, y=1.08)

    for ax, row in zip(axs[:, 0], rows):
        ax.set_ylabel(row, fontsize=20)
    
    # relRMSE
    rmsesbox_z = [rmse_map.z.values[i, :] for i in range(len(rmse_map.z.values))]
    axs[0, 0].boxplot(rmsesbox_z)
    axs[0, 0].plot(np.arange(1, n_ticks+1, 1), rmse.z.values, color='red')
    axs[0, 0].set_ylim(relrmse_ylim)
    axs[0, 0].tick_params(axis='y', labelsize=16)
    
    rmsesbox_t = [rmse_map.t.values[i, :] for i in range(len(rmse_map.t.values))]
    axs[0, 1].boxplot(rmsesbox_t)
    axs[0, 1].plot(np.arange(1, n_ticks+1, 1), rmse.t.values, color='red')
    

    # relBIAS
    rbiasbox_z = [rbias_map.z.values[i, :] for i in range(len(rbias_map.z.values))]
    axs[1, 0].boxplot(rbiasbox_z)
    axs[1, 0].plot(np.arange(1, n_ticks+1, 1), rbias.z.values, color='red')
    axs[1, 0].set_ylim(relbias_ylim)
    axs[1, 0].tick_params(axis='y', labelsize=16)

    rbiasbox_t = [rbias_map.t.values[i, :] for i in range(len(rbias_map.t.values))]
    axs[1, 1].boxplot(rbiasbox_t)
    axs[1, 1].plot(np.arange(1, n_ticks+1, 1), rbias.t.values, color='red')
    
    
    # rSD
    rsdbox_z = [rsd_map.z.values[i, :] for i in range(len(rsd_map.z.values))]
    axs[2, 0].boxplot(rsdbox_z)
    axs[2, 0].plot(np.arange(1, n_ticks+1, 1), rsd.z.values, color='red')
    axs[2, 0].set_ylim(rsd_ylim)
    axs[2, 0].tick_params(axis='y', labelsize=16)

    rsdbox_t = [rsd_map.t.values[i, :] for i in range(len(rsd_map.t.values))]
    axs[2, 1].boxplot(rsdbox_t)
    axs[2, 1].plot(np.arange(1, n_ticks+1, 1), rsd.t.values, color='red')
    
    
    # R2
    r2box_z = [corr_map.z.values[i, :] for i in range(len(corr_map.z.values))]
    axs[3, 0].boxplot(r2box_z)
    axs[3, 0].plot(np.arange(1, n_ticks+1, 1), corr.z.values, color='red')
    axs[3, 0].set_ylim(r2_ylim)
    axs[3, 0].tick_params(axis='y', labelsize=16)
    
    r2box_t = [corr_map.t.values[i, :] for i in range(len(corr_map.t.values))]
    axs[3, 1].boxplot(r2box_t)
    axs[3, 1].plot(np.arange(1, n_ticks+1, 1), corr.t.values, color='red')


    f.legend(lines, labels, loc=[0.03, 0.94], fontsize=18)
    
    
    axs[3, 0].set_xlabel('Forecast time [hours]', fontsize=20)
    axs[3, 1].set_xlabel('Forecast time [hours]', fontsize=20)
    axs[3, 0].set_xticklabels(xlabels, fontsize=16)
    axs[3, 1].set_xticklabels(xlabels, fontsize=16)
    
    plt.tight_layout()

    filename = model_description + '_general_skills.png'
    plt.savefig(output_dir + filename, bbox_inches='tight')
    
    plt.show()
    

def plot_skillmaps(rmse_map, rsd_map, rbias_map, corr_map, model_description, lead_times, resolution, 
                   output_dir):
    for i, lead in enumerate(lead_times):  
        
        rmse_min = 0
        rmse_max = 0.04
        
        rsd_min = 0.2
        rsd_max = 1.8

        rbias_min = -0.025
        rbias_max = 0.025

        corr_min = 0
        corr_max = 1

        rmse_equi = hp_to_equiangular(rmse_map.isel(lead_time=i), resolution)
        rsd_equi = hp_to_equiangular(rsd_map.isel(lead_time=i), resolution)
        rbias_equi = hp_to_equiangular(rbias_map.isel(lead_time=i), resolution)
        corr_equi = hp_to_equiangular(corr_map.isel(lead_time=i), resolution)

        proj = ccrs.PlateCarree()

        f, axs = plt.subplots(4, 2, figsize=(15, 15), subplot_kw=dict(projection=proj))
        f.suptitle('Skillmaps between forecast and observation, lead time: {}h'.format(lead), 
                   fontsize=26, y=1.05, x=0.45)
        
        cols = ['Z500', 'T850']

        for ax, col in zip(axs[0, :], cols):
            ax.set_title(col, fontsize=24, y=1.08)
        
        plot_signal(f, sample=rmse_equi, var='z', vmin=rmse_min, vmax=rmse_max, proj=proj, ax=axs[0, 0], 
                    cmap='Reds', colorbar=False, cbar_label='', extend='max')
        plot_signal(f, sample=rmse_equi, var='t', vmin=rmse_min, vmax=rmse_max, proj=proj, ax=axs[0, 1], 
                    cmap='Reds', colorbar=True, cbar_label='relRMSE', extend='max')

        plot_signal(f, sample=rbias_equi, var='z', vmin=rbias_min, vmax=rbias_max, proj=proj, ax=axs[1, 0], 
                    cmap='RdBu_r', colorbar=False, cbar_label='', extend='both')
        plot_signal(f, sample=rbias_equi, var='t', vmin=rbias_min, vmax=rbias_max, proj=proj, ax=axs[1, 1], 
                    cmap='RdBu_r', colorbar=True, cbar_label='relBIAS', extend='both')

        plot_signal(f, sample=rsd_equi, var='z', vmin=rsd_min, vmax=rsd_max, proj=proj, ax=axs[2, 0], 
                    cmap='PuOr_r', colorbar=False, cbar_label='', extend='both')
        plot_signal(f, sample=rsd_equi, var='t', vmin=rsd_min, vmax=rsd_max, proj=proj, ax=axs[2, 1], 
                    cmap='PuOr_r', colorbar=True, cbar_label='rSD', extend='both')

        plot_signal(f, sample=corr_equi, var='z', vmin=corr_min, vmax=corr_max, proj=proj, ax=axs[3, 0], 
                    cmap='Greens', colorbar=False, cbar_label='', extend='neither')
        plot_signal(f, sample=corr_equi, var='t', vmin=corr_min, vmax=corr_max, proj=proj, ax=axs[3, 1], 
                    cmap='Greens', colorbar=True, cbar_label='R2', extend='neither')


        f.tight_layout(pad=-2)
        filename = model_description + '_' + str(i) + '_maps.png'
        plt.savefig(output_dir + filename, bbox_inches='tight')

        plt.show()
       
'''   
def plot_general_skills(rmse, corr, rbias, rsd, model_description, lead_times, 
                        input_dir, output_dir):
    
    f, axs = plt.subplots(4, 2, figsize=(17, 18), sharex=True)
    f.suptitle('Skills between forecast and observation as a function of forecast time', fontsize=20, y=0.93)

    # RMSE
    rmses_weyn = xr.open_dataset(input_dir + 'rmses_weyn.nc')
    rmses_rasp = xr.open_dataset(input_dir + 'rmses_rasp.nc')
    rmses_climatology = xr.open_dataset(input_dir + 'rmses_clim.nc')
    rmses_weekly_clim = xr.open_dataset(input_dir + 'rmses_weekly_clim.nc')
    rmses_persistence = xr.open_dataset(input_dir + 'rmses_persistence.nc')
    rmses_tigge = xr.open_dataset(input_dir + 'rmses_tigge.nc')


    axs[0, 0].plot(lead_times, rmses_persistence.z.values, label='Persistence', linestyle='--')
    axs[0, 0].plot(lead_times, rmses_climatology.z.values, label='Global climatology', linestyle='--')
    axs[0, 0].plot(lead_times, rmses_weekly_clim.z.values, label='Weekly climatology', linestyle='--')
    axs[0, 0].plot(lead_times, rmses_tigge.z.values, label='Operational IFS', linestyle='--')
    axs[0, 0].plot(lead_times, rmses_weyn.z.values, label='Weyn 2020', linestyle='--')
    axs[0, 0].scatter([72, 120], rmses_rasp.z.values, label='Rasp 2020', marker='_', linewidth=5, s=60, color='maroon')
    axs[0, 0].plot(lead_times, rmse.z.values, label='Spherical', color='black')

    axs[0, 0].set_ylabel('RMSE [m2 s−2]', fontsize=16)
    axs[0, 0].set_title('RMSE Z500', fontsize=18)
    axs[0, 0].tick_params(axis='both', which='major', labelsize=14)
    axs[0, 0].set_xticks(np.arange(24, 120+24, 24))
    axs[0, 0].set_xticklabels((1, 2, 3, 4, 5))
    axs[0, 0].legend(loc='upper left', fontsize=14)


    axs[0, 1].plot(lead_times, rmses_persistence.t.values, label='Persistence', linestyle='--')
    axs[0, 1].plot(lead_times, rmses_climatology.t.values, label='Global climatology', linestyle='--')
    axs[0, 1].plot(lead_times, rmses_weekly_clim.t.values, label='Weekly climatology', linestyle='--')
    axs[0, 1].plot(lead_times, rmses_tigge.t.values, label='Operational IFS', linestyle='--')
    axs[0, 1].plot(lead_times, rmses_weyn.t.values, label='Weyn et al., 2020', linestyle='--')
    axs[0, 1].scatter([72, 120], rmses_rasp.t.values, label='Rasp et al., 2020', marker='_', linewidth=5, s=60, color='maroon')
    axs[0, 1].plot(lead_times, rmse.t.values, label='Spherical', color='black')


    axs[0, 1].set_ylabel('RMSE [K]', fontsize=16)
    axs[0, 1].set_title('RMSE T850', fontsize=18)
    axs[0, 1].set_xticks(np.arange(24, 120+24, 24))
    axs[0, 1].set_xticklabels((1, 2, 3, 4, 5))
    axs[0, 1].tick_params(axis='both', which='major', labelsize=14)
    axs[0, 1].legend(loc='upper left', fontsize=14)

    # R2
    axs[1, 0].plot(lead_times, corr.z.values, label='Spherical', color='black')

    axs[1, 0].set_ylabel('R2', fontsize=16)
    axs[1, 0].set_title('R2 Z500', fontsize=18)
    axs[1, 0].tick_params(axis='both', which='major', labelsize=14)
    axs[1, 0].set_xticks(np.arange(24, 120+24, 24))
    axs[1, 0].set_xticklabels((1, 2, 3, 4, 5))
    axs[1, 0].legend(loc='upper right', fontsize=14)
    axs[1, 0].set_ylim([0, 1])

    axs[1, 1].plot(lead_times, corr.t.values, label='Spherical', color='black')

    axs[1, 1].set_ylabel('R2', fontsize=16)
    axs[1, 1].set_title('R2 T850', fontsize=18)
    axs[1, 1].set_xticks(np.arange(24, 120+24, 24))
    axs[1, 1].set_xticklabels((1, 2, 3, 4, 5))
    axs[1, 1].tick_params(axis='both', which='major', labelsize=14)
    axs[1, 1].legend(loc='upper right', fontsize=14)
    axs[1, 1].set_ylim([0, 1])

    # relBIAS
    axs[2, 0].plot(lead_times, rbias.z.values, label='Spherical', color='black')

    axs[2, 0].set_ylabel('relBIAS', fontsize=16)
    axs[2, 0].set_title('relBIAS Z500', fontsize=18)
    axs[2, 0].tick_params(axis='both', which='major', labelsize=14)
    axs[2, 0].set_xticks(np.arange(24, 120+24, 24))
    axs[2, 0].set_xticklabels((1, 2, 3, 4, 5))
    axs[2, 0].legend(loc='upper right', fontsize=14)
    axs[2, 0].set_ylim([-0.0005, 0.0005])

    axs[2, 1].plot(lead_times, rbias.t.values, label='Spherical', color='black')

    axs[2, 1].set_ylabel('relBIAS', fontsize=16)
    axs[2, 1].set_title('relBIAS T850', fontsize=18)
    axs[2, 1].set_xticks(np.arange(24, 120+24, 24))
    axs[2, 1].set_xticklabels((1, 2, 3, 4, 5))
    axs[2, 1].tick_params(axis='both', which='major', labelsize=14)
    axs[2, 1].legend(loc='upper right', fontsize=14)
    axs[2, 1].set_ylim([-0.005, 0.005])

    # rSD
    axs[3, 0].plot(lead_times, rsd.z.values, label='Spherical', color='black')

    axs[3, 0].set_ylabel('rSD', fontsize=16)
    axs[3, 0].set_title('rSD Z500', fontsize=18)
    axs[3, 0].tick_params(axis='both', which='major', labelsize=14)
    axs[3, 0].set_xticks(np.arange(24, 120+24, 24))
    axs[3, 0].set_xticklabels((1, 2, 3, 4, 5))
    axs[3, 0].legend(loc='upper right', fontsize=14)
    axs[3, 0].set_ylim([0.8, 1.2])
    axs[3, 0].set_xlabel('Forecast time [days]', fontsize=16)

    axs[3, 1].plot(lead_times, rsd.t.values, label='Spherical', color='black')

    axs[3, 1].set_ylabel('rSD', fontsize=16)
    axs[3, 1].set_title('rSD T850', fontsize=18)
    axs[3, 1].set_xticks(np.arange(24, 120+24, 24))
    axs[3, 1].set_xticklabels((1, 2, 3, 4, 5))
    axs[3, 1].tick_params(axis='both', which='major', labelsize=14)
    axs[3, 1].legend(loc='upper right', fontsize=14)
    axs[3, 1].set_ylim([0.8, 1.2])
    axs[3, 1].set_xlabel('Forecast time [days]', fontsize=16)

    filename = model_description + '_general_skills.png'
    plt.savefig(output_dir + filename, bbox_inches='tight')

    plt.show()


def plot_general_skills_boxplot(rmse_map, corr_map, rbias_map, rsd_map, model_description, lead_times, 
                                input_dir, output_dir):    
    f, axs = plt.subplots(4, 2, figsize=(17, 18), sharex=True)
    f.suptitle('Skill boxplots', fontsize=20, y=0.93)

    # RMSE

    rmsesbox_z = [rmse_map.z.values[i, :] for i in range(len(rmse_map.z.values))]
    axs[0, 0].boxplot(rmsesbox_z)

    axs[0, 0].set_ylabel('RMSE [m2 s−2]', fontsize=16)
    axs[0, 0].set_title('RMSE Z500', fontsize=18)
    axs[0, 0].tick_params(axis='both', which='major', labelsize=14)
    axs[0, 0].set_xticklabels(lead_times)
    axs[0, 1].set_ylim([0, 2000])


    rmsesbox_t = [rmse_map.t.values[i, :] for i in range(len(rmse_map.t.values))]
    axs[0, 1].boxplot(rmsesbox_t)

    axs[0, 1].set_ylabel('RMSE [K]', fontsize=16)
    axs[0, 1].set_title('RMSE T850', fontsize=18)
    axs[0, 1].set_xticklabels(lead_times)
    axs[0, 1].tick_params(axis='both', which='major', labelsize=14)
    axs[0, 1].set_ylim([0, 10])

    # R2
    r2box_z = [corr_map.z.values[i, :] for i in range(len(corr_map.z.values))]
    axs[1, 0].boxplot(r2box_z)

    axs[1, 0].set_ylabel('R2', fontsize=16)
    axs[1, 0].set_title('R2 Z500', fontsize=18)
    axs[1, 0].tick_params(axis='both', which='major', labelsize=14)
    axs[1, 0].set_xticklabels(lead_times)
    axs[1, 0].set_ylim([0, 1])

    r2box_t = [corr_map.t.values[i, :] for i in range(len(corr_map.t.values))]
    axs[1, 1].boxplot(r2box_t)

    axs[1, 1].set_ylabel('R2', fontsize=16)
    axs[1, 1].set_title('R2 T850', fontsize=18)
    axs[1, 1].set_xticklabels(lead_times)
    axs[1, 1].tick_params(axis='both', which='major', labelsize=14)
    axs[1, 1].set_ylim([0, 1])

    # relBIAS
    rbiasbox_z = [rbias_map.z.values[i, :] for i in range(len(rbias_map.z.values))]
    axs[2, 0].boxplot(rbiasbox_z)

    axs[2, 0].set_ylabel('relBIAS', fontsize=16)
    axs[2, 0].set_title('relBIAS Z500', fontsize=18)
    axs[2, 0].tick_params(axis='both', which='major', labelsize=14)
    axs[2, 0].set_xticklabels(lead_times)
    axs[2, 0].set_ylim([-0.02, 0.02])

    rbiasbox_t = [rbias_map.t.values[i, :] for i in range(len(rbias_map.t.values))]
    axs[2, 1].boxplot(rbiasbox_t)

    axs[2, 1].set_ylabel('relBIAS', fontsize=16)
    axs[2, 1].set_title('relBIAS T850', fontsize=18)
    axs[2, 1].set_xticklabels(lead_times)
    axs[2, 1].tick_params(axis='both', which='major', labelsize=14)
    axs[2, 1].set_ylim([-0.02, 0.02])

    # rSD
    rsdbox_z = [rsd_map.z.values[i, :] for i in range(len(rsd_map.z.values))]
    axs[3, 0].boxplot(rsdbox_z)

    axs[3, 0].set_ylabel('rSD', fontsize=16)
    axs[3, 0].set_title('rSD Z500', fontsize=18)
    axs[3, 0].tick_params(axis='both', which='major', labelsize=14)
    axs[3, 0].set_xticklabels(lead_times)
    axs[3, 0].set_ylim([0.7, 1.3])
    axs[3, 0].set_xlabel('Forecast time [hours]', fontsize=16)

    rsdbox_t = [rsd_map.t.values[i, :] for i in range(len(rsd_map.t.values))]
    axs[3, 1].boxplot(rsdbox_t)

    axs[3, 1].set_ylabel('rSD', fontsize=16)
    axs[3, 1].set_title('rSD T850', fontsize=18)
    axs[3, 1].set_xticklabels(lead_times)
    axs[3, 1].tick_params(axis='both', which='major', labelsize=14)
    axs[3, 1].set_ylim([0.25, 1.75])
    axs[3, 1].set_xlabel('Forecast time [days]', fontsize=16)

    filename = model_description + '_general_skills_boxplot.png'
    plt.savefig(output_dir + filename, bbox_inches='tight')

    plt.show()


def plot_skillmaps(rsd_map, rbias_map, corr_map, model_description, lead_times, resolution, output_dir):
    for i, lead in enumerate(lead_times):    
        rsd_min = 0.35
        rsd_max = 1.65

        rbias_min = -0.015
        rbias_max = 0.015

        corr_min = 0
        corr_max = 1

        rsd_equi = hp_to_equiangular(rsd_map.isel(lead_time=i), resolution)
        rbias_equi = hp_to_equiangular(rbias_map.isel(lead_time=i), resolution)
        corr_equi = hp_to_equiangular(corr_map.isel(lead_time=i), resolution)

        proj = ccrs.PlateCarree()

        f, axs = plt.subplots(3, 2, figsize=(17, 15), subplot_kw=dict(projection=proj))
        f.suptitle('Skillmaps between forecast and observation, lead time: {}h'.format(lead), fontsize=26, y=1.02)

        plot_signal(f, sample=rbias_equi, var='z', vmin=rbias_min, vmax=rbias_max, proj=proj, ax=axs[0, 0], 
                    cmap='RdBu_r')
        plot_signal(f, sample=rbias_equi, var='t', vmin=rbias_min, vmax=rbias_max, proj=proj, ax=axs[0, 1], 
                    cmap='RdBu_r')

        plot_signal(f, sample=rsd_equi, var='z', vmin=rsd_min, vmax=rsd_max, proj=proj, ax=axs[1, 0], 
                    cmap='PuOr_r')
        plot_signal(f, sample=rsd_equi, var='t', vmin=rsd_min, vmax=rsd_max, proj=proj, ax=axs[1, 1], 
                    cmap='PuOr_r')

        plot_signal(f, sample=corr_equi, var='z', vmin=corr_min, vmax=corr_max, proj=proj, ax=axs[2, 0], 
                    cmap='Greens')
        plot_signal(f, sample=corr_equi, var='t', vmin=corr_min, vmax=corr_max, proj=proj, ax=axs[2, 1], 
                    cmap='Greens')

        fontsize=24


        axs[0, 0].set_title('relBIAS Z500', fontsize=fontsize)
        axs[0, 1].set_title('relBIAS T850', fontsize=fontsize)

        axs[1, 0].set_title('rSD Z500', fontsize=fontsize)
        axs[1, 1].set_title('rSD T850', fontsize=fontsize)

        axs[2, 0].set_title('R2 Z500', fontsize=fontsize)
        axs[2, 1].set_title('R2 T850', fontsize=fontsize)

        f.tight_layout(pad=-2)
        filename = model_description + '_' + str(i) + '_maps.png'
        plt.savefig(output_dir + filename, bbox_inches='tight')

        plt.show()
'''  
    
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
    model_description : str
        Description of the model
    save_path : str
        Path to save the figure
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