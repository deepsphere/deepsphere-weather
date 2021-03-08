#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 21:39:12 2021

@author: ghiggi
"""
import os
import glob
import tempfile
import shutil
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
import xarray as xr
import pygsp as pg
import numpy as np
from PIL import Image 
import modules.xsphere as xsphere

# TODO: ylabels add unit
# 'RMSE [$m^2 s^{−2}$]'

def get_var_cmap(var, arg): 
    cmap_dict = {'t850': {'state': plt.get_cmap('RdYlBu_r'),
                          'error': plt.get_cmap('RdBu')
                          },
                 'z500': {'state': plt.get_cmap('BuPu'),
                          'error': plt.get_cmap('RdBu')
                          },      
                 'q500': {'state': plt.get_cmap('GnBu'),
                          'error': plt.get_cmap('RdBu'),       
                          }
                 }
    if var in list(cmap_dict.keys()):
        if arg in list(cmap_dict[var].keys()):
            cmap = cmap_dict[var][arg]
        else: 
            cmap = plt.get_cmap('RdYlBu')
    else: 
        cmap = plt.get_cmap('RdYlBu')
    return cmap

def get_var_clim(var, arg): 
    clim_dict = {'t850': {'state': (225, 310),
                          'error': (-10, 10)
                          },
                 'z500': {'state': (44000, 59000),
                          'error': (-5000, 5000)
                          },        
                 }
    if var in list(clim_dict.keys()):
        if arg in list(clim_dict[var].keys()):
            clim = clim_dict[var][arg]
        else: 
            clim = (None, None)
    else: 
        clim = (None, None)
    return clim

##----------------------------------------------------------------------------.
def get_global_ylim(skill, var): 
    ylim_dict = {'RMSE': {'z500': (100, 700),
                          't850': (0.5, 4)},
                 'rSD':  {'z500': (0.9, 1.1),
                          't850': (0.9, 1.1)},
                 'relBIAS': {'z500': (-0.002, 0.002),
                             't850': (-0.002, 0.002)},
                 'pearson_R2': {'z500': (0.5, 1),
                                't850': (0.5, 1)},
                 'KGE': {'z500': (0.4, 1),
                          't850': (0.4, 1)},
                 'NSE': {'z500': (0.4, 1),
                         't850': (0.4, 1)},
                 'percBIAS': {'z500': (-1, 1),
                          't850': (-2, 2)},
                 'percMAE': {'z500': (0, 2),
                          't850': (0, 2)},
                 'error_CoV': {'z500': (-40,40),
                          't850': (-40, 40)},
                 'MAE': {'z500': (50, 700),
                         't850': (0.5, 2.5)},
                 'BIAS': {'z500': (-100, 100),
                          't850': (-1, 1)},
                 'diffSD': {'z500': (-60, 60),
                           't850': (-0.2, 0.2)},
    }
    if skill in list(ylim_dict.keys()):
        if var in list(ylim_dict[skill].keys()):
            ylim = ylim_dict[skill][var]
        else: 
            ylim = (None, None)
    else: 
        ylim = (None, None)
    return ylim

def get_spatial_ylim(skill, var): 
    ylim_dict = {'BIAS': {'z500': (-400, 400),
                          't850': (-4, 4)},
                 'RMSE': {'z500': (0, 2000),
                          't850': (0, 8)},
                 'rSD':  {'z500': (0.6, 1.4),
                          't850': (0.6, 1.4)},
                 'pearson_R2': {'z500': (0, 1),
                                't850': (0, 1)},
                 'KGE': {'z500': (0, 1),
                          't850': (0, 1)},
                 'NSE': {'z500': (0, 1),
                         't850': (0, 1)},

                 'relBIAS': {'z500': (-0.006, 0.006),
                             't850': (-0.01, 0.01)},
                 'percBIAS': {'z500': (-2.5, 2.5),
                              't850': (-2.5, 2.5)},
                 'percMAE': {'z500': (0, 2.5),
                          't850': (0, 2.5)},
                 'error_CoV': {'z500': (-40,40),
                          't850': (-40, 40)},
                 'MAE': {'z500': (50, 700),
                         't850': (0.5, 2.5)},
                
                 'diffSD': {'z500': (-300, 300),
                           't850': (-1,1)},
    }
    if skill in list(ylim_dict.keys()):
        if var in list(ylim_dict[skill].keys()):
            ylim = ylim_dict[skill][var]
        else: 
            ylim = (None, None)
    else: 
        ylim = (None, None)
    return ylim

def get_skill_cbar_extend(skill):
    skill_dict = {"error_CoV": 'both',
                  "obs_CoV": 'both',
                  "pred_CoV": 'both',
                  # Magnitude 
                  "BIAS": 'both',
                  "relBIAS": 'both',
                  "percBIAS": 'both',
                  "MAE": 'max',
                  "relMAE": 'max',
                  "percMAE": 'max',
                  "MSE": 'max',
                  "relMSE": 'max',
                  "RMSE": 'max',
                  "relRMSE": 'max',
                  # Average
                  "rMean": 'both',
                  "diffMean": 'both',
                  # Variability 
                  'rSD': 'both',
                  'diffSD': 'both',
                  "rCoV": 'max',
                  "diffCoV": 'both',
                  # Correlation 
                  "pearson_R": 'neither',
                  "pearson_R2": 'neither',
                  "spearman_R": "neither",
                  "spearman_R2": "neither",
                  "pearson_R2_pvalue": 'neither',
                  "spearman_R2_pvalue": "max",
                  # Overall skills
                  "NSE": 'min',
                  "KGE": 'neither',
                  }
    return skill_dict[skill]

def get_skill_cmap(skill):
    skill_dict = {"error_CoV": plt.get_cmap('RdYlBu'),
                  "obs_CoV": plt.get_cmap('YlOrRd'),
                  "pred_CoV": plt.get_cmap('YlOrRd'),
                  # Magnitude 
                  "BIAS": plt.get_cmap('BrBG'),
                  "relBIAS": plt.get_cmap('BrBG'),
                  "percBIAS": plt.get_cmap('BrBG'),
                  "MAE": plt.get_cmap('Reds'),
                  "relMAE": plt.get_cmap('Reds'),
                  "percMAE": plt.get_cmap('Reds'),
                  "MSE": plt.get_cmap('Reds'),
                  "relMSE": plt.get_cmap('Reds'),
                  "RMSE": plt.get_cmap('Reds'),
                  "relRMSE": plt.get_cmap('Reds'),
                  # Average
                  "rMean": plt.get_cmap('BrBG'),
                  "diffMean": plt.get_cmap('BrBG'),
                  # Variability 
                  'rSD': plt.get_cmap('PRGn'),
                  'diffSD': plt.get_cmap('PRGn'),
                  "rCoV": plt.get_cmap('PRGn'),
                  "diffCoV": plt.get_cmap('PRGn'),
                  # Correlation 
                  "pearson_R": plt.get_cmap('Greens'),
                  "pearson_R2": plt.get_cmap('Greens'),
                  "spearman_R": plt.get_cmap('Greens'),
                  "spearman_R2": plt.get_cmap('Greens'),
                  "pearson_R_pvalue": plt.get_cmap('Purples'),
                  "spearman_R_pvalue": plt.get_cmap('Purples'),
                  # Overall skills
                  "NSE": plt.get_cmap('Spectral'),
                  "KGE": plt.get_cmap('Spectral'),
                  }
    return skill_dict[skill]

##----------------------------------------------------------------------------.
def plot_map(da, ax, 
             cmap, 
             cbar_label,
             vmin = None, 
             vmax = None,
             crs_ref=ccrs.Geodetic(), 
             add_colorbar = True, 
             cbar_shrink=0.7, 
             cbar_pad=0.03, extend='neither'):
    """ Plots a weather signal drawing coastlines

    Parameters
    ----------
    ds : xr.Dataset
         ds containing signals to plot
    ax : cartopy.mpl.geoaxes
        Axes where plot is drawn
    vmin : float
        Minimum value for colorbar
    vmax : float
        Maximum value for colorbar
    proj: cartopy.crs.CRS 
        Coordinate Reference System of the xr.Dataset
    cmap : string
        Colormap
    add_colorbar : bool
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
    cbar_kwargs = {'pad': cbar_pad, 
                   'shrink': cbar_shrink,
                   'label': cbar_label}             
    primitive = da.sphere.plot(ax=ax,
                               transform=crs_ref, 
                               cmap=cmap,
                               vmin=vmin, 
                               vmax=vmax, 
                               extend=extend, 
                               add_colorbar=add_colorbar,
                               cbar_kwargs=cbar_kwargs)                   
    ax.coastlines()
    return primitive

##----------------------------------------------------------------------------.
def plot_skill_maps(ds_skill,  
                    figs_dir,
                    crs_proj = ccrs.Robinson(),
                    variables = ['z500', 't850'],
                    skills = ['BIAS','RMSE','rSD', 'pearson_R2', 'error_CoV'],
                    suffix="",
                    prefix=""):
    ##------------------------------------------------------------------------.
    # Check figs_dir 
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)
    ##------------------------------------------------------------------------.
    # Create a figure for each leadtime 
    for i, leadtime in enumerate(ds_skill.leadtime.values):  
        # Temporary dataset for a specific leadtime 
        ds = ds_skill.sel(leadtime=leadtime)
        ##--------------------------------------------------------------------.
        # Define super title 
        # - TODO maybe need to convert leadtime in hour if it's not 
        suptitle = 'Forecast skill at lead time: {}'.format(str(leadtime.astype('timedelta64[h]')))
        ##--------------------------------------------------------------------.
        # Create figure 
        fig, axs = plt.subplots(len(skills), len(variables), 
                                figsize=(15, 20), 
                                subplot_kw=dict(projection=crs_proj))
        ##--------------------------------------------------------------------.
        # Add supertitle
        fig.suptitle(suptitle,
                     fontsize=26, y=1.05, x=0.45)
        ##--------------------------------------------------------------------.
        # Set the variable title
        for ax, var in zip(axs[0, :], variables):
            ax.set_title(var.upper(), fontsize=24, y=1.08)
        ##--------------------------------------------------------------------.
        # Display skill maps 
        ax_count = 0 
        axs = axs.flatten()
        for skill in skills:
            for var in variables: 
                # Define colorbar options
                cbar_kwargs = {'pad': 0.03, 
                               'shrink': 0.7,
                               'label': skill}
                # xsphere._contourf(ds[var].sel(skill=skill),
                #                   ax=axs[ax_count],
                #                   transform=ccrs.PlateCarree(), 
                #                   cmap=get_skill_cmap[skill],
                #                   vmin=plot_options[skill]['vmin'],
                #                   vmax=plot_options[skill]['vmax'],
                #                   extend=plot_options[skill]['extend'], 
                #                   add_colorbar=True,
                #                   add_labels=False,
                #                   cbar_kwargs=cbar_kwargs)
                xsphere._plot(ds[var].sel(skill=skill),
                              ax=axs[ax_count],
                              transform=ccrs.Geodetic(), 
                              edgecolors=None,
                              cmap=get_skill_cmap(skill),
                              vmin=get_spatial_ylim(skill, var)[0],
                              vmax=get_spatial_ylim(skill, var)[1],
                              extend=get_skill_cbar_extend(skill), 
                              add_colorbar=True,
                              add_labels=False,
                              cbar_kwargs=cbar_kwargs)                
                axs[ax_count].coastlines()
                axs[ax_count].outline_patch.set_linewidth(5)
                ax_count += 1
        ##--------------------------------------------------------------------.         
        # Figure tight layout 
        fig.tight_layout(pad=-2)
        plt.show()
        ##--------------------------------------------------------------------.    
        # Define figure filename 
        if prefix != "": 
            prefix = prefix + "_"
        if suffix != "":
            suffix = "_" + suffix 
        leadtime_str = str(int(leadtime/np.timedelta64(1,'h')))
        fname = prefix + "L" + leadtime_str + suffix + '.png'
        ##--------------------------------------------------------------------.    
        # Save figure 
        fig.savefig(os.path.join(figs_dir, fname), bbox_inches='tight')
        ##--------------------------------------------------------------------.    
    return 
    
##----------------------------------------------------------------------------.
def plot_global_skill(ds_global_skill, skill="RMSE", 
                      variables=['z500','t850'],
                      n_leadtimes=20): 
    # Plot first n_leadtimes
    ds_global_skill = ds_global_skill.isel(leadtime=slice(0, n_leadtimes))
    # Retrieve leadtime
    leadtimes = ds_global_skill['leadtime'].values
    leadtimes = [str(l).split(" ")[0] for l in leadtimes.astype('timedelta64[h]')]
    # Create figure
    fig, axs = plt.subplots(1, len(variables), figsize=(15, 4))
    for ax, var in zip(axs.flatten(), variables):
        ax.plot(leadtimes, ds_global_skill[var].sel(skill=skill).values)
        ax.set_ylim(get_global_ylim(skill, var))
        ax.set_xlabel('Leadtime (h)')  
        ax.set_ylabel(skill)
        ax.set_title(var.upper())
    fig.tight_layout()
    return fig 
 
def plot_global_skills(ds_global_skill, 
                       skills=['BIAS','RMSE','rSD','pearson_R2','KGE','error_CoV'],
                       variables=['z500','t850'],
                       n_leadtimes=20):
    # Plot first n_leadtimes
    ds_global_skill = ds_global_skill.isel(leadtime=slice(0, n_leadtimes))
    # Retrieve leadtime
    leadtimes = ds_global_skill['leadtime'].values
    leadtimes = [str(l).split(" ")[0] for l in leadtimes.astype('timedelta64[h]')]
    # Create figure
    fig, axs = plt.subplots(len(skills), len(variables), figsize=(17, 18))
    # Initialize axes
    ax_i = 0
    axs = axs.flatten()
    for skill in skills: 
        for var in variables:
            axs[ax_i].plot(leadtimes, ds_global_skill[var].sel(skill=skill).values)
            if skill in ['relBIAS','BIAS','percBIAS','diffMean','diffSD','diffCoV','error_CoV']:
                axs[ax_i].axhline(y=0, linestyle='dashed', color="gray")
            elif skill in ['rSD','rMean','rCoV']:
                axs[ax_i].axhline(y=1, linestyle='dashed', color="gray")
            axs[ax_i].set_ylim(get_global_ylim(skill, var))
            axs[ax_i].set_xlabel('Leadtime (h)')  
            axs[ax_i].set_ylabel(skill)
            if ax_i <= 1:
                axs[ax_i].set_title(var.upper())
            # Update ax count 
            ax_i += 1
    # Figure tight layout
    fig.tight_layout()
    return fig

def plot_skills_distribution(ds_skill, 
                             skills=['BIAS','RMSE','rSD','pearson_R2','KGE','error_CoV'],
                             variables=['z500','t850'],
                             n_leadtimes=20):
    # Plot first n_leadtimes
    ds_skill = ds_skill.isel(leadtime=slice(0, n_leadtimes))           
    # Retrieve leadtime
    leadtimes = ds_skill['leadtime'].values
    leadtimes = [str(l).split(" ")[0] for l in leadtimes.astype('timedelta64[h]')]
    # Create figure
    fig, axs = plt.subplots(len(skills), len(variables), figsize=(17, 18))
    # Initialize axes
    ax_i = 0
    axs = axs.flatten()
    for skill in skills: 
        for var in variables:
            tmp_boxes = [ds_skill[var].sel(skill=skill).values[i, :] for i in range(len(ds_skill[var].sel(skill=skill).values))]
            axs[ax_i].boxplot(tmp_boxes, showfliers=False)
            if skill in ['relBIAS','BIAS','percBIAS','diffMean','diffSD','diffCoV','error_CoV']:
                axs[ax_i].axhline(y=0, linestyle='dashed', color="gray")
            elif skill in ['rSD','rMean','rCoV']:
                axs[ax_i].axhline(y=1, linestyle='dashed', color="gray")
            axs[ax_i].set_ylim(get_spatial_ylim(skill, var))
            axs[ax_i].set_xlabel('Leadtime (h)')  
            axs[ax_i].set_ylabel(skill)
            axs[ax_i].set_xticklabels(leadtimes)
            axs[ax_i].tick_params(axis='both', which='major', labelsize=14)
            ##------------------------------------------------------------------.
            # Violin plots
            # import seaborn as sns
            # da = ds_skill[var].sel(skill=skill)
            # da.to_dataframe().reset_index()
            # ax = sns.boxplot(x=df.time.dt.hour, y=name, data=df)
            ##------------------------------------------------------------------.
            if ax_i <= 1:
                axs[ax_i].set_title(var.upper())
            # Update ax count 
            ax_i += 1
    # Figure tight layout
    fig.tight_layout()
    return fig 

##------------------------------------------------------------------------------.
def create_GIF_forecast_error(GIF_fpath,
                              ds_forecast,
                              ds_obs, 
                              aspect_cbar = 40,
                              antialiased = False,
                              edgecolors = None):
    ##----------------------------------------------------------------------------.                              
    # Check Datasets have mesh attached 
    if 'mesh' not in list(ds_forecast.coords.keys()):
        raise ValueError("No 'mesh' coordinate in ds_forecast.")
    if 'mesh' not in list(ds_obs.coords.keys()):
        raise ValueError("No 'mesh' coordinate in ds_obs.")
    ##----------------------------------------------------------------------------.
    # Retrieve forecast reference time 
    forecast_reference_time = str(ds_forecast['forecast_reference_time'].values.astype('datetime64[s]')) # remove nanosecs

    # Load data into memory
    ds_forecast = ds_forecast.load()

    # Retrieve valid time 
    valid_time = ds_forecast['forecast_reference_time'].values + ds_forecast['leadtime'].values
    ds_forecast = ds_forecast.assign_coords({'time': ('leadtime', valid_time)})
    ds_forecast = ds_forecast.swap_dims({'leadtime': 'time'})

    # Subset observations and load in memory
    ds_obs = ds_obs.sel(time=ds_forecast['time'].values)
    ds_obs = ds_obs.load()

    # Compute error 
    ds_error = ds_forecast - ds_obs 
    ds_error = ds_error.assign_coords({'mesh': ("node", ds_obs['mesh'].values)})

    # Create a dictionary with relevant infos  
    ds_dict = {"pred": ds_forecast, "obs": ds_obs, "error": ds_error}

    # Retrieve common variables to plot 
    variables = list(ds_forecast.data_vars.keys())

    ##----------------------------------------------------------------------------.
    # Check GIF fpath 
    if not os.path.exists(os.path.dirname(GIF_fpath)):
        os.makedirs(os.path.dirname(GIF_fpath))
        
    # Check GIF fpath ends with .gif
    if GIF_fpath[-4:] != ".gif":
        print("Added .gif to GIF_fpath.")
        GIF_fpath = GIF_fpath + ".gif"
        
    # Create temporary directory to store temporary GIF image frames
    tmp_dir = tempfile.mkdtemp()

    ##----------------------------------------------------------------------------.
    # Create GIF image frames for each leadtime 
    ds_forecast['leadtime'].values
    for i in range(len(ds_forecast['leadtime'])):
        # Select frame super title 
        tmp_leadtime = str(ds_forecast['leadtime'].values[i].astype('timedelta64[h]'))
        tmp_valid_time = str(ds_forecast['time'].values[i].astype('datetime64[s]'))
        suptitle_str = "Forecast reference time: {}, Leadtime: {}".format(forecast_reference_time, tmp_leadtime)
        ##------------------------------------------------------------------------.
        # Create figure 
        fig, axs = plt.subplots(nrows=len(variables), ncols=3, 
                        figsize=(18, 4*len(variables)),
                        subplot_kw={'projection': ccrs.Robinson()})
        fig.suptitle(suptitle_str)
        # fig.subplots_adjust(wspace=0.1, hspace=0.2)
        ##------------------------------------------------------------------------.
        # Initialize 
        axs = axs.flatten()
        ax_count = 0
        ##------------------------------------------------------------------------.
        # Plot each variable
        for var in variables:
            # Plot obs 
            tmp_obs = ds_dict['obs'][var].isel(time=i)
            xsphere._plot(tmp_obs,
                            ax=axs[ax_count],
                            edgecolors = edgecolors,
                            antialiased = antialiased,
                            vmin=get_var_clim(var,'state')[0],
                            vmax=get_var_clim(var,'state')[1],
                            cmap=get_var_cmap(var,'state')
                            )
            axs[ax_count].set_title(None)
            axs[ax_count].coastlines(alpha=0.2)
            # Plot pred 
            tmp_pred = ds_dict['pred'][var].isel(time=i)
            s_p = xsphere._plot(tmp_pred,
                                ax=axs[ax_count+1],
                                edgecolors = edgecolors, 
                                antialiased = antialiased,
                                vmin=get_var_clim(var,'state')[0],
                                vmax=get_var_clim(var,'state')[1],
                                cmap=get_var_cmap(var,'state'),
                                )
            axs[ax_count+1].set_title(None)    
            axs[ax_count+1].coastlines(alpha=0.2)
            # - Add state colorbar
            cbar = fig.colorbar(s_p, ax=axs[[ax_count, ax_count+1]], 
                                orientation="horizontal", 
                                extend = 'both',
                                aspect=aspect_cbar)       
            cbar.set_label(var.upper())
            cbar.ax.xaxis.set_label_position('top')
            # Plot error 
            tmp_error = ds_dict['error'][var].isel(time=i)
            e_p = xsphere._plot(tmp_error,
                                ax=axs[ax_count+2],
                                edgecolors = edgecolors, 
                                antialiased = antialiased,
                                vmin=get_var_clim(var,'error')[0],
                                vmax=get_var_clim(var,'error')[1],
                                cmap=get_var_cmap(var,'error'),
                                # add_colorbar = True, 
                                # cbar_kwargs={'orientation': 'horizontal',
                                #              'label': var.upper() + " Error"}
                                )
            axs[ax_count+2].set_title(None)
            axs[ax_count+2].coastlines(alpha=0.2)
            # - Add error colorbar
            # cb = plt.colorbar(e_p, ax=axs[ax_count+2], orientation="horizontal") # pad=0.15)
            # cb.set_label(label=var.upper() + " Error") # size='large', weight='bold'
            cbar_err = fig.colorbar(e_p, ax=axs[ax_count+2],
                                    orientation="horizontal",
                                    extend = 'both',
                                    aspect = aspect_cbar/2)      
            cbar_err.set_label(var.upper() + " Error")
            cbar_err.ax.xaxis.set_label_position('top')
            # Add plot labels 
            # if ax_count == 0: 
            axs[ax_count].set_title("Observed")     
            axs[ax_count+1].set_title("Predicted")  
            axs[ax_count+2].set_title("Error")
            # Update ax_count 
            ax_count += 3
        ##------------------------------------------------------------------------.    
        # Save figure in temporary directory 
        fig.savefig(os.path.join(tmp_dir, '{:04}.png'.format(i)))
    ##----------------------------------------------------------------------------. 
    # Load all figures (in PIL PngImageFile)
    im_fpaths = glob.glob(tmp_dir + "/" + '*')
    im_fpaths.sort()
    img, *l_imgs = [Image.open(fpath) for fpath in im_fpaths]
    ##----------------------------------------------------------------------------.
    # Create a GIF
    img.save(fp = GIF_fpath, 
            format='GIF', 
            append_images = l_imgs,
            save_all=True, 
            duration=0.5*1000,  # The time to display the current frame of the GIF, in milliseconds
            loop=0) # Number of times the GIF should loop 
    ##----------------------------------------------------------------------------.
    # Remove temporary images 
    shutil.rmtree(tmp_dir)
    ##----------------------------------------------------------------------------.


##------------------------------------------------------------------------------.
def create_GIF_forecast_anom_error(GIF_fpath,
                                   ds_forecast,
                                   ds_obs, 
                                   scaler,
                                   anom_title, 
                                   aspect_cbar = 40,
                                   antialiased = False,
                                   edgecolors = None):
    ##----------------------------------------------------------------------------.                              
    # Check Datasets have mesh attached 
    if 'mesh' not in list(ds_forecast.coords.keys()):
        raise ValueError("No 'mesh' coordinate in ds_forecast.")
    if 'mesh' not in list(ds_obs.coords.keys()):
        raise ValueError("No 'mesh' coordinate in ds_obs.")
    ##----------------------------------------------------------------------------.
    # Retrieve forecast reference time 
    forecast_reference_time = str(ds_forecast['forecast_reference_time'].values.astype('datetime64[s]')) # remove nanosecs

    # Load data into memory
    ds_forecast = ds_forecast.load()

    # Retrieve valid time 
    valid_time = ds_forecast['forecast_reference_time'].values + ds_forecast['leadtime'].values
    ds_forecast = ds_forecast.assign_coords({'time': ('leadtime', valid_time)})
    ds_forecast = ds_forecast.swap_dims({'leadtime': 'time'})

    # Subset observations and load in memory
    ds_obs = ds_obs.sel(time=ds_forecast['time'].values)
    ds_obs = ds_obs.load()
    
    # Compute anomalies 
    ds_obs = scaler.transform(ds_obs).compute()
    ds_forecast = scaler.transform(ds_forecast).compute()

    # Compute error 
    ds_error = ds_forecast - ds_obs 
    ds_error = ds_error.assign_coords({'mesh': ("node", ds_obs['mesh'].values)})

    # Create a dictionary with relevant infos  
    ds_dict = {"pred": ds_forecast, "obs": ds_obs, "error": ds_error}

    # Retrieve common variables to plot 
    variables = list(ds_forecast.data_vars.keys())

    ##----------------------------------------------------------------------------.
    # Check GIF fpath 
    if not os.path.exists(os.path.dirname(GIF_fpath)):
        os.makedirs(os.path.dirname(GIF_fpath))
        
    # Check GIF fpath ends with .gif
    if GIF_fpath[-4:] != ".gif":
        print("Added .gif to GIF_fpath.")
        GIF_fpath = GIF_fpath + ".gif"
        
    # Create temporary directory to store temporary GIF image frames
    tmp_dir = tempfile.mkdtemp()

    ##----------------------------------------------------------------------------.
    # Create GIF image frames for each leadtime 
    ds_forecast['leadtime'].values
    for i in range(len(ds_forecast['leadtime'])):
        # Select frame super title 
        tmp_leadtime = str(ds_forecast['leadtime'].values[i].astype('timedelta64[h]'))
        tmp_valid_time = str(ds_forecast['time'].values[i].astype('datetime64[s]'))
        suptitle_str = "Forecast reference time: {}, Leadtime: {}".format(forecast_reference_time, tmp_leadtime)
        ##------------------------------------------------------------------------.
        # Create figure 
        fig, axs = plt.subplots(nrows=len(variables), ncols=3, 
                        figsize=(18, 4*len(variables)),
                        subplot_kw={'projection': ccrs.Robinson()})
        fig.suptitle(suptitle_str)
        # fig.subplots_adjust(wspace=0.1, hspace=0.2)
        ##------------------------------------------------------------------------.
        # Initialize 
        axs = axs.flatten()
        ax_count = 0
        ##------------------------------------------------------------------------.
        # Plot each variable
        for var in variables:
            # Plot obs 
            tmp_obs = ds_dict['obs'][var].isel(time=i)
            xsphere._plot(tmp_obs,
                          ax=axs[ax_count],
                          edgecolors = edgecolors,
                          antialiased = antialiased,
                          vmin = -4,
                          vmax = 4,
                          cmap = plt.get_cmap('Spectral')
                          )
            axs[ax_count].set_title(None)
            axs[ax_count].coastlines(alpha=0.2)
            # Plot pred 
            tmp_pred = ds_dict['pred'][var].isel(time=i)
            s_p = xsphere._plot(tmp_pred,
                                ax=axs[ax_count+1],
                                edgecolors = edgecolors, 
                                antialiased = antialiased,
                                vmin = -4,
                                vmax = 4,
                                cmap = plt.get_cmap('Spectral')
                                )
            axs[ax_count+1].set_title(None)    
            axs[ax_count+1].coastlines(alpha=0.2)
            # - Add state colorbar
            cbar = fig.colorbar(s_p, ax=axs[[ax_count, ax_count+1]], 
                                orientation="horizontal", 
                                extend = 'both',
                                aspect=aspect_cbar)       
            cbar.set_label(var.upper() + " " + anom_title)
            cbar.ax.xaxis.set_label_position('top')
            # Plot error 
            tmp_error = ds_dict['error'][var].isel(time=i)
            e_p = xsphere._plot(tmp_error,
                                ax=axs[ax_count+2],
                                edgecolors = edgecolors, 
                                antialiased = antialiased,
                                vmin=-6,
                                vmax=6,
                                cmap=get_var_cmap(var,'error'),
                                # add_colorbar = True, 
                                # cbar_kwargs={'orientation': 'horizontal',
                                #              'label': var.upper() + " Error"}
                                )
            axs[ax_count+2].set_title(None)
            axs[ax_count+2].coastlines(alpha=0.2)
            # - Add error colorbar
            # cb = plt.colorbar(e_p, ax=axs[ax_count+2], orientation="horizontal") # pad=0.15)
            # cb.set_label(label=var.upper() + " Error") # size='large', weight='bold'
            cbar_err = fig.colorbar(e_p, ax=axs[ax_count+2],
                                    orientation="horizontal",
                                    extend = 'both',
                                    aspect = aspect_cbar/2)      
            cbar_err.set_label(var.upper() + " " + anom_title + " Error")
            cbar_err.ax.xaxis.set_label_position('top')
            # Add plot labels 
            # if ax_count == 0: 
            axs[ax_count].set_title("Observed")     
            axs[ax_count+1].set_title("Predicted")  
            axs[ax_count+2].set_title("Error")
            # Update ax_count 
            ax_count += 3
        ##------------------------------------------------------------------------.    
        # Save figure in temporary directory 
        fig.savefig(os.path.join(tmp_dir, '{:04}.png'.format(i)))
    ##----------------------------------------------------------------------------. 
    # Load all figures (in PIL PngImageFile)
    im_fpaths = glob.glob(tmp_dir + "/" + '*')
    im_fpaths.sort()
    img, *l_imgs = [Image.open(fpath) for fpath in im_fpaths]
    ##----------------------------------------------------------------------------.
    # Create a GIF
    img.save(fp = GIF_fpath, 
            format='GIF', 
            append_images = l_imgs,
            save_all=True, 
            duration=0.5*1000,  # The time to display the current frame of the GIF, in milliseconds
            loop=0) # Number of times the GIF should loop 
    ##----------------------------------------------------------------------------.
    # Remove temporary images 
    shutil.rmtree(tmp_dir)
    ##----------------------------------------------------------------------------.    
    
 
    
 
    
 
 
    
 