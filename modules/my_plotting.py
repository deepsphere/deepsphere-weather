#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 21:39:12 2021

@author: ghiggi
"""
import os
import numpy as np
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
import modules.xsphere as xsphere

# TODO: ylabels add unit
# 'RMSE [$m^2 s^{−2}$]'

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
def plot_global_skill(ds_global_skill, skill="RMSE", variables=['z500','t850']): 
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
                       variables=['z500','t850']):
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
                             variables=['z500','t850']):           
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

 
    
    
 
    
 
    
 
 
    
 