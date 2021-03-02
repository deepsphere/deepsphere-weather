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

def get_skill_vmin_vmax(skill):
    skill_dict = {"error_CoV": (None,None),
                  "obs_CoV": (None,None),
                  "pred_CoV": (None,None),
                  # Magnitude 
                  "BIAS": (None,None),
                  "relBIAS": (None,None),
                  "percBIAS": (-100, 100),
                  "MAE": (0, None),
                  "relMAE": (0, None),
                  "percMAE": (0, 100),
                  "MSE": (0, None),
                  "relMSE": (0, None),
                  "RMSE": (0, None),
                  "relRMSE": (0, None),
                  # Average
                  "rMean": (0, 2),
                  "diffMean": (None,None),
                  # Variability 
                  'rSD': (0, 2),
                  'diffSD': (None,None),
                  "rCoV": (0, 2),
                  "diff_CoV": (None,None),
                  # Correlation 
                  "pearson_R": (-1, 1),
                  "pearson_R2": (0, 1),
                  # "spearman_R": spearman_R,
                  # "spearman_R2": spearman_R2,
                  # Overall skills
                  "NSE": (-1, 1),
                  "KGE": (0, 1),
                  }
    return skill_dict[skill]


def get_skill_cbar_extent(skill):
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
                  "rMean": 'max',
                  "diffMean": 'both',
                  # Variability 
                  'rSD': 'max',
                  'diffSD': 'both',
                  "rCoV": 'max',
                  "diff_CoV": 'both',
                  # Correlation 
                  "pearson_R": 'neither',
                  "pearson_R2": 'neither',
                  # "spearman_R": spearman_R,
                  # "spearman_R2": spearman_R2,
                  # Overall skills
                  "NSE": 'min',
                  "KGE": 'neither',
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
                    suffix="",
                    prefix=""):
    ##------------------------------------------------------------------------.
    # Check figs_dir 
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)
    ##------------------------------------------------------------------------.
    # Define plot options
    plot_options = {'relBIAS': {'vmin': -0.02, 'vmax': 0.02,
                                "extend": 'both', 
                                "cmap": plt.get_cmap('BrBG')},
                    'relRMSE': {'vmin': 0, 'vmax': 0.04,
                                "extend": 'max',
                                "cmap": plt.get_cmap('Reds')},
                    'rSD': {'vmin': 0.2, 'vmax': 1.8, 
                            "extend": 'both',
                            "cmap": plt.get_cmap('PRGn')},
                    'pearson_R2': {'vmin': 0, 'vmax': 1, 
                                   "extend": 'neither',
                                   "cmap": plt.get_cmap('Greens')},
                    }
    variables = ['z500', 't850']
    skills = ['relBIAS','relRMSE','rSD', 'pearson_R2']
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
                                figsize=(15, 15), 
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
                #                   cmap=plot_options[skill]['cmap'],
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
                              cmap=plot_options[skill]['cmap'],
                              vmin=plot_options[skill]['vmin'],
                              vmax=plot_options[skill]['vmax'],
                              extend=plot_options[skill]['extend'], 
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
       


 
    
    
 
    
 
    
 
 
    
 