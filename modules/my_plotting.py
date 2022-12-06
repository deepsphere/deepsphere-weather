#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 21:39:12 2021

@author: ghiggi
"""
import os
import subprocess
import tempfile
import shutil
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from xforecasting.utils.xr import (
    xr_common_vars,
    xr_has_coord,
    xr_has_dim,
    xr_n_vars,
    xr_Dataset_vars,
)
from xscaler import HovmollerDiagram
import xsphere

# TODO: ylabels add unit
# 'RMSE [$m^2 s^{−2}$]'


def get_var_cmap(var, arg):
    cmap_dict = {
        "t850": {"state": plt.get_cmap("RdYlBu_r"), "error": plt.get_cmap("RdBu")},
        "z500": {"state": plt.get_cmap("BuPu"), "error": plt.get_cmap("RdBu")},
        "q500": {
            "state": plt.get_cmap("GnBu"),
            "error": plt.get_cmap("RdBu"),
        },
    }
    if var in list(cmap_dict.keys()):
        if arg in list(cmap_dict[var].keys()):
            cmap = cmap_dict[var][arg]
        else:
            cmap = plt.get_cmap("RdYlBu_r")
    else:
        cmap = plt.get_cmap("RdYlBu_r")
    return cmap


def get_var_clim(var, arg):
    clim_dict = {
        "t850": {"state": (225, 310), "error": (-10, 10)},
        "z500": {"state": (44000, 59000), "error": (-5000, 5000)},
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
    ylim_dict = {
        "RMSE": {"z500": (50, 700), "t850": (0.5, 4)},
        "rSD": {"z500": (0.6, 1.4), "t850": (0.6, 1.4)},
        "relBIAS": {"z500": (-0.002, 0.002), "t850": (-0.002, 0.002)},
        "pearson_R2": {"z500": (0.3, 1), "t850": (0.3, 1)},
        "KGE": {"z500": (0.4, 1), "t850": (0.4, 1)},
        "NSE": {"z500": (0, 1), "t850": (0, 1)},
        "percBIAS": {"z500": (-1, 1), "t850": (-2, 2)},
        "percMAE": {"z500": (0, 2), "t850": (0, 2)},
        "error_CoV": {"z500": (-40, 40), "t850": (-40, 40)},
        "MAE": {"z500": (50, 700), "t850": (0.5, 2.5)},
        "BIAS": {"z500": (-110, 110), "t850": (-1, 1)},
        "diffSD": {"z500": (-300, 300), "t850": (-1.5, 1.5)},
    }
    if skill in list(ylim_dict.keys()):
        if var in list(ylim_dict[skill].keys()):
            ylim = ylim_dict[skill][var]
        else:
            ylim = (None, None)
    else:
        ylim = (None, None)
    return ylim


def get_legend_loc(skill):
    loc_dict = {
        "RMSE": "upper left",
        "rSD": "best",
        "relBIAS": "best",
        "pearson_R2": "upper right",
        "KGE": "upper right",
        "NSE": "upper right",
        "percBIAS": "best",
        "percMAE": "upper left",
        "error_CoV": "best",
        "MAE": "upper left",
        "BIAS": "best",
        "diffSD": "best",
    }
    if skill in list(loc_dict.keys()):
        loc = loc_dict[skill]
    else:
        loc = "best"
    return loc


def get_spatial_ylim(skill, var):
    ylim_dict = {
        "BIAS": {"z500": (-400, 400), "t850": (-4, 4)},
        "RMSE": {"z500": (0, 2000), "t850": (0, 8)},
        "rSD": {"z500": (0.6, 1.4), "t850": (0.6, 1.4)},
        "pearson_R2": {"z500": (0, 1), "t850": (0, 1)},
        "KGE": {"z500": (0, 1), "t850": (0, 1)},
        "NSE": {"z500": (0, 1), "t850": (0, 1)},
        "relBIAS": {"z500": (-0.006, 0.006), "t850": (-0.01, 0.01)},
        "percBIAS": {"z500": (-2.5, 2.5), "t850": (-2.5, 2.5)},
        "percMAE": {"z500": (0, 2.5), "t850": (0, 2.5)},
        "error_CoV": {"z500": (-40, 40), "t850": (-40, 40)},
        "MAE": {"z500": (50, 700), "t850": (0.5, 2.5)},
        "diffSD": {"z500": (-300, 300), "t850": (-1, 1)},
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
    skill_dict = {
        "error_CoV": "both",
        "obs_CoV": "both",
        "pred_CoV": "both",
        # Magnitude
        "BIAS": "both",
        "relBIAS": "both",
        "percBIAS": "both",
        "MAE": "max",
        "relMAE": "max",
        "percMAE": "max",
        "MSE": "max",
        "relMSE": "max",
        "RMSE": "max",
        "relRMSE": "max",
        # Average
        "rMean": "both",
        "diffMean": "both",
        # Variability
        "rSD": "both",
        "diffSD": "both",
        "rCoV": "max",
        "diffCoV": "both",
        # Correlation
        "pearson_R": "neither",
        "pearson_R2": "neither",
        "spearman_R": "neither",
        "spearman_R2": "neither",
        "pearson_R2_pvalue": "neither",
        "spearman_R2_pvalue": "max",
        # Overall skills
        "NSE": "min",
        "KGE": "neither",
    }
    return skill_dict[skill]


def get_skill_cmap(skill):
    skill_dict = {
        "error_CoV": plt.get_cmap("RdYlBu"),
        "obs_CoV": plt.get_cmap("YlOrRd"),
        "pred_CoV": plt.get_cmap("YlOrRd"),
        # Magnitude
        "BIAS": plt.get_cmap("BrBG"),
        "relBIAS": plt.get_cmap("BrBG"),
        "percBIAS": plt.get_cmap("BrBG"),
        "MAE": plt.get_cmap("Reds"),
        "relMAE": plt.get_cmap("Reds"),
        "percMAE": plt.get_cmap("Reds"),
        "MSE": plt.get_cmap("Reds"),
        "relMSE": plt.get_cmap("Reds"),
        "RMSE": plt.get_cmap("Reds"),
        "relRMSE": plt.get_cmap("Reds"),
        # Average
        "rMean": plt.get_cmap("BrBG"),
        "diffMean": plt.get_cmap("BrBG"),
        # Variability
        "rSD": plt.get_cmap("PRGn"),
        "diffSD": plt.get_cmap("PRGn"),
        "rCoV": plt.get_cmap("PRGn"),
        "diffCoV": plt.get_cmap("PRGn"),
        # Correlation
        "pearson_R": plt.get_cmap("Greens"),
        "pearson_R2": plt.get_cmap("Greens"),
        "spearman_R": plt.get_cmap("Greens"),
        "spearman_R2": plt.get_cmap("Greens"),
        "pearson_R_pvalue": plt.get_cmap("Purples"),
        "spearman_R_pvalue": plt.get_cmap("Purples"),
        # Overall skills
        "NSE": plt.get_cmap("Spectral"),
        "KGE": plt.get_cmap("Spectral"),
    }
    return skill_dict[skill]


##----------------------------------------------------------------------------.
def plot_map(
    da,
    ax,
    cmap,
    cbar_label,
    vmin=None,
    vmax=None,
    crs_ref=ccrs.Geodetic(),
    add_colorbar=True,
    cbar_shrink=0.7,
    cbar_pad=0.03,
    extend="neither",
):
    """Plots a weather signal drawing coastlines

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
    cbar_kwargs = {"pad": cbar_pad, "shrink": cbar_shrink, "label": cbar_label}
    primitive = da.sphere.plot(
        ax=ax,
        transform=crs_ref,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extend=extend,
        add_colorbar=add_colorbar,
        cbar_kwargs=cbar_kwargs,
    )
    ax.coastlines()
    return primitive


##----------------------------------------------------------------------------.
def plot_skill_maps(
    ds_skill,
    figs_dir,
    crs_proj=ccrs.Robinson(),
    variables=["z500", "t850"],
    skills=["BIAS", "RMSE", "rSD", "pearson_R2", "error_CoV"],
    suffix="",
    prefix="",
):
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
        suptitle = "Forecast skill at lead time: {}".format(
            str(leadtime.astype("timedelta64[h]"))
        )
        ##--------------------------------------------------------------------.
        # Create figure
        fig, axs = plt.subplots(
            len(skills),
            len(variables),
            figsize=(15, 20),
            subplot_kw=dict(projection=crs_proj),
        )
        ##--------------------------------------------------------------------.
        # Add supertitle
        fig.suptitle(suptitle, fontsize=26, y=1.05, x=0.45)
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
                cbar_kwargs = {"pad": 0.03, "shrink": 0.7, "label": skill}
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
                xsphere.plot.plot(
                    ds[var].sel(skill=skill),
                    ax=axs[ax_count],
                    transform=ccrs.Geodetic(),
                    edgecolors=None,
                    cmap=get_skill_cmap(skill),
                    vmin=get_spatial_ylim(skill, var)[0],
                    vmax=get_spatial_ylim(skill, var)[1],
                    extend=get_skill_cbar_extend(skill),
                    add_colorbar=True,
                    add_labels=False,
                    cbar_kwargs=cbar_kwargs,
                )
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
        leadtime_str = str(int(leadtime / np.timedelta64(1, "h")))
        fname = prefix + "L" + leadtime_str + suffix + ".png"
        ##--------------------------------------------------------------------.
        # Save figure
        fig.savefig(os.path.join(figs_dir, fname), bbox_inches="tight")
        ##--------------------------------------------------------------------.
    return


##----------------------------------------------------------------------------.
def plot_global_skill(
    ds_global_skill, skill="RMSE", variables=["z500", "t850"], n_leadtimes=20
):
    # Plot first n_leadtimes
    ds_global_skill = ds_global_skill.isel(leadtime=slice(0, n_leadtimes))
    # Retrieve leadtime
    leadtimes = ds_global_skill["leadtime"].values
    leadtimes = [str(l).split(" ")[0] for l in leadtimes.astype("timedelta64[h]")]
    # Create figure
    fig, axs = plt.subplots(1, len(variables), figsize=(15, 4))
    for ax, var in zip(axs.flatten(), variables):
        # Plot global average skill
        ax.plot(leadtimes, ds_global_skill[var].sel(skill=skill).values)
        ##------------------------------------------------------------------.
        # Add best skill line
        if skill in [
            "relBIAS",
            "BIAS",
            "percBIAS",
            "diffMean",
            "diffSD",
            "diffCoV",
            "error_CoV",
        ]:
            ax.axhline(y=0, linestyle="solid", color="gray", alpha=0.2)
        elif skill in ["rSD", "rMean", "rCoV"]:
            ax.axhline(y=1, linestyle="solid", color="gray", alpha=0.2)
        ##------------------------------------------------------------------.
        # Add labels
        ax.set_ylim(get_global_ylim(skill, var))
        ax.set_xlabel("Leadtime (h)")
        ax.set_ylabel(skill)
        # Set axis appearance
        ax.margins(x=0, y=0)
        # Set xticks
        ax.set_xticks(leadtimes[::2])
        ax.set_xticklabels(leadtimes[::2])
        ##------------------------------------------------------------------.
        # Add title
        ax.set_title(var.upper())
        ##------------------------------------------------------------------.
    fig.tight_layout()
    return fig


def plot_global_skills(
    ds_global_skill,
    skills=["BIAS", "RMSE", "rSD", "pearson_R2", "KGE", "error_CoV"],
    variables=["z500", "t850"],
    n_leadtimes=20,
):
    # Plot first n_leadtimes
    ds_global_skill = ds_global_skill.isel(leadtime=slice(0, n_leadtimes))
    # Retrieve leadtime
    leadtimes = ds_global_skill["leadtime"].values
    leadtimes = [str(l).split(" ")[0] for l in leadtimes.astype("timedelta64[h]")]
    # Create figure
    fig, axs = plt.subplots(len(skills), len(variables), figsize=(17, 18))
    # Initialize axes
    ax_i = 0
    axs = axs.flatten()
    for skill in skills:
        for var in variables:
            # Plot global average skill
            axs[ax_i].plot(leadtimes, ds_global_skill[var].sel(skill=skill).values)
            ##------------------------------------------------------------------.
            # Add best skill line
            if skill in [
                "relBIAS",
                "BIAS",
                "percBIAS",
                "diffMean",
                "diffSD",
                "diffCoV",
                "error_CoV",
            ]:
                axs[ax_i].axhline(y=0, linestyle="solid", color="gray", alpha=0.2)
            elif skill in ["rSD", "rMean", "rCoV"]:
                axs[ax_i].axhline(y=1, linestyle="solid", color="gray", alpha=0.2)
            ##------------------------------------------------------------------.
            # Add labels
            axs[ax_i].set_ylim(get_global_ylim(skill, var))
            axs[ax_i].set_xlabel("Leadtime (h)")
            axs[ax_i].set_ylabel(skill)
            # Set axis appearance
            axs[ax_i].margins(x=0, y=0)
            # Set xticks
            axs[ax_i].set_xticks(leadtimes[::2])
            axs[ax_i].set_xticklabels(leadtimes[::2])
            ##------------------------------------------------------------------.
            # Add title
            if ax_i < len(variables):
                axs[ax_i].set_title(var.upper())
            ##------------------------------------------------------------------.
            # Update ax count
            ax_i += 1
    # Figure tight layout
    fig.tight_layout()
    return fig


def plot_skills_distribution(
    ds_skill,
    skills=["BIAS", "RMSE", "rSD", "pearson_R2", "KGE", "error_CoV"],
    variables=["z500", "t850"],
    n_leadtimes=20,
):
    # Plot first n_leadtimes
    ds_skill = ds_skill.isel(leadtime=slice(0, n_leadtimes))
    # Retrieve leadtime
    leadtimes = ds_skill["leadtime"].values
    leadtimes = [str(l).split(" ")[0] for l in leadtimes.astype("timedelta64[h]")]
    # Create figure
    fig, axs = plt.subplots(len(skills), len(variables), figsize=(17, 18))
    # Initialize axes
    ax_i = 0
    axs = axs.flatten()
    for skill in skills:
        for var in variables:
            # Plot skill distribution
            tmp_boxes = [
                ds_skill[var].sel(skill=skill).values[i, :]
                for i in range(len(ds_skill[var].sel(skill=skill).values))
            ]
            axs[ax_i].boxplot(tmp_boxes, showfliers=False)
            ##------------------------------------------------------------------.
            # Add best skill line
            if skill in [
                "relBIAS",
                "BIAS",
                "percBIAS",
                "diffMean",
                "diffSD",
                "diffCoV",
                "error_CoV",
            ]:
                axs[ax_i].axhline(y=0, linestyle="solid", color="gray")
            elif skill in ["rSD", "rMean", "rCoV"]:
                axs[ax_i].axhline(y=1, linestyle="solid", color="gray")
            ##------------------------------------------------------------------.
            # Add labels
            axs[ax_i].set_ylim(get_spatial_ylim(skill, var))
            axs[ax_i].set_xlabel("Leadtime (h)")
            axs[ax_i].set_ylabel(skill)
            axs[ax_i].set_xticklabels(leadtimes)
            axs[ax_i].tick_params(axis="both", which="major", labelsize=14)
            ##------------------------------------------------------------------.
            # Violin plots
            # import seaborn as sns
            # da = ds_skill[var].sel(skill=skill)
            # da.to_dataframe().reset_index()
            # ax = sns.boxplot(x=df.time.dt.hour, y=name, data=df)
            ##------------------------------------------------------------------.
            # Add title
            if ax_i < len(variables):
                axs[ax_i].set_title(var.upper())
            ##------------------------------------------------------------------.
            # Update ax count
            ax_i += 1
    # Figure tight layout
    fig.tight_layout()
    return fig


##-----------------------------------------------------------------------------.
def benchmark_global_skill(
    skills_dict,
    skill="RMSE",
    variables=["z500", "t850"],
    colors_dict=None,
    ylim="fixed",
    n_leadtimes=20,
):
    # Get model/forecast names
    forecast_names = list(skills_dict.keys())
    # Get colors_dict if not specified
    if colors_dict is None:
        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colors_dict = {
            k: default_colors[i] for i, (k, v) in enumerate(skills_dict.items())
        }
    else:
        # Check that a colors for each forecast is specified
        forecast_without_color = np.array(forecast_names)[
            np.isin(forecast_names, list(colors_dict.keys()), invert=True)
        ]
        if len(forecast_without_color) > 0:
            raise ValueError(
                "Color must be specified also for {}".format(forecast_without_color)
            )
    # Retrieve leadtimes
    leadtimes = (
        skills_dict[forecast_names[0]]
        .isel(leadtime=slice(0, n_leadtimes))["leadtime"]
        .values
    )
    leadtimes = [str(l).split(" ")[0] for l in leadtimes.astype("timedelta64[h]")]
    # Create figure
    fig, axs = plt.subplots(1, len(variables), figsize=(15, 4))
    # Linestyles for climatology baselines
    climatology_linestyles = ["--", ":", "-."]
    for ax, var in zip(axs.flatten(), variables):
        clim_i = 0
        for forecast_name in forecast_names:
            ## Plot forecast skill
            # - Get the skill of the forecast
            tmp_skill = skills_dict[forecast_name].sel(skill=skill)
            # - If dynamic or persistence forecast (not climatology forecast)
            if "leadtime" in list(tmp_skill.dims):
                ax.plot(
                    leadtimes,
                    tmp_skill.isel(leadtime=slice(0, n_leadtimes))[var].values,
                    color=colors_dict[forecast_name],
                )
            # - If climatology forecast
            else:
                ax.axhline(
                    y=tmp_skill[var].values,
                    linestyle=climatology_linestyles[clim_i],
                    color=colors_dict[forecast_name],
                )
                clim_i += 1
        ##---------------------------------------------------------------------.
        # Set axis limit based on skill and model variable
        if isinstance(ylim, str) and ylim == "fixed":
            ax.set_ylim(get_global_ylim(skill, var))
        elif isinstance(ylim, tuple):
            ax.set_ylim(ylim)
        # Add labels
        ax.set_xlabel("Leadtime (h)")
        ax.set_ylabel(skill)
        ax.set_title(var.upper())
        # Set axis appearance
        ax.margins(x=0, y=0)
        # Set xticks
        ax.set_xticks(leadtimes[::2])
        ax.set_xticklabels(leadtimes[::2])
        ##---------------------------------------------------------------------.
        # Add legend
        ax.legend(
            forecast_names,
            loc="best",  # get_legend_loc(skill),
            frameon=True,
            fancybox=True,
            framealpha=1,
            shadow=False,
            borderpad=0,
        )
        ##---------------------------------------------------------------------.
        # Add best skill line
        if skill in [
            "relBIAS",
            "BIAS",
            "percBIAS",
            "diffMean",
            "diffSD",
            "diffCoV",
            "error_CoV",
        ]:
            ax.axhline(y=0, linestyle="solid", color="gray", alpha=0.2)
        elif skill in ["rSD", "rMean", "rCoV"]:
            ax.axhline(y=1, linestyle="solid", color="gray", alpha=0.2)
        ##------------------------------------------------------------------.
    fig.tight_layout()
    return fig


def benchmark_global_skills(
    skills_dict,
    skills=["BIAS", "RMSE", "rSD", "pearson_R2", "KGE", "error_CoV"],
    variables=["z500", "t850"],
    colors_dict=None,
    legend_everywhere=False,
    n_leadtimes=20,
):
    # Get model/forecast names
    forecast_names = list(skills_dict.keys())
    # Get colors_dict if not specified
    if colors_dict is None:
        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colors_dict = {
            k: default_colors[i] for i, (k, v) in enumerate(skills_dict.items())
        }
    else:
        # Check that a colors for each forecast is specified
        forecast_without_color = np.array(forecast_names)[
            np.isin(forecast_names, list(colors_dict.keys()), invert=True)
        ]
        if len(forecast_without_color) > 0:
            raise ValueError(
                "Color must be specified also for {}".format(forecast_without_color)
            )
    # Retrieve leadtimes
    leadtimes = (
        skills_dict[forecast_names[0]]
        .isel(leadtime=slice(0, n_leadtimes))["leadtime"]
        .values
    )
    leadtimes = [str(l).split(" ")[0] for l in leadtimes.astype("timedelta64[h]")]
    # Create figure
    fig, axs = plt.subplots(
        len(skills), len(variables), figsize=(17, 18)
    )  # 19 cm x 23 = (7.4, 9)
    # Linestyles for climatology baselines
    climatology_linestyles = ["--", ":", "-."]
    # Initialize axes
    ax_i = 0
    leg_i = 0
    axs = axs.flatten()
    for skill in skills:
        for var in variables:
            clim_i = 0
            for forecast_name in forecast_names:
                ## Plot forecast skill
                # - Get the skill of the forecast
                tmp_skill = skills_dict[forecast_name].sel(skill=skill)
                # - If dynamic or persistence forecast (not climatology forecast)
                if "leadtime" in list(tmp_skill.dims):
                    axs[ax_i].plot(
                        leadtimes,
                        tmp_skill.isel(leadtime=slice(0, n_leadtimes))[var].values,
                        color=colors_dict[forecast_name],
                    )
                # - If climatology forecast
                else:
                    axs[ax_i].axhline(
                        y=tmp_skill[var].values,
                        linestyle=climatology_linestyles[clim_i],
                        color=colors_dict[forecast_name],
                    )
                    clim_i += 1
            ##-----------------------------------------------------------------.
            # Set axis limit based on skill and model variable
            axs[ax_i].set_ylim(get_global_ylim(skill, var))
            # Add labels
            axs[ax_i].set_xlabel("Leadtime (h)")
            axs[ax_i].set_ylabel(skill)
            # Set axis appearance
            axs[ax_i].margins(x=0, y=0)
            # Set xticks
            axs[ax_i].set_xticks(leadtimes[::2])
            axs[ax_i].set_xticklabels(leadtimes[::2])

            ##-----------------------------------------------------------------.
            # Set title
            if ax_i < len(variables):
                axs[ax_i].set_title(var.upper())
            ##------------------------------------------------------------------.
            # Add legend
            if not legend_everywhere:
                if leg_i == 0:
                    axs[ax_i].legend(
                        forecast_names,
                        loc=get_legend_loc(skill),
                        frameon=True,
                        fancybox=True,
                        framealpha=1,
                        shadow=False,
                        borderpad=0,
                    )
                leg_i += 1
            else:
                axs[ax_i].legend(
                    forecast_names,
                    loc=get_legend_loc(skill),
                    frameon=True,
                    fancybox=True,
                    framealpha=1,
                    shadow=False,
                    borderpad=0,
                )
            ##------------------------------------------------------------------.
            # Add best skill line
            if skill in [
                "relBIAS",
                "BIAS",
                "percBIAS",
                "diffMean",
                "diffSD",
                "diffCoV",
                "error_CoV",
            ]:
                axs[ax_i].axhline(y=0, linestyle="solid", color="gray", alpha=0.2)
            elif skill in ["rSD", "rMean", "rCoV"]:
                axs[ax_i].axhline(y=1, linestyle="solid", color="gray", alpha=0.2)
            ##------------------------------------------------------------------.
            # Update ax count
            ax_i += 1
        fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------.
def create_hovmoller_plots(ds_obs, ds_pred, scaler=None, arg="state", time_groups=None):
    # Retrieve time
    if xr_has_dim(ds_pred, dim="leadtime"):
        if not xr_has_coord(ds_pred, coord="forecast_reference_time"):
            raise ValueError(
                "If 'leadtime' is a dimension, 'forecast_reference_time' must be a coordinate."
            )
        if ds_pred["forecast_reference_time"].size != 1:
            raise ValueError("Provide data with a single 'forecast_reference_time'")
        # Drop forecast_reference_time dimension of size 1
        if xr_has_dim(ds_pred, dim="forecast_reference_time"):
            ds_pred = ds_pred.isel(forecast_reference_time=0)
        # Compute prediction time
        ds_pred["time"] = (
            ds_pred["forecast_reference_time"].values + ds_pred["leadtime"]
        )
        ds_pred = ds_pred.set_coords("time").swap_dims({"leadtime": "time"})
    # Check time dimension exist
    if not xr_has_dim(ds_pred, dim="time"):
        raise ValueError(
            "Either 'time' or 'forecast_reference_time' must be dimension of ds_pred."
        )
    ##-----------------------------------------------------------------------.
    # Ensure pred and obs have same variables
    variables = xr_common_vars(ds_obs, ds_pred)
    n_vars = len(variables)
    if n_vars == 0:
        raise ValueError("No common variables between obs and pred.")
    ds_obs = ds_obs[variables]
    ds_pred = ds_pred[variables]

    # Align dimensions and load data
    ds_obs, ds_pred = xr.align(ds_obs, ds_pred)
    ds_pred = ds_pred.compute()
    ds_obs = ds_obs.compute()

    ##-----------------------------------------------------------------------.
    # Scale data (i.e. anomalies)
    if scaler is not None:
        ds_obs = scaler.transform(ds_obs)
        ds_pred = scaler.transform(ds_pred)
    ##-----------------------------------------------------------------------.
    # Compute hovmollers
    hovmoller_obs = HovmollerDiagram(
        ds_obs,
        time_dim="time",
        time_groups=time_groups,
        spatial_dim="lat",
        bin_width=5,
        time_average_before_binning=True,
    )
    hovmoller_pred = HovmollerDiagram(
        ds_pred,
        time_dim="time",
        time_groups=time_groups,
        spatial_dim="lat",
        bin_width=5,
        time_average_before_binning=True,
    )

    hovmoller_diff = hovmoller_pred - hovmoller_obs
    ##-----------------------------------------------------------------------.
    ### Create figure
    fig, axs = plt.subplots(
        n_vars, 3, figsize=(17, 5 * n_vars), gridspec_kw={"width_ratios": [0.83, 1, 1]}
    )
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    axs = axs.flatten()
    for i, var in enumerate(variables):
        # Define vmin, vmax
        vmin, vmax = get_var_clim(var=var, arg=arg)
        if vmin is None:
            vmin = hovmoller_obs[var].values.min()
            vmax = hovmoller_obs[var].values.max()

        # Plot obs
        _ = hovmoller_obs[var].plot(
            ax=axs[i * 3],
            vmin=vmin,
            vmax=vmax,
            cmap=get_var_cmap(var=var, arg=arg),
            add_colorbar=False,
        )
        axs[i * 3].set_ylabel("Latitude")

        # Plot predictions
        p1 = hovmoller_pred[var].plot(
            ax=axs[i * 3 + 1],
            vmin=vmin,
            vmax=vmax,
            cmap=get_var_cmap(var=var, arg=arg),
            add_colorbar=True,
            cbar_kwargs={"pad": 0.07},
        )
        p1.colorbar.ax.yaxis.set_label_position("left")

        # Plot difference
        p2 = hovmoller_diff[var].plot(
            ax=axs[i * 3 + 2],
            add_colorbar=True,
            cbar_kwargs={"label": var + " error", "pad": 0.07},
        )
        p2.colorbar.ax.yaxis.set_label_position("left")

        # Place titles
        if i == 0:
            axs[i * 3].set_title("Observed")
            axs[i * 3 + 1].set_title("Predicted")
            axs[i * 3 + 2].set_title("Difference")
        else:
            axs[i * 3].set_title("")
            axs[i * 3 + 1].set_title("")
            axs[i * 3 + 2].set_title("")
        # Display axis label only at the figure border
        # axs[i*3+1].set_ylabel("")
        # axs[i*3+2].set_ylabel("")
        # Display axis only at the figure border
        axs[i * 3 + 1].get_yaxis().set_visible(False)
        axs[i * 3 + 2].get_yaxis().set_visible(False)

        if i != n_vars - 1:
            axs[i * 3].get_xaxis().set_visible(False)
            axs[i * 3 + 1].get_xaxis().set_visible(False)
            axs[i * 3 + 2].get_xaxis().set_visible(False)
    # -------------------------------------------------------------------------.
    # Return figure
    return fig


# ------------------------------------------------------------------------------.
def create_gif_forecast_error(
    gif_fpath,
    ds_forecast,
    ds_obs,
    fps=4,
    aspect_cbar=40,
    antialiased=False,
    edgecolors=None,
    verbose=False,
):
    ##-------------------------------------------------------------------------.
    # Check Datasets have mesh attached
    if "mesh" not in list(ds_forecast.coords.keys()):
        raise ValueError("No 'mesh' coordinate in ds_forecast.")
    if "mesh" not in list(ds_obs.coords.keys()):
        raise ValueError("No 'mesh' coordinate in ds_obs.")
    ##-------------------------------------------------------------------------.
    # Retrieve forecast reference time
    forecast_reference_time = str(
        ds_forecast["forecast_reference_time"].values.astype("datetime64[s]")
    )  # remove nanosecs

    # Load data into memory
    ds_forecast = ds_forecast.load()

    # Retrieve valid time
    valid_time = (
        ds_forecast["forecast_reference_time"].values + ds_forecast["leadtime"].values
    )
    ds_forecast = ds_forecast.assign_coords({"time": ("leadtime", valid_time)})
    ds_forecast = ds_forecast.swap_dims({"leadtime": "time"})

    # Subset observations and load in memory
    ds_obs = ds_obs.sel(time=ds_forecast["time"].values)
    ds_obs = ds_obs.load()

    # Compute error
    ds_error = ds_forecast - ds_obs
    ds_error = ds_error.assign_coords({"mesh": ("node", ds_obs["mesh"].values)})

    # Create a dictionary with relevant infos
    ds_dict = {"pred": ds_forecast, "obs": ds_obs, "error": ds_error}

    # Retrieve common variables to plot
    variables = list(ds_forecast.data_vars.keys())

    ##-------------------------------------------------------------------------.
    # Check fpath
    if not os.path.exists(os.path.dirname(gif_fpath)):
        os.makedirs(os.path.dirname(gif_fpath))

    # Remove gif file format
    if gif_fpath[-4:] == ".gif":
        gif_fpath = gif_fpath[:-4]

    # Create temporary directory to store temporary GIF image frames
    tmp_dir = tempfile.mkdtemp()

    ##-------------------------------------------------------------------------.
    # Create GIF image frames for each leadtime
    for i in range(len(ds_forecast["leadtime"])):
        # Select frame super title
        tmp_leadtime = str(ds_forecast["leadtime"].values[i].astype("timedelta64[h]"))
        # tmp_valid_time = str(ds_forecast['time'].values[i].astype('datetime64[s]'))
        suptitle_str = "Forecast reference time: {}, Leadtime: {}".format(
            forecast_reference_time, tmp_leadtime
        )
        ##----------------------------------------------------------------------.
        # Create figure
        # pix = 1/plt.rcParams['figure.dpi'] figsize=(1920*pix,1080*pix),
        fig, axs = plt.subplots(
            nrows=len(variables),
            ncols=3,
            figsize=(18, 4 * len(variables)),  # (8, 2*len(variables))
            subplot_kw={"projection": ccrs.Robinson()},
        )
        fig.suptitle(suptitle_str)
        # fig.subplots_adjust(wspace=0.1, hspace=0.2)
        ##---------------------------------------------------------------------.
        # Initialize
        axs = axs.flatten()
        ax_count = 0
        ##---------------------------------------------------------------------.
        # Plot each variable
        for var in variables:
            # Plot obs
            tmp_obs = ds_dict["obs"][var].isel(time=i)
            xsphere.plot.plot(
                tmp_obs,
                ax=axs[ax_count],
                edgecolors=edgecolors,
                antialiased=antialiased,
                vmin=get_var_clim(var, "state")[0],
                vmax=get_var_clim(var, "state")[1],
                cmap=get_var_cmap(var, "state"),
            )
            axs[ax_count].set_title(None)
            axs[ax_count].coastlines(alpha=0.2)
            # Plot pred
            tmp_pred = ds_dict["pred"][var].isel(time=i)
            s_p = xsphere.plot.plot(
                tmp_pred,
                ax=axs[ax_count + 1],
                edgecolors=edgecolors,
                antialiased=antialiased,
                vmin=get_var_clim(var, "state")[0],
                vmax=get_var_clim(var, "state")[1],
                cmap=get_var_cmap(var, "state"),
            )
            axs[ax_count + 1].set_title(None)
            axs[ax_count + 1].coastlines(alpha=0.2)
            # - Add state colorbar
            cbar = fig.colorbar(
                s_p,
                ax=axs[[ax_count, ax_count + 1]],
                orientation="horizontal",
                extend="both",
                aspect=aspect_cbar,
            )
            cbar.set_label(var.upper())
            cbar.ax.xaxis.set_label_position("top")
            # Plot error
            tmp_error = ds_dict["error"][var].isel(time=i)
            e_p = xsphere.plot.plot(
                tmp_error,
                ax=axs[ax_count + 2],
                edgecolors=edgecolors,
                antialiased=antialiased,
                vmin=get_var_clim(var, "error")[0],
                vmax=get_var_clim(var, "error")[1],
                cmap=get_var_cmap(var, "error"),
                # add_colorbar = True,
                # cbar_kwargs={'orientation': 'horizontal',
                #              'label': var.upper() + " Error"}
            )
            axs[ax_count + 2].set_title(None)
            axs[ax_count + 2].coastlines(alpha=0.2)
            # - Add error colorbar
            # cb = plt.colorbar(e_p, ax=axs[ax_count+2], orientation="horizontal") # pad=0.15)
            # cb.set_label(label=var.upper() + " Error") # size='large', weight='bold'
            cbar_err = fig.colorbar(
                e_p,
                ax=axs[ax_count + 2],
                orientation="horizontal",
                extend="both",
                aspect=aspect_cbar / 2,
            )
            cbar_err.set_label(var.upper() + " Error")
            cbar_err.ax.xaxis.set_label_position("top")
            # Add plot labels
            # if ax_count == 0:
            axs[ax_count].set_title("Observed")
            axs[ax_count + 1].set_title("Predicted")
            axs[ax_count + 2].set_title("Error")
            # Update ax_count
            ax_count += 3
        ##---------------------------------------------------------------------.
        # Save figure in temporary directory
        fig.savefig(os.path.join(tmp_dir, "{:04}.png".format(i)))
        # Close the figure
        plt.close(fig)
    ##-------------------------------------------------------------------------.
    ## Create MP4 and GIF with FFMPEG
    # --> YouTube and Vimeo won’t really appreciate video with < 0.5 FPS
    # --> Duplicate frames by specifying (again) the desired FPS before -codec
    # -y : overwrite existing file
    # -r:v 30 : write at 30 frames per seconds
    # -r 4: write 4 frames per seconds
    # -r:v 1/4 : write a frame every 4 seconds
    # -codec:v lix264
    # Create MP4
    cmd = 'ffmpeg -r:v {} -i "{}/%04d.png" -codec:v libx264 -preset placebo -an -y "{}.mp4"'.format(
        fps, tmp_dir, gif_fpath
    )
    capture_output = not verbose
    _ = subprocess.run(cmd, shell=True, capture_output=capture_output)
    # Create GIF
    cmd = 'ffmpeg -i {}.mp4 -vf "fps={},scale=2560:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 -y {}.gif'.format(
        gif_fpath, fps, gif_fpath
    )
    _ = subprocess.run(cmd, shell=True, capture_output=capture_output)

    ##-------------------------------------------------------------------------.
    # Remove temporary images
    shutil.rmtree(tmp_dir)

    ##-------------------------------------------------------------------------.


##------------------------------------------------------------------------------.
def create_gif_forecast_anom_error(
    gif_fpath,
    ds_forecast,
    ds_obs,
    scaler,
    anom_title,
    fps=4,
    aspect_cbar=40,
    antialiased=False,
    edgecolors=None,
    verbose=False,
):
    ##-------------------------------------------------------------------------.
    # Check Datasets have mesh attached
    if "mesh" not in list(ds_forecast.coords.keys()):
        raise ValueError("No 'mesh' coordinate in ds_forecast.")
    if "mesh" not in list(ds_obs.coords.keys()):
        raise ValueError("No 'mesh' coordinate in ds_obs.")
    ##-------------------------------------------------------------------------.
    # Retrieve forecast reference time
    forecast_reference_time = str(
        ds_forecast["forecast_reference_time"].values.astype("datetime64[s]")
    )  # remove nanosecs

    # Load data into memory
    ds_forecast = ds_forecast.load()

    # Retrieve valid time
    valid_time = (
        ds_forecast["forecast_reference_time"].values + ds_forecast["leadtime"].values
    )
    ds_forecast = ds_forecast.assign_coords({"time": ("leadtime", valid_time)})
    ds_forecast = ds_forecast.swap_dims({"leadtime": "time"})

    # Subset observations and load in memory
    ds_obs = ds_obs.sel(time=ds_forecast["time"].values)
    ds_obs = ds_obs.load()

    # Compute anomalies
    ds_obs = scaler.transform(ds_obs).compute()
    ds_forecast = scaler.transform(ds_forecast).compute()

    # Compute error
    ds_error = ds_forecast - ds_obs
    ds_error = ds_error.assign_coords({"mesh": ("node", ds_obs["mesh"].values)})

    # Create a dictionary with relevant infos
    ds_dict = {"pred": ds_forecast, "obs": ds_obs, "error": ds_error}

    # Retrieve common variables to plot
    variables = list(ds_forecast.data_vars.keys())

    ##-------------------------------------------------------------------------.
    # Check GIF fpath
    if not os.path.exists(os.path.dirname(gif_fpath)):
        os.makedirs(os.path.dirname(gif_fpath))

    # Check GIF fpath ends with .gif
    if gif_fpath[-4:] == ".gif":
        gif_fpath = gif_fpath[:-4]

    # Create temporary directory to store temporary GIF image frames
    tmp_dir = tempfile.mkdtemp()

    ##-------------------------------------------------------------------------.
    # Create GIF image frames for each leadtime
    for i in range(len(ds_forecast["leadtime"])):
        # Select frame super title
        tmp_leadtime = str(ds_forecast["leadtime"].values[i].astype("timedelta64[h]"))
        # tmp_valid_time = str(ds_forecast['time'].values[i].astype('datetime64[s]'))
        suptitle_str = "Forecast reference time: {}, Leadtime: {}".format(
            forecast_reference_time, tmp_leadtime
        )
        ##---------------------------------------------------------------------.
        # Create figure
        fig, axs = plt.subplots(
            nrows=len(variables),
            ncols=3,
            figsize=(18, 4 * len(variables)),  # # (8, 2*len(variables))
            subplot_kw={"projection": ccrs.Robinson()},
        )
        fig.suptitle(suptitle_str)
        # fig.subplots_adjust(wspace=0.1, hspace=0.2)
        ##---------------------------------------------------------------------.
        # Initialize
        axs = axs.flatten()
        ax_count = 0
        ##---------------------------------------------------------------------.
        # Plot each variable
        for var in variables:
            # Plot obs
            tmp_obs = ds_dict["obs"][var].isel(time=i)
            xsphere.plot.plot(
                tmp_obs,
                ax=axs[ax_count],
                edgecolors=edgecolors,
                antialiased=antialiased,
                vmin=-4,
                vmax=4,
                cmap=plt.get_cmap("Spectral_r"),
            )
            axs[ax_count].set_title(None)
            axs[ax_count].coastlines(alpha=0.2)
            # Plot pred
            tmp_pred = ds_dict["pred"][var].isel(time=i)
            s_p = xsphere.plot.plot(
                tmp_pred,
                ax=axs[ax_count + 1],
                edgecolors=edgecolors,
                antialiased=antialiased,
                vmin=-4,
                vmax=4,
                cmap=plt.get_cmap("Spectral_r"),
            )
            axs[ax_count + 1].set_title(None)
            axs[ax_count + 1].coastlines(alpha=0.2)
            # - Add state colorbar
            cbar = fig.colorbar(
                s_p,
                ax=axs[[ax_count, ax_count + 1]],
                orientation="horizontal",
                extend="both",
                aspect=aspect_cbar,
            )
            cbar.set_label(var.upper() + " " + anom_title)
            cbar.ax.xaxis.set_label_position("top")
            # Plot error
            tmp_error = ds_dict["error"][var].isel(time=i)
            e_p = xsphere.plot.plot(
                tmp_error,
                ax=axs[ax_count + 2],
                edgecolors=edgecolors,
                antialiased=antialiased,
                vmin=-6,
                vmax=6,
                cmap=get_var_cmap(var, "error"),
                # add_colorbar = True,
                # cbar_kwargs={'orientation': 'horizontal',
                #              'label': var.upper() + " Error"}
            )
            axs[ax_count + 2].set_title(None)
            axs[ax_count + 2].coastlines(alpha=0.2)
            # - Add error colorbar
            # cb = plt.colorbar(e_p, ax=axs[ax_count+2], orientation="horizontal") # pad=0.15)
            # cb.set_label(label=var.upper() + " Error") # size='large', weight='bold'
            cbar_err = fig.colorbar(
                e_p,
                ax=axs[ax_count + 2],
                orientation="horizontal",
                extend="both",
                aspect=aspect_cbar / 2,
            )
            cbar_err.set_label(var.upper() + " " + anom_title + " Error")
            cbar_err.ax.xaxis.set_label_position("top")
            # Add plot labels
            # if ax_count == 0:
            axs[ax_count].set_title("Observed")
            axs[ax_count + 1].set_title("Predicted")
            axs[ax_count + 2].set_title("Error")
            # Update ax_count
            ax_count += 3
        ##---------------------------------------------------------------------.
        # Save figure in temporary directory
        fig.savefig(os.path.join(tmp_dir, "{:04}.png".format(i)))
        # Close the figure
        plt.close(fig)
    ##-------------------------------------------------------------------------.
    ## Create MP4 and GIF with FFMPEG
    # --> YouTube and Vimeo won’t really appreciate video with < 0.5 FPS
    # --> Duplicate frames by specifying (again) the desired FPS before -codec
    # -y : overwrite existing file
    # -r:v 30 : write at 30 frames per seconds
    # -r 4: write 4 frames per seconds
    # -r:v 1/4 : write a frame every 4 seconds
    # -codec:v lix264
    # Create MP4
    cmd = 'ffmpeg -r:v {} -i "{}/%04d.png" -codec:v libx264 -preset placebo -an -y "{}.mp4"'.format(
        fps, tmp_dir, gif_fpath
    )
    capture_output = not verbose
    _ = subprocess.run(cmd, shell=True, capture_output=capture_output)
    # Create GIF
    cmd = 'ffmpeg -i {}.mp4 -vf "fps={},scale=2560:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 -y {}.gif'.format(
        gif_fpath, fps, gif_fpath
    )
    capture_output = not verbose
    _ = subprocess.run(cmd, shell=True, capture_output=capture_output)

    ##-------------------------------------------------------------------------.
    # Remove temporary images
    shutil.rmtree(tmp_dir)
    ##-------------------------------------------------------------------------.


def create_gif_forecast_evolution(
    gif_fpath,
    dict_data,
    # Plot options
    aspect_cbar=40,
    antialiased=False,
    edgecolors=None,
    # GIF options
    fps=30,
    create_gif=True,
    verbose=False,
):
    ##------------------------------------------------------------------------.
    # Check dict_data
    if not isinstance(dict_data, dict):
        raise TypeError(
            "dict_data must be a dictionary with sub-keys ('data','clim','cmap')."
        )

    for k in dict_data.keys():
        # Check there is a subdictionary with keys ('data','clim','cmap')
        if not isinstance(dict_data[k], dict):
            raise ValueError(
                "For {!r}', you must specify a subdictionary with keys ('data','clim','cmap').".format(
                    k
                )
            )
        required_subkeys = np.array(["data", "clim", "cmap"])
        subkeys = list(dict_data[k].keys())
        missing_subkeys = required_subkeys[
            np.isin(required_subkeys, subkeys, invert=True)
        ]
        if len(missing_subkeys) > 0:
            raise ValueError(
                "{!r} miss the following subkeys: {}".format(k, missing_subkeys)
            )

        # Check data are in the expected format
        x = dict_data[k]["data"]
        if not x.sphere.has_mesh():
            raise ValueError("No 'mesh' coordinate in data.")
        if not isinstance(x, (xr.DataArray, xr.Dataset)):
            raise TypeError("Provide a list of xr.DataArray or xr.Dataset objects.")
        if not xr_has_dim(x, dim="time"):
            raise ValueError("Provide xarray objects with the 'time' dimension.")
        if isinstance(x, xr.Dataset):
            if xr_n_vars(x) != 1:
                raise ValueError(
                    "Unless providing a single xr.Dataset, each xr.Dataset in the provided list must have an unique variable."
                )
            # Convert to DataArray
            dict_data[k]["data"] = x[xr_Dataset_vars(x)[0]]

    ##------------------------------------------------------------------------.
    # Align DataArray
    l_DataArrays = [dict_data[k]["data"] for k in dict_data.keys()]
    l_DataArrays = list(xr.align(*l_DataArrays))

    ##------------------------------------------------------------------------.
    # Load the data
    for i, k in enumerate(list(dict_data.keys())):
        dict_data[k]["data"] = l_DataArrays[i].compute()

    ##------------------------------------------------------------------------.
    # Compute hovmollers
    print("- Computing hovmoller diagram")
    dict_hovmollers = {}
    for k, d in dict_data.items():
        dict_hovmollers[k] = HovmollerDiagram(
            d["data"],
            time_dim="time",
            time_groups=None,
            spatial_dim="lat",
            bin_width=5,
            time_average_before_binning=True,
        )

    ##------------------------------------------------------------------------.
    # Check GIF fpath
    if not os.path.exists(os.path.dirname(gif_fpath)):
        os.makedirs(os.path.dirname(gif_fpath))

    # Remove gif file format
    if gif_fpath[-4:] == ".gif":
        gif_fpath = gif_fpath[:-4]

    # Create temporary directory to store temporary GIF image frames
    tmp_dir = tempfile.mkdtemp()

    ##-----------------------------------------------------------------------.
    # Retrieve timesteps
    timesteps = l_DataArrays[0]["time"].values
    n_timesteps = len(timesteps)

    # Retrieve number of DataArrays to plot
    n_rows = len(dict_data)

    ##-----------------------------------------------------------------------.
    # i = n_timesteps-1
    # antialiased = False
    # edgecolors = None
    ##-----------------------------------------------------------------------.
    ### Create GIF image frames for each timestep
    for i in range(n_timesteps):
        tmp_timestep = timesteps[i]
        tmp_timestep_str = str(tmp_timestep.astype("datetime64[m]"))

        ##----------------------------------------------------------------------.
        # Define figure settings
        # pix = 1/plt.rcParams['figure.dpi'] figsize=(1920*pix,1080*pix),
        figsize = (17, 4 * n_rows)
        crs_proj = ccrs.Robinson()
        fig = plt.figure(figsize=figsize)

        # Define figure grids
        gs = gridspec.GridSpec(
            nrows=n_rows,
            ncols=2,
            # height_ratios=[1, 6],
            hspace=0.03,
            wspace=0.1,
        )
        axs = []
        for row in range(n_rows):
            # Left plot (geographic map)
            axs.append(fig.add_subplot(gs[row, 0], projection=crs_proj))
            # Right plot (hovmoller plot)
            axs.append(fig.add_subplot(gs[row, 1]))

        ##---------------------------------------------------------------------.
        # Plot each DataArray
        for row, (k, d) in enumerate(dict_data.items()):
            # Retrieve infos
            data = d["data"]
            vmin, vmax = d["clim"]
            cmap = d["cmap"]
            clabel = k
            cbar_kwargs = {"label": clabel}
            # Plot field map
            tmp_map = data.isel(time=i)
            tmp_map.sphere.plot(
                ax=axs[row * 2],
                edgecolors=edgecolors,
                antialiased=antialiased,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
            axs[row * 2].set_title(None)
            axs[row * 2].coastlines(alpha=0.2)
            # Plot hovmoller
            tmp_hovmoller = dict_hovmollers[k]
            tmp_hovmoller = tmp_hovmoller.where(tmp_hovmoller.time <= tmp_timestep)
            # mask tmp_hovmoller till > i
            _ = tmp_hovmoller.plot(
                ax=axs[row * 2 + 1],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                add_colorbar=True,
                cbar_kwargs=cbar_kwargs,
            )
            axs[row * 2 + 1].set_ylabel("Latitude")
            axs[row * 2 + 1].set_title(None)
            # Display title ony on top plot
            if row == 0:
                axs[row * 2 + 0].set_title(tmp_timestep_str)
                axs[row * 2 + 1].set_title("Hovmoller diagram")
            # Display time axis only on the bottom plot
            if row != n_rows - 1:
                axs[row * 2 + 1].get_xaxis().set_visible(False)

        ##---------------------------------------------------------------------.
        # Save figure in temporary directory
        fig.savefig(os.path.join(tmp_dir, "{:04}.png".format(i)))
        # Close the figure
        plt.close(fig)

    ##-------------------------------------------------------------------------.
    ## Create MP4 and GIF with FFMPEG
    # --> YouTube and Vimeo won’t really appreciate video with < 0.5 FPS
    # --> Duplicate frames by specifying (again) the desired FPS before -codec
    # -y : overwrite existing file
    # -r:v 30 : write at 30 frames per seconds
    # -r 4: write 4 frames per seconds
    # -r:v 1/4 : write a frame every 4 seconds
    # -codec:v lix264

    # Create MP4
    cmd = 'ffmpeg -r:v {} -i "{}/%04d.png" -codec:v libx264 -preset placebo -an -y "{}.mp4"'.format(
        fps, tmp_dir, gif_fpath
    )
    capture_output = not verbose
    _ = subprocess.run(cmd, shell=True, capture_output=capture_output)

    # Create GIF
    if create_gif:
        cmd = 'ffmpeg -i {}.mp4 -vf "fps={},scale=2560:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 -y {}.gif'.format(
            gif_fpath, fps, gif_fpath
        )
        capture_output = not verbose
        _ = subprocess.run(cmd, shell=True, capture_output=capture_output)

    ##-------------------------------------------------------------------------.
    # Remove temporary images
    shutil.rmtree(tmp_dir)
