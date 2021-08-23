#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 17:50:11 2021

@author: ghiggi
"""
import os
import glob
import sys
import shutil
import argparse
import xarray as xr 
import modules.xverif as xverif

from modules.utils_config import read_config_file
from modules.utils_config import get_model_settings
from modules.my_plotting import plot_skill_maps
from modules.my_plotting import plot_global_skill
from modules.my_plotting import plot_global_skills
from modules.my_plotting import plot_skills_distribution

### TODO: This script is in development 
# - Check same sampling 
# - Define chunking of deep ensemble 
# - Plot deterministic skill of each model (ens median skill in bold)

#-----------------------------------------------------------------------------.
def main(data_dir, 
         model_dirs,
         DeepEnsemble_dir,
         remove_model_dirs=False, force=False):
    ##------------------------------------------------------------------------.
    ### TODO: 
    # check exist DeepEnsemble_dir
    # force
    ##------------------------------------------------------------------------.
    # Define paths
    ens_forecast_fpath = os.path.join(DeepEnsemble_dir, f"model_predictions/temporal_chunks/test_pred_ens.zarr")
    ens_median_forecast_fpath = os.path.join(DeepEnsemble_dir, f"model_predictions/temporal_chunks/test_pred_ens_median.zarr")
    ##------------------------------------------------------------------------.
    # Get configs 
    cfg_paths = [os.path.join(model_dir[0], "config.json") for model_dir in model_dirs]
    cfg_list = [read_config_file(fpath=cfg_path) for cfg_path in cfg_paths]
    # TODO: check same sampling 
    ##------------------------------------------------------------------------.
    #### Load Datasets
    model_settings = get_model_settings(cfg_list[0]) 
    data_sampling_dir = os.path.join(data_dir, model_settings["sampling_name"])
    ds_dynamic = xr.open_zarr(os.path.join(data_sampling_dir, "Data","dynamic", "time_chunked", "dynamic.zarr"))["data"]

    ##------------------------------------------------------------------------.
    ### - Create an ensemble prediction Dataset
    ens_members_fpaths = [os.path.join(model_dir, "model_predictions/temporal_chunks/test_pred.zarr") for model_dir in model_dirs]
    list_ds_member = [xr.open_zarr(fpath) for fpath in ens_members_fpaths]
    ds_ensemble = xr.concat(list_ds_member, dim="member")
    del list_ds_member

    ### - Save ensemble prediction Dataset
    if not os.path.exists(ens_forecast_fpath):
        ds_ensemble.to_zarr(ens_forecast_fpath, mode='w') # Create
    else:                        
        ds_ensemble.to_zarr(ens_forecast_fpath, append_dim='member') # Append

    ### - Remove optionally individual model_dir 
    if remove_model_dirs:
        for model_dir in model_dirs:
            shutil.rmtree(model_dir)

    ##------------------------------------------------------------------------.
    ### - Compute ensemble median 
    ds_ensemble = xr.open_zarr(ens_forecast_fpath, chunks="auto")
    ds_ensemble_median = ds_ensemble.median(dim="member")
    ds_ensemble_median.to_zarr(ens_median_forecast_fpath, mode='w')  
    del ds_ensemble
    del ds_ensemble_median

    ##------------------------------------------------------------------------.
    ### - Run deterministic verification 
    print("========================================================================================")
    print("- Run deterministic verification")
    # dask.config.set(scheduler='processes')
    # - Compute skills
    ds_ensemble_median = xr.open_zarr(ens_median_forecast_fpath, chunks="auto")
    ds_obs_test , ds_ensemble_median = xr.align(ds_dynamic, ds_ensemble_median) # TODO.. left side?
    ds_obs_test = ds_obs_test.chunk({'time': -1,'node': 1})
    ds_skill = xverif.deterministic(pred = ds_ensemble_median,
                                    obs = ds_obs_test, 
                                    forecast_type="continuous",
                                    aggregating_dim='time')
    # - Save sptial skills 
    ds_skill.to_netcdf(os.path.join(DeepEnsemble_dir, f"model_skills/deterministic_spatial_skill.nc"))

    ##------------------------------------------------------------------------.
    print("========================================================================================")
    print("- Run probabilistic verification")
    # TODO : prob skills
    ds_ensemble_median = xr.open_zarr(ens_forecast_fpath, chunks="auto")

    ##------------------------------------------------------------------------.
    ### - Create verification summary plots and maps
    print("========================================================================================")
    print("- Create verification summary plots and maps")
    # - Add mesh information 
    ds_skill = ds_skill.sphere.add_SphericalVoronoiMesh(x='lon', y='lat')

    # - Compute global and latitudinal skill summary statistics    
    ds_global_skill = xverif.global_summary(ds_skill, area_coords="area")
    # ds_latitudinal_skill = xverif.latitudinal_summary(ds_skill, lat_dim='lat', lon_dim='lon', lat_res=5) 
    # ds_longitudinal_skill = xverif.longitudinal_summary(ds_skill, lat_dim='lat', lon_dim='lon', lon_res=5) 

    # - Save global skills
    ds_global_skill.to_netcdf(os.path.join(DeepEnsemble_dir, "model_skills/deterministic_global_skill.nc"))

    # - Create spatial maps 
    plot_skill_maps(ds_skill = ds_skill,  
                    figs_dir = os.path.join(DeepEnsemble_dir, "figs/skills/SpatialSkill"),
                    crs_proj = ccrs.Robinson(),
                    skills = ['BIAS','RMSE','rSD', 'pearson_R2', 'error_CoV'],
                    # skills = ['percBIAS','percMAE','rSD', 'pearson_R2', 'KGE'],
                    suffix="",
                    prefix="")

    # - Create skill vs. leadtime plots 
    plot_global_skill(ds_global_skill).savefig(os.path.join(DeepEnsemble_dir, "figs/skills/RMSE_skill.png"))
    plot_global_skills(ds_global_skill).savefig(os.path.join(DeepEnsemble_dir, "figs/skills/skills_global.png"))
    plot_skills_distribution(ds_skill).savefig(os.path.join(DeepEnsemble_dir, "figs/skills/skills_distribution.png"))
        
    ##-------------------------------------------------------------------------.                                     
    print("   ---> Elapsed time: {:.1f} minutes ".format((time.time() - t_i)/60))
    ##-------------------------------------------------------------------------.                            
    print("========================================================================================")
    print("- DeepEnsemble verification terminated. Elapsed time: {:.1f} hours ".format((time.time() - t_start)/60/60))  
    print("========================================================================================")

if __name__ == '__main__':
    default_data_dir = "/ltenas3/DeepSphere/data/preprocessed/ERA5_HRES"
    default_model_dirs = glob.glob("/data/weather_prediction/experiments/exp_DeepEnsemble/*")
    default_model_dirs = " ".join(default_model_dirs)
    default_DeepEnsemble_dir = "/data/weather_prediction/experiments/exp_DeepEnsemble/DeepEnsemble"
 
    parser = argparse.ArgumentParser(description='Verify a DeepEnsemble of DeepSphere-Weather models')
    parser.add_argument('--data_dir', type=str, default=default_data_dir)
    parser.add_argument('--DeepEnsemble_dir', type=str, default=default_DeepEnsemble_dir)
    parser.add_argument('--model_dirs', nargs="*", default=default_model_dirs)
    parser.add_argument('--remove_model_dirs', type=str, default='False')
    parser.add_argument('--force', type=str, default='True')                    
    
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] ="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda  
    if args.remove_model_dirs == 'True':
        remove_model_dirs = True
    else: 
        remove_model_dirs = False
        
    if args.force == 'True':
        force = True
    else: 
        force = False
        
    main(data_dir = args.data_dir,
         model_dirs = args.model_dirs, 
         DeepEnsemble_dir = args.DeepEnsemble_dir, 
         remove_model_dirs = remove_model_dirs, 
         force = force)

 