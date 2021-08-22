#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 17:50:11 2021

@author: ghiggi
"""
import os
import sys
import shutil
import xarray as xr 

# TODO: in development 
# - exp_deep_ensemble.py
# - exp_swag.py 

remove_exp_dirs = False 
exp_dir =        # of DeepEnsemble
list_exp_dir =   # models exp_dir

## TODO: define chunking of deep ensemble 

# Define paths
ens_forecast_fpath = os.path.join(exp_dir, f"model_predictions/temporal_chunks/test_pred_ens.zarr")
ens_median_forecast_fpath = os.path.join(exp_dir, f"model_predictions/temporal_chunks/test_pred_ens_median.zarr")
##------------------------------------------------------------------------.
### - Create an ensemble prediction Dataset
ens_members_fpaths = [os.path.join(exp_dir, "model_predictions/temporal_chunks/test_pred.zarr") for model_exp_dir in list_exp_dir]
list_ds_member = [xr.open_zarr(fpath) for fpath in ens_members_fpaths]
ds_ensemble = xr.concat(list_ds_member, dim="member")
del list_ds_member

### - Save ensemble prediction Dataset
if not os.path.exists(ens_forecast_zarr_fpath):
    ds_ensemble.to_zarr(ens_forecast_zarr_fpath, mode='w') # Create
else:                        
    ds_ensemble.to_zarr(ens_forecast_zarr_fpath, append_dim='member') # Append

### - Remove optionally individual exp_dir 
if remove_exp_dirs = True:
    for old_exp_dir in list_exp_dir:
        shutil.rmtree(old_exp_dir)

##------------------------------------------------------------------------.
### - Compute ensemble median 
ds_ensemble = xr.open_zarr(ens_forecast_zarr_fpath, chunks="auto")
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
ds_obs_test = da_test_dynamic.to_dataset('feature')
ds_obs_test = ds_obs_test.chunk({'time': -1,'node': 1})
ds_skill = xverif.deterministic(pred = ds_ensemble_median,
                                obs = ds_obs_test, 
                                forecast_type="continuous",
                                aggregating_dim='time')
# - Save sptial skills 
ds_skill.to_netcdf(os.path.join(exp_dir, f"model_skills/deterministic_spatial_skill.nc"))

##------------------------------------------------------------------------.
print("========================================================================================")
print("- Run probabilistic verification")
# TODO 
ds_ensemble_median = xr.open_zarr(forecast_zarr_fpath, chunks="auto")

##------------------------------------------------------------------------.
### - Create verification summary plots and maps
print("========================================================================================")
print("- Create verification summary plots and maps")
# - Add mesh information 
# ---> TODO: To generalize based on cfg sampling !!!
pygsp_graph = get_pygsp_graph(sampling = model_settings['sampling'], 
                              resolution = model_settings['resolution'],
                              knn = model_settings['knn'])
ds_skill = ds_skill.sphere.add_nodes_from_pygsp(pygsp_graph=pygsp_graph)
ds_skill = ds_skill.sphere.add_SphericalVoronoiMesh(x='lon', y='lat')

# - Compute global and latitudinal skill summary statistics    
ds_global_skill = xverif.global_summary(ds_skill, area_coords="area")
ds_latitudinal_skill = xverif.latitudinal_summary(ds_skill, lat_dim='lat', lon_dim='lon', lat_res=5) 
ds_longitudinal_skill = xverif.longitudinal_summary(ds_skill, lat_dim='lat', lon_dim='lon', lon_res=5) 

# - Save global skills
ds_global_skill.to_netcdf(os.path.join(exp_dir, "model_skills/deterministic_global_skill.nc"))

# - Create spatial maps 
plot_skill_maps(ds_skill = ds_skill,  
                figs_dir = os.path.join(exp_dir, "figs/skills/SpatialSkill"),
                crs_proj = ccrs.Robinson(),
                skills = ['BIAS','RMSE','rSD', 'pearson_R2', 'error_CoV'],
                # skills = ['percBIAS','percMAE','rSD', 'pearson_R2', 'KGE'],
                suffix="",
                prefix="")

# - Create skill vs. leadtime plots 
plot_global_skill(ds_global_skill).savefig(os.path.join(exp_dir, "figs/skills/RMSE_skill.png"))
plot_global_skills(ds_global_skill).savefig(os.path.join(exp_dir, "figs/skills/skills_global.png"))
plot_skills_distribution(ds_skill).savefig(os.path.join(exp_dir, "figs/skills/skills_distribution.png"))
    
##-------------------------------------------------------------------------.                                     
print("   ---> Elapsed time: {:.1f} minutes ".format((time.time() - t_i)/60))
##-------------------------------------------------------------------------.                            
print("========================================================================================")
print("- DeepEnsemble verification terminated. Elapsed time: {:.1f} hours ".format((time.time() - t_start)/60/60))  
print("========================================================================================")