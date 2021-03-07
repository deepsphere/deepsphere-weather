#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 19:16:10 2021

@author: ghiggi
"""
import os
import sys
import numpy as np
import xarray as xr
sys.path.append('../')
from modules.my_io import readDatasets   
from modules.xscaler import LoadClimatology
from modules.utils_models import get_pygsp_graph
import modules.xsphere as xsphere #  it load the xarray sphere accessor ! 
import modules.xverif as xverif
##----------------------------------------------------------------------------.
# - Define sampling data directory
base_data_dir = "/data/weather_prediction/data"
# - Define samplings
sampling_infos = {
                #   'Healpix_400km': {'sampling': 'healpix', 
                #                     'resolution': 16},                     
                  'Equiangular_400km': {'sampling': 'equiangular',
                                        'resolution': [36,72]},         
                  'Equiangular_400km_tropics': {'sampling': 'equiangular',
                                                'resolution': [46,92]},    
                  'Icosahedral_400km': {'sampling': 'icosahedral', 
                                        'resolution': 16},                      
                  'O24': {'sampling': 'gauss', 
                          'resolution': 48},  
                  'Cubed_400km': {'sampling': 'cubed', 
                                  'resolution': 24}
                  }
sampling_name_list = list(sampling_infos.keys())

# - Define climatology forecasts to analyze
list_climatologies = ["MonthlyClimatology", "WeeklyClimatology", "DailyClimatology", 
                      "HourlyMonthlyClimatology", "HourlyWeeklyClimatology"]
    
#### Compute climatology and persistence skills  
for sampling_name in sampling_name_list:
    print("- Computing climatology forecasts skills for {} sampling".format(sampling_name))
    ### Define the sampling-specific folder 
    data_dir = os.path.join(base_data_dir, sampling_name)
    ##------------------------------------------------------------------------.
    ### Load netCDF4 Dataset 
    # - Dynamic data (i.e. pressure and surface levels variables)
    ds_dynamic = readDatasets(data_dir=data_dir, feature_type='dynamic')
    ds_dynamic = ds_dynamic.drop(["level","lat","lon"])
    # - Get test set data  
    test_years = np.array(['2017-01-01T00:00','2018-12-31T23:00'], dtype='M8[m]')   
    # - Split data sets     
    ds_test_dynamic = ds_dynamic.sel(time=slice(test_years[0], test_years[-1]))
    ds_test_dynamic = ds_test_dynamic.load()
    ##------------------------------------------------------------------------.
    #### Compute climatology forecast skils 
    for clim_name in list_climatologies:
        print("- Computing forecast skill of", clim_name)
        # - Load Climatology 
        tmp_clim_fpath = os.path.join(data_dir, "Climatology", clim_name + "_dynamic.nc")
        tmp_clim = LoadClimatology(tmp_clim_fpath)  
        # - Create climatology forecasts
        ds_clim_forecast = tmp_clim.forecast(ds_test_dynamic['time'].values)
        # - Rechunk dataset
        ds_test_dynamic = ds_test_dynamic.chunk({'time': -1, 'node': 1})
        ds_clim_forecast = ds_clim_forecast.chunk({'time': -1, 'node': 1})
        # - Compute deterministic spatial skills 
        ds_skill = xverif.deterministic(pred = ds_clim_forecast,
                                        obs = ds_test_dynamic, 
                                        forecast_type="continuous",
                                        aggregating_dim='time')
        # - Save spatial skills
        if not os.path.exists(os.path.join(data_dir, "Benchmarks")):
            os.makedirs(os.path.join(data_dir, "Benchmarks"))
        ds_skill.to_netcdf(os.path.join(data_dir, "Benchmarks", clim_name + "_Spatial_Skills.nc"))
        # - Compute deterministic global skills 
        pygsp_graph = get_pygsp_graph(sampling = sampling_infos[sampling_name]['sampling'], 
                                      resolution = sampling_infos[sampling_name]['resolution'])                        
        ds_skill = ds_skill.sphere.add_nodes_from_pygsp(pygsp_graph=pygsp_graph)
        ds_skill = ds_skill.sphere.add_SphericalVoronoiMesh(x='lon', y='lat')
        ds_global_skill = xverif.global_summary(ds_skill, area_coords="area")
        ds_global_skill.to_netcdf(os.path.join(data_dir, "Benchmarks", clim_name + "_Global_Skills.nc"))
     
    ##------------------------------------------------------------------------.
    #### Compute persistence forecast skils 
    print("- Computing persistence forecasts skills for {} sampling".format(sampling_name))
    # - Rechunk dataset
    ds_test_dynamic = ds_test_dynamic.chunk({'time': -1, 'node': 1})
    # - Define leadtimes 
    forecast_cycle = 6 
    AR_iterations = 40
    leadtimes = np.arange(1, AR_iterations)*np.timedelta64(forecast_cycle, 'h')
    # - Compute persistence forecast at each leadtime
    list_skills = []
    for leadtime in leadtimes:
        lagged_ds = ds_test_dynamic.copy()
        lagged_ds['time'] = lagged_ds['time'] + leadtime
        ds_skill = xverif.deterministic(pred = lagged_ds,
                                        obs = ds_test_dynamic, 
                                        forecast_type="continuous",
                                        aggregating_dim='time')
        ds_skill = ds_skill.assign_coords({'leadtime': np.array(leadtime)})
        ds_skill = ds_skill.expand_dims("leadtime")
        list_skills.append(ds_skill)
    # - Combine peristence forecast skill at all leadtimes 
    ds_persistence_skill = xr.merge(list_skills)
    ds_persistence_skill.to_netcdf(os.path.join(data_dir, "Benchmarks", "Persistence_Spatial_Skills.nc"))
    # - Compute peristence forecast deterministic global skills 
    pygsp_graph = get_pygsp_graph(sampling = sampling_infos[sampling_name]['sampling'], 
                                  resolution = sampling_infos[sampling_name]['resolution'])
    ds_skill = ds_skill.sphere.add_nodes_from_pygsp(pygsp_graph=pygsp_graph)
    ds_skill = ds_skill.sphere.add_SphericalVoronoiMesh(x='lon', y='lat')
    ds_global_skill = xverif.global_summary(ds_skill, area_coords="area")
    ds_global_skill.to_netcdf(os.path.join(data_dir, "Benchmarks", "Persistence_Global_Skills.nc"))
    ##------------------------------------------------------------------------.    
##----------------------------------------------------------------------------.  
