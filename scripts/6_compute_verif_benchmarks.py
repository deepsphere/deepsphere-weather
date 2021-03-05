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
from modules.xscaler import Climatology
from modules.xscaler import LoadClimatology
import modules.xverif as xverif
##----------------------------------------------------------------------------.
# - Define sampling data directory
base_data_dir = "/data/weather_prediction/data"
# - Define samplings
sampling_name_list = ['Healpix_400km','Equiangular_400km','Equiangular_400km_tropics',
                      'Icosahedral_400km','O24','Cubed_400km']
# - Define climatologies to analyze
list_climatologies = ["MonthlyClimatology", "WeeklyClimatology", "DailyClimatology", 
                      "HourlyMonthlyClimatology", "HourlyWeeklyClimatology"]
    
#### Compute climatology and persistence skills  
for sampling_name in sampling_name_list:
    
    ### Define the sampling-specific folder 
    data_dir = os.path.join(base_data_dir, sampling_name)
    print(data_dir)
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
        # TODO: 
        # pygsp_graph = get_pygsp_graph(sampling = model_settings['sampling'], 
        #                               resolution = model_settings['resolution'],
        #                               knn = model_settings['knn'])
        # ds_skill = ds_skill.sphere.add_nodes_from_pygsp(pygsp_graph=pygsp_graph)
        ds_skill = ds_skill.sphere.add_SphericalVoronoiMesh(x='lon', y='lat')
        ds_global_skill = xverif.global_summary(ds_skill, area_coords="area")
        ds_global_skill.to_netcdf(os.path.join(data_dir, "Benchmarks", clim_name + "_Global_Skills.nc"))
     
    ##------------------------------------------------------------------------.
    #### Compute persistence forecast skils 
    forecast_cycle = 6 
    AR_iterations = 20
    leadtimes = np.arange(1, AR_iterations)*np.timedelta64(forecast_cycle, 'h')
    
    dt * timedelta(milliseconds=1)
     ds_test_dynamic
    