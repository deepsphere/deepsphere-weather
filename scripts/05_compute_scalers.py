"""
Created on Fri Feb 12 21:44:06 2021

@author: ghiggi
"""
import os
import sys
import xarray as xr
sys.path.append('../')

from modules.xscaler import GlobalStandardScaler   
from modules.xscaler import GlobalMinMaxScaler
from modules.xscaler import AnomalyScaler
from modules.xscaler import Climatology
from modules.my_io import readDatasets   
# Notes: tisr have outliers ...up to 8 std anomaly deviations !!!

##----------------------------------------------------------------------------.
# - Define sampling data directory
base_data_dir = "/data/weather_prediction/data"
# - Define samplings
sampling_name_list = ['Healpix_400km','Equiangular_400km','Equiangular_400km_tropics',
                      'Icosahedral_400km','O24','Cubed_400km']
# - Define reference period (for computing climatological statistics)
reference_period = ('1980-01-01T00:00','2010-12-31T23:00')

#### Compute scalers and climatology 
for sampling_name in sampling_name_list:
    ### Define the sampling-specific folder 
    data_dir = os.path.join(base_data_dir, sampling_name)
    print(data_dir)
    ##------------------------------------------------------------------------.
    ### Load data 
    # - Dynamic data (i.e. pressure and surface levels variables)
    ds_dynamic = readDatasets(data_dir=data_dir, feature_type='dynamic')
    # - Boundary conditions data (i.e. TOA)
    ds_bc = readDatasets(data_dir=data_dir, feature_type='bc')
    # - Static features
    ds_static = readDatasets(data_dir=data_dir, feature_type='static')
    
    ds_dynamic = ds_dynamic.drop(["level","lat","lon"])
    ds_bc = ds_bc.drop(["lat","lon"])
    ds_static = ds_static.drop(["lat","lon"])
  
    ds_dynamic = ds_dynamic.load()
    ds_bc = ds_bc.load()
    ds_static = ds_static.load()
    ##------------------------------------------------------------------------.
    #### Define Global Standard Scaler
    dynamic_scaler = GlobalStandardScaler(data=ds_dynamic)
    dynamic_scaler.fit()
    dynamic_scaler.save(os.path.join(data_dir, "Scalers", "GlobalStandardScaler_dynamic.nc"))
     
    bc_scaler = GlobalStandardScaler(data=ds_bc)
    bc_scaler.fit()
    bc_scaler.save(os.path.join(data_dir, "Scalers", "GlobalStandardScaler_bc.nc"))
    
    static_scaler = GlobalStandardScaler(data=ds_static)
    static_scaler.fit()
    static_scaler.save(os.path.join(data_dir, "Scalers", "GlobalStandardScaler_static.nc"))
    
    ##------------------------------------------------------------------------.
    #### Define Global MinMax Scaler
    dynamic_scaler = GlobalMinMaxScaler(data=ds_dynamic)
    dynamic_scaler.fit()
    dynamic_scaler.save(os.path.join(data_dir, "Scalers", "GlobalMinMaxScaler_dynamic.nc"))
     
    bc_scaler = GlobalMinMaxScaler(data=ds_bc)
    bc_scaler.fit()
    bc_scaler.save(os.path.join(data_dir, "Scalers", "GlobalMinMaxScaler_bc.nc"))
    
    static_scaler = GlobalMinMaxScaler(data=ds_static)
    static_scaler.fit()
    static_scaler.save(os.path.join(data_dir, "Scalers", "GlobalMinMaxScaler_static.nc"))
    
    ##------------------------------------------------------------------------.
    #### Define Monthly Standard Anomaly Scaler  
    dynamic_scaler = AnomalyScaler(ds_dynamic, time_dim = "time", time_groups = "month", 
                                   groupby_dims = 'node',
                                   standardized = True,
                                   reference_period = reference_period)
    dynamic_scaler.fit()
    dynamic_scaler.save(os.path.join(data_dir, "Scalers", "MonthlyStdAnomalyScaler_dynamic.nc"))
    
    bc_scaler = AnomalyScaler(ds_bc, time_dim = "time", time_groups = "month", 
                              groupby_dims = 'node',
                              standardized = True,
                              reference_period = reference_period)
    bc_scaler.fit()
    bc_scaler.save(os.path.join(data_dir, "Scalers", "MonthlyStdAnomalyScaler_bc.nc"))
    
    # - Define Global MinMax Scaler of Monthly Standard Anomalies fields 
    # --> Limit standard anomalies between -8 and 8 
    ds_dynamic_anom = dynamic_scaler.transform(ds_dynamic).compute() 
    ds_dynamic_anom = xr.where(ds_dynamic_anom > 8, 8, ds_dynamic_anom) 
    ds_dynamic_anom = xr.where(ds_dynamic_anom < -8, -8, ds_dynamic_anom) 
     
    dynamic_minmax_scaler = GlobalMinMaxScaler(data=ds_dynamic_anom)
    dynamic_minmax_scaler.fit()
    dynamic_minmax_scaler.save(os.path.join(data_dir, "Scalers", "MinMaxScaler_MonthlyStdAnomaly_dynamic.nc"))
    
    ds_bc_anom = bc_scaler.transform(ds_bc).compute()
    ds_bc_anom = xr.where(ds_bc_anom > 8, 8, ds_bc_anom) 
    ds_bc_anom = xr.where(ds_bc_anom < -8, -8, ds_bc_anom) 
    bc_minmax_scaler = GlobalMinMaxScaler(data=ds_bc_anom)
    bc_minmax_scaler.fit()
    bc_minmax_scaler.save(os.path.join(data_dir, "Scalers", "MinMaxScaler_MonthlyStdAnomaly_bc.nc"))
    
    ##------------------------------------------------------------------------.
    #### Define Weekly Standard Anomaly Scaler  
    dynamic_scaler = AnomalyScaler(ds_dynamic, time_dim = "time", time_groups = "weekofyear", 
                                   groupby_dims = 'node',
                                   standardized = True,
                                   reference_period = reference_period)
    dynamic_scaler.fit()
    dynamic_scaler.save(os.path.join(data_dir, "Scalers", "WeeklyStdAnomalyScaler_dynamic.nc"))
    
    bc_scaler = AnomalyScaler(ds_bc, time_dim = "time", time_groups = "weekofyear", 
                              groupby_dims = 'node',
                              standardized = True,
                              reference_period = reference_period)
    bc_scaler.fit()
    bc_scaler.save(os.path.join(data_dir, "Scalers", "WeeklyStdAnomalyScaler_bc.nc"))
    
    # - Define Global MinMax Scaler of Weekly Standard Anomalies fields 
    # --> Limit standard anomalies between -8 and 8 
    ds_dynamic_anom = dynamic_scaler.transform(ds_dynamic).compute() 
    ds_dynamic_anom = xr.where(ds_dynamic_anom > 8, 8, ds_dynamic_anom) 
    ds_dynamic_anom = xr.where(ds_dynamic_anom < -8, -8, ds_dynamic_anom) 
     
    dynamic_minmax_scaler = GlobalMinMaxScaler(data=ds_dynamic_anom)
    dynamic_minmax_scaler.fit()
    dynamic_minmax_scaler.save(os.path.join(data_dir, "Scalers", "MinMaxScaler_WeeklyStdAnomaly_dynamic.nc"))
    
    ds_bc_anom = bc_scaler.transform(ds_bc).compute()
    ds_bc_anom = xr.where(ds_bc_anom > 8, 8, ds_bc_anom) 
    ds_bc_anom = xr.where(ds_bc_anom < -8, -8, ds_bc_anom) 
    bc_minmax_scaler = GlobalMinMaxScaler(data=ds_bc_anom)
    bc_minmax_scaler.fit()
    bc_minmax_scaler.save(os.path.join(data_dir, "Scalers", "MinMaxScaler_WeeklyStdAnomaly_bc.nc"))
    
    ##------------------------------------------------------------------------.
    #### Define Hourly Weekly Standard Anomaly Scaler  
    dynamic_scaler = AnomalyScaler(ds_dynamic, time_dim = "time", time_groups = ["weekofyear", "hour"],
                                   groupby_dims = 'node',
                                   standardized = True,
                                   reference_period = reference_period)
    dynamic_scaler.fit()
    dynamic_scaler.save(os.path.join(data_dir, "Scalers", "WeeklyHourlyStdAnomalyScaler_dynamic.nc"))
    
    bc_scaler = AnomalyScaler(ds_bc, time_dim = "time", time_groups = ["weekofyear", "hour"], 
                              groupby_dims = 'node',
                              standardized = True,
                              reference_period = reference_period)
    bc_scaler.fit()
    bc_scaler.save(os.path.join(data_dir, "Scalers", "WeeklyHourlyStdAnomalyScaler_bc.nc"))
    
    # - Define Global MinMax Scaler of Weekly Hourly Standard Anomalies fields 
    # --> Limit standard anomalies between -8 and 8 
    ds_dynamic_anom = dynamic_scaler.transform(ds_dynamic).compute() 
    ds_dynamic_anom = xr.where(ds_dynamic_anom > 8, 8, ds_dynamic_anom) 
    ds_dynamic_anom = xr.where(ds_dynamic_anom < -8, -8, ds_dynamic_anom) 
     
    dynamic_minmax_scaler = GlobalMinMaxScaler(data=ds_dynamic_anom)
    dynamic_minmax_scaler.fit()
    dynamic_minmax_scaler.save(os.path.join(data_dir, "Scalers", "MinMaxScaler_WeeklyHourlyStdAnomaly_dynamic.nc"))
    
    ds_bc_anom = bc_scaler.transform(ds_bc).compute()
    ds_bc_anom = xr.where(ds_bc_anom > 8, 8, ds_bc_anom) 
    ds_bc_anom = xr.where(ds_bc_anom < -8, -8, ds_bc_anom) 
    bc_minmax_scaler = GlobalMinMaxScaler(data=ds_bc_anom)
    bc_minmax_scaler.fit()
    bc_minmax_scaler.save(os.path.join(data_dir, "Scalers", "MinMaxScaler_WeeklyHourlyStdAnomaly_bc.nc"))
    
    ##------------------------------------------------------------------------.
    #### Compute Monthly climatology
    monthly_clim = Climatology(data = ds_dynamic,
                               time_dim = 'time',
                               time_groups= "month",  
                               groupby_dims = "node",  
                               reference_period = reference_period, 
                               mean = True, variability=True)
    monthly_clim.compute()
    monthly_clim.save(os.path.join(data_dir, "Climatology", "MonthlyClimatology_dynamic.nc"))  
    
    ##------------------------------------------------------------------------.
    #### Compute Weekly climatology
    weekly_clim = Climatology(data = ds_dynamic,
                              time_dim = 'time',
                              time_groups= "weekofyear",  
                              groupby_dims = "node",  
                              reference_period = reference_period, 
                              mean = True, variability=True)
    weekly_clim.compute()
    weekly_clim.save(os.path.join(data_dir, "Climatology", "WeeklyClimatology_dynamic.nc"))
    
    ##------------------------------------------------------------------------.
    #### Compute Daily climatology
    daily_clim = Climatology(data = ds_dynamic,
                             time_dim = 'time',
                             time_groups= "dayofyear",  
                             groupby_dims = "node",  
                             reference_period = reference_period, 
                             mean = True, variability=True)
    daily_clim.compute()
    daily_clim.save(os.path.join(data_dir, "Climatology", "DailyClimatology_dynamic.nc"))  
    
    ##------------------------------------------------------------------------.
    #### Compute Hourly Monthly climatology 
    hourlymonthly_clim = Climatology(data = ds_dynamic,
                                     time_dim = 'time',
                                     time_groups= ['hour', 'month'],  
                                     groupby_dims = "node",  
                                     reference_period = reference_period, 
                                     mean = True, variability=True)
    hourlymonthly_clim.compute()
    hourlymonthly_clim.save(os.path.join(data_dir, "Climatology", "HourlyMonthlyClimatology_dynamic.nc"))  
    
    ##------------------------------------------------------------------------.
    #### Compute Hourly Weekly climatology 
    hourlyweekly_clim = Climatology(data = ds_dynamic,
                                    time_dim = 'time',
                                    time_groups= ['hour', 'weekofyear'],  
                                    groupby_dims = "node",  
                                    reference_period = reference_period, 
                                    mean = True, variability=True)
    hourlyweekly_clim.compute()
    hourlyweekly_clim.save(os.path.join(data_dir, "Climatology", "HourlyWeeklyClimatology_dynamic.nc"))  

    ##------------------------------------------------------------------------.



    
