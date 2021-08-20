"""
Created on Fri Feb 12 21:44:06 2021

@author: ghiggi
"""

import os
import sys
sys.path.append('../')
import time
import xarray as xr
from multiprocessing import Pool
from functools import partial

from modules.xscaler import GlobalStandardScaler   
from modules.xscaler import GlobalMinMaxScaler
from modules.xscaler import AnomalyScaler
from modules.xscaler import Climatology

# Notes: tisr have outliers ...up to 8 std anomaly deviations !!!

#### Compute scalers and climatology 
def compute_scalers(sampling_name, base_data_dir, reference_period):
    ### Define the sampling-specific folder 
    data_dir = os.path.join(base_data_dir, sampling_name)
    print("Data directory:", data_dir)
    ##------------------------------------------------------------------------.
    ### Load data 
    # - Dynamic data (i.e. pressure and surface levels variables)
    dynamic_fpath = os.path.join(data_dir, "Data","dynamic", "time_chunked", "dynamic.zarr")
    da_dynamic = xr.open_zarr(dynamic_fpath)['data']
    # - Boundary conditions data (i.e. TOA)
    bc_fpath = os.path.join(data_dir, "Data","bc", "time_chunked", "bc.zarr")
    da_bc = xr.open_zarr(bc_fpath)['data']
    # - Static features
    # static_fpath = os.path.join(data_dir, "Data", "static.zarr")
    # ds_static = xr.open_zarr(static_fpath)
    
    # - Load in memory the data 
    da_dynamic = da_dynamic.load()
    da_bc = da_bc.load()
   
    ##------------------------------------------------------------------------.
    #### Define Global Standard Scaler
    dynamic_scaler = GlobalStandardScaler(data=da_dynamic, variable_dim="feature")
    dynamic_scaler.fit()
    dynamic_scaler.save(os.path.join(data_dir, "Scalers", "GlobalStandardScaler_dynamic.nc"))
     
    bc_scaler = GlobalStandardScaler(data=da_bc, variable_dim="feature")
    bc_scaler.fit()
    bc_scaler.save(os.path.join(data_dir, "Scalers", "GlobalStandardScaler_bc.nc"))
        
    ##------------------------------------------------------------------------.
    #### Define Global MinMax Scaler
    dynamic_scaler = GlobalMinMaxScaler(data=da_dynamic, variable_dim="feature")
    dynamic_scaler.fit()
    dynamic_scaler.save(os.path.join(data_dir, "Scalers", "GlobalMinMaxScaler_dynamic.nc"))
     
    bc_scaler = GlobalMinMaxScaler(data=da_bc, variable_dim="feature")
    bc_scaler.fit()
    bc_scaler.save(os.path.join(data_dir, "Scalers", "GlobalMinMaxScaler_bc.nc"))
    
    ##------------------------------------------------------------------------.
    #### Define Monthly Standard Anomaly Scaler  
    dynamic_scaler = AnomalyScaler(da_dynamic, time_dim = "time", time_groups = "month", 
                                   variable_dim = "feature",
                                   groupby_dims = 'node',
                                   standardized = True,
                                   reference_period = reference_period)
    dynamic_scaler.fit()
    dynamic_scaler.save(os.path.join(data_dir, "Scalers", "MonthlyStdAnomalyScaler_dynamic.nc"))
    
    bc_scaler = AnomalyScaler(da_bc, time_dim = "time", time_groups = "month", 
                              variable_dim = "feature",
                              groupby_dims = 'node',
                              standardized = True,
                              reference_period = reference_period)
    bc_scaler.fit()
    bc_scaler.save(os.path.join(data_dir, "Scalers", "MonthlyStdAnomalyScaler_bc.nc"))
    
    # - Define Global MinMax Scaler of Monthly Standard Anomalies fields 
    # --> Limit standard anomalies between -8 and 8 
    da_dynamic_anom = dynamic_scaler.transform(da_dynamic).compute() 
    da_dynamic_anom = xr.where(da_dynamic_anom > 8, 8, da_dynamic_anom) 
    da_dynamic_anom = xr.where(da_dynamic_anom < -8, -8, da_dynamic_anom) 
     
    dynamic_minmax_scaler = GlobalMinMaxScaler(data=da_dynamic_anom, variable_dim="feature")
    dynamic_minmax_scaler.fit()
    dynamic_minmax_scaler.save(os.path.join(data_dir, "Scalers", "MinMaxScaler_MonthlyStdAnomaly_dynamic.nc"))
    
    da_bc_anom = bc_scaler.transform(da_bc).compute()
    da_bc_anom = xr.where(da_bc_anom > 8, 8, da_bc_anom) 
    da_bc_anom = xr.where(da_bc_anom < -8, -8, da_bc_anom) 
    bc_minmax_scaler = GlobalMinMaxScaler(data=da_bc_anom, variable_dim="feature")
    bc_minmax_scaler.fit()
    bc_minmax_scaler.save(os.path.join(data_dir, "Scalers", "MinMaxScaler_MonthlyStdAnomaly_bc.nc"))
    
    ##------------------------------------------------------------------------.
    #### Define Weekly Standard Anomaly Scaler  
    dynamic_scaler = AnomalyScaler(da_dynamic, time_dim = "time", time_groups = "weekofyear", 
                                   variable_dim = "feature",
                                   groupby_dims = 'node',
                                   standardized = True,
                                   reference_period = reference_period)
    dynamic_scaler.fit()
    dynamic_scaler.save(os.path.join(data_dir, "Scalers", "WeeklyStdAnomalyScaler_dynamic.nc"))
    
    bc_scaler = AnomalyScaler(da_bc, time_dim = "time", time_groups = "weekofyear", 
                              variable_dim = "feature",
                              groupby_dims = 'node',
                              standardized = True,
                              reference_period = reference_period)
    bc_scaler.fit()
    bc_scaler.save(os.path.join(data_dir, "Scalers", "WeeklyStdAnomalyScaler_bc.nc"))
    
    # - Define Global MinMax Scaler of Weekly Standard Anomalies fields 
    # --> Limit standard anomalies between -8 and 8 
    da_dynamic_anom = dynamic_scaler.transform(da_dynamic).compute() 
    da_dynamic_anom = xr.where(da_dynamic_anom > 8, 8, da_dynamic_anom) 
    da_dynamic_anom = xr.where(da_dynamic_anom < -8, -8, da_dynamic_anom) 
     
    dynamic_minmax_scaler = GlobalMinMaxScaler(data=da_dynamic_anom, variable_dim="feature")
    dynamic_minmax_scaler.fit()
    dynamic_minmax_scaler.save(os.path.join(data_dir, "Scalers", "MinMaxScaler_WeeklyStdAnomaly_dynamic.nc"))
    
    da_bc_anom = bc_scaler.transform(da_bc).compute()
    da_bc_anom = xr.where(da_bc_anom > 8, 8, da_bc_anom) 
    da_bc_anom = xr.where(da_bc_anom < -8, -8, da_bc_anom) 
    bc_minmax_scaler = GlobalMinMaxScaler(data=da_bc_anom, variable_dim="feature")
    bc_minmax_scaler.fit()
    bc_minmax_scaler.save(os.path.join(data_dir, "Scalers", "MinMaxScaler_WeeklyStdAnomaly_bc.nc"))
    
    ##------------------------------------------------------------------------.
    #### Define Hourly Weekly Standard Anomaly Scaler  
    dynamic_scaler = AnomalyScaler(da_dynamic, time_dim = "time", time_groups = ["weekofyear", "hour"],
                                   variable_dim = "feature",
                                   groupby_dims = 'node',
                                   standardized = True,
                                   reference_period = reference_period)
    dynamic_scaler.fit()
    dynamic_scaler.save(os.path.join(data_dir, "Scalers", "WeeklyHourlyStdAnomalyScaler_dynamic.nc"))
    
    bc_scaler = AnomalyScaler(da_bc, time_dim = "time", time_groups = ["weekofyear", "hour"], 
                              variable_dim = "feature",
                              groupby_dims = 'node',
                              standardized = True,
                              reference_period = reference_period)
    bc_scaler.fit()
    bc_scaler.save(os.path.join(data_dir, "Scalers", "WeeklyHourlyStdAnomalyScaler_bc.nc"))
    
    # - Define Global MinMax Scaler of Weekly Hourly Standard Anomalies fields 
    # --> Limit standard anomalies between -8 and 8 
    da_dynamic_anom = dynamic_scaler.transform(da_dynamic).compute() 
    da_dynamic_anom = xr.where(da_dynamic_anom > 8, 8, da_dynamic_anom) 
    da_dynamic_anom = xr.where(da_dynamic_anom < -8, -8, da_dynamic_anom) 
     
    dynamic_minmax_scaler = GlobalMinMaxScaler(data=da_dynamic_anom, variable_dim="feature")
    dynamic_minmax_scaler.fit()
    dynamic_minmax_scaler.save(os.path.join(data_dir, "Scalers", "MinMaxScaler_WeeklyHourlyStdAnomaly_dynamic.nc"))
    
    da_bc_anom = bc_scaler.transform(da_bc).compute()
    da_bc_anom = xr.where(da_bc_anom > 8, 8, da_bc_anom) 
    da_bc_anom = xr.where(da_bc_anom < -8, -8, da_bc_anom) 
    bc_minmax_scaler = GlobalMinMaxScaler(data=da_bc_anom, variable_dim="feature")
    bc_minmax_scaler.fit()
    bc_minmax_scaler.save(os.path.join(data_dir, "Scalers", "MinMaxScaler_WeeklyHourlyStdAnomaly_bc.nc"))
    
    ##------------------------------------------------------------------------.
    #### Compute Monthly climatology
    monthly_clim = Climatology(data = da_dynamic,
                               time_dim = 'time',
                               time_groups = "month", 
                               variable_dim = "feature", 
                               groupby_dims = "node",  
                               reference_period = reference_period, 
                               mean = True, variability=True)
    monthly_clim.compute()
    monthly_clim.save(os.path.join(data_dir, "Climatology", "MonthlyClimatology_dynamic.nc"))  
    
    ##------------------------------------------------------------------------.
    #### Compute Weekly climatology
    weekly_clim = Climatology(data = da_dynamic,
                              time_dim = 'time',
                              time_groups = "weekofyear",  
                              variable_dim = "feature",
                              groupby_dims = "node",  
                              reference_period = reference_period, 
                              mean = True, variability=True)
    weekly_clim.compute()
    weekly_clim.save(os.path.join(data_dir, "Climatology", "WeeklyClimatology_dynamic.nc"))
    
    ##------------------------------------------------------------------------.
    #### Compute Daily climatology
    daily_clim = Climatology(data = da_dynamic,
                             time_dim = 'time',
                             time_groups = "dayofyear",  
                             variable_dim = "feature",
                             groupby_dims = "node",  
                             reference_period = reference_period, 
                             mean = True, variability=True)
    daily_clim.compute()
    daily_clim.save(os.path.join(data_dir, "Climatology", "DailyClimatology_dynamic.nc"))  
    
    ##------------------------------------------------------------------------.
    #### Compute Hourly Monthly climatology 
    hourlymonthly_clim = Climatology(data = da_dynamic,
                                     time_dim = 'time',
                                     time_groups = ['hour', 'month'],  
                                     variable_dim = "feature",
                                     groupby_dims = "node",  
                                     reference_period = reference_period, 
                                     mean = True, variability=True)
    hourlymonthly_clim.compute()
    hourlymonthly_clim.save(os.path.join(data_dir, "Climatology", "HourlyMonthlyClimatology_dynamic.nc"))  
    
    ##------------------------------------------------------------------------.
    #### Compute Hourly Weekly climatology 
    hourlyweekly_clim = Climatology(data = da_dynamic,
                                    time_dim = 'time',
                                    time_groups = ['hour', 'weekofyear'],  
                                    variable_dim = "feature",
                                    groupby_dims = "node",  
                                    reference_period = reference_period, 
                                    mean = True, variability=True)
    hourlyweekly_clim.compute()
    hourlyweekly_clim.save(os.path.join(data_dir, "Climatology", "HourlyWeeklyClimatology_dynamic.nc"))  
    
    ##------------------------------------------------------------------------.

#-----------------------------------------------------------------------------.
if __name__ == '__main__':
    ##------------------------------------------------------------------------.
    # - Define sampling data directory
    base_data_dir = "/ltenas3/DeepSphere/data/preprocessed/ERA5_HRES/"
    # - Define samplings
    sampling_name_list = ['Healpix_400km','Equiangular_400km','Equiangular_400km_tropics',
                          'Icosahedral_400km','O24','Cubed_400km']
    # - Define reference period (for computing climatological statistics)
    reference_period = ('1980-01-07T00:00','2010-12-31T23:00')
    
    # DYNAMIC size 75.62
    # BC size 8 GB 
    # num_workers = min(len(sampling_name_list), os.cpu_count()//2)
    # process_func = partial(compute_scalers, base_data_dir=base_data_dir, reference_period=reference_period)
    # with Pool(num_workers) as p:
    #     p.map(process_func, sampling_name_list)
    for sampling_name in sampling_name_list:
        print("==================================================================")
        print("- Computing scalers for", sampling_name, "data")
        t_i = time.time()
        #----------------------------------------------------------------------.
        compute_scalers(sampling_name,
                        base_data_dir=base_data_dir, 
                        reference_period=reference_period)
        #----------------------------------------------------------------------.
        # Report elapsed time 
        print("---> Elapsed time: {:.1f} minutes ".format((time.time() - t_i)/60))
        print("==================================================================")