"""
Created on Fri Feb 12 21:44:06 2021

@author: ghiggi
"""

import os
import sys
sys.path.append('../')
import time
import xarray as xr
import dask
import warnings
from modules.xscaler import GlobalStandardScaler   
from modules.xscaler import GlobalMinMaxScaler
from modules.xscaler import AnomalyScaler
from modules.xscaler import Climatology

warnings.filterwarnings("ignore")

# Notes: tisr have outliers ...up to 8 std anomaly deviations !!!

#### Compute scalers and climatology 
def compute_scalers(sampling_name, base_data_dir, reference_period, DATA_ARRAY_FLAG=False):
    ##------------------------------------------------------------------------.
    ### Define the sampling-specific folder 
    data_dir = os.path.join(base_data_dir, sampling_name)
    print("Data directory:", data_dir)
    ##------------------------------------------------------------------------.
    ### Define zarr fpaths
    # - Define path for dynamic data (i.e. pressure and surface levels variables)
    dynamic_fpath = os.path.join(data_dir, "Data","dynamic", "space_chunked", "dynamic.zarr")
    # - Define path for boundary conditions data (i.e. TOA)
    bc_fpath = os.path.join(data_dir, "Data","bc", "space_chunked", "bc.zarr")
    ##------------------------------------------------------------------------.
    ### Read data 
    if DATA_ARRAY_FLAG:
        # - Define dask chunks (per thread available)
        dask_chunks = {'feature': 1, # each single feature 
                    'time': -1,   # all across time
                    'node': 400,  # 2.73 MB per disk chunk --> 1092 MB each chunk 
                    }

        variable_dim="feature"
        # - Read data 
        print("- Reading Datasets")
        data_dynamic = xr.open_zarr(dynamic_fpath)['data']
        data_bc = xr.open_zarr(bc_fpath)['data']
    else: 
        # - Define dask chunks (per thread available)
        dask_chunks = {'time': -1,   # all across time
                       'node': 200,  # 2.73 MB per disk chunk --> 546 MB each chunk 
                      }
        variable_dim=None
        # - Read data 
        print("- Reading Datasets")
        data_dynamic = xr.open_zarr(dynamic_fpath) 
        data_bc = xr.open_zarr(bc_fpath)

    ##------------------------------------------------------------------------.
    ### Redefine dask chunks to speed up calculatins
    data_dynamic = data_dynamic.chunk(dask_chunks)
    data_bc = data_bc.chunk(dask_chunks)
   
    ##------------------------------------------------------------------------.
    # #### Define Global Standard Scaler
    print("- Compute Global Standard Scaler")
    dynamic_scaler = GlobalStandardScaler(data=data_dynamic, variable_dim=variable_dim)
    dynamic_scaler.fit()
    dynamic_scaler.save(os.path.join(data_dir, "Scalers", "GlobalStandardScaler_dynamic.nc"))
    del dynamic_scaler

    bc_scaler = GlobalStandardScaler(data=data_bc, variable_dim=variable_dim)
    bc_scaler.fit()
    bc_scaler.save(os.path.join(data_dir, "Scalers", "GlobalStandardScaler_bc.nc"))
    del bc_scaler  

    # ##------------------------------------------------------------------------.
    # #### Define Global MinMax Scaler
    print("- Compute  Global MinMax Scaler")
    dynamic_scaler = GlobalMinMaxScaler(data=data_dynamic, variable_dim=variable_dim)
    dynamic_scaler.fit()
    dynamic_scaler.save(os.path.join(data_dir, "Scalers", "GlobalMinMaxScaler_dynamic.nc"))
    del dynamic_scaler

    bc_scaler = GlobalMinMaxScaler(data=data_bc, variable_dim=variable_dim)
    bc_scaler.fit()
    bc_scaler.save(os.path.join(data_dir, "Scalers", "GlobalMinMaxScaler_bc.nc"))
    del bc_scaler

    ##------------------------------------------------------------------------.
    #### Define Monthly Standard Anomaly Scaler  
    print("- Compute Monthly Standard Anomaly Scaler")
    dynamic_scaler = AnomalyScaler(data_dynamic, time_dim = "time", time_groups = "month", 
                                   variable_dim = variable_dim,
                                   groupby_dims = 'node',
                                   standardized = True,
                                   reference_period = reference_period)
    dynamic_scaler.fit()
    dynamic_scaler.save(os.path.join(data_dir, "Scalers", "MonthlyStdAnomalyScaler_dynamic.nc"))
  

    bc_scaler = AnomalyScaler(data_bc, time_dim = "time", time_groups = "month", 
                              variable_dim = variable_dim,
                              groupby_dims = 'node',
                              standardized = True,
                              reference_period = reference_period)
    bc_scaler.fit()
    bc_scaler.save(os.path.join(data_dir, "Scalers", "MonthlyStdAnomalyScaler_bc.nc"))

    # - Define Global MinMax Scaler of Monthly Standard Anomalies fields 
    # --> Limit standard anomalies between -8 and 8
    print("- Compute Global MinMax Scaler of Monthly Standard Anomaly Scaler") 
    data_dynamic_anom = dynamic_scaler.transform(data_dynamic).clip(-8, 8)  
    dynamic_minmax_scaler = GlobalMinMaxScaler(data=data_dynamic_anom, variable_dim=variable_dim)
    dynamic_minmax_scaler.fit()
    dynamic_minmax_scaler.save(os.path.join(data_dir, "Scalers", "MinMaxScaler_MonthlyStdAnomaly_dynamic.nc"))
    del dynamic_scaler
    del dynamic_minmax_scaler
    del data_dynamic_anom

    data_bc_anom = bc_scaler.transform(data_bc).clip(-8, 8)
    bc_minmax_scaler = GlobalMinMaxScaler(data=data_bc_anom, variable_dim=variable_dim)
    bc_minmax_scaler.fit()
    bc_minmax_scaler.save(os.path.join(data_dir, "Scalers", "MinMaxScaler_MonthlyStdAnomaly_bc.nc"))
    del bc_minmax_scaler
    del data_bc_anom
    del bc_scaler

    ##------------------------------------------------------------------------.
    #### Define Weekly Standard Anomaly Scaler  
    print("- Compute Weekly Standard Anomaly Scaler")
    dynamic_scaler = AnomalyScaler(data_dynamic, time_dim = "time", time_groups = "weekofyear", 
                                   variable_dim = variable_dim,
                                   groupby_dims = 'node',
                                   standardized = True,
                                   reference_period = reference_period)
    dynamic_scaler.fit()
    dynamic_scaler.save(os.path.join(data_dir, "Scalers", "WeeklyStdAnomalyScaler_dynamic.nc"))
  
    bc_scaler = AnomalyScaler(data_bc, time_dim = "time", time_groups = "weekofyear", 
                              variable_dim = variable_dim,
                              groupby_dims = 'node',
                              standardized = True,
                              reference_period = reference_period)
    bc_scaler.fit()
    bc_scaler.save(os.path.join(data_dir, "Scalers", "WeeklyStdAnomalyScaler_bc.nc"))

    # - Define Global MinMax Scaler of Weekly Standard Anomalies fields 
    # --> Limit standard anomalies between -8 and 8 
    print("- Compute Global MinMax Scaler of Weekly Standard Anomaly Scaler") 
    data_dynamic_anom = dynamic_scaler.transform(data_dynamic).clip(-8, 8)     
    dynamic_minmax_scaler = GlobalMinMaxScaler(data=data_dynamic_anom, variable_dim=variable_dim)
    dynamic_minmax_scaler.fit()
    dynamic_minmax_scaler.save(os.path.join(data_dir, "Scalers", "MinMaxScaler_WeeklyStdAnomaly_dynamic.nc"))
    del dynamic_scaler
    del dynamic_minmax_scaler
    del data_dynamic_anom

    data_bc_anom = bc_scaler.transform(data_bc).clip(-8, 8) 
    bc_minmax_scaler = GlobalMinMaxScaler(data=data_bc_anom, variable_dim=variable_dim)
    bc_minmax_scaler.fit()
    bc_minmax_scaler.save(os.path.join(data_dir, "Scalers", "MinMaxScaler_WeeklyStdAnomaly_bc.nc"))
    del bc_minmax_scaler
    del data_bc_anom
    del bc_scaler

    ##------------------------------------------------------------------------.
    #### Define Hourly Weekly Standard Anomaly Scaler  
    print("- Compute Hourly Weekly Standard Anomaly Scaler")
    dynamic_scaler = AnomalyScaler(data_dynamic, time_dim = "time", time_groups = ["weekofyear", "hour"],
                                   variable_dim = variable_dim,
                                   groupby_dims = 'node',
                                   standardized = True,
                                   reference_period = reference_period)
    dynamic_scaler.fit()
    dynamic_scaler.save(os.path.join(data_dir, "Scalers", "WeeklyHourlyStdAnomalyScaler_dynamic.nc"))
    
    bc_scaler = AnomalyScaler(data_bc, time_dim = "time", time_groups = ["weekofyear", "hour"], 
                              variable_dim = variable_dim,
                              groupby_dims = 'node',
                              standardized = True,
                              reference_period = reference_period)
    bc_scaler.fit()
    bc_scaler.save(os.path.join(data_dir, "Scalers", "WeeklyHourlyStdAnomalyScaler_bc.nc"))
    
    # - Define Global MinMax Scaler of Hourly Weekly Standard Anomalies fields 
    # --> Limit standard anomalies between -8 and 8 
    print("- Compute Global MinMax Scaler of Hourly Weekly Standard Anomaly Scaler") 
    data_dynamic_anom = dynamic_scaler.transform(data_dynamic).clip(-8, 8) 
     
    dynamic_minmax_scaler = GlobalMinMaxScaler(data=data_dynamic_anom, variable_dim=variable_dim)
    dynamic_minmax_scaler.fit()
    dynamic_minmax_scaler.save(os.path.join(data_dir, "Scalers", "MinMaxScaler_WeeklyHourlyStdAnomaly_dynamic.nc"))
    del dynamic_scaler
    del dynamic_minmax_scaler
    del data_dynamic_anom

    data_bc_anom = bc_scaler.transform(data_bc).clip(-8, 8) 
    bc_minmax_scaler = GlobalMinMaxScaler(data=data_bc_anom, variable_dim=variable_dim)
    bc_minmax_scaler.fit()
    bc_minmax_scaler.save(os.path.join(data_dir, "Scalers", "MinMaxScaler_WeeklyHourlyStdAnomaly_bc.nc"))
    del bc_scaler
    del bc_minmax_scaler
    del data_bc_anom

    ##------------------------------------------------------------------------.
    #### Compute Monthly climatology
    print("- Compute Monthly Climatology")
    monthly_clim = Climatology(data = data_dynamic,
                               time_dim = 'time',
                               time_groups = "month", 
                               variable_dim = variable_dim, 
                               groupby_dims = "node",  
                               reference_period = reference_period, 
                               mean = True, variability=True)
    monthly_clim.compute()
    monthly_clim.save(os.path.join(data_dir, "Climatology", "MonthlyClimatology_dynamic.nc"))  
    del monthly_clim

    ##------------------------------------------------------------------------.
    #### Compute Weekly climatology
    print("- Compute Weekly Climatology")
    weekly_clim = Climatology(data = data_dynamic,
                              time_dim = 'time',
                              time_groups = "weekofyear",  
                              variable_dim = variable_dim,
                              groupby_dims = "node",  
                              reference_period = reference_period, 
                              mean = True, variability=True)
    weekly_clim.compute()
    weekly_clim.save(os.path.join(data_dir, "Climatology", "WeeklyClimatology_dynamic.nc"))
    del weekly_clim

    ##------------------------------------------------------------------------.
    #### Compute Daily climatology
    print("- Compute Daily Climatology")
    daily_clim = Climatology(data = data_dynamic,
                             time_dim = 'time',
                             time_groups = "dayofyear",  
                             variable_dim = variable_dim,
                             groupby_dims = "node",  
                             reference_period = reference_period, 
                             mean = True, variability=True)
    daily_clim.compute()
    daily_clim.save(os.path.join(data_dir, "Climatology", "DailyClimatology_dynamic.nc"))  
    del daily_clim

    ##------------------------------------------------------------------------.
    #### Compute Hourly Monthly climatology 
    print("- Compute Hourly Monthly Climatology")
    hourlymonthly_clim = Climatology(data = data_dynamic,
                                     time_dim = 'time',
                                     time_groups = ['hour', 'month'],  
                                     variable_dim = variable_dim,
                                     groupby_dims = "node",  
                                     reference_period = reference_period, 
                                     mean = True, variability=True)
    hourlymonthly_clim.compute()
    hourlymonthly_clim.save(os.path.join(data_dir, "Climatology", "HourlyMonthlyClimatology_dynamic.nc"))  
    del hourlymonthly_clim

    ##------------------------------------------------------------------------.
    #### Compute Hourly Weekly climatology 
    print("- Compute Hourly Weekly Climatology")
    hourlyweekly_clim = Climatology(data = data_dynamic,
                                    time_dim = 'time',
                                    time_groups = ['hour', 'weekofyear'],  
                                    variable_dim = variable_dim,
                                    groupby_dims = "node",  
                                    reference_period = reference_period, 
                                    mean = True, variability=True)
    hourlyweekly_clim.compute()
    hourlyweekly_clim.save(os.path.join(data_dir, "Climatology", "HourlyWeeklyClimatology_dynamic.nc"))  
    del hourlyweekly_clim

    ##------------------------------------------------------------------------.

#-----------------------------------------------------------------------------.
if __name__ == '__main__':
    ##------------------------------------------------------------------------.
    ### Set dask configs 
    # - By default, Xarray and dask.array use thee multi-threaded scheduler (dask.config.set(scheduler='threads')
    # - 'num_workers' defaults to the number of cores
    # - dask.config.set(scheduler='threads') # Uses a ThreadPoolExecutor in the local process
    # - dask.config.set(scheduler='processes') # Uses a ProcessPoolExecutor to spread work between processes
    
    # - Set array.chunk-size default
    dask.config.set({"array.chunk-size": "1024 MiB"})
    # - Avoid to split large dask chunks 
    dask.config.set(**{'array.slicing.split_large_chunks': False})
    
    ##------------------------------------------------------------------------.
    ### Define data settings
    # - Define data directory
    DATA_ARRAY_FLAG = False
    # DATA_ARRAY_FLAG = True
    base_data_dir = "/ltenas3/DeepSphere/data/preprocessed_ds/ERA5_HRES/"
    # base_data_dir = "/ltenas3/DeepSphere/data/preprocessed/ERA5_HRES/"
        
    # - Define samplings
    sampling_name_list = ['Cubed_400km','O24','Cubed_400km',
                          'Healpix_400km',
                          'Equiangular_400km','Equiangular_400km_tropics',
                          'Icosahedral_400km']
    # - Define reference period (for computing climatological statistics)
    reference_period = ('1980-01-07T00:00','2010-12-31T23:00')
    
    ##------------------------------------------------------------------------.
    # Launch computations 
    for sampling_name in sampling_name_list:
        print("==================================================================")
        print("- Computing scalers for", sampling_name, "data")
        t_i = time.time()
        #----------------------------------------------------------------------.
        compute_scalers(sampling_name = sampling_name, 
                        base_data_dir = base_data_dir, 
                        reference_period = reference_period, DATA_ARRAY_FLAG = DATA_ARRAY_FLAG)
        #----------------------------------------------------------------------.
        # Report elapsed time 
        print("---> Elapsed time: {:.1f} minutes ".format((time.time() - t_i)/60))
        print("==================================================================")