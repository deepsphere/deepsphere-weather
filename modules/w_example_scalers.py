#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 12:27:05 2021

@author: ghiggi
"""
import os
import xarray as xr 
import numpy as np
os.chdir("/home/ghiggi/Projects/weather_prediction/")

from modules.xscaler import LoadScaler
from modules.xscaler import GlobalStandardScaler
from modules.xscaler import TemporalStandardScaler
from modules.xscaler import SequentialScaler
from modules.my_io import readDatasets   

##----------------------------------------------------------------------------.
data_dir = "/home/ghiggi/Projects/DeepSphere/ToyData/Healpix_400km/" # to change to scratch/... 
# - Dynamic data (i.e. pressure and surface levels variables)
ds_dynamic = readDatasets(data_dir=data_dir, feature_type='dynamic')
# - Boundary conditions data (i.e. TOA)
ds_bc = readDatasets(data_dir=data_dir, feature_type='bc')
# - Static features
ds_static = readDatasets(data_dir=data_dir, feature_type='static')

ds_dynamic = ds_dynamic.drop(["level","lat","lon"])
ds_bc = ds_bc.drop(["lat","lon"])
ds_static = ds_static.drop(["lat","lon"])

##----------------------------------------------------------------------------.
# ds = ds_static
# xr.testing.assert_identical
# xr.testing.assert_equal
# xr.ALL_DIMS # ...  
##----------------------------------------------------------------------------.
# ######################
#### GlobalScalers #####
# ######################
ds = ds_dynamic.compute()
da = ds.to_array(dim='feature', name='whatever').transpose('time', 'node', 'feature')

# Dataset 
gs = GlobalStandardScaler(data=ds)
gs.fit() 
ds_trans = gs.transform(ds).compute()
ds_invert = gs.inverse_transform(ds_trans).compute()
xr.testing.assert_equal(ds, ds_invert)

# DataArray with variable dimension
gs = GlobalStandardScaler(data=da, variable_dim="feature")
gs.fit()
da_trans = gs.transform(da, variable_dim="feature").compute()
da_invert = gs.inverse_transform(da_trans, variable_dim="feature").compute()
xr.testing.assert_equal(da, da_invert)

# DataArray without variable dimension
da1 = ds['z500']
gs = GlobalStandardScaler(data=da1, variable_dim=None)
gs.fit()
da_trans = gs.transform(da1, variable_dim=None).compute()
da_invert = gs.inverse_transform(da_trans, variable_dim=None).compute()
xr.testing.assert_equal(da1, da_invert)

# DataArray with groupby option (i.e. scaling each member ...)
gs = GlobalStandardScaler(data=da, variable_dim="feature", groupby_dims="node")  
gs.fit()
da_trans = gs.transform(da, variable_dim="feature").compute()
da_invert = gs.inverse_transform(da_trans, variable_dim="feature").compute()
xr.testing.assert_equal(da, da_invert)

# DataArray with variable_dimension but not specified  
gs = GlobalStandardScaler(data=da, variable_dim=None)
gs.fit()
da_trans = gs.transform(da, variable_dim=None).compute()
da_invert = gs.inverse_transform(da_trans, variable_dim=None).compute()
# xr.testing.assert_equal(da, da_invert)     # Not equal
xr.testing.assert_allclose(da, da_invert)  # Actually close ... 

##----------------------------------------------------------------------------.
### - Fit with DataSet - Transform with DataArray 
gs = GlobalStandardScaler(data=ds)
gs.fit()
da_trans = gs.transform(da, variable_dim="feature").compute()
da_invert = gs.inverse_transform(da_trans, variable_dim="feature").compute()
xr.testing.assert_equal(da, da_invert)

### - Fit with DataArray - Transform with Dataset  
gs = GlobalStandardScaler(data=da, variable_dim="feature")
gs.fit()
ds_trans = gs.transform(ds).compute()
ds_invert = gs.inverse_transform(ds_trans).compute()
xr.testing.assert_equal(ds, ds_invert) 

##----------------------------------------------------------------------------.
### - Write and load from disk 
fpath = "/home/ghiggi/scaler_test.nc"
gs = GlobalStandardScaler(data=ds)
gs.fit()
gs.save(fpath)

gs = LoadScaler(fpath)

# Dataset
ds_trans = gs.transform(ds).compute()
ds_invert = gs.inverse_transform(ds_trans).compute()
xr.testing.assert_equal(ds, ds_invert)

# DataArray (with variable dimension)
da_trans = gs.transform(da, variable_dim="feature").compute()
da_invert = gs.inverse_transform(da_trans, variable_dim="feature").compute()
xr.testing.assert_equal(da, da_invert)

# DataArray (without variable dimension)
da_trans = gs.transform(ds['z500'], variable_dim=None).compute()
da_invert = gs.inverse_transform(da_trans, variable_dim=None).compute()
xr.testing.assert_equal(ds['z500'], da_invert) 

##-----------------------------------------------------------------------------.
# ########################
#### Temporal Scalers ####
# ########################
# Pixelwise --> groupby_dims = "node"
# Anomalies: center=True, standardize=False
# Standardized Anomalies: center=True, standardize=True
from modules.xscaler import TemporalStandardScaler

ds = ds_dynamic.compute()
da = ds.to_array(dim='feature', name='whatever').transpose('time', 'node', 'feature')

variable_dim = 'feature'
time_dim = 'time'
groupby_dims="node" 
time_groups = ['month','day']
time_groups = ['dayofyear']
time_groups = ["hour","weekofyear"]
time_groups = 'season'

time_groups = {'hour': 6, 
               'month': 2}

gs = TemporalStandardScaler(data=ds, 
                            time_dim=time_dim, 
                            time_groups=None)
gs.fit()
ds_trans = gs.transform(ds).compute()
ds_invert = gs.inverse_transform(ds_trans).compute()
xr.testing.assert_equal(ds, ds_invert)

# Dataset 
gs = TemporalStandardScaler(data=ds, 
                            time_dim=time_dim, 
                            time_groups=time_groups)
gs.fit()
ds_trans = gs.transform(ds).compute()
ds_invert = gs.inverse_transform(ds_trans).compute()
xr.testing.assert_equal(ds, ds_invert)

# DataArray with variable dimension
gs = TemporalStandardScaler(data=da, 
                            variable_dim=variable_dim, 
                            time_dim=time_dim,
                            time_groups=time_groups)
gs.fit()
da_trans = gs.transform(da, variable_dim=variable_dim).compute()
da_invert = gs.inverse_transform(da_trans, variable_dim=variable_dim).compute()
xr.testing.assert_equal(da, da_invert)

# DataArray without variable dimension
da1 = ds['z500']
gs = TemporalStandardScaler(data=da1, 
                            variable_dim=None, 
                            time_dim=time_dim, 
                            time_groups=time_groups)
gs.fit()
da_trans = gs.transform(da1, variable_dim=None).compute()
da_invert = gs.inverse_transform(da_trans, variable_dim=None).compute()
xr.testing.assert_equal(da1, da_invert)

ds_trans = gs.transform(ds, variable_dim=None).compute()
ds_invert = gs.inverse_transform(ds_trans, variable_dim=None).compute()
xr.testing.assert_equal(ds, ds_invert)

# DataArray with groupby option (i.e. scaling each member ...)
gs = TemporalStandardScaler(data=da, 
                            variable_dim=variable_dim, 
                            groupby_dims=groupby_dims, 
                            time_dim=time_dim,
                            time_groups=time_groups)
gs.fit()
da_trans = gs.transform(da, variable_dim=variable_dim).compute()
da_invert = gs.inverse_transform(da_trans, variable_dim=variable_dim).compute()
xr.testing.assert_equal(da, da_invert)

# DataArray with variable_dimension but not specified  
gs = TemporalStandardScaler(data=da, 
                            variable_dim=None,
                            time_dim=time_dim,
                            time_groups=time_groups)
gs.fit()
da_trans = gs.transform(da, variable_dim=None).compute()
da_invert = gs.inverse_transform(da_trans, variable_dim=None).compute()
# xr.testing.assert_equal(da, da_invert)     # Not equal
xr.testing.assert_allclose(da, da_invert)  # Actually close ... 

##----------------------------------------------------------------------------.
## Fit with DataSet - Transform with DataArray 
gs = TemporalStandardScaler(data=ds,  
                            time_dim=time_dim,
                            time_groups=time_groups)
gs.fit()
da_trans = gs.transform(da, variable_dim=variable_dim).compute()
da_invert = gs.inverse_transform(da_trans, variable_dim=variable_dim).compute()
xr.testing.assert_equal(da, da_invert)

## Fit with DataArray - Transform with Dataset  
gs = TemporalStandardScaler(data=da, 
                          variable_dim=variable_dim,
                          time_dim=time_dim,
                          time_groups=time_groups)
gs.fit()
ds_trans = gs.transform(ds).compute()
ds_invert = gs.inverse_transform(ds_trans).compute()
xr.testing.assert_equal(ds, ds_invert) 

##----------------------------------------------------------------------------.
# Check consistency 
gs = TemporalStandardScaler(data=ds, 
                            time_dim=time_dim, 
                            time_groups=['day','month'])
gs.fit()
ds_trans = gs.transform(ds).compute()

gs = TemporalStandardScaler(data=ds, 
                            time_dim=time_dim, 
                            time_groups='dayofyear')
gs.fit()

ds_trans1 = gs.transform(ds).compute()

xr.testing.assert_equal(ds_trans, ds_trans1) 

##----------------------------------------------------------------------------.
### - Write and load from disk 
fpath = "/home/ghiggi/scaler_test.nc"
gs = TemporalStandardScaler(data=ds, 
                            time_dim=time_dim, 
                            time_groups=['day','month'])
gs.fit()
gs.save(fpath)

gs = LoadScaler(fpath)
ds_trans = gs.transform(ds).compute()
ds_invert = gs.inverse_transform(ds_trans).compute()
xr.testing.assert_equal(ds, ds_invert)

##----------------------------------------------------------------------------.
### Check rename_dict 
rename_dict = {'time': 'forecast_time', 
               'node': 'space'}
new_ds = ds 
new_ds = new_ds.rename(rename_dict)

# - GlobalScaler
from modules.xscaler import GlobalStandardScaler

groupby_dims = ["node","time"]
groupby_dims = ["node"]
groupby_dims = None
 
gs = GlobalStandardScaler(data=ds, groupby_dims=groupby_dims)
gs.fit()

ds_trans1 = gs.transform(new_ds).compute()
ds_trans = gs.transform(new_ds, rename_dict=rename_dict).compute()

ds_invert1 = gs.inverse_transform(ds_trans).compute()
ds_invert = gs.inverse_transform(ds_trans, rename_dict=rename_dict).compute()
xr.testing.assert_equal(new_ds, ds_invert)
 
# - Temporal scaler
from modules.xscaler import TemporalStandardScaler 
time_dim = 'time'
groupby_dims = "node" 
groupby_dims = None
time_groups = ['month','day']
time_groups = {'hour': 6, 
               'month': 2}
time_groups = None

gs = TemporalStandardScaler(data=ds, 
                            time_dim=time_dim, 
                            time_groups=time_groups,
                            groupby_dims=groupby_dims)
gs.fit()

gs.mean_ 

ds_trans1 = gs.transform(new_ds).compute()
ds_trans = gs.transform(new_ds, rename_dict=rename_dict).compute()

ds_invert1 = gs.inverse_transform(ds_trans).compute()
ds_invert = gs.inverse_transform(ds_trans, rename_dict=rename_dict).compute()
xr.testing.assert_equal(new_ds, ds_invert)
 
##-----------------------------------------------------------------------------.
# ########################
#### SequentialScaler ####
# ########################
from modules.xscaler import SequentialScaler

# Pixelwise over all time   
scaler1 = GlobalStandardScaler(data=ds['z500'], groupby_dims="node")  
# Pixelwise per month 
scaler2 = TemporalStandardScaler(data=ds['t850'], time_dim = "time",  
                                 time_groups="month", groupby_dims="node")

final_scaler = SequentialScaler(scaler1, scaler2)

final_scaler.fit()
print(final_scaler.list_scalers)

ds_trans = final_scaler.transform(ds).compute()
ds_invert = final_scaler.inverse_transform(ds_trans).compute()
xr.testing.assert_equal(ds, ds_invert)

##----------------------------------------------------------------------------.
# Global (over space) 
scaler1 = GlobalStandardScaler(data=ds, groupby_dims="node")  
# Global (over time) 
scaler2 = GlobalStandardScaler(data=ds, groupby_dims="time")  

final_scaler = SequentialScaler(scaler1, scaler2)
final_scaler.fit()
print(final_scaler.list_scalers)

ds_trans = final_scaler.transform(ds).compute()
ds_invert = final_scaler.inverse_transform(ds_trans).compute()
xr.testing.assert_allclose(ds, ds_invert)

##----------------------------------------------------------------------------.
# Pixelwise over all time   
scaler1 = GlobalStandardScaler(data=ds['z500'], groupby_dims="node")  
# Pixelwise per month 
scaler2 = TemporalStandardScaler(data=ds['t850'], time_dim = "time",  
                                 time_groups="month", groupby_dims="node")
# Global (over space) 
scaler3 = GlobalStandardScaler(data=ds, groupby_dims="node")  # minmax

final_scaler = SequentialScaler(scaler1, scaler2, scaler3)

final_scaler.fit()
print(final_scaler.list_scalers)

ds_trans = final_scaler.transform(ds).compute()
ds_invert = final_scaler.inverse_transform(ds_trans).compute()
xr.testing.assert_allclose(ds, ds_invert)

##----------------------------------------------------------------------------.
 


##----------------------------------------------------------------------------.
# ###################
#### Climatology ####
# ###################
from modules.xscaler import Climatology
from modules.xscaler import LoadClimatology
### Daily climatology
daily_clim = Climatology(data = ds_dynamic,
                         time_dim = 'time',
                         time_groups= ['day', 'month'],  
                         groupby_dims = "node",  
                         mean = True, variability=True)
daily_clim1 = Climatology(data = ds_dynamic,
                         time_dim = 'time',
                         time_groups= 'dayofyear',
                         groupby_dims = "node",  
                         mean = True, variability=True)
# - Compute the climatology
daily_clim.compute()
daily_clim1.compute()

print(daily_clim.mean)
print(daily_clim1.mean)
print(daily_clim.variability)
print(daily_clim1.variability)

# - Forecast climatology 
ds_forecast = daily_clim.forecast(ds_dynamic['time'].values)
ds_forecast1 = daily_clim.forecast(ds_dynamic['time'].values)
print(ds_forecast)
xr.testing.assert_allclose(ds_forecast, ds_forecast1)

### 3-hourly weekly climatology
custom_clim = Climatology(data = ds_dynamic,
                          time_dim = 'time',
                          time_groups= {'hour': 3, 'weekofyear': 1},  
                          groupby_dims = "node",  
                          mean = True, variability=True)
# - Compute the climatology
custom_clim.compute()

print(custom_clim.mean)
print(custom_clim.variability)

# - Forecast climatology 
ds_forecast = custom_clim.forecast(ds_dynamic['time'].values)

# - Save
fpath = "/home/ghiggi/clim_test.nc"
custom_clim.save(fpath)

# - Reload
custom_clim = LoadClimatology(fpath)

# - Forecast
custom_clim.forecast(ds_dynamic['time'].values)
##----------------------------------------------------------------------------.
 