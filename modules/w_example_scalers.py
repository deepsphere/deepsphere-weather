#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 12:27:05 2021

@author: ghiggi
"""
import os
import xarray as xr 
import numpy as np
os.chdir("/home/ghiggi/Projects/DeepSphere/")

from modules.xscaler import LoadScaler
from modules.xscaler import GlobalStandardScaler
from modules.xscaler import TemporalStandardScaler
from modules.my_io import readDatasets   

##----------------------------------------------------------------------------.
data_dir = "/home/ghiggi/Projects/DeepSphere/ToyData/Healpix_400km/data/" # to change to scratch/... 
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
#----------------------------------------------------------------------------.
#######################
### GlobalScalers #####
#######################
ds = ds_dynamic.compute()
da = ds.to_array(dim='feature', name='whatever').transpose('time', 'node', 'feature')

# Dataset 
gs = GlobalStandardScaler(data=ds)
gs.fit()
gs.fit_transform()
ds_trans = gs.transform(ds).compute()
ds_invert = gs.inverse_transform(ds_trans).compute()
xr.testing.assert_equal(ds, ds_invert)

# DataArray with variable dimension
gs = GlobalStandardScaler(data=da, variable_dim="feature")
gs.fit()
gs.fit_transform()
da_trans = gs.transform(da, variable_dim="feature").compute()
da_invert = gs.inverse_transform(da_trans, variable_dim="feature").compute()
xr.testing.assert_equal(da, da_invert)

# DataArray without variable dimension
da1 = ds['z500']
gs = GlobalStandardScaler(data=da1, variable_dim=None)
gs.fit()
gs.fit_transform()
da_trans = gs.transform(da1, variable_dim=None).compute()
da_invert = gs.inverse_transform(da_trans, variable_dim=None).compute()
xr.testing.assert_equal(da1, da_invert)

# DataArray with groupby option (i.e. scaling each member ...)
gs = GlobalStandardScaler(data=da, variable_dim="feature", groupby_dims="node")  
gs.fit()
gs.fit_transform()
da_trans = gs.transform(da, variable_dim="feature").compute()
da_invert = gs.inverse_transform(da_trans, variable_dim="feature").compute()
xr.testing.assert_equal(da, da_invert)

# DataArray with variable_dimension but not specified  
gs = GlobalStandardScaler(data=da, variable_dim=None)
gs.fit()
gs.fit_transform()
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

#-----------------------------------------------------------------------------.
##########################
#### Temporal Scalers ####
##########################
# Pixelwise --> groupby_dims = "node"
# Anomalies: center=True, standardize=False
# Standardized Anomalies: center=True, standardize=True

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

from modules.xscaler import TemporalStandardScaler

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
gs.fit_transform()
ds_trans = gs.transform(ds).compute()
ds_invert = gs.inverse_transform(ds_trans).compute()
xr.testing.assert_equal(ds, ds_invert)

# DataArray with variable dimension
gs = TemporalStandardScaler(data=da, 
                            variable_dim=variable_dim, 
                            time_dim=time_dim,
                            time_groups=time_groups)
gs.fit()
gs.fit_transform()
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
gs.fit_transform()
da_trans = gs.transform(da1, variable_dim=None).compute()
da_invert = gs.inverse_transform(da_trans, variable_dim=None).compute()
xr.testing.assert_equal(da1, da_invert)

# DataArray with groupby option (i.e. scaling each member ...)
gs = TemporalStandardScaler(data=da, 
                            variable_dim=variable_dim, 
                            groupby_dims=groupby_dims, 
                            time_dim=time_dim,
                            time_groups=time_groups)
gs.fit()
gs.fit_transform()
da_trans = gs.transform(da, variable_dim=variable_dim).compute()
da_invert = gs.inverse_transform(da_trans, variable_dim=variable_dim).compute()
xr.testing.assert_equal(da, da_invert)

# DataArray with variable_dimension but not specified  
gs = TemporalStandardScaler(data=da, 
                            variable_dim=None,
                            time_dim=time_dim,
                            time_groups=time_groups)
gs.fit()
gs.fit_transform()
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
gs.fit_transform()
ds_trans = gs.transform(ds).compute()

gs = TemporalStandardScaler(data=ds, 
                            time_dim=time_dim, 
                            time_groups='dayofyear')
gs.fit()
gs.fit_transform()
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
# TODO: 
# - Check dimensions of _mean ... are in new_data 
# - Rename_dictionary 

# - Option to add time group dimensions to mean_, std_

# - TemporalScalers
# --> check time_idx present in groupby 
# --> insert Nan in mean_, std_ 


#----------------------------------------------------------------------------.
# - Monthly standardized anomalies 
ds_mean = ds.groupby("time.month").mean("time").compute()
ds_std = ds.groupby("time.month").std("time").compute()
 

ds_std_anom1 = (ds.groupby("time.month") - ds_mean)/ ds_std
ds_std_anom1 = ds_std_anom1.compute() # dimension month (duplicated values?)

ds_std_anom2 = xr.apply_ufunc(lambda x, m, s: (x - m) / s,
                              ds.groupby("time.month"), 
                              ds_mean,
                              ds_std)  # dimensio month not dim coordinates

ds_tmp = ds.groupby("time.month") - ds_mean
ds_std_anom3 = ds_tmp.groupby("time.month") / ds_std
ds_std_anom3 = ds_std_anom3.compute()  

xr.testing.assert_equal(ds_std_anom2, ds_std_anom3) 

# groupby.().reduce
# da.reduce(np.mean, dim=) 

### Apply function at each timestep  (over other dimensions)
# - Global normalization at each timestep
da.groupby("time").map(lambda x: ((x - x.min())/x.max())) 
# - Global standardization at each timestep
da.groupby("time").map(lambda x: ((x - x.mean())/x.std())) 


##----------------------------------------------------------------------------.
 
 
 
##----------------------------------------------------------------------------.
# CompositeScaler(scaler1, scaler2, ...) --> (multiple variables)
# SequentialScaler(scaler1, scaler2) --> (Anomalies + GlobalStandardScaler)
# --> Fit (delayed) 
##----------------------------------------------------------------------------.


##----------------------------------------------------------------------------. 
### Robust anomalies/std
# http://xarray.pydata.org/en/stable/dask.html
# https://stackoverflow.com/questions/54938180/get-95-percentile-of-the-variables-for-son-djf-mam-over-multiple-years-data

ds.median('time','node').compute()
ds.quantile([0.25, 0.75], dim=('time','node')).compute()    
ds_std = IQR


def my_iqr(x, axis=None): 
    return np.diff(np.quantile(x, q=[0.25, 0.75]), axis=axis)
da.reduce(my_iqr, dim=('time','node')).compute()

# da.map_blocks
da.reduce(np.quantile, dim=('time','node'), q=0.25).compute()
da.reduce(np.quantile, dim=('time','node'), q=[0.25, 0.75]).compute()



##----------------------------------------------------------------------------.
### Linear trend 
# define a function to compute a linear trend of a timeseries
def linear_trend(x):
    # x = x.dropna(dim='time')
    pf = np.polyfit(x.time, x, 1)
    # we need to return a dataarray or else xarray's groupby won't be happy
    return xr.DataArray(pf[0])

# stack lat and lon into a single dimension called allpoints
stacked = da.stack(allpoints=['x','y'])
# apply the function over allpoints to calculate the trend at each point
trend = stacked.groupby('allpoints').apply(linear_trend)
# unstack back to lat lon coordinates
trend_unstacked = trend.unstack('allpoints')

def _calc_allpoints(ds, function):
    """
    Helper function to do a pixel-wise calculation that requires using x and y dimension values
    as inputs. This version does the computation over all available timesteps as well.

    """

    # note: the below code will need to be generalized for other dimensions

    def _time_wrapper(gb):
        gb = gb.groupby('dtime', squeeze=False).apply(function)
        return gb
    
    # stack x and y into a single dimension called allpoints
    stacked = ds.stack(allpoints=['x','y'])
    # groupby time and apply the function over allpoints to calculate the trend at each point
    newelev = stacked.groupby('allpoints', squeeze=False).apply(_time_wrapper)
    # unstack back to x y coordinates
    ds = newelev.unstack('allpoints')

    return ds
    
#-----------------------------------------------------------------------------.
### Persistence Class
class Persistence():
	def init() :
        
	def forecast(<np.datetime64_arr>)
	       return ds_persistence_forecast

##----------------------------------------------------------------------------.

## OneHotEncoder 
## Binarizer
## PowerTransformer 
## QuantileTransformer 
## MaxAbsScaler
## https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html#sklearn.preprocessing.QuantileTransformer
# https://phausamann.github.io/sklearn-xarray/content/pipeline.html


##----------------------------------------------------------------------------.
#### Possible future improvements
### RollingScalers 
# -- No rolling yet implemented for groupby xarray object 
 
### SpatialScaler 
# --> Requires a groupby_spatially(geopandas)

##----------------------------------------------------------------------------.









 