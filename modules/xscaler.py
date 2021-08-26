#!/usr/bin/env python3
"""
Created on Mon Dec 14 13:25:46 2020

@author: ghiggi
"""
import xarray as xr
import numpy as np
import os
import time 
 
# import xscaler
# xscaler.GlobalScaler.MinMaxScaler
# xscaler.GlobalScaler.StandardScaler  

# GlobalScaler  
# TemporalScaler 
# xr.ALL_DIMS # ...

## Make "elapsed time" optional

## GitHub issues related to groupby(time)
# - https://github.com/pydata/xarray/issues/2237
##----------------------------------------------------------------------------.
## TODO 
# - Robust standardization (IQR, MEDIAN) (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler)
# - feature_min, feature_max as dictionary per variable for MinMaxScaler ... 
# - Add lw and up to std scalers (avoid outliers alter the distribution)
# - In TemporalScalers, when new_data contain new time_groupby indices values, insert NaN values
#     in mean_, std_  for the missing time_groupby values
# - check_time_groups might also check that t_res specified is > to t_res data
##----------------------------------------------------------------------------.
# # Loop over each variable (for Datasets)
# gs = GlobalStandardScaler(data=ds)
# gs.fit()
# mean_ = gs.mean_ 
# ds['z500'] = ds['z500'] - mean_['z500'] 

# # Loop over each variable (for DataArray)
# gs = GlobalStandardScaler(data=da, variable_dim="feature")
# gs.fit()
# mean_ = gs.mean_ 
# da.loc[dict(feature='z500')] = da.loc[dict(feature='z500')] - mean_.loc[dict(feature='z500')] 

# How to generalize to Dataset and DataArray:
# var = "z500"
# sel = "['" + var + "']"
# sel = ".loc[dict(" + variable_dim + "='" + var + "')]"
# exec_cmd = "x" + sel + " = x" + sel "- mean_" + sel 
# exec(exec_cmd)

##----------------------------------------------------------------------------.
#### Possible future improvements
## RollingScaler
# - No rolling yet implemented for groupby xarray object 
## SpatialScaler 
# - Requires a groupby_spatially(gpd_poly or xr.grid)

## In future: multidimensional groupby? :
# - http://xarray.pydata.org/en/stable/groupby.html
# - http://xarray.pydata.org/en/stable/generated/xarray.IndexVariable.html#xarray.IndexVariable 
# - https://github.com/pydata/xarray/issues/324
# - https://github.com/pydata/xarray/issues/1569

## sklearn-xarray
# https://phausamann.github.io/sklearn-xarray/content/pipeline.html

##----------------------------------------------------------------------------.
#### Utils ####
def get_valid_scaler_class():
    """Return list of implemented xscaler objects."""
    scaler_classes = ['GlobalStandardScaler', 'TemporalStandardScaler',
                      'GlobalMinMaxScaler', 'TemporalMinMaxScaler',
                      'SequentialScaler']
    return scaler_classes

def check_valid_scaler(scaler): 
    """Check is a valid scaler."""
    # TODO : Check type/class instead looking for attribute...
    if scaler.scaler_class not in get_valid_scaler_class():
        print(scaler.scaler_class)
        raise ValueError("Not a valid scaler")
        
def check_variable_dim(variable_dim, data):
    """Check that the correct variable dimension (for DataArray) is specified."""
    # Check type
    if variable_dim is None: 
        return None
    if not isinstance(variable_dim, str):
        raise TypeError("Provide 'variable_dim' as a string")  
    # Check validity
    dims = list(data.dims)
    if variable_dim not in dims:
        raise ValueError("'variable_dim' must be a dimension coordinate of the xarray object") 
    # Return variable_dim as a string    
    return variable_dim

def check_groupby_dims(groupby_dims, data):
    """Check that valid groupby dimensions are specified."""
    # Check type
    if isinstance(groupby_dims, str):
        groupby_dims = [groupby_dims]
    if not (isinstance(groupby_dims, list) or isinstance(groupby_dims, tuple)):
        raise TypeError("Provide 'groupby_dims' as a string, list or tuple")  
    # Check validity
    dims = np.array(list(data.dims))
    if not np.all(np.isin(groupby_dims, dims)):
        raise ValueError("'groupby_dims' must be dimension coordinates of the xarray object") 
    # Return grouby_dims as a list of strings     
    return groupby_dims     

def check_rename_dict(data, rename_dict): 
    """Check rename_dict validity."""
    if not isinstance(rename_dict, dict):
        raise ValueError("'rename_dict' must be a dictionary.")
    data_dims = list(data.dims)
    keys = list(rename_dict.keys())
    vals = list(rename_dict.values())
    keys_all = np.all(np.isin(keys, data_dims))
    vals_all = np.all(np.isin(vals, data_dims))
    if keys_all: 
        new_dict = rename_dict 
    elif vals_all: 
        new_dict = {v: k for k,v in rename_dict.items()}
    else:
        raise ValueError("The specified dimensions in 'rename_dict' are not dimensions of the supplied data.")
    return new_dict    
            
def get_xarray_variables(data, variable_dim = None):
    """Return the variables of an xarray Dataset or DataArray."""
    if isinstance(data, xr.Dataset):
        return list(data.data_vars.keys())  
    elif isinstance(data, xr.DataArray):
        if variable_dim is None:
            return data.name 
        else: 
            variable_dim = check_variable_dim(variable_dim = variable_dim, data = data)
            return data[variable_dim].values.tolist()
    else: 
        raise TypeError("Provide an xarray Dataset or DataArray")

#-----------------------------------------------------------------------------.
# ################################
#### Utils for TemporalScalers ###
# ################################
def check_time_dim(time_dim, data):
    """Check that the correct time dimension is specified."""
    # Check type
    if not isinstance(time_dim, str):
        raise TypeError("Specify 'time_dim' as a string.")  
    # Check validity
    dims = list(data.dims)
    if time_dim not in dims:
        raise ValueError("'time_dim' must specify the time dimension coordinate of the xarray object.") 
    if not isinstance(data[time_dim].values[0], np.datetime64):
        raise ValueError("'time_dim' must indicate a time dimension coordinate with np.datetime64 dtype.")  
    # Return time_dim as a string    
    return time_dim

def get_valid_time_groups():
    """Return valid time groups."""
    time_groups = ['year','season','quarter','month','day',
                   'weekofyear','dayofweek','dayofyear',
                   'hour','minute','second']  
    return time_groups

def get_dict_season():
    """Return dictionary for conversion (to integers) of season strings."""
    dict_season = {'DJF': 1,
                   'MAM': 2,
                   'JJA': 3,
                   'SON': 4,
                   }
    return dict_season

def get_time_group_max(time_group):
    """Return dictionary with max values for each time group."""
    dict_time_max = {'year': 5000,      # dummy large value for year since unbounded ... 
                     'season': 4,    
                     'quarter': 4, 
                     'month': 12, 
                     'weekofyear': 53,  
                     'dayofweek': 7,    # init 0, end 6
                     'dayofyear': 366,
                     'day': 31, 
                     'hour': 24,        # init 0, end 23
                     'minute': 60,      # init 0, end 59
                     'second': 60,      # init 0, end 59
                     }
    return dict_time_max[time_group]

def get_time_group_min(time_group):
    """Return dictionary with min values for each time group."""
    dict_time_min = {'year': 0,   
                     'season': 1,    
                     'quarter': 1,
                     'month': 1, 
                     'weekofyear': 1, 
                     'dayofweek': 0,
                     'dayofyear': 1,
                     'day': 1,
                     'hour': 0,
                     'minute': 0,
                     'second': 0
                     }
    return dict_time_min[time_group] 

def get_time_groupby_name(time_groups):
    """Return a name reflecting the temporal groupby operation."""
    # Define time groupby name 
    time_groups_list = []
    for k, v in time_groups.items():
        if v == 1:
            time_groups_list.append(k) 
        else: 
            time_groups_list.append(str(v) + k) 
    time_groupby_name = "Time_GroupBy " + '-'.join(time_groups_list)    
    return time_groupby_name

def check_time_groups(time_groups): 
    """Check validity of time_groups."""
    if time_groups is None: 
        return None 
    # Check type 
    if isinstance(time_groups, str):
        time_groups = [time_groups]
    if isinstance(time_groups, list):
        time_groups = {k: 1 for k in time_groups}
    if not isinstance(time_groups, dict):
        raise TypeError("Provide time_groups as string, list or dictionary.")
    ##------------------------------------------------------------------------.
    # Check time_groups name validity 
    time_groups_name = np.array(list(time_groups.keys()))
    unvalid_time_groups_name = time_groups_name[np.isin(time_groups_name, get_valid_time_groups(), invert=True)]
    if len(unvalid_time_groups_name) > 0: 
        raise ValueError("{} are not valid 'time_groups' keys".format(unvalid_time_groups_name))
    ##------------------------------------------------------------------------.
    # Check time_groups time aggregation validity
    for k, v in time_groups.items():
        # Check min value 
        if v < 1:
            raise ValueError("The aggregation period of '{}' must be at least 1".format(k)) 
        # Check max value
        max_val = get_time_group_max(time_group=k)
        if v > get_time_group_max(time_group=k):
            raise ValueError("The maximum aggregation period of '{}' is {}".format(k, max_val))
        # Check max value is divisible by specified time aggregation
        if ((max_val % v) != 0):
            print("Attention, the specified aggregation period ({}) does not allow uniform subdivision of '{}'".format(v, k)) 
    ##------------------------------------------------------------------------.
    return time_groups

def check_time_groupby_factors(time_groupby_factors, time_groups):
    """Check validity of time_groupby_factors."""
    if time_groupby_factors is None: 
        return {} 
    if time_groups is not None: 
        if not np.all(np.isin(time_groups.keys(), time_groupby_factors.keys())):
            raise ValueError("All time groups must be included in time_groupby_factors.")
        return time_groupby_factors
    else:
        return {} 

def check_new_time_groupby_idx(time_groupby_idx, scaler_stat):
    """Check that the fitted scaler contains all time_groupby_idx of new_data."""
    time_groupby_idx_orig = np.unique(scaler_stat[time_groupby_idx.name].values)
    time_groupby_idx_new = np.unique(time_groupby_idx.values)
    if not np.all(np.isin(time_groupby_idx_new, time_groupby_idx_orig)):
        raise ValueError("The TemporalScaler does not contain representative statistics for all time_groups indices of 'new_data'.")       
        
##----------------------------------------------------------------------------.   
def get_time_groupby_idx(data, time_dim, time_groups, time_groupby_factors=None): 
    """Return a 1D array with unique index for temporal groupby operation."""
    # Check time groups 
    time_groups_dict = check_time_groups(time_groups=time_groups)
    # Check time_groupby_factors 
    time_groupby_factors = check_time_groupby_factors(time_groupby_factors, time_groups=time_groups_dict)  
    no_time_groupby_factors = len(time_groupby_factors) == 0
    ##------------------------------------------------------------------------.
    # Retrieve groupby indices 
    if time_groups is not None: 
        tmp_min_interval = 0
        l_time_groups_dims = []     
        for i, (time_group, time_agg) in enumerate(time_groups_dict.items()):
            # Retrieve max time aggregation
            time_agg_max = get_time_group_max(time_group=time_group)
            # Retrieve time index (for specific time group)
            # -  dt.week, dt.weekofyear has been deprecated in Pandas ...
            if time_group == "weekofyear":
                idx = data[time_dim].dt.isocalendar().week
            else:
                idx = data[time_dim].dt.__getattribute__(time_group)
            l_time_groups_dims.append(idx)
            # Preprocessing if 'season' (string to integer)
            if time_group == 'season':
                dict_season = get_dict_season()
                idx_values = [dict_season[s] for s in idx.values]
                idx.values = idx_values
            ##----------------------------------------------------------------.
            # Define (numeric) indexing for groupby
            idx_agg = np.floor(idx/time_agg)            # set equal indices within time_agg period 
            idx_norm = idx_agg/(time_agg_max/time_agg)  # value between 0 and 1  
            ##----------------------------------------------------------------.
            if no_time_groupby_factors: 
                # get_numeric_combo_factor()
                if tmp_min_interval == 0:
                    idx_scaled = idx_norm # *10⁰
                    tmp_min_interval = np.max(np.unique(idx_scaled))
                    time_groupby_factors[time_group] = 0 # 10^0 = 1
                    time_groupby_idx = idx_scaled                   
                else: 
                    factor = 0
                    while True: 
                        idx_scaled = idx_norm*(10**factor)
                        unique_idx = np.unique(idx_scaled)
                        if np.min(np.diff(unique_idx)) > tmp_min_interval:
                            break
                        else: 
                            factor = factor + 1
                    tmp_min_interval = tmp_min_interval + np.max(unique_idx)
                    time_groupby_idx = time_groupby_idx + idx_scaled
                    time_groupby_factors[time_group] = factor
            else: 
                idx_scaled = idx_norm*(10**time_groupby_factors[time_group])  
                if i == 0:
                    time_groupby_idx = idx_scaled
                else:
                    time_groupby_idx = time_groupby_idx + idx_scaled
        ##--------------------------------------------------------------------.
        # Add name to time groupby indices     
        time_groupby_idx_name = get_time_groupby_name(time_groups_dict)
        time_groupby_idx.name = time_groupby_idx_name
        # Retrieve time_groups_dims coords
        time_groups_dims = xr.merge(l_time_groups_dims)
        # Retrieve unique extended time_groupby_dims coords
        time_groupby_dims = xr.merge([time_groupby_idx, time_groups_dims])
        _, index = np.unique(time_groupby_dims[time_groupby_idx_name], return_index=True)
        time_groupby_dims = time_groupby_dims.isel({time_dim: index})
        time_groupby_dims = time_groupby_dims.swap_dims({time_dim: time_groupby_idx_name}).drop(time_dim)
    # If no time_groups are specified --> Long-term mean    
    else: 
        # Set all indices to 0 (unique group over time --> long-term mean)
        time_groupby_idx = data.time.dt.month
        time_groupby_idx[:] = 0
        time_groupby_idx.name = "Long-term mean"
        time_groupby_dims = None
    ##------------------------------------------------------------------------.
    # Create time_groupby info dictionary
    time_groupby_info = {}
    time_groupby_info['time_groupby_idx'] = time_groupby_idx
    time_groupby_info['time_groupby_idx_name'] = time_groupby_idx.name
    time_groupby_info['time_groupby_factors'] = time_groupby_factors
    time_groupby_info['time_groupby_dims'] = time_groupby_dims
    return time_groupby_info 

##----------------------------------------------------------------------------.
def check_reference_period(reference_period):
    """Check reference_period validity."""
    # Check type 
    if reference_period is None: 
        return None
    if not isinstance(reference_period, (list,tuple, np.ndarray)):
        raise TypeError("'reference period' must be either a list, tuple or numpy array with start and end time period.")
    if len(reference_period) != 2: 
        raise ValueError("'reference period' require 2 elements: start time and end time.")
    ##------------------------------------------------------------------------.
    # If np.array with np.datetime64
    if isinstance(reference_period, np.ndarray):
        if not np.issubdtype(reference_period.dtype, np.datetime64):
            raise ValueError('If a numpy array, must have np.datetime64 dtype.')
        else:
            return reference_period
    ##------------------------------------------------------------------------.
    if isinstance(reference_period, (list,tuple)):
        try:
            reference_period = np.array(reference_period, dtype='M8')
        except ValueError:
            raise ValueError("The values of reference_period can not be converted to datetime64.")      
    return reference_period

##----------------------------------------------------------------------------.
### Utils for Hovmoller 
def check_spatial_dim(spatial_dim, data):
    """Check that a valid spatial dimension is specified."""
    # Check type
    if not isinstance(spatial_dim, str):
        raise TypeError("Specify 'spatial_dim' as a string.")  
    # Check validity
    coords = list(data.coords.keys())
    if spatial_dim not in coords:
        raise ValueError("'spatial_dim' must be a coordinate of the xarray object.") 
    # Return spatial_dim as a list of strings     
    return spatial_dim  

##----------------------------------------------------------------------------.
def check_bin_width(bin_width): 
    if not isinstance(bin_width, (int,float)):
        raise TypeError("'bin_width' must be an integer or float number.")
    if bin_width <= 0:
        raise ValueError("'bin_width' must be a positive number larger than 0.")
    return bin_width

##----------------------------------------------------------------------------.
def check_bin_edges(bin_edges, lb, ub): 
    if not isinstance(bin_edges, (list,np.ndarray)):
        raise TypeError("'bin_edges' must be a list or numpy.ndarray.")
    if isinstance(bin_edges, list):
        bin_edges = np.array(bin_edges)
    # Select and sort only unique values 
    bin_edges = np.sort(np.unique(bin_edges))
    # Check that at least 2 bins can be defined
    if len(bin_edges) < 3:
        raise ValueError("'bin_edges' must have minimum 3 unique values.")
    # Ensure that some data falls within the bins 
    if bin_edges[0] >= ub:
        raise ValueError("The left edge exceed the max value.")
    if bin_edges[-1] <= lb:
        raise ValueError("The right edge exceed the min value.")
    n_bins_within_data_range = sum(np.logical_and(bin_edges > lb, bin_edges < ub))
    if n_bins_within_data_range < 2:
        raise ValueError("Too much values in 'bin_edges' are outside data range to create at least 1 bin.")
    return bin_edges

#-----------------------------------------------------------------------------.
# #####################
#### Global Scalers ###
# #####################
# - Statistics over all space, at each timestep: groupby_dims = "time"
# - Statistics over all timestep, at each pixel: groupby_dims = "node"

class GlobalStandardScaler():
    """StandardScaler aggregating over all dimensions (except variable_dim and groupby_dims)."""
    
    def __init__(self,
                 data,
                 variable_dim=None, groupby_dims=None, 
                 center=True, standardize=True, eps=0.0001, ds_scaler=None):
        # ds_scaler must not be specified. Use load_scaler(fpath) if you want to load an existing scaler from disk.
        ##--------------------------------------------------------------------.
        ### Load an already existing scaler (if ds_scaler is provided) 
        if isinstance(ds_scaler, xr.Dataset):
            self.scaler_class = ds_scaler.attrs['scaler_class']
            self.eps = ds_scaler.attrs['eps']
            self.aggregating_dims = ds_scaler.attrs['aggregating_dims'] if ds_scaler.attrs['aggregating_dims'] != 'None' else None
            self.groupby_dims = ds_scaler.attrs['groupby_dims'] if ds_scaler.attrs['groupby_dims'] != 'None' else None   
            self.center = True if ds_scaler.attrs['center'] == 'True' else False
            self.standardize = True if ds_scaler.attrs['standardize'] == 'True' else False
            self.fitted = True
            if self.center:
                self.mean_ = ds_scaler['mean_'].to_dataset(dim='variable')
            if self.standardize:
                self.std_ = ds_scaler['std_'].to_dataset(dim='variable')
            
        ##--------------------------------------------------------------------.
        ### Create the scaler object 
        else:
            # Check center, standardize
            if not isinstance(center, bool):
                raise TypeError("'center' must be True or False'")
            if not isinstance(standardize, bool):
                raise TypeError("'standardize' must be True or False'")
            if not center and not standardize:
                raise ValueError("At least one between 'center' and 'standardize' must be 'true'.")
            ##--------------------------------------------------------------------.   
            # Check data is an xarray Dataset or DataArray  
            if not (isinstance(data, xr.Dataset) or isinstance(data, xr.DataArray)):
                raise TypeError("'data' must be an xarray Dataset or xarray DataArray")
            ##--------------------------------------------------------------------.                        
            ## Checks for Dataset 
            if isinstance(data, xr.Dataset):
                # Check variable_dim is not specified ! 
                if variable_dim is not None: 
                    raise ValueError("'variable_dim' must not be specified for Dataset objects. Use groupby_dims instead.")
            ##--------------------------------------------------------------------.  
            # Checks for DataArray (and convert to Dataset)
            if isinstance(data, xr.DataArray):
                # Check variable_dim
                if variable_dim is None: 
                    # If not specified, data name will become the dataset variable name
                    data = data.to_dataset() 
                else: 
                    variable_dim = check_variable_dim(variable_dim = variable_dim, data = data)
                    data = data.to_dataset(dim=variable_dim) 
            ##---------------------------------------------------------------------.
            # Define groupby dimensions (over which to groupby)
            if groupby_dims is not None:
                groupby_dims = check_groupby_dims(groupby_dims=groupby_dims, data = data) # list (or None)
            self.groupby_dims = groupby_dims
            ##--------------------------------------------------------------------.    
            # Retrieve dimensions over which to aggregate 
            # - If DataArray, exclude variable_dims 
            dims = np.array(list(data.dims))
            if groupby_dims is None: 
                self.aggregating_dims = dims.tolist()  
            else:
                self.aggregating_dims = dims[np.isin(dims, groupby_dims, invert=True)].tolist()
            ##--------------------------------------------------------------------. 
            # Initialize 
            self.scaler_class = "GlobalStandardScaler"
            self.data = data
            self.fitted = False
            self.eps = eps 
            self.center = center
            self.standardize = standardize
            self.mean_ = None
            self.std_ = None
            # Save variable_dim if data is DataArray and using fit_transform()
            self.variable_dim = variable_dim

    ##------------------------------------------------------------------------.
    def fit(self, show_progress=True):
        """Fit the GlobalStandardScaler."""
        #---------------------------------------------------------------------.
        # Checks
        if self.fitted: 
            raise ValueError("The scaler has been already fitted!")
        #---------------------------------------------------------------------.
        # Fit the scaler 
        t_i = time.time()
        if self.center:
            self.mean_ = self.data.mean(self.aggregating_dims).compute()
        if self.standardize:
            self.std_ = self.data.std(self.aggregating_dims).compute()
        print('- Elapsed time: {:.2f}min'.format((time.time() - t_i)/60))
        # Set fitted flag to True 
        self.fitted = True
        # del self.data
    
    def save(self, fpath):
        """Save the scaler object to disk in netCDF format."""
        ##--------------------------------------------------------------------.
        # Checks
        if not self.fitted:
            raise ValueError("Please fit() the scaler before saving it!")
        # Check basepath exists 
        if not os.path.exists(os.path.dirname(fpath)):
            # If not exist, create directory 
            os.makedirs(os.path.dirname(fpath))
            print("The directory {} did not exist and has been created !".format(os.path.dirname(fpath)))
        # Check end with .nc 
        if fpath[-3:] != ".nc":
            fpath = fpath + ".nc"
            print("Added .nc extension to the provided fpath.")
        ##--------------------------------------------------------------------.
        # Create xarray Dataset (to save as netCDF)
        mean_ = self.mean_.to_array()
        mean_.name = "mean_"
        std_ = self.std_.to_array()
        std_.name = "std_"  
        # - Pack data into a Dataset based on center and standardize arguments
        if self.center and self.standardize:
            ds_scaler = xr.merge((mean_, std_))
        elif self.center:
            ds_scaler = mean_.to_dataset()
        else:
            ds_scaler = std_.to_dataset()
        ds_scaler.attrs = {'scaler_class': self.scaler_class,
                           'eps': self.eps,
                           'aggregating_dims': self.aggregating_dims if self.aggregating_dims is not None else 'None', 
                           'groupby_dims': self.groupby_dims if self.groupby_dims is not None else 'None',
                           'center': str(self.center),
                           'standardize': str(self.standardize),
                           }
        ds_scaler.to_netcdf(fpath)
        
    ##------------------------------------------------------------------------.   
    def transform(self, new_data, variable_dim=None, rename_dict=None): 
        """Transform data using the fitted GlobalStandardScaler."""
        ##--------------------------------------------------------------------.
        if not self.fitted: 
            raise ValueError("The GlobalStandardScaler need to be first fit() !.")
        
        ##--------------------------------------------------------------------.   
        # Get variables to transform 
        data_vars = get_xarray_variables(new_data, variable_dim=variable_dim)
        if self.center:
            transform_vars = get_xarray_variables(self.mean_)
        else:
            transform_vars = get_xarray_variables(self.std_)    
        transform_vars = np.array(transform_vars)[np.isin(transform_vars, data_vars)]
        if len(transform_vars) == 0:
            return new_data         
        
        ##--------------------------------------------------------------------.
        # If input is DataArray --> Convert to Dataset
        flag_DataArray = False
        if isinstance(new_data, xr.DataArray):
            flag_DataArray = True
            da_name = new_data.name
            da_dims_order = new_data.dims
            variable_dim = check_variable_dim(variable_dim = variable_dim, data = new_data)
            new_data = new_data.to_dataset(dim=variable_dim)
        
        ##--------------------------------------------------------------------.
        # Rename new_data dimensions if rename_dict is provided 
        flag_dim_renamed = False
        if rename_dict is not None:  
            flag_dim_renamed = True
            # Check rename_dict (ensure {from:to} format )
            rename_dict = check_rename_dict(data=new_data, rename_dict=rename_dict)   
            # Create dictionary for resetting dimensions name as original 
            inv_rename_dict = {v: k for k,v in rename_dict.items()}
            # Rename dimensions 
            new_data = new_data.rename(rename_dict)

        ##--------------------------------------------------------------------.
        # Check dimension name coincides 
        new_data_dims = list(new_data.dims)
        if self.center: 
            required_dims = list(self.mean_.dims)
        else:
            required_dims = list(self.std_.dims)
        
        if len(required_dims) >= 1:
            idx_missing_dims = np.isin(required_dims, new_data_dims, invert=True)
            if np.any(idx_missing_dims):
                raise ValueError("Missing {} dimensions in new_data. You might want to specify the 'rename_dict' argument.".format(np.array(required_dims)[idx_missing_dims]))
            
        ##--------------------------------------------------------------------.
        ## Transform variables 
        new_data = new_data.copy()
        for var in transform_vars:
            if self.center: 
                new_data[var] = new_data[var] - self.mean_[var] 
            if self.standardize: 
                new_data[var] = new_data[var] / (self.std_[var] + self.eps)    
                
        ##--------------------------------------------------------------------.   
        # Rename dimension as new_data (if necessary)
        if flag_dim_renamed: 
            new_data = new_data.rename(inv_rename_dict)

        ##--------------------------------------------------------------------.      
        # Reshape to DataArray if new_data was a DataArray
        if flag_DataArray:
            if variable_dim is None:
                return new_data.to_array(dim='variable', name=da_name).squeeze().drop('variable').transpose(*da_dims_order) 
            else:
                return new_data.to_array(dim=variable_dim, name=da_name).transpose(*da_dims_order)
        else: 
            return new_data 
        
    ##------------------------------------------------------------------------.     
    def inverse_transform(self, new_data, variable_dim=None, rename_dict=None):
        """Inverse transform data using the fitted GlobalStandardScaler."""
        ##--------------------------------------------------------------------.
        if not self.fitted: 
            raise ValueError("The GlobalStandardScaler need to be first fit() !.")
               
        ##--------------------------------------------------------------------.   
        # Get variables to transform 
        data_vars = get_xarray_variables(new_data, variable_dim=variable_dim)
        if self.center:
            transform_vars = get_xarray_variables(self.mean_)
        else:
            transform_vars = get_xarray_variables(self.std_)    
        transform_vars = np.array(transform_vars)[np.isin(transform_vars, data_vars)]
        if len(transform_vars) == 0:
            return new_data         
        
        ##--------------------------------------------------------------------.    
        # If input is DataArray --> Convert to Dataset
        flag_DataArray = False
        if isinstance(new_data, xr.DataArray):
            flag_DataArray = True
            da_name = new_data.name
            da_dims_order = new_data.dims
            variable_dim = check_variable_dim(variable_dim = variable_dim, data = new_data)
            new_data = new_data.to_dataset(dim=variable_dim)
        
        ##--------------------------------------------------------------------.
        # Rename new_data dimensions if rename_dict is provided 
        flag_dim_renamed = False
        if rename_dict is not None:
            flag_dim_renamed = True
            # Check rename_dict (ensure {from:to} format )
            rename_dict = check_rename_dict(data=new_data, rename_dict=rename_dict)   
            # Create dictionary for resetting dimensions name as original 
            inv_rename_dict = {v: k for k,v in rename_dict.items()}
            # Rename dimensions 
            new_data = new_data.rename(rename_dict)
            
        ##--------------------------------------------------------------------.
        # Check dimension name coincides 
        new_data_dims = list(new_data.dims)
        if self.center: 
            required_dims = list(self.mean_.dims)
        else:
            required_dims = list(self.std_.dims)
        
        if len(required_dims) >= 1:
            idx_missing_dims = np.isin(required_dims, new_data_dims, invert=True)
            if np.any(idx_missing_dims):
                raise ValueError("Missing {} dimensions in new_data. You might want to specify the 'rename_dict' argument.".format(np.array(required_dims)[idx_missing_dims]))
                    
        ##--------------------------------------------------------------------.
        ## Transform variables 
        new_data = new_data.copy()
        for var in transform_vars:
            if self.standardize: 
                new_data[var] = new_data[var] * (self.std_[var] + self.eps)  
                
            if self.center: 
                new_data[var] = new_data[var] + self.mean_[var] 
    
        ##--------------------------------------------------------------------.   
        # Rename dimension as new_data (if necessary)
        if flag_dim_renamed: 
            new_data = new_data.rename(inv_rename_dict)

        ##--------------------------------------------------------------------.      
        # Reshape to DataArray if new_data was a DataArray           
        if flag_DataArray:
            if variable_dim is None:
                return new_data.to_array(dim='variable', name=da_name).squeeze().drop('variable').transpose(*da_dims_order)
            else:
                return new_data.to_array(dim=variable_dim, name=da_name).transpose(*da_dims_order)
        else: 
            return new_data 
        
    ##------------------------------------------------------------------------. 
    def fit_transform(self):
        """Fit and transform directly the data."""
        ##--------------------------------------------------------------------.
        if self.fitted:
            raise ValueError("The scaler has been already fitted. Please use .transform().") 
        ##--------------------------------------------------------------------.
        self.fit()
        return self.transform(new_data=self.data, variable_dim=self.variable_dim)
    
##----------------------------------------------------------------------------.
class GlobalMinMaxScaler():
    """MinMaxScaler aggregating over all dimensions (except variable_dim and groupby_dims)."""
    # TODO: feature_min, feature_max as dictionary per variable ... 
    def __init__(self,
                 data,
                 variable_dim=None, groupby_dims=None, 
                 feature_min = 0, feature_max = 1,
                 ds_scaler=None):
        # ds_scaler must not be specified. Use load_scaler(fpath) if you want to load an existing scaler from disk.
        ##--------------------------------------------------------------------.
        ### Load an already existing scaler (if ds_scaler is provided) 
        if isinstance(ds_scaler, xr.Dataset):
            self.scaler_class = ds_scaler.attrs['scaler_class']
            self.aggregating_dims = ds_scaler.attrs['aggregating_dims'] if ds_scaler.attrs['aggregating_dims'] != 'None' else None
            self.groupby_dims = ds_scaler.attrs['groupby_dims'] if ds_scaler.attrs['groupby_dims'] != 'None' else None   
            self.feature_min = ds_scaler.attrs['feature_min']  
            self.feature_max = ds_scaler.attrs['feature_min']  
            self.fitted = True
            # Data
            self.min_ = ds_scaler['min_'].to_dataset(dim='variable')
            self.max_ = ds_scaler['max_'].to_dataset(dim='variable')
            self.range_ = ds_scaler['range_'].to_dataset(dim='variable')
            self.scaling = ds_scaler.attrs['scaling']  
        ##--------------------------------------------------------------------.
        ### Create the scaler object 
        else:
            # Check feature_min, feature_max
            if not isinstance(feature_min, (int, float)):
                raise TypeError("'feature_min' must be a single number.'")
            if not isinstance(feature_max, (int, float)):
                raise TypeError("'feature_max' must be a single number.'")
            ##--------------------------------------------------------------------.   
            # Check data is an xarray Dataset or DataArray  
            if not (isinstance(data, xr.Dataset) or isinstance(data, xr.DataArray)):
                raise TypeError("'data' must be an xarray Dataset or xarray DataArray")
            ##--------------------------------------------------------------------.                        
            ## Checks for Dataset 
            if isinstance(data, xr.Dataset):
                # Check variable_dim is not specified ! 
                if variable_dim is not None: 
                    raise ValueError("'variable_dim' must not be specified for Dataset objects. Use groupby_dims instead.")
            ##--------------------------------------------------------------------.  
            # Checks for DataArray (and convert to Dataset)
            if isinstance(data, xr.DataArray):
                # Check variable_dim
                if variable_dim is None: 
                    # If not specified, data name will become the dataset variable name
                    data = data.to_dataset() 
                else: 
                    variable_dim = check_variable_dim(variable_dim = variable_dim, data = data)
                    data = data.to_dataset(dim=variable_dim) 
            ##---------------------------------------------------------------------.
            # Define groupby dimensions (over which to groupby)
            if groupby_dims is not None:
                groupby_dims = check_groupby_dims(groupby_dims=groupby_dims, data = data) # list (or None)
            self.groupby_dims = groupby_dims
            ##--------------------------------------------------------------------.    
            # Retrieve dimensions over which to aggregate 
            # - If DataArray, exclude variable_dims 
            dims = np.array(list(data.dims))
            if groupby_dims is None: 
                self.aggregating_dims = dims.tolist()  
            else:
                self.aggregating_dims = dims[np.isin(dims, groupby_dims, invert=True)].tolist()
            ##--------------------------------------------------------------------. 
            # Initialize 
            self.scaler_class = "GlobalMinMaxScaler"
            self.data = data
            self.fitted = False
      
            self.feature_min = feature_min
            self.feature_max = feature_max
            self.scaling = self.feature_max - self.feature_min

            # Save variable_dim if data is DataArray and using fit_transform()
            self.variable_dim = variable_dim

    ##------------------------------------------------------------------------.
    def fit(self, show_progress=True):
        """Fit the GlobalMinMaxScaler."""
        #---------------------------------------------------------------------.
        # Checks
        if self.fitted: 
            raise ValueError("The scaler has been already fitted!")
        #---------------------------------------------------------------------.
        # Fit the scaler 
        t_i = time.time()
        self.min_ = self.data.min(self.aggregating_dims).compute()
        self.max_ = self.data.max(self.aggregating_dims).compute()
        self.range_ = self.max_ - self.min_  
        print('- Elapsed time: {:.2f}min'.format((time.time() - t_i)/60))
        self.fitted = True
        # del self.data
    
    def save(self, fpath):
        """Save the scaler object to disk in netCDF format."""
        ##--------------------------------------------------------------------.
        # Checks
        if not self.fitted:
            raise ValueError("Please fit() the scaler before saving it!")
        # Check basepath exists 
        if not os.path.exists(os.path.dirname(fpath)):
            # If not exist, create directory 
            os.makedirs(os.path.dirname(fpath))
            print("The directory {} did not exist and has been created !".format(os.path.dirname(fpath)))
        # Check end with .nc 
        if fpath[-3:] != ".nc":
            fpath = fpath + ".nc"
            print("Added .nc extension to the provided fpath.")
        ##--------------------------------------------------------------------.
        # Create xarray Dataset (to save as netCDF)
        # - Convert to DataArray
        min_ = self.min_.to_array()
        min_.name = "min_"
        max_ = self.max_.to_array()
        max_.name = "max_"  
        range_ = self.range_.to_array()
        range_.name = "range_"  
        # - Pack data into a Dataset based on feature_min and feature_max arguments
        ds_scaler = xr.merge((min_, max_, range_))
        ds_scaler.attrs = {'scaler_class': self.scaler_class,
                           'aggregating_dims': self.aggregating_dims if self.aggregating_dims is not None else 'None', 
                           'groupby_dims': self.groupby_dims if self.groupby_dims is not None else 'None',
                           'scaling': self.scaling,
                           'feature_min': self.feature_min,
                           'feature_max': self.feature_max,
                           }
        ds_scaler.to_netcdf(fpath)
        
    ##------------------------------------------------------------------------.   
    def transform(self, new_data, variable_dim=None, rename_dict=None): 
        """Transform data using the fitted GlobalStandardScaler."""
        ##--------------------------------------------------------------------.
        if not self.fitted: 
            raise ValueError("The GlobalStandardScaler need to be first fit() !.")
        
        ##--------------------------------------------------------------------.
        # Get variables to transform 
        data_vars = get_xarray_variables(new_data, variable_dim=variable_dim)
        transform_vars = get_xarray_variables(self.min_)
        transform_vars = np.array(transform_vars)[np.isin(transform_vars, data_vars)]
        if len(transform_vars) == 0:
            return new_data  
        
        ##--------------------------------------------------------------------.
        # If input is DataArray --> Convert to Dataset
        flag_DataArray = False
        if isinstance(new_data, xr.DataArray):
            flag_DataArray = True
            da_name = new_data.name
            da_dims_order = new_data.dims
            variable_dim = check_variable_dim(variable_dim = variable_dim, data = new_data)
            new_data = new_data.to_dataset(dim=variable_dim)
        
        ##--------------------------------------------------------------------.
        # Rename new_data dimensions if rename_dict is provided 
        flag_dim_renamed = False
        if rename_dict is not None:  
            flag_dim_renamed = True
            # Check rename_dict (ensure {from:to} format )
            rename_dict = check_rename_dict(data=new_data, rename_dict=rename_dict)   
            # Create dictionary for resetting dimensions name as original 
            inv_rename_dict = {v: k for k,v in rename_dict.items()}
            # Rename dimensions 
            new_data = new_data.rename(rename_dict)
                    
        ##--------------------------------------------------------------------.
        # Check dimension name coincides 
        new_data_dims = list(new_data.dims)
        required_dims = list(self.min_.dims)        
        if len(required_dims) >= 1:
            idx_missing_dims = np.isin(required_dims, new_data_dims, invert=True)
            if np.any(idx_missing_dims):
                raise ValueError("Missing {} dimensions in new_data. You might want to specify the 'rename_dict' argument.".format(np.array(required_dims)[idx_missing_dims]))
            
        ##--------------------------------------------------------------------.
        ## Transform variables 
        new_data = new_data.copy()
        for var in transform_vars:
            new_data[var] = (new_data[var] - self.min_[var]) / self.range_[var] * self.scaling + self.feature_min

        ##--------------------------------------------------------------------.   
        # Rename dimension as new_data (if necessary)
        if flag_dim_renamed: 
            new_data = new_data.rename(inv_rename_dict)

        ##--------------------------------------------------------------------.      
        # Reshape to DataArray if new_data was a DataArray
        if flag_DataArray:
            if variable_dim is None:
                return new_data.to_array(dim='variable', name=da_name).squeeze().drop('variable').transpose(*da_dims_order) 
            else:
                return new_data.to_array(dim=variable_dim, name=da_name).transpose(*da_dims_order)
        else: 
            return new_data 
        
    ##------------------------------------------------------------------------.     
    def inverse_transform(self, new_data, variable_dim=None, rename_dict=None):
        """Inverse transform data using the fitted GlobalMinMaxScaler."""
        ##--------------------------------------------------------------------.
        if not self.fitted: 
            raise ValueError("The GlobalMinMaxScaler need to be first fit() !.")
        
        ##--------------------------------------------------------------------.
        # Get variables to transform 
        data_vars = get_xarray_variables(new_data, variable_dim=variable_dim)
        transform_vars = get_xarray_variables(self.min_)
        transform_vars = np.array(transform_vars)[np.isin(transform_vars, data_vars)]
        if len(transform_vars) == 0:
            return new_data  
        
        ##--------------------------------------------------------------------.    
        # If input is DataArray --> Convert to Dataset
        flag_DataArray = False
        if isinstance(new_data, xr.DataArray):
            flag_DataArray = True
            da_name = new_data.name
            da_dims_order = new_data.dims
            variable_dim = check_variable_dim(variable_dim = variable_dim, data = new_data)
            new_data = new_data.to_dataset(dim=variable_dim)
        
        ##--------------------------------------------------------------------.
        # Rename new_data dimensions if rename_dict is provided 
        flag_dim_renamed = False
        if rename_dict is not None:
            flag_dim_renamed = True
            # Check rename_dict (ensure {from:to} format )
            rename_dict = check_rename_dict(data=new_data, rename_dict=rename_dict)   
            # Create dictionary for resetting dimensions name as original 
            inv_rename_dict = {v: k for k,v in rename_dict.items()}
            # Rename dimensions 
            new_data = new_data.rename(rename_dict)
        
        ##--------------------------------------------------------------------.
        # Check dimension name coincides 
        new_data_dims = list(new_data.dims)
        required_dims = list(self.min_.dims)       
        if len(required_dims) >= 1:
            idx_missing_dims = np.isin(required_dims, new_data_dims, invert=True)
            if np.any(idx_missing_dims):
                raise ValueError("Missing {} dimensions in new_data. You might want to specify the 'rename_dict' argument.".format(np.array(required_dims)[idx_missing_dims]))
                    
        ##--------------------------------------------------------------------.
        ## Transform variables 
        new_data = new_data.copy()
        for var in transform_vars: 
            new_data[var] = (new_data[var] - self.feature_min) * self.range_[var] / self.scaling + self.min_[var]  
    
        ##--------------------------------------------------------------------.   
        # Rename dimension as new_data (if necessary)
        if flag_dim_renamed: 
            new_data = new_data.rename(inv_rename_dict)

        ##--------------------------------------------------------------------.      
        # Reshape to DataArray if new_data was a DataArray           
        if flag_DataArray:
            if variable_dim is None:
                return new_data.to_array(dim='variable', name=da_name).squeeze().drop('variable').transpose(*da_dims_order)
            else:
                return new_data.to_array(dim=variable_dim, name=da_name).transpose(*da_dims_order)
        else: 
            return new_data 
        
    ##------------------------------------------------------------------------. 
    def fit_transform(self):
        """Fit and transform directly the data."""
        ##--------------------------------------------------------------------.
        if self.fitted:
            raise ValueError("The scaler has been already fitted. Please use .transform().") 
        ##--------------------------------------------------------------------.
        self.fit()
        return self.transform(new_data=self.data, variable_dim=self.variable_dim)
    
#-----------------------------------------------------------------------------.
# #######################
#### Temporal Scalers ###
# #######################
class TemporalStandardScaler():
    """TemporalStandardScaler aggregating over all dimensions (except variable_dim and groupby_dims)."""
    # TODO: - Add option to bound values to i.e. -5, 5 std devs.
    def __init__(self, 
                 data,
                 time_dim, time_groups=None,
                 variable_dim=None, groupby_dims=None, 
                 reference_period=None,
                 center=True, standardize=True, eps=0.0001, ds_scaler = None):
        # ds_scaler must not be specified. Use load_scaler(fpath) if you want to load an existing scaler from disk.
        ##--------------------------------------------------------------------.
        ### Load an already existing scaler (if ds_scaler is provided) 
        if isinstance(ds_scaler, xr.Dataset):
            self.scaler_class = ds_scaler.attrs['scaler_class']
            self.eps = ds_scaler.attrs['eps']
            self.aggregating_dims = ds_scaler.attrs['aggregating_dims'] if ds_scaler.attrs['aggregating_dims'] != 'None' else None
            self.groupby_dims = ds_scaler.attrs['groupby_dims'] if ds_scaler.attrs['groupby_dims'] != 'None' else None   
            self.time_dim = ds_scaler.attrs['time_dim']
            self.time_groups = eval(ds_scaler.attrs['time_groups'])
            self.time_groupby_factors = eval(ds_scaler.attrs['time_groupby_factors'])
            self.time_groupby_name = ds_scaler.attrs['time_groupby_name']  
            self.center = True if ds_scaler.attrs['center'] == 'True' else False
            self.standardize = True if ds_scaler.attrs['standardize'] == 'True' else False
            self.fitted = True
            if self.center:
                self.mean_ = ds_scaler['mean_'].to_dataset(dim='variable')
            if self.standardize:
                self.std_ = ds_scaler['std_'].to_dataset(dim='variable')
            
        ##--------------------------------------------------------------------.
        ### Create the scaler object 
        else:
            # Check center, standardize
            if not isinstance(center, bool):
                raise TypeError("'center' must be True or False'")
            if not isinstance(standardize, bool):
                raise TypeError("'standardize' must be True or False'")
            if not center and not standardize:
                raise ValueError("At least one between 'center' and 'standardize' must be 'true'.")
            ##----------------------------------------------------------------.   
            # Check data is an xarray Dataset or DataArray  
            if not (isinstance(data, xr.Dataset) or isinstance(data, xr.DataArray)):
                raise TypeError("'data' must be an xarray Dataset or xarray DataArray")
            ##----------------------------------------------------------------.                        
            ## Checks for Dataset 
            if isinstance(data, xr.Dataset):
                # Check variable_dim is not specified ! 
                if variable_dim is not None: 
                    raise ValueError("'variable_dim' must not be specified for Dataset objects. Use groupby_dims instead.")
                    
            ##----------------------------------------------------------------.  
            # Checks for DataArray (and convert to Dataset)
            if isinstance(data, xr.DataArray):
                # Check variable_dim
                if variable_dim is None: 
                    # If not specified, data name will become the dataset variable name
                    data = data.to_dataset() 
                else: 
                    variable_dim = check_variable_dim(variable_dim = variable_dim, data = data)
                    data = data.to_dataset(dim=variable_dim) 
                    
            ##----------------------------------------------------------------.
            # Check time_dim  
            time_dim = check_time_dim(time_dim=time_dim, data=data)
            self.time_dim = time_dim 
            
            ##----------------------------------------------------------------.
            # Select data within the reference period 
            reference_period = check_reference_period(reference_period)
            if reference_period is not None:
                data = data.sel({time_dim: slice(reference_period[0], reference_period[-1])})
                
            ##----------------------------------------------------------------.
            # Define groupby dimensions (over which to groupby)
            if groupby_dims is not None:
                groupby_dims = check_groupby_dims(groupby_dims=groupby_dims, data = data) 
                if time_dim in groupby_dims:
                    raise ValueError("TemporalScalers does not allow 'time_dim' to be included in 'groupby_dims'.")
            self.groupby_dims = groupby_dims
            
            ##----------------------------------------------------------------.
            # Check time_groups 
            time_groups = check_time_groups(time_groups=time_groups)
            self.time_groups = time_groups
            
            ##----------------------------------------------------------------.
            # Retrieve indexing for temporal groupby   
            time_groupby_info = get_time_groupby_idx(data=data,
                                                     time_dim=time_dim, 
                                                     time_groups=time_groups)
            self.time_groupby_idx = time_groupby_info['time_groupby_idx']
            self.time_groupby_name = time_groupby_info['time_groupby_idx_name']
            self.time_groupby_factors = time_groupby_info['time_groupby_factors'] 
            self.time_groupby_dims = time_groupby_info['time_groupby_dims']
            
            ##--------------------------------------------------------------------.    
            # Retrieve dimensions over which to aggregate 
            # - If DataArray, exclude variable_dims 
            # - It include 'time_dim' by default (since groupby_dims do not include 'time_dim')
            dims = np.array(list(data.dims))
            if groupby_dims is None: 
                self.aggregating_dims = dims.tolist()  
            else:
                self.aggregating_dims = dims[np.isin(dims, groupby_dims, invert=True)].tolist()
                
            ##--------------------------------------------------------------------. 
            # Initialize 
            self.scaler_class = 'TemporalStandardScaler'
            self.data = data
            self.fitted = False
            self.eps = eps 
            self.center = center
            self.standardize = standardize
            self.mean_ = None
            self.std_ = None
            # Save variable_dim
            # - Used if data is DataArray and using fit_transform()
            # - Used by Climatology().compute() ...
            self.variable_dim = variable_dim  
        
    ##------------------------------------------------------------------------. 
    def fit(self, show_progress=True):
        """Fit the TemporalStandardScaler."""
        ##---------------------------------------------------------------------.
        if self.fitted: 
            raise ValueError("The scaler has been already fitted!")
        ##---------------------------------------------------------------------.
        # Fit the scaler 
        t_i = time.time()
        if self.center:
            self.mean_ = self.data.groupby(self.time_groupby_idx).mean(self.aggregating_dims).compute()
        if self.standardize:
            self.std_ = self.data.groupby(self.time_groupby_idx).std(self.aggregating_dims).compute()
        print('- Elapsed time: {:.2f}min'.format((time.time() - t_i)/60))
        self.fitted = True
        # del self.data
    
    def save(self, fpath):
        """Save the scaler object to disk in netCDF format."""
        if not self.fitted:
            raise ValueError("Please fit() the scaler before saving it!")
        # Check basepath exists 
        if not os.path.exists(os.path.dirname(fpath)):
            # If not exist, create directory 
            os.makedirs(os.path.dirname(fpath))
            print("The directory {} did not exist and has been created !".format(os.path.dirname(fpath)))
        # Check end with .nc 
        if fpath[-3:] != ".nc":
            fpath = fpath + ".nc"
            print("Added .nc extension to the provided fpath.")
        #---------------------------------------------------------------------.
        ## Create xarray Dataset (to save as netCDF)
        if self.center:
            mean_ = self.mean_.to_array()
            mean_.name = "mean_"
        if self.standardize:
            std_ = self.std_.to_array()  
            std_.name = "std_"
        # - Pack data into a Dataset based on center and standardize arguments
        if self.center and self.standardize:
            ds_scaler = xr.merge((mean_, std_))
        elif self.center:
            ds_scaler = mean_.to_dataset()
        else:
            ds_scaler = std_.to_dataset()
        # Add attributes 
        ds_scaler.attrs = {'scaler_class': self.scaler_class,
                           'eps': self.eps,
                           'aggregating_dims': self.aggregating_dims if self.aggregating_dims is not None else 'None', 
                           'groupby_dims': self.groupby_dims if self.groupby_dims is not None else 'None',
                           "time_dim": self.time_dim, 
                           'center': str(self.center),
                           'standardize': str(self.standardize),
                           'time_groups': str(self.time_groups),
                           'time_groupby_factors': str(self.time_groupby_factors), 
                           'time_groupby_name': self.time_groupby_name
                           }
        ds_scaler.to_netcdf(fpath)
        
    ##------------------------------------------------------------------------.   
    def transform(self, new_data, variable_dim=None, rename_dict=None): 
        """Transform data using the fitted TemporalStandardScaler."""
        ##--------------------------------------------------------------------.
        if not self.fitted: 
            raise ValueError("The TemporalStandardScaler need to be first fit() !")
            
        ##--------------------------------------------------------------------.   
        # Get variables to transform 
        data_vars = get_xarray_variables(new_data, variable_dim=variable_dim)
        if self.center:
            transform_vars = get_xarray_variables(self.mean_)
        else:
            transform_vars = get_xarray_variables(self.std_)    
        transform_vars = np.array(transform_vars)[np.isin(transform_vars, data_vars)]
        if len(transform_vars) == 0:
            return new_data         
        
        ##--------------------------------------------------------------------.
        # If input is DataArray --> Convert to Dataset
        flag_DataArray = False
        if isinstance(new_data, xr.DataArray):
            flag_DataArray = True
            da_name = new_data.name
            da_dims_order = new_data.dims
            variable_dim = check_variable_dim(variable_dim = variable_dim, data = new_data)
            new_data = new_data.to_dataset(dim=variable_dim)
        
        ##--------------------------------------------------------------------.
        # Rename new_data dimensions if rename_dict is provided 
        flag_dim_renamed = False
        if rename_dict is not None:
            flag_dim_renamed = True
            # Check rename_dict (ensure {from:to} format )
            rename_dict = check_rename_dict(data=new_data, rename_dict=rename_dict)   
            # Create dictionary for resetting dimensions name as original 
            inv_rename_dict = {v: k for k,v in rename_dict.items()}
            # Rename dimensions 
            new_data = new_data.rename(rename_dict)
        
        ##--------------------------------------------------------------------.
        # Check dimension name coincides 
        new_data_dims = list(new_data.dims)
        if self.center: 
            required_dims = list(self.mean_.dims)
        else:
            required_dims = list(self.std_.dims)
            
        # - Replace time_grouby dim, with time_dim   
        required_dims = [ self.time_dim if nm == self.time_groupby_name else nm for nm in required_dims]      
        
        # - Check no missing dims in new data 
        idx_missing_dims = np.isin(required_dims, new_data_dims, invert=True)
        if np.any(idx_missing_dims):
            raise ValueError("Missing {} dimensions in new_data. You might want to specify the 'rename_dict' argument.".format(np.array(required_dims)[idx_missing_dims]))
                                    
        ##--------------------------------------------------------------------.
        # Get time grouby indices
        time_groupby_info = get_time_groupby_idx(data=new_data,
                                                 time_dim=self.time_dim, 
                                                 time_groups=self.time_groups,
                                                 time_groupby_factors=self.time_groupby_factors)
        time_groupby_idx = time_groupby_info['time_groupby_idx']
        # Check that the fitted scaler contains all time_groupby_idx of new_data
        if self.center: 
            check_new_time_groupby_idx(time_groupby_idx, scaler_stat = self.mean_)
        else: 
            check_new_time_groupby_idx(time_groupby_idx, scaler_stat = self.std_)    
        ##--------------------------------------------------------------------.
        ## Transform variables 
        new_data = new_data.copy()
        for var in transform_vars:
            if self.center: 
                new_data[var] = new_data[var].groupby(time_groupby_idx) - self.mean_[var] 
            if self.standardize: 
                new_data[var] = new_data[var].groupby(time_groupby_idx) / (self.std_[var] + self.eps)    
                
        ##----------------------------------------------------------------.
        ## Remove non-dimension (time groupby) coordinate      
        new_data = new_data.drop(time_groupby_idx.name)
            
        ##--------------------------------------------------------------------.      
        # Rename dimension as new_data (if necessary)
        if flag_dim_renamed: 
            new_data = new_data.rename(inv_rename_dict)
            
        ##--------------------------------------------------------------------.   
        # Reshape to DataArray if new_data was a DataArray
        if flag_DataArray:
            if variable_dim is None:
                return new_data.to_array(dim='variable', name=da_name).squeeze().drop('variable').transpose(*da_dims_order) 
            else:
                return new_data.to_array(dim=variable_dim, name=da_name).transpose(*da_dims_order)
        else: 
            return new_data 
        
    ##------------------------------------------------------------------------.     
    def inverse_transform(self, new_data, variable_dim=None, rename_dict=None):
        """Inverse transform data using the fitted TemporalStandardScaler."""
        ##--------------------------------------------------------------------.
        if not self.fitted: 
            raise ValueError("The TemporalStandardScaler need to be first fit() !")
        ##--------------------------------------------------------------------.    
        # Get variables to transform 
        data_vars = get_xarray_variables(new_data, variable_dim=variable_dim)
        if self.center:
            transform_vars = get_xarray_variables(self.mean_)
        else:
            transform_vars = get_xarray_variables(self.std_)    
        transform_vars = np.array(transform_vars)[np.isin(transform_vars, data_vars)]
        if len(transform_vars) == 0:
            return new_data    
        
        ##--------------------------------------------------------------------.    
        # If input is DataArray --> Convert to Dataset
        flag_DataArray = False
        if isinstance(new_data, xr.DataArray):
            flag_DataArray = True
            da_name = new_data.name
            da_dims_order = new_data.dims
            variable_dim = check_variable_dim(variable_dim = variable_dim, data = new_data)
            new_data = new_data.to_dataset(dim=variable_dim)
        
        ##--------------------------------------------------------------------.
        # Rename new_data dimensions if rename_dict is provided 
        flag_dim_renamed = False
        if rename_dict is not None:
            flag_dim_renamed = True
            # Check rename_dict (ensure {from:to} format )
            rename_dict = check_rename_dict(data=new_data, rename_dict=rename_dict)   
            # Create dictionary for resetting dimensions name as original 
            inv_rename_dict = {v: k for k,v in rename_dict.items()}
            # Rename dimensions 
            new_data = new_data.rename(rename_dict)
        
        ##--------------------------------------------------------------------.
        # Check dimension name coincides 
        new_data_dims = list(new_data.dims)
        if self.center: 
            required_dims = list(self.mean_.dims)
        else:
            required_dims = list(self.std_.dims)
            
        # - Replace time_grouby dim, with time_dim   
        required_dims = [ self.time_dim if nm == self.time_groupby_name else nm for nm in required_dims]      
        
        # - Check no missing dims in new data 
        idx_missing_dims = np.isin(required_dims, new_data_dims, invert=True)
        if np.any(idx_missing_dims):
            raise ValueError("Missing {} dimensions in new_data. You might want to specify the 'rename_dict' argument.".format(np.array(required_dims)[idx_missing_dims]))
                  
        ##--------------------------------------------------------------------.
        # Get time grouby indices
        time_groupby_info = get_time_groupby_idx(data=new_data,
                                                 time_dim=self.time_dim, 
                                                 time_groups=self.time_groups,
                                                 time_groupby_factors=self.time_groupby_factors)
        time_groupby_idx = time_groupby_info['time_groupby_idx']        
        # Check that the fitted scaler contains all time_groupby_idx of new_data
        if self.center: 
            check_new_time_groupby_idx(time_groupby_idx, scaler_stat = self.mean_)
        else: 
            check_new_time_groupby_idx(time_groupby_idx, scaler_stat = self.std_)

        ##--------------------------------------------------------------------.
        ## Transform variables 
        new_data = new_data.copy()
        for var in transform_vars:
            if self.standardize: 
                new_data[var] = new_data[var].groupby(time_groupby_idx) * (self.std_[var] + self.eps)  
                
            if self.center: 
                new_data[var] = new_data[var].groupby(time_groupby_idx) + self.mean_[var] 
        ##----------------------------------------------------------------.
        ## Remove non-dimension (time groupby) coordinate      
        new_data = new_data.drop(time_groupby_idx.name)
        
        ##--------------------------------------------------------------------.      
        # Rename dimension as new_data (if necessary)
        if flag_dim_renamed: 
            new_data = new_data.rename(inv_rename_dict)
            
        ##--------------------------------------------------------------------.   
        # Reshape to DataArray if new_data was a DataArray
        if flag_DataArray:
            if variable_dim is None:
                return new_data.to_array(dim='variable', name=da_name).squeeze().drop('variable').transpose(*da_dims_order)
            else:
                return new_data.to_array(dim=variable_dim, name=da_name).transpose(*da_dims_order)
        else: 
            return new_data 
        
    ##------------------------------------------------------------------------. 
    def fit_transform(self):
        """Fit and transform directly the data."""
        ##--------------------------------------------------------------------.
        if self.fitted:
            raise ValueError("The scaler has been already fitted. Please use .transform().") 
        ##--------------------------------------------------------------------.
        self.fit()
        return self.transform(new_data=self.data, variable_dim=self.variable_dim)        
            
class TemporalMinMaxScaler():
    """TemporalMinMaxScaler aggregating over all dimensions (except variable_dim and groupby_dims)."""
    
    def __init__(self, 
                 data,
                 time_dim, time_groups=None,
                 variable_dim=None, groupby_dims=None, 
                 feature_min=0, feature_max=1, 
                 reference_period = None,
                 ds_scaler = None):
        # ds_scaler must not be specified. Use load_scaler(fpath) if you want to load an existing scaler from disk.
        ##--------------------------------------------------------------------.
        ### Load an already existing scaler (if ds_scaler is provided) 
        if isinstance(ds_scaler, xr.Dataset):
            self.scaler_class = ds_scaler.attrs['scaler_class']
            self.aggregating_dims = ds_scaler.attrs['aggregating_dims'] if ds_scaler.attrs['aggregating_dims'] != 'None' else None
            self.groupby_dims = ds_scaler.attrs['groupby_dims'] if ds_scaler.attrs['groupby_dims'] != 'None' else None   
            self.time_dim = ds_scaler.attrs['time_dim']
            self.time_groups = eval(ds_scaler.attrs['time_groups'])
            self.time_groupby_factors = eval(ds_scaler.attrs['time_groupby_factors'])
            self.time_groupby_name = ds_scaler.attrs['time_groupby_name']  
            self.feature_min = ds_scaler.attrs['feature_min']  
            self.feature_max = ds_scaler.attrs['feature_max']  
            self.fitted = True
            # Data
            self.min_ = ds_scaler['min_'].to_dataset(dim='variable')
            self.max_ = ds_scaler['max_'].to_dataset(dim='variable')
            self.range_ = ds_scaler['range_'].to_dataset(dim='variable')
            self.scaling = ds_scaler.attrs['scaling'] 
        ##--------------------------------------------------------------------.
        ### Create the scaler object 
        else:
            # Check feature_min, feature_max
            if not isinstance(feature_min, (int,float)):
                raise TypeError("'feature_min' must be a single number.'")
            if not isinstance(feature_max, (int, float)):
                raise TypeError("'feature_max' must be a single number.'")
            ##----------------------------------------------------------------.   
            # Check data is an xarray Dataset or DataArray  
            if not (isinstance(data, xr.Dataset) or isinstance(data, xr.DataArray)):
                raise TypeError("'data' must be an xarray Dataset or xarray DataArray")
            ##----------------------------------------------------------------.                        
            ## Checks for Dataset 
            if isinstance(data, xr.Dataset):
                # Check variable_dim is not specified ! 
                if variable_dim is not None: 
                    raise ValueError("'variable_dim' must not be specified for Dataset objects. Use groupby_dims instead.")
                    
            ##----------------------------------------------------------------.  
            # Checks for DataArray (and convert to Dataset)
            if isinstance(data, xr.DataArray):
                # Check variable_dim
                if variable_dim is None: 
                    # If not specified, data name will become the dataset variable name
                    data = data.to_dataset() 
                else: 
                    variable_dim = check_variable_dim(variable_dim = variable_dim, data = data)
                    data = data.to_dataset(dim=variable_dim) 
                    
            ##----------------------------------------------------------------.
            # Check time_dim  
            time_dim = check_time_dim(time_dim=time_dim, data=data)
            self.time_dim = time_dim 
            
            ##----------------------------------------------------------------.
            # Select data within the reference period 
            reference_period = check_reference_period(reference_period)
            if reference_period is not None:
                data = data.sel({time_dim: slice(reference_period[0], reference_period[-1])})
                
            ##----------------------------------------------------------------.
            # Define groupby dimensions (over which to groupby)
            if groupby_dims is not None:
                groupby_dims = check_groupby_dims(groupby_dims=groupby_dims, data = data) 
                if time_dim in groupby_dims:
                    raise ValueError("TemporalScalers does not allow 'time_dim' to be included in 'groupby_dims'.")
            self.groupby_dims = groupby_dims
            
            ##----------------------------------------------------------------.
            # Check time_groups 
            time_groups = check_time_groups(time_groups=time_groups)
            self.time_groups = time_groups
            
            ##----------------------------------------------------------------.
            # Retrieve indexing for temporal groupby   
            time_groupby_info = get_time_groupby_idx(data=data,
                                                     time_dim=time_dim, 
                                                     time_groups=time_groups)
            self.time_groupby_idx = time_groupby_info['time_groupby_idx']
            self.time_groupby_name = time_groupby_info['time_groupby_idx_name']
            self.time_groupby_factors = time_groupby_info['time_groupby_factors'] 
            self.time_groupby_dims = time_groupby_info['time_groupby_dims']
            ##----------------------------------------------------------------.    
            # Retrieve dimensions over which to aggregate 
            # - If DataArray, exclude variable_dims 
            # - It include 'time_dim' by default (since groupby_dims do not include 'time_dim')
            dims = np.array(list(data.dims))
            if groupby_dims is None: 
                self.aggregating_dims = dims.tolist()  
            else:
                self.aggregating_dims = dims[np.isin(dims, groupby_dims, invert=True)].tolist()
            ##----------------------------------------------------------------. 
            # Initialize 
            self.scaler_class = 'TemporalMinMaxScaler'
            self.data = data
            self.fitted = False
            self.feature_min = feature_min
            self.feature_max = feature_max
            self.scaling = self.feature_max - self.feature_min
            # Save variable_dim
            # - Used if data is DataArray and using fit_transform()
            # - Used by Climatology().compute() ...
            self.variable_dim = variable_dim  
        
    ##------------------------------------------------------------------------. 
    def fit(self, show_progress=True):
        """Fit the TemporalMinMaxScaler."""
        ##--------------------------------------------------------------------.
        if self.fitted: 
            raise ValueError("The scaler has been already fitted!")
        ##--------------------------------------------------------------------.
        # Fit the scaler 
        t_i = time.time()
        self.min_ = self.data.groupby(self.time_groupby_idx).min(self.aggregating_dims).compute()
        self.max_ = self.data.groupby(self.time_groupby_idx).max(self.aggregating_dims).compute()
        self.range_ = self.max_ - self.min_  
        print('- Elapsed time: {:.2f}min'.format((time.time() - t_i)/60))
        self.fitted = True
        # del self.data
    
    def save(self, fpath):
        """Save the scaler object to disk in netCDF format."""
        if not self.fitted:
            raise ValueError("Please fit() the scaler before saving it!")
        # Check basepath exists 
        if not os.path.exists(os.path.dirname(fpath)):
            # If not exist, create directory 
            os.makedirs(os.path.dirname(fpath))
            print("The directory {} did not exist and has been created !".format(os.path.dirname(fpath)))
        # Check end with .nc 
        if fpath[-3:] != ".nc":
            fpath = fpath + ".nc"
            print("Added .nc extension to the provided fpath.")
        ##---------------------------------------------------------------------.
        ## Create xarray Dataset (to save as netCDF)
        # - Convert to DataArray
        min_ = self.min_.to_array()
        min_.name = "min_"
        max_ = self.max_.to_array()
        max_.name = "max_"  
        range_ = self.range_.to_array()
        range_.name = "range_"  
        # - Pack data into a Dataset based on feature_min and feature_max arguments
        ds_scaler = xr.merge((min_, max_, range_))
        # Add attributes 
        ds_scaler.attrs = {'scaler_class': self.scaler_class,
                           'aggregating_dims': self.aggregating_dims if self.aggregating_dims is not None else 'None', 
                           'groupby_dims': self.groupby_dims if self.groupby_dims is not None else 'None',
                           "time_dim": self.time_dim, 
                           'feature_min': self.feature_min,
                           'feature_max': self.feature_max,
                           'scaling': self.scaling,
                           'time_groups': str(self.time_groups),
                           'time_groupby_factors': str(self.time_groupby_factors), 
                           'time_groupby_name': self.time_groupby_name
                           }
        ds_scaler.to_netcdf(fpath)
        
    ##------------------------------------------------------------------------.   
    def transform(self, new_data, variable_dim=None, rename_dict=None): 
        """Transform data using the fitted TemporalMinMaxScaler."""
        ##--------------------------------------------------------------------.
        if not self.fitted: 
            raise ValueError("The TemporalMinMaxScaler need to be first fit() !")
        ##--------------------------------------------------------------------.
        # Get variables to transform 
        data_vars = get_xarray_variables(new_data, variable_dim=variable_dim)
        transform_vars = get_xarray_variables(self.min_)
        transform_vars = np.array(transform_vars)[np.isin(transform_vars, data_vars)]
        if len(transform_vars) == 0:
            return new_data   
        
        ##--------------------------------------------------------------------.
        # If input is DataArray --> Convert to Dataset
        flag_DataArray = False
        if isinstance(new_data, xr.DataArray):
            flag_DataArray = True
            da_name = new_data.name
            da_dims_order = new_data.dims
            variable_dim = check_variable_dim(variable_dim = variable_dim, data = new_data)
            new_data = new_data.to_dataset(dim=variable_dim)
        
        ##--------------------------------------------------------------------.
        # Rename new_data dimensions if rename_dict is provided 
        flag_dim_renamed = False
        if rename_dict is not None:
            flag_dim_renamed = True
            # Check rename_dict (ensure {from:to} format )
            rename_dict = check_rename_dict(data=new_data, rename_dict=rename_dict)   
            # Create dictionary for resetting dimensions name as original 
            inv_rename_dict = {v: k for k,v in rename_dict.items()}
            # Rename dimensions 
            new_data = new_data.rename(rename_dict)
        
        ##--------------------------------------------------------------------.
        # Check dimension name coincides 
        new_data_dims = list(new_data.dims)
        required_dims = list(self.min_.dims)           
        # - Replace time_grouby_idx dim, with time_dim   
        required_dims = [ self.time_dim if nm == self.time_groupby_name else nm for nm in required_dims]      
        
        # - Check no missing dims in new data 
        idx_missing_dims = np.isin(required_dims, new_data_dims, invert=True)
        if np.any(idx_missing_dims):
            raise ValueError("Missing {} dimensions in new_data. You might want to specify the 'rename_dict' argument.".format(np.array(required_dims)[idx_missing_dims]))
                      
        ##--------------------------------------------------------------------.
        # Get time grouby indices
        time_groupby_info = get_time_groupby_idx(data=new_data,
                                                 time_dim=self.time_dim, 
                                                 time_groups=self.time_groups,
                                                 time_groupby_factors=self.time_groupby_factors)
        time_groupby_idx = time_groupby_info['time_groupby_idx']
        # Check that the fitted scaler contains all time_groupby_idx of new_data
        check_new_time_groupby_idx(time_groupby_idx, scaler_stat = self.min_)
  
        ##--------------------------------------------------------------------.
        ## Transform variables 
        new_data = new_data.copy()
        for var in transform_vars:
            new_data[var] = xr.apply_ufunc(lambda x, min_, range_, scaling_, feature_min: (x - min_)/range_*scaling_ + feature_min, 
                                           # Args
                                           new_data[var].groupby(time_groupby_idx),
                                           self.min_[var],
                                           self.range_[var],
                                           self.scaling,
                                           self.feature_min,
                                           dask="allowed", # "parallelized", # 
                                           output_dtypes=[float],
                                           keep_attrs=True)
                            
        ##----------------------------------------------------------------.
        ## Remove non-dimension (time groupby) coordinate      
        new_data = new_data.drop(time_groupby_idx.name)
            
        ##--------------------------------------------------------------------.      
        # Rename dimension as new_data (if necessary)
        if flag_dim_renamed: 
            new_data = new_data.rename(inv_rename_dict)
            
        ##--------------------------------------------------------------------.   
        # Reshape to DataArray if new_data was a DataArray
        if flag_DataArray:
            if variable_dim is None:
                return new_data.to_array(dim='variable', name=da_name).squeeze().drop('variable').transpose(*da_dims_order) 
            else:
                return new_data.to_array(dim=variable_dim, name=da_name).transpose(*da_dims_order)
        else: 
            return new_data 
        
    ##------------------------------------------------------------------------.     
    def inverse_transform(self, new_data, variable_dim=None, rename_dict=None):
        """Inverse transform data using the fitted TemporalMinMaxScaler."""
        ##--------------------------------------------------------------------.
        if not self.fitted: 
            raise ValueError("The TemporalMinMaxScaler need to be first fit() !")
        ##--------------------------------------------------------------------.
        # Get variables to transform 
        data_vars = get_xarray_variables(new_data, variable_dim=variable_dim)
        transform_vars = get_xarray_variables(self.min_)
        transform_vars = np.array(transform_vars)[np.isin(transform_vars, data_vars)]
        if len(transform_vars) == 0:
            return new_data   

        ##--------------------------------------------------------------------.    
        # If input is DataArray --> Convert to Dataset
        flag_DataArray = False
        if isinstance(new_data, xr.DataArray):
            flag_DataArray = True
            da_name = new_data.name
            da_dims_order = new_data.dims
            variable_dim = check_variable_dim(variable_dim = variable_dim, data = new_data)
            new_data = new_data.to_dataset(dim=variable_dim)
        
        ##--------------------------------------------------------------------.
        # Rename new_data dimensions if rename_dict is provided 
        flag_dim_renamed = False
        if rename_dict is not None:
            flag_dim_renamed = True
            # Check rename_dict (ensure {from:to} format )
            rename_dict = check_rename_dict(data=new_data, rename_dict=rename_dict)   
            # Create dictionary for resetting dimensions name as original 
            inv_rename_dict = {v: k for k,v in rename_dict.items()}
            # Rename dimensions 
            new_data = new_data.rename(rename_dict)
        
        ##--------------------------------------------------------------------.
        # Check dimension name coincides 
        new_data_dims = list(new_data.dims)
        required_dims = list(self.min_.dims)
        # - Replace time_grouby dim, with time_dim   
        required_dims = [ self.time_dim if nm == self.time_groupby_name else nm for nm in required_dims]      
        
        # - Check no missing dims in new data 
        idx_missing_dims = np.isin(required_dims, new_data_dims, invert=True)
        if np.any(idx_missing_dims):
            raise ValueError("Missing {} dimensions in new_data. You might want to specify the 'rename_dict' argument.".format(np.array(required_dims)[idx_missing_dims]))
                  
        ##--------------------------------------------------------------------.
        # Get time grouby indices
        time_groupby_info = get_time_groupby_idx(data=new_data,
                                                 time_dim=self.time_dim, 
                                                 time_groups=self.time_groups,
                                                 time_groupby_factors=self.time_groupby_factors)
        time_groupby_idx = time_groupby_info['time_groupby_idx']        
        # Check that the fitted scaler contains all time_groupby_idx of new_data
        check_new_time_groupby_idx(time_groupby_idx, scaler_stat = self.min_)

        ##--------------------------------------------------------------------.
        ## Transform variables 
        new_data = new_data.copy()
        for var in transform_vars:
            new_data[var] = xr.apply_ufunc(lambda x, min_, range_, scaling_, feature_min: (x - feature_min) * range_ / scaling_ + min_, 
                                           # Args
                                           new_data[var].groupby(time_groupby_idx),
                                           self.min_[var],
                                           self.range_[var],
                                           self.scaling,
                                           self.feature_min,
                                           dask="allowed", # "parallelized", # 
                                           output_dtypes=[float],
                                           keep_attrs=True)
            
        ##----------------------------------------------------------------.
        ## Remove non-dimension (time groupby) coordinate  
        new_data = new_data.drop(time_groupby_idx.name)
            
        ##--------------------------------------------------------------------.      
        # Rename dimension as new_data (if necessary)
        if flag_dim_renamed: 
            new_data = new_data.rename(inv_rename_dict)
            
        ##--------------------------------------------------------------------.   
        # Reshape to DataArray if new_data was a DataArray
        if flag_DataArray:
            if variable_dim is None:
                return new_data.to_array(dim='variable', name=da_name).squeeze().drop('variable').transpose(*da_dims_order)
            else:
                return new_data.to_array(dim=variable_dim, name=da_name).transpose(*da_dims_order)
        else: 
            return new_data 
        
    ##------------------------------------------------------------------------. 
    def fit_transform(self):
        """Fit and transform directly the data."""
        ##--------------------------------------------------------------------.
        if self.fitted:
            raise ValueError("The scaler has been already fitted. Please use .transform().") 
        ##--------------------------------------------------------------------.
        self.fit()
        return self.transform(new_data=self.data, variable_dim=self.variable_dim)        
   
#-----------------------------------------------------------------------------.
# ####################
#### Load Scalers ####
# ####################
def LoadScaler(fpath):
    """Load xarray scalers."""
    # Check .nc 
    if fpath[-3:] != ".nc":
        fpath = fpath + ".nc"
    # Check exist 
    if not os.path.exists(fpath):
        raise ValueError("{} does not exist on disk.".format(fpath))
    # Create scaler 
    ds_scaler = xr.open_dataset(fpath)
    scaler_class = ds_scaler.attrs['scaler_class']
    if scaler_class == "GlobalStandardScaler":
        return GlobalStandardScaler(data=None, ds_scaler=ds_scaler)
    if scaler_class == "GlobalMinMaxScaler":
        return GlobalMinMaxScaler(data=None, ds_scaler=ds_scaler)
    if scaler_class == "TemporalStandardScaler":
        return TemporalStandardScaler(data=None, time_dim=None, ds_scaler=ds_scaler)
    if scaler_class == "TemporalMinMaxScaler":
        return TemporalMinMaxScaler(data=None, time_dim=None, ds_scaler=ds_scaler)

#-----------------------------------------------------------------------------.
# ######################### 
#### SequentialScalers ####
# ######################### 
# SequentialScaler(scaler1, scaler2, ... ) 
# --> (PixelwiseAnomalies + GlobalStandardScaler)
# --> (Ad-hoc scaler for specific variables)
# --> Allow nested SequentialScaler
   
class SequentialScaler():
    """Enable sequential scaling operations."""
    
    def __init__(self, *scalers):  
        # Check is a valid scaler 
        for scaler in scalers: 
            check_valid_scaler(scaler)  
        self.list_scalers = scalers 
        self.fitted = False
        self.scaler_class = 'SequentialScaler'

    def fit(self, show_progress=True): 
        """Fit all scalers within a SequentialScaler."""
        new_list_scaler = []
        for scaler in self.list_scalers:
            if not scaler.fitted:
                scaler.fit(show_progress=show_progress)
            new_list_scaler.append(scaler)
        self.list_scalers = new_list_scaler
        self.fitted = True
        
    def save(self): 
        """Save a SequentialScaler to disk."""
        raise NotImplementedError("Saving of SequentialScaler has not been yet implemented!")
        
    def transform(self, new_data, variable_dim = None, rename_dict=None): 
        """Transform data using the fitted SequentialScaler."""
        for scaler in self.list_scalers:
            if not scaler.fitted:
                raise ValueError("The SequentialScaler contains scalers that have not been fit. Use .fit() first!")
        for scaler in self.list_scalers:
            new_data = scaler.transform(new_data=new_data, variable_dim = variable_dim, rename_dict=rename_dict)
        return new_data
    
    def inverse_transform(self, new_data, variable_dim = None, rename_dict=None): 
        """Inverse transform data using the fitted SequentialScaler."""
        reversed_scalers = self.list_scalers[::-1] 
        for scaler in reversed_scalers:
            if not scaler.fitted:
                raise ValueError("The SequentialScaler contains scalers that have not been fit. Use .fit() first!")
        for scaler in reversed_scalers:
            new_data = scaler.inverse_transform(new_data=new_data, variable_dim = variable_dim, rename_dict=rename_dict)
        return new_data 

#-----------------------------------------------------------------------------.
# ###################
#### Climatology ####
# ###################
class Climatology():
    """Compute climatology."""
    
    def __init__(self,
                 data, 
                 time_dim, time_groups=None, 
                 groupby_dims = None, 
                 variable_dim=None, 
                 mean = True, variability=True, 
                 reference_period = None,
                 ds_climatology=None):
        # reference_period : numpy array or tuple of len 2 with datetime 
        # ('1980-01-01T00:00','2010-12-31T23:00')
        # np.array(['1980-01-01T00:00','2010-12-31T23:00'], dtype='M8') 
        ##--------------------------------------------------------------------.
        # Create a climatology object if ds_climatology is a xr.Dataset 
        if isinstance(ds_climatology, xr.Dataset):            
            self.fitted = True
            self.mean = ds_climatology['Mean'].to_dataset(dim='variable')
            self.variability = ds_climatology['Variability'].to_dataset(dim='variable')
            self.variable_dim = ds_climatology.attrs['variable_dim'] if ds_climatology.attrs['variable_dim'] != 'None' else None
            self.aggregating_dims = ds_climatology.attrs['aggregating_dims'] if ds_climatology.attrs['aggregating_dims'] != 'None' else None
            self.groupby_dims = ds_climatology.attrs['groupby_dims'] if ds_climatology.attrs['groupby_dims'] != 'None' else None   
            self.time_dim = ds_climatology.attrs['time_dim']
            self.time_groups = eval(ds_climatology.attrs['time_groups'])
            self.time_groupby_factors = eval(ds_climatology.attrs['time_groupby_factors'])
            self.time_groupby_name = ds_climatology.attrs['time_groupby_name']
            # Ensure arguments are list 
            if isinstance(self.aggregating_dims, str):
                self.aggregating_dims = [self.aggregating_dims]
            if isinstance(self.groupby_dims, str):
                self.groupby_dims = [self.groupby_dims]
            
        ##--------------------------------------------------------------------.
        # Initialize climatology object
        else:
            self.scaler = TemporalStandardScaler(data=data, 
                                                 time_dim = time_dim, 
                                                 time_groups = time_groups,
                                                 variable_dim = variable_dim, 
                                                 groupby_dims = groupby_dims,
                                                 reference_period = reference_period, 
                                                 center=mean, standardize = variability)
            self.fitted = False 
    
    ##------------------------------------------------------------------------.
    def compute(self):
        """Compute climatology mean and variability."""
        # Fit scaler 
        self.scaler.fit()
        self.fitted = self.scaler.fitted 
        # Retrieve mean and variability 
        self.mean = self.scaler.mean_  
        self.variability = self.scaler.std_
        # Extract time group dimensions 
        self.time_dim=self.scaler.time_dim
        self.time_groups = self.scaler.time_groups
        self.time_groupby_factors = self.scaler.time_groupby_factors
        self.time_groupby_name = self.scaler.time_groupby_name
        time_groupby_dims = self.scaler.time_groupby_dims # not self because not saved to disk
        # Extract other infos
        self.variable_dim = self.scaler.variable_dim
        self.aggregating_dims = self.scaler.aggregating_dims
        self.groupby_dims = self.scaler.groupby_dims
        
        # Add extended time group dimensions
        if self.mean is not None:
            for k in self.time_groups.keys():
                self.mean[k] = time_groupby_dims[k]
                self.mean = self.mean.set_coords(k)
                
        if self.variability is not None: 
            for k in self.time_groups.keys():
                self.variability[k] = time_groupby_dims[k]
                self.variability = self.variability.set_coords(k)

        # Return DataArray if input data is dataarray 
        if self.variable_dim is not None:
            if self.mean is not None:
                self.mean = self.mean.to_array(self.variable_dim)
            if self.variability is not None: 
                self.variability = self.variability.to_array(self.variable_dim)
   
    ##------------------------------------------------------------------------. 
    def save(self, fpath):
        """Save the Climatogy object to disk in netCDF format."""
        if not self.fitted:
            raise ValueError("Please fit() the Climatology object before saving it!")
        # Check basepath exists 
        if not os.path.exists(os.path.dirname(fpath)):
            # If not exist, create directory 
            os.makedirs(os.path.dirname(fpath))
            print("The directory {} did not exist and has been created !".format(os.path.dirname(fpath)))
        # Check end with .nc 
        if fpath[-3:] != ".nc":
            fpath = fpath + ".nc"
            print("Added .nc extension to the provided fpath.")
        ##---------------------------------------------------------------------.
        ## Create Climatology xarray Dataset (to save as netCDF)
        # - Reshape mean and variability into DataArray
        if self.mean is not None:
            mean_ = self.mean.to_array()
            mean_.name = "Mean"
        if self.variability is not None:
            std_ = self.variability.to_array()  
            std_.name = "Variability"
        # - Pack data into a Dataset  
        if self.mean is not None and self.variability is not None:
            ds_clim = xr.merge((mean_, std_))
        elif self.mean is not None:
            ds_clim = mean_.to_dataset()
        else:
            ds_clim = std_.to_dataset()
        # Add attributes 
        
        ds_clim.attrs = {'aggregating_dims': self.aggregating_dims if self.aggregating_dims is not None else 'None', 
                         'groupby_dims': self.groupby_dims if self.groupby_dims is not None else 'None',
                         'variable_dim': self.variable_dim if self.variable_dim is not None else 'None',
                         "time_dim": self.time_dim, 
                         'time_groups': str(self.time_groups),
                         'time_groupby_factors': str(self.time_groupby_factors), 
                         'time_groupby_name': self.time_groupby_name
                         }
        ds_clim.to_netcdf(fpath)
        
    ##------------------------------------------------------------------------.
    def forecast(self, time, mean=True):
        """
        Forecast the climatology.

        Parameters
        ----------
        time : np.narray
            Timesteps at which retrieve the climatology.
        mean : bool, optional
            Wheter to forecast the climatological mean (when True) or variability.
            The default is True.

        Returns
        -------
        ds_forecast : xr.Dataset
            xarray Dataset with the forecasted climatology.

        """
        ##--------------------------------------------------------------------.
        # Check time_arr type
        if not isinstance(time, np.ndarray):
            raise TypeError("'time' must be a numpy array with np.datetime64 values.")
        if not np.issubdtype(time.dtype, np.datetime64):
            raise TypeError("The 'time' numpy array must have np.datetime64 values.")
            
        ##--------------------------------------------------------------------.    
        # Define dims names 
        groupby_dims = self.groupby_dims
        time_dim = self.time_dim
        dims = []
        dims.append(time_dim)
        if groupby_dims is not None:
            dims.append(*groupby_dims)
        
        ##--------------------------------------------------------------------.
        # Define dims shape
        dims_shape = [] 
        dims_shape.append(len(time))
        if groupby_dims is not None:
            for groupbydim in groupby_dims:
                dims_shape.append(self.mean.dims[groupbydim])
                
        ##--------------------------------------------------------------------. 
        # Define coords
        coords = []
        coords.append(time)
        if groupby_dims is not None:
            for groupbydim in groupby_dims:
                coords.append(self.mean[groupbydim].values)
                
        ##--------------------------------------------------------------------.
        # Create DataArray of 1 
        da_ones = xr.DataArray(data=np.ones(dims_shape),
                               coords=coords, 
                               dims=dims) 
        
        ##--------------------------------------------------------------------.
        # Create the climatological forecast
        time_groupby_info = get_time_groupby_idx(data=da_ones,
                                                 time_dim=self.time_dim, 
                                                 time_groups=self.time_groups,
                                                 time_groupby_factors=self.time_groupby_factors)
        time_groupby_dims = time_groupby_info['time_groupby_dims']
        time_groupby_idx = time_groupby_info['time_groupby_idx']
        # - Mean 
        ds_forecast = da_ones.groupby(time_groupby_idx) * self.mean
        
        ##--------------------------------------------------------------------.
        # Remove time groups 
        vars_to_remove = [self.time_groupby_name] + list(time_groupby_dims.data_vars.keys())
        ds_forecast = ds_forecast.drop(vars_to_remove)   
        
        ##--------------------------------------------------------------------.
        # Return the forecast                
        return ds_forecast

def LoadClimatology(fpath):
    """Load Climatology object."""
    # Check .nc 
    if fpath[-3:] != ".nc":
        fpath = fpath + ".nc"
    # Check exist 
    if not os.path.exists(fpath):
        raise ValueError("{} does not exist on disk.".format(fpath))
    # Create scaler 
    ds_clim = xr.open_dataset(fpath)
    return Climatology(data=None, time_dim=None, ds_climatology=ds_clim)

#-----------------------------------------------------------------------------.
# ############### 
#### AnomalyScaler ####
# ############### 
class AnomalyScaler(TemporalStandardScaler):
    """Class object to transform data into anomalies (and back)."""
    
    def __init__(self, 
                 data,
                 time_dim, time_groups=None,
                 variable_dim=None, groupby_dims=None, 
                 reference_period=None, standardized=True, 
                 eps=0.0001, ds_anomaly = None):
        super().__init__(data = data, 
                         time_dim = time_dim,
                         time_groups = time_groups,
                         variable_dim=variable_dim, groupby_dims=groupby_dims, 
                         reference_period=reference_period,
                         center=True, standardize=True, 
                         eps=eps, ds_scaler=ds_anomaly)
        # Set default method
        self.standardize = standardized
    
    def transform(self, new_data, standardized=None, variable_dim=None, rename_dict=None):
        """Transform new_data to anomalies."""
        # Standardize option
        standardize_default = self.standardize
        if standardized is not None:
            self.standardize = standardized 
        # Get anomalies 
        anom = TemporalStandardScaler.transform(self, new_data=new_data, variable_dim=variable_dim, rename_dict=rename_dict)
        # Reset default 
        self.standardize = standardize_default 
        return anom
    
    def inverse_transform(self, new_data, standardized=None, variable_dim=None, rename_dict=None):
        """Retrieve original values from anomalies."""
        # Standardize option
        standardize_default = self.standardize
        if standardized is not None:
            self.standardize = standardized 
        # Inverse
        x = TemporalStandardScaler.inverse_transform(self, new_data=new_data, variable_dim=variable_dim, rename_dict=rename_dict)
        # Reset default 
        self.standardize = standardize_default 
        return x

def LoadAnomaly(fpath):
    """Load xarray scalers.
    
    Useful because return an AnomalyScaler class... which allows to choose 
    between anomalies and std anomalies.
    """
    # Check .nc 
    if fpath[-3:] != ".nc":
        fpath = fpath + ".nc"
    # Check exist 
    if not os.path.exists(fpath):
        raise ValueError("{} does not exist on disk.".format(fpath))
    # Create scaler 
    ds_anomaly = xr.open_dataset(fpath)
    return AnomalyScaler(data=None, time_dim=None, ds_anomaly=ds_anomaly)

#-----------------------------------------------------------------------------.
# ###################### 
#### OneHotEncoding ####
# ######################       
def OneHotEnconding(data, n_categories=None):
    """
    Perform OneHotEnconding of a categorical xarray DataArray.

    Parameters
    ----------
    data : xr.DataArray
        xarray DataArray to OneHotEncode.
    n_categories : int, optional
        Specify the number of categories. The default is None.

    Returns
    -------
    Returns an xarray Dataset with OneHotEncoding variables.

    """
    if not isinstance(data, xr.DataArray):
        raise TypeError("'data' must be a xarray DataArray.")
    if not isinstance(n_categories, (int, type(None))):
        raise TypeError("'n_categories' must be an integer.")
    ##------------------------------------------------------------------------.
    # Convert data as integers
    x = data.values.astype(int) 
    # Compute n_categories 
    if n_categories is None:
        n_categories = np.max(x) + 1
    else: 
        min_n_categories = np.max(x) + 1 
        if n_categories < min_n_categories:
            raise ValueError("'n_categories' must be equal or larger than {}.".format(min_n_categories))
    ##------------------------------------------------------------------------.
    # Compute OHE tensor
    OHE = np.eye(n_categories)[x]
    ##------------------------------------------------------------------------.
    # Create Dataset 
    da_name = data.name
    list_da = []
    for cat in range(n_categories):
        tmp_da = data.copy()
        tmp_da.values = OHE[..., cat]
        tmp_da.name = da_name + " (OHE Class " + str(cat) + ")"
        list_da.append(tmp_da)
    ds = xr.merge(list_da)
    ##------------------------------------------------------------------------.
    return ds 
    
def InvertOneHotEnconding(data, name=None):
    """
    Invert OneHotEnconded variables of an xarray Dataset.

    Parameters
    ----------
    data : xr.Dataset
        xarray Dataset with OneHotEncoded variables
    name: str
        Name of the output xarray DataArray
    Returns
    -------
    Returns an xarray DataArray with categorical labels.

    """   
    if not isinstance(data, xr.Dataset):
        raise TypeError("'data' must be a xarray DataArray.")
    if not isinstance(name, (str, type(None))):
        raise TypeError("'name' must be a string (or None).")
    ##------------------------------------------------------------------------.
    # Convert Dataset to numpy tensor
    OHE = data.to_array('OHE').transpose(...,'OHE').values
    
    (OHE > 1).any() or (OHE < 0).any()
    # Check all values are between 0 or 1 (so that works also for probs)
    if (OHE > 1).any() or (OHE < 0).any():
        raise ValueError("Expects all values to be between 0 and 1")
    ##-----------------------------------------------------------------------.
    # Inverse
    x = np.argmax(OHE, axis=len(OHE.shape) - 1)
    ##------------------------------------------------------------------------.
    # Create DataArray 
    da = data[list(data.data_vars.keys())[0]]
    da.values = x
    da.name = name
    ##------------------------------------------------------------------------.
    return da
 
#-----------------------------------------------------------------------------.
### Hovmoller 
def HovmollerDiagram(data,
                     spatial_dim, 
                     time_dim, bin_edges=None, 
                     bin_width=None,
                     time_groups=None, time_average_before_binning=True, 
                     variable_dim=None):
    """
    Compute an Hovmoller diagram.

    Parameters
    ----------
    data : xr.Data.Array or xr.Data.Array
        Either xr.Data.Array or xr.Dataset.
    spatial_dim : str
        The name of the spatial dimension over which to average values.
    time_dim : str
        The name of the time dimension.
    bin_edges : (list, np.ndarray), optional
        The bin edges over which to aggregate values across the spatial dimension.
        If not specified, bin_width must be specified.
    bin_width : (int, float), optional
        This argument is required if 'bin_edges' is not specified.
        Bins with 'bin_width' are automatically defined based 'spatial_dim' data range. 
    time_groups : TYPE, optional
        DESCRIPTION. The default is None.
    time_average_before_binning : bool, optional
        If 'time_groups' is provided, wheter to average data over time groups before
        or after computation of the Hovmoller diagram. 
        The default is True. 
    variable_dim : str, optional
        If data is a DataArray, 'variable_dim' is used to reshape the tensor to 
        an xr.Dataset with as variables the values of 'variable_dim' 
        This allows to compute the statistic for each 'variable_dim' value. 

    Returns
    -------
    xr.Data.Array or xr.Data.Array
        An Hovmoller diagram.
  
    """
    ##----------------------------------------------------------------.   
    # Check data is an xarray Dataset or DataArray  
    if not (isinstance(data, xr.Dataset) or isinstance(data, xr.DataArray)):
        raise TypeError("'data' must be an xarray Dataset or xarray DataArray.")
                    
    # - Checks for Dataset 
    if isinstance(data, xr.Dataset):
        # Check variable_dim is not specified ! 
        if variable_dim is not None: 
            raise ValueError("'variable_dim' must not be specified for Dataset objects. Use groupby_dims instead.")
            
    # - Checks for DataArray (and convert to Dataset)
    flag_DataArray = False      
    if isinstance(data, xr.DataArray):
        flag_DataArray = True
        da_name = data.name
        # Check variable_dim
        if variable_dim is None: 
            # If not specified, data name will become the dataset variable name
            data = data.to_dataset() 
        else: 
            variable_dim = check_variable_dim(variable_dim = variable_dim, data = data)
            data = data.to_dataset(dim=variable_dim) 
    ##----------------------------------------------------------------.   
    # - Check for spatial_dim  
    spatial_dim = check_spatial_dim(spatial_dim, data)

    # - If spatial_dim is not a dimension-coordinate, swap dimensions
    dims = np.array(list(data.dims))
    if not np.all(np.isin(spatial_dim, dims)):
        dim_tuple = data[spatial_dim].dims
        if len(dim_tuple) != 1:
            raise ValueError("{} 'spatial_dim' coordinate must be 1-dimensional."
                                .format(spatial_dim))
        data = data.swap_dims({dim_tuple[0]: spatial_dim})
    ##----------------------------------------------------------------.
    # - Check for bin_width and bin_edges 
    if bin_edges is None and bin_width is None: 
        raise ValueError("If 'bin_edges' are not specified, specify the desired 'bin_width'.")
    bin_width = check_bin_width(bin_width)
    # - Define bin_edges if not provided 
    min_val = data[spatial_dim].min().values
    max_val = data[spatial_dim].max().values
    tol = 1.e-8
    if bin_edges is None: 
        bin_edges = np.arange(min_val,max_val+tol, bin_width)
    # - Define bin midpoints
    midpoints = bin_edges[:-1] + np.ediff1d(bin_edges)*0.5   
    # - Extend outermost edges to ensure min and max values to be included
    bin_edges[0] -= tol
    bin_edges[-1] += tol
    # - Check bin_edges validity (at least 2 bins) 
    bin_edges = check_bin_edges(bin_edges, lb=min_val, ub=max_val)
    ##----------------------------------------------------------------.
    # Check time_dim  
    time_dim = check_time_dim(time_dim=time_dim, data=data)
    if time_dim == spatial_dim:
        raise ValueError("'spatial_dim' can not be equal to 'time_dim'.")       
    ##----------------------------------------------------------------.
    # Check time_groups 
    time_groups = check_time_groups(time_groups=time_groups)

    ##----------------------------------------------------------------.
    # Retrieve indexing for temporal groupby   
    time_groupby_info = get_time_groupby_idx(data=data,
                                                time_dim=time_dim, 
                                                time_groups=time_groups)
    time_groupby_idx = time_groupby_info['time_groupby_idx']
    time_groupby_dims = time_groupby_info['time_groupby_dims']   

    ##-----------------------------------------------------------------.     
    # Optional aggregation over time before binning by spatial_dim  
    if time_average_before_binning and time_groups is not None: 
        data = data.groupby(time_groupby_idx).mean(time_dim)
        
    ##-----------------------------------------------------------------. 
    # Compute average across spatial dimension bins 
    hovmoller = data.groupby_bins(spatial_dim, bin_edges, right=True).mean(spatial_dim).compute()
    hovmoller[spatial_dim + "_bins"] = midpoints

    ##-----------------------------------------------------------------. 
    # Optional aggregation over time after binning
    if not time_average_before_binning and time_groups is not None: 
        hovmoller = hovmoller.groupby(time_groupby_idx).mean(time_dim)

    ##----------------------------------------------------------------.
    ## Remove non-dimension (Time_GroupBy) coordinate   
    if time_groups is not None:
        time_groups_vars = list(time_groups.keys())
        for time_group in time_groups_vars: 
            hovmoller[time_group] = time_groupby_dims[time_group]
            hovmoller = hovmoller.set_coords(time_group)
        if len(time_groups_vars) == 1: 
            hovmoller = hovmoller.swap_dims({time_groupby_idx.name: time_groups_vars[0]})
        hovmoller = hovmoller.drop(time_groupby_idx.name)
        
    ##--------------------------------------------------------------------.   
    # Reshape to DataArray if new_data was a DataArray
    if flag_DataArray:
        if variable_dim is None:
            return hovmoller.to_array(dim='variable', name=da_name).squeeze().drop('variable')
        else:
            return hovmoller.to_array(dim=variable_dim, name=da_name)
    else: 
        return hovmoller 

    ##----------------------------------------------------------------------------.   