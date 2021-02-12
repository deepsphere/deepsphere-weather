#!/usr/bin/env python3
"""
Created on Mon Dec 14 13:25:46 2020

@author: ghiggi
"""
import xarray as xr
import numpy as np
import os
# import xscaler
# xscaler.GlobalScaler.MinMaxScaler
# xscaler.GlobalScaler.StandardScaler  
# xscaler.GlobalScaler.TrendScaler

### TemporalScalers 
# --> When new_data contain new time_groupby indices values, insert NaN values
#     in mean_, std_  for the missing time_groupby values
 
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

##----------------------------------------------------------------------------.
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
        raise ValueError("'groupby_dims' must be a dimension coordinates of the xarray object") 
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
            return data[variable_dim].values.tolist()
    else: 
        raise TypeError("Provide an xarray Dataset or DataArray")
   
#-----------------------------------------------------------------------------.
######################
### Global Scalers ###
######################
# - Statistics over all space, at each timestep: groupby_dims = "time"

class GlobalStandardScaler():
    """Aggregate over all dimensions (except variable_dim and groupby_dims)."""
    
    def __init__(self, data, variable_dim=None, groupby_dims=None, 
                 center=True, standardize=True, eps=0.0001, ds_scaler=None):
        # ds_scaler must not be specified. Use load_scaler(fpath) if you want to load an existing scaler from disk.
        ##--------------------------------------------------------------------.
        ### Load an already existing scaler (if ds_scaler is provided) 
        if isinstance(ds_scaler, xr.Dataset):
            self.scaler_class = ds_scaler.attrs['scaler_class']
            self.eps = ds_scaler.attrs['eps']
            self.aggregating_dims = ds_scaler.attrs['aggregating_dims']
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
            if ((center is False) and (standardize is False)):
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
            # Save variable_dim if data is DataArray and using fit_transform()
            self.variable_dim = variable_dim

    ##------------------------------------------------------------------------.
    def fit(self):
        """Fit the GlobalStandardScaler."""
        #---------------------------------------------------------------------.
        # Checks
        if self.fitted is True: 
            raise ValueError("The scaler has been already fitted!")
        #---------------------------------------------------------------------.
        # Fit the scaler 
        if self.center:
            self.mean_ = self.data.mean(self.aggregating_dims).compute() 
        if self.standardize:
            self.std_ = self.data.std(self.aggregating_dims).compute() 
        self.fitted = True
        # del self.data
    
    def save(self, fpath):
        """Save the scaler object to disk in netCDF format."""
        ##--------------------------------------------------------------------.
        # Checks
        if self.fitted is False:
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
                           'aggregating_dims': self.aggregating_dims, 
                           'center': str(self.center),
                           'standardize': str(self.standardize),
                           }
        ds_scaler.to_netcdf(fpath)
        print("The GlobalStandardScaler has been written to disk!")
        
    ##------------------------------------------------------------------------.   
    def transform(self, new_data, variable_dim=None, rename_dict=None): 
        """Transform data using the fitted GlobalStandardScaler."""
        ##--------------------------------------------------------------------.
        if self.fitted is False: 
            raise ValueError("The GlobalStandardScaler need to be first fit() !.")
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
        # Get variables to transform 
        data_vars = get_xarray_variables(new_data)
        if self.center:
            transform_vars = get_xarray_variables(self.mean_)
        else:
            transform_vars = get_xarray_variables(self.std_)    
        transform_vars = np.array(transform_vars)[np.isin(transform_vars, data_vars)]
        
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
        if len(transform_vars) > 0: 
            new_data = new_data.copy()
            for var in transform_vars:
                if self.center: 
                    new_data[var] = new_data[var] - self.mean_[var] 
                if self.standardize: 
                    new_data[var] = new_data[var] / (self.std_[var] + self.eps)    
                    
        ##--------------------------------------------------------------------.   
        # Rename dimension as new_data (if necessary)
        if flag_dim_renamed is True: 
            new_data = new_data.rename(inv_rename_dict)

        ##--------------------------------------------------------------------.      
        # Reshape to DataArray if new_data was a DataArray
        if flag_DataArray is True:
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
        if self.fitted is False: 
            raise ValueError("The GlobalStandardScaler need to be first fit() !.")
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
        # Get variables to transform 
        data_vars = get_xarray_variables(new_data)
        if self.center:
            transform_vars = get_xarray_variables(self.mean_)
        else:
            transform_vars = get_xarray_variables(self.std_)    
        transform_vars = np.array(transform_vars)[np.isin(transform_vars, data_vars)]
        
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
        if len(transform_vars) > 0: 
            new_data = new_data.copy()
            for var in transform_vars:
                if self.standardize: 
                    new_data[var] = new_data[var] * (self.std_[var] + self.eps)  
                    
                if self.center: 
                    new_data[var] = new_data[var] + self.mean_[var] 
        
        ##--------------------------------------------------------------------.   
        # Rename dimension as new_data (if necessary)
        if flag_dim_renamed is True: 
            new_data = new_data.rename(inv_rename_dict)

        ##--------------------------------------------------------------------.      
        # Reshape to DataArray if new_data was a DataArray           
        if flag_DataArray is True:
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
        if self.fitted is True:
            raise ValueError("The scaler has been already fitted. Please use .transform().") 
        ##--------------------------------------------------------------------.
        self.fit()
        return self.transform(new_data=self.data, variable_dim=self.variable_dim)
    
##----------------------------------------------------------------------------.
class GlobalMinMaxScaler():   
    """GlobalMinMaxScaler."""
    
    def __init__(self, variable_dims=None,
                 feature_min = 0, feature_max = 1):
        """TODO."""
        self.feature_min = feature_min
        self.feature_max = feature_max
    
    def fit(self):
        """TODO."""
        self.fitted = True
        self.min_ = self.data.min().compute()
        self.max_ = self.data.max().compute()
        self.range_ = self.data_max_ - self.data_min_  
        self.scaling = self.feature_max - self.feature_min
    
    def transform(self, data):
        """TODO."""
        return (data - self.min_) / self.range_ *self.scaling + self.feature_min

    def fit_transform(self, data):
        """TODO."""
        self.fit()
        return self.transform(data)
    
    def inverse_transform(self, data): 
        """TODO."""
        return (data - self.feature_min) * self.range_ / self.scaling + self.min_

### GlobalTrendScaler
#---------------------------------------------------------------------------.
##################################
### Utils for TemporalScalers ####
##################################
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
    time_groupby_name = '-'.join(time_groups_list)    
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

def check_dict_factors(dict_factors, time_groups):
    """Check validity of dict_factors."""
    if dict_factors is None: 
        return {} 
    if time_groups is not None: 
        if not np.all(np.isin(time_groups.keys(), dict_factors.keys())):
            raise ValueError("All time groups must be included in dict_factors.")
        return dict_factors
    else:
        return {} 

def check_new_time_groupby_idx(time_groupby_idx, scaler_stat):
    """Check that the fitted scaler contains all time_groupby_idx of new_data."""
    time_groupby_idx_orig = np.unique(scaler_stat[time_groupby_idx.name].values)
    time_groupby_idx_new = np.unique(time_groupby_idx.values)
    if not np.all(np.isin(time_groupby_idx_new, time_groupby_idx_orig)):
        raise ValueError("The TemporalScaler does not contain representative statistics for all time_groups indices of 'new_data'.")       
        
##----------------------------------------------------------------------------.   
def get_time_groupby_idx(data, time_dim, time_groups, dict_factors=None): 
    """Return a 1D array with unique index for temporal groupby operation."""
    # Check dict_factors 
    dict_factors = check_dict_factors(dict_factors, time_groups=time_groups)  
    no_dict_factors = len(dict_factors) == 0
    ##------------------------------------------------------------------------.
    # Retrieve groupby indices 
    if time_groups is not None: 
        tmp_min_interval = 0
        l_idx = []     # TODO: remove in future
        for i, (time_group, time_agg) in enumerate(time_groups.items()):
            # Retrieve max time aggregation
            time_agg_max = get_time_group_max(time_group=time_group)
            # Retrieve time index (for specific time group)
            # idx = data[time_dim].dt.isocalendar().week  # dt.week, dt.weekofyear has been deprecated in Pandas ... but xarray not updated
            idx = data[time_dim].dt.__getattribute__(time_group)
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
            if no_dict_factors: 
                # get_numeric_combo_factor()
                if tmp_min_interval == 0:
                    idx_scaled = idx_norm # *10⁰
                    tmp_min_interval = np.max(np.unique(idx_scaled))
                    dict_factors[time_group] = 0 # 10^0 = 1
                    time_groupby_idx = idx_scaled
                    
                    l_idx.append(idx_scaled)     # TODO: remove in future 
                    
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
                    dict_factors[time_group] = factor
                     
                    l_idx.append(idx_scaled)  # TODO: remove in future 
            else: 
                idx_scaled = idx_norm*(10**dict_factors[time_group])  
                if i == 0:
                    time_groupby_idx = idx_scaled
                else:
                    time_groupby_idx = time_groupby_idx + idx_scaled
        ##---------------------------------------------------------------------.
        # Add name to time grouby indices     
        time_groupby_idx.name = get_time_groupby_name(time_groups)    
    # If no time_groups are specified --> Long-term mean    
    else: 
        # Set all indices to 0 (unique group over time --> long-term mean)
        time_groupby_idx = data.time.dt.month
        time_groupby_idx[:] = 0
        time_groupby_idx.name = "Long-term mean"
    return time_groupby_idx, dict_factors
  
#-----------------------------------------------------------------------------.
########################
### Temporal Scalers ###
########################
# In future: multidimensional groupby? :
# - http://xarray.pydata.org/en/stable/groupby.html
# - http://xarray.pydata.org/en/stable/generated/xarray.IndexVariable.html#xarray.IndexVariable 
# - https://github.com/pydata/xarray/issues/324
# - https://github.com/pydata/xarray/issues/1569

class TemporalStandardScaler():
    """Aggregate over all dimensions (except variable_dim and groupby_dims)."""
    
    def __init__(self, data, time_dim, time_groups=None,
                 variable_dim=None, groupby_dims=None, 
                 center=True, standardize=True, eps=0.0001, ds_scaler = None):
        # ds_scaler must not be specified. Use load_scaler(fpath) if you want to load an existing scaler from disk.
        ##--------------------------------------------------------------------.
        ### Load an already existing scaler (if ds_scaler is provided) 
        if isinstance(ds_scaler, xr.Dataset):
            self.scaler_class = ds_scaler.attrs['scaler_class']
            self.eps = ds_scaler.attrs['eps']
            self.aggregating_dims = ds_scaler.attrs['aggregating_dims']
            self.time_dim = ds_scaler.attrs['time_dim']
            self.time_groups = eval(ds_scaler.attrs['time_groups'])
            self.dict_factors = eval(ds_scaler.attrs['dict_factors'])
            self.time_groupby_name = 'time_groupby_name'
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
            if ((center is False) and (standardize is False)):
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
            ##--------------------------------------------------------------------.
            # Check time_dim  
            time_dim = check_time_dim(time_dim=time_dim, data=data)
            self.time_dim = time_dim 
            ##---------------------------------------------------------------------.
            # Define groupby dimensions (over which to groupby)
            if groupby_dims is not None:
                groupby_dims = check_groupby_dims(groupby_dims=groupby_dims, data = data) 
                if time_dim in groupby_dims:
                    raise ValueError("TemporalScalers does not allow 'time_dim' to be included in 'groupby_dims'.")
            ##--------------------------------------------------------------------.
            # Check time_groups 
            time_groups = check_time_groups(time_groups=time_groups)
            self.time_groups = time_groups
            ##--------------------------------------------------------------------.
            # Retrieve indexing for temporal groupby   
            time_groupby_idx, dict_factors = get_time_groupby_idx(data=data, time_dim=time_dim, time_groups=time_groups)
            self.time_groupby_idx = time_groupby_idx
            self.time_groupby_name = time_groupby_idx.name
            self.dict_factors = dict_factors
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
            # Save variable_dim if data is DataArray and using fit_transform()
            self.variable_dim = variable_dim  
        
    ##------------------------------------------------------------------------. 
    def fit(self):
        """Fit the TemporalStandardScaler."""
        ##---------------------------------------------------------------------.
        if self.fitted is True: 
            raise ValueError("The scaler has been already fitted!")
        ##---------------------------------------------------------------------.
        # Fit the scaler 
        if self.center:
            self.mean_ = self.data.groupby(self.time_groupby_idx).mean(self.aggregating_dims).compute() 
        if self.standardize:
            self.std_ = self.data.groupby(self.time_groupby_idx).std(self.aggregating_dims).compute() 
        self.fitted = True
        # del self.data
    
    def save(self, fpath):
        """Save the scaler object to disk in netCDF format."""
        if self.fitted is False:
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
                           'aggregating_dims': self.aggregating_dims, 
                           "time_dim": self.time_dim, 
                           'center': str(self.center),
                           'standardize': str(self.standardize),
                           'dict_factors': str(self.dict_factors), 
                           'time_groups': str(self.time_groups),
                           'time_groupby_name': self.time_groupby_name
                           }
        ds_scaler.to_netcdf(fpath)
        print("The TemporalStandardScaler has been written to disk!")
        
    ##------------------------------------------------------------------------.   
    def transform(self, new_data, variable_dim=None, rename_dict=None): 
        """Transform data using the fitted TemporalStandardScaler."""
        ##--------------------------------------------------------------------.
        if self.fitted is False: 
            raise ValueError("The TemporalStandardScaler need to be first fit() !")
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
        # Get variables to transform 
        data_vars = get_xarray_variables(new_data)
        if self.center:
            transform_vars = get_xarray_variables(self.mean_)
        else:
            transform_vars = get_xarray_variables(self.std_)    
        transform_vars = np.array(transform_vars)[np.isin(transform_vars, data_vars)]
                
        ##--------------------------------------------------------------------.
        # Get time grouby indices
        time_groupby_idx, _ = get_time_groupby_idx(data=new_data,
                                                   time_dim=self.time_dim, 
                                                   time_groups=self.time_groups,
                                                   dict_factors=self.dict_factors)
        
        # Check that the fitted scaler contains all time_groupby_idx of new_data
        if self.center: 
            check_new_time_groupby_idx(time_groupby_idx, scaler_stat = self.mean_)
        else: 
            check_new_time_groupby_idx(time_groupby_idx, scaler_stat = self.std_)    
        ##--------------------------------------------------------------------.
        ## Transform variables 
        if len(transform_vars) > 0: 
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
        if flag_dim_renamed is True: 
            new_data = new_data.rename(inv_rename_dict)
            
        ##--------------------------------------------------------------------.   
        # Reshape to DataArray if new_data was a DataArray
        if flag_DataArray is True:
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
        if self.fitted is False: 
            raise ValueError("The TemporalStandardScaler need to be first fit() !")
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
        # Get variables to transform 
        data_vars = get_xarray_variables(new_data)
        if self.center:
            transform_vars = get_xarray_variables(self.mean_)
        else:
            transform_vars = get_xarray_variables(self.std_) 
        transform_vars = np.array(transform_vars)[np.isin(transform_vars, data_vars)]
        
        ##--------------------------------------------------------------------.
        # Get time grouby indices
        time_groupby_idx, _ = get_time_groupby_idx(data=new_data,
                                                   time_dim=self.time_dim, 
                                                   time_groups=self.time_groups,
                                                   dict_factors=self.dict_factors)
        
        # Check that the fitted scaler contains all time_groupby_idx of new_data
        if self.center: 
            check_new_time_groupby_idx(time_groupby_idx, scaler_stat = self.mean_)
        else: 
            check_new_time_groupby_idx(time_groupby_idx, scaler_stat = self.std_)

        ##--------------------------------------------------------------------.
        ## Transform variables 
        if len(transform_vars) > 0: 
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
        if flag_dim_renamed is True: 
            new_data = new_data.rename(inv_rename_dict)
            
        ##--------------------------------------------------------------------.   
        # Reshape to DataArray if new_data was a DataArray
        if flag_DataArray is True:
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
        if self.fitted is True:
            raise ValueError("The scaler has been already fitted. Please use .transform().") 
        ##--------------------------------------------------------------------.
        self.fit()
        return self.transform(new_data=self.data, variable_dim=self.variable_dim)        
            
##-----------------------------------------------------------------------------. 
### TemporalMinMaxScaler 
### TemporalTrendScaler
 
#-----------------------------------------------------------------------------.
#####################
### Load Scalers ####
#####################
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
    if scaler_class == "TemporalStandardScaler":
        return TemporalStandardScaler(data=None, time_dim=None, ds_scaler=ds_scaler)


#-----------------------------------------------------------------------------.
########################## 
### SequentialScalers ####
########################## 
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

    def fit(self): 
        """Fit all scalers within a SequentialScaler."""
        new_list_scaler = []
        for scaler in self.list_scalers:
            if scaler.fitted is False:
                scaler.fit()
            new_list_scaler.append(scaler)
        self.list_scalers = new_list_scaler
        self.fitted = True
        
    def save(self): 
        """Save a SequentialScaler to disk."""
        raise NotImplementedError("Saving of SequentialScaler has not been yet implemented!")
        
    def transform(self, new_data, variable_dim = None, rename_dict=None): 
        """Transform data using the fitted SequentialScaler."""
        for scaler in self.list_scalers:
            if scaler.fitted is False:
                raise ValueError("The SequentialScaler contains scalers that have not been fit. Use .fit() first!")
        for scaler in self.list_scalers:
            new_data = scaler.transform(new_data=new_data, variable_dim = variable_dim, rename_dict=rename_dict)
        return new_data
    
    def inverse_transform(self, new_data, variable_dim = None, rename_dict=None): 
        """Inverse transform data using the fitted SequentialScaler."""
        reversed_scalers = self.list_scalers[::-1] 
        for scaler in reversed_scalers:
            if scaler.fitted is False:
                raise ValueError("The SequentialScaler contains scalers that have not been fit. Use .fit() first!")
        for scaler in reversed_scalers:
            new_data = scaler.inverse_transform(new_data=new_data, variable_dim = variable_dim, rename_dict=rename_dict)
        return new_data 

#-----------------------------------------------------------------------------.
####################
### Climatology ####
####################
class Climatology():
    def __init__(self, data, time_dim, time_groups=None,
                 variable_dim=None, groupby_dims = None, ds_climatology=None):
        # Init 
        if ds_climatology is None: 
            self.scaler = TemporalStandardScaler(data=data, 
                                                time_dim = time_dim, 
                                                time_groups = time_groups,
                                                variable_dim = variable_dim, 
                                                groupby_dims = groupby_dims,
                                                center=True, standardize = False)
            self.fitted = False
            
    def compute(self):
        self.data = self.scaler.fit().mean_ # TODO: return dataarray if if data is dataarray
        self.fitted = True 
    
    # # - Option to add time group dimensions to mean_, std_
    
    # groupby ... and then multiply by 1 
        
    # def forecast(datetime64_arr):
         # Retrieve the values for each timestep based on ds_clim 
         # ds_forecast = None # TODO
         # return ds_forecast

#-----------------------------------------------------------------------------.
####################
### Variability ####
####################
class Variability():
    def __init__(self, data, time_dim, time_groups=None,
                 variable_dim=None, groupby_dims = None, ds_variability=None):
        # Init 
        if ds_variability is None: 
            self.scaler = TemporalStandardScaler(data=data, 
                                                time_dim = time_dim, 
                                                time_groups = time_groups,
                                                variable_dim = variable_dim, 
                                                groupby_dims = groupby_dims,
                                                center=True, standardize = False)
            self.fitted = False
            
    def compute(self):
        self.data = self.scaler.fit().mean_ # TODO: return dataarray if if data is dataarray
        self.fitted = True 
        
    # def forecast(datetime64_arr):
         # Retrieve the values for each timestep based on ds_clim 
         # ds_forecast = None # TODO
         # return ds_forecast


##----------------------------------------------------------------------------.