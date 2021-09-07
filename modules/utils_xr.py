import numpy as np 
import xarray as xr

def is_dask_DataArray(da):
    """Check if data in the xarray DataArray are lazy loaded."""
    if da.chunks is not None:
        return True
    else: 
        return False
    
def xr_has_dim(x, dim): 
    if not isinstance(dim, str): 
        raise TypeError("'dim' must be a string.")
    if isinstance(x, xr.Dataset):
        dims = list(x.dims.keys())
    elif isinstance(x, xr.DataArray):
        dims = list(x.dims)
    if dim not in dims: 
        return False
    return True 

def xr_has_coord(x, coord): 
    if not isinstance(coord, str): 
        raise TypeError("'coord' must be a string.")
    coords = list(x.coords.keys())
    if coord not in coords: 
        return False
    return True 

def xr_Dataset_vars(x): 
    if isinstance(x, xr.Dataset): 
        return list(x.data_vars.keys())
    else:
        raise TypeError("Expecting xr.Dataset")    
    
def xr_n_vars(x):
    if isinstance(x, xr.DataArray): 
        return 0 
    elif isinstance(x, xr.Dataset): 
        return len(list(x.data_vars.keys()))
    else:
        raise TypeError("Expecting xr.DataArray or xr.Dataset")

def xr_has_uniform_resolution(x, dim = "time"):
    dt = np.unique(np.diff(x[dim].values))
    if len(dt) == 1: 
        return True 
    else:
        return False 
        
def xr_align_dim(x,y, dim='time'):
    if x is None or y is None:
        return x, y
    all_dims = set(list(x.dims) + list(y.dims))
    exclude_dims = all_dims.remove(dim) 
    x, y = xr.align(x, y, join='inner', exclude=exclude_dims) 
    return x,y 

def xr_align_start_time(x, y, time_dim='time'):
    if x is None or y is None:
        return x, y
    time_start = np.min([x[time_dim].values, y[time_dim].values])
    x = x.sel({time_dim: slice(time_start, None)})
    y = y.sel({time_dim: slice(time_start, None)})
    return x, y

def xr_is_aligned(x,y, exclude=None):
    if isinstance(exclude, str):
        exclude = [exclude]
    # - Retrieve dims  
    dims_x = set(list(x.dims))
    dims_y = set(list(y.dims))
    # - Remove dims to exclude
    if exclude is not None:
        _ = [dims_x.discard(excl) for excl in exclude]
        _ = [dims_y.discard(excl) for excl in exclude]
    # - Check dim order 
    if not np.array_equal(list(dims_x), list(dims_y)):
        return False 
    # - Check dimension values matching
    dims = list(dims_x)
    x_dims_dict = {dim: x[dim].values for dim in dims}
    y_dims_dict = {dim: y[dim].values for dim in dims}
    for dim in dims: 
        if not np.array_equal(x_dims_dict[dim], y_dims_dict[dim]):
            return False 
    return True 

def xr_have_same_timesteps(x, y, time_dim='time'):
    if x is None or y is None: 
        return True 
    return np.array_equal(x[time_dim].values, y[time_dim].values)

def xr_common_vars(x,y):
    """ Retrieve common variables between two xr.Dataset."""  
    if not isinstance(x, xr.Dataset):
        raise TypeError("Expecting xr.Dataset.")
    if not isinstance(y, xr.Dataset):
        raise TypeError("Expecting xr.Dataset.")
    # Retrieve common vars
    x_vars = list(x.data_vars.keys())
    y_vars = list(y.data_vars.keys())
    common_vars = list(set(x_vars).intersection(set(y_vars)))
    if len(common_vars) == 0: 
        return None 
    else: 
        return common_vars

def xr_have_Dataset_vars_same_dims(ds):
    if not isinstance(ds, xr.Dataset):
        raise TypeError("Expecting an xr.Dataset within xr_have_Dataset_vars_same_dims().")
    unordered_dims = list(ds.dims) # This does not correspond to dims of DataArrays !!!!
    variables = list(ds.data_vars.keys()) 
    # - Get the dimension of the first DataArray as reference (and check it has all the Dataset dimensions)
    dims = list(ds[variables[0]].dims)
    missing_dim = np.array(unordered_dims)[np.isin(unordered_dims, dims, invert=True)].tolist()
    if len(missing_dim) >= 1:
        print("The Dataset variable {!r} does not have dimensions {!r}.".format(variables[0], missing_dim)) 
    for var in variables:
        da_dims = list(ds[var].dims)
        if not np.array_equal(da_dims, dims):
            missing_dim = np.array(dims)[np.isin(dims, da_dims, invert=True)].tolist()
            if len(missing_dim) >= 1:
                print("The Dataset variable {!r} does not have dimensions {!r}.".format(var, missing_dim))
            else:
                print("The Dataset variable {!r} have dimension {!r} instead of {!r}.".format(var, da_dims, dims))
            return False
    return True 