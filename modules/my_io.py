#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 18:46:41 2021

@author: ghiggi
"""
import zarr
import os
import time
import xarray as xr
from modules.utils_io import check_AR_Datasets
from dask.diagnostics import ProgressBar
from modules.utils_zarr import check_chunks
from modules.utils_zarr import check_compressor

##----------------------------------------------------------------------------.   
def _write_zarr(zarr_fpath, ds, chunks="auto",
                compressor="auto", 
                consolidated=True, append=False):
    """Write Xarray Dataset to zarr with custom chunks and compressor per Dataset variable."""
    ### - Define file chunking  
    default_chunks = {'node': -1,
                      'time': 72,
                      'feature': 1}
    chunks = check_chunks(chunks = chunks, 
                          default_chunks = default_chunks,
                          variable_names = list(ds.data_vars.keys())) 
    
    ##------------------------------------------------------------------------.
    # - Define compressor and filters
    default_compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
    compressor = check_compressor(compressor = compressor,  
                                  default_compressor = default_compressor,
                                  variable_names = list(ds.data_vars.keys()))
    
    ##------------------------------------------------------------------------.
    # - Add chunk encoding the dataset
    for var, chunk in chunks.items():
        ds[var] = ds[var].chunk(chunk)  
    # - Add compressor encoding to each DataArray
    for var, comp in compressor.items(): 
        ds[var].encoding['compressor'] = comp
        
    ##------------------------------------------------------------------------.
    ### - Write zarr files
    # - Define zarr store  
    zarr_store = zarr.DirectoryStore(zarr_fpath)  
    # - Write data to new zarr store 
    if not append: 
        r = ds.to_zarr(store=zarr_store, 
                       mode='w', # overwrite if exists already
                       synchronizer=None, group=None, 
                       consolidated = consolidated,
                       compute=False) 
        with ProgressBar():
            r.compute()   
    # - Append data to existing zarr store 
    # ---> !!! Do not check if data are repeated !!! 
    else: 
        r = ds.to_zarr(zarr_store,
                       append_dim="time",
                       mode="a", 
                       consolidated = consolidated, 
                       compute=False) 
        with ProgressBar():
            r.compute()   
            
##----------------------------------------------------------------------------.   
def pl_to_zarr(ds,
               zarr_fpath, 
               var_dict = None, 
               unstack_plev = True, 
               stack_variables = True, 
               chunks="auto", compressor="auto",
               consolidated = True, 
               append = False, 
               attrs={}):
    """Convert pressure level data netCDFs to zarr.
    
    unstack_plev: bool
         If True, it suppress the vertical dimension and it treats every 
           <variable>-<pressure level> as a separate feature
         If False, it keeps the vertical (pressure) dimension
    stack_variables: bool
        If True, all Dataset variables are stacked across a new 'feature' dimension        
    """
    ##------------------------------------------------------------------------.  
    # Check arguments
    if not isinstance(attrs, dict):
        raise TypeError("attrs must be a dictionary")
    if var_dict is not None:
        if not isinstance(var_dict, dict):
            raise TypeError("var_dict must be a dictionary") 
    if not zarr_fpath.endswith(".zarr"):
        zarr_fpath = zarr_fpath + ".zarr"
    if not append:
        if os.path.exists(zarr_fpath):
            raise ValueError(zarr_fpath + " already exists !")

    # Drop information related to mesh vertices 
    ds = ds.drop_vars(['lon_bnds', 'lat_bnds'])
    # Drop lat lon info of nodes 
    # ds = ds.drop_vars(['lon', 'lat'])  
    # Rename ncells to nodes 
    ds = ds.rename({'ncells':'node'})
    # Rename variables 
    if var_dict is not None:
        ds = ds.rename(var_dict)
        
    ##------------------------------------------------------------------------.
    ### - Unstack pressure levels (as separate variables)
    if unstack_plev:
        l_da_vars = []
        for var in ds.keys():
            for i, p_level in enumerate(ds[var]['plev'].values):
                # - Select variable at given pressure level
                tmp_da = ds[var].isel(plev=i)
                # - Update the name to (<var>_<plevel_in_hPa>)
                tmp_da.name = tmp_da.name + str(int(p_level/100))
                # - Remove plev dimension 
                tmp_da = tmp_da.drop_vars('plev')
                # - Append to the list of DataArrays to then be merged
                l_da_vars.append(tmp_da)
                
        ds = xr.merge(l_da_vars)
        
    ##------------------------------------------------------------------------.
    ### - Stack variables 
    # - new_dim : new dimension name 
    # - variable_dim : level name of the new stacked coordinate <new_dim>
    # - name : DataArray name 
    if stack_variables:
        da_stacked = ds.to_stacked_array(new_dim="feature", variable_dim='variable', 
                                         sample_dims=list(ds.dims.keys()), 
                                         name = 'data')
        # - Remove MultiIndex for compatibility with netCDF and Zarr
        da_stacked = da_stacked.reset_index('feature')
        da_stacked = da_stacked.set_index(feature='variable', append=True)
        
        # - Remove attributes from DataArray 
        da_stacked.attrs = {}
        # - Reshape to Dataset
        ds = da_stacked.to_dataset() 
        # - Reorder dimension as requested by PyTorch input 
        ds = ds.transpose('time', 'node', ..., 'feature')
    else: 
        # - Reorder dimension as requested by PyTorch input 
        ds = ds.transpose('time', 'node', ...)
        
    ##------------------------------------------------------------------------.
    ### - Write Zarr store
    # - Add attributes 
    ds.attrs = attrs
    # - Write data
    _write_zarr(zarr_fpath=zarr_fpath, 
                ds = ds, 
                chunks = chunks,
                compressor = compressor,
                consolidated = consolidated,
                append = append)
    ##------------------------------------------------------------------------.
    return None

def toa_to_zarr(ds,
               zarr_fpath, 
               var_dict = None, 
               stack_variables = True, 
               chunks="auto", compressor="auto",
               consolidated = True, 
               append = False, 
               attrs={}):
    """Convert TOA data netCDFs to zarr."""
    ##------------------------------------------------------------------------.  
    # Check arguments
    if not isinstance(attrs, dict):
        raise TypeError("attrs must be a dictionary")
    if var_dict is not None:
        if not isinstance(var_dict, dict):
            raise TypeError("var_dict must be a dictionary") 
    if not zarr_fpath.endswith(".zarr"):
        zarr_fpath = zarr_fpath + ".zarr"
    if not append:
        if os.path.exists(zarr_fpath):
            raise ValueError(zarr_fpath + " already exists !")
    ##------------------------------------------------------------------------.
    # Drop information related to mesh vertices 
    ds = ds.drop_vars(['lon_bnds', 'lat_bnds'])
    # Drop lat lon info of nodes 
    # ds = ds.drop_vars(['lon', 'lat'])  
    # Rename ncells to nodes 
    ds = ds.rename({'ncells':'node'})
    # Rename variables 
    if var_dict is not None:
        ds = ds.rename(var_dict)
        
    ##------------------------------------------------------------------------.
    ### - Stack variables 
    # - new_dim : new dimension name 
    # - variable_dim : level name of the new stacked coordinate <new_dim>
    # - name : DataArray name 
    if stack_variables:
        da_stacked = ds.to_stacked_array(new_dim="feature", variable_dim='variable', 
                                         sample_dims=list(ds.dims.keys()), 
                                         name = 'data')
        # - Remove MultiIndex for compatibility with netCDF and Zarr
        da_stacked = da_stacked.reset_index('feature')
        da_stacked = da_stacked.set_index(feature='variable', append=True)
        
        # - Remove attributes from DataArray 
        da_stacked.attrs = {}
        # - Reshape to Dataset
        ds = da_stacked.to_dataset() 
        # - Reorder dimension as requested by PyTorch input 
        ds = ds.transpose('time', 'node', ..., 'feature')
    else: 
        # - Reorder dimension as requested by PyTorch input 
        ds = ds.transpose('time', 'node', ...)
        
    ##------------------------------------------------------------------------.
    ### - Write Zarr store
    # - Add attributes 
    ds.attrs = attrs
    # - Write data
    _write_zarr(zarr_fpath=zarr_fpath, 
                ds = ds, 
                chunks = chunks,
                compressor = compressor,
                consolidated = consolidated,
                append = append)
    ##------------------------------------------------------------------------.
    return None

### OLD 
##----------------------------------------------------------------------------.
def load_dynamic_dataset(data_dir, chunk_size="auto"):
    z500 = xr.open_mfdataset(f'{data_dir}/data/geopotential_500/*.nc', 
                             lock = False, 
                             combine='by_coords', 
                             chunks={'time': chunk_size}).rename({'z': 'z500'})
                            
    t850 = xr.open_mfdataset(f'{data_dir}/data/temperature_850/*.nc', 
                             lock = False,
                             combine='by_coords',  
                             chunks={'time': chunk_size}).rename({'t': 't850'})
    return xr.merge([z500, t850], compat='override')    
   
def load_bc_dataset(data_dir, chunk_size = "auto"): 
    return xr.open_mfdataset(f'{data_dir}/data/toa_incident_solar_radiation/*.nc',
                             lock = False,
                             combine='by_coords',  
                             chunks={'time': chunk_size})

def load_static_dataset(data_dir):
    return xr.open_dataset(f'{data_dir}/data/constants/constants_5.625deg.nc')
    
##----------------------------------------------------------------------------.
 
def readDatasets(data_dir, feature_type, chunk_size="auto"):
    if feature_type=="dynamic":
        return load_dynamic_dataset(data_dir, chunk_size)
    elif feature_type=="bc":       
        return load_bc_dataset(data_dir, chunk_size)
    elif feature_type=="static":
        return load_static_dataset(data_dir)
    else:
        raise ValueError('Specify valid feature_type')
        
    
def reformat_Datasets(ds_training_dynamic,
                      ds_validation_dynamic = None,
                      ds_static = None,              
                      ds_training_bc = None,         
                      ds_validation_bc = None,
                      preload_data_in_CPU = False):
    ##------------------------------------------------------------------------. 
    # Check Datasets are in the expected format for AR training 
    check_AR_Datasets(ds_training_dynamic = ds_training_dynamic,
                      ds_validation_dynamic = ds_validation_dynamic,
                      ds_static = ds_static,              
                      ds_training_bc = ds_training_bc,         
                      ds_validation_bc = ds_validation_bc)   
                  
    ##------------------------------------------------------------------------.
    ### Load all data into CPU memory here if asked 
    if preload_data_in_CPU is True:
        ##  Dynamic data
        print("- Preload xarray Dataset of dynamic data into CPU memory:")
        t_i = time.time()
        ds_training_dynamic = ds_training_dynamic.compute()
        print('  --> Training Dynamic Dataset: {:.2f}s'.format(time.time() - t_i))
        if ds_validation_dynamic is not None:
            t_i = time.time()
            ds_validation_dynamic = ds_validation_dynamic.compute()
            print('  --> Validation Dynamic Dataset: {:.2f}s'.format(time.time() - t_i))
        
        ##--------------------------------------------------------------------.
        ## Boundary conditions data
        if ds_training_bc is not None: 
            print("- Preload xarray Dataset of boundary conditions data into CPU memory:")
            t_i = time.time()
            ds_training_bc = ds_training_bc.compute()
            print('  --> Training Boundary Condition Dataset: {:.2f}s'.format(time.time() - t_i))
            if ds_validation_bc is not None:
                t_i = time.time()
                ds_validation_bc = ds_validation_bc.compute()
                print('  --> Validation Boundary Condition Dataset: {:.2f}s'.format(time.time() - t_i))
            
    ##------------------------------------------------------------------------. 
    ### Conversion to DataArray and order dimensions 
    # - For dynamic and bc: ['time', 'node', 'features']
    # - For static: ['node', 'features']
    t_i = time.time()
    da_training_dynamic = ds_training_dynamic.to_array(dim='feature', name='Dynamic').transpose('time', 'node', 'feature')
    if ds_validation_dynamic is not None:
        da_validation_dynamic = ds_validation_dynamic.to_array(dim='feature', name='Dynamic').transpose('time', 'node', 'feature')
    else: 
        da_validation_dynamic = None
    if ds_training_bc is not None:
        da_training_bc = ds_training_bc.to_array(dim='feature', name='BC').transpose('time', 'node', 'feature')
    else:
        da_training_bc = None
    if ds_validation_bc is not None:
        da_validation_bc = ds_validation_bc.to_array(dim='feature', name='BC').transpose('time', 'node', 'feature')
    else:
        da_validation_bc = None
    if ds_static is not None: 
        da_static = ds_static.to_array(dim='feature', name='Static').transpose('node','feature') 
    else: 
        da_static = None
    print('- Conversion to xarray DataArrays: {:.2f}s'.format(time.time() - t_i))
    
    ##------------------------------------------------------------------------.
    dict_DataArrays = {'da_training_dynamic': da_training_dynamic, 
                       'da_validation_dynamic': da_validation_dynamic,
                       'da_training_bc': da_training_bc, 
                       'da_validation_bc': da_validation_bc,
                       'da_static': da_static
                       }
    return dict_DataArrays
