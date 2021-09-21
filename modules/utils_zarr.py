#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 16:11:33 2021

@author: ghiggi
"""
import os
import shutil
import time 
import itertools
import dask 
import zarr 
import numcodecs
import numpy as np
import xarray as xr 
from rechunker import rechunk
from dask.diagnostics import ProgressBar

#-----------------------------------------------------------------------------.
##########################
#### Compressor utils ####
##########################
def is_numcodecs(compressor):
    """Check is a numcodec compressor."""
    if type(compressor).__module__.find("numcodecs") == -1:
        return False
    else:
        return True

##----------------------------------------------------------------------------.
def check_compressor(compressor, variable_names, 
                     default_compressor = None):
    """Check compressor validity for zarr writing.
    
    compressor = None --> No compression.
    compressor = "auto" --> Use default_compressor if specified. Otherwise will default to ds.to_zarr() default compressor.
    compressor = <numcodecs class> --> Specify the same compressor to all Dataset variables
    compressor = {..} --> A dictionary specifying a compressor for each Dataset variable.
   
    default_compressor: None or numcodecs compressor. None will default to ds.to_zarr() default compressor.
    variable_names: list of xarray Dataset variables
    """
    ##------------------------------------------------------------------------.
    # Check variable_names type 
    if not isinstance(variable_names, (list, str)):
        raise TypeError("'variable_names' must be a string or a list")
    if isinstance(variable_names, str):
        variable_names = [variable_names]
    if not all([isinstance(s, str) for s in variable_names]):
        raise TypeError("Specify all variable names as string within the 'variable_names' list.")
    
    # Check compressor type 
    if not (isinstance(compressor, (str, dict, type(None))) or is_numcodecs(compressor)):
        raise TypeError("'compressor' must be a dictionary, numcodecs compressor, 'auto' string or None.")
    if isinstance(compressor, str): 
        if not compressor == "auto":
            raise ValueError("If 'compressor' is specified as string, must be 'auto'.")
    if isinstance(compressor, dict):
        if not np.all(np.isin(list(compressor.keys()), variable_names)):
            raise ValueError("The 'compressor' dictionary must contain the keys {}".format(variable_names))
    # Check default_compressor type 
    if not (isinstance(default_compressor, (dict, type(None))) or is_numcodecs(default_compressor)):
        raise TypeError("'default_compressor' must be a numcodecs compressor or None.")
    if isinstance(default_compressor, dict):
        if not np.all(np.isin(list(default_compressor.keys()), variable_names)):
            raise ValueError("The 'default_compressor' dictionary must contain the keys {}".format(variable_names))
    ##------------------------------------------------------------------------.
    # If a string --> "Auto" --> Apply default_compressor (if specified)
    if isinstance(compressor, str): 
        if compressor == "auto":
            compressor = default_compressor
  
    ##------------------------------------------------------------------------.        
    # If a dictionary, check keys validity and compressor validity 
    if isinstance(compressor, dict):
        if not all([is_numcodecs(cmp) or isinstance(cmp, type(None)) for cmp in compressor.values()]):
            raise ValueError("The compressors specified in the 'compressor' dictionary must be numcodecs (or None).")
    ##------------------------------------------------------------------------.
    # If a unique compressor, create a dictionary with the same compressor for all variables
    if is_numcodecs(compressor) or isinstance(compressor, type(None)):
        compressor = {var: compressor for var in variable_names}
    ##------------------------------------------------------------------------.    
    return compressor

#-----------------------------------------------------------------------------.
######################
#### Chunks utils ####
######################
def _all_valid_chunks_values(values): 
    bool_list = []
    for x in values: 
        if isinstance(x, str): 
            if x == "auto":
                bool_list.append(True)
            else:
                bool_list.append(False)
        elif isinstance(x, int):
            bool_list.append(True)
        elif isinstance(x, type(None)):
            bool_list.append(True) # Require caution 
        else:
            bool_list.append(False)
    return all(bool_list)   

def get_ds_chunks(ds): 
    variable_names = list(ds.data_vars.keys())
    chunks = {} 
    for var in variable_names: 
        if ds[var].chunks is not None:
            chunks[var] = {dim: v[0] for dim,v in zip(ds[var].dims, ds[var].chunks)}
        else:
            chunks[var] = None
    return chunks 
                     
def check_chunks(ds, chunks, 
                 default_chunks = None):
    """Check chunks validity.
    
    chunks = None --> Keeps current chunks.
    chunks = "auto" --> Use default_chunks if specified, otherwise defaults to xarray "auto" chunks.
    chunks = {.-.}  -->  Custom chunks  
    # - Option 1: A dictionary of chunks definitions for each Dataset variable 
    # - Option 2: A single chunk definition to be applied to all Dataset variables 
    # --> Attention: -1 and None are equivalent chunk values !!!
   
    default_chunks is used only if chunks = "auto"
    """
    # Check chunks  
    if not isinstance(chunks, (str, dict, type(None))):
        raise TypeError("'chunks' must be a dictionary, 'auto' or None.")
    if isinstance(chunks, str): 
        if not chunks == "auto":
            raise ValueError("If 'chunks' is specified as string, must be 'auto'.")
    # Check default chunks
    if not isinstance(default_chunks, (dict, type(None))):
        raise TypeError("'default_chunks' must be either a dictionary or None.")
    # Check variable_names 
    if not isinstance(ds, xr.Dataset):
        raise TypeError("'ds' must be an xarray Dataset.")
    #-------------------------------------------------------------------------.
    # Retrieve Dataset infos 
    variable_names = list(ds.data_vars.keys())
    dim_names = list(ds.dims)
    #-------------------------------------------------------------------------.        
    # Retrieve chunks and default_chunks formats when provided as dictionary
    if isinstance(chunks, dict):
        CHUNKS_PER_VARIABLE = np.all(np.isin(list(chunks.keys()), variable_names))
        CHUNKS_DIMS = np.all(np.isin(list(chunks.keys()), dim_names))
        if not CHUNKS_PER_VARIABLE and not CHUNKS_DIMS:
            if np.any(np.isin(list(chunks.keys()), dim_names)):
                print("The 'chunks' dictionary must contain the keys {}".format(dim_names))
                raise ValueError("Please specify specific chunks for each Dataset dimension.")
            if np.any(np.isin(list(chunks.keys()), variable_names)):
                print("The 'chunks' dictionary must contain the keys {}".format(variable_names))
                raise ValueError("Please specify specific chunks for each Dataset variable.")
    if isinstance(default_chunks, dict): 
        DEFAULT_CHUNKS_PER_VARIABLE = np.all(np.isin(list(default_chunks.keys()), variable_names))
        DEFAULT_CHUNKS_DIMS = np.all(np.isin(list(default_chunks.keys()), dim_names))
        if not DEFAULT_CHUNKS_PER_VARIABLE and not DEFAULT_CHUNKS_DIMS:
            if np.any(np.isin(list(default_chunks.keys()), dim_names)):
                raise ValueError("Please specify specific default_chunks for each Dataset dimension.")
            if np.any(np.isin(list(default_chunks.keys()), variable_names)):
                raise ValueError("Please specify specific default_chunks for each Dataset variable.")
    ##------------------------------------------------------------------------.
    # If chunks = "auto"  
    if isinstance(chunks, str): 
        # --> If default_chunks is a dict, assign to chunks  
        if isinstance(default_chunks, dict): 
            chunks = default_chunks
            CHUNKS_PER_VARIABLE = DEFAULT_CHUNKS_PER_VARIABLE
            CHUNKS_DIMS = DEFAULT_CHUNKS_DIMS
        # --> If default_chunks is None, assign "auto" to all dimensions  
        else: 
            chunks = {dim: "auto" for dim in dim_names}
            CHUNKS_PER_VARIABLE = False
            CHUNKS_DIMS = True
            
    ##------------------------------------------------------------------------.
    # If chunks = None
    if isinstance(chunks, type(None)):
        chunks = get_ds_chunks(ds)
        CHUNKS_PER_VARIABLE = True
        CHUNKS_DIMS = False
        
    ##------------------------------------------------------------------------.
    # If a dictionary, check chunks valid keys and values  
    if isinstance(chunks, dict):       
        # If 'chunks' specify specific chunks for each Dataset variable 
        if CHUNKS_PER_VARIABLE:
            # - For each variable 
            for var in variable_names:
                # - Check that the chunk value for each dimension is specified
                if not np.all(np.isin(list(chunks[var].keys()), list(ds[var].dims))):
                    raise ValueError("The 'chunks' dictionary of {} must contain the keys {}".format(var, list(ds[var].dims)))
                # - Check chunks value validity 
                if not _all_valid_chunks_values(list(chunks[var].values())):
                    raise ValueError("Unvalid 'chunks' values for {}.".format(var))
        # If 'chunks' specify specific chunks for all Dataset dimensions 
        elif not CHUNKS_PER_VARIABLE:
            # - Checks chunks value validity
            if not _all_valid_chunks_values(list(chunks.values())):
                raise ValueError("Unvalid 'chunks' values")
            # - Create dictionary for each variable  
            new_chunks = {}
            for var in variable_names: 
                new_chunks[var] = {dim: chunks[dim] for dim in ds[var].dims}
            chunks = new_chunks 
        else: 
            raise ValueError("This chunks option has not been implemented.")
    ##------------------------------------------------------------------------.    
    return chunks 

#-----------------------------------------------------------------------------.
#######################
#### Rounding utils ###
#######################
def check_rounding(rounding, variable_names):
    """Check rounding validity.
    
    rounding = None --> No rounding.
    rounding = int --> All variables will be round to the specified decimals.
    rounding = dict --> Specify specific rounding for each variable
    """
    ##------------------------------------------------------------------------.
    # Check variable_names type 
    if not isinstance(variable_names, (list, str)):
        raise TypeError("'variable_names' must be a string or a list.")
    if isinstance(variable_names, str):
        variable_names = [variable_names]
    if not all([isinstance(s,str) for s in variable_names]):
        raise ValueError("Specify all variable names as string within the 'variable_names' list.")
    # Check rounding type 
    if not isinstance(rounding, (int, dict, type(None))):
        raise TypeError("'rounding' must be a dictionary, integer or None.")
    ##------------------------------------------------------------------------.   
    # If a dictionary, check valid keys and valid compressor
    if isinstance(rounding, dict):
        if not np.all(np.isin(list(rounding.keys()), variable_names)):
            raise ValueError("The 'rounding' dictionary must contain the keys {}.".format(variable_names))
        if not all([isinstance(v, (int, type(None))) for v in rounding.values()]):
            raise ValueError("The rounding decimals specified in the 'rounding' dictionary must be integers (or None).")
        if any([v < 0 for v in rounding.values() if v is not None]):
            raise ValueError("The rounding decimals specified in the 'rounding' dictionary must be positive integers (or None).") 
    ##------------------------------------------------------------------------.
    # If a unique compressor, create a dictionary with the same compressor for all variables
    if isinstance(rounding, int):
        if rounding < 0: 
            raise ValueError("'rounding' decimal value must be larger than 0.")
    ##------------------------------------------------------------------------.    
    return rounding

#-----------------------------------------------------------------------------.
#######################
#### Writing wrapper ##
#######################
def write_zarr(zarr_fpath, ds, 
               chunks="auto", default_chunks = None, 
               compressor="auto", default_compressor = None, 
               rounding = None, 
               consolidated=True, 
               append=False, append_dim = None, 
               show_progress=True):
    """Write Xarray Dataset to zarr with custom chunks and compressor per Dataset variable."""
    # Good to know: chunks=None: keeps current chunks, chunks='auto' rely on xarray defaults
    # append=True: if zarr_fpath do not exists, set to False (for the first write)
    ##-------------------------------------------------------------------------.
    ### Check zarr_fpath and append options
    # - Check fpath
    if not zarr_fpath.endswith(".zarr"):
        zarr_fpath = zarr_fpath + ".zarr"
    # - Check append options
    ZARR_EXIST = os.path.exists(zarr_fpath)
    if not isinstance(append, bool):
        raise TypeError("'append' must be either True or False'.")
    # If append = False and a Zarr store already exist --> Raise Error 
    if not append and ZARR_EXIST:
        raise ValueError(zarr_fpath + " already exists!")
    # If the Zarr store do not exist yet but append = True, append is turned to False 
    # --> Useful when calling this function to write data subset by subset
    if append and not ZARR_EXIST:
        append = False
    if append:
        if not isinstance(append_dim, str):
            raise TypeError("Please specify the 'append_dim' (as a string).")
    else:
        append_dim = None

    ##------------------------------------------------------------------------.
    ### - Define file chunking  
    chunks = check_chunks(ds, chunks = chunks, 
                          default_chunks = default_chunks)
    
    ##------------------------------------------------------------------------.
    # - Define compressor and filters
    compressor = check_compressor(compressor = compressor,  
                                  default_compressor = default_compressor,
                                  variable_names = list(ds.data_vars.keys()))
    
    ##------------------------------------------------------------------------.
    # - Define rounding option 
    rounding = check_rounding(rounding = rounding,
                              variable_names = list(ds.data_vars.keys()))
    
    ##------------------------------------------------------------------------.
    # - Rounding (if required)
    if rounding is not None: 
        if isinstance(rounding, int):
            ds = ds.round(decimals=rounding)   
        elif isinstance(rounding, dict):
            for var, decimal in rounding.items():
                if decimal is not None: 
                    ds[var] = ds[var].round(decimal)                
        else: 
            raise NotImplementedError("'rounding' should be int, dict or None.")
    ##------------------------------------------------------------------------.        
    # - Remove previous encoding filters
    # - https://github.com/pydata/xarray/issues/3476
    # for var in ds.data_vars.keys(): 
    #     ds[var].encoding['filters'] = None
    for dim in list(ds.dims.keys()): 
        ds[dim].encoding['filters'] = None # Without this, bug when coords are str objects
    
    ##------------------------------------------------------------------------.
    # - Add chunk encoding the dataset
    for var, chunk in chunks.items():
        if chunk is not None: 
            ds[var] = ds[var].chunk(chunk)  
            
    ##------------------------------------------------------------------------.
    # - Add compressor encoding to each DataArray
    for var, comp in compressor.items(): 
        ds[var].encoding['compressor'] = comp
        
    ##------------------------------------------------------------------------.
    ### - Write zarr files
    compute = not show_progress
    # - Write data to new zarr store 
    if not append: 
        # - Define zarr store  
        zarr_store = zarr.DirectoryStore(zarr_fpath)  
        r = ds.to_zarr(store=zarr_store, 
                       mode='w', # overwrite if exists already
                       synchronizer=None, group=None, 
                       consolidated = consolidated,
                       compute=compute) 
        if show_progress:
            with ProgressBar():
                r.compute()    
    # - Append data to existing zarr store 
    # ---> !!! Attention: It do not check if data are repeated !!! 
    else: 
        r = ds.to_zarr(store = zarr_fpath,
                       mode="a", 
                       append_dim = append_dim,
                       synchronizer = None, group = None, 
                       consolidated = consolidated, 
                       compute=compute) 
        if show_progress:
            with ProgressBar():
                r.compute()   

#-----------------------------------------------------------------------------.
#########################
#### Rechunker wrapper ##
#########################
def rechunk_Dataset(ds, chunks, target_store, temp_store, max_mem = '1GB', force=False):
    """
    Rechunk on disk a xarray Dataset read lazily from a zarr store.

    Parameters
    ----------
    ds : xarray.Dataset
        A Dataset opened with open_zarr().
    chunks : dict
        Custom chunks of the new Dataset.
        If not specified for each Dataset variable, implicitly assumed.
    target_store : str
        Filepath of the zarr store where to save the new Dataset.
    temp_store : str
        Filepath of a zarr store where to save temporary data.
        This store is removed at the end of the rechunking operation. 
    max_mem : str, optional
        The amount of memory (in bytes) that workers are allowed to use.
        The default is '1GB'.

    Returns
    -------
    None.

    """
    # TODO 
    # - Add compressors options 
    # compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.BITSHUFFLE)
    # options = dict(compressor=compressor)
    # rechunk(..., target_options=options)
    ##------------------------------------------------------------------------.
    # Check target_store do not exist already
    if os.path.exists(target_store):
        if force: 
            shutil.rmtree(target_store)
        else:
            raise ValueError("A zarr store already exists at {}. If you want to overwrite, specify force=True".format(target_store))
    
    ##------------------------------------------------------------------------.
    # Remove temp_store if still exists 
    if os.path.exists(temp_store):
        shutil.rmtree(temp_store)

    ##------------------------------------------------------------------------.
    # Check chunks 
    target_chunks = check_chunks(ds=ds, chunks=chunks, default_chunks=None) 
    ##------------------------------------------------------------------------.
    # Change chunk value '-1' to length of the dimension 
    # - rechunk and zarr do not currently support -1 specification used by dask and xarray 
    dict_dims = dict(ds.dims)
    for var in target_chunks.keys():
        if target_chunks[var] is not None: 
            for k, v in target_chunks[var].items():
                if v == -1: 
                    target_chunks[var][k] = dict_dims[k]   
                    
    ##------------------------------------------------------------------------.
    # Plan rechunking                
    r = rechunk(ds, 
                target_chunks=target_chunks, 
                max_mem=max_mem,
                target_store=target_store, temp_store=temp_store)
    
    ##------------------------------------------------------------------------.
    # Execute rechunking
    with ProgressBar():
        r.execute()
        
    ##------------------------------------------------------------------------.    
    # Remove temporary store 
    shutil.rmtree(temp_store)
    ##------------------------------------------------------------------------.

#-----------------------------------------------------------------------------.
###############################################
#### Utils for zarr io profiling and optim ####
###############################################
#### Storage 
def _get_zarr_array_info_dict(arr):
    return {k:v for k,v in arr.info_items()}

def _get_zarr_array_storage_ratio(arr):
    # arr.info.obj.nbytes/arr.info.obj.nbytes_stored
    return float(_get_zarr_array_info_dict(arr)['Storage ratio'])

def _get_zarr_nbytes_stored(arr): 
    return round(arr.info.obj.nbytes_stored/1024/1024,3)

def _get_zarr_nbytes(arr): 
    return round(arr.info.obj.nbytes/1024/1024,3)

def get_nbytes_stored_zarr_variables(fpath):
    """Return nbytes stored each variable."""
    ds = xr.open_zarr(fpath)
    source_group = zarr.open(fpath)
    variables = list(ds.data_vars.keys())
    nbytes_stored = {}
    for variable in variables: 
        nbytes_stored[variable] = _get_zarr_nbytes_stored(source_group[variable])
    return nbytes_stored

def get_nbytes_stored_zarr_coordinates(fpath):
    """Return storage ratio for each variable."""
    ds = xr.open_zarr(fpath)
    source_group = zarr.open(fpath)
    coords = list(ds.coords.keys())
    nbytes_stored = {}
    for coord in coords: 
        nbytes_stored[coord] = _get_zarr_nbytes_stored(source_group[coord])
    return nbytes_stored

def get_storage_ratio_zarr(fpath):
    """Return storage ratio for the entire store."""
    stored_nbytes_coords = get_nbytes_stored_zarr_coordinates(fpath) 
    stored_nbytes_variables = get_nbytes_stored_zarr_variables(fpath)
    stored_nbytes = sum(stored_nbytes_coords.values()) + sum(stored_nbytes_variables.values())
    ds = xr.open_zarr(fpath)
    nbytes = ds.nbytes/1024/1024
    storage_ratio = nbytes / stored_nbytes
    storage_ratio = round(storage_ratio,3)
    return storage_ratio

##----------------------------------------------------------------------------. 
#### Memory size 
def get_memory_size_dataset(ds): 
    """Return size in MB of variables and coordinates."""
    size_dict = {k: ds[k].nbytes/1024/1024 for k in list(ds.keys()) + list(ds.coords.keys())}
    return size_dict

def get_memory_size_dataset_variables(ds): 
    """Return the memory size in MB of the Dataset variables."""
    size_dict = {k: ds[k].nbytes/1024/1024 for k in list(ds.keys())}
    return size_dict

def get_memory_size_dataset_coordinates(ds): 
    """Return the memory size in MB of the Dataset coordinates."""
    size_dict = {k: ds[k].nbytes/1024/1024 for k in list(ds.coords.keys())}
    return size_dict

def get_memory_size_zarr(fpath, isel_dict={}): 
    """Return size in MB of variables and coordinates."""
    ds = xr.open_zarr(fpath)
    ds = ds.isel(isel_dict)
    return get_memory_size_dataset(ds)

def get_memory_size_zarr_variables(fpath, isel_dict={}):
    """Return the memory size in MB of the Zarr variables."""
    ds = xr.open_zarr(fpath)
    ds = ds.isel(isel_dict)
    return get_memory_size_dataset_variables(ds)

def get_memory_size_zarr_coordinates(fpath, isel_dict={}):
    """Return the memory size in MB of the Zarr coordinates."""
    ds = xr.open_zarr(fpath)
    ds = ds.isel(isel_dict)
    return get_memory_size_dataset_coordinates(ds)

def get_memory_size_chunk(x):
    """Return the size in MB of a single chunk.
    
    If x is an xr.Dataset, it returns a dictionary with the chunk size of each variable. 
    """
    if isinstance(x, xr.Dataset):
        size_dict = {}
        for var in list(x.data_vars.keys()):
            size_dict[var] = get_memory_size_chunk(x[var])
        return size_dict
    if isinstance(x, xr.DataArray):
        # If chunked: return the size of the chunk 
        if x.chunks is not None: 
            isel_dict = {dim: slice(0, chunks[0]) for dim, chunks in zip(x.dims, x.chunks)}
            x = x.isel(isel_dict)
            return x.nbytes/1024/1024
        # If not chunked, return the size of the entire array
        else: 
            return x.nbytes/1024/1024
    elif isinstance(x, dask.array.core.Array):
        # slice_list = [slice(None, chunk[0]) for chunk in x.chunks]
        # x[*slice_list]
        raise NotImplementedError("Dask arrays")
    elif isinstance(x, np.ndarray):
        return x.nbytes/1024/1024
    else: 
        raise NotImplementedError("What array you provided?")  

#-----------------------------------------------------------------------------.
#### IO Timing  
def get_reading_time(fpath, isel_dict={}, n_repetitions=5):
    """Return the reading time of a Dataset (subset)."""
    def _load(fpath, isel_dict):
        ds = xr.open_zarr(fpath)
        ds = ds.isel(isel_dict)
        ds.load()
        return None
    times = []
    for i in range(1, n_repetitions):
        t_i = time.time() 
        _load(fpath, isel_dict)
        times.append(time.time() - t_i)
    return times

def get_reading_throughput(fpath, isel_dict={}, n_repetitions=10):
    """Return the reading throughput (MB/s) of a Dataset (subset)."""
    times = get_reading_time(fpath=fpath, isel_dict=isel_dict, n_repetitions=n_repetitions)
    size_dict = get_memory_size_zarr(fpath, isel_dict=isel_dict)
    throughput = sum(size_dict.values())/np.array(times)
    return throughput.tolist()

def get_writing_time(ds, fpath, 
                     chunks = None,     # Use current chunking 
                     compressor = None,
                     consolidated = True, 
                     n_repetitions=5, remove_last=True):
    """Return the writing time of a Dataset."""
    times = []
    for i in range(1, n_repetitions):
        t_i = time.time() 
        write_zarr(zarr_fpath = fpath, 
                   ds = ds,
                   chunks = chunks,  
                   compressor = compressor, 
                   consolidated = consolidated, 
                   show_progress = False)  
        times.append(time.time() - t_i)
        if i < n_repetitions - 1:
            # Remove store 
            shutil.rmtree(fpath)
    if remove_last:
        shutil.rmtree(fpath)
    return times

def get_writing_throughput(ds, fpath, 
                           chunks = None,     # Use current chunking 
                           compressor = None,
                           consolidated = True, 
                           n_repetitions=5, remove_last=True):
    """Return the writing throughput (MB/s) of a Dataset."""
    times = get_writing_time(ds, fpath, 
                             chunks = chunks,     
                             compressor = compressor,
                             consolidated = consolidated, 
                             n_repetitions=n_repetitions, remove_last=remove_last)
    size_dict = get_memory_size_dataset(ds)
    throughput = sum(size_dict.values())/np.array(times)
    return throughput.tolist()        

def profile_zarr_io(ds, fpath, 
                    chunks = None,        # Use current chunking 
                    compressor = None,
                    isel_dict = {}, 
                    consolidated = True, 
                    n_repetitions=5):
    """Profile reading and writing of a Dataset."""
    io_dict = {}
    io_dict['writing'] = get_writing_time(ds = ds, 
                                          fpath = fpath, 
                                          chunks = chunks,    
                                          compressor = compressor,
                                          consolidated = consolidated, 
                                          n_repetitions=n_repetitions, 
                                          remove_last=False)
    io_dict['reading'] = get_reading_time(fpath = fpath,
                                          isel_dict=isel_dict,
                                          n_repetitions=n_repetitions)
    io_dict['reading_throughput'] = get_reading_throughput(fpath = fpath, 
                                                           isel_dict=isel_dict, 
                                                           n_repetitions=n_repetitions)
    io_dict['compression_ratio'] = get_storage_ratio_zarr(fpath = fpath)
    shutil.rmtree(fpath)
    return io_dict 

#-----------------------------------------------------------------------------. 
#########################################
#### Define default zarr compressors ####
#########################################
def _get_blosc_compressors(clevels=[0,1,3,5,9]):
    cnames = numcodecs.blosc.list_compressors()   
    shuffles = [numcodecs.Blosc.BITSHUFFLE, numcodecs.Blosc.SHUFFLE, numcodecs.Blosc.NOSHUFFLE]
    possible_args = list(itertools.product(shuffles, clevels, cnames))
    compressors = {}
    for shuffle, clevel, cname in possible_args:
        k_name = cname + "_cl" + str(clevel) + "_s" + str(shuffle)
        compressors[k_name] = numcodecs.blosc.Blosc(cname=cname, clevel=clevel, shuffle=shuffle)
    return compressors

def _get_lmza_compressors(clevels=[0,1,3,5,9]): 
    # - preset: compression level between 0 and 9 
    # - dist: distance between bytes to be subtracted (default 1)
    # Cannot specify filters except with FORMAT_RAW
    import lzma
    delta_dist = [None, 1,2,4]
    possible_args = list(itertools.product(clevels, delta_dist))
    compressors = {}
    for clevel, delta_dist in possible_args:
        if delta_dist is not None:
            lzma_filters = [dict(id=lzma.FILTER_DELTA, dist=delta_dist),
                           dict(id=lzma.FILTER_LZMA2, preset=clevel)]
            k_name = "LZMA" + "_cl" + str(clevel) + "_delta" + str(delta_dist)
            compressors[k_name] = numcodecs.LZMA(preset=None, filters=lzma_filters) 
        else:
            k_name = "LZMA" + "_cl" + str(clevel) + "_nodelta"
            compressors[k_name] = numcodecs.LZMA(preset=clevel, filters=None)
    return compressors 

def _get_zip_compressors(clevels=[0,1,3,5,9]):
    # - BZ2 do not accept clevel = 0
    compressors = {}
    for clevel in clevels:
        k_name = "GZip" + "_cl" + str(clevel)       
        compressors[k_name] = numcodecs.gzip.GZip(level=clevel)
        if clevel > 0:
            k_name = "BZ2" + "_cl" + str(clevel)           
            compressors[k_name] = numcodecs.bz2.BZ2(level=clevel)
    return compressors
 
def _get_zfpy_compressors():
    # TODO define some options 
    # - Not yet available for Python 3.8.5
    # - precision: A integer number, specifying the compression precision needed
    compressors = {}
    compressors['zfpy'] = numcodecs.zfpy.ZFPY(tolerance=-1, rate=-1, precision=-1)
    return compressors

def _getlossless_compressors(clevels=[0,1,3,5,9]):
    compressors = _get_blosc_compressors(clevels=clevels)
    # compressors.update(_get_lmza_compressors(clevels=clevels))
    compressors.update(_get_zip_compressors(clevels=clevels))
    # compressors.update(_get_zfpy_compressors())
    return compressors  
 
#-----------------------------------------------------------------------------. 






