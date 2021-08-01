#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 16:11:33 2021

@author: ghiggi
"""
import shutil
import dask 
import numpy as np
from rechunker import rechunk
from dask.diagnostics import ProgressBar

##----------------------------------------------------------------------------.
def is_numcodecs(compressor):
    """Check is a numcodec compressor."""
    if type(compressor).__module__.find("numcodecs") == -1:
        return False
    else:
        return True

##----------------------------------------------------------------------------.
def check_compressor(compressor, variable_names, default_compressor = None):
    """Check compressor validity for zarr writing.
    
    compressor = None --> No compression.
    compressor = "auto" --> Use default_compressor if specified. Otherwise no compression is applied.
    compressor = {..} --> If compressor dictionary is specified, check that is valid.
    compressor = numcodecs class --> Create a dictionary for the specified compressor for each variable_name
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
    if not (isinstance(default_compressor, type(None)) or is_numcodecs(default_compressor)):
        raise TypeError("'default_compressor' must be a numcodecs compressor or None.")
    ##------------------------------------------------------------------------.
    # If a string --> Apply default compressor (if specified)
    if isinstance(compressor, str): 
        if compressor == "auto" and default_compressor is not None:
            compressor = default_compressor
        else: 
            raise ValueError("If 'compressor' is specified as string, must be 'auto'.")    
    ##------------------------------------------------------------------------.        
    # If a dictionary, check valid keys and valid compressor
    if isinstance(compressor, dict):
        if not np.all(np.isin(list(compressor.keys()), variable_names)):
            raise ValueError("The 'compressor' dictionary must contain the keys {}".format(variable_names))
        if not all([is_numcodecs(cmp) or isinstance(cmp, type(None)) for cmp in compressor.values()]):
            raise ValueError("The compressors specified in the 'compressor' dictionary must be numcodecs (or None).")
    ##------------------------------------------------------------------------.
    # If a unique compressor, create a dictionary with the same compressor for all variables
    if is_numcodecs(compressor):
        compressor = {var: compressor for var in variable_names}
    ##------------------------------------------------------------------------.    
    return compressor

##----------------------------------------------------------------------------.
def check_chunks(chunks, variable_names, default_chunks = None):
    """Check chunks validity.
    
    chunks = None --> No chunking --> Contiguous.
    chunks = "auto" --> Use default_chunks is specified, otherwise default xarray chunks .
    chunks = {..}  
    # - If default_chunks is specified, check that keys are the same.
    # - Option1: A dictionary of chunks definitions for each variable_names 
    # - Option1: A single chunk definition to be applied to all variables 
    """
    # Check variable_names and chunks types
    if not isinstance(chunks, (str, dict, type(None))):
        raise TypeError("'chunks' must be a dictionary, 'auto' or None.")
    if isinstance(variable_names, str):
        variable_names = [variable_names]
    if not all([isinstance(s, str) for s in variable_names]):
        raise TypeError("Specify all variable names as string within the 'variable_names' list.")
    ##------------------------------------------------------------------------.
    # If a string --> Auto --> Apply default_chunks (if specified)  
    if isinstance(chunks, str): 
        if chunks == "auto" and default_chunks is not None:
            chunks = default_chunks
        elif chunks == "auto" and default_chunks is None:
            chunks = None
        else: 
            raise ValueError("If 'chunks' is specified as string, must be 'auto'.")
    ##------------------------------------------------------------------------.
    # If a dictionary, check valid keys and values  
    if isinstance(chunks, dict):
        # If a chunk specific for each variable is specified (keys are variable_names)
        if np.all(np.isin(list(chunks.keys()), variable_names)):
            if not np.all(np.isin(variable_names, list(chunks.keys()))):
                raise ValueError("If you specify specific chunks for each variable, please specify it for all variables.")
            # - Check that the chunk for each dimension is specified
            for key in chunks.keys():
                if default_chunks is not None:
                    if not np.all(np.isin(list(chunks[key].keys()), list(default_chunks.keys()))):
                        raise ValueError("The 'chunks' dictionary of {} must contain the keys {}".format(key, list(default_chunks.keys())))
                # - Check that the chunk value are integers
                if not all([isinstance(v, int) for v in chunks[key].values()]):
                    raise ValueError("The 'chunks' values of the {} dictionary must be integers.".format(key))
        # If a common chunk is specified for all variable_names (chunks keys are not variable_names)
        elif np.all(np.isin(list(chunks.keys()), variable_names, invert=True)):
            # - Check that the chunk for each dimension is specified
            if default_chunks is not None:
                if not np.all(np.isin(list(chunks.keys()), list(default_chunks.keys()))):
                    raise ValueError("The 'chunks' dictionary must contain the keys {}".format(list(default_chunks.keys())))
            # - Check that the chunk value are integers
            if not all([isinstance(v, int) for v in chunks.values()]):
                raise ValueError("The 'chunks' values of the dictionary must be integers.")
            # - Specify chunks for each variable
            chunks = {var: chunks for var in variable_names}
        else: 
            raise ValueError("This chunks option has not been implemented.")
    ##------------------------------------------------------------------------.    
    return chunks 

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
def rechunk_Dataset(ds, chunks, target_store, temp_store, max_mem = '1GB'):
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
    ##------------------------------------------------------------------------.
    # Retrieve variables
    variable_names = list(ds.data_vars.keys())
    # Check chunks 
    target_chunks = check_chunks(chunks=chunks, default_chunks=None, variable_names=variable_names) 
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
