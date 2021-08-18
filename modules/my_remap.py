#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 17:07:10 2021

@author: ghiggi
"""
import os 
import glob 
import re 
import numpy as np 

from modules.remap import check_normalization
from modules.remap import cdo_remapping

########################################
### Ad-hoc funtion for WeatherBench #### 
########################################
### Datasets / Samplings / Variables ####
def get_available_datasets():
    """Available datasets."""
    datasets = ['ERA5_HRES',
                'ERA5_EDA',
                'IFS_HRES',
                'IFS_ENS',
                'IFS_ENS_Extended',
                'SEAS5'] 
    return datasets

def get_native_grids_dict(): 
    """Native grid dictionary of datasets."""
    d = {'ERA5_HRES': 'N320',
         'ERA5_EDA': 'N160', 
         'IFS_HRES': 'O1280',
         'IFS_ENS': 'O640',
         'IFS_ENS_Extended': 'O320',
         'SEAS5': 'O320'}
    return d

def get_native_grid(dataset): 
    """Native grid of a dataset."""
    return get_native_grids_dict()[dataset]
             
def get_available_dynamic_variables():
    """Available dynamic variables."""
    # https://github.com/pangeo-data/WeatherBench 
    variables = ['geopotential',
                 'temperature',
                 'specific_humidity',
                 'toa_incident_solar_radiation']
    return variables        
         
def get_available_static_variables():   
    """Available static variables."""
    variables = ['topography', 'land_sea_mask', 'soil_type']
    return variables 

def get_available_variables():
    """Available variables."""
    variables = get_available_dynamic_variables()
    variables.extend(get_available_static_variables())
    return variables

def get_variable_interp_method_dict(): 
    """Interpolation method dictionary for each variable."""
    d = {'dynamic_variables': 'conservative',
         'topography': 'conservative',
         'land_sea_mask': 'conservative',
         'soil_type': 'largest_area_fraction'}
    return d

def get_variable_interp_method(variable):
    """Return the interpolation method that should be used for a specific variable."""
    return get_variable_interp_method_dict()[variable]   

def get_dir_path(data_dir, dataset, sampling, variable_type, variable=None): 
    """Get directory path."""
    dir_path = os.path.join(data_dir, dataset, sampling, variable_type)  
    # Create a subdirectory for each static variable 
    if (variable_type == "static"):
        dir_path = os.path.join(dir_path, variable)   
    return dir_path 

def get_cdo_grid_fpath(CDO_grids_dir, sampling): 
    """Check if CDO grid description file exists and return its path."""
    fpath = os.path.join(CDO_grids_dir, sampling) 
    if not os.path.exists(fpath):
        raise ValueError("Please create a CDO grid description infile into the CDO grids folder")
    return fpath

def get_cdo_weights_filename(method, input_sampling, output_sampling): 
    """Generate the filename where to save the CDO interpolation weights."""
    # Normalization option
    # Nearest_neighbor option 
    filename = "CDO_" + method + "_weights_IN_" + input_sampling + "_OUT_"+ output_sampling + ".nc"
    return filename 

################
### Checks #####
################
def check_dataset(dataset): 
    """Check dataset name."""
    if not isinstance(dataset, str):
        raise TypeError("Provide 'dataset' name as a string")
    if (dataset not in get_available_datasets()): 
        raise ValueError("Provide valid dataset. get_available_datasets()")

def check_sampling(CDO_grids_dir, sampling): 
    """Check sampling name."""
    if not isinstance(sampling, str):
        raise TypeError("Provide 'sampling' name as a string")
    files = os.listdir(CDO_grids_dir)
    if (sampling not in files): 
        raise ValueError("Provide sampling name for which a CDO grid has been defined")

def check_variable(variable):
    """Check variable name."""
    if not isinstance(variable, str):
        raise TypeError("Provide 'variable' name as a string")
    if (variable not in get_available_variables()): 
        raise ValueError("Provide valid variable. get_available_variables()")

def check_variable_type(variable_type):
    """Check variable type name."""
    if not isinstance(variable_type, str):
        raise TypeError("Provide 'variable_type' name as a string")
    if (variable_type not in ['static', 'dynamic']): 
        raise ValueError("Provide either 'static' or 'dynamic'")

def ensure_dir_exists(dir_paths):
    """Create directory if not existing."""
    # Accept str or list of directories.
    if isinstance(dir_paths, str):
        dir_paths = [dir_paths]
    if not isinstance(dir_paths, list):
        raise ValueError('Either provide a string or a list of string')
    dir_paths = np.unique(dir_paths).tolist()   
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    return 

#############################
### Remapping GRIB files ####
#############################
def change_to_nc_extension(fpath): 
    """Replace .grib with .nc ."""
    pre, ext = os.path.splitext(fpath)
    return pre + ".nc"

def get_src_dst_fpaths(src_dir, dst_dir):
    """Return input, output tuple of filepaths for a specific input folder."""
    src_fpaths = sorted(glob.glob(src_dir + "/**/*.grib", recursive=True))
    dst_fpaths = [re.sub(pattern=src_dir, repl=dst_dir, string=fpath) for fpath in src_fpaths]
    # Ensure output dir exists 
    _ = [ensure_dir_exists(os.path.dirname(fpath)) for fpath in dst_fpaths]
    # Change extension from grib to nc 
    dst_fpaths = [change_to_nc_extension(fpath) for fpath in dst_fpaths]
    return (src_fpaths, dst_fpaths)

def remap_grib_files(data_dir, 
                     CDO_grids_dir,
                     CDO_grids_weights_dir, 
                     dataset,
                     sampling,
                     variable_type,
                     precompute_weights = True, 
                     normalization='fracarea',
                     n_threads = 1,
                     compression_level = 1, 
                     force_remapping = False): 
    """
    Functions to remap nc/grib files between unstructured grid with cdo.

    Parameters
    ----------
    data_dir : str
        Base directory where data are stored.
    CDO_grids_dir : str
        Directory where CDO grids description files are stored.
    CDO_grids_weights_dir : str
        Directory where the generated weights for interpolation can be stored.
    dataset : str
        A valid dataset name [to access <dataset>/<sampling>/variable_type].
    variable_type : str
        A valid variable type [to access <dataset>/<sampling>/variable_type].
        Either dynamic or static.
    sampling : str
        A valid variable name [to access <dataset>/<sampling>/variable_type].
    precompute_weights : bool, optional
        Whether to first precompute once the iterpolation weights and then remap. 
        The default is True.
    normalization : str, optional
        Normalization option for conservative remapping. 
        The default is 'fracarea'.
        Options:
        - fracarea uses the sum of the non-masked source cell intersected 
          areas to normalize each target cell field value. 
          Flux is not locally conserved.
        - destarea’ uses the total target cell area to normalize each target
          cell field value. 
          Local flux conservation is ensured, but unreasonable flux values 
          may result [i.e. in small patches].
    compression_level : int, optional
        Compression level of output netCDF4. Default 1. 0 for no compression.
    n_threads : int, optional
        Number of OpenMP threads to use within CDO. The default is 1.

    Returns
    -------
    None.

    """
    ##------------------------------------------------------------------------.
    ## Checks 
    check_dataset(dataset)
    check_sampling(CDO_grids_dir, sampling)
    check_variable_type(variable_type)   
    check_normalization(normalization)
    ##------------------------------------------------------------------------.
    ## Define input and output sampling 
    native_sampling = get_native_grid(dataset=dataset)
    # Retrieve the CDO grid description path of inputs and outputs 
    src_CDO_grid_fpath = get_cdo_grid_fpath(CDO_grids_dir = CDO_grids_dir,
                                            sampling = native_sampling)
    dst_CDO_grid_fpath = get_cdo_grid_fpath(CDO_grids_dir = CDO_grids_dir,
                                            sampling = sampling)
    ##------------------------------------------------------------------------.
    if (variable_type == "static"):
        variables = get_available_static_variables()
    else:  
        variables = ["dynamic_variables"]
    ##------------------------------------------------------------------------.
    for variable in variables:
        print("Remapping", variable, "from", native_sampling, "to", sampling)
        ### Define input and output folders 
        src_dir = get_dir_path(data_dir = data_dir,
                               dataset = dataset, 
                               sampling = native_sampling, 
                               variable_type = variable_type,
                               variable = variable)
        dst_dir = get_dir_path(data_dir = data_dir,
                               dataset = dataset, 
                               sampling = sampling, 
                               variable_type = variable_type,
                               variable = variable)
        ##--------------------------------------------------------------------.
        ### List input filepaths and define output filepaths for a specific folder
        src_fpaths, dst_fpaths = get_src_dst_fpaths(src_dir = src_dir,
                                                    dst_dir = dst_dir)
        ##--------------------------------------------------------------------.
        if (len(src_fpaths) == 0):
            print(variable, "data are not available")
            continue
        ##--------------------------------------------------------------------. 
        ## Remap only data not already remapped 
        if force_remapping is not True:
            idx_not_existing = [not os.path.exists(dst_fpath) for dst_fpath in dst_fpaths]
            src_fpaths = np.array(src_fpaths)[np.array(idx_not_existing)].tolist()
            dst_fpaths = np.array(dst_fpaths)[np.array(idx_not_existing)].tolist() 
        if (len(src_fpaths) == 0):
            print("Data were already remapped. Set force_remapping=True to force remapping.")
            continue    
        ##--------------------------------------------------------------------.
        ### Define interpolation method based on variable_type and variable  
        method = get_variable_interp_method(variable)
        ##--------------------------------------------------------------------. 
        ### Specify filename and path for the interpolation weights  
        cdo_weights_name = get_cdo_weights_filename(method = method,
                                                    input_sampling = native_sampling, 
                                                    output_sampling = sampling)
        weights_fpath = os.path.join(CDO_grids_weights_dir, cdo_weights_name)
        ##--------------------------------------------------------------------. 
        # Remap the data 
        cdo_remapping(method = method,
                      src_CDO_grid_fpath = src_CDO_grid_fpath,
                      dst_CDO_grid_fpath = dst_CDO_grid_fpath, 
                      src_fpaths = src_fpaths,
                      dst_fpaths = dst_fpaths,
                      precompute_weights = precompute_weights,
                      weights_fpath = weights_fpath, 
                      normalization = normalization,
                      compression_level = compression_level,
                      n_threads = n_threads)
        ##--------------------------------------------------------------------.
    ##-----------------------------------------------------------------------.
    return 

##----------------------------------------------------------------------------.
