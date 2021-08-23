#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 22:05:51 2021

@author: ghiggi
"""
import pygsp
import numpy as np
from collections.abc import Iterable

def get_pygsp_graph_dict():
    """Return a dictionary matching sampling name with pygsp graph function."""
    ALL_GRAPH = {'healpix': pygsp.graphs.SphereHealpix,
                 'equiangular': pygsp.graphs.SphereEquiangular,
                 'icosahedral': pygsp.graphs.SphereIcosahedral,
                 'cubed': pygsp.graphs.SphereCubed,
                 'gauss': pygsp.graphs.SphereGaussLegendre
                 }
    return ALL_GRAPH

def get_valid_pygsp_graph():
    """Return the spherical samplings implemented in pygsp."""
    return list(get_pygsp_graph_dict().keys())

def check_sampling(sampling):
    """Check valid sampling name."""
    if not isinstance(sampling, str):
        raise TypeError("'sampling' must be a string.")
    sampling = sampling.lower() 
    valid_samplings = get_valid_pygsp_graph()
    if sampling not in valid_samplings:
        raise ValueError("'sampling' must be one of {}.".format(valid_samplings))
    return sampling

def check_conv_type(conv_type, sampling):
    """Check valid convolution type."""
    if not isinstance(conv_type, str):
        raise TypeError("'conv_type' must be a string.")
    if not isinstance(sampling, str):
        raise TypeError("'sampling' must be a string.")
    conv_type = conv_type.lower()
    if conv_type not in ['graph','image']:
        raise ValueError("'conv_type' must be either 'graph' or 'image'.")
    if conv_type == 'image':
        sampling = sampling.lower()
        if sampling != 'equiangular':
            raise ValueError("conv_type='image' is available only if sampling='equiangular'.")
    return conv_type

def check_pool_method(pool_method):
    """Check valid pooling method."""
    # ('interp', 'maxval', 'maxarea', 'learn')  graph 
    pool_method = pool_method.lower()
    return(pool_method)

def check_skip_connection(skip_connection):
    """Check skip connection type."""
    if not isinstance(skip_connection, (str, type(None))):
        raise TypeError("'skip_connection' must be a string.")
    if skip_connection is None: 
        skip_connection = 'none'
    valid_options = ('none','stack','sum','avg')
    if skip_connection not in valid_options:
        raise ValueError("'skip_connection' must be one of {}".format(valid_options))
    return skip_connection
        
def get_pygsp_graph_fun(sampling):
    """Return the pygsp function to generate the spherical graph."""
    check_sampling(sampling)
    return get_pygsp_graph_dict()[sampling]

def get_pygsp_graph(sampling, sampling_kwargs,  knn=20):
    """Return the pygsp graph for a specific spherical sampling."""
    check_sampling(sampling)
    graph_initializer = get_pygsp_graph_fun(sampling)
    sampling_kwargs['k'] = knn
    pygsp_graph = graph_initializer(**sampling_kwargs)
    return pygsp_graph

def pygsp_graph_coarsening(sampling, sampling_kwargs, coarsening):
    new_kwargs = sampling_kwargs.copy()
    if sampling == "equiangular":
         new_kwargs['nlat'] = new_kwargs['nlat'] // coarsening 
         new_kwargs['nlon'] = new_kwargs['nlon'] // coarsening
    elif sampling in ["icosahedral", "cubed", "healpix"]:
        new_kwargs['subdivisions'] = new_kwargs['subdivisions'] // coarsening 
    elif sampling == "gauss":
        new_kwargs['nlat'] = new_kwargs['nlat'] // coarsening 
    else:
        NotImplementedError()
    return new_kwargs  
    
