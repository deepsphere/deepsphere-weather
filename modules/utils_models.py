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

def check_resolution(resolution):
    """Check valid resolution."""
    if not isinstance(resolution, Iterable):
        resolution = [resolution]
    resolution = np.array(resolution) 
    return resolution

def get_pygsp_graph_fun(sampling):
    """Return the pygsp function to generate the spherical graph."""
    return get_pygsp_graph_dict()[sampling]
    
def get_pygsp_graph_params(sampling):
    """Return the graph parameters to generate the spherical graph."""
    ALL_GRAPH_PARAMS = {'healpix': {'nest': True},
                        'equiangular': {'poles': 0},
                        'icosahedral': {},
                        'cubed': {},
                        'gauss': {'nlon': 'ecmwf-octahedral'}}
    return ALL_GRAPH_PARAMS[sampling]

def get_pygsp_graph(sampling, resolution,  knn=20):
    """Return the pygsp graph for a specific spherical sampling."""
    check_sampling(sampling)
    graph_initializer = get_pygsp_graph_fun(sampling)
    params = get_pygsp_graph_params(sampling)
    if sampling.lower() == "equiangular":
        # TODO: update pgsp.SphereEquiangular to accept k ... 
        pygsp_graph = graph_initializer(resolution[0], resolution[1], **params)
    else:
        params['k'] = knn
        pygsp_graph = graph_initializer(resolution, **params)
    return pygsp_graph
