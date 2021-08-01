#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 17:51:40 2021

@author: ghiggi
"""
import pygsp
import numpy as np
from typing import List
from abc import ABC, abstractmethod
from modules.layers import prepare_torch_laplacian
from modules.utils_models import check_sampling
# from modules.utils_models import get_pygsp_graph
from modules.utils_models import get_pygsp_graph_fun
from modules.utils_models import get_pygsp_graph_params
from modules.utils_models import check_conv_type
from modules.utils_torch import get_torch_dtype
import igl # conda install igl

def triangulate(graph):
    sv = SphericalVoronoi(graph.coords)
    assert sv.points.shape[0] == graph.n_vertices
    return sv.points, sv._simplices

def compute_cotan_laplacian(graph, return_mass=False):
    v, f = triangulate(graph)
    L = -igl.cotmatrix(v, f)
    assert len((L - L.T).data) == 0
    M = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
    # M = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_BARYCENTRIC)
    if return_mass:
        # Eliminate zeros for speed (appears for equiangular).
        L.eliminate_zeros()  
        return L, M
    else:
        Minv = sparse.diags(1 / M.diagonal())
        return Minv @ L

# TODO 
# - Add RemappingNet (just pooling)
# - Add DownscalingNet
# - Add ResNet (without pooling)

##----------------------------------------------------------------------------.
class UNet(ABC):
    """Define general UNet class."""
    
    @abstractmethod
    def encode(self, *args, **kwargs):
        """Encode an input into a lower dimensional space."""
        pass

    @abstractmethod
    def decode(self, *args, **kwargs):
        """Decode low dimensional data into a high dimensional space."""
        pass

    def forward(self, x):
        """Implement a forward pass."""
        x_encoded = self.encode(x)
        output = self.decode(*x_encoded)
        return output
  
    @staticmethod
    def build_graph(resolutions: List[List[int]], 
                    sampling: str = 'healpix', 
                    knn: int = 10) -> List["pygsp.graphs"]:
        """Build the graph for each specified resolution."""
        # Create a list of pygsp graph 
        # pygsp_graphs_list = [get_pygsp_graph(sampling=sampling, resolution=*res, knn=knn) for res in resolutions]

        # Check sampling 
        check_sampling(sampling)
        # Retrieve pygsp function to create the spherical graph
        graph_initializer = get_pygsp_graph_fun(sampling)
        # Retrieve parameters to customize the spherical graph
        params = get_pygsp_graph_params(sampling)
        params['lap_type'] = 'normalized'
        params['k'] = knn
        # Create a list of pygsp graph 
        pygsp_graphs_list = [graph_initializer(*res, **params) for res in resolutions]
 
        return pygsp_graphs_list
    
    @staticmethod
    def get_laplacian_kernels(graphs: List["pygsp.graphs"],
                              torch_dtype,
                              gtype='knn'):
        """Compute the laplacian for each specified graph."""
        # TODO 
        # - Add gtype in config file
        assert gtype in ['knn', 'mesh']
        laplacians_list = [graph.L if gtype == 'knn' else compute_cotan_laplacian(graph, return_mass=False) for graph in graphs]
        laplacians_list = [prepare_torch_laplacian(L, torch_dtype=torch_dtype) for L in laplacians_list]
        return laplacians_list
    
    def init_graph_and_laplacians(self, 
                                  conv_type,
                                  resolution,
                                  sampling, 
                                  knn, 
                                  kernel_size_pooling,
                                  UNet_depth,
                                  gtype="knn",
                                  numeric_precision='float32'):
        """Initialize graph and laplacian.
        
        Parameters
        ----------
        conv_type : TYPE
            Convolution type. Either 'graph' or 'image'.
        resolution : int
            Resolution of the input tensor.
        sampling : str
            Name of the spherical sampling.
        knn : int
            DESCRIPTION.
        kernel_size_pooling : int
            DESCRIPTION.
        UNet_depth : int
            Depth of the UNet.
        numeric_precision: str
            Numeric precision for model training.
            The default is 'float32'.
        """
        # Check inputs 
        sampling = check_sampling(sampling)
        conv_type = check_conv_type(conv_type, sampling)
        torch_dtype = get_torch_dtype(numeric_precision)
        ##--------------------------------------------------------------------.
        # Define resolutions 
        coarsening = int(np.sqrt(kernel_size_pooling))
        resolutions = [resolution]
        for i in range(1, UNet_depth):
            resolutions.append(resolutions[i-1] // coarsening)
            
        # Initialize graph and laplacians 
        if conv_type == 'graph':
            self.graphs = UNet.build_graph(resolutions=resolutions,
                                           sampling=sampling,
                                           knn=knn)
            self.laplacians = UNet.get_laplacian_kernels(graphs=self.graphs, 
                                                         torch_dtype=torch_dtype,
                                                         gtype=gtype)
        # Option for equiangular sampling  
        elif conv_type == 'image':
            self.graphs = UNet.build_graph(resolutions=resolutions,
                                           sampling=sampling,
                                           knn=knn)
            self.laplacians = [None] * UNet_depth
