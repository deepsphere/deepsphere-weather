#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 17:51:40 2021

@author: ghiggi
"""
import pygsp
import numpy as np
import torch
from typing import List
from abc import ABC, abstractmethod
from modules.layers import compute_cotan_laplacian
from modules.layers import prepare_torch_laplacian
from modules.utils_models import check_sampling
# from modules.utils_models import get_pygsp_graph
from modules.utils_models import get_pygsp_graph_fun
from modules.utils_models import get_pygsp_graph_params
from modules.utils_models import check_conv_type
 
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
        # Check sampling 
        check_sampling(sampling)
        # Retrieve pygsp function to create the spherical graph
        # .i.e. pygsp.graphs.SphereHealpix
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
                              gtype='knn'):
        """Compute the laplacian for each specified graph."""
        # TODO 
        # - Add gtype in config file
        assert gtype in ['knn', 'mesh']
        torch_dtype = torch.get_default_dtype()  
        laplacians_list = [graph.L if gtype == 'knn' else compute_cotan_laplacian(graph, return_mass=False) for graph in graphs]
        laplacians_list = [prepare_torch_laplacian(L, torch_dtype=torch_dtype) for L in laplacians_list]
        return laplacians_list
    
    def init_graph_and_laplacians(self, 
                                  sampling, 
                                  resolution,
                                  conv_type,
                                  knn, 
                                  kernel_size_pooling,
                                  UNet_depth,
                                  gtype="knn"):
        """Initialize graph and laplacian.
        
        Parameters
        ----------
        sampling : str
            Name of the spherical sampling.
        resolution : int
            Resolution of the spherical sampling.
        conv_type : str, optional
            Convolution type. Either 'graph' or 'image'.
            The default is 'graph'.
            conv_type='image' can be used only when sampling='equiangular'.
        knn : int
            DESCRIPTION.
        gtype : str
            DESCRIPTION
        kernel_size_pooling
            The size of the window to max/avg over.
            kernel_size_pooling = 4 means halving the resolution when pooling
            The default is 4.
        UNet_depth : int
            Depth of the UNet.
        """
        # Check inputs 
        sampling = check_sampling(sampling)
        conv_type = check_conv_type(conv_type, sampling)
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
                                                         gtype=gtype)
        # Option for equiangular sampling  
        elif conv_type == 'image':
            self.graphs = UNet.build_graph(resolutions=resolutions,
                                           sampling=sampling,
                                           knn=knn)
            self.laplacians = [None] * UNet_depth
