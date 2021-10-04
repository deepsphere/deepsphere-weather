#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 17:51:40 2021

@author: ghiggi
"""
import pygsp
from typing import List, Dict
from abc import ABC, abstractmethod
from modules.layers import compute_cotan_laplacian
from modules.layers import prepare_torch_laplacian
from modules.utils_models import get_pygsp_graph_fun

##----------------------------------------------------------------------------.
class DeepSphere(ABC):
    """Define general DeepSphere model class."""
    
    @abstractmethod
    def forward(self, x):
        """Implement a forward pass."""
        pass    
    
    @staticmethod
    def build_pygsp_graphs(sampling_list: List[str],
                           sampling_kwargs_list: List[Dict]) -> List["pygsp.graphs"]:
        """Build the graphs for the specified samplings."""
        # Checks 
        if not isinstance(sampling_list, list):
            raise TypeError("sampling_list must be a list specifying the sampling of each graph.")
        if not isinstance(sampling_list, list):
            raise TypeError("sampling_kwargs_list must be a list specifying the sampling_kwargs_list of each sampling.")
        if len(sampling_list) != len(sampling_kwargs_list): 
            raise ValueError("sampling_list must have same length of sampling_kwargs_list.")
        # Create a list of pygsp graphs
        pygsp_graphs_list = [get_pygsp_graph_fun(sampl)(**sampl_kwargs, lap_type="normalized") for sampl, sampl_kwargs in zip(sampling_list, sampling_kwargs_list)]
        return pygsp_graphs_list
    
    @staticmethod
    def get_laplacian_kernels(graphs: List["pygsp.graphs"],
                              graph_type: str = 'knn'):
        """Compute the laplacian for each specified graph."""
        assert graph_type in ['knn', 'voronoi']
        laplacians_list = [graph.L if graph_type == 'knn' else compute_cotan_laplacian(graph, return_mass=False) for graph in graphs]
        laplacians_list = [prepare_torch_laplacian(L) for L in laplacians_list]
        return laplacians_list
    
    def init_graph_and_laplacians(self, 
                                  sampling_list: List[str], 
                                  sampling_kwargs_list: List[Dict],
                                  graph_type="knn",
                                  conv_type="graph"):
        """Initialize graph and laplacian.
        
        Parameters
        ----------
        sampling_list : list[str]
            List  of spherical sampling.
        sampling_kwargs_list : list[dict]
            List of kwargs for each spherical sampling in sampling_list   
        conv_type : str, optional
            Convolution type. Either 'graph' or 'image'.
            The default is 'graph'.
            conv_type='image' can be used only when sampling='equiangular'.
        graph_type : str
            DESCRIPTION
        """
        ##--------------------------------------------------------------------.
        # Initialize graph and laplacians 
        if conv_type == 'graph':
            self.graphs = DeepSphere.build_pygsp_graphs(sampling_list = sampling_list,
                                                        sampling_kwargs_list = sampling_kwargs_list)
                                                   
            self.laplacians = DeepSphere.get_laplacian_kernels(graphs = self.graphs, 
                                                               graph_type = graph_type)
        # Option for equiangular sampling  
        elif conv_type == 'image':
            self.graphs = DeepSphere.build_pygsp_graphs(sampling_list = sampling_list,
                                                        sampling_kwargs_list = sampling_kwargs_list)
            self.laplacians = [None] * len(sampling_list)

##----------------------------------------------------------------------------.  
class UNet(DeepSphere):
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
  
##----------------------------------------------------------------------------.
class ConvNet(DeepSphere):
    """Define general ResNet class."""
    
    @abstractmethod
    def forward(self, x):
        """Implement a forward pass."""
        pass    
    
##----------------------------------------------------------------------------.  
class DownscalingNet(DeepSphere):
    """Define general DownscalingNet class."""
    
    @abstractmethod
    def decode(self, *args, **kwargs):
        """Decode low dimensional data into a high dimensional space."""
        pass

    def forward(self, x):
        """Implement a forward pass."""
        output = self.decode(x)
        return output   