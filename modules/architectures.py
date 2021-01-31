from typing import List, Union, Dict
from abc import ABC, abstractmethod
from collections.abc import Iterable

import torch
import pygsp
import numpy as np
from torch.nn import functional as F
from torch.nn import BatchNorm1d, Linear

from modules import layers
from modules.layers import *


HEALPIX_POOL = {'max': (PoolMaxHealpix, UnpoolMaxHealpix), 
                'avg': (PoolAvgHealpix, UnpoolAvgHealpix)}

EQUIANGULAR_POOl = {'max': (PoolMaxEquiangular, UnpoolMaxEquiangular), 
                    'avg': (PoolAvgEquiangular, UnpoolAvgEquiangular)}

ALL_POOL = {'healpix': HEALPIX_POOL,
            'equiangular': EQUIANGULAR_POOl}

ALL_CONV = {'image': Conv2dEquiangular,
            'graph': ConvCheb}

ALL_GRAPH = {'healpix': pygsp.graphs.SphereHealpix,
             'equiangular': pygsp.graphs.SphereEquiangular,
             'icosahedral': pygsp.graphs.SphereIcosahedral,
             'cubed': pygsp.graphs.SphereCubed,
             'gauss': pygsp.graphs.SphereGaussLegendre}

ALL_GRAPH_PARAMS = {'healpix': {'nest': True},
                    'equiangular': {'poles': 0},
                    'icosahedral': {},
                    'cubed': {},
                    'gauss': {'nlon': 'ecmwf-octahedral'}}

def _compute_laplacian(graph, laplacian_type="normalized"):
    graph.compute_laplacian(laplacian_type)
    laplacian = layers.prepare_laplacian(graph.L.astype(np.float32))
    return laplacian


class ConvBlock(torch.nn.Module):
    """ Spherical graph convolution block

    Parameters
    ----------
    in_channels : int
        Number of channels in the input graph.
    out_channels : int
        Number of channels in the output graph.
    kernel_size : int
        Chebychev polynomial degree
    laplacian : torch.sparse_coo_tensor
        Graph laplacian
    """

    def __init__(self, in_channels, out_channels, kernel_size, conv_type = 'graph', batch_norm=True, relu_activation=True, **kwargs):
        super().__init__()
    
        self.conv = ConvBlock.getConvLayer(in_channels, out_channels, kernel_size, conv_type, **kwargs)
        self.bn = BatchNorm1d(out_channels)
        self.norm = batch_norm
        self.act = relu_activation
        
    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.act:
            x = F.relu(x)
        return x
    
    @staticmethod
    def getConvLayer(in_channels: int, out_channels: int, kernel_size: int, conv_type: str= 'graph', **kwargs):
        conv_type = conv_type.lower()
        conv = None
        if conv_type == 'graph':
            assert 'laplacian' in kwargs
            conv = ALL_CONV[conv_type](in_channels, out_channels, kernel_size, **kwargs)
        elif conv_type == 'image':
            assert 'ratio' in kwargs
            conv = ALL_CONV[conv_type](in_channels, out_channels, kernel_size, **kwargs)
        else:
            raise ValueError(f'{conv_type} convolution is not supported')
        return conv


class PoolUnpoolBlock(torch.nn.Module):
    @staticmethod
    def getPoolUnpoolLayer(sampling: str, pool_method: str, **kwargs):
        sampling = sampling.lower()
        pool_method = pool_method.lower()
        assert sampling in ('healpix', 'equiangular')
        assert pool_method in ('max', 'avg')

        pooling, unpool = ALL_POOL[sampling][pool_method]
        return pooling(**kwargs), unpool(**kwargs)
    
    @staticmethod
    def getGeneralPoolUnpoolLayer(g1, g2, pool_method):
        if g1.n_vertices < g2.n_vertices:
            g1, g2 = g2, g1
        
        pool_mat, unpool_mat = build_pooling_matrices(g1, g2)
        if pool_method == 'interp':
            pool = GeneralAvgPool(pool_mat)
            unpool = GeneralAvgUnpool(unpool_mat)
            return pool, unpool
        elif pool_method == 'maxarea':
            pool = GeneralMaxAreaPool(pool_mat)
            unpool = GeneralMaxAreaUnpool(pool_mat.T)
            return pool, unpool
        elif pool_method == 'learn':
            pool = GeneralLearnablePool(pool_mat)
            unpool = GeneralLearnableUnpool(unpool_mat)
            return pool, unpool
        else:
            raise ValueError(f'{pool_method} is not supoorted.')


class UNet(ABC):
    @abstractmethod
    def encode(self, *args, **kwargs):
        """ Encodes an input into a lower dimensional space applying convolutional,
        batch normalisation and pooling layers
        """
        pass

    @abstractmethod
    def decode(self, *args, **kwargs):
        """
        Decodes low dimensional data into high dimensional applying convolutional, batch normalisation,
        unpooling layers and skip connections
        """
        pass

    def forward(self, x):
        x_encoded = self.encode(x)
        output = self.decode(*x_encoded)
        return output


class UNetSpherical(UNet, torch.nn.Module):
    """Spherical GCNN UNet

    Parameters
    ----------
    resolution : Union[int, List[int]]
        Resolution of data. Can be subdivision, nlat or (nlat. nlon)
    kernel_size_pooling : int
        Pooling's kernel size
    """

    def __init__(self,
                 dim_info: Dict[str, int],
                 resolution: Union[int, List[int]],    
                 conv_type: str='graph', 
                 kernel_size: int=3,
                 sampling: str='healpix',
                 knn: int=10, 
                 pool_method: str='max', 
                 kernel_size_pooling: int=4, 
                 periodic: Union[int, bool, None]=None, 
                 ratio: Union[int, float, bool, None]=None):
        super().__init__()
        in_channels = dim_info['input_feature_dim']*dim_info['input_time_dim']
        self.in_channels = in_channels
        out_channels = dim_info['output_feature_dim']*dim_info['output_time_dim']
        self.out_channels = out_channels
                                 

        conv_type = conv_type.lower()
        sampling = sampling.lower()
        pool_method = pool_method.lower()
        assert conv_type != 'image' or periodic is not None
        periodic = bool(periodic)


        if not isinstance(resolution, Iterable):
            resolution = [resolution]
        resolution = np.array(resolution)
        self.sphere_graph = UNetSpherical.build_graph([resolution], sampling, knn)[0]
        coarsening = int(np.sqrt(kernel_size_pooling))
        resolutions = [resolution, resolution // coarsening, resolution // coarsening // coarsening]
        self.graphs = []
        if conv_type == 'graph':
            self.graphs = UNetSpherical.build_graph(resolutions, sampling, knn)
            self.laplacians = UNetSpherical.get_laplacian_kernels(self.graphs)
        elif conv_type == 'image':
            self.laplacians = [None] * 20
        else:
            raise ValueError(f'{conv_type} convolution is not supported')
        

        # Pooling - unpooling
        if pool_method in ('interp', 'maxval', 'maxarea', 'learn'):
            assert conv_type == 'graph'
            self.pool1, self.unpool1 = PoolUnpoolBlock.getGeneralPoolUnpoolLayer(self.graphs[0], self.graphs[1], pool_method)
            self.pool2, self.unpool2 = PoolUnpoolBlock.getGeneralPoolUnpoolLayer(self.graphs[1], self.graphs[2], pool_method)
        else:
            self.pool1, self.unpool1 = PoolUnpoolBlock.getPoolUnpoolLayer(sampling, pool_method, kernel_size=kernel_size_pooling, ratio=ratio)
            self.pool2, self.unpool2 = PoolUnpoolBlock.getPoolUnpoolLayer(sampling, pool_method, kernel_size=kernel_size_pooling, ratio=ratio)

        # Encoding block 1
        self.conv11 = ConvBlock(in_channels, 32 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[0], periodic=periodic, ratio=ratio)
        self.conv13 = ConvBlock(32 * 2, 64 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[0], periodic=periodic, ratio=ratio)

        self.conv1_res = Linear(in_channels, 64 * 2)

        # Encoding block 2
        self.conv21 = ConvBlock(64 * 2, 96 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[1], periodic=periodic, ratio=ratio)
        self.conv23 = ConvBlock(96 * 2, 128 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[1], periodic=periodic, ratio=ratio)

        self.conv2_res = Linear(64 * 2, 128 * 2)

        # Encoding block 3
        self.conv31 = ConvBlock(128 * 2, 256 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[2], periodic=periodic, ratio=ratio)
        self.conv33 = ConvBlock(256 * 2, 128 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[2], periodic=periodic, ratio=ratio)

        self.conv3_res = Linear(128 * 2, 128 * 2)

        # Decoding block 2
        self.uconv21 = ConvBlock(256 * 2, 128 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[1], periodic=periodic, ratio=ratio)
        self.uconv22 = ConvBlock(128 * 2, 64 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[1], periodic=periodic, ratio=ratio)

        # Decoding block 1
        self.uconv11 = ConvBlock(128 * 2, 64 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[0], periodic=periodic, ratio=ratio)
        self.uconv12 = ConvBlock(64 * 2, 32 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[0], periodic=periodic, ratio=ratio)
        self.uconv13 = ConvBlock(32 * 2 * 2, out_channels, kernel_size, conv_type, False, False, laplacian=self.laplacians[0], periodic=periodic, ratio=ratio)
    

    def encode(self, x):
        # ['sample', 'time', 'node', 'feature'] ==> ['sample', 'node', 'time-feature']
        batchsize, time, nodes, _ = x.shape
        self.time = time
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batchsize, nodes, self.in_channels)

        # Block 1
        x_enc11 = self.conv11(x)
        x_enc1 = self.conv13(x_enc11)

        x_enc1 += self.conv1_res(x)

        # Block 2
        x_enc2_ini, idx1 = self.pool1(x_enc1)
        x_enc2 = self.conv21(x_enc2_ini)
        x_enc2 = self.conv23(x_enc2)

        x_enc2 += self.conv2_res(x_enc2_ini)

        # Block 3
        x_enc3_ini, idx2 = self.pool2(x_enc2)
        x_enc3 = self.conv31(x_enc3_ini)
        x_enc3 = self.conv33(x_enc3)

        x_enc3 += self.conv3_res(x_enc3_ini)

        return x_enc3, x_enc2, x_enc1, idx2, idx1, x_enc11


    def decode(self, x_enc3, x_enc2, x_enc1, idx2, idx1, x_enc11):
        # Block 2
        x = self.unpool2(x_enc3, idx2)
        x_cat = torch.cat((x, x_enc2), dim=2)
        x = self.uconv21(x_cat)
        x = self.uconv22(x)

        # Block 1
        x = self.unpool1(x, idx1)
        x_cat = torch.cat((x, x_enc1), dim=2)
        x = self.uconv11(x_cat)
        x = self.uconv12(x)
        x_cat = torch.cat((x, x_enc11), dim=2)
        x = self.uconv13(x_cat)

        # ['sample', 'time', 'node', 'feature'] <== ['sample', 'node', 'time-feature']
        batchsize, nodes, _ = x.shape
        x = x.reshape(batchsize, nodes, self.time, -1)
        x = x.permute(0, 2, 1, 3)
        return x

    @staticmethod
    def get_laplacian_kernels(graphs: List["pygsp.graphs"]):
        container = []
        for _, G in enumerate(graphs):
            laplacian = _compute_laplacian(G)
            container.append(laplacian)
        return container
    
    @staticmethod
    def build_graph(resolutions: List[List[int]], sampling='healpix', k: int=10) -> List["pygsp.graphs"]:
        sampling = sampling.lower()
        try:
            graph_initializer = ALL_GRAPH[sampling]
            params = ALL_GRAPH_PARAMS[sampling]
        except:
            raise ValueError(f'{sampling} is not supported')

        container = []
        params['k'] = k
        for _, res in enumerate(resolutions):
            G = graph_initializer(*res, **params)
            container.append(G)
        return container
