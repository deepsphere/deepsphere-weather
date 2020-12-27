from typing import Container, List, Union
from abc import ABC, abstractmethod

import torch
import pygsp
import numpy as np
from torch.nn import functional as F
from torch.nn import BatchNorm1d, Linear
from modules.layers import equiangular_dimension_unpack

from modules import layers
from modules.layers import (ConvCheb, Conv2dEquiangular,
    PoolMaxHealpix, 
    UnpoolMaxHealpix, 
    PoolAvgHealpix,
    UnpoolAvgHealpix,
    PoolMaxEquiangular,
    UnpoolMaxEquiangular,
    PoolAvgEquiangular,
    UnpoolAvgEquiangular)

from modules.layers import (GeneralInterpPoolUnpool, 
    convert_to_torch_sparse,
    build_pooling_matrices)

# Deprecated import, delete in the future
from modules.layers import Conv1dAuto

HEALPIX_POOL = {'max': (PoolMaxHealpix, UnpoolMaxHealpix), 
                'avg': (PoolAvgHealpix, UnpoolAvgHealpix)}

EQUIANGULAR_POOl = {'max': (PoolMaxEquiangular, UnpoolMaxEquiangular), 
                    'avg': (PoolAvgEquiangular, UnpoolAvgEquiangular)}

ALL_POOL = {'healpix': HEALPIX_POOL,
            'equiangular': EQUIANGULAR_POOl}

ALL_CONV = {'image': Conv2dEquiangular,
            'graph': ConvCheb}

def _compute_healpix(nodes, k=20):
    resolution = int(np.sqrt(nodes / 12))
    G = pygsp.graphs.SphereHealpix(subdivisions=resolution, k=k)
    return G


def _compute_equiangular(nodes):
    G = pygsp.graphs.SphereEquiangular(*nodes)
    return G


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
    def getGeneralPoolUnpoolLayer(graphs):
        assert len(graphs) >= 2
        pool = GeneralInterpPoolUnpool(isPool=True)
        unpool = GeneralInterpPoolUnpool(isPool=False)
        for i in range(len(graphs) - 1):
            g1, g2 = graphs[i], graphs[i + 1]
            n1, n2 = g1.n_vertices, g2.n_vertices
            pool_mat, unpool_mat = build_pooling_matrices(g1, g2)
            pool[n1] = convert_to_torch_sparse(pool_mat)
            unpool[n2] = convert_to_torch_sparse(unpool_mat)
        return pool, unpool


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
    num_nodes : List[int]
        Number of nodes in the input graph
    kernel_size_pooling : int
        Pooling's kernel size
    """

    def __init__(self, N: int, in_channels: int, out_channels: int, conv_type: str='graph', 
                 kernel_size: int=3, sampling: str='healpix', knn: int = 20, pool_method: str='max', kernel_size_pooling: int=4, 
                 periodic: Union[int, bool, None]=None, ratio: Union[int, float, bool, None]=None):
        super().__init__()

        conv_type = conv_type.lower()
        sampling = sampling.lower()
        pool_method = pool_method.lower()

        if sampling == 'equiangular':
            assert ratio is not None
            N = equiangular_dimension_unpack(N, ratio)
            N = np.array(N)

        num_nodes = [N, N / kernel_size_pooling, N / (kernel_size_pooling * kernel_size_pooling)]
        graphs = []
        if conv_type == 'graph':
            graphs = UNetSpherical.build_graph(num_nodes, sampling, knn)
            self.laplacians = UNetSpherical.get_laplacian_kernels(graphs)
        elif conv_type == 'image':
            self.laplacians = [None] * 20
        else:
            raise ValueError(f'{conv_type} convolution is not supported')
            
        if sampling == 'equiangular' and conv_type == 'image':
            assert periodic is not None
            periodic = bool(periodic)

        # Pooling - unpooling
        if pool_method == 'interp':
            assert conv_type == 'graph'
            self.pooling, self.unpool = PoolUnpoolBlock.getGeneralPoolUnpoolLayer(graphs)
        else:
            self.pooling, self.unpool = PoolUnpoolBlock.getPoolUnpoolLayer(sampling, pool_method, kernel_size=kernel_size_pooling, ratio=ratio)

        # Encoding block 1
        self.conv11 = ConvBlock(in_channels, 32 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[0], periodic=periodic, ratio=ratio)
        self.conv13 = ConvBlock(32 * 2, 64 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[0], periodic=periodic, ratio=ratio)

        self.conv1_res = Conv1dAuto(in_channels, 64 * 2, 1)

        # Encoding block 2
        self.conv21 = ConvBlock(64 * 2, 96 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[1], periodic=periodic, ratio=ratio)
        self.conv23 = ConvBlock(96 * 2, 128 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[1], periodic=periodic, ratio=ratio)

        self.conv2_res = Conv1dAuto(64 * 2, 128 * 2, 1)

        # Encoding block 3
        self.conv31 = ConvBlock(128 * 2, 256 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[2], periodic=periodic, ratio=ratio)
        self.conv33 = ConvBlock(256 * 2, 128 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[2], periodic=periodic, ratio=ratio)

        self.conv3_res = Conv1dAuto(128 * 2, 128 * 2, 1)

        # Decoding block 2
        self.uconv21 = ConvBlock(256 * 2, 128 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[1], periodic=periodic, ratio=ratio)
        self.uconv22 = ConvBlock(128 * 2, 64 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[1], periodic=periodic, ratio=ratio)

        # Decoding block 1
        self.uconv11 = ConvBlock(128 * 2, 64 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[0], periodic=periodic, ratio=ratio)
        self.uconv12 = ConvBlock(64 * 2, 32 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[0], periodic=periodic, ratio=ratio)
        self.uconv13 = ConvBlock(32 * 2 * 2, out_channels, kernel_size, conv_type, False, False, laplacian=self.laplacians[0], periodic=periodic, ratio=ratio)
    

    def encode(self, x):
        # Block 1

        x_enc11 = self.conv11(x)
        x_enc1 = self.conv13(x_enc11)

        x_enc1 += torch.transpose(self.conv1_res(torch.transpose(x, 2, 1)), 2, 1)

        # Block 2
        x_enc2_ini, idx1 = self.pooling(x_enc1)
        x_enc2 = self.conv21(x_enc2_ini)
        x_enc2 = self.conv23(x_enc2)

        x_enc2 += torch.transpose(self.conv2_res(torch.transpose(x_enc2_ini, 2, 1)), 2, 1)

        # Block 3
        x_enc3_ini, idx2 = self.pooling(x_enc2)
        x_enc3 = self.conv31(x_enc3_ini)
        x_enc3 = self.conv33(x_enc3)

        x_enc3 += torch.transpose(self.conv3_res(torch.transpose(x_enc3_ini, 2, 1)), 2, 1)

        return x_enc3, x_enc2, x_enc1, idx2, idx1, x_enc11


    def decode(self, x_enc3, x_enc2, x_enc1, idx2, idx1, x_enc11):
        # Block 2
        x = self.unpool(x_enc3, idx2)
        x_cat = torch.cat((x, x_enc2), dim=2)
        x = self.uconv21(x_cat)
        x = self.uconv22(x)

        # Block 1
        x = self.unpool(x, idx1)
        x_cat = torch.cat((x, x_enc1), dim=2)
        x = self.uconv11(x_cat)
        x = self.uconv12(x)
        x_cat = torch.cat((x, x_enc11), dim=2)
        x = self.uconv13(x_cat)
        return x

    @staticmethod
    def get_laplacian_kernels(graphs: List["pygsp.graphs"]):
        container = []
        for _, G in enumerate(graphs):
            laplacian = _compute_laplacian(G)
            container.append(laplacian)
        return container
    
    @staticmethod
    def build_graph(nodes_list: Union[List[int], List[List[int]]], sampling='healpix', k: int=20) -> List["pygsp.graphs"]:
        container = []
        for _, nodes in enumerate(nodes_list):
            if sampling == 'healpix':
                G = _compute_healpix(nodes, k=k)
            elif sampling == 'equiangular':
                G = _compute_equiangular(nodes)
            else:
                raise ValueError(f'{sampling} is not supported')
            container.append(G)
        return container


class SphericalGraphUNet(UNet, torch.nn.Module):
    """Spherical GCNN UNet

    Parameters
    ----------
    num_nodes : List[int]
        Number of nodes in the input graph
    kernel_size_pooling : int
        Pooling's kernel size
    """

    def __init__(self, N: int, in_channels: int, out_channels: int, kernel_size: int=3, 
                 sampling: str='healpix', knn: int=20, kernel_size_pooling: int=4, 
                 ratio: Union[int, float, bool, None]=None):
        super().__init__()

        conv_type = 'graph'
        sampling = sampling.lower()

        if sampling == 'equiangular':
            assert ratio is not None
            N = equiangular_dimension_unpack(N, ratio)
            N = np.array(N)

        num_nodes = [N, N / kernel_size_pooling, N / (kernel_size_pooling * kernel_size_pooling)]
        graphs = SphericalGraphUNet.build_graph(num_nodes, sampling, knn)
        self.laplacians = UNetSpherical.get_laplacian_kernels(graphs)

        self.pooling, self.unpool = PoolUnpoolBlock.getGeneralPoolUnpoolLayer(graphs)

        # Encoding block 1
        self.conv11 = ConvBlock(in_channels, 32 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[0], ratio=ratio)
        self.conv13 = ConvBlock(32 * 2, 64 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[0], ratio=ratio)

        # self.conv1_res = Conv1dAuto(in_channels, 64 * 2, 1)
        self.conv1_res = Linear(in_channels, 64 * 2)

        # Encoding block 2
        self.conv21 = ConvBlock(64 * 2, 96 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[1], ratio=ratio)
        self.conv23 = ConvBlock(96 * 2, 128 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[1], ratio=ratio)

        # self.conv2_res = Conv1dAuto(64 * 2, 128 * 2, 1)
        self.conv2_res = Linear(64 * 2, 128 * 2)

        # Encoding block 3
        self.conv31 = ConvBlock(128 * 2, 256 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[2], ratio=ratio)
        self.conv33 = ConvBlock(256 * 2, 128 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[2], ratio=ratio)

        # self.conv3_res = Conv1dAuto(128 * 2, 128 * 2, 1)
        self.conv3_res = Linear(128 * 2, 128 * 2)

        # Decoding block 2
        self.uconv21 = ConvBlock(256 * 2, 128 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[1], ratio=ratio)
        self.uconv22 = ConvBlock(128 * 2, 64 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[1], ratio=ratio)

        # Decoding block 1
        self.uconv11 = ConvBlock(128 * 2, 64 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[0], ratio=ratio)
        self.uconv12 = ConvBlock(64 * 2, 32 * 2, kernel_size, conv_type, True, True, laplacian=self.laplacians[0], ratio=ratio)
        self.uconv13 = ConvBlock(32 * 2 * 2, out_channels, kernel_size, conv_type, False, False, laplacian=self.laplacians[0], ratio=ratio)
    

    def encode(self, x):
        # Block 1

        x_enc11 = self.conv11(x)
        x_enc1 = self.conv13(x_enc11)

        # x_enc1 += torch.transpose(self.conv1_res(torch.transpose(x, 2, 1)), 2, 1)
        x_enc1 += self.conv1_res(x)

        # Block 2
        x_enc2_ini, idx1 = self.pooling(x_enc1)
        x_enc2 = self.conv21(x_enc2_ini)
        x_enc2 = self.conv23(x_enc2)

        # x_enc2 += torch.transpose(self.conv2_res(torch.transpose(x_enc2_ini, 2, 1)), 2, 1)
        x_enc2 += self.conv2_res(x_enc2_ini)

        # Block 3
        x_enc3_ini, idx2 = self.pooling(x_enc2)
        x_enc3 = self.conv31(x_enc3_ini)
        x_enc3 = self.conv33(x_enc3)

        # x_enc3 += torch.transpose(self.conv3_res(torch.transpose(x_enc3_ini, 2, 1)), 2, 1)
        x_enc3 += self.conv3_res(x_enc3_ini)

        return x_enc3, x_enc2, x_enc1, idx2, idx1, x_enc11


    def decode(self, x_enc3, x_enc2, x_enc1, idx2, idx1, x_enc11):
        # Block 2
        x = self.unpool(x_enc3, idx2)
        x_cat = torch.cat((x, x_enc2), dim=2)
        x = self.uconv21(x_cat)
        x = self.uconv22(x)

        # Block 1
        x = self.unpool(x, idx1)
        x_cat = torch.cat((x, x_enc1), dim=2)
        x = self.uconv11(x_cat)
        x = self.uconv12(x)
        x_cat = torch.cat((x, x_enc11), dim=2)
        x = self.uconv13(x_cat)
        return x

    @staticmethod
    def get_laplacian_kernels(graphs: List["pygsp.graphs"]):
        container = []
        for _, G in enumerate(graphs):
            laplacian = _compute_laplacian(G)
            container.append(laplacian)
        return container
    
    @staticmethod
    def build_graph(nodes_list: Union[List[int], List[List[int]]], sampling='healpix', k: int=20) -> List["pygsp.graphs"]:
        container = []
        for _, nodes in enumerate(nodes_list):
            if sampling == 'healpix':
                G = _compute_healpix(nodes, k=k)
            elif sampling == 'equiangular':
                G = _compute_equiangular(nodes)
            else:
                raise ValueError(f'{sampling} is not supported')
            container.append(G)
        return container