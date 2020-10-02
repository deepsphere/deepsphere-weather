from typing import List
from abc import ABC, abstractmethod

import torch
import pygsp
import numpy as np
from torch.nn import functional as F
from torch.nn import BatchNorm1d

from modules import layers
from modules.layers import ConvCheb, PoolMaxHealpix, UnpoolMaxHealpix, Conv1dAuto


def _compute_laplacian_healpix(nodes, laplacian_type="normalized"):
    """ Computes laplacian of spherical graph sampled as a HEALpix grid

    Parameters
    ----------
    nodes : int
        Number of nodes in the graph
    laplacian_type : string
        Type of laplacian. Options are {´normalized´, ´combinatorial´}

    Returns
    -------
    laplacian : torch.sparse_coo_tensor
        Graph laplacian
    """
    resolution = int(np.sqrt(nodes / 12))

    G = pygsp.graphs.SphereHealpix(nside=resolution, n_neighbors=20)
    G.compute_laplacian(laplacian_type)
    laplacian = layers.prepare_laplacian(G.L.astype(np.float32))

    return laplacian


def get_laplacian_kernels(container: List, nodes_list: List[int]) -> None:
    for i, nodes in enumerate(nodes_list):
        laplacian = _compute_laplacian_healpix(nodes)
        container.append(laplacian)


class BaseModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def state_dict(self, *args, **kwargs):
        """
        This function overrides the state dict in order to be able to save the model.
        This can be removed as soon as saving sparse matrices has been added to Pytorch.
        """
        state_dict = super().state_dict(*args, **kwargs)
        del_keys = []
        for key in state_dict:
            if "laplacian" in key:
                del_keys.append(key)
        for key in del_keys:
            del state_dict[key]
        return state_dict


class ConvBlock(BaseModule):
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

    def __init__(self, in_channels, out_channels, kernel_size, laplacian):
        super().__init__()

        self.conv = ConvCheb(in_channels, out_channels, kernel_size, laplacian)
        self.bn = BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        return x


class BottleNeckBlock(BaseModule):
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

    def __init__(self, in_channels, out_channels, laplacian):
        super().__init__()

        self.conv1 = ConvCheb(in_channels, out_channels, 1, laplacian)
        self.conv2 = ConvCheb(out_channels, out_channels, 3, laplacian)
        self.conv3 = ConvCheb(out_channels, in_channels, 1, laplacian)
        self.bn1 = BatchNorm1d(out_channels)
        self.bn2 = BatchNorm1d(out_channels)
        self.bn3 = BatchNorm1d(in_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class SphericalHealpixBlottleNeck(BaseModule):
    """Spherical GCNN UNet

     Parameters
    ----------
    N : int
        Number of nodes in the input graph
    in_channels : int
        Number of channels in the input graph.
    out_channels : int
        Number of channels in the output graph.
    kernel_size : int
        Chebychev polynomial degree

    Residual connections based on: https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, N, in_channels, out_channels, kernel_size, kernel_size_pooling=4):
        super().__init__()

        self.kernel_size = kernel_size

        laplacians = []
        get_laplacian_kernels(laplacians, [N])

        # First convolution
        self.conv1 = ConvBlock(in_channels, 64, 3, laplacians[0])
        self.conv2 = ConvBlock(64, 256, 3, laplacians[0])

        # First BottleNeck Block
        self.bottleneck1 = BottleNeckBlock(256, 64, laplacians[0])

        # Second BottleNeck Block
        self.bottleneck2 = BottleNeckBlock(256, 64, laplacians[0])

        # Third BottleNeck Block
        self.bottleneck3 = BottleNeckBlock(256, 64, laplacians[0])

        self.conv3 = ConvCheb(256, out_channels, kernel_size, laplacians[0])

    def forward(self, x):
        """Forward Pass
        Parameters
        ----------
        x : torch.Tensor of shape batch_size x n_vertices x in_channels
            Input data
        Returns
        -------
        x : torch.Tensor of shape batch_size x n_vertices x out_channels
            Model output
        """
        x = self.conv1(x)
        x = self.conv2(x)

        x += self.bottleneck1(x)
        x += self.bottleneck2(x)
        x += self.bottleneck3(x)

        output = self.conv3(x)

        return output


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


class UNetSphericalHealpix(UNet, BaseModule, ABC):
    """Spherical GCNN UNet

    Parameters
    ----------
    num_nodes : List[int]
        Number of nodes in the input graph
    kernel_size_pooling : int
        Pooling's kernel size
    """

    def __init__(self, num_nodes, kernel_size_pooling=4):
        super().__init__()

        laplacians = []
        get_laplacian_kernels(laplacians, num_nodes)
        self.laplacians = laplacians

        # Pooling - unpooling
        self.pooling = PoolMaxHealpix(kernel_size=kernel_size_pooling)
        self.unpool = UnpoolMaxHealpix(kernel_size=kernel_size_pooling)


class UNetSphericalHealpixResidualLongConnections(UNetSphericalHealpix):
    def __init__(self, N, in_channels, out_channels, kernel_size, kernel_size_pooling=4):
        num_nodes = [N, N / kernel_size_pooling, N / (kernel_size_pooling * kernel_size_pooling)]
        super().__init__(num_nodes, kernel_size_pooling)

        # Encoding block 1
        self.conv11 = ConvBlock(in_channels, max(in_channels, 32 * 2), kernel_size, self.laplacians[0])
        self.conv13 = ConvBlock(max(in_channels, 32 * 2), 64 * 2, kernel_size, self.laplacians[0])

        self.conv1_res = Conv1dAuto(in_channels, 64 * 2, 1)

        # Encoding block 2
        self.conv21 = ConvBlock(64 * 2, 96 * 2, kernel_size, self.laplacians[1])
        self.conv23 = ConvBlock(96 * 2, 128 * 2, kernel_size, self.laplacians[1])

        self.conv2_res = Conv1dAuto(64 * 2, 128 * 2, 1)

        # Encoding block 3
        self.conv31 = ConvBlock(128 * 2, 256 * 2, kernel_size, self.laplacians[2])
        self.conv33 = ConvBlock(256 * 2, 128 * 2, kernel_size, self.laplacians[2])

        self.conv3_res = Conv1dAuto(128 * 2, 128 * 2, 1)

        # Decoding block 2
        self.uconv21 = ConvBlock(256 * 2, 128 * 2, kernel_size, self.laplacians[1])
        self.uconv22 = ConvBlock(128 * 2, 64 * 2, kernel_size, self.laplacians[1])

        # Decoding block 1
        self.uconv11 = ConvBlock(128 * 2, 64 * 2, kernel_size, self.laplacians[0])
        self.uconv12 = ConvBlock(64 * 2, 32 * 2, kernel_size, self.laplacians[0])
        self.uconv13 = ConvCheb(32 * 2 * 2, out_channels, kernel_size, self.laplacians[0])

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
