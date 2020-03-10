import pygsp

import torch
from torch.nn import functional as F

from deepsphere.utils.samplings import equiangular_dimension_unpack

import modules.layers as layers


def compute_laplacian(nodes, ratio, laplacian_type="normalized"):
    dim1, dim2 = equiangular_dimension_unpack(nodes, ratio)
    
    bw = [int(dim1/2), int(dim2/2)]

    G = pygsp.graphs.SphereEquiangular(bandwidth=bw, sampling="SOFT")
    G.compute_laplacian(laplacian_type)
    laplacian = layers.prepare_laplacian(G.L)
    
    return laplacian


class SphericalCNN(torch.nn.Module):
    """Spherical GCNN with WeatherBench CNN architecture
    """

    def __init__(self, N, ratio, in_channels, out_channels, kernel_size):
        """Initialization.
        Args:
            N (int): number of pixels in the input image
            ratio (float): parameter for equiangular sampling -> width/height
            in_channels (int): number of input features
            out_channel (int): number of output features
            kernel_size (int): chebychev polynomial degree
        """
        super().__init__()
        
        self.kernel_size = kernel_size
        self.laplacian = compute_laplacian(N, ratio)
        
        self.conv1 = layers.ChebConv(in_channels, 64, self.kernel_size)
        self.conv2 = layers.ChebConv(64, 64, self.kernel_size)
        self.conv3 = layers.ChebConv(64, 64, self.kernel_size)
        self.conv4 = layers.ChebConv(64, 64, self.kernel_size)
        self.conv5 = layers.ChebConv(64, out_channels, self.kernel_size)
        
        for m in self.modules():
            if isinstance(m, layers.ChebConv):
                m.reset_parameters(activation='linear', fan='avg', distribution='uniform')

    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): input to be forwarded.
        Returns:
            :obj:`torch.Tensor`: output
        """
        
        x = F.elu(self.conv1(self.laplacian, x))
        x = F.elu(self.conv2(self.laplacian, x))
        x = F.elu(self.conv3(self.laplacian, x))
        x = F.elu(self.conv4(self.laplacian, x))
        x = self.conv5(x)
        
        return x
    
    
class PeriodicCNN(torch.nn.Module):
    """Fully convolutional CNN with periodic padding on the longitude dimension.
    WeatherBench CNN architecture.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size):
        """Initialization.
        Args:
            in_channels (int): number of input features
            out_channel (int): number of output features
            kernel_size (int): convolutional kernel width 
        """
        
        super().__init__()
        
        self.conv1 = layers.PeriodicConv2D(in_channels, 64, kernel_size)
        self.conv2 = layers.PeriodicConv2D(64, 64, kernel_size)
        self.conv3 = layers.PeriodicConv2D(64, 64, kernel_size)
        self.conv4 = layers.PeriodicConv2D(64, 64, kernel_size)
        self.conv5 = layers.PeriodicConv2D(64, out_channels, kernel_size)
        
    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): input to be forwarded.
        Returns:
            :obj:`torch.Tensor`: output
        """
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = self.conv5(x)
        
        return x
