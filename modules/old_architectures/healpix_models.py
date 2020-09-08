import pygsp
import numpy as np

import torch
from torch.nn import functional as F
from torch.nn import BatchNorm1d, BatchNorm2d, Conv1d, Identity

from modules import layers

from modules.layers import (ConvCheb, PoolMaxHealpix, UnpoolMaxHealpix,
                            ConvChebTemp, PoolMaxTempHealpix, UnpoolMaxTempHealpix)


def weights_init(m):
    if isinstance(m, ConvCheb):
        torch.nn.init.normal_(m.weight.data, 0, 1.5)
        torch.nn.init.normal_(m.bias.data, 0, 1.5)

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
    def __init__(self, in_channels, out_channels, kernel_size, laplacian):
        super().__init__()

        self.conv = ConvCheb(in_channels, out_channels, kernel_size, laplacian)
        self.bn = BatchNorm1d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        return x


class BottleNeckBlock(torch.nn.Module):
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

    
class UNetSphericalHealpix(torch.nn.Module):
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
    """

    def __init__(self, N, in_channels, out_channels, kernel_size):        
        super().__init__()

        self.kernel_size = kernel_size
       
        laplacians = []
        for i, nodes in enumerate([3072, 768, 192]):
            laplacian = _compute_laplacian_healpix(nodes)
            laplacians.append(laplacian)
        
        
        # Pooling - unpooling
        self.pooling = PoolMaxHealpix(kernel_size=4)
        self.unpool = UnpoolMaxHealpix(kernel_size=4)
        
        # Encoding block 1
        self.conv11 = ConvBlock(in_channels, max(in_channels, 16), kernel_size, laplacians[0])
        self.conv12 = ConvBlock(max(in_channels, 16), max(in_channels, 32), kernel_size, laplacians[0])
        self.conv13 = ConvBlock(max(in_channels, 32), 64, kernel_size, laplacians[0])
        
        # Encoding block 2
        self.conv21 = ConvBlock(64, 88, kernel_size, laplacians[1])
        self.conv22 = ConvBlock(88, 110, kernel_size, laplacians[1])
        self.conv23 = ConvBlock(110, 128, kernel_size, laplacians[1])
       
        # Encoding block 3
        self.conv31 = ConvBlock(128, 256, kernel_size, laplacians[2])
        self.conv32 = ConvBlock(256, 256, kernel_size, laplacians[2])
        self.conv33 = ConvBlock(256, 128, kernel_size, laplacians[2])
        
        # Decoding block 4
        self.uconv21 = ConvBlock(256, 128, kernel_size, laplacians[1])
        self.uconv22 = ConvBlock(128, 64, kernel_size, laplacians[1])
        
        # Decoding block 4
        self.uconv11 = ConvBlock(128, 64, kernel_size, laplacians[0])
        self.uconv12 = ConvBlock(64, 32, kernel_size, laplacians[0])
        self.uconv13 = ConvCheb(32, out_channels, kernel_size, laplacians[0])
        
         
    def encode(self, x):
        """ Encodes an input into a lower dimensional space applying convolutional, batch normalisation and pooling layers
        Parameters
        ----------
        x : torch.Tensor of shape batch_size x n_vertices x in_channels
            Input data
        Returns
        -------
       x_enc3, x_enc2, x_enc1, idx2, idx1 : torch.Tensors of shapes batch_size x n_vertices x layer_channels + list(int)
            Encoded data at the different encoding stages and the indices indicating the locations of the maxium values in
            unpooled images.
        """
        #x_enc1 = self.dropout1(x_enc1)
        
        # Block 1
        
        x_enc1 = self.conv11(x)
        x_enc1 = self.conv12(x_enc1)
        x_enc1 = self.conv13(x_enc1)
        
        # Block 2
        x_enc2, idx1 = self.pooling(x_enc1)
        x_enc2 = self.conv21(x_enc2)
        x_enc2 = self.conv22(x_enc2)
        x_enc2 = self.conv23(x_enc2)
        
        # Block 3
        x_enc3, idx2 = self.pooling(x_enc2)
        x_enc3 = self.conv31(x_enc3)
        x_enc3 = self.conv32(x_enc3)
        x_enc3 = self.conv33(x_enc3)
        
        return x_enc3, x_enc2, x_enc1, idx2, idx1
    
    def decode(self, x_enc3, x_enc2, x_enc1, idx2, idx1):
        """ Decodes low dimensional data into high dimensional applying convolutional, batch normalisation, 
        unpooling layers and skip connections
        
        Parameters
        ----------
        x_enc3, x_enc2, x_enc1, idx2, idx1 : torch.Tensors of shapes batch_size x n_vertices x layer_channels + list(int)
            Encoded data at the different encoding stages and the indices indicating the locations of the maxium values in
            unpooled images.
        Returns
        -------
        x : torch.Tensor of shape batch_size x n_vertices x out_channels
            Decoded data
        """

        # Block 2
        x = self.unpool(x_enc3, idx2)
        x = torch.cat((x, x_enc2), dim=2)
        x = self.uconv21(x)
        x = self.uconv22(x)
        
        # Block 1
        x = self.unpool(x, idx1)
        x = torch.cat((x, x_enc1), dim=2)
        x = self.uconv11(x)
        x = self.uconv12(x)
        x = self.uconv13(x)
            
        return x
    
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
        x_encoded = self.encode(x)
        output = self.decode(*x_encoded)
        return output


class UNetSphericalHealpixDeep(torch.nn.Module):
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
    """

    def __init__(self, N, in_channels, out_channels, kernel_size, kernel_size_pooling=4):
        super().__init__()

        self.kernel_size = kernel_size

        num_nodes = 3072
        laplacians = []
        for i, nodes in enumerate(
                [num_nodes, num_nodes / kernel_size_pooling, num_nodes / (kernel_size * kernel_size_pooling)]):
            laplacian = _compute_laplacian_healpix(nodes)
            laplacians.append(laplacian)

        # Pooling - unpooling
        self.pooling = PoolMaxHealpix(kernel_size=kernel_size_pooling)
        self.unpool = UnpoolMaxHealpix(kernel_size=kernel_size_pooling)

        # Encoding block 1
        self.conv11 = ConvBlock(in_channels, max(in_channels, 32 * 2), kernel_size, laplacians[0])
        self.conv13 = ConvBlock(max(in_channels, 32 * 2), 64 * 2, kernel_size, laplacians[0])

        self.conv1_res = Conv1dAuto(in_channels, 64 * 2, 1)

        # Encoding block 2
        self.conv21 = ConvBlock(64 * 2, 96 * 2, kernel_size, laplacians[1])
        self.conv23 = ConvBlock(96 * 2, 128 * 2, kernel_size, laplacians[1])

        self.conv2_res = Conv1dAuto(64 * 2, 128 * 2, 1)

        # Encoding block 3
        self.conv31 = ConvBlock(128 * 2, 256 * 2, kernel_size, laplacians[2])
        # self.conv32 = ConvBlock(256*2, 256*2, kernel_size, laplacians[2])
        self.conv33 = ConvBlock(256 * 2, 128 * 2, kernel_size, laplacians[2])

        self.conv3_res = Conv1dAuto(128 * 2, 128 * 2, 1)

        # Decoding block 4
        self.uconv21 = ConvBlock(256 * 2, 128 * 2, kernel_size, laplacians[1])
        self.uconv22 = ConvBlock(128 * 2, 64 * 2, kernel_size, laplacians[1])

        # Decoding block 4
        self.uconv11 = ConvBlock(128 * 2, 64 * 2, kernel_size, laplacians[0])
        self.uconv12 = ConvBlock(64 * 2, 32 * 2, kernel_size, laplacians[0])
        self.uconv13 = ConvCheb(32 * 2, out_channels, kernel_size, laplacians[0])

    def encode(self, x):
        """ Encodes an input into a lower dimensional space applying convolutional, batch normalisation and pooling layers
        Parameters
        ----------
        x : torch.Tensor of shape batch_size x n_vertices x in_channels
            Input data
        Returns
        -------
       x_enc3, x_enc2, x_enc1, idx2, idx1 : torch.Tensors of shapes batch_size x n_vertices x layer_channels + list(int)
            Encoded data at the different encoding stages and the indices indicating the locations of the maxium values in
            unpooled images.
        """
        # x_enc1 = self.dropout1(x_enc1)

        # Block 1

        x_enc1 = self.conv11(x)
        x_enc1 = self.conv13(x_enc1)

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

        return x_enc3, x_enc2, x_enc1, idx2, idx1

    def decode(self, x_enc3, x_enc2, x_enc1, idx2, idx1):
        """ Decodes low dimensional data into high dimensional applying convolutional, batch normalisation,
        unpooling layers and skip connections

        Parameters
        ----------
        x_enc3, x_enc2, x_enc1, idx2, idx1 : torch.Tensors of shapes batch_size x n_vertices x layer_channels + list(int)
            Encoded data at the different encoding stages and the indices indicating the locations of the maxium values in
            unpooled images.
        Returns
        -------
        x : torch.Tensor of shape batch_size x n_vertices x out_channels
            Decoded data
        """

        # Block 2
        x = self.unpool(x_enc3, idx2)
        x = torch.cat((x, x_enc2), dim=2)
        x = self.uconv21(x)
        x = self.uconv22(x)

        # Block 1
        x = self.unpool(x, idx1)
        x = torch.cat((x, x_enc1), dim=2)
        x = self.uconv11(x)
        x = self.uconv12(x)
        x = self.uconv13(x)

        return x

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
        x_encoded = self.encode(x)
        output = self.decode(*x_encoded)
        return output
    
##### Spatio-temporal convolutions

class ConvBlockTemp(torch.nn.Module):
    """ Spherical graph spatio-convolution block
    
    Parameters
    ----------
    in_channels : int
        Number of channels in the input graph.
    out_channels : int
        Number of channels in the output graph.
    graph_width : int
        Chebychev polynomial degree
    temp_width : int
        Width of the convolutional kernel in the temporal dimension
    laplacian : torch.sparse_coo_tensor
        Graph laplacian
    conv : functional
        Type of convolution (options are {cheb_conv_temp, cheb_conv_temp_reduce}
    """
    
    def __init__(self, in_channels, out_channels, graph_width, temp_width, laplacian):
        super().__init__()

        self.conv = ConvChebTemp(in_channels, out_channels, graph_width, temp_width, laplacian)
        self.bn = BatchNorm2d(out_channels)
        
    def forward(self, x):
        """Forward Pass
        Parameters
        ----------
        x : torch.Tensor of shape batch_size x n_vertices x len_sqce x in_channels
            Input data
        Returns
        -------
        x : torch.Tensor of shape batch_size x n_vertices x len_sqce x out_channels
            Model output
        """
        x = self.conv(x)
        print('Before bn ', x.shape)
        print('After bn ', self.bn(x.permute(0, 3, 1, 2)).shape)
        x = self.bn(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = F.relu(x)
        return x
    

class UNetSphericalTempHealpix(torch.nn.Module):
    """Spherical GCNN UNet
    
    Parameters
    ----------
    N : int
        Number of nodes in the input graph
    in_channels : int
        Number of channels in the input graph.
    out_channels : int
        Number of channels in the output graph.
    graph_width : int
        Chebychev polynomial degree
    """

    def __init__(self, N, len_sqce, in_channels, out_channels, graph_width):        
        super().__init__()

        self.graph_width = graph_width
        self.temp_width = len_sqce
       
        laplacians = []
        for i, nodes in enumerate([3072, 768, 192]):
            laplacian = _compute_laplacian_healpix(nodes)
            laplacians.append(laplacian)
        
        
        # Pooling - unpooling
        self.pool1 = PoolMaxTempHealpix(kernel_size=(4, 2))
        self.unpool1 = UnpoolMaxTempHealpix(kernel_size=(4, 2))
        
        if len_sqce < 4:
            self.pool2 = PoolMaxTempHealpix(kernel_size=(4, 1))
            self.unpool2 = UnpoolMaxTempHealpix(kernel_size=(4, 1))
        else:
            self.pool2 = PoolMaxTempHealpix(kernel_size=(4, 2))
            self.unpool2 = UnpoolMaxTempHealpix(kernel_size=(4, 2))
        
        self.pool_temp = PoolMaxTempHealpix(kernel_size=(1, len_sqce))
        
        
        # Encoding block 1
        self.conv11 = ConvBlockTemp(in_channels, 16, graph_width, len_sqce, laplacians[0])
        self.conv12 = ConvBlockTemp(16, 32, graph_width, len_sqce, laplacians[0])
        self.conv13 = ConvBlockTemp(32, 64, graph_width, len_sqce, laplacians[0])
        
        # Encoding block 2
        self.conv21 = ConvBlockTemp(64, 88, graph_width, int(len_sqce/2), laplacians[1])
        self.conv22 = ConvBlockTemp(88, 110, graph_width, int(len_sqce/2), laplacians[1])
        self.conv23 = ConvBlockTemp(110, 128, graph_width, int(len_sqce/2), laplacians[1])
       
        # Encoding block 3
        self.conv31 = ConvBlockTemp(128, 256, graph_width, max(1, int(len_sqce/4)), laplacians[2])
        self.conv32 = ConvBlockTemp(256, 256, graph_width, max(1, int(len_sqce/4)), laplacians[2])
        self.conv33 = ConvBlockTemp(256, 128, graph_width, max(1, int(len_sqce/4)), laplacians[2])
        
        # Decoding block 4
        self.uconv21 = ConvBlockTemp(256, 128, graph_width, int(len_sqce/2), laplacians[1])
        self.uconv22 = ConvBlockTemp(128, 64, graph_width, int(len_sqce/2), laplacians[1])
        
        # Decoding block 4
        self.uconv11 = ConvBlockTemp(128, 64, graph_width, len_sqce, laplacians[0])
        self.uconv12 = ConvBlockTemp(64, 32, graph_width, len_sqce, laplacians[0])
        self.uconv13 = ConvChebTemp(32, out_channels, graph_width, len_sqce, laplacians[0])
        
         
    def encode(self, x):
        """ Encodes an input into a lower dimensional space applying convolutional, batch normalisation and pooling layers
        Parameters
        ----------
        x : torch.Tensor of shape batch_size x len_sqce x n_vertices x in_channels
            Input data
        Returns
        -------
       x_enc3, x_enc2, x_enc1, idx2, idx1 : torch.Tensors of shapes batch_size x len_sqce x n_vertices x layer_channels
                                            + list(int)
            Encoded data at the different encoding stages and the indices indicating the locations of the maxium 
            values in unpooled images.
        """
        
        # Block 1
        x_enc1 = self.conv11(x)
        x_enc1 = self.conv12(x_enc1)
        x_enc1 = self.conv13(x_enc1)
        
        # Block 2
        x_enc2, idx1 = self.pool1(x_enc1)
        x_enc2 = self.conv21(x_enc2)
        x_enc2 = self.conv22(x_enc2)
        x_enc2 = self.conv23(x_enc2)
        
        # Block 3
        x_enc3, idx2 = self.pool2(x_enc2)
        x_enc3 = self.conv31(x_enc3)
        x_enc3 = self.conv32(x_enc3)
        x_enc3 = self.conv33(x_enc3)
        
        return x_enc3, x_enc2, x_enc1, idx2, idx1
    
    def decode(self, x_enc3, x_enc2, x_enc1, idx2, idx1):
        """ Decodes low dimensional data into high dimensional applying convolutional, batch normalisation, 
        unpooling layers and skip connections
        
        Parameters
        ----------
        x_enc3, x_enc2, x_enc1, idx2, idx1 : torch.Tensors of shapes batch_size x len_sqce x n_vertices x layer_channels
                                            + list(int)
            Encoded data at the different encoding stages and the indices indicating the locations of the maxium values in
            unpooled images.
        Returns
        -------
        x : torch.Tensor of shape batch_size x len_sqce x n_vertices x out_channels
            Decoded data
        """

        # Block 2
        x = self.unpool2(x_enc3, idx2)
        x = torch.cat((x, x_enc2), dim=3)
        x = self.uconv21(x)
        x = self.uconv22(x)
        
        # Block 1
        x = self.unpool1(x, idx1)
        x = torch.cat((x, x_enc1), dim=3)        
        x = self.uconv11(x)
        x = self.uconv12(x)
        x = self.uconv13(x)
            
        return x
    
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
        x_encoded = self.encode(x)
        output = self.decode(*x_encoded)
        return output
    

### add residual connections

class Conv1dAuto(Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2) # dynamic add padding based on the kernel_size
        
        
        
class UNetSphericalHealpixResidual(torch.nn.Module):
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
    """

    def __init__(self, N, in_channels, out_channels, kernel_size, kernel_size_pooling=4):
        super().__init__()

        self.kernel_size = kernel_size

        num_nodes = 3072
        laplacians = []
        for i, nodes in enumerate(
                [num_nodes, num_nodes / kernel_size_pooling, num_nodes / (kernel_size * kernel_size_pooling)]):
            laplacian = _compute_laplacian_healpix(nodes)
            laplacians.append(laplacian)

        # Pooling - unpooling
        self.pooling = PoolMaxHealpix(kernel_size=kernel_size_pooling)
        self.unpool = UnpoolMaxHealpix(kernel_size=kernel_size_pooling)

        # Encoding block 1
        self.conv11 = ConvBlock(in_channels, max(in_channels, 32 * 2), kernel_size, laplacians[0])
        self.conv13 = ConvBlock(max(in_channels, 32 * 2), 64 * 2, kernel_size, laplacians[0])
        self.conv1_res = Conv1dAuto(64 * 2, 64 * 2, 1)

        # Encoding block 2
        self.conv21 = ConvBlock(64 * 2, 96 * 2, kernel_size, laplacians[1])
        self.conv23 = ConvBlock(96 * 2, 128 * 2, kernel_size, laplacians[1])
        self.conv2_res = Conv1dAuto(128 * 2, 128 * 2, 1)

        # Encoding block 3
        self.conv31 = ConvBlock(128 * 2, 256 * 2, kernel_size, laplacians[2])
        # self.conv32 = ConvBlock(256*2, 256*2, kernel_size, laplacians[2])
        self.conv33 = ConvBlock(256 * 2, 128 * 2, kernel_size, laplacians[2])

        # Decoding block 4
        self.uconv21 = ConvBlock(128 * 2, 128 * 2, kernel_size, laplacians[1])
        self.uconv22 = ConvBlock(128 * 2, 64 * 2, kernel_size, laplacians[1])

        # Decoding block 4
        self.uconv11 = ConvBlock(64 * 2, 64 * 2, kernel_size, laplacians[0])
        self.uconv12 = ConvBlock(64 * 2, 32 * 2, kernel_size, laplacians[0])
        self.uconv13 = ConvCheb(32 * 2, out_channels, kernel_size, laplacians[0])

    def encode(self, x):
        """ Encodes an input into a lower dimensional space applying convolutional, batch normalisation and pooling layers
        Parameters
        ----------
        x : torch.Tensor of shape batch_size x n_vertices x in_channels
            Input data
        Returns
        -------
       x_enc3, x_enc2, x_enc1, idx2, idx1 : torch.Tensors of shapes batch_size x n_vertices x layer_channels + list(int)
            Encoded data at the different encoding stages and the indices indicating the locations of the maxium values in
            unpooled images.
        """
        # x_enc1 = self.dropout1(x_enc1)

        # Block 1

        x_enc1 = self.conv11(x)
        x_enc1 = self.conv13(x_enc1)


        # Block 2
        x_enc2, idx1 = self.pooling(x_enc1)
        x_enc2 = self.conv21(x_enc2)
        x_enc2 = self.conv23(x_enc2)

        # Block 3
        x_enc3, idx2 = self.pooling(x_enc2)
        x_enc3 = self.conv31(x_enc3)
        x_enc3 = self.conv33(x_enc3)

        return x_enc3, x_enc2, x_enc1, idx2, idx1

    def decode(self, x_enc3, x_enc2, x_enc1, idx2, idx1):
        """ Decodes low dimensional data into high dimensional applying convolutional, batch normalisation,
        unpooling layers and skip connections

        Parameters
        ----------
        x_enc3, x_enc2, x_enc1, idx2, idx1 : torch.Tensors of shapes batch_size x n_vertices x layer_channels + list(int)
            Encoded data at the different encoding stages and the indices indicating the locations of the maxium values in
            unpooled images.
        Returns
        -------
        x : torch.Tensor of shape batch_size x n_vertices x out_channels
            Decoded data
        """

        # Block 2
        x = self.unpool(x_enc3, idx2)
        x += torch.transpose(self.conv2_res(torch.transpose(x_enc2, 2, 1)), 2, 1)
        #x = torch.cat((x, x_enc2), dim=2)
        #x *= x_enc2
        x = self.uconv21(x)
        x = self.uconv22(x)

        # Block 1
        x = self.unpool(x, idx1)
        #x = torch.cat((x, x_enc1), dim=2)
        x += torch.transpose(self.conv1_res(torch.transpose(x_enc1, 2, 1)), 2, 1)
        #x += x_enc1
        x = self.uconv11(x)
        x = self.uconv12(x)
        x = self.uconv13(x)
        return x
    
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
        x_encoded = self.encode(x)
        output = self.decode(*x_encoded)
        return output
    
class UNetSphericalHealpixResidual_2(torch.nn.Module):
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
    """

    def __init__(self, N, in_channels, out_channels, kernel_size):        
        super().__init__()

        self.kernel_size = kernel_size
       
        laplacians = []
        for i, nodes in enumerate([3072, 768, 192]):
            laplacian = _compute_laplacian_healpix(nodes)
            laplacians.append(laplacian)
        
        
        # Pooling - unpooling
        self.pooling = PoolMaxHealpix(kernel_size=4)
        self.unpool = UnpoolMaxHealpix(kernel_size=4)
        
        # Encoding block 1
        self.conv11 = ConvBlock(in_channels, max(in_channels, 64), kernel_size, laplacians[0])
        self.conv12 = ConvBlock(max(in_channels, 64), max(in_channels, 64), kernel_size, laplacians[0])
        self.conv13 = ConvCheb(max(in_channels, 64), max(in_channels, 64), kernel_size, laplacians[0])#ConvBlock(max(in_channels, 32), 64, kernel_size, laplacians[0])
        
        self.conv1_res = Conv1dAuto(in_channels, 64, 1)
        
        # Encoding block 2
        self.conv21 = ConvBlock(64, 128, kernel_size, laplacians[1])
        self.conv22 = ConvBlock(128, 128, kernel_size, laplacians[1])
        self.conv23 = ConvCheb(128, 128, kernel_size, laplacians[1])
        
        self.conv2_res = Conv1dAuto(64, 128, 1)
       
        # Encoding block 3
        self.conv31 = ConvBlock(128, 256, kernel_size, laplacians[2])
        self.conv32 = ConvBlock(256, 256, kernel_size, laplacians[2])
        self.conv33 = ConvCheb(256, 128, kernel_size, laplacians[2])
        
        self.conv3_res = Conv1dAuto(128, 128, 1)
        
        # Decoding block 4
        self.uconv21 = ConvBlock(256, 128, kernel_size, laplacians[1])
        self.uconv22 = ConvBlock(128, 64, kernel_size, laplacians[1])
        
        # Decoding block 4
        self.uconv11 = ConvBlock(128, 64, kernel_size, laplacians[0])
        self.uconv12 = ConvBlock(64, 32, kernel_size, laplacians[0])
        self.uconv13 = ConvCheb(32, out_channels, kernel_size, laplacians[0])
        
         
    def encode(self, x):
        """ Encodes an input into a lower dimensional space applying convolutional, batch normalisation and pooling layers
        Parameters
        ----------
        x : torch.Tensor of shape batch_size x n_vertices x in_channels
            Input data
        Returns
        -------
       x_enc3, x_enc2, x_enc1, idx2, idx1 : torch.Tensors of shapes batch_size x n_vertices x layer_channels + list(int)
            Encoded data at the different encoding stages and the indices indicating the locations of the maxium values in
            unpooled images.
        """
        #x_enc1 = self.dropout1(x_enc1)
        
        # Block 1
        
        x_enc1 = self.conv11(x)
        x_enc1 = self.conv12(x_enc1)
        x_enc1 = self.conv13(x_enc1)
        
        x_enc1 += torch.transpose(self.conv1_res(torch.transpose(x, 2,1)), 2,1)
        x_enc1 = F.relu(x_enc1)
        #x_enc1 += self.conv1_res(x.transpose(0,2,1)).transpose(0,2,1)
        
        # Block 2
        x_enc2_ini, idx1 = self.pooling(x_enc1)
        x_enc2 = self.conv21(x_enc2_ini)
        x_enc2 = self.conv22(x_enc2)
        x_enc2 = self.conv23(x_enc2)
        
        x_enc2 += torch.transpose(self.conv2_res(torch.transpose(x_enc2_ini, 2,1)),2,1)
        x_enc2 = F.relu(x_enc2)
        #x_enc2 += self.conv2_res(x_enc1.transpose(0,2,1)).transpose(0,2,1)
        
        # Block 3
        x_enc3_ini, idx2 = self.pooling(x_enc2)
        x_enc3 = self.conv31(x_enc3_ini)
        x_enc3 = self.conv32(x_enc3)
        x_enc3 = self.conv33(x_enc3)
        x_enc3 += torch.transpose(self.conv3_res(torch.transpose(x_enc3_ini, 2,1)),2,1)
        x_enc3 = F.relu(x_enc3)
        #x_enc3 += self.conv3_res(x_enc2.transpose(0,2,1)).transpose(0,2,1)
        
        return x_enc3, x_enc2, x_enc1, idx2, idx1
    
    def decode(self, x_enc3, x_enc2, x_enc1, idx2, idx1):
        """ Decodes low dimensional data into high dimensional applying convolutional, batch normalisation, 
        unpooling layers and skip connections
        
        Parameters
        ----------
        x_enc3, x_enc2, x_enc1, idx2, idx1 : torch.Tensors of shapes batch_size x n_vertices x layer_channels + list(int)
            Encoded data at the different encoding stages and the indices indicating the locations of the maxium values in
            unpooled images.
        Returns
        -------
        x : torch.Tensor of shape batch_size x n_vertices x out_channels
            Decoded data
        """

        # Block 2
        x = self.unpool(x_enc3, idx2)
        x = torch.cat((x, x_enc2), dim=2)
        x = self.uconv21(x)
        x = self.uconv22(x)
        
        # Block 1
        x = self.unpool(x, idx1)
        x = torch.cat((x, x_enc1), dim=2)
        x = self.uconv11(x)
        x = self.uconv12(x)
        x = self.uconv13(x)
            
        return x
    
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
        x_encoded = self.encode(x)
        output = self.decode(*x_encoded)
        return output