import pygsp
import numpy as np

import torch
from torch.nn import functional as F
from torch.nn import BatchNorm1d, BatchNorm2d

from deepsphere.utils.samplings import equiangular_dimension_unpack

from modules import layers
from modules.layers import (ConvCheb, Conv2dPeriodic, PoolAvgEquiangular, UnpoolAvgEquiangular, PoolMaxEquiangular,
                            UnpoolMaxEquiangular, PoolMaxHealpix, UnpoolMaxHealpix, PoolAvgHealpix, UnpoolAvgHealpix,
                            ConvChebTemp, PoolAvgTempHealpix, UnpoolAvgTempHealpix)



def _compute_laplacian_equiangular(nodes, ratio, laplacian_type="normalized"):
    """ Computes laplacian of spherical graph sampled as a HEALpix grid 
    
    Parameters
    ----------
    nodes : int
        Number of nodes in the graph
    ratio : float
        width / height
    laplacian_type : string
        Type of laplacian. Options are {´normalized´, ´combinatorial´}
        
    Returns
    -------
    laplacian : torch.sparse_coo_tensor
        Graph laplacian
    """
    dim1, dim2 = equiangular_dimension_unpack(nodes, ratio)
    
    bw = [int(dim1/2), int(dim2/2)]

    G = pygsp.graphs.SphereEquiangular(bandwidth=bw, sampling="SOFT")
    G.compute_laplacian(laplacian_type)
    laplacian = layers.prepare_laplacian(G.L.astype(np.float32))
    
    return laplacian


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


# 2D Models
class CNN2dPeriodic(torch.nn.Module):
    """Fully convolutional CNN with periodic padding on the longitude dimension.
    WeatherBench CNN architecture.
    
     Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels in the output image.
    kernel_size : int
        Width of the square kernel. Actual kernel size is kernel_size**2.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        
        self.conv1 = Conv2dPeriodic(in_channels, 64, kernel_size)
        self.conv2 = Conv2dPeriodic(64, 64, kernel_size)
        self.conv3 = Conv2dPeriodic(64, 64, kernel_size)
        self.conv4 = Conv2dPeriodic(64, 64, kernel_size)
        self.conv5 = Conv2dPeriodic(64, out_channels, kernel_size)
        
    def forward(self, x):
        """Forward Pass
        Parameters
        ----------
        x : torch.Tensor of shape batch_size x lat x lon x in_channels
            Input data
        Returns
        -------
        x : torch.Tensor of shape batch_size x lat x lon x out_channels
            Model output
        """
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = self.conv5(x)
        
        return x

class CNN2d(torch.nn.Module):
    """Regular CNN 2D
    
    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels in the output image.
    kernel_size : int
        Width of the square kernel. Actual kernel size is kernel_size**2.
    """

    def __init__(self, channels_in, channels_out, kernel_size):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        self.conv1 = torch.nn.Conv2d(channels_in, 64, self.kernel_size, padding=int((self.kernel_size - 1)/2))
        self.conv2 = torch.nn.Conv2d(64, 64, self.kernel_size, padding=int((self.kernel_size - 1)/2))
        self.conv3 = torch.nn.Conv2d(64, 64, self.kernel_size, padding=int((self.kernel_size - 1)/2))
        self.conv4 = torch.nn.Conv2d(64, 64, self.kernel_size, padding=int((self.kernel_size - 1)/2))
        self.conv5 = torch.nn.Conv2d(64, channels_out, self.kernel_size, padding=int((self.kernel_size - 1)/2))

    def forward(self, x):
        """Forward Pass
        Parameters
        ----------
        x : torch.Tensor of shape batch_size x lat x lon x in_channels
            Input data
        Returns
        -------
        x : torch.Tensor of shape batch_size x lat x lon x out_channels
            Model output
        """
        
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = self.conv5(x)
        
        return x


# Spherical GCNN models
class CNNSpherical(torch.nn.Module):
    """Spherical GCNN with WeatherBench CNN architecture
    
    Parameters
    ----------
    N : int
        Number of nodes in the input graph
    ratio : float
        Parameter for equiangular sampling -> width/height
    in_channels : int
        Number of channels in the input graph.
    out_channels : int
        Number of channels in the output graph.
    kernel_size : int
        Chebychev polynomial degree
    """

    def __init__(self, N, ratio, in_channels, out_channels, kernel_size):
        super().__init__()
        
        self.kernel_size = kernel_size
        laplacian = _compute_laplacian_equiangular(N, ratio)
        self.register_buffer(f'laplacian', laplacian)
        
        self.conv1 = ConvCheb(in_channels, 64, self.kernel_size)
        self.conv2 = ConvCheb(64, 64, self.kernel_size)
        self.conv3 = ConvCheb(64, 64, self.kernel_size)
        self.conv4 = ConvCheb(64, 64, self.kernel_size)
        self.conv5 = ConvCheb(64, out_channels, self.kernel_size)
        
        for m in self.modules():
            if isinstance(m, ConvCheb):
                m.reset_parameters(activation='linear', fan='avg', distribution='uniform')
                
    def state_dict(self, *args, **kwargs):
        """ This function overrides the state dict in order to be able to save the model.
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
    
        x = F.elu(self.conv1(self.laplacian, x))
        x = F.elu(self.conv2(self.laplacian, x))
        x = F.elu(self.conv3(self.laplacian, x))
        x = F.elu(self.conv4(self.laplacian, x))
        x = self.conv5(self.laplacian, x)
        
        return x
    
    
class CNNSphericalBatchNorm(torch.nn.Module):
    """Spherical GCNN with WeatherBench CNN architecture
   
    
    Parameters
    ----------
    N : int
        Number of nodes in the input graph
    ratio : float
        Parameter for equiangular sampling -> width/height
    in_channels : int
        Number of channels in the input graph.
    out_channels : int
        Number of channels in the output graph.
    kernel_size : int
        Chebychev polynomial degree
    """

    def __init__(self, N, ratio, in_channels, out_channels, kernel_size):
        super().__init__()
        
        self.kernel_size = kernel_size
        laplacian = _compute_laplacian_equiangular(N, ratio)
        self.register_buffer(f'laplacian', laplacian)
        
        self.conv1 = ConvCheb(in_channels, 64, self.kernel_size)
        self.bn1 = BatchNorm1d(64)
        self.conv2 = ConvCheb(64, 64, self.kernel_size)
        self.bn2 = BatchNorm1d(64)
        self.conv3 = ConvCheb(64, 64, self.kernel_size)
        self.bn3 = BatchNorm1d(64)
        self.conv4 = ConvCheb(64, 64, self.kernel_size)
        self.bn4 = BatchNorm1d(64)
        self.conv5 = ConvCheb(64, out_channels, self.kernel_size)
        
        for m in self.modules():
            if isinstance(m, ConvCheb):
                m.reset_parameters(activation='linear', fan='avg', distribution='uniform')
                
    def state_dict(self, *args, **kwargs):
        """ This function overrides the state dict in order to be able to save the model.
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
    
        x = F.elu(self.conv1(self.laplacian, x))
        x = self.bn1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.elu(self.conv2(self.laplacian, x))
        x = self.bn2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.elu(self.conv3(self.laplacian, x))
        x = self.bn3(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.elu(self.conv4(self.laplacian, x))
        x = self.bn4(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.conv5(self.laplacian, x)
        
        return x

    
class UNetSpherical(torch.nn.Module):
    """Spherical GCNN UNet
    
     Parameters
    ----------
    N : int
        Number of nodes in the input graph
    ratio : float
        Parameter for equiangular sampling -> width/height
    in_channels : int
        Number of channels in the input graph.
    out_channels : int
        Number of channels in the output graph.
    kernel_size : int
        Chebychev polynomial degree
    """

    def __init__(self, N, ratio, in_channels, out_channels, kernel_size):        
        super().__init__()
        
        self.ratio = ratio
        self.kernel_size = kernel_size
       
        for i, nodes in enumerate([8, 32, 128, 512, 2048]):
            laplacian = _compute_laplacian_equiangular(nodes, ratio)
            self.register_buffer(f'laplacian_{i+1}', laplacian)
        
        
        # Pooling - unpooling
        self.pooling = PoolAvgEquiangular(ratio, 2)
        self.unpool = UnpoolAvgEquiangular(ratio, 2)
        
        
        # Encoding block 1
        self.conv1_enc_l5 = ConvCheb(in_channels, 32, self.kernel_size)
        self.bn1_enc_l5 = BatchNorm1d(32)
        self.conv2_enc_l5 = ConvCheb(32, 64, self.kernel_size)
        self.bn2_enc_l5 = BatchNorm1d(64)
        
        # Encoding block 2
        self.conv_enc_l4 = ConvCheb(64, 128, self.kernel_size)
        self.bn_enc_l4 = BatchNorm1d(128)
        
        # Encoding block 3
        self.conv_enc_l3 = ConvCheb(128, 256, self.kernel_size)
        self.bn_enc_l3 = BatchNorm1d(256)
        
        # Encoding block 4
        self.conv_enc_l2 = ConvCheb(256, 512, self.kernel_size)
        self.bn_enc_l2 = BatchNorm1d(512)
        
        # Encoding block 5
        self.conv_enc_l1 = ConvCheb(512, 512, self.kernel_size)
        self.bn_enc_l1 = BatchNorm1d(512)
        
        # Encoding block 6
        self.conv_enc_l0 = ConvCheb(512, 512, self.kernel_size)
        

        
        # Decoding block 1
        self.conv1_dec_l1 = ConvCheb(512, 512, self.kernel_size)
        self.bn1_dec_l1 = BatchNorm1d(512)
        
        self.conv2_dec_l1 = ConvCheb(512+512, 512, self.kernel_size)
        self.bn2_dec_l1 = BatchNorm1d(512)
        
        # Decoding block 2
        self.conv1_dec_l2 = ConvCheb(512, 256, self.kernel_size)
        self.bn1_dec_l2 = BatchNorm1d(256)
        
        self.conv2_dec_l2 = ConvCheb(512+256, 256, self.kernel_size)
        self.bn2_dec_l2 = BatchNorm1d(256)
        
        # Decoding block 3
        self.conv1_dec_l3 = ConvCheb(256, 128, self.kernel_size)
        self.bn1_dec_l3 = BatchNorm1d(128)
        
        self.conv2_dec_l3 = ConvCheb(256+128, 128, self.kernel_size)
        self.bn2_dec_l3 = BatchNorm1d(128)
        
        # Decoding block 4
        self.conv1_dec_l4 = ConvCheb(128, 64, self.kernel_size)
        self.bn1_dec_l4 = BatchNorm1d(64)
        
        self.conv2_dec_l4 = ConvCheb(128+64, 64, self.kernel_size)
        self.bn2_dec_l4 = torch.nn.BatchNorm1d(64)
        
        # Decoding block 5
        self.conv1_dec_l5 = ConvCheb(64, 32, self.kernel_size)
        self.bn_dec_l5 = BatchNorm1d(32)
        
        self.conv2_dec_l5 = ConvCheb(32, out_channels, self.kernel_size)
        
        
    def encode(self, x):
        """ Encodes an input into a lower dimensional space applying convolutional, batch normalisation and pooling layers
        Parameters
        ----------
        x : torch.Tensor of shape batch_size x n_vertices x in_channels
            Input data
        Returns
        -------
        x_enc0, x_enc1, x_enc2, x_enc3, x_enc4 : torch.Tensors of shapes batch_size x n_vertices x layer_channels
            Encoded data at the different encoding stages
        """
        
        # Block 1
        x_enc5 = self.conv1_enc_l5(self.laplacian_5, x)
        x_enc5 = self.bn1_enc_l5(x_enc5.permute(0, 2, 1)).permute(0, 2, 1)
        x_enc5 = F.relu(x_enc5)
        
        x_enc5 = self.conv2_enc_l5(self.laplacian_5, x_enc5)
        x_enc5 = self.bn2_enc_l5(x_enc5.permute(0, 2, 1)).permute(0, 2, 1)
        x_enc5 = F.relu(x_enc5)
        
        # Block 2
        x_enc4 = self.pooling(x_enc5)
        x_enc4 = self.conv_enc_l4(self.laplacian_4, x_enc4)
        x_enc4 = self.bn_enc_l4(x_enc4.permute(0, 2, 1)).permute(0, 2, 1)
        x_enc4 = F.relu(x_enc4)
        
        # Block 3
        x_enc3 = self.pooling(x_enc4)
        x_enc3 = self.conv_enc_l3(self.laplacian_3, x_enc3)
        x_enc3 = self.bn_enc_l3(x_enc3.permute(0, 2, 1)).permute(0, 2, 1)
        x_enc3 = F.relu(x_enc3)
        
        # Block 4
        x_enc2 = self.pooling(x_enc3)
        x_enc2 = self.conv_enc_l2(self.laplacian_2, x_enc2)
        x_enc2 = self.bn_enc_l2(x_enc2.permute(0, 2, 1)).permute(0, 2, 1)
        x_enc2 = F.relu(x_enc2)
        
        # Block 5
        x_enc1 = self.pooling(x_enc2)
        x_enc1 = self.conv_enc_l1(self.laplacian_1, x_enc1)
        x_enc1 = self.bn_enc_l1(x_enc1.permute(0, 2, 1)).permute(0, 2, 1)
        x_enc1 = F.relu(x_enc1)
        
        # Block 6
        x_enc0 = self.conv_enc_l0(self.laplacian_1, x_enc1)

        return x_enc0, x_enc1, x_enc2, x_enc3, x_enc4
    
    def decode(self, x_enc0, x_enc1, x_enc2, x_enc3, x_enc4):
        """ Decodes low dimensional data into high dimensional applying convolutional, batch normalisation, 
        unpooling layers and skip connections
        
        Parameters
        ----------
        x_enc0, x_enc1, x_enc2, x_enc3, x_enc4 : torch.Tensors of shapes batch_size x n_vertices x layer_channels
            Encoded data at the different encoding stages
        Returns
        -------
        x : torch.Tensor of shape batch_size x n_vertices x out_channels
            Decoded data
        """
        # Block 1
        x = self.conv1_dec_l1(self.laplacian_1, x_enc0)
        x = self.bn1_dec_l1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        x = torch.cat((x, x_enc1), dim=2)
        
        x = self.conv2_dec_l1(self.laplacian_1, x)
        x = self.bn2_dec_l1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        
        # Block 2
        x = self.unpool(x)
        x = self.conv1_dec_l2(self.laplacian_2, x)
        x = self.bn1_dec_l2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        x = torch.cat((x, x_enc2), dim=2)
        
        x = self.conv2_dec_l2(self.laplacian_2, x)
        x = self.bn2_dec_l2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        
        # Block 3
        x = self.unpool(x)
        x = self.conv1_dec_l3(self.laplacian_3, x)
        x = self.bn1_dec_l3(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        x = torch.cat((x, x_enc3), dim=2)
        
        x = self.conv2_dec_l3(self.laplacian_3, x)
        x = self.bn2_dec_l3(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        
        # Block 4
        x = self.unpool(x)
        x = self.conv1_dec_l4(self.laplacian_4, x)
        x = self.bn1_dec_l4(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        x = torch.cat((x, x_enc4), dim=2)
        
        x = self.conv2_dec_l4(self.laplacian_4, x)
        x = self.bn2_dec_l4(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        
        # Block 5
        x = self.unpool(x)
        x = self.conv1_dec_l5(self.laplacian_5, x)
        x = self.bn_dec_l5(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.conv2_dec_l5(self.laplacian_5, x)
            
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

    
    
class UNetNoBNSpherical(torch.nn.Module):
    """Spherical GCNN UNet without batch normalisation
    
    Parameters
    ----------
    N : int 
        Number of pixels in the input image
    ratio : float
        Parameter for equiangular sampling -> width/height
    in_channels : int 
        Number of input features
    out_channels : int
        Number of output features
    kernel_size : int 
        Chebychev polynomial degree
    """

    def __init__(self, N, ratio, in_channels, out_channels, kernel_size):
        super().__init__()
        
        self.ratio = ratio
        self.kernel_size = kernel_size
       
        for i, nodes in enumerate([8, 32, 128, 512, 2048]):
            laplacian = _compute_laplacian(nodes, ratio)
            self.register_buffer(f'laplacian_{i+1}', laplacian)
        
        # Pooling - unpooling
        self.pooling = PoolAvgEquiangular(ratio, 2)
        self.unpool = UnpoolAvgEquiangular(ratio, 2)
      
    
        # Encoding block 1
        self.conv1_enc_l5 = ConvCheb(in_channels, 32, self.kernel_size) 
        self.conv2_enc_l5 = ConvCheb(32, 64, self.kernel_size)

        # Encoding block 2
        self.conv_enc_l4 = ConvCheb(64, 128, self.kernel_size)
        
        # Encoding block 3
        self.conv_enc_l3 = ConvCheb(128, 256, self.kernel_size)
        
        # Encoding block 4
        self.conv_enc_l2 = ConvCheb(256, 512, self.kernel_size)
        
        # Encoding block 5
        self.conv_enc_l1 = ConvCheb(512, 512, self.kernel_size)
        
        # Encoding block 6
        self.conv_enc_l0 = ConvCheb(512, 512, self.kernel_size)
        

        
        # Decoding block 1
        self.conv1_dec_l1 = ConvCheb(512, 512, self.kernel_size)
        
        self.conv2_dec_l1 = ConvCheb(512+512, 512, self.kernel_size)
        
        # Decoding block 2
        self.conv1_dec_l2 = ConvCheb(512, 256, self.kernel_size)
        
        self.conv2_dec_l2 = ConvCheb(512+256, 256, self.kernel_size)
        
        # Decoding block 3
        self.conv1_dec_l3 = ConvCheb(256, 128, self.kernel_size)
        
        self.conv2_dec_l3 = ConvCheb(256+128, 128, self.kernel_size)
        
        # Decoding block 4
        self.conv1_dec_l4 = ConvCheb(128, 64, self.kernel_size)
        
        self.conv2_dec_l4 = ConvCheb(128+64, 64, self.kernel_size)
        
        # Decoding block 5
        self.conv1_dec_l5 = ConvCheb(64, 32, self.kernel_size)
        self.conv2_dec_l5 = ConvCheb(32, out_channels, self.kernel_size)
        
        
    def encode(self, x):
        """ Encodes an input into a lower dimensional space applying convolutional and pooling layers
        Parameters
        ----------
        x : torch.Tensor of shape batch_size x n_vertices x in_channels
            Input data
        Returns
        -------
        x_enc0, x_enc1, x_enc2, x_enc3, x_enc4 : torch.Tensors of shapes batch_size x n_vertices x layer_channels
            Encoded data at the different encoding stages
        """
        # Block 1
        x_enc5 = self.conv1_enc_l5(self.laplacian_5, x)
        x_enc5 = F.relu(x_enc5)
        x_enc5 = self.conv2_enc_l5(self.laplacian_5, x_enc5)
        x_enc5 = F.relu(x_enc5)
        
        # Block 2
        x_enc4 = self.pooling(x_enc5)
        x_enc4 = self.conv_enc_l4(self.laplacian_4, x_enc4)
        x_enc4 = F.relu(x_enc4)
        
        # Block 3
        x_enc3 = self.pooling(x_enc4)
        x_enc3 = self.conv_enc_l3(self.laplacian_3, x_enc3)
        x_enc3 = F.relu(x_enc3)
        
        # Block 4
        x_enc2 = self.pooling(x_enc3)
        x_enc2 = self.conv_enc_l2(self.laplacian_2, x_enc2)
        x_enc2 = F.relu(x_enc2)
        
        # Block 5
        x_enc1 = self.pooling(x_enc2)
        x_enc1 = self.conv_enc_l1(self.laplacian_1, x_enc1)
        x_enc1 = F.relu(x_enc1)
        
        # Block 6
        x_enc0 = self.conv_enc_l0(self.laplacian_1, x_enc1)

        return x_enc0, x_enc1, x_enc2, x_enc3, x_enc4
    
    def decode(self, x_enc0, x_enc1, x_enc2, x_enc3, x_enc4):
        """ Decodes low dimensional data into high dimensional applying convolutional and unpooling layers and skip connections
        Parameters
        ----------
        x_enc0, x_enc1, x_enc2, x_enc3, x_enc4 : torch.Tensors of shapes batch_size x n_vertices x layer_channels
            Encoded data at the different encoding stages
        Returns
        -------
        x : torch.Tensor of shape batch_size x n_vertices x out_channels
            Decoded data
        """
        # Block 1
        x = self.conv1_dec_l1(self.laplacian_1, x_enc0)
        x = F.relu(x)
        x = torch.cat((x, x_enc1), dim=2)
        x = self.conv2_dec_l1(self.laplacian_1, x)
        x = F.relu(x)
        
        # Block 2
        x = self.unpool(x)
        x = self.conv1_dec_l2(self.laplacian_2, x)
        x = F.relu(x)
        x = torch.cat((x, x_enc2), dim=2)
        
        x = self.conv2_dec_l2(self.laplacian_2, x)
        x = F.relu(x)
        
        # Block 3
        x = self.unpool(x)
        x = self.conv1_dec_l3(self.laplacian_3, x)
        x = F.relu(x)
        x = torch.cat((x, x_enc3), dim=2)
        
        x = self.conv2_dec_l3(self.laplacian_3, x)
        x = F.relu(x)
        
        # Block 4
        x = self.unpool(x)
        x = self.conv1_dec_l4(self.laplacian_4, x)
        x = F.relu(x)
        x = torch.cat((x, x_enc4), dim=2)
        
        x = self.conv2_dec_l4(self.laplacian_4, x)
        x = F.relu(x)
        
        # Block 5
        x = self.unpool(x)
        x = self.conv1_dec_l5(self.laplacian_5, x)
        x = self.conv2_dec_l5(self.laplacian_5, x)
            
        return x
    
    def state_dict(self, *args, **kwargs):
        """ This function overrides the state dict in order to be able to save the model.
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
    
########## TESTING MODEL ############


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, laplacian):
        super().__init__()

        self.conv = ConvCheb(in_channels, out_channels, kernel_size, laplacian)
        self.bn = BatchNorm1d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        return x

    
class UNetSphericalTest(torch.nn.Module):
    """Spherical GCNN UNet
    
     Parameters
    ----------
    N : int
        Number of nodes in the input graph
    ratio : float
        Parameter for equiangular sampling -> width/height
    in_channels : int
        Number of channels in the input graph.
    out_channels : int
        Number of channels in the output graph.
    kernel_size : int
        Chebychev polynomial degree
    """

    def __init__(self, N, ratio, in_channels, out_channels, kernel_size):        
        super().__init__()
        
        self.ratio = ratio
        self.kernel_size = kernel_size
       
        laplacians = []
        for i, nodes in enumerate([2048, 512, 128]):
            laplacian = _compute_laplacian_equiangular(nodes, ratio)
            laplacians.append(laplacian)
        
        
        # Pooling - unpooling
        self.pooling = PoolAvgEquiangular(ratio=ratio, kernel_size=2)
        self.unpool = UnpoolAvgEquiangular(ratio=ratio, kernel_size=2)
        
        # Encoding block 1
        self.conv11 = ConvBlock(in_channels, 6, kernel_size, laplacians[0])
        self.conv12 = ConvBlock(6, 11, kernel_size, laplacians[0])
        self.conv13 = ConvBlock(11, 16, kernel_size, laplacians[0])

        # Encoding block 2
        self.conv21 = ConvBlock(16, 22, kernel_size, laplacians[1])
        self.conv22 = ConvBlock(22, 27, kernel_size, laplacians[1])
        self.conv23 = ConvBlock(27, 32, kernel_size, laplacians[1])

        # Encoding block 3
        self.conv31 = ConvBlock(32, 64, kernel_size, laplacians[2])
        self.conv32 = ConvBlock(64, 32, kernel_size, laplacians[2])

        # Decoding block 2
        self.uconv21 = ConvBlock(64, 32, kernel_size, laplacians[1])
        self.uconv22 = ConvBlock(32, 16, kernel_size, laplacians[1])

        # Decoding block 1
        self.uconv11 = ConvBlock(32, 16, kernel_size, laplacians[0])
        self.uconv12 = ConvBlock(16, out_channels, kernel_size, laplacians[0])
        self.uconv13 = ConvCheb(out_channels, out_channels, kernel_size, laplacians[0])
        
         
    def encode(self, x):
        """ Encodes an input into a lower dimensional space applying convolutional, batch normalisation and pooling layers
        Parameters
        ----------
        x : torch.Tensor of shape batch_size x n_vertices x in_channels
            Input data
        Returns
        -------
        x_enc3, x_enc2, x_enc1 : torch.Tensors of shapes batch_size x n_vertices x layer_channels
            Encoded data at the different encoding stages
        """
        #x_enc1 = self.dropout1(x_enc1)
        
        # Block 1
        x_enc1 = self.conv11(x)
        x_enc1 = self.conv12(x_enc1)
        x_enc1 = self.conv13(x_enc1)
        
        # Block 2
        x_enc2 = self.pooling(x_enc1)
        x_enc2 = self.conv21(x_enc2)
        x_enc2 = self.conv22(x_enc2)
        x_enc2 = self.conv23(x_enc2)
        
        # Block 3
        x_enc3 = self.pooling(x_enc2)
        x_enc3 = self.conv31(x_enc3)
        x_enc3 = self.conv32(x_enc3)
        #x_enc3 = self.conv33(x_enc3)
        
        return x_enc3, x_enc2, x_enc1
    
    def decode(self, x_enc3, x_enc2, x_enc1):
        """ Decodes low dimensional data into high dimensional applying convolutional, batch normalisation, 
        unpooling layers and skip connections
        
        Parameters
        ----------
        x_enc3, x_enc2, x_enc1 : torch.Tensors of shapes batch_size x n_vertices x layer_channels
            Encoded data at the different encoding stages
        Returns
        -------
        x : torch.Tensor of shape batch_size x n_vertices x out_channels
            Decoded data
        """
        
        
        # Block 2
        #x = self.convT2(x_enc3)
        x = self.unpool(x_enc3)
        x = torch.cat((x, x_enc2), dim=2)
        x = self.uconv21(x)
        x = self.uconv22(x)
        
        # Block 1
        #x = self.convT1(x)
        x = self.unpool(x)
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
    
    
class UNetSphericalTestMax(torch.nn.Module):
    """Spherical GCNN UNet
    
     Parameters
    ----------
    N : int
        Number of nodes in the input graph
    ratio : float
        Parameter for equiangular sampling -> width/height
    in_channels : int
        Number of channels in the input graph.
    out_channels : int
        Number of channels in the output graph.
    kernel_size : int
        Chebychev polynomial degree
    """

    def __init__(self, N, ratio, in_channels, out_channels, kernel_size):        
        super().__init__()
        
        self.ratio = ratio
        self.kernel_size = kernel_size
       
        laplacians = []
        for i, nodes in enumerate([2048, 512, 128]):
            laplacian = _compute_laplacian_equiangular(nodes, ratio)
            laplacians.append(laplacian)
        
        
        # Pooling - unpooling
        self.pooling = PoolMaxEquiangular(ratio=ratio, kernel_size=2)
        self.unpool = UnpoolMaxEquiangular(ratio=ratio, kernel_size=2)
        
        
        # Encoding block 1
        self.conv11 = ConvBlock(in_channels, 16, kernel_size, laplacians[0])
        self.conv12 = ConvBlock(16, 32, kernel_size, laplacians[0])
        self.conv13 = ConvBlock(32, 64, kernel_size, laplacians[0])
        
        # Encoding block 2
        self.conv21 = ConvBlock(64, 88, kernel_size, laplacians[1])
        self.conv22 = ConvBlock(88, 110, kernel_size, laplacians[1])
        self.conv23 = ConvBlock(110, 128, kernel_size, laplacians[1])
       
        # Encoding block 3
        self.conv31 = ConvBlock(128, 174, kernel_size, laplacians[2])
        self.conv32 = ConvBlock(174, 218, kernel_size, laplacians[2])
        self.conv33 = ConvBlock(218, 256, kernel_size, laplacians[2])
        
        # Decoding block 4
        self.convT2 = ConvBlock(256, 128, kernel_size, laplacians[2])
        self.uconv21 = ConvBlock(256, 128, kernel_size, laplacians[1])
        self.uconv22 = ConvBlock(128, 128, kernel_size, laplacians[1])
        
        # Decoding block 4
        self.convT1 = ConvBlock(128, 64, kernel_size, laplacians[1])
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
            Encoded data at the different encoding stages and the indices indicating the locations of the maxium values in unpooled images.
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
            Encoded data at the different encoding stages and the indices indicating the locations of the maxium values in unpooled images.
        Returns
        -------
        x : torch.Tensor of shape batch_size x n_vertices x out_channels
            Decoded data
        """

        # Block 2
        x = self.convT2(x_enc3)
        x = self.unpool(x, idx2)
        x = torch.cat((x, x_enc2), dim=2)
        x = self.uconv21(x)
        x = self.uconv22(x)
        
        # Block 1
        x = self.convT1(x)
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