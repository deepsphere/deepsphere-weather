import pygsp
import numpy as np

import torch
from torch.nn import functional as F
from torch.nn import BatchNorm1d

from deepsphere.utils.samplings import equiangular_dimension_unpack

from modules import layers
from modules.layers import ConvCheb, Conv2dPeriodic, PoolAvgEquiangular, UnpoolAvgEquiangular



def _compute_laplacian(nodes, ratio, laplacian_type="normalized"):
    dim1, dim2 = equiangular_dimension_unpack(nodes, ratio)
    
    bw = [int(dim1/2), int(dim2/2)]

    G = pygsp.graphs.SphereEquiangular(bandwidth=bw, sampling="SOFT")
    G.compute_laplacian(laplacian_type)
    laplacian = layers.prepare_laplacian(G.L.astype(np.float32))
    
    return laplacian



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
        laplacian = compute_laplacian(N, ratio)
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
        laplacian = compute_laplacian(N, ratio)
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
        x : torch.Tensor of shape batch_size x n_vertices x in_channels
            Input data
        Returns
        -------
        x : torch.Tensor of shape batch_size x n_vertices x out_channels
            Model output
        """
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = self.conv5(x)
        
        return x
    
    
class EncoderDecoderSpherical(torch.nn.Module):
    """Spherical GCNN encoder-decoder.
    
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
        
        for nodes in [2048, 512, 128, 32]:
            laplacian = compute_laplacian(nodes, ratio)
            self.register_buffer(f'laplacian_{nodes}', laplacian)
         
        self.conv1 = ConvCheb(in_channels, 64, kernel_size=self.kernel_size)
        self.conv2 = ConvCheb(64, 128, kernel_size=self.kernel_size)
        self.conv3 = ConvCheb(128, 512, kernel_size=self.kernel_size)

        self.conv4 = ConvCheb(512, 128, kernel_size=self.kernel_size)
        self.conv5 = ConvCheb(128, 64, kernel_size=self.kernel_size)
        self.conv6 = ConvCheb(64, out_channels, kernel_size=self.kernel_size)

        self.pool = PoolAvgEquiangular(ratio, 2)
        self.unpool = UnpoolAvgEquiangular(ratio, 2)
        
        
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

    def encode(self, x):
        """ Encodes an input into a lower dimensional space applying convolutional and pooling layers
        Parameters
        ----------
        x : torch.Tensor of shape batch_size x n_vertices x in_channels
            Input data
        Returns
        -------
        x : torch.Tensor of shape batch_size x n_vertices x 512
            Encoded data
        """
        x = self.pool(F.relu(self.conv1(self.laplacian_2048, x)))
        x = self.pool(F.relu(self.conv2(self.laplacian_512, x)))
        x = self.pool(F.relu(self.conv3(self.laplacian_128, x)))
        
        return x
    

    def decode(self, x):
        """ Decodes low dimensional data into high dimensional applying convolutional and unpooling layers
        Parameters
        ----------
        x : torch.Tensor of shape batch_size x n_vertices x 512
            Encoded data
        Returns
        -------
        x : torch.Tensor of shape batch_size x n_vertices x out_channels
            Decoded data
        """
        x = self.unpool(F.relu(self.conv4(self.laplacian_32, x)))
        x = self.unpool(F.relu(self.conv5(self.laplacian_128, x)))
        x = self.unpool(F.relu(self.conv6(self.laplacian_512, x)))
        
        return x

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
        encoded = self.encode(x)
        output = self.decode(encoded)
        return output
    
    
    
class EncoderDecoder(torch.nn.Module):
    """Spherical GCNN encoder-decoder.
    
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
        
        for nodes in [2048, 512, 128, 32]:
            laplacian = compute_laplacian(nodes, ratio)
            self.register_buffer(f'laplacian_{nodes}', laplacian)
            
            
        self.conv1 = ConvCheb(in_channels, 6, kernel_size=self.kernel_size)
        self.conv2 = ConvCheb(6, 16, kernel_size=self.kernel_size)

        self.conv3 = ConvCheb(16, 6, kernel_size=self.kernel_size)
        self.conv4 = ConvCheb(6, out_channels, kernel_size=self.kernel_size)
    
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
          
            
    def encode(self, x):
        """ Encodes an input into a lower dimensional space applying convolutional and pooling layers
        Parameters
        ----------
        x : torch.Tensor of shape batch_size x n_vertices x in_channels
            Input data
        Returns
        -------
        x : torch.Tensor of shape batch_size x n_vertices x 16
            Encoded data
        """
        x = F.relu(self.conv1(self.laplacian_2048, x))
        x = F.relu(self.conv2(self.laplacian_2048, x))
        return x
    

    def decode(self, x):
        """ Decodes low dimensional data into high dimensional applying convolutional and unpooling layers
        Parameters
        ----------
        x : torch.Tensor of shape batch_size x n_vertices x 16
            Encoded data
        Returns
        -------
        x : torch.Tensor of shape batch_size x n_vertices x out_channels
            Decoded data
        """
        x = F.relu(self.conv3(self.laplacian_2048, x))
        x = F.relu(self.conv4(self.laplacian_2048, x))
        return x
    
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
        encoded = self.encode(x)
        output = self.decode(encoded)
        return output
    
    
    
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
            laplacian = compute_laplacian(nodes, ratio)
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
            laplacian = compute_laplacian(nodes, ratio)
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
        
        x = self.conv2_dec_l3(self.laplacian_3, x
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