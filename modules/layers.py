import math

import numpy as np
from scipy import sparse
import scipy.sparse.linalg
import torch
from torch.nn import functional as F

from deepsphere.utils.samplings import equiangular_dimension_unpack
from deepsphere.layers.samplings.equiangular_pool_unpool import reformat

# 2D CNN layers
class Conv2dPeriodic(torch.nn.Module):
    """ 2D Convolutional layer, periodic in the longitude (width) dimension.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels in the output image.
    kernel_size : int
        Width of the square convolutional kernel.
        The actual size of the kernel is kernel_size*+2
    """
    
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.pad_width = int((self.kernel_size - 1)/2)
        
        self.conv = torch.nn.Conv2d(in_channels, out_channels, self.kernel_size, padding=0)
        
        torch.nn.init.xavier_uniform_(self.conv.weight)
        torch.nn.init.zeros_(self.conv.bias)
    
    def pad(self, x):
        padded = torch.cat((x[:, :, :, -self.pad_width:], x, x[:, :, :, :self.pad_width]), dim=3)
        padded = F.pad(padded, (0, 0, self.pad_width, self.pad_width), 'constant', 0)
        
        return padded
    
    def forward(self, x):
        padded = self.pad(x)
        output = self.conv(padded)
        
        return output


# Graph CNN layers
"""
PyTorch implementation of a convolutional neural network on graphs based on
Chebyshev polynomials of the graph Laplacian.
See https://arxiv.org/abs/1606.09375 for details.
Copyright 2018 Michaël Defferrard.
Released under the terms of the MIT license.
"""


def prepare_laplacian(laplacian):
    """Prepare a graph Laplacian to be fed to a graph convolutional layer
    """

    def estimate_lmax(laplacian, tol=5e-3):
        r"""Estimate the largest eigenvalue of an operator."""
        lmax = sparse.linalg.eigsh(laplacian, k=1, tol=tol,
                                   ncv=min(laplacian.shape[0], 10),
                                   return_eigenvectors=False)
        lmax = lmax[0]
        lmax *= 1 + 2*tol  # Be robust to errors.
        return lmax

    def scale_operator(L, lmax, scale=1):
        r"""Scale the eigenvalues from [0, lmax] to [-scale, scale]."""
        I = sparse.identity(L.shape[0], format=L.format, dtype=L.dtype)
        L *= 2 * scale / lmax
        L -= I
        return L

    lmax = estimate_lmax(laplacian)
    laplacian = scale_operator(laplacian, lmax)

    laplacian = sparse.coo_matrix(laplacian)

    # PyTorch wants a LongTensor (int64) as indices (it'll otherwise convert).
    indices = np.empty((2, laplacian.nnz), dtype=np.int64)
    np.stack((laplacian.row, laplacian.col), axis=0, out=indices)
    indices = torch.from_numpy(indices)

    laplacian = torch.sparse_coo_tensor(indices, laplacian.data, laplacian.shape)
    laplacian = laplacian.coalesce()  # More efficient subsequent operations.
    return laplacian


def cheb_conv(laplacian, inputs, weight):
    """Chebyshev convolution.
    Parameters
    ----------
    laplacian : torch.sparse.Tensor
        The laplacian corresponding to the current sampling of the sphere
    inputs : torch.Tensor
        The current input data being forwarded
    weight : torch.Tensor
        The weights of the current layer
        
    Returns
    -------
    x : torch.Tensor
        Inputs after applying Chebyshev convolution.
    """
    
    B, V, Fin1 = inputs.shape
    #print('B: {}, V. {}, Fin: {}'.format(B,V,Fin1))
    Fin, K, Fout = weight.shape
    #print('Fin: {}, K: {}, Fout: {}'.format(Fin, K, Fout))
    assert Fin1 == Fin
    # B = batch size
    # V = nb vertices
    # Fin = nb input features
    # Fout = nb output features
    # K = order of Chebyshev polynomials (kenel size)
    
    # transform to Chebyshev basis    
    x0 = inputs.permute(1, 2, 0).contiguous()  # V x Fin x B
    x0 = x0.view([V, Fin*B])              # V x Fin*B
    x = x0.unsqueeze(0)                   # 1 x V x Fin*B

    if K > 1:
        x1 = torch.sparse.mm(laplacian, x0)     # V x Fin*B
        x = torch.cat((x, x1.unsqueeze(0)), 0)  # 2 x V x Fin*B
    for _ in range(2, K):
        x2 = 2 * torch.sparse.mm(laplacian, x1) - x0
        x = torch.cat((x, x2.unsqueeze(0)), 0)  # M x Fin*B
        x0, x1 = x1, x2

    x = x.view([K, V, Fin, B])              # K x V x Fin x B
    x = x.permute(3, 1, 2, 0).contiguous()  # B x V x Fin x K
    x = x.view([B*V, Fin*K])                # B*V x Fin*K

    # Linearly compose Fin features to get Fout features
    weight = weight.view(Fin*K, Fout)
    x = x.matmul(weight)      # B*V x Fout
    x = x.view([B, V, Fout])  # B x V x Fout

    return x


def cheb_conv_temp(laplacian, inputs, weight):
    """Chebyshev convolution.
    Parameters
    ----------
    laplacian : torch.sparse.Tensor
        The laplacian corresponding to the current sampling of the sphere
    inputs : torch.Tensor
        The current input data being forwarded
    weight : torch.Tensor
        The weights of the current layer
        
    Returns
    -------
    x : torch.Tensor of shape [batch_size x num_vertex x in_channels]
        Inputs after applying Chebyshev convolution.
    """
    B, V, T, Fin = inputs.shape
    Fin, Kv, Kt, Fout = weight.shape
    # B = batch size
    # V = nb vertices
    # T = nb time steps
    # Fin = nb input features
    # Fout = nb output features
    # Kv = order of Chebyshev polynomials (spatial kernel width)
    # Kt = Temporal kernel width
    
    # transform to Chebyshev basis    
    x0 = inputs.permute(1, 2, 3, 0).contiguous()  # V x T x Fin x B
    x0 = x0.view([V, T*Fin*B])              # V x T*Fin*B
    x = x0.unsqueeze(0)                     # 1 x V x T*Fin*B

    if Kv > 1:
        x1 = torch.sparse.mm(laplacian, x0)     # V x T*Fin*B
        x = torch.cat((x, x1.unsqueeze(0)), 0)  # 2 x V x T*Fin*B
    for _ in range(2, Kv):
        x2 = 2 * torch.sparse.mm(laplacian, x1) - x0
        x = torch.cat((x, x2.unsqueeze(0)), 0)  # M x T*Fin*B
        x0, x1 = x1, x2
    
    x = x.view([Kv, V, T, Fin, B])              # Kv x V x T x Fin x B
    x = x.permute(4, 1, 3, 0, 2).contiguous()   # B x V x Fin x Kv x T
    x = x.view([B*V, Fin*Kv*T])                 # B*V x Fin*K*T
    
    # Linearly compose Fin features to get Fout features
    weight = weight.view(Fin*Kv*Kt, Fout)
    x = x.matmul(weight)      # B*V x Fout
    x = x.view([B, V, Fout])  # B x V x Fout

    return x



class ConvCheb(torch.nn.Module):
    """Graph convolutional layer.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input graph.
    out_channels : int
        Number of channels in the output graph.
    kernel_size : int
        Number of trainable parameters per filter, which is also the size of
        the convolutional kernel.
        The order of the Chebyshev polynomials is kernel_size - 1.

        * A kernel_size of 1 won't take the neighborhood into account.
        * A kernel_size of 2 will look up to the 1-neighborhood (1 hop away).
        * A kernel_size of 3 will look up to the 2-neighborhood (2 hops away).

        A kernel_size of 0 is equivalent to not having a graph (or an empty
        adjacency matrix). All the vertices are treated independently and form
        a set. Every element of that set is given to a fully connected layer
        with a weight matrix of size (out_channels x in_channels).
    bias : bool
        Whether to add a bias term.
    conv : callable
        Function which will perform the actual convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size, laplacian, bias=True,
                 conv=cheb_conv):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.register_buffer(f'laplacian', laplacian)
        self._conv = conv

        # shape = (kernel_size, out_channels, in_channels)
        shape = (in_channels, kernel_size, out_channels)
        self.weight = torch.nn.Parameter(torch.Tensor(*shape))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self, activation='relu', fan='in',
                         distribution='normal'):
        r"""Reset weight and bias.

        * Kaiming / He is given by `activation='relu'`, `fan='in'` or
        `fan='out'`, and `distribution='normal'`.
        * Xavier / Glorot is given by `activation='linear'`, `fan='avg'`,
          and `distribution='uniform'`.
        * LeCun is given by `activation='linear'` and `fan='in'`.

        Motivation based on inits from PyTorch, TensorFlow, Keras.

        Parameters
        ----------
        activation : {'relu', 'linear', 'sigmoid', 'tanh'}
            Select the activation function your are using.
        fan : {'in', 'out', 'avg'}
            Select `'in'` to preserve variance in the forward pass.
            Select `'out'` to preserve variance in the backward pass.
            Select `'avg'` for a balance.
        distribution : {'normal', 'uniform'}
            Whether to draw weights from a normal or random distribution.

        References
        ----------
        Delving Deep into Rectifiers: Surpassing Human-Level Performance on
        ImageNet Classification, Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian
        Sun, https://arxiv.org/abs/1502.01852

        Understanding the difficulty of training deep feedforward neural
        networks, Xavier Glorot, Yoshua Bengio,
        http://proceedings.mlr.press/v9/glorot10a.html
        """

        if fan == 'in':
            fan = self.in_channels * self.kernel_size
        elif fan == 'out':
            fan = self.out_channels * self.kernel_size
        elif fan == 'avg':
            fan = (self.in_channels + self.out_channels) / 2 * self.kernel_size
        else:
            raise ValueError('unknown fan')

        if activation == 'relu':
            scale = 2  # relu kills half the activations, from He et al.
        elif activation in ['linear', 'sigmoid', 'tanh']:
            # sigmoid and tanh are linear around 0
            scale = 1  # from Glorot et al.
        else:
            raise ValueError('unknown activation')

        if distribution == 'normal':
            std = math.sqrt(scale / fan)
            self.weight.data.normal_(0, std)
        elif distribution == 'uniform':
            limit = math.sqrt(3 * scale / fan)
            self.weight.data.uniform_(-limit, limit)
        else:
            raise ValueError('unknown distribution')

        if self.bias is not None:
            self.bias.data.fill_(0)

    def set_parameters(self, weight, bias=None):
        r"""Set weight and bias.

        Parameters
        ----------
        weight : array of shape in_channels x kernel_size x out_channels
            The coefficients of the Chebyshev polynomials.
        bias : vector of length out_channels
            The bias.
        """
        self.weight = torch.nn.Parameter(torch.as_tensor(weight))
        if bias is not None:
            self.bias = torch.nn.Parameter(torch.as_tensor(bias))

    def extra_repr(self):
        s = '{in_channels} -> {out_channels}, kernel_size={kernel_size}'
        s += ', bias=' + str(self.bias is not None)
        return s.format(**self.__dict__)

    def forward(self, inputs):
        r"""Forward graph convolution.

        Parameters
        ----------
        laplacian : sparse matrix of shape n_vertices x n_vertices
            Encode the graph structure.
        inputs : tensor of shape n_signals x n_vertices x n_features
            Data, i.e., features on the vertices.
        """
        
        outputs = self._conv(self.laplacian, inputs, self.weight)
        if self.bias is not None:
            outputs += self.bias
        return outputs
    
    
class ConvChebTemp(torch.nn.Module):
    """Graph spatio-temporal convolutional layer.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input graph.
    out_channels : int
        Number of channels in the output graph.
    graph_width : int
        Width of the spatial convolutional kernel.
        The order of the Chebyshev polynomials is graph_width - 1.

        * A graph_width of 1 won't take the neighborhood into account.
        * A graph_width of 2 will look up to the 1-neighborhood (1 hop away).
        * A graph_width of 3 will look up to the 2-neighborhood (2 hops away).

        A graph_width of 0 is equivalent to not having a graph (or an empty
        adjacency matrix). All the vertices are treated independently and form
        a set. Every element of that set is given to a fully connected layer
        with a weight matrix of size (out_channels x in_channels).
    temp_width : int
        Width of the temporal convolutional kernel. 
    bias : bool
        Whether to add a bias term.
    conv : callable
        Function which will perform the actual convolution.
    """

    def __init__(self, in_channels, out_channels, graph_width, temp_width, laplacian, bias=True,
                 conv=cheb_conv_temp):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.graph_width = graph_width
        self.temp_width = temp_width
        self._conv = conv
        self.register_buffer(f'laplacian', laplacian)
        
        shape = (in_channels, graph_width, temp_width, out_channels)
        self.weight = torch.nn.Parameter(torch.Tensor(*shape))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self, activation='relu', fan='in',
                         distribution='normal'):
        r"""Reset weight and bias.

        * Kaiming / He is given by `activation='relu'`, `fan='in'` or
        `fan='out'`, and `distribution='normal'`.
        * Xavier / Glorot is given by `activation='linear'`, `fan='avg'`,
          and `distribution='uniform'`.
        * LeCun is given by `activation='linear'` and `fan='in'`.

        Motivation based on inits from PyTorch, TensorFlow, Keras.

        Parameters
        ----------
        activation : {'relu', 'linear', 'sigmoid', 'tanh'}
            Select the activation function your are using.
        fan : {'in', 'out', 'avg'}
            Select `'in'` to preserve variance in the forward pass.
            Select `'out'` to preserve variance in the backward pass.
            Select `'avg'` for a balance.
        distribution : {'normal', 'uniform'}
            Whether to draw weights from a normal or random distribution.

        References
        ----------
        Delving Deep into Rectifiers: Surpassing Human-Level Performance on
        ImageNet Classification, Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian
        Sun, https://arxiv.org/abs/1502.01852

        Understanding the difficulty of training deep feedforward neural
        networks, Xavier Glorot, Yoshua Bengio,
        http://proceedings.mlr.press/v9/glorot10a.html
        """

        if fan == 'in':
            fan = self.in_channels * self.graph_width * self.temp_width
        elif fan == 'out':
            fan = self.out_channels * self.graph_width * self.temp_width
        elif fan == 'avg':
            fan = (self.in_channels + self.out_channels) / 2 * self.graph_width * self.temp_width
        else:
            raise ValueError('unknown fan')

        if activation == 'relu':
            scale = 2  # relu kills half the activations, from He et al.
        elif activation in ['linear', 'sigmoid', 'tanh']:
            # sigmoid and tanh are linear around 0
            scale = 1  # from Glorot et al.
        else:
            raise ValueError('unknown activation')

        if distribution == 'normal':
            std = math.sqrt(scale / fan)
            self.weight.data.normal_(0, std)
        elif distribution == 'uniform':
            limit = math.sqrt(3 * scale / fan)
            self.weight.data.uniform_(-limit, limit)
        else:
            raise ValueError('unknown distribution')

        if self.bias is not None:
            self.bias.data.fill_(0)

    def set_parameters(self, weight, bias=None):
        r"""Set weight and bias.

        Parameters
        ----------
        weight : array of shape in_channels x kernel_size x out_channels
            The coefficients of the Chebyshev polynomials.
        bias : vector of length out_channels
            The bias.
        """
        self.weight = torch.nn.Parameter(torch.as_tensor(weight))
        if bias is not None:
            self.bias = torch.nn.Parameter(torch.as_tensor(bias))

    def extra_repr(self):
        s = '{in_channels} -> {out_channels}, kernel_size={kernel_size}'
        s += ', bias=' + str(self.bias is not None)
        return s.format(**self.__dict__)

    def forward(self, inputs):
        r"""Forward graph convolution.

        Parameters
        ----------
        laplacian : sparse matrix of shape n_vertices x n_vertices
            Encode the graph structure.
        inputs : tensor of shape n_signals x n_vertices x len_sqce x n_features
            Data, i.e., features on the vertices.
        """
        outputs = self._conv(self.laplacian, inputs, self.weight)
        if self.bias is not None:
            outputs += self.bias
        return outputs

    
# Pooling layers
def _equiangular_calculator(tensor, ratio):
    N, M, F = tensor.size()
    dim1, dim2 = equiangular_dimension_unpack(M, ratio)
    bw_dim1, bw_dim2 = dim1/2, dim2/2
    tensor = tensor.view(N, dim1, dim2, F)
    return tensor, [bw_dim1, bw_dim2]


class PoolMaxEquiangular(torch.nn.MaxPool1d):
    """EquiAngular max pooling module
    
    Parameters
    ----------
    ratio : float
        Ratio between latitude and longitude dimensions of the data
    kernel_size : int
        Pooling kernel width
    return_indices : bool (default : True)
        Whether to return the indices corresponding to the locations of the maximum value retained at pooling
    """

    def __init__(self, ratio, kernel_size, return_indices=True):
        self.ratio = ratio
        super().__init__(kernel_size=kernel_size, return_indices=return_indices)

    def forward(self, inputs):
        """calls Maxpool1d and if desired, keeps indices of the pixels pooled to unpool them
        Parameters
        ----------
        x : torch.tensor of shape batch x pixels x features
            Input data
            
        Returns
        -------
        x : torch.tensor of shape batch x unpooled pixels x features
            Layer output
        indices : list(int)
            Indices of the pixels pooled
        """
        x, _ = _equiangular_calculator(inputs, self.ratio)
        x = x.permute(0, 3, 1, 2)

        if self.return_indices:
            x, indices = F.max_pool2d(x, self.kernel_size, return_indices=self.return_indices)
        else:
            x = F.max_pool2d(x, self.kernel_size)
        x = reformat(x)

        if self.return_indices:
            output = x, indices
        else:
            output = x

        return output


class UnpoolMaxEquiangular(torch.nn.MaxUnpool1d):
    """Equiangular max unpooling module
    
    Parameters
    ----------
    ratio : float
        Ratio between latitude and longitude dimensions of the data
    kernel_size : int
        Pooling kernel width
    """

    def __init__(self, ratio, kernel_size):
        self.ratio = ratio
        
        super().__init__(kernel_size=(kernel_size, kernel_size))

    def forward(self, inputs, indices):
        """calls MaxUnpool1d using the indices returned previously by PoolMaxEquiangular
        Parameters
        ----------
        inputs : torch.tensor of shape batch x pixels x features
            Input data
        indices : int
            Indices of pixels equiangular maxpooled previously
            
        Returns
        -------
        x : torch.tensor of shape batch x unpooled pixels x features
            Layer output
        """
        x, _ = _equiangular_calculator(inputs, self.ratio)
        x = x.permute(0, 3, 1, 2)
        x = F.max_unpool2d(x, indices, self.kernel_size)
        x = reformat(x)
        return x

class PoolAvgEquiangular(torch.nn.AvgPool1d):
    """EquiAngular average pooling
    
    Parameters
    ----------
    ratio : float
        Parameter for equiangular sampling -> width/height
    kernel_size : int
        Pooling kernel width
    """

    def __init__(self, ratio, kernel_size):
        self.ratio = ratio
        super().__init__(kernel_size=(kernel_size, kernel_size))

    def forward(self, inputs):
        """calls Avgpool1d
        Parameters
        ----------
        inputs : torch.tensor of shape batch x pixels x features
            Input data
        
        Returns
        -------
        x : torch.tensor of shape batch x pooled pixels x features
            Layer output
        """
        x, _ = _equiangular_calculator(inputs, self.ratio)
        x = x.permute(0, 3, 1, 2)
        x = F.avg_pool2d(x, self.kernel_size)
        x = reformat(x)

        return x
    
class UnpoolAvgEquiangular(torch.nn.Module):
    """EquiAngular average unpooling
    
    Parameters
    ----------
    ratio : float
        Parameter for equiangular sampling -> width/height
    """

    def __init__(self, ratio, kernel_size):
        self.ratio = ratio
        self.kernel_size = kernel_size
        super().__init__()

    def forward(self, inputs):
        """calls pytorch's interpolate function to create the values while unpooling based on the nearby values
        Parameters
        ----------
        inputs : torch.tensor of shape batch x pixels x features
            Input data
        
        Returns
        -------
        x : torch.tensor of shape batch x unpooled pixels x features
            Layer output
        """

        x, _ = _equiangular_calculator(inputs, self.ratio)
        x = x.permute(0, 3, 1, 2)
        x = F.interpolate(x, scale_factor=(self.kernel_size, self.kernel_size), mode="nearest")
        x = reformat(x)
        return x
    
    
class PoolMaxHealpix(torch.nn.MaxPool1d):
    """Healpix Maxpooling module
     
    Parameters
    ----------
    kernel_size : int
        Pooling kernel width
    return_indices : bool (default : True)
        Whether to return the indices corresponding to the locations of the maximum value retained at pooling
    """

    def __init__(self, kernel_size, return_indices=True):
        super().__init__(kernel_size=kernel_size, return_indices=return_indices)

    def forward(self, x):
        """calls Maxpool1d and if desired, keeps indices of the pixels pooled to unpool them
        Parameters
        ----------
        x : torch.tensor of shape batch x pixels x features
            Input data  
        indices : list
            Indices where the max value was located in unpooled image
            
        Returns
        -------
        x : torch.tensor of shape batch x unpooled pixels x features
            Layer output
        indices : list(int)
            Indices of the pixels pooled
        """
        x = x.permute(0, 2, 1)
        if self.return_indices:
            x, indices = F.max_pool1d(x, self.kernel_size, return_indices=self.return_indices)
        else:
            x = F.max_pool1d(x)
        x = x.permute(0, 2, 1)

        if self.return_indices:
            output = x, indices
        else:
            output = x
        return output


    
class PoolAvgHealpix(torch.nn.Module):
    """Healpix average pooling module
    
    Parameters
    ----------
    kernel_size : int
        Pooling kernel width
    """

    def __init__(self, kernel_size):
        """kernel_size should be 4, 16, 64, etc."""
        super().__init__()
        self.kernel_size = kernel_size

    def extra_repr(self):
        return 'kernel_size={kernel_size}'.format(**self.__dict__)

    def forward(self, x):
        """x has shape (batch, pixels, channels) and is in nested ordering"""
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.avg_pool1d(x, self.kernel_size)
        return x.permute(0, 2, 1)


class UnpoolAvgHealpix(torch.nn.Module):
    """Healpix Average Unpooling module
    
    Parameters
    ----------
    kernel_size : int
        Pooling kernel width
    """

    def __init__(self, kernel_size):
        """kernel_size should be 4, 16, 64, etc."""
        super().__init__()
        self.kernel_size = kernel_size

    def extra_repr(self):
        return 'kernel_size={kernel_size}'.format(**self.__dict__)

    def forward(self, x):
        """x has shape (batch, pixels, channels) and is in nested ordering"""
        # return x.repeat_interleave(self.kernel_size, dim=1)
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.interpolate(x, scale_factor=self.kernel_size, mode='nearest')
        return x.permute(0, 2, 1)
    

class UnpoolMaxHealpix(torch.nn.MaxUnpool1d):
    """HEALpix max unpooling module
    
    Parameters
    ----------
    kernel_size : int
        Pooling kernel width
    """

    def __init__(self, kernel_size):
        super().__init__(kernel_size=kernel_size)
        

    def forward(self, x, indices):
        """calls pytorch's unpool1d function to create the values while unpooling based on the nearby values
        Parameters
        ----------
        inputs : torch.tensor of shape batch x pixels x features
            Input data
        indices : list
            Indices where the max value was located in unpooled image
        
        Returns
        -------
        x : torch.tensor of shape batch x unpooled pixels x features
            Layer output
        """

        x = x.permute(0, 2, 1)
        x = F.max_unpool1d(x, indices, self.kernel_size)
        x = x.permute(0, 2, 1)
        return x
    
    
 # Temporal + graph 2D pooling
class PoolAvgTempHealpix(torch.nn.Module):
    """Healpix with temporal convolutions average pooling module for 2D data
    
    Parameters
    ----------
    kernel_size : int
        Pooling kernel width
    """

    def __init__(self, kernel_size):
        """kernel_size should be 4, 16, 64, etc."""
        super().__init__()
        self.kernel_size = kernel_size

    def extra_repr(self):
        return 'kernel_size={kernel_size}'.format(**self.__dict__)

    def forward(self, x):
        """x has shape (batch, nodes, len_sqce, channels) and is in nested ordering"""
        x = x.permute(0, 3, 1, 2) # batch, channels, nodes, len_sqce
        x = F.avg_pool2d(x, self.kernel_size)
        return x.permute(0, 2, 3, 1)


class UnpoolAvgTempHealpix(torch.nn.Module):
    """Healpix with temporal convolutions Average Unpooling module
    
    Parameters
    ----------
    kernel_size : int
        Pooling kernel width
    """

    def __init__(self, kernel_size):
        """kernel_size should be 4, 16, 64, etc."""
        super().__init__()
        self.kernel_size = kernel_size

    def extra_repr(self):
        return 'kernel_size={kernel_size}'.format(**self.__dict__)

    def forward(self, x):
        """x has shape (batch, nodes, len_sqce, channels) and is in nested ordering"""
        # return x.repeat_interleave(self.kernel_size, dim=1)
        x = x.permute(0, 3, 1, 2) # batch, channels, nodes, len_sqce
        x = F.interpolate(x, scale_factor=self.kernel_size, mode='nearest')
        return x.permute(0, 2, 3, 1)
    
    

class PoolMaxTempHealpix(torch.nn.MaxPool1d):
    """Healpix Maxpooling module for spatio-temporal convolutions
     
    Parameters
    ----------
    kernel_size : tuple
        Pooling kernel shape. First dimension indicates spatial kernel with, second dimension is 
        temporal kernel width
    return_indices : bool (default : True)
        Whether to return the indices corresponding to the locations of the maximum value retained at pooling
    """

    def __init__(self, kernel_size, return_indices=True):
        super().__init__(kernel_size=kernel_size, return_indices=return_indices)

    def forward(self, x):
        """calls Maxpool1d and if desired, keeps indices of the pixels pooled to unpool them
        Parameters
        ----------
        x : torch.tensor of shape batch x pixels x features
            Input data  
        indices : list
            Indices where the max value was located in unpooled image
            
        Returns
        -------
        x : torch.tensor of shape batch x unpooled pixels x features
            Layer output
        indices : list(int)
            Indices of the pixels pooled
        """
        x = x.permute(0, 3, 1, 2)
        if self.return_indices:
            x, indices = F.max_pool2d(x, self.kernel_size, return_indices=self.return_indices)
        else:
            x = F.max_pool2d(x)
        x = x.permute(0, 2, 3, 1)

        if self.return_indices:
            output = x, indices
        else:
            output = x
        return output
    
    
class UnpoolMaxTempHealpix(torch.nn.MaxUnpool1d):
    """HEALpix max unpooling module for spatio-temporal convolutions
    
    Parameters
    ----------
    kernel_size : tuple
        Pooling kernel shape. First dimension indicates spatial kernel width, second dimension is 
        temporal kernel width
    """

    def __init__(self, kernel_size):
        super().__init__(kernel_size=kernel_size)
        

    def forward(self, x, indices):
        """calls pytorch's unpool1d function to create the values while unpooling based on the nearby values
        Parameters
        ----------
        inputs : torch.tensor of shape [batch x nodes x len_sqce x features] in nested ordering
            Input data
        indices : list
            Indices where the max value was located in unpooled image
        
        Returns
        -------
        x : torch.tensor of shape [batch x nodes x len_sqce x features] in nested ordering
            Layer output
        """
        x = x.permute(0, 3, 1, 2)
        x = F.max_unpool2d(x, indices, self.kernel_size)
        x = x.permute(0, 2, 3, 1)
        return x
    
