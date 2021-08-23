#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 21:51:40 2021

@author: ghiggi
"""
import math
import torch
import numpy as np
from collections import Counter
from abc import ABC, abstractmethod
from scipy import sparse
from scipy.spatial import SphericalVoronoi
from torch.nn import functional as F
from modules import remap
from sparselinear import SparseLinear
##----------------------------------------------------------------------------.
# ##############################
#### Laplacian computations ####
# ##############################
def _import_igl():
    try:
        import igl
    except Exception as e:
        raise ImportError('Cannot import igl. Build a knn graph '
                          'instead of a voronoi mesh graph or install it with '
                          'conda install igl. '
                          'Original exception: {}'.format(e))
    return igl

def triangulate(graph):
    sv = SphericalVoronoi(graph.coords)
    assert sv.points.shape[0] == graph.n_vertices
    return sv.points, sv._simplices

def compute_cotan_laplacian(graph, return_mass=False):
    igl = _import_igl()
    v, f = triangulate(graph)
    L = -igl.cotmatrix(v, f)
    assert len((L - L.T).data) == 0
    M = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
    # M = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_BARYCENTRIC)
    if return_mass:
        # Eliminate zeros for speed (appears for equiangular).
        L.eliminate_zeros()
        return L, M
    else:
        Minv = sparse.diags(1 / M.diagonal())
        return Minv @ L
    
def estimate_lmax(laplacian, tol=5e-3):
    """Estimate the Laplacian's largest eigenvalue."""
    # eigs(Minv @ L) is faster than eigsh(L, M) and we use Minv @ L to convolve
    lmax = sparse.linalg.eigs(laplacian, k=1, tol=tol,
                              ncv=min(laplacian.shape[0], 10),
                              return_eigenvectors=False)
    lmax = np.real(lmax[0])  # Always real even if not symmetric.
    lmax *= 1 + 2 * tol  # Margin to be robust to estimation errors.
    return lmax

def scale_operator(laplacian, lmax, scale=1):
    """Scale the eigenvalues from [0, lmax] to [-scale, scale]."""
    sparse_I = sparse.identity(laplacian.shape[0], format=laplacian.format, dtype=laplacian.dtype)
    laplacian *= 2 * scale / lmax
    laplacian -= sparse_I
    return laplacian
    
def prepare_torch_laplacian(laplacian):
    """Prepare a graph Laplacian to be fed to a graph convolutional layer."""
    # Change type 
    laplacian = laplacian.astype(np.float32) # from float64 to float32
    # Scale the eigenvalues
    lmax = estimate_lmax(laplacian) 
    laplacian = scale_operator(laplacian, lmax)
    # Construct the Laplacian sparse matrix in COOrdinate format
    laplacian = sparse.coo_matrix(laplacian, laplacian.dtype) 
    # Get torch dtype
    torch_dtype = torch.get_default_dtype()  
    # Build Torch sparse tensor in COOrdinate format
    # - PyTorch wants a LongTensor (int64) as indices (it'll otherwise convert).
    indices = np.empty((2, laplacian.nnz), dtype=np.int64)
    np.stack((laplacian.row, laplacian.col), axis=0, out=indices)
    indices = torch.from_numpy(indices)
    laplacian = torch.sparse_coo_tensor(indices=indices, 
                                        values=laplacian.data,
                                        size=laplacian.shape,
                                        dtype=torch_dtype,
                                        device=indices.device)  
    laplacian = laplacian.coalesce()  # More efficient subsequent operations.
    return laplacian

#----------------------------------------------------------------------------.
# ################################
#### Graph Convolution layer  ####
# ################################
def conv_cheb(laplacian, inputs, weight):
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
    # Terminology
    # B = batch size
    # V = nb vertices (nodes)
    # Fin = nb input features
    # Fout = nb output features
    # K = order of Chebyshev polynomials (kenel size)
    ##------------------------------------------------------------------------.
    # Get input tensor shape 
    B, V, Fin1 = inputs.shape
    # print('B: {}, V. {}, Fin: {}'.format(B,V,Fin1))
    ##------------------------------------------------------------------------.
    # Get weight tensor shape
    Fin, K, Fout = weight.shape
    # print('Fin: {}, K: {}, Fout: {}'.format(Fin, K, Fout))
    ##------------------------------------------------------------------------.
    # Check the tensor shape is correct 
    if not Fin1 == Fin:
        raise ValueError("Input tensor shape does not match the expected shape: \n" +
                         "- Input tensor shape :{} \n".format(Fin1) + 
                         "- Expected tensor shape :{} \n".format(Fin))
    ##------------------------------------------------------------------------.  
    # Transform to Chebyshev basis    
    x0 = inputs.permute(1, 2, 0).contiguous()  # V x Fin x B
    x0 = x0.view([V, Fin * B])  # V x Fin*B
    x = x0.unsqueeze(0)  # 1 x V x Fin*B
    ##------------------------------------------------------------------------.
    # Perform sparse matrix multiplications 
    if K > 1:
        x1 = torch.sparse.mm(laplacian, x0)  # V x Fin*B
        x = torch.cat((x, x1.unsqueeze(0)), 0)  # 2 x V x Fin*B
    for _ in range(2, K):
        x2 = 2 * torch.sparse.mm(laplacian, x1) - x0
        x = torch.cat((x, x2.unsqueeze(0)), 0)  # M x Fin*B
        x0, x1 = x1, x2
    ##------------------------------------------------------------------------.
    x = x.view([K, V, Fin, B])  # K x V x Fin x B
    x = x.permute(3, 1, 2, 0).contiguous()  # B x V x Fin x K
    x = x.view([B * V, Fin * K])  # B*V x Fin*K
    ##------------------------------------------------------------------------.
    # Linearly compose Fin features to get Fout features
    weight = weight.view(Fin * K, Fout)
    x = x.matmul(weight)  # B*V x Fout
    x = x.view([B, V, Fout])  # B x V x Fout
    ##------------------------------------------------------------------------.
    return x

class ConvCheb(torch.nn.Module):
    """Graph Convolutional Layer.

    PyTorch implementation of a convolutional neural network on graphs based on
    Chebyshev polynomials of the graph Laplacian.
    
    See https://arxiv.org/abs/1606.09375 for details.
    Copyright 2018 Michaël Defferrard.
    Released under the terms of the MIT license.
    
    It expects the input data to have dimensions: 
      
        (n_signals, n_vertices, n_features) aka (sample, node, time-feature)   
        
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
        Function which will perform the actual convolution. Default to Chebyshev 
        convolution.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, laplacian,
                 bias=True, conv = conv_cheb, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # - Define the Chebyshev Convolution
        self._conv = conv
        # - Add the laplacina as a buffer (to not be considered a model parameter)
        self.register_buffer('laplacian', laplacian)
        # - Define the weights 
        shape = (in_channels, kernel_size, out_channels)
        self.weight = torch.nn.Parameter(torch.Tensor(*shape))
        # - Define the bias parameters 
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        # - Initialize parameters
        self.reset_parameters()

    def reset_parameters(self, 
                         activation='relu',
                         fan='in',
                         distribution='normal'):
        """Reset weight and bias.

        * Kaiming / He initialization is given by:
            - `activation='relu'`
            - `fan='in'` or `fan='out'`
            - `distribution='normal'`
        * Xavier / Glorot initialization is given by:
            - `activation='linear'` 
            - `fan='avg'` 
            - `distribution='uniform'`
        * LeCun is given by:
            - `activation='linear'` 
            - `fan='in'`.

        Motivation based on initalizations from PyTorch, TensorFlow, Keras.

        Parameters
        ----------
        activation : str
            Name of a torch.nn.functional activation function
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
        
        if activation in ['relu', 'celu', 'selu', 'prelu','hardswish', 'mish',
                          'silu','gelu','softplus', 'softmax','logsigmoid',
                          'relu6', # upper bounded at 6
                          'rrlu', 'leaky_relu', 'elu']:  
            # relu kills half the activations, from He et al.
            scale = 2  
        elif activation in ['linear', 'hardshrink ',
                            'sigmoid', 'hardsigmoid', 
                            'tanh', 'hardtanh', 'softsign']:
            # sigmoid and tanh are linear around 0
            scale = 1  # from Glorot et al.
        else:
            raise ValueError('Unknown activation')

        if distribution == 'normal':
            std = math.sqrt(scale / fan)
            self.weight.data.normal_(0, std)
        elif distribution == 'uniform':
            limit = math.sqrt(3 * scale / fan)
            self.weight.data.uniform_(-limit, limit)
        else:
            raise ValueError('Unknown distribution')

        if self.bias is not None:
            self.bias.data.fill_(0)

    def set_parameters(self, weight, bias=None):
        """Set weight and bias.

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
        """Extra repr."""
        s = '{in_channels} -> {out_channels}, kernel_size={kernel_size}'
        s += ', bias=' + str(self.bias is not None)
        return s.format(**self.__dict__)

    def forward(self, inputs):
        """Forward graph convolution.

        Parameters
        ----------
        inputs : tensor of shape n_signals x n_vertices x n_features
            Data, i.e., features on the vertices.
        """
        outputs = self._conv(self.laplacian, inputs, self.weight)
        if self.bias is not None:
            outputs += self.bias
        return outputs

#----------------------------------------------------------------------------.
# ######################################
#### Equiangular Convolution layer  ####
# ######################################  
def get_nlat_nlon(n_nodes, lonlat_ratio):
    """Calculate the width (longitudes) and height (latitude) of an equiangular grid provided in 1D.
    
    Parameters
    ----------
    lonlat_ratio : int
        lonlat_ratio = H // W = n_longitude rings / n_latitude rings
        Aspect ratio to reshape the input 1D data to a 2D image.
        A ratio of 2 means the equiangular grid has the same resolution
        in latitude and longitude.
    """
    # ratio = n_lon/n_lat (width/height)
    n_lat = int((n_nodes / lonlat_ratio) ** 0.5)
    n_lon = int((n_nodes * lonlat_ratio) ** 0.5)
    if n_lat * n_lon != n_nodes: # Try to correct n_lat or n_lon if ratio is wrong
        if n_nodes % n_lat == 0:
            n_lon = n_nodes // n_lat
        if n_nodes % n_lon == 0:
            n_lat = n_nodes // n_lon
    assert n_lat * n_lon == n_nodes, f'Unable to unpack nodes: {n_nodes}, lonlat_ratio: {lonlat_ratio}'
    return n_lat, n_lon

def equiangular_1d_to_2d(tensor, lonlat_ratio):
    """Reformat the input from a 3D tensor to a 4D tensor.
    
    Shapes: (sample, node, time-feature) --> (sample, lat, lon, time-feature)
    """
    batch_size, n_nodes, features = tensor.size()
    n_lat, n_lon = get_nlat_nlon(n_nodes, lonlat_ratio)
    tensor = tensor.view(batch_size, n_lat, n_lon, features)
    return tensor

def equiangular_2d_to_1d(x): 
    """Reformat the input from a 4D tensor to a 3D tensor.
    
    Shapes: (sample, lat, lon, time-feature) --> (sample, node, time-feature)
    """
    batch_size, n_lat, n_lon, features = x.size()
    x = x.view(batch_size, n_lat * n_lon, features)
    return x

class Conv2dEquiangular(torch.nn.Module):
    """Equiangular 2D Convolutional layer.
    
    Enable to treat the equiangular spherical sampling as: 
    - planar 2D projection (when periodic_padding=False)
    - cylindrical projection (when periodic_padding=True)  
    
    When periodic_padding=True no "borders" exists along the longitude (width) dimension.
           
    It expects the input data to have dimensions: (sample, node, time-feature)

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels in the output image.
    kernel_size : int
        Width of the square convolutional kernel.
        The actual size of the kernel is kernel_size**2
    lonlat_ratio : int
        lonlat_ratio = H // W = n_longitude rings / n_latitude rings
        Aspect ratio to reshape the input 1D data to a 2D image.
        A ratio of 2 means the equiangular grid has the same resolution
        in latitude and longitude.
    periodic_padding : bool
        whether to use periodic padding along the longitude dimension. (default: True)
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, lonlat_ratio, periodic_padding, bias, **kwargs):
        super().__init__()
        self.lonlat_ratio = lonlat_ratio
        self.periodic_padding = periodic_padding

        self.kernel_size = kernel_size
        self.pad_width = int((self.kernel_size - 1) / 2)
        
        self.conv = torch.nn.Conv2d(in_channels = in_channels, 
                                    out_channels = out_channels, 
                                    kernel_size = self.kernel_size,
                                    bias = bias,
                                    **kwargs)
        # Re-initialize parameters 
        torch.nn.init.xavier_uniform_(self.conv.weight)
        torch.nn.init.zeros_(self.conv.bias)

    def periodic_pad(self, x, width, periodic_padding=True):       
        """Periodic padding function.
        
        When periodic_padding=True no "borders" exists along the longitude (width) dimension. 

        Parameters
        ----------
        x : torch.Tensor
            The tensor format should be N * C * H * W
        width : int
            The padding width.
        periodic_padding: bool
            Whether to use periodic padding along the longitude dimension. 
            If False, the function is same as ZeroPad2D. 
            The default is True.
        
        Returns
        -------
        padded : torch.tensor 
            Return the padded tensor.   
        """
        if periodic_padding:
            x = torch.cat((x[:, :, :, -width:], x, x[:, :, :, :width]), dim=3)
            padded = F.pad(x, (0, 0, width, width), 'constant', 0)
        else:
            padded = F.pad(x, (width, width, width, width), 'constant', 0)
        return padded

    def forward(self, x):
        """Perform convolution."""
        x = equiangular_1d_to_2d(x, self.lonlat_ratio)
        x = x.permute(0, 3, 1, 2)

        x = self.periodic_pad(x, self.pad_width, self.periodic_padding)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = equiangular_2d_to_1d(x)
        return x

#----------------------------------------------------------------------------.
# ##################### 
#### Utils Pooling ####
# #####################
def _build_interpolation_matrix(src_graph, dst_graph):
    """Return the sparse matrix that interpolates between two spherical samplings."""
    ds = remap.compute_interpolation_weights(src_graph=src_graph,
                                             dst_graph=dst_graph, 
                                             method='conservative',
                                             normalization='fracarea') 

    # Sanity checks
    np.testing.assert_allclose(ds.src_grid_center_lat, src_graph.signals['lat'])
    np.testing.assert_allclose(ds.src_grid_center_lon, src_graph.signals['lon'])
    np.testing.assert_allclose(ds.dst_grid_center_lat, dst_graph.signals['lat'])
    np.testing.assert_allclose(ds.dst_grid_center_lon, dst_graph.signals['lon'])
    np.testing.assert_allclose(ds.src_grid_frac, 1)
    np.testing.assert_allclose(ds.dst_grid_frac, 1)
    np.testing.assert_allclose(ds.src_grid_imask, 1)
    np.testing.assert_allclose(ds.dst_grid_imask, 1)

    col = ds.src_address
    row = ds.dst_address
    dat = ds.remap_matrix.squeeze()
    
    # CDO indexing starts at 1
    row = np.array(row) - 1
    col = np.array(col) - 1
    weights = sparse.csr_matrix((dat, (row, col)))
    assert weights.shape == (dst_graph.n_vertices, src_graph.n_vertices)

    # Destination pixels are normalized to 1 (row-sum = 1)
    # Weights represent the fractions of area attributed to source pixels
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    # Interpolation is conservative: it preserves area
    np.testing.assert_allclose(weights.T @ ds.dst_grid_area, ds.src_grid_area)

    # Unnormalize
    weights = weights.multiply(ds.dst_grid_area.values[:, np.newaxis])

    # Another way to assert that the interpolation is conservative
    np.testing.assert_allclose(np.asarray(weights.sum(1)).squeeze(), ds.dst_grid_area)
    np.testing.assert_allclose(np.asarray(weights.sum(0)).squeeze(), ds.src_grid_area)

    return weights

def build_pooling_matrices(src_graph, dst_graph):
    """Create pooling and unpooling matrix."""
    weights = _build_interpolation_matrix(src_graph, dst_graph)
    pool = weights.multiply(1/weights.sum(1))
    unpool = weights.multiply(1/weights.sum(0)).T
    return pool, unpool

def convert_to_torch_sparse(mat: "sparse.coo.coo_matrix"):
    """Convert a sparse matrix to a torch.sparse COO matrix."""
    torch_dtype = torch.get_default_dtype()
    indices = np.empty((2, mat.nnz), dtype=np.int64)
    np.stack((mat.row, mat.col), axis=0, out=indices)
    indices = torch.from_numpy(indices)
    mat = torch.sparse_coo_tensor(indices, mat.data, mat.shape, 
                                  dtype=torch_dtype, 
                                  device=indices.device)
    mat = mat.coalesce()
    return mat
    
#----------------------------------------------------------------------------.
# ################################## 
#### Graph-based Pooling layers ####
# ##################################    
class EquiangularMaxPool(torch.nn.MaxPool1d):
    """Equiangular Max Pooling Layer.
    
    Parameters
    ----------
    lonlat_ratio : int
        lonlat_ratio = H // W = n_longitude rings / n_latitude rings
        Aspect ratio to reshape the input 1D data to a 2D image.
        A ratio of 2 means the equiangular grid has the same resolution
        in latitude and longitude.
    kernel_size : int
        Pooling kernel width
    return_indices : bool (default : True)
        Whether to return the indices corresponding to the locations of the
        maximum value retained at pooling. Useful to unpool. The default is True.
    """

    def __init__(self, lonlat_ratio, kernel_size, return_indices=True, *args, **kwargs):
        self.lonlat_ratio = lonlat_ratio
        kernel_size = int(kernel_size ** 0.5)
        super().__init__(kernel_size=kernel_size, return_indices=return_indices)

    def forward(self, x):
        """Perform pooling.
        
        Parameters
        ----------
        x : torch.tensor 
            Torch tensor of shape batch x pixels x features.
            
        Returns
        -------
        x : torch.tensor
            Pooling output tensor of shape batch x pooled pixels x features 
        indices : list(int)
            Indices of the pooled pixels.
        """
        x = equiangular_1d_to_2d(x, self.lonlat_ratio)
        x = x.permute(0, 3, 1, 2)

        if self.return_indices:
            x, indices = F.max_pool2d(x, self.kernel_size, return_indices=self.return_indices)
        else:
            x = F.max_pool2d(x, self.kernel_size)
        x = x.permute(0, 2, 3, 1)
        x = equiangular_2d_to_1d(x)

        if self.return_indices:
            return x, indices
        else:
            return x

class EquiangularMaxUnpool(torch.nn.MaxUnpool1d):
    """Equiangular Max Unpooling Layer.
    
    Parameters
    ----------
    lonlat_ratio : int
        lonlat_ratio = H // W = n_longitude rings / n_latitude rings
        Aspect ratio to reshape the input 1D data to a 2D image.
        A ratio of 2 means the equiangular grid has the same resolution
        in latitude and longitude.
    kernel_size : int
        Pooling kernel width
    """

    def __init__(self, lonlat_ratio, kernel_size, *args, **kwargs):
        self.lonlat_ratio =lonlat_ratio
        kernel_size = int(kernel_size ** 0.5)
        super().__init__(kernel_size=(kernel_size, kernel_size))

    def forward(self, x, indices):
        """Perform unpooling.
        
        Parameters
        ----------
        x : torch.tensor 
            Torch tensor of shape batch x pixels x features.
        indices : list(int)
            Indices of the pooled pixels. 
            
        Returns
        -------
        x : torch.tensor 
            Unpooling output tensor output of shape batch x unpooled pixels x features
         
        """
        x = equiangular_1d_to_2d(x, self.lonlat_ratio)
        x = x.permute(0, 3, 1, 2)
        x = F.max_unpool2d(x, indices, self.kernel_size)
        x = x.permute(0, 2, 3, 1)
        x = equiangular_2d_to_1d(x)
        return x


class EquiangularAvgPool(torch.nn.AvgPool1d):
    """Equiangular Average Pooling Layer.
    
    Parameters
    ----------
    lonlat_ratio : int
        lonlat_ratio = H // W = n_longitude rings / n_latitude rings
        Aspect ratio to reshape the input 1D data to a 2D image.
        A ratio of 2 means the equiangular grid has the same resolution
        in latitude and longitude.
    kernel_size : int
        Pooling kernel width
    """

    def __init__(self, lonlat_ratio, kernel_size, *args, **kwargs):
        self.lonlat_ratio = lonlat_ratio
        kernel_size = int(kernel_size ** 0.5)
        super().__init__(kernel_size=(kernel_size, kernel_size))

    def forward(self, x):
        """Perform pooling.
        
        Parameters
        ----------
        x : torch.tensor 
            Torch tensor of shape batch x pixels x features.
            
        Returns
        -------
        x : torch.tensor
            Pooling output tensor of shape batch x pooled pixels x features 
        
        """
        x = equiangular_1d_to_2d(x, self.lonlat_ratio)
        x = x.permute(0, 3, 1, 2)
        x = F.avg_pool2d(x, self.kernel_size)
        x = x.permute(0, 2, 3, 1)
        x = equiangular_2d_to_1d(x)

        return x, None


class EquiangularAvgUnpool(torch.nn.Module):
    """Equiangular Average Pooling Layer.
    
    Parameters
    ----------
    lonlat_ratio : int
        lonlat_ratio = H // W = n_longitude rings / n_latitude rings
        Aspect ratio to reshape the input 1D data to a 2D image.
        A ratio of 2 means the equiangular grid has the same resolution
        in latitude and longitude.
    kernel_size : int
        Pooling kernel width
    """

    def __init__(self, lonlat_ratio, kernel_size, *args, **kwargs):
        self.lonlat_ratio = lonlat_ratio
        self.kernel_size = int(kernel_size ** 0.5)
        super().__init__()

    def forward(self, x, *args):
        """Perform unpooling.
        
        Parameters
        ----------
        x : torch.tensor 
            Torch tensor of shape batch x pixels x features.
            
        Returns
        -------
        x : torch.tensor 
            Unpooling output tensor output of shape batch x unpooled pixels x features
         
        """
        x = equiangular_1d_to_2d(x, self.lonlat_ratio)
        x = x.permute(0, 3, 1, 2)
        x = F.interpolate(x, scale_factor=(self.kernel_size, self.kernel_size), mode="nearest")
        x = x.permute(0, 2, 3, 1)
        x = equiangular_2d_to_1d(x)
        return x


class HealpixMaxPool(torch.nn.MaxPool1d):
    """Healpix Max Pooling Layer.
    
    Parameters
    ----------
    kernel_size : int
        Pooling kernel width
    return_indices : bool (default : True)
        Whether to return the indices corresponding to the locations of the
        maximum value retained at pooling. Useful to unpool. The default is True.
    """
    
    def __init__(self, kernel_size, return_indices=True, *args, **kwargs):
        super().__init__(kernel_size=kernel_size, return_indices=return_indices)

    def forward(self, x):
        """Perform pooling.
        
        Parameters
        ----------
        x : torch.tensor 
            Torch tensor of shape batch x pixels x features.
            x is expected to have nested ordering.
            
        Returns
        -------
        x : torch.tensor
            Pooling output tensor of shape batch x pooled pixels x features 
        indices : list(int)
            Indices of the pooled pixels.
        """
        x = x.permute(0, 2, 1)
        if self.return_indices:
            x, indices = F.max_pool1d(x, self.kernel_size, return_indices=self.return_indices)
        else:
            x = F.max_pool1d(x, self.kernel_size)
        x = x.permute(0, 2, 1)

        if self.return_indices:
            output = x, indices
        else:
            output = x
        return output

class HealpixMaxUnpool(torch.nn.MaxUnpool1d):
    """Healpix Max Unpooling Layer.
    
    Parameters
    ----------
    kernel_size : int
        Pooling kernel width
    """

    def __init__(self, kernel_size, *args, **kwargs):
        super().__init__(kernel_size=kernel_size)

    def forward(self, x, indices, **kwargs):
        """Perform unpooling.
        
        Parameters
        ----------
        x : torch.tensor 
            Torch tensor of shape batch x pixels x features.
            x is expected to have nested ordering.
        indices : list(int)
            Indices of the pooled pixels. 
            
        Returns
        -------
        x : torch.tensor 
            Unpooling output tensor output of shape batch x unpooled pixels x features
         
        """
        x = x.permute(0, 2, 1)
        x = F.max_unpool1d(x, indices, self.kernel_size)
        x = x.permute(0, 2, 1)
        return x

class HealpixAvgPool(torch.nn.Module):
    """Healpix Average Pooling Layer.
    
    Parameters
    ----------
    kernel_size : int
        Pooling kernel width
    """
    
    def __init__(self, kernel_size, *args, **kwargs):
        """kernel_size should be 4, 16, 64, etc."""
        super().__init__()
        self.kernel_size = kernel_size

    def extra_repr(self):
        """Extra repr."""
        return 'kernel_size={kernel_size}'.format(**self.__dict__)

    def forward(self, x):
        """Perform pooling.
        
        Parameters
        ----------
        x : torch.tensor 
            Torch tensor of shape batch x pixels x features.
            x is expected to have nested ordering.
            
        Returns
        -------
        x : torch.tensor
            Pooling output tensor of shape batch x pooled pixels x features 
        
        """
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.avg_pool1d(x, self.kernel_size)
        return x.permute(0, 2, 1), None


class HealpixAvgUnpool(torch.nn.Module):
    """Healpix Average Unpooling Layer.
    
    Parameters
    ----------
    kernel_size : int
        Pooling kernel width
    """

    def __init__(self, kernel_size, *args, **kwargs):
        """kernel_size should be 4, 16, 64, etc."""
        super().__init__()
        self.kernel_size = kernel_size

    def extra_repr(self):
        """Extra repr."""
        return 'kernel_size={kernel_size}'.format(**self.__dict__)

    def forward(self, x, *args):
        """Perform unpooling.
        
        Parameters
        ----------
        x : torch.tensor 
            Torch tensor of shape batch x pixels x features.
            x is expected to have nested ordering.
            
        Returns
        -------
        x : torch.tensor 
            Unpooling output tensor output of shape batch x unpooled pixels x features
         
        """
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.interpolate(x, scale_factor=self.kernel_size, mode='nearest')
        return x.permute(0, 2, 1)

#----------------------------------------------------------------------------.
# ##################################
#### Mesh-based Pooling layers  ####
# ##################################
class RemapBlock(torch.nn.Module):
    """General function for mesh-based pool/unpooling."""
    
    def __init__(self, remap_matrix: "sparse.coo.coo_matrix"):
        super().__init__()
        remap_matrix = self.process_remap_matrix(remap_matrix)
        self.register_buffer('remap_matrix', remap_matrix)
    
    def forward(self, x, *args, **kwargs):
        """Perform pooling or unpooling."""
        n_batch, n_nodes, n_val = x.shape
        matrix = self.remap_matrix
        new_nodes, _ = matrix.shape
        x = x.permute(1, 2, 0).reshape(n_nodes, n_batch * n_val)
        x = torch.sparse.mm(matrix, x)
        x = x.reshape(new_nodes, n_val, n_batch).permute(2, 0, 1)
        return x
    
    def process_remap_matrix(self, mat):
        """Compute the remapping matrix."""
        return convert_to_torch_sparse(mat)
    
class GeneralAvgPool(RemapBlock):
    """Generalized Average Pooling."""
    
    def forward(self, x, *args, **kwargs):
        """Perform pooling."""
        x = super().forward(x, *args, **kwargs)
        # Some pooling methods (e.g. Avg) do not give source indices, please return None for compatibility reason
        return x, None

class GeneralAvgUnpool(RemapBlock):
    """Generalized Average Unpooling."""
    
    def forward(self, x, *args, **kwargs):
        """Perform unpooling."""
        x = super().forward(x, *args, **kwargs)
        return x

##----------------------------------------------------------------------------.
class GeneralMaxAreaPool(RemapBlock):
    """Generalized Max Area Pooling."""
    
    def forward(self, x, *args, **kwargs):
        """Perform pooling."""
        x = super().forward(x, *args, **kwargs)
        # Some pooling methods (e.g. Avg) do not give source indices, please return None for compatibility reason
        return x, None
        
    def process_remap_matrix(self, mat):
        """Compute the remapping matrix."""
        torch_dtype = torch.get_default_dtype()
        max_ind_col = np.argmax(mat, axis=1).T
        row = np.arange(max_ind_col.shape[1]).reshape(1, -1)
        indices = np.concatenate([row, max_ind_col], axis=0)
        indices = torch.from_numpy(indices.astype(np.int64))
        mat = torch.sparse_coo_tensor(indices, torch.ones(indices.shape[1]), mat.shape, 
                                      dtype=torch_dtype,
                                      device=indices.device)
        mat = mat.coalesce()
        return mat


class GeneralMaxAreaUnpool(RemapBlock):
    """Generalized Max Area Unpooling."""
    
    def process_remap_matrix(self, mat):
        """Compute the remapping matrix."""
        torch_dtype = torch.get_default_dtype()
        max_ind_row = np.argmax(mat, axis=0)
        col = np.arange(max_ind_row.shape[1]).reshape(1, -1)
        indices = np.concatenate([max_ind_row, col], axis=0)
        indices = torch.from_numpy(indices.astype(np.int64))
        mat = torch.sparse_coo_tensor(indices, torch.ones(indices.shape[1]), mat.shape,
                                      dtype=torch_dtype,
                                      device=indices.device)
        mat = mat.coalesce()
        return mat

##----------------------------------------------------------------------------.
class GeneralMaxValPool(RemapBlock):
    """Generalized Max Value Pooling."""
    
    def forward(self, x, *args, **kwargs):
        """Perform pooling."""
        n_batch, n_nodes, n_val = x.shape
        matrix = self.remap_matrix
        new_nodes, old_nodes = matrix.shape
        assert n_nodes == old_nodes, 'remap_matrix.shape[1] != x.shape[1]'
        x = x.permute(1, 2, 0).reshape(n_nodes, n_batch * n_val)

        indices = matrix.indices()
        row, col = indices
        weights = matrix.values()

        cnt = Counter([i.item() for i in row])
        kernel_sizes = [cnt[i] for i in sorted(cnt)]

        col = col.repeat(n_batch * n_val, 1).T

        val = torch.gather(x, dim=0, index=col).detach()
        val.requires_grad_(False)
        weighted_val = weights.view(-1, 1) * val

        start_row = 0
        max_val_index = []
        for k in kernel_sizes:
            curr = weighted_val[start_row:start_row+k]
            max_val_index.append(torch.argmax(curr, dim=0) + start_row)
            start_row += k
        max_val_index = torch.stack(max_val_index)
        nnz_row = torch.gather(col, dim=0, index=max_val_index)

        x_pooled = torch.gather(x, dim=0, index=nnz_row)

        _, nnz_col = np.indices(x_pooled.shape)
        nnz_col = torch.LongTensor(nnz_col).to(nnz_row.device)
        nnz_ind = torch.stack([nnz_row, nnz_col], dim=2)
        nnz_ind = nnz_ind.permute(1, 0, 2).reshape(-1, 2).T
        nnz_ind.requires_grad_(False)

        x_pooled = x_pooled.reshape(new_nodes, n_val, n_batch).permute(2, 0, 1)

        return x_pooled, nnz_ind

class GeneralMaxValUnpool(RemapBlock):
    """Generalized Max Value Unpooling."""
    
    def forward(self, x, index, *args, **kwargs):
        """Perform unpooling."""
        matrix = self.remap_matrix

        n_batch, _, n_val = x.shape
        new_nodes, _ = matrix.shape

        x = x.permute(2, 0, 1).flatten()
        x_unpooled = torch.zeros([new_nodes, n_batch * n_val], dtype=x.dtype, device=x.device)
        row, col = index
        x_unpooled = torch.index_put(x_unpooled, (row, col), x)
        x_unpooled = x_unpooled.reshape(new_nodes, n_val, n_batch).permute(2, 0, 1)
        return x_unpooled
    
##----------------------------------------------------------------------------.
class GeneralLearnableUnpool(SparseLinear):
    """Generalized Learnable Unooling."""
    
    def __init__(self, remap_matrix: "sparse.coo.coo_matrix"):
        out_feature, in_feature = remap_matrix.shape
        bias = False
        indices = np.empty((2, remap_matrix.nnz), dtype=np.int64)
        np.stack((remap_matrix.row, remap_matrix.col), axis=0, out=indices)
        indices = torch.from_numpy(indices)
        dynamic = False
        super().__init__(in_feature, out_feature, bias, connectivity=indices, dynamic=dynamic)
    
    def forward(self, x, *args, **kwargs):
        """Perform unpooling."""
        x = x.permute(0, 2, 1)
        x = super().forward(x)
        x = x.permute(0, 2, 1)
        return x
    
class GeneralLearnablePool(GeneralLearnableUnpool):  
    """Generalized Learnable Pooling."""
    
    def forward(self, x):
        """Perform pooling."""
        output = super().forward(x)
        return output, None

#----------------------------------------------------------------------------.
# #################################### 
#### Generalized PoolUnpoolBlock  ####
# ####################################
HEALPIX_POOL = {'max': (HealpixMaxPool, HealpixMaxUnpool), 
                'avg': (HealpixAvgPool, HealpixAvgUnpool)}

EQUIANGULAR_POOl = {'max': (EquiangularMaxPool, EquiangularMaxUnpool), 
                    'avg': (EquiangularAvgPool, EquiangularAvgUnpool)}

ALL_POOL = {'healpix': HEALPIX_POOL,
            'equiangular': EQUIANGULAR_POOl}

class PoolUnpoolBlock(torch.nn.Module):
    """Define Pooling and Unpooling Layers."""
    
    @staticmethod
    def getPoolUnpoolLayer(sampling: str, pool_method: str, **kwargs):
        """Retrieve ad-hoc pooling and unpooling layers for healpix and equiangular."""
        sampling = sampling.lower()
        pool_method = pool_method.lower()
        assert sampling in ('healpix', 'equiangular')
        assert pool_method in ('max', 'avg')

        pooling, unpool = ALL_POOL[sampling][pool_method]
        return pooling(**kwargs), unpool(**kwargs)
    
    @staticmethod
    def getGeneralPoolUnpoolLayer(src_graph, dst_graph, pool_method: str):
        """Retrieve general pooling and unpooling layers."""
        if src_graph.n_vertices < dst_graph.n_vertices:
            src_graph, dst_graph = dst_graph, src_graph
        
        pool_mat, unpool_mat = build_pooling_matrices(src_graph, dst_graph)
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
        elif pool_method == 'maxval':
            pool = GeneralMaxValPool(pool_mat)
            unpool = GeneralMaxValUnpool(unpool_mat)
            return pool, unpool
        else:
            raise ValueError(f'{pool_method} is not supoorted.')

#----------------------------------------------------------------------------.
# ############################## 
#### Generalized ConvBlock  ####
# ############################## 
def get_conv_fun(conv_type):
    """Retrieve the ConvLayer based on conv_type."""
    ALL_CONV = {'image': Conv2dEquiangular,
                'graph': ConvCheb}
    return ALL_CONV[conv_type]

class GeneralConvBlock(ABC, torch.nn.Module):
    """Define General ConvBlock class."""
    
    @abstractmethod    
    def forward(self, *args, **kwargs):
        """Define forward pass of a ConvBlock."""
        pass
    
    @staticmethod
    def getConvLayer(in_channels: int,
                     out_channels: int,
                     kernel_size: int,
                     conv_type: str = 'graph',
                     **kwargs):
        """Retrieve the required ConvLayer."""        
        conv_type = conv_type.lower()
        conv = None
        if conv_type == 'graph':
            assert 'laplacian' in kwargs
            _ = kwargs.pop('lonlat_ratio')
            _ = kwargs.pop('periodic_padding')
            conv = get_conv_fun(conv_type)(in_channels, out_channels, kernel_size, **kwargs)
        elif conv_type == 'image':
            assert 'lonlat_ratio' in kwargs
            conv = get_conv_fun(conv_type)(in_channels, out_channels, kernel_size, **kwargs)
        else:
            raise ValueError("{} conv_type is not supported. Choose either 'graph' or 'image'".format(conv_type))
        return conv
