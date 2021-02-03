import math
from collections import Counter

import numpy as np
from scipy import sparse
import torch
from torch.nn import functional as F
from torch.nn import Conv1d
from modules import remap
from sparselinear import SparseLinear


def reformat(x):
    """Reformat the input from a 4D tensor to a 3D tensor
    Args:
        x (:obj:`torch.tensor`): a 4D tensor
    Returns:
        :obj:`torch.tensor`: a 3D tensor
    """
    x = x.permute(0, 2, 3, 1)
    N, D1, D2, Feat = x.size()
    x = x.view(N, D1 * D2, Feat)
    return x


def equiangular_dimension_unpack(nodes, ratio):
    """Calculate the two underlying dimensions
    from the total number of nodes
    Args:
        nodes (int): combined dimensions
        ratio (float): ratio between the two dimensions
    Returns:
        int, int: separated dimensions
    """
    dim1 = int((nodes / ratio) ** 0.5)
    dim2 = int((nodes * ratio) ** 0.5)
    if dim1 * dim2 != nodes: # Try to correct dim1 or dim2 if ratio is wrong
        if nodes % dim1 == 0:
            dim2 = nodes // dim1
        if nodes % dim2 == 0:
            dim1 = nodes // dim2
    assert dim1 * dim2 == nodes, f'Unable to unpack nodes: {nodes}, ratio: {ratio}'
    return dim1, dim2


def equiangular_calculator(tensor, ratio):
    N, M, F = tensor.size()
    dim1, dim2 = equiangular_dimension_unpack(M, ratio)
    # bw_dim1, bw_dim2 = dim1 / 2, dim2 / 2
    tensor = tensor.view(N, dim1, dim2, F)
    return tensor


# 2D CNN layers
class Conv2dEquiangular(torch.nn.Module):
    """ Equiangular 2D Convolutional layer, periodic in the longitude (width) dimension.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels in the output image.
    kernel_size : int
        Width of the square convolutional kernel.
        The actual size of the kernel is kernel_size**2
    ratio : int
        ratio = H // W. Aspect ratio to reorganize the equiangular map from 1D to 2D
    periodic : bool
        whether to use periodic padding. (default: True)
    """

    def __init__(self, in_channels, out_channels, kernel_size, ratio, periodic, **kwargs):
        super().__init__()
        self.ratio = ratio
        self.periodic = periodic

        self.kernel_size = kernel_size
        self.pad_width = int((self.kernel_size - 1) / 2)

        self.conv = torch.nn.Conv2d(in_channels, out_channels, self.kernel_size)

        torch.nn.init.xavier_uniform_(self.conv.weight)
        torch.nn.init.zeros_(self.conv.bias)

    def periodicPad(self, x, width, periodic=True):
        """ Periodic padding function.

        Parameters
        ----------
        x : torch.Tensor
            The tensor format should be N * C * H * W
        width : int
            The padding width.
        periodic: bool
            Whether to use periodic padding. If False, the function is same as ZeroPad2D. (default: True)
        """
        if periodic:
            x = torch.cat((x[:, :, :, -width:], x, x[:, :, :, :width]), dim=3)
            padded = F.pad(x, (0, 0, width, width), 'constant', 0)
        else:
            padded = F.pad(x, (width, width, width, width), 'constant', 0)
        return padded

    def forward(self, x):
        x = equiangular_calculator(x, self.ratio)
        x = x.permute(0, 3, 1, 2)

        x = self.periodicPad(x, self.pad_width, self.periodic)
        x = self.conv(x)

        x = reformat(x)
        return x



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
        lmax *= 1 + 2 * tol  # Be robust to errors.
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
    # print('B: {}, V. {}, Fin: {}'.format(B,V,Fin1))
    Fin, K, Fout = weight.shape
    # print('Fin: {}, K: {}, Fout: {}'.format(Fin, K, Fout))
    assert Fin1 == Fin
    # B = batch size
    # V = nb vertices
    # Fin = nb input features
    # Fout = nb output features
    # K = order of Chebyshev polynomials (kenel size)

    # transform to Chebyshev basis    
    x0 = inputs.permute(1, 2, 0).contiguous()  # V x Fin x B
    x0 = x0.view([V, Fin * B])  # V x Fin*B
    x = x0.unsqueeze(0)  # 1 x V x Fin*B

    if K > 1:
        x1 = torch.sparse.mm(laplacian, x0)  # V x Fin*B
        x = torch.cat((x, x1.unsqueeze(0)), 0)  # 2 x V x Fin*B
    for _ in range(2, K):
        x2 = 2 * torch.sparse.mm(laplacian, x1) - x0
        x = torch.cat((x, x2.unsqueeze(0)), 0)  # M x Fin*B
        x0, x1 = x1, x2

    x = x.view([K, V, Fin, B])  # K x V x Fin x B
    x = x.permute(3, 1, 2, 0).contiguous()  # B x V x Fin x K
    x = x.view([B * V, Fin * K])  # B*V x Fin*K

    # Linearly compose Fin features to get Fout features
    weight = weight.view(Fin * K, Fout)
    x = x.matmul(weight)  # B*V x Fout
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
                 conv=cheb_conv, **kwargs):
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


# Conv layers
class Conv1dAuto(Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2)  # dynamic add padding based on the kernel_size



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

    def __init__(self, ratio, kernel_size, return_indices=True, *args, **kwargs):
        self.ratio = ratio
        kernel_size = int(kernel_size ** 0.5)
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
        x = equiangular_calculator(inputs, self.ratio)
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

    def __init__(self, ratio, kernel_size, *args, **kwargs):
        self.ratio = ratio
        kernel_size = int(kernel_size ** 0.5)
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
        x = equiangular_calculator(inputs, self.ratio)
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

    def __init__(self, ratio, kernel_size, *args, **kwargs):
        self.ratio = ratio
        kernel_size = int(kernel_size ** 0.5)
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
        x = equiangular_calculator(inputs, self.ratio)
        x = x.permute(0, 3, 1, 2)
        x = F.avg_pool2d(x, self.kernel_size)
        x = reformat(x)

        return x, None


class UnpoolAvgEquiangular(torch.nn.Module):
    """EquiAngular average unpooling
    
    Parameters
    ----------
    ratio : float
        Parameter for equiangular sampling -> width/height
    """

    def __init__(self, ratio, kernel_size, *args, **kwargs):
        self.ratio = ratio
        self.kernel_size = int(kernel_size ** 0.5)
        super().__init__()

    def forward(self, inputs, *args):
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

        x = equiangular_calculator(inputs, self.ratio)
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

    def __init__(self, kernel_size, return_indices=True, *args, **kwargs):
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

    def __init__(self, kernel_size, *args, **kwargs):
        """kernel_size should be 4, 16, 64, etc."""
        super().__init__()
        self.kernel_size = kernel_size

    def extra_repr(self):
        return 'kernel_size={kernel_size}'.format(**self.__dict__)

    def forward(self, x):
        """x has shape (batch, pixels, channels) and is in nested ordering"""
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.avg_pool1d(x, self.kernel_size)
        return x.permute(0, 2, 1), None


class UnpoolAvgHealpix(torch.nn.Module):
    """Healpix Average Unpooling module
    
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
        return 'kernel_size={kernel_size}'.format(**self.__dict__)

    def forward(self, x, *args):
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

    def __init__(self, kernel_size, *args, **kwargs):
        super().__init__(kernel_size=kernel_size)

    def forward(self, x, indices, **kwargs):
        """calls pytorch's unpool1d function to create the values while unpooling based on the nearby values
        Parameters
        ----------
        **kwargs
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


def _build_interpolation_matrix(src_graph, dst_graph):
    """Return the sparse matrix that interpolates between two spherical samplings."""

    ds = remap.compute_interpolation_weights(src_graph, dst_graph, method='conservative', normalization='fracarea') # destarea’

    # Sanity checks.
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

    # Destination pixels are normalized to 1 (row-sum = 1).
    # Weights represent the fractions of area attributed to source pixels.
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    # Interpolation is conservative: it preserves area.
    np.testing.assert_allclose(weights.T @ ds.dst_grid_area, ds.src_grid_area)

    # Unnormalize.
    weights = weights.multiply(ds.dst_grid_area.values[:, np.newaxis])

    # Another way to assert that the interpolation is conservative.
    np.testing.assert_allclose(np.asarray(weights.sum(1)).squeeze(), ds.dst_grid_area)
    np.testing.assert_allclose(np.asarray(weights.sum(0)).squeeze(), ds.src_grid_area)

    return weights


def build_pooling_matrices(src_graph, dst_graph):
    weights = _build_interpolation_matrix(src_graph, dst_graph)
    pool = weights.multiply(1/weights.sum(1))
    unpool = weights.multiply(1/weights.sum(0)).T
    return pool, unpool


def convert_to_torch_sparse(mat: "sparse.coo.coo_matrix"):
    indices = np.empty((2, mat.nnz), dtype=np.int64)
    np.stack((mat.row, mat.col), axis=0, out=indices)
    indices = torch.from_numpy(indices)
    mat = torch.sparse_coo_tensor(indices, mat.data, mat.shape, dtype=torch.float32)
    mat = mat.coalesce()
    return mat


class RemapBlock(torch.nn.Module):
    def __init__(self, remap_matrix: "sparse.coo.coo_matrix"):
        super().__init__()
        remap_matrix = self.process_remap_matrix(remap_matrix)
        self.register_buffer('remap_matrix', remap_matrix)
    
    def forward(self, x, *args, **kwargs):
        n_batch, n_nodes, n_val = x.shape
        matrix = self.remap_matrix
        new_nodes, _ = matrix.shape
        x = x.permute(1, 2, 0).reshape(n_nodes, n_batch * n_val)
        x = torch.sparse.mm(matrix, x)
        x = x.reshape(new_nodes, n_val, n_batch).permute(2, 0, 1)
        return x
    
    def process_remap_matrix(self, mat):
        return convert_to_torch_sparse(mat)


class GeneralAvgPool(RemapBlock):
    def forward(self, x, *args, **kwargs):
        x = super().forward(x, *args, **kwargs)
        # Some pooling methods (e.g. Avg) do not give source indices, please return None for compatibility reason
        return x, None


class GeneralAvgUnpool(RemapBlock):
    def forward(self, x, *args, **kwargs):
        x = super().forward(x, *args, **kwargs)
        return x


class GeneralMaxAreaPool(RemapBlock):
    def forward(self, x, *args, **kwargs):
        x = super().forward(x, *args, **kwargs)
        # Some pooling methods (e.g. Avg) do not give source indices, please return None for compatibility reason
        return x, None
        
    def process_remap_matrix(self, mat):
        max_ind_col = np.argmax(mat, axis=1).T
        row = np.arange(max_ind_col.shape[1]).reshape(1, -1)
        indices = np.concatenate([row, max_ind_col], axis=0)
        indices = torch.from_numpy(indices.astype(np.int64))
        mat = torch.sparse_coo_tensor(indices, torch.ones(indices.shape[1]), mat.shape, dtype=torch.float32)
        mat = mat.coalesce()
        return mat


class GeneralMaxAreaUnpool(RemapBlock):
    def process_remap_matrix(self, mat):
        max_ind_row = np.argmax(mat, axis=0)
        col = np.arange(max_ind_row.shape[1]).reshape(1, -1)
        indices = np.concatenate([max_ind_row, col], axis=0)
        indices = torch.from_numpy(indices.astype(np.int64))
        mat = torch.sparse_coo_tensor(indices, torch.ones(indices.shape[1]), mat.shape, dtype=torch.float32)
        mat = mat.coalesce()
        return mat


class GeneralLearnableUnpool(SparseLinear):
    def __init__(self, remap_matrix: "sparse.coo.coo_matrix"):
        out_feature, in_feature = remap_matrix.shape
        bias = False
        indices = np.empty((2, remap_matrix.nnz), dtype=np.int64)
        np.stack((remap_matrix.row, remap_matrix.col), axis=0, out=indices)
        indices = torch.from_numpy(indices)
        dynamic = False
        super().__init__(in_feature, out_feature, bias, connectivity=indices, dynamic=dynamic)
    
    def forward(self, x, *args, **kwargs):
        x = x.permute(0, 2, 1)
        x = super().forward(x)
        x = x.permute(0, 2, 1)
        return x


class GeneralLearnablePool(GeneralLearnableUnpool):    
    def forward(self, inputs):
        output = super().forward(inputs)
        return output, None


class GeneralMaxValPool(RemapBlock):
    def forward(self, x, *args, **kwargs):
        n_batch, n_nodes, n_val = x.shape
        matrix = self.remap_matrix
        new_nodes, old_nodes = matrix.shape
        assert n_nodes == old_nodes, 'remap_matrix.shape[1] != input.shape[1]'
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
    def forward(self, x, index, *args, **kwargs):
        matrix = self.remap_matrix

        n_batch, _, n_val = x.shape
        new_nodes, _ = matrix.shape

        x = x.permute(2, 0, 1).flatten()
        x_unpooled = torch.zeros([new_nodes, n_batch * n_val], dtype=x.dtype, device=x.device)
        row, col = index
        x_unpooled =  torch.index_put(x_unpooled, (row, col), x)
        x_unpooled = x_unpooled.reshape(new_nodes, n_val, n_batch).permute(2, 0, 1)
        return x_unpooled