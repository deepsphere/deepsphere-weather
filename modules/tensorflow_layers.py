#!/usr/bin/env python3

r"""
TensorFlow implementation of a convolutional neural network on graphs based on
Chebyshev polynomials of the graph Laplacian.
See https://arxiv.org/abs/1606.09375 for details.
Copyright 2018 Michaël Defferrard.
Released under the terms of the MIT license.
"""

import math

import numpy as np
from scipy import sparse
import scipy.sparse.linalg
import tensorflow as tf


def prepare_laplacian(laplacian):
    r"""Prepare a graph Laplacian to be fed to a graph convolutional layer."""

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

    # TensorFlow expects the indices to be in row-major order.
    # Enforced by scipy CSR as tf.sparse_reorder() is not available on GPU.
    laplacian = sparse.csr_matrix(laplacian, copy=False)

    laplacian = sparse.coo_matrix(laplacian)

    # Indices can be int32 or int64.
    indices = np.stack((laplacian.row, laplacian.col), axis=1)

    # https://stackoverflow.com/questions/48570140/difference-between-sparsetensor-and-sparsetensorvalue
    return tf.SparseTensorValue(indices, laplacian.data, laplacian.shape)

'''
# State-less function.
def cheb_conv(laplacian, x, weight):
    r"""Convolution on graph with Chebyshev polynomials."""
    n_features_in, kernel_size, n_features_out = weight.get_shape()
    x = [x]  # shape: #nodes x #in_channels
    if kernel_size > 1:
        x.append(tf.sparse_tensor_dense_matmul(laplacian, x[0]))
    for k in range(2, kernel_size):
        x.append(2 * tf.sparse_tensor_dense_matmul(laplacian, x[k-1]) - x[k-2])
    x = tf.stack(x, axis=2)  # shape: #nodes x #in_channels x kernel_size
    x = tf.reshape(x, [-1, n_features_in*kernel_size])
    weight = tf.reshape(weight, [-1, n_features_out])
    return tf.matmul(x, weight)  # shape: #nodes x #out_channels
'''

# State-less function.
def cheb_conv(laplacian, x, weight):
    r"""Convolution on graph with Chebyshev polynomials."""
    n_features_in, kernel_size, n_features_out = weight.get_shape()
    batch_size, n_vertex, _ = x.get_shape()
    
    # transform to Chebyshev basis
    # Input shape: 
    x0 = tf.transpose(x, perm=[1, 2, 0])  # n_vertex x n_features x batch_size
    x0 = tf.reshape(x0, [n_vertex, -1])  # n_vertex x n_features*batch_size
    x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
    
    def concat(x, x_):
        x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
        return tf.concat([x, x_], axis=0)  # K x M x Fin*N
    if kernel_size > 1:
        x1 = tf.sparse_tensor_dense_matmul(laplacian, x0)
        x = concat(x, x1)
    for k in range(2, kernel_size):
        x2 = 2 * tf.sparse_tensor_dense_matmul(laplacian, x1) - x0  # M x Fin*N
        x = concat(x, x2)
        x0, x1 = x1, x2

    x = tf.reshape(x, [kernel_size, n_vertex, n_features_in, -1])  # K x M x Fin x N
    x = tf.transpose(x, perm=[3, 1, 2, 0])
    x = tf.reshape(x, [-1, n_features_in*kernel_size])  # N*M x Fin*K
    # Filter: Fin*Fout filters of order K, i.e. one filterbank per output feature.
    weight = tf.reshape(weight, [-1, n_features_out])
    x = tf.matmul(x, weight)  # N*M x Fout
    return tf.reshape(x, [-1, n_vertex, n_features_out])  # N x M x Fout

# State-full class.
class ChebConv():
    """Graph convolutional layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
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

    def __init__(self, in_channels, out_channels, kernel_size, name, bias=True, conv=cheb_conv):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = True if bias else None
        self._conv = conv
        self.name = name
        self.reset_parameters()

    def __repr__(self):
        s = ('{in_channels} -> {out_channels}, kernel_size={kernel_size}')
        s += ', bias=' + str(self.bias is not None)
        s = s.format(**self.__dict__)
        return f'{self.__class__.__name__}({s})'

    def reset_parameters(self, activation='relu', fan='avg',
                         distribution='uniform'):
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
            init = tf.initializers.random_normal(0, std, dtype=tf.float32)
            # init = tf.initializers.truncated_normal(0, std, dtype=tf.float32)
        elif distribution == 'uniform':
            lim = math.sqrt(3 * scale / fan)
            init = tf.initializers.random_uniform(-lim, lim, dtype=tf.float32)
        else:
            raise ValueError('unknown distribution')
            
        
        self.weight = self.add_weight(
            'weight_'+self.name,
            (self.in_channels, self.kernel_size, self.out_channels),
            tf.float32,
            init,
        )

        if self.bias is not None:
            self.bias = tf.get_variable(
                'bias_'+self.name,
                self.out_channels,
                tf.float32,
                tf.initializers.zeros(tf.float32),
            )
          
        '''
        weight = tf.get_variable(
            'weight_'+self.name,
            (self.in_channels, self.kernel_size, self.out_channels),
            tf.float32,
            init,
        )
        self.weight.assign(weight)
        
        if self.bias is not None:
            bias = tf.get_variable(
                    'bias_'+self.name,
                    self.out_channels,
                    tf.float32,
                    tf.initializers.zeros(tf.float32),)
            self.bias.assign(bias)
        '''


    def set_parameters(self, weight, bias=None):
        r"""Set weight and bias.

        Parameters
        ----------
        weight : array of shape in_channels x kernel_size x out_channels
            The coefficients of the Chebyshev polynomials.
        bias : vector of length out_channels
            The bias.
        """
        self.weight.assign(weight)
        if bias is not None:
            self.bias.assign(bias)

    def __call__(self, laplacian, inputs):
        r"""Forward graph convolution.

        Parameters
        ----------
        laplacian : sparse matrix of shape n_nodes x n_nodes
            Encode the graph structure.
        inputs : tensor of shape n_nodes x n_features
            Data, i.e. node features.
        """
        outputs = self._conv(laplacian, inputs, self.weight)
        if self.bias is not None:
            outputs += self.bias
        return outputs
