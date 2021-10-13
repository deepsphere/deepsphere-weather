#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 17:51:40 2021

@author: ghiggi
"""
import torch
import numpy as np
from typing import Dict
from torch.nn import Identity
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch.nn import functional as F       
from modules.models import UNet 
from modules.layers import GeneralConvBlock, PoolUnpoolBlock
from modules.utils_models import check_sampling
from modules.utils_models import check_conv_type
from modules.utils_models import check_pool_method
from modules.utils_models import check_skip_connection
from modules.utils_models import pygsp_graph_coarsening

# https://towardsdatascience.com/resnet-with-identity-mapping-over-1000-layers-reached-image-classification-bb50a42af03e 

##----------------------------------------------------------------------------.
class ConvBlock(GeneralConvBlock):
    """Spherical graph convolution block.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input graph.
    out_channels : int
        Number of channels in the output graph.
    laplacian : TYPE
        DESCRIPTION
    kernel_size : int
        Chebychev polynomial degree
    conv_type : str, optional
        'graph' or 'image'. The default is 'graph'.
        'image' can be used only when sampling='equiangular'
    bias : bool, optional
        Whether to add bias parameters. The default is True.
        If batch_norm = True, bias is set to False.
    batch_norm : bool, optional
        Wheter to use batch normalization. The default is False.
    batch_norm_before_activation : bool, optional
        Whether to apply the batch norm before or after the activation function.
        The default is False.
    activation : bool, optional
        Whether to apply an activation function. The default is True.
    activation_fun : str, optional
        Name of an activation function implemented in torch.nn.functional
        The default is 'relu'.
    periodic_padding : bool, optional
        Matters only if sampling='equiangular' and conv_type='image'.
        whether to use periodic padding along the longitude dimension. The default is True.
    lonlat_ratio : int
        Matters only if sampling='equiangular' and conv_type='image.
        Aspect ratio to reshape the input 1D data to a 2D image.
        lonlat_ratio = H // W = n_longitude rings / n_latitude rings
        A ratio of 2 means the equiangular grid has the same resolution.
        in latitude and longitude.
    """

    def __init__(self,
                 in_channels, 
                 out_channels, 
                 laplacian,
                 kernel_size = 3,
                 conv_type = 'graph', 
                 bias = True, 
                 batch_norm = False, 
                 batch_norm_before_activation = False,
                 activation = True,
                 activation_fun = 'relu',
                 periodic_padding = True, 
                 lonlat_ratio = 2):
        
        super().__init__()
        # If batch norm is used, set conv bias = False  
        if batch_norm: 
            bias = False
        # Define convolution 
        self.conv = GeneralConvBlock.getConvLayer(in_channels=in_channels,
                                                  out_channels=out_channels, 
                                                  kernel_size=kernel_size, 
                                                  laplacian=laplacian,
                                                  conv_type=conv_type,
                                                  bias=bias,
                                                  periodic_padding=periodic_padding,
                                                  lonlat_ratio = lonlat_ratio)
        if batch_norm:
            self.bn = BatchNorm1d(out_channels)
        self.bn_before_act = batch_norm_before_activation
        self.norm = batch_norm
        self.act = activation
        self.act_fun = getattr(F, activation_fun)
        
    def forward(self, x):
        """Define forward pass of a ConvBlock.
        
        It expect a tensor with shape: (sample, nodes, time-feature).
        """
        x = self.conv(x)
        if self.norm and self.bn_before_act:   
            # [batch, node, time-feature]
            x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.act:
            x = self.act_fun(x)
        if self.norm and not self.bn_before_act:   
            # [batch, node, time-feature]
            x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x
        
class ResBlock(torch.nn.Module):
    """
    General definition of a Residual Block. 
    
    Parameters
    ----------
    in_channels : int
        Input dimension of the tensor.   
    out_channels :(int, tuple)
        Output dimension of each ConvBlock within a ResBlock.
        The length of the tuple determine the number of ConvBlock layers 
        within the ResBlock. 
    laplacian : TYPE
        DESCRIPTION.
    convblock_kwargs : TYPE
        Arguments for the ConvBlock layer.
    """    
    def __init__(self,
                 in_channels, 
                 out_channels, 
                 laplacian, 
                 convblock_kwargs,
                 **kwargs):
        super().__init__()
        ##--------------------------------------------------------
        # ResBlock options
        self.rezero = True 
        ##--------------------------------------------------------
        # Check out_channels type (if an int --> a single conv)
        if not isinstance(out_channels, (int, tuple, list)):
            raise TypeError("'output_channels' must be int or list/tuple of int.")
        if not isinstance(out_channels, (tuple, list)): 
            out_channels = [out_channels]
        out_channels = list(out_channels)
        ##--------------------------------------------------------
        ### Define list of convolutional layer
        # - Initialize list of convolutional layer within the ResBlock 
        conv_names_list = []
        tmp_in = in_channels
        n_layers = len(out_channels)
        for i, tmp_out in enumerate(out_channels):
            tmp_convblock_kwargs = convblock_kwargs.copy()
            #--------------------------------------------------------.
            # - If last conv layer, do not add the activation function 
            # --> If act_fun like Relu, the ResBlock can only output positive increments
            if i == n_layers -1: 
                tmp_convblock_kwargs["activation"] = False
            #--------------------------------------------------------.    
            # - If last conv layer, do not add the batch_norm (or only add on the last?)
            # TODO:

            #--------------------------------------------------------.
            # - Define conv layer name 
            tmp_conv_name = 'convblock' + str(i+1)
            #--------------------------------------------------------.
            # - Create the conv layer
            tmp_conv = ConvBlock(tmp_in, tmp_out,  
                                laplacian = laplacian, 
                                **tmp_convblock_kwargs)
            setattr(self, tmp_conv_name, tmp_conv)
            conv_names_list.append(tmp_conv_name)
            tmp_in = tmp_out   
            
        self.conv_names_list = conv_names_list 
        ##--------------------------------------------------------
        ### Define the residual connection 
        if in_channels == out_channels[-1]: 
            self.res_connection = Identity()
        else:
            self.res_connection = Linear(in_channels, out_channels[-1])
        
        ##--------------------------------------------------------
        ### Define multiplier initialized at 0 for ReZero trick 
        if self.rezero:
            self.rezero_weight = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        
        ##--------------------------------------------------------
        # Zero-initialize the last BN in each residual branch
        # --> Each ResBlock behave like identity if input_channels == output_channels
        if convblock_kwargs["batch_norm"]:
            last_convblock = getattr(self, conv_names_list[-1])          
            torch.nn.init.constant_(last_convblock.bn.weight, 0)
            torch.nn.init.constant_(last_convblock.bn.bias, 0)
    
        ##--------------------------------------------------------

    def forward(self, x):
        """Define forward pass of a ResBlock."""
        # Perform convolutions 
        x_out = x  
        for conv_name in self.conv_names_list:
            x_out = getattr(self, conv_name)(x_out)
        # Rezero trick
        if self.rezero:
            x_out *= self.rezero_weight 
        # Add residual connection 
        x_out += self.res_connection(x)
        return x_out

##----------------------------------------------------------------------------.     
class UNetSpherical(UNet, torch.nn.Module):
    """Spherical UNet with residual layers.
        
    Parameters
    ----------
    tensor_info: dict
        Dictionary with all relevant shape, dimension and feature order informations 
        regarding input and output tensors.
    sampling : str
        Name of the spherical sampling.
    sampling_kwargs : int
        Arguments to define the spherical pygsp graph.
    conv_type : str, optional
        Convolution type. Either 'graph' or 'image'.
        The default is 'graph'.
        conv_type='image' can be used only when sampling='equiangular'.
    knn : int 
        DESCRIPTION
    graph_type : str , optional 
        DESCRIPTION
       'voronoi' or 'knn'.
       'knn' build a knn graph.
       'voronoi' build a voronoi mesh graph and require the igl package
        The default is 'knn'.
    kernel_size_conv : int
        Size ("width") of the convolutional kernel.
        If conv_type='graph':
        - A kernel_size of 1 won't take the neighborhood into account.
        - A kernel_size of 2 will look up to the 1-neighborhood (1 hop away).
        - A kernel_size of 3 will look up to the 2-neighborhood (2 hops away). 
        --> The order of the Chebyshev polynomials is kernel_size_conv - 1.
        If conv_type='image':
        - Width of the square convolutional kernel.
        - The number of pixels of the kernel is kernel_size_conv**2
    bias : bool, optional
        Whether to add bias parameters. The default is True.
        If batch_norm = True, bias is set to False.
    batch_norm : bool, optional
        Wheter to use batch normalization. The default is False.
    batch_norm_before_activation : bool, optional
        Whether to apply the batch norm before or after the activation function.
        The default is False.
    activation : bool, optional
        Whether to apply an activation function. The default is True.
    activation_fun : str, optional
        Name of an activation function implemented in torch.nn.functional
        The default is 'relu'.
    pool_method : str, optional 
        Pooling method:
        - ('interp', 'maxval', 'maxarea', 'learn') if conv_type='graph' 
        - ('max','avg') if sampling in ('healpix','equiangular') or conv_type="image"        
    kernel_size_pooling : int, optional 
        The size of the window to max/avg over.
        kernel_size_pooling = 4 means halving the resolution when pooling
        The default is 4.
    skip_connection : str, optional 
        Possibilities: 'none','stack','sum','avg'
        The default is 'stack.
    increment_learning: bool, optional 
        If increment_learning = True, the network is forced internally to learn the increment
        from the previous timestep.
        If increment_learning = False, the network learn the full state.
        The default is False.
    periodic_padding : bool, optional
        Matters only if sampling='equiangular' and conv_type='image'.
        whether to use periodic padding along the longitude dimension. The default is True.
    """

    def __init__(self,
                 tensor_info: Dict,
                 sampling: str,
                 sampling_kwargs: Dict, 
                 # Convolutions options 
                 kernel_size_conv: int = 3,
                 conv_type: str = "graph",
                 graph_type: str = 'knn', 
                 knn: int = 20, 
                 # Options for classical image convolution on equiangular sampling 
                 periodic_padding: bool = True, 
                 # ConvBlock Options 
                 bias: bool = True, 
                 batch_norm: bool = False, 
                 batch_norm_before_activation: bool = False,
                 activation: bool = True,
                 activation_fun: str = 'relu',
                 # Pooling options 
                 pool_method: str = 'max', 
                 kernel_size_pooling: int = 4,
                 # Architecture options 
                 skip_connection: str = 'stack',
                 increment_learning: bool = False):
        # TODO: code for flexible skip_connection not yet implemented
        ##--------------------------------------------------------------------.
        super().__init__()
        ##--------------------------------------------------------------------.
        # Retrieve tensor informations 
        self.dim_names = tensor_info['dim_order']['dynamic']  
        
        # dim_info = tensor_info['dynamic']['dim_info']  
        # feature_static = tensor_info['feature_order']['static']
        # feature_bc = tensor_info['feature_order']['bc']
        # feature_dynamic = tensor_info['feature_order']['dynamic']
        # input_features =  feature_static + feature_bc + feature_dynamic
        # output_features = feature_dynamic 
        
        self.input_n_feature = tensor_info['input_n_feature']
        self.output_n_feature = tensor_info['output_n_feature']
        self.input_n_time = tensor_info['input_n_time']
        self.output_n_time = tensor_info['output_n_time']
        self.input_n_node = tensor_info['input_shape_info']['dynamic']['node']  
        self.output_n_node = tensor_info['output_shape_info']['dynamic']['node']  
        
        ##--------------------------------------------------------------------.
        # Define size of last dimension for ConvChen conv (merging time-feature dimension)
        self.input_channels = self.input_n_feature*self.input_n_time
        self.output_channels = self.output_n_feature*self.output_n_time
        
        ##--------------------------------------------------------------------.
        # Decide whether to predict the difference from the previous timestep 
        #  instead of the full state 
        self.increment_learning = increment_learning
        ##--------------------------------------------------------------------.
        ### Check arguments 
        sampling = check_sampling(sampling)
        conv_type = check_conv_type(conv_type, sampling)
        pool_method = check_pool_method(pool_method)
        skip_connection = check_skip_connection(skip_connection)
        ##--------------------------------------------------------------------.   
        # Derive lonlat ratio from sampling_kwargs if equiangular 
       if sampling == "equiangular":
            lonlat_ratio = sampling_kwargs['nlon'] / sampling_kwargs['nlat']
        else:
            lonlat_ratio = None
        ##--------------------------------------------------------------------.
        ### Define ConvBlock options 
        convblock_kwargs = {"kernel_size": kernel_size_conv, 
                            "conv_type": conv_type,
                            "bias": bias, 
                            "batch_norm": batch_norm, 
                            "batch_norm_before_activation": batch_norm_before_activation,
                            "activation": activation,
                            "activation_fun": activation_fun,
                            # Options for conv_type = "image", sampling='equiangular'
                            "periodic_padding": periodic_padding, 
                            "lonlat_ratio": lonlat_ratio,
                            }
        ##--------------------------------------------------------------------.
        ### Define graph and laplacian 
        # - Update knn based on model settings 
        sampling_kwargs['k'] = knn  
        # - Define sampling_kwargs for coarsed UNet levels 
        UNet_depth = 3
        coarsening = int(np.sqrt(kernel_size_pooling))
        sampling_list = [sampling]
        sampling_kwargs_list = [sampling_kwargs]
        for i in range(1, UNet_depth):
            sampling_list.append(sampling)
            sampling_kwargs_list.append(pygsp_graph_coarsening(sampling = sampling,
                                                               sampling_kwargs = sampling_kwargs_list[i-1], 
                                                               coarsening = coarsening))
        
        ##--------------------------------------------------------------------.
        ### Initialize graphs and laplacians
        # - If conv_type == 'image', self.laplacians = [None] * UNet_depth
        # - self.init_graph_and_laplacians() defines self.graphs and self.laplacians
        self.init_graph_and_laplacians(sampling_list = sampling_list, 
                                       sampling_kwargs_list = sampling_kwargs_list,
                                       graph_type = graph_type,
                                       conv_type = conv_type)
        
        ##--------------------------------------------------------------------.
        ### Define Pooling - Unpooling layers
        if pool_method in ('interp', 'maxval', 'maxarea', 'learn'):
            assert conv_type == 'graph'
            self.pool1, self.unpool1 = PoolUnpoolBlock.getGeneralPoolUnpoolLayer(src_graph=self.graphs[0], 
                                                                                 dst_graph=self.graphs[1], 
                                                                                 pool_method=pool_method)
            self.pool2, self.unpool2 = PoolUnpoolBlock.getGeneralPoolUnpoolLayer(src_graph=self.graphs[1],
                                                                                 dst_graph=self.graphs[2], 
                                                                                 pool_method=pool_method)
        elif pool_method in ('max', 'avg'):
            assert sampling in ['healpix','equiangular']
            self.pool1, self.unpool1 = PoolUnpoolBlock.getPoolUnpoolLayer(sampling=sampling, 
                                                                          pool_method=pool_method, 
                                                                          kernel_size=kernel_size_pooling,
                                                                          lonlat_ratio=lonlat_ratio)
            
            self.pool2, self.unpool2 = PoolUnpoolBlock.getPoolUnpoolLayer(sampling=sampling, 
                                                                          pool_method=pool_method, 
                                                                          kernel_size=kernel_size_pooling, 
                                                                          lonlat_ratio=lonlat_ratio)
        else:
            if pool_method is not None:
                raise ValueError("Not valid pooling method provided.")
                
        ##--------------------------------------------------------------------.
        ### Define Encoding blocks
        # Initial ResBlock (?)
        
        ##------------------------------------.
        # Encoding block 1
        self.conv1 = ResBlock(self.input_channels, (32 * 2, 64 * 2),
                              laplacian = self.laplacians[0],
                              convblock_kwargs = convblock_kwargs)
               
        # Encoding block 2
        self.conv2 = ResBlock(64 * 2, (96 * 2, 128 * 2),
                              laplacian = self.laplacians[1],
                              convblock_kwargs = convblock_kwargs)
    
        # Encoding block 3
        self.conv3 = ResBlock(128 * 2, (256 * 2, 128 * 2),
                              laplacian = self.laplacians[2],
                              convblock_kwargs = convblock_kwargs)
        
        ##--------------------------------------------------------------------.
        ### Decoding blocks 
        # Decoding level 2
        self.uconv2 = ResBlock(256 * 2, (128 * 2, 64 * 2),
                      laplacian = self.laplacians[1],
                      convblock_kwargs = convblock_kwargs)
                
        # Decoding block 1
        self.uconv1 = ResBlock(128 * 2, (64 * 2, 32 * 2),
                      laplacian = self.laplacians[0],
                      convblock_kwargs = convblock_kwargs)
        ##------------------------------------.
        # Final ResBlock 
        self.uconv1_final = ResBlock(32 * 2, self.output_channels, 
                                     laplacian = self.laplacians[0], 
                                     convblock_kwargs = convblock_kwargs)
        ##--------------------------------------------------------------------.
        # Rezero trick for the full architecture if increment learning ... 
        if self.increment_learning:
            self.res_increment = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        
    ##------------------------------------------------------------------------.
    def encode(self, x):
        """Define UNet encoder."""
        # Recommended optimal shape: [sample, node, time, feature]
        # --> Contiguous inputs can be reshaped without copying --> ['sample', 'node', 'time-feature']
        ##--------------------------------------------------------------------.
        batch_size = x.shape[0]
        ##--------------------------------------------------------------------.
        ## Extract last timestep (to add after decoding) to make the network learn the increment
        x_last_timestep = x[:,-1,:,-2:].unsqueeze(dim=1)
        
        ##--------------------------------------------------------------------.
        # Reorder and reshape data to ['sample', 'node', 'time-feature'] --> [B, V, Fin] for ConvCheb
        x = x.rename(*self.dim_names).align_to('sample','node','time','feature').rename(None)   
        x = x.reshape(batch_size, self.input_n_node, self.input_channels)  # reshape to ['sample', 'node', 'time-feature'] 
        ##--------------------------------------------------------------------.
        ## Define encoder computation
        # Level 1
        x_enc1 = self.conv1(x)

        # Level 2
        x_enc2_ini, idx1 = self.pool1(x_enc1)
        x_enc2 = self.conv2(x_enc2_ini)
    
        # Level 3
        x_enc3_ini, idx2 = self.pool2(x_enc2)
        x_enc3 = self.conv3(x_enc3_ini)
        
        return x_enc3, x_enc2, x_enc1, idx2, idx1, x_last_timestep  
    
    ##------------------------------------------------------------------------.
    def decode(self, x_enc3, x_enc2, x_enc1, idx2, idx1, x_last_timestep): 
        """Define UNet decoder."""
        ## Define decoder computation
        # - Level 2
        x = self.unpool2(x_enc3, idx2)
        x_cat = torch.cat((x, x_enc2), dim=2)
        x = self.uconv2(x_cat)

        # - Level 1
        x = self.unpool1(x, idx1)
        x_cat = torch.cat((x, x_enc1), dim=2)
        x = self.uconv1(x_cat)
        
        # - Level 1 final block 
        x = self.uconv1_final(x)
        
        ##--------------------------------------------------------------------.
        # Reshape data to ['sample', 'time', 'node', 'feature']
        batch_size = x.shape[0]   # 
        # Reshape from ['sample', 'node', 'time-feature']  to ['sample', 'node', 'time', 'feature']
        x = x.reshape(batch_size, self.output_n_node, self.output_n_time, self.output_n_feature)   
        x = x.rename(*['sample', 'node', 'time', 'feature']).align_to(*self.dim_names).rename(None)  
        ##--------------------------------------------------------------------.
        # Add tensor of most recent past timestep
        if self.increment_learning:
            x *= self.res_increment 
            # print(x.shape)
            # print(x_last_timestep.shape)
            x += x_last_timestep
        ##--------------------------------------------------------------------.
        return x
