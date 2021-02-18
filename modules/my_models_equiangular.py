#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 17:51:40 2021

@author: ghiggi
"""
import torch
from typing import List, Union, Dict
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch.nn import functional as F       
from modules.models import UNet 
from modules.layers import GeneralConvBlock, PoolUnpoolBlock
from modules.utils_models import check_sampling
from modules.utils_models import check_resolution
from modules.utils_models import check_pool_method

##----------------------------------------------------------------------------.
class ConvBlock(GeneralConvBlock):
    """Spherical graph convolution block.

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

    def __init__(self,
                 in_channels, 
                 out_channels, 
                 kernel_size,
                 conv_type = 'graph', 
                 batch_norm=True, 
                 relu_activation=True, **kwargs):
        super().__init__()
        # TODO a 
        # - add bias=False to getConvLayer() if batch_norm=True
        # - add option BN_before_act_fun = False --> Add BN before or after act_fun  
        # - replace relu activation with activation_function='relu'
        #   --> self.act = getattr(torch.nn, activation_function)
        
        self.conv = GeneralConvBlock.getConvLayer(in_channels, out_channels,
                                                  kernel_size=kernel_size, 
                                                  conv_type=conv_type, **kwargs)
        self.bn = BatchNorm1d(out_channels)
        self.norm = batch_norm
        self.act = relu_activation
        
    def forward(self, x):
        """Define forward pass of a ConvBlock."""
        x = self.conv(x)
        if self.norm:   # [batch, node, time-feature]
            x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.act:
            x = F.relu(x)
        return x
    
##----------------------------------------------------------------------------.     
class UNetSpherical(UNet, torch.nn.Module):
    """Classical spherical UNet with residual connections."""

    def __init__(self,
                 dim_info: Dict[str, int],
                 resolution: Union[int, List[int]],    
                 kernel_size_conv: int = 3,
                 sampling: str = 'healpix',
                 knn: int = 10, 
                 pool_method: str = 'max', 
                 kernel_size_pooling: int = 4,
                 numeric_precision: str = 'float32'):
        ##--------------------------------------------------------------------.
        super().__init__()
        ##--------------------------------------------------------------------.
        # Retrieve tensor informations 
        self.dim_names = dim_info['dim_order']
        self.input_feature_dim = dim_info['input_feature_dim']
        self.input_time_dim = dim_info['input_time_dim']
        self.input_node_dim = dim_info['input_node_dim'] 
        
        self.output_time_dim = dim_info['output_time_dim']
        self.output_feature_dim = dim_info['output_feature_dim']
        self.output_node_dim = dim_info['output_node_dim']   
        
        self.input_channels = dim_info['input_feature_dim']*dim_info['input_time_dim']
        self.output_channels = dim_info['output_feature_dim']*dim_info['output_time_dim']
   
        ##--------------------------------------------------------------------.
        ### Check arguments 
        sampling = check_sampling(sampling)
        resolution = check_resolution(resolution)
        pool_method = check_pool_method(pool_method)
        
        ##--------------------------------------------------------------------.
        ### Define convolution type 
        # --> For all spherical samplings: conv_type='graph'   
        # --> For equiangular sampling, it is possible to specify conv_type='image'
        # - If conv_type = 'image':
        #   --> ratio = 2
        #   --> periodic = True or False
        conv_type = "image"  
        ratio = 2        
        periodic = True # False 
        
        ##--------------------------------------------------------------------.
        ### Initialize graphs and laplacians
        self.init_graph_and_laplacians(conv_type = conv_type, 
                                       resolution = resolution, 
                                       sampling=sampling, 
                                       knn=knn,
                                       kernel_size_pooling=kernel_size_pooling,
                                       numeric_precision=numeric_precision,
                                       UNet_depth=3)
        
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
        else:
            self.pool1, self.unpool1 = PoolUnpoolBlock.getPoolUnpoolLayer(sampling=sampling, 
                                                                          pool_method=pool_method, 
                                                                          kernel_size=kernel_size_pooling,
                                                                          ratio=ratio)
            
            self.pool2, self.unpool2 = PoolUnpoolBlock.getPoolUnpoolLayer(sampling=sampling, 
                                                                          pool_method=pool_method, 
                                                                          kernel_size=kernel_size_pooling, 
                                                                          ratio=ratio)
            
        ##--------------------------------------------------------------------.
        ### Define Encoding blocks 
        # Encoding block 1
        self.conv11 = ConvBlock(self.input_channels, 32 * 2,  
                                kernel_size=kernel_size_conv, 
                                conv_type=conv_type,
                                batch_norm=True, relu_activation=True,
                                laplacian=self.laplacians[0], 
                                periodic=periodic, ratio=ratio)
        
        self.conv13 = ConvBlock(32 * 2, 64 * 2,  
                                kernel_size=kernel_size_conv,
                                conv_type=conv_type,
                                batch_norm=True, relu_activation=True, 
                                laplacian=self.laplacians[0],
                                periodic=periodic, ratio=ratio)

        self.conv1_res = Linear(self.input_channels, 64 * 2)

        # Encoding block 2
        self.conv21 = ConvBlock(64 * 2, 96 * 2, 
                                kernel_size=kernel_size_conv, 
                                conv_type=conv_type,
                                batch_norm=True, relu_activation=True, 
                                laplacian=self.laplacians[1], 
                                periodic=periodic, ratio=ratio)
        
        self.conv23 = ConvBlock(96 * 2, 128 * 2,  
                                kernel_size=kernel_size_conv, 
                                conv_type=conv_type, 
                                batch_norm=True, relu_activation=True, 
                                laplacian=self.laplacians[1], 
                                periodic=periodic, ratio=ratio)

        self.conv2_res = Linear(64 * 2, 128 * 2)

        # Encoding block 3
        self.conv31 = ConvBlock(128 * 2, 256 * 2, 
                                kernel_size=kernel_size_conv,
                                conv_type=conv_type,
                                batch_norm=True, relu_activation=True, 
                                laplacian=self.laplacians[2], 
                                periodic=periodic, ratio=ratio)
        
        self.conv33 = ConvBlock(256 * 2, 128 * 2,  
                                kernel_size=kernel_size_conv, 
                                conv_type=conv_type, 
                                batch_norm=True, relu_activation=True, 
                                laplacian=self.laplacians[2],
                                periodic=periodic, ratio=ratio)

        self.conv3_res = Linear(128 * 2, 128 * 2)
        
        ##--------------------------------------------------------------------.
        ### Decoding blocks 
        # Decoding block 2
        self.uconv21 = ConvBlock(256 * 2, 128 * 2, 
                                 kernel_size=kernel_size_conv,
                                 conv_type=conv_type, 
                                 batch_norm=True, relu_activation=True,
                                 laplacian=self.laplacians[1], 
                                 periodic=periodic, ratio=ratio)
        self.uconv22 = ConvBlock(128 * 2, 64 * 2,  
                                 kernel_size=kernel_size_conv, 
                                 conv_type=conv_type, 
                                 batch_norm=True, relu_activation=True, 
                                 laplacian=self.laplacians[1], 
                                 periodic=periodic, ratio=ratio)

        # Decoding block 1
        self.uconv11 = ConvBlock(128 * 2, 64 * 2,
                                 kernel_size=kernel_size_conv, 
                                 conv_type=conv_type, 
                                 batch_norm=True, relu_activation=True, 
                                 laplacian=self.laplacians[0],
                                 periodic=periodic, ratio=ratio)
        self.uconv12 = ConvBlock(64 * 2, 32 * 2, 
                                 kernel_size=kernel_size_conv,
                                 conv_type=conv_type, 
                                 batch_norm=True, relu_activation=True, 
                                 laplacian=self.laplacians[0], 
                                 periodic=periodic, ratio=ratio)
        self.uconv13 = ConvBlock(32 * 2 * 2, self.output_channels, 
                                 kernel_size=kernel_size_conv, 
                                 conv_type=conv_type, 
                                 batch_norm=False, relu_activation=False, 
                                 laplacian=self.laplacians[0], 
                                 periodic=periodic, ratio=ratio)
        
    ##------------------------------------------------------------------------.

    def encode(self, x):
        """Define UNet encoder."""
        # Current input shape: ['sample', 'time', 'node', 'feature'] 
        # Desired shape: ['sample', 'node', 'time-feature']
        ##--------------------------------------------------------------------.
        # TODO? 
        # - Provide tensor with sample [sample, node, time, feature]?
        # - Contiguous inputs can be reshaped without copying
        ##--------------------------------------------------------------------.
        batch_size = x.shape[0]
        
        ##--------------------------------------------------------------------.
        # Reorder and reshape data 
        x = x.rename(*self.dim_names).align_to('sample','node','time','feature').rename(None) # x.permute(0, 2, 1, 3)   
        x = x.reshape(batch_size, self.input_node_dim, self.input_channels)  # reshape to ['sample', 'node', 'time-feature'] 
        ##--------------------------------------------------------------------.
        # Block 1
        x_enc11 = self.conv11(x)
        x_enc1 = self.conv13(x_enc11)

        x_enc1 += self.conv1_res(x)

        # Block 2
        x_enc2_ini, idx1 = self.pool1(x_enc1)
        x_enc2 = self.conv21(x_enc2_ini)
        x_enc2 = self.conv23(x_enc2)

        x_enc2 += self.conv2_res(x_enc2_ini)

        # Block 3
        x_enc3_ini, idx2 = self.pool2(x_enc2)
        x_enc3 = self.conv31(x_enc3_ini)
        x_enc3 = self.conv33(x_enc3)

        x_enc3 += self.conv3_res(x_enc3_ini)

        return x_enc3, x_enc2, x_enc1, idx2, idx1, x_enc11  
    
    ##------------------------------------------------------------------------.
    def decode(self, x_enc3, x_enc2, x_enc1, idx2, idx1, x_enc11): 
        """Define UNet decoder."""
        # Block 2
        x = self.unpool2(x_enc3, idx2)
        x_cat = torch.cat((x, x_enc2), dim=2)
        x = self.uconv21(x_cat)
        x = self.uconv22(x)

        # Block 1
        x = self.unpool1(x, idx1)
        x_cat = torch.cat((x, x_enc1), dim=2)
        x = self.uconv11(x_cat)
        x = self.uconv12(x)
        x_cat = torch.cat((x, x_enc11), dim=2)
        x = self.uconv13(x_cat)
        
        ##--------------------------------------------------------------------.
        # Reshape data to ['sample', 'time', 'node', 'feature']
        batch_size = x.shape[0]   # ['sample', 'node', 'time-feature'] 
        x = x.reshape(batch_size, self.output_node_dim, self.output_time_dim, self.output_feature_dim)   # ==> ['sample', 'node', 'time', 'feature']
        x = x.rename(*['sample', 'node', 'time', 'feature']).align_to(*self.dim_names).rename(None) # x.permute(0, 2, 1, 3)  
        return x

class UNetDiffSpherical(UNet, torch.nn.Module):
    """Spherical UNet learning the increment to add the the previous timestep."""

    def __init__(self,
                 dim_info: Dict[str, int],
                 resolution: Union[int, List[int]],    
                 kernel_size_conv: int = 3,
                 sampling: str = 'healpix',
                 knn: int = 10, 
                 pool_method: str = 'max', 
                 kernel_size_pooling: int = 4,
                 numeric_precision: str = 'float32'):
        ##--------------------------------------------------------------------.
        super().__init__()
        ##--------------------------------------------------------------------.
        # Retrieve tensor informations 
        self.dim_names = dim_info['dim_order']
        self.input_feature_dim = dim_info['input_feature_dim']
        self.input_time_dim = dim_info['input_time_dim']
        self.input_node_dim = dim_info['input_node_dim'] 
        
        self.output_time_dim = dim_info['output_time_dim']
        self.output_feature_dim = dim_info['output_feature_dim']
        self.output_node_dim = dim_info['output_node_dim']   
        
        self.input_channels = dim_info['input_feature_dim']*dim_info['input_time_dim']
        self.output_channels = dim_info['output_feature_dim']*dim_info['output_time_dim']
   
        ##--------------------------------------------------------------------.
        ### Check arguments 
        sampling = check_sampling(sampling)
        resolution = check_resolution(resolution)
        pool_method = check_pool_method(pool_method)
        
        ##--------------------------------------------------------------------.
        ### Define convolution type 
        # --> For all spherical samplings: conv_type='graph'   
        # --> For equiangular sampling, it is possible to specify conv_type='image'
        # - If conv_type = 'image':
        #   --> ratio = 2
        #   --> periodic = True or False
        conv_type = "graph" # image
        ratio = None        # 2
        periodic = None     # True or False
        
        ##--------------------------------------------------------------------.
        ### Initialize graphs and laplacians
        self.init_graph_and_laplacians(conv_type = conv_type, 
                                       resolution = resolution, 
                                       sampling=sampling, 
                                       knn=knn,
                                       kernel_size_pooling=kernel_size_pooling,
                                       numeric_precision=numeric_precision,
                                       UNet_depth=3)
        
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
        else:
            self.pool1, self.unpool1 = PoolUnpoolBlock.getPoolUnpoolLayer(sampling=sampling, 
                                                                          pool_method=pool_method, 
                                                                          kernel_size=kernel_size_pooling,
                                                                          ratio=ratio)
            
            self.pool2, self.unpool2 = PoolUnpoolBlock.getPoolUnpoolLayer(sampling=sampling, 
                                                                          pool_method=pool_method, 
                                                                          kernel_size=kernel_size_pooling, 
                                                                          ratio=ratio)
            
        ##--------------------------------------------------------------------.
        ### Define Encoding blocks 
        # Encoding block 1
        self.conv11 = ConvBlock(self.input_channels, 32 * 2,  
                                kernel_size=kernel_size_conv, 
                                conv_type=conv_type,
                                batch_norm=True, relu_activation=True,
                                laplacian=self.laplacians[0], 
                                periodic=periodic, ratio=ratio)
        
        self.conv13 = ConvBlock(32 * 2, 64 * 2,  
                                kernel_size=kernel_size_conv,
                                conv_type=conv_type,
                                batch_norm=True, relu_activation=True, 
                                laplacian=self.laplacians[0],
                                periodic=periodic, ratio=ratio)

        self.conv1_res = Linear(self.input_channels, 64 * 2)

        # Encoding block 2
        self.conv21 = ConvBlock(64 * 2, 96 * 2, 
                                kernel_size=kernel_size_conv, 
                                conv_type=conv_type,
                                batch_norm=True, relu_activation=True, 
                                laplacian=self.laplacians[1], 
                                periodic=periodic, ratio=ratio)
        
        self.conv23 = ConvBlock(96 * 2, 128 * 2,  
                                kernel_size=kernel_size_conv, 
                                conv_type=conv_type, 
                                batch_norm=True, relu_activation=True, 
                                laplacian=self.laplacians[1], 
                                periodic=periodic, ratio=ratio)

        self.conv2_res = Linear(64 * 2, 128 * 2)

        # Encoding block 3
        self.conv31 = ConvBlock(128 * 2, 256 * 2, 
                                kernel_size=kernel_size_conv,
                                conv_type=conv_type,
                                batch_norm=True, relu_activation=True, 
                                laplacian=self.laplacians[2], 
                                periodic=periodic, ratio=ratio)
        
        self.conv33 = ConvBlock(256 * 2, 128 * 2,  
                                kernel_size=kernel_size_conv, 
                                conv_type=conv_type, 
                                batch_norm=True, relu_activation=True, 
                                laplacian=self.laplacians[2],
                                periodic=periodic, ratio=ratio)

        self.conv3_res = Linear(128 * 2, 128 * 2)
        
        ##--------------------------------------------------------------------.
        ### Decoding blocks 
        # Decoding block 2
        self.uconv21 = ConvBlock(256 * 2, 128 * 2, 
                                 kernel_size=kernel_size_conv,
                                 conv_type=conv_type, 
                                 batch_norm=True, relu_activation=True,
                                 laplacian=self.laplacians[1], 
                                 periodic=periodic, ratio=ratio)
        self.uconv22 = ConvBlock(128 * 2, 64 * 2,  
                                 kernel_size=kernel_size_conv, 
                                 conv_type=conv_type, 
                                 batch_norm=True, relu_activation=True, 
                                 laplacian=self.laplacians[1], 
                                 periodic=periodic, ratio=ratio)

        # Decoding block 1
        self.uconv11 = ConvBlock(128 * 2, 64 * 2,
                                 kernel_size=kernel_size_conv, 
                                 conv_type=conv_type, 
                                 batch_norm=True, relu_activation=True, 
                                 laplacian=self.laplacians[0],
                                 periodic=periodic, ratio=ratio)
        self.uconv12 = ConvBlock(64 * 2, 32 * 2, 
                                 kernel_size=kernel_size_conv,
                                 conv_type=conv_type, 
                                 batch_norm=True, relu_activation=True, 
                                 laplacian=self.laplacians[0], 
                                 periodic=periodic, ratio=ratio)
        self.uconv13 = ConvBlock(32 * 2 * 2, self.output_channels, 
                                 kernel_size=kernel_size_conv, 
                                 conv_type=conv_type, 
                                 batch_norm=False, relu_activation=False, 
                                 laplacian=self.laplacians[0], 
                                 periodic=periodic, ratio=ratio)
        
    ##------------------------------------------------------------------------.
    def encode(self, x):
        """Define UNet encoder."""
        # Current input shape: ['sample', 'time', 'node', 'feature'] 
        ##--------------------------------------------------------------------.
        batch_size = x.shape[0]
        
        ##--------------------------------------------------------------------.
        ## Extract last timestep (to add after decoding) (to make the network learn the increment)
        x_last_timestep = x[:,-1,:,-2:].unsqueeze(dim=1)
        
        ##--------------------------------------------------------------------.
        # Reorder and reshape data 
        # --> Desired shape: ['sample', 'node', 'time-feature']
        x = x.rename(*self.dim_names).align_to('sample','node','time','feature').rename(None) # as x.permute(0, 2, 1, 3)   
        x = x.reshape(batch_size, self.input_node_dim, self.input_channels)  # reshape to ['sample', 'node', 'time-feature'] 
        ##--------------------------------------------------------------------.
        # Block 1
        x_enc11 = self.conv11(x)
        x_enc1 = self.conv13(x_enc11)

        x_enc1 += self.conv1_res(x)

        # Block 2
        x_enc2_ini, idx1 = self.pool1(x_enc1)
        x_enc2 = self.conv21(x_enc2_ini)
        x_enc2 = self.conv23(x_enc2)

        x_enc2 += self.conv2_res(x_enc2_ini)

        # Block 3
        x_enc3_ini, idx2 = self.pool2(x_enc2)
        x_enc3 = self.conv31(x_enc3_ini)
        x_enc3 = self.conv33(x_enc3)

        x_enc3 += self.conv3_res(x_enc3_ini)

        return x_enc3, x_enc2, x_enc1, idx2, idx1, x_enc11, x_last_timestep
    
    ##------------------------------------------------------------------------.
    def decode(self, x_enc3, x_enc2, x_enc1, idx2, idx1, x_enc11, x_last_timestep): 
        """Define UNet decoder."""
        # Block 2
        x = self.unpool2(x_enc3, idx2)
        x_cat = torch.cat((x, x_enc2), dim=2)
        x = self.uconv21(x_cat)
        x = self.uconv22(x)

        # Block 1
        x = self.unpool1(x, idx1)
        x_cat = torch.cat((x, x_enc1), dim=2)
        x = self.uconv11(x_cat)
        x = self.uconv12(x)
        x_cat = torch.cat((x, x_enc11), dim=2)
        x = self.uconv13(x_cat)
        
        ##--------------------------------------------------------------------.
        # Reshape data to ['sample', 'time', 'node', 'feature']
        batch_size = x.shape[0]   # ['sample', 'node', 'time-feature'] 
        x = x.reshape(batch_size, self.output_node_dim, self.output_time_dim, self.output_feature_dim)   # ==> ['sample', 'node', 'time', 'feature']
        x = x.rename(*['sample', 'node', 'time', 'feature']).align_to(*self.dim_names).rename(None) # as x.permute(0, 2, 1, 3)  
        ##--------------------------------------------------------------------.
        # Add tensor of most recent past timestep
        # print(x.shape)
        # print(x_last_timestep.shape)
        x = x + x_last_timestep
        return x
