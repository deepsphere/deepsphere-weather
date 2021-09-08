import torch
import numpy as np
from typing import List, Union, Dict
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

#  "knn": 20
# "learning_rate": 0.007
# "training_batch_size": 16
# "validation_batch_size": 16
# "scoring_interval": 10
# "deterministic_training": false
# "deterministic_training_seed": 100
# "benchmark_cuDNN": true
# patience = 500
# minimum_iterations = 500
# minimum_improvement = 0.001 # 0 to not stop 
# static scalers ... # load old static and old static scalers 
# ds_static = readDatasets(data_dir=data_sampling_dir, feature_type='static')  
# static_scaler = LoadScaler(os.path.join(data_sampling_dir, "Scalers", "GlobalStandardScaler_static.nc"))
# # # - Create single scaler 
# scaler = SequentialScaler(dynamic_scaler, bc_scaler, static_scaler)

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

##----------------------------------------------------------------------------.     
class UNetSpherical(UNet, torch.nn.Module):
    """Classical spherical UNet with residual connections."""

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
                 batch_norm: bool = True, 
                 batch_norm_before_activation: bool = True,
                 activation: bool = True,
                 activation_fun: str = 'relu',
                 # Pooling options 
                 pool_method: str = 'max', 
                 kernel_size_pooling: int = 4,
                 # Architecture options 
                 skip_connection: str = 'stack',
                 increment_learning: bool = False):
        ##--------------------------------------------------------------------.
        super().__init__()
        ##--------------------------------------------------------------------.
        # Retrieve tensor informations 
        self.dim_names = tensor_info['dim_order']['dynamic']  
        self.input_feature_dim = tensor_info['input_n_feature']
        self.input_time_dim = tensor_info['input_n_time']
        self.input_node_dim = tensor_info['input_shape_info']['dynamic']['node'] 
        
        self.output_time_dim = tensor_info['output_n_time']
        self.output_feature_dim = tensor_info['output_n_feature']
        self.output_node_dim = tensor_info['output_shape_info']['dynamic']['node']  
        
        ##--------------------------------------------------------------------.
        # Define size of last dimension for ConvChen conv (merging time-feature dimension)
        self.input_channels = self.input_feature_dim* self.input_time_dim 
        self.output_channels = self.output_feature_dim*self.output_time_dim 

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
        if sampling != "equiangular":
            # (pygsp.graphs.SphereEquiangular do not accept k)
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
        else:
            self.pool1, self.unpool1 = PoolUnpoolBlock.getPoolUnpoolLayer(sampling=sampling, 
                                                                          pool_method=pool_method, 
                                                                          kernel_size=kernel_size_pooling,
                                                                          lonlat_ratio=lonlat_ratio)
            
            self.pool2, self.unpool2 = PoolUnpoolBlock.getPoolUnpoolLayer(sampling=sampling, 
                                                                          pool_method=pool_method, 
                                                                          kernel_size=kernel_size_pooling, 
                                                                          lonlat_ratio=lonlat_ratio)
            
        ##--------------------------------------------------------------------.
        ### Define Encoding blocks 
        # Encoding block 1
        self.conv11 = ConvBlock(self.input_channels, 32 * 2,  
                                laplacian = self.laplacians[0],
                                **convblock_kwargs)
        
        self.conv13 = ConvBlock(32 * 2, 64 * 2,  
                                laplacian = self.laplacians[0],
                                **convblock_kwargs)

        self.conv1_res = Linear(self.input_channels, 64 * 2)

        # Encoding block 2
        self.conv21 = ConvBlock(64 * 2, 96 * 2, 
                                laplacian = self.laplacians[1],
                                **convblock_kwargs)
        
        self.conv23 = ConvBlock(96 * 2, 128 * 2,  
                                laplacian = self.laplacians[1],
                                **convblock_kwargs)

        self.conv2_res = Linear(64 * 2, 128 * 2)

        # Encoding block 3
        self.conv31 = ConvBlock(128 * 2, 256 * 2, 
                                laplacian = self.laplacians[2],
                                **convblock_kwargs)
        
        self.conv33 = ConvBlock(256 * 2, 128 * 2,  
                                 laplacian = self.laplacians[2],
                                 **convblock_kwargs)

        self.conv3_res = Linear(128 * 2, 128 * 2)
        
        ##--------------------------------------------------------------------.
        ### Decoding blocks 
        # Decoding block 2
        self.uconv21 = ConvBlock(256 * 2, 128 * 2, 
                                 laplacian = self.laplacians[1],
                                 **convblock_kwargs)
        self.uconv22 = ConvBlock(128 * 2, 64 * 2,  
                                 laplacian = self.laplacians[1],
                                 **convblock_kwargs)

        # Decoding block 1
        self.uconv11 = ConvBlock(128 * 2, 64 * 2,
                                 laplacian = self.laplacians[0],
                                 **convblock_kwargs)
        self.uconv12 = ConvBlock(64 * 2, 32 * 2, 
                                 laplacian = self.laplacians[0],
                                 **convblock_kwargs)

        special_kwargs = convblock_kwargs.copy()
        special_kwargs["batch_norm"] = False 
        special_kwargs["activation"] = False    
        self.uconv13 = ConvBlock(32 * 2 * 2, self.output_channels, 
                                 laplacian = self.laplacians[0],
                                 **special_kwargs)                           
        
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
