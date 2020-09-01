import torch

from modules.layers import ConvCheb, PoolMaxHealpix, UnpoolMaxHealpix
from modules.healpix_models import _compute_laplacian_healpix, ConvBlock, Conv1dAuto, BottleNeckBlock

class UNetSphericalHealpixResidualShort4Levels(torch.nn.Module):
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

    Residual connections based on: https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, N, in_channels, out_channels, kernel_size, kernel_size_pooling=4):
        super().__init__()

        self.kernel_size = kernel_size

        num_nodes = 3072
        laplacians = []
        for i, nodes in enumerate([num_nodes, num_nodes/kernel_size_pooling, num_nodes/(kernel_size_pooling*kernel_size_pooling), num_nodes/(kernel_size_pooling**3)]):
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

        self.conv2_res = Conv1dAuto(64*2, 128*2, 1)

        # Encoding block 3
        self.conv31 = ConvBlock(128 * 2, 256 * 2, kernel_size, laplacians[2])
        self.conv33 = ConvBlock(256 * 2, 512 * 2, kernel_size, laplacians[2])

        self.conv3_res = Conv1dAuto(128 * 2, 512 * 2, 1)

        # Encoding block 4
        self.conv41 = ConvBlock(512 * 2, 256 * 2, kernel_size, laplacians[3])
        self.conv43 = ConvBlock(256 * 2, 256 * 2, kernel_size, laplacians[3])

        self.conv4_res = Conv1dAuto(512 * 2, 256 * 2, 1)

        # Decoding block 4
        self.uconv31 = ConvBlock(1024 * 2, 256 * 2, kernel_size, laplacians[2])
        self.uconv32 = ConvBlock(256 * 2, 128 * 2, kernel_size, laplacians[2])

        self.uconv3_res = Conv1dAuto(1024 * 2, 128 * 2, 1)

        # Decoding block 4
        self.uconv21 = ConvBlock(256 * 2, 128 * 2, kernel_size, laplacians[1])
        self.uconv22 = ConvBlock(128 * 2, 64 * 2, kernel_size, laplacians[1])

        self.uconv2_res = Conv1dAuto(256 * 2, 64 * 2, 1)

        # Decoding block 4
        self.uconv11 = ConvBlock(128 * 2, 64 * 2, kernel_size, laplacians[0])
        self.uconv12 = ConvBlock(64 * 2, 32 * 2, kernel_size, laplacians[0])

        self.uconv1_res = Conv1dAuto(128 * 2, 32 * 2, 1)

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

        # Block 3
        x_enc4_ini, idx3 = self.pooling(x_enc3)
        x_enc4 = self.conv41(x_enc4_ini)
        x_enc4 = self.conv43(x_enc4)

        x_enc4 += torch.transpose(self.conv4_res(torch.transpose(x_enc4_ini, 2, 1)), 2, 1)

        return x_enc4_ini, idx3, x_enc3, x_enc2, x_enc1, idx2, idx1

    def decode(self, x_enc4, idx3, x_enc3, x_enc2, x_enc1, idx2, idx1):
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
        # Block 3
        x = self.unpool(x_enc4, idx3)
        x_cat = torch.cat((x, x_enc3), dim=2)
        x = self.uconv31(x_cat)
        x = self.uconv32(x)

        x += torch.transpose(self.uconv3_res(torch.transpose(x_cat, 2, 1)), 2, 1)

        # Block 2
        x = self.unpool(x, idx2)
        x_cat = torch.cat((x, x_enc2), dim=2)
        x = self.uconv21(x_cat)
        x = self.uconv22(x)
        x += torch.transpose(self.uconv2_res(torch.transpose(x_cat, 2, 1)), 2, 1)

        # Block 1
        x = self.unpool(x, idx1)
        x_cat = torch.cat((x, x_enc1), dim=2)
        x = self.uconv11(x_cat)
        x = self.uconv12(x)

        x += torch.transpose(self.uconv1_res(torch.transpose(x_cat, 2, 1)), 2, 1)

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

class UNetSphericalHealpixResidualShort3Levels(torch.nn.Module):
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

    Residual connections based on: https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, N, in_channels, out_channels, kernel_size, kernel_size_pooling=4):
        super().__init__()

        self.kernel_size = kernel_size

        num_nodes = 3072
        laplacians = []
        for i, nodes in enumerate([num_nodes, num_nodes/kernel_size_pooling, num_nodes/(kernel_size_pooling*kernel_size_pooling)]):
            laplacian = _compute_laplacian_healpix(nodes)
            laplacians.append(laplacian)

        # Pooling - unpooling
        self.pooling = PoolMaxHealpix(kernel_size=kernel_size_pooling)
        self.unpool = UnpoolMaxHealpix(kernel_size=kernel_size_pooling)

        # Encoding block 1
        self.conv11 = ConvBlock(in_channels, max(in_channels, 32*2), kernel_size, laplacians[0])
        self.conv13 = ConvBlock(max(in_channels, 32*2), 64*2, kernel_size, laplacians[0])

        self.conv1_res = Conv1dAuto(in_channels, 64*2, 1)

        # Encoding block 2
        self.conv21 = ConvBlock(64*2, 96*2, kernel_size, laplacians[1])
        self.conv23 = ConvBlock(96*2, 128*2, kernel_size, laplacians[1])

        self.conv2_res = Conv1dAuto(64*2, 128*2, 1)

        # Encoding block 3
        self.conv31 = ConvBlock(128*2, 256*2, kernel_size, laplacians[2])
        self.conv33 = ConvBlock(256*2, 128*2, kernel_size, laplacians[2])

        self.conv3_res = Conv1dAuto(128*2, 128*2, 1)

        # Decoding block 4
        self.uconv21 = ConvBlock(256*2, 128*2, kernel_size, laplacians[1])
        self.uconv22 = ConvBlock(128*2, 64*2, kernel_size, laplacians[1])

        self.uconv2_res = Conv1dAuto(256*2, 64*2, 1)

        # Decoding block 4
        self.uconv11 = ConvBlock(128*2, 64*2, kernel_size, laplacians[0])
        self.uconv12 = ConvBlock(64*2, 32*2, kernel_size, laplacians[0])

        self.uconv1_res = Conv1dAuto(128 * 2, 32 * 2, 1)

        self.uconv13 = ConvCheb(32*2, out_channels, kernel_size, laplacians[0])

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
        x_cat = torch.cat((x, x_enc2), dim=2)
        x = self.uconv21(x_cat)
        x = self.uconv22(x)

        x += torch.transpose(self.uconv2_res(torch.transpose(x_cat, 2, 1)), 2, 1)

        # Block 1
        x = self.unpool(x, idx1)
        x_cat = torch.cat((x, x_enc1), dim=2)
        x = self.uconv11(x_cat)
        x = self.uconv12(x)

        x += torch.transpose(self.uconv1_res(torch.transpose(x_cat, 2, 1)), 2, 1)

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

class UNetSphericalHealpixResidualShort3LevelsOnlyEncoder(torch.nn.Module):
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

    Residual connections based on: https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, N, in_channels, out_channels, kernel_size, kernel_size_pooling=4):
        super().__init__()

        self.kernel_size = kernel_size

        num_nodes = 3072
        laplacians = []
        for i, nodes in enumerate([num_nodes, num_nodes/kernel_size_pooling, num_nodes/(kernel_size_pooling*kernel_size_pooling)]):
            laplacian = _compute_laplacian_healpix(nodes)
            laplacians.append(laplacian)

        # Pooling - unpooling
        self.pooling = PoolMaxHealpix(kernel_size=kernel_size_pooling)
        self.unpool = UnpoolMaxHealpix(kernel_size=kernel_size_pooling)

        # Encoding block 1
        self.conv11 = ConvBlock(in_channels, max(in_channels, 32*2), kernel_size, laplacians[0])
        self.conv13 = ConvBlock(max(in_channels, 32*2), 64*2, kernel_size, laplacians[0])

        self.conv1_res = Conv1dAuto(in_channels, 64*2, 1)

        # Encoding block 2
        self.conv21 = ConvBlock(64*2, 96*2, kernel_size, laplacians[1])
        self.conv23 = ConvBlock(96*2, 128*2, kernel_size, laplacians[1])

        self.conv2_res = Conv1dAuto(64*2, 128*2, 1)

        # Encoding block 3
        self.conv31 = ConvBlock(128*2, 256*2, kernel_size, laplacians[2])
        self.conv33 = ConvBlock(256*2, 128*2, kernel_size, laplacians[2])

        self.conv3_res = Conv1dAuto(128*2, 128*2, 1)

        # Decoding block 4
        self.uconv21 = ConvBlock(256*2, 128*2, kernel_size, laplacians[1])
        self.uconv22 = ConvBlock(128*2, 64*2, kernel_size, laplacians[1])

        self.uconv2_res = Conv1dAuto(256*2, 64*2, 1)

        # Decoding block 4
        self.uconv11 = ConvBlock(128*2, 64*2, kernel_size, laplacians[0])
        self.uconv12 = ConvBlock(64*2, 32*2, kernel_size, laplacians[0])

        self.uconv1_res = Conv1dAuto(128 * 2, 32 * 2, 1)

        self.uconv13 = ConvCheb(32*2, out_channels, kernel_size, laplacians[0])

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
        x_cat = torch.cat((x, x_enc2), dim=2)
        x = self.uconv21(x_cat)
        x = self.uconv22(x)

        #x += torch.transpose(self.uconv2_res(torch.transpose(x_cat, 2, 1)), 2, 1)

        # Block 1
        x = self.unpool(x, idx1)
        x_cat = torch.cat((x, x_enc1), dim=2)
        x = self.uconv11(x_cat)
        x = self.uconv12(x)

        #x += torch.transpose(self.uconv1_res(torch.transpose(x_cat, 2, 1)), 2, 1)

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

class UNetSphericalHealpixResidualShort3LevelsOnlyEncoder3Channels(torch.nn.Module):
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

    Residual connections based on: https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, N, in_channels, out_channels, kernel_size, kernel_size_pooling=4):
        super().__init__()

        self.kernel_size = kernel_size

        num_nodes = 3072
        laplacians = []
        for i, nodes in enumerate([num_nodes, num_nodes/kernel_size_pooling, num_nodes/(kernel_size_pooling*kernel_size_pooling)]):
            laplacian = _compute_laplacian_healpix(nodes)
            laplacians.append(laplacian)

        # Pooling - unpooling
        self.pooling = PoolMaxHealpix(kernel_size=kernel_size_pooling)
        self.unpool = UnpoolMaxHealpix(kernel_size=kernel_size_pooling)

        # Encoding block 1
        self.conv11 = ConvBlock(in_channels, max(in_channels, 32*2), kernel_size, laplacians[0])
        self.conv12 = ConvBlock(max(in_channels, 32*2), 64*2, kernel_size, laplacians[0])
        self.conv13 = ConvBlock(64 * 2, 64 * 2, kernel_size, laplacians[0])

        self.conv1_res = Conv1dAuto(in_channels, 64*2, 1)

        # Encoding block 2
        self.conv21 = ConvBlock(64*2, 96*2, kernel_size, laplacians[1])
        self.conv22 = ConvBlock(96*2, 128*2, kernel_size, laplacians[1])
        self.conv23 = ConvBlock(128 * 2, 128 * 2, kernel_size, laplacians[1])

        self.conv2_res = Conv1dAuto(64*2, 128*2, 1)

        # Encoding block 3
        self.conv31 = ConvBlock(128 * 2, 256 * 2, kernel_size, laplacians[2])
        self.conv32 = ConvBlock(256 * 2, 128 * 2, kernel_size, laplacians[2])
        self.conv33 = ConvBlock(128 * 2, 128 * 2, kernel_size, laplacians[2])

        self.conv3_res = Conv1dAuto(128*2, 128*2, 1)

        # Decoding block 4
        self.uconv21 = ConvBlock(256*2, 128*2, kernel_size, laplacians[1])
        self.uconv22 = ConvBlock(128*2, 64*2, kernel_size, laplacians[1])
        self.uconv23 = ConvBlock(64 * 2, 64 * 2, kernel_size, laplacians[1])


        # Decoding block 4
        self.uconv11 = ConvBlock(128*2, 64*2, kernel_size, laplacians[0])
        self.uconv12 = ConvBlock(64*2, 32*2, kernel_size, laplacians[0])
        self.uconv13 = ConvBlock(32 * 2, 32 * 2, kernel_size, laplacians[0])


        self.uconv14 = ConvCheb(32*2, out_channels, kernel_size, laplacians[0])

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
        x_enc1 = self.conv12(x_enc1)
        x_enc1 = self.conv13(x_enc1)

        x_enc1 += torch.transpose(self.conv1_res(torch.transpose(x, 2, 1)), 2, 1)

        # Block 2
        x_enc2_ini, idx1 = self.pooling(x_enc1)
        x_enc2 = self.conv21(x_enc2_ini)
        x_enc2 = self.conv22(x_enc2)
        x_enc2 = self.conv23(x_enc2)

        x_enc2 += torch.transpose(self.conv2_res(torch.transpose(x_enc2_ini, 2, 1)), 2, 1)

        # Block 3
        x_enc3_ini, idx2 = self.pooling(x_enc2)
        x_enc3 = self.conv31(x_enc3_ini)
        x_enc3 = self.conv32(x_enc3)
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
        x_cat = torch.cat((x, x_enc2), dim=2)
        x = self.uconv21(x_cat)
        x = self.uconv22(x)
        x = self.uconv23(x)

        #x += torch.transpose(self.uconv2_res(torch.transpose(x_cat, 2, 1)), 2, 1)

        # Block 1
        x = self.unpool(x, idx1)
        x_cat = torch.cat((x, x_enc1), dim=2)
        x = self.uconv11(x_cat)
        x = self.uconv12(x)
        x = self.uconv13(x)

        #x += torch.transpose(self.uconv1_res(torch.transpose(x_cat, 2, 1)), 2, 1)

        x = self.uconv14(x)

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

class UNetSphericalHealpixResidualShort2Levels(torch.nn.Module):
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

    Residual connections based on: https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, N, in_channels, out_channels, kernel_size, kernel_size_pooling=4):
        super().__init__()

        self.kernel_size = kernel_size

        num_nodes = 3072
        laplacians = []
        for i, nodes in enumerate([num_nodes, num_nodes/kernel_size_pooling]):
            laplacian = _compute_laplacian_healpix(nodes)
            laplacians.append(laplacian)

        # Pooling - unpooling
        self.pooling = PoolMaxHealpix(kernel_size=kernel_size_pooling)
        self.unpool = UnpoolMaxHealpix(kernel_size=kernel_size_pooling)

        # Encoding block 1
        self.conv11 = ConvBlock(in_channels, max(in_channels, 32*2), kernel_size, laplacians[0])
        self.conv13 = ConvBlock(max(in_channels, 32*2), 64*2, kernel_size, laplacians[0])

        self.conv1_res = Conv1dAuto(in_channels, 64*2, 1)

        # Encoding block 2
        self.conv21 = ConvBlock(64*2, 96*2, kernel_size, laplacians[1])
        self.conv23 = ConvBlock(96*2, 64*2, kernel_size, laplacians[1])

        self.conv2_res = Conv1dAuto(64*2, 64*2, 1)


        # Decoding block 4
        self.uconv11 = ConvBlock(128 * 2, 64*2, kernel_size, laplacians[0])
        self.uconv12 = ConvBlock(64*2, 32*2, kernel_size, laplacians[0])

        self.uconv1_res = Conv1dAuto(128 * 2, 32 * 2, 1)

        self.uconv13 = ConvCheb(32*2, out_channels, kernel_size, laplacians[0])

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


        return x_enc2, x_enc1, idx1

    def decode(self, x_enc2, x_enc1, idx1):
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


        # Block 1

        x = self.unpool(x_enc2, idx1)
        x_cat = torch.cat((x, x_enc1), dim=2)
        x = self.uconv11(x_cat)
        x = self.uconv12(x)

        x += torch.transpose(self.uconv1_res(torch.transpose(x_cat, 2, 1)), 2, 1)

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

class SphericalHealpixBlottleNeck(torch.nn.Module):
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

    Residual connections based on: https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, N, in_channels, out_channels, kernel_size, kernel_size_pooling=4):
        super().__init__()

        self.kernel_size = kernel_size

        num_nodes = 3072
        laplacians = []
        for i, nodes in enumerate([num_nodes]):
            laplacian = _compute_laplacian_healpix(nodes)
            laplacians.append(laplacian)

        # First convolution
        self.conv1 = ConvBlock(in_channels, 64, 3, laplacians[0])
        self.conv2 = ConvBlock(64, 256, 3, laplacians[0])

        # First BottleNeck Block
        self.bottleneck1 = BottleNeckBlock(256, 64, laplacians[0])

        # Second BottleNeck Block
        self.bottleneck2 = BottleNeckBlock(256, 64, laplacians[0])

        # Third BottleNeck Block
        self.bottleneck3 = BottleNeckBlock(256, 64, laplacians[0])

        self.conv3 = ConvCheb(256, out_channels, kernel_size, laplacians[0])


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
        x = self.conv1(x)
        x = self.conv2(x)

        x += self.bottleneck1(x)
        x += self.bottleneck2(x)
        x += self.bottleneck3(x)

        output = self.conv3(x)

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
        for i, nodes in enumerate([num_nodes, num_nodes/kernel_size_pooling, num_nodes/(kernel_size*kernel_size_pooling)]):
            laplacian = _compute_laplacian_healpix(nodes)
            laplacians.append(laplacian)

        # Pooling - unpooling
        self.pooling = PoolMaxHealpix(kernel_size=kernel_size_pooling)
        self.unpool = UnpoolMaxHealpix(kernel_size=kernel_size_pooling)

        # Encoding block 1
        self.conv11 = ConvBlock(in_channels, max(in_channels, 32*2), kernel_size, laplacians[0])
        self.conv13 = ConvBlock(max(in_channels, 32*2), 64*2, kernel_size, laplacians[0])

        # Encoding block 2
        self.conv21 = ConvBlock(64*2, 96*2, kernel_size, laplacians[1])
        self.conv23 = ConvBlock(96*2, 128*2, kernel_size, laplacians[1])

        # Encoding block 3
        self.conv31 = ConvBlock(128*2, 256*2, kernel_size, laplacians[2])
        #self.conv32 = ConvBlock(256*2, 256*2, kernel_size, laplacians[2])
        self.conv33 = ConvBlock(256*2, 128*2, kernel_size, laplacians[2])

        # Decoding block 4
        self.uconv21 = ConvBlock(256*2, 128*2, kernel_size, laplacians[1])
        self.uconv22 = ConvBlock(128*2, 64*2, kernel_size, laplacians[1])

        # Decoding block 4
        self.uconv11 = ConvBlock(128*2, 64*2, kernel_size, laplacians[0])
        self.uconv12 = ConvBlock(64*2, 32*2, kernel_size, laplacians[0])
        self.uconv13 = ConvCheb(32*2, out_channels, kernel_size, laplacians[0])

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

class UNetSphericalHealpixResidualShort4LevelsNoDec(torch.nn.Module):
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

    Residual connections based on: https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, N, in_channels, out_channels, kernel_size, kernel_size_pooling=4):
        super().__init__()

        self.kernel_size = kernel_size

        num_nodes = 3072
        laplacians = []
        for i, nodes in enumerate([num_nodes, num_nodes/kernel_size_pooling, num_nodes/(kernel_size_pooling*kernel_size_pooling), num_nodes/(kernel_size_pooling**3)]):
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

        self.conv2_res = Conv1dAuto(64*2, 128*2, 1)

        # Encoding block 3
        self.conv31 = ConvBlock(128 * 2, 256 * 2, kernel_size, laplacians[2])
        self.conv33 = ConvBlock(256 * 2, 512 * 2, kernel_size, laplacians[2])

        self.conv3_res = Conv1dAuto(128 * 2, 512 * 2, 1)

        # Encoding block 4
        self.conv41 = ConvBlock(512 * 2, 256 * 2, kernel_size, laplacians[3])
        self.conv43 = ConvBlock(256 * 2, 256 * 2, kernel_size, laplacians[3])

        self.conv4_res = Conv1dAuto(512 * 2, 256 * 2, 1)

        # Decoding block 4
        self.uconv31 = ConvBlock(1024 * 2, 256 * 2, kernel_size, laplacians[2])
        self.uconv32 = ConvBlock(256 * 2, 128 * 2, kernel_size, laplacians[2])

        #self.uconv3_res = Conv1dAuto(512 * 2, 128 * 2, 1)

        # Decoding block 4
        self.uconv21 = ConvBlock(256 * 2, 128 * 2, kernel_size, laplacians[1])
        self.uconv22 = ConvBlock(128 * 2, 64 * 2, kernel_size, laplacians[1])

        #self.uconv2_res = Conv1dAuto(128 * 2, 64 * 2, 1)

        # Decoding block 4
        self.uconv11 = ConvBlock(128 * 2, 64 * 2, kernel_size, laplacians[0])
        self.uconv12 = ConvBlock(64 * 2, 32 * 2, kernel_size, laplacians[0])

        #self.uconv1_res = Conv1dAuto(64 * 2, 32 * 2, 1)

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

        # Block 3
        x_enc4_ini, idx3 = self.pooling(x_enc3)
        x_enc4 = self.conv41(x_enc4_ini)
        x_enc4 = self.conv43(x_enc4)

        x_enc4 += torch.transpose(self.conv4_res(torch.transpose(x_enc4_ini, 2, 1)), 2, 1)

        return x_enc4_ini, idx3, x_enc3, x_enc2, x_enc1, idx2, idx1

    def decode(self, x_enc4, idx3, x_enc3, x_enc2, x_enc1, idx2, idx1):
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
        # Block 3
        x = self.unpool(x_enc4, idx3)
        x_cat = torch.cat((x, x_enc3), dim=2)
        x = self.uconv31(x_cat)
        x = self.uconv32(x)

        #x += torch.transpose(self.uconv3_res(torch.transpose(x_cat, 2, 1)), 2, 1)

        # Block 2
        x = self.unpool(x, idx2)
        x_cat = torch.cat((x, x_enc2), dim=2)
        x = self.uconv21(x_cat)
        x = self.uconv22(x)
        #x += torch.transpose(self.uconv2_res(torch.transpose(x_cat, 2, 1)), 2, 1)

        # Block 1
        x = self.unpool(x, idx1)
        x_cat = torch.cat((x, x_enc1), dim=2)
        x = self.uconv11(x_cat)
        x = self.uconv12(x)

        #x += torch.transpose(self.uconv1_res(torch.transpose(x_cat, 2, 1)), 2, 1)

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

class UNetSphericalHealpixResidualLongConnections(torch.nn.Module):
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

    Residual connections based on: https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, N, in_channels, out_channels, kernel_size, kernel_size_pooling=4):
        super().__init__()

        self.kernel_size = kernel_size

        num_nodes = 3072
        laplacians = []
        for i, nodes in enumerate([num_nodes, num_nodes/kernel_size_pooling, num_nodes/(kernel_size_pooling*kernel_size_pooling)]):
            laplacian = _compute_laplacian_healpix(nodes)
            laplacians.append(laplacian)

        # Pooling - unpooling
        self.pooling = PoolMaxHealpix(kernel_size=kernel_size_pooling)
        self.unpool = UnpoolMaxHealpix(kernel_size=kernel_size_pooling)

        # Encoding block 1
        self.conv11 = ConvBlock(in_channels, max(in_channels, 32*2), kernel_size, laplacians[0])
        self.conv13 = ConvBlock(max(in_channels, 32*2), 64*2, kernel_size, laplacians[0])

        self.conv1_res = Conv1dAuto(in_channels, 64*2, 1)

        # Encoding block 2
        self.conv21 = ConvBlock(64*2, 96*2, kernel_size, laplacians[1])
        self.conv23 = ConvBlock(96*2, 128*2, kernel_size, laplacians[1])

        self.conv2_res = Conv1dAuto(64*2, 128*2, 1)

        # Encoding block 3
        self.conv31 = ConvBlock(128*2, 256*2, kernel_size, laplacians[2])
        self.conv33 = ConvBlock(256*2, 128*2, kernel_size, laplacians[2])

        self.conv3_res = Conv1dAuto(128*2, 128*2, 1)

        # Decoding block 4
        self.uconv21 = ConvBlock(256*2, 128*2, kernel_size, laplacians[1])
        self.uconv22 = ConvBlock(128*2, 64*2, kernel_size, laplacians[1])

        self.uconv2_res = Conv1dAuto(256*2, 64*2, 1)

        # Decoding block 4
        self.uconv11 = ConvBlock(128*2, 64*2, kernel_size, laplacians[0])
        self.uconv12 = ConvBlock(64*2, 32*2, kernel_size, laplacians[0])

        self.uconv1_res = Conv1dAuto(128 * 2, 32 * 2, 1)

        self.uconv13 = ConvCheb(32*2*2, out_channels, kernel_size, laplacians[0])

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

        x_enc11 = self.conv11(x)
        x_enc1 = self.conv13(x_enc11)

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

        return x_enc3, x_enc2, x_enc1, idx2, idx1, x_enc11

    def decode(self, x_enc3, x_enc2, x_enc1, idx2, idx1, x_enc11):
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
        x_cat = torch.cat((x, x_enc2), dim=2)
        x = self.uconv21(x_cat)
        x = self.uconv22(x)

        #x += torch.transpose(self.uconv2_res(torch.transpose(x_cat, 2, 1)), 2, 1)

        # Block 1
        x = self.unpool(x, idx1)
        x_cat = torch.cat((x, x_enc1), dim=2)
        x = self.uconv11(x_cat)
        x = self.uconv12(x)

        #x += torch.transpose(self.uconv1_res(torch.transpose(x_cat, 2, 1)), 2, 1)
        x_cat = torch.cat((x, x_enc11), dim=2)
        x = self.uconv13(x_cat)

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