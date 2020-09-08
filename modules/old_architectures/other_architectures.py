class UNetSphericalHealpixDeeperBlocks(torch.nn.Module):
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

    def __init__(self, N, in_channels, out_channels, kernel_size):
        super().__init__()

        self.kernel_size = kernel_size

        laplacians = []
        for i, nodes in enumerate([3072, 768, 192, int(192 / 4)]):
            laplacian = _compute_laplacian_healpix(nodes)
            laplacians.append(laplacian)

        # print('hey')
        # Pooling - unpooling
        self.pooling = PoolMaxHealpix(kernel_size=4)
        self.unpool = UnpoolMaxHealpix(kernel_size=4)

        # Encoding block 1
        self.conv11 = ConvBlock(in_channels, max(in_channels, 16), kernel_size, laplacians[0])
        self.conv12 = ConvBlock(max(in_channels, 16), max(in_channels, 32), kernel_size, laplacians[0])
        self.conv13 = ConvBlock(max(in_channels, 32), 64, kernel_size, laplacians[0])

        # Encoding block 2
        self.conv21 = ConvBlock(64, 88, kernel_size, laplacians[1])
        self.conv22 = ConvBlock(88, 110, kernel_size, laplacians[1])
        self.conv23 = ConvBlock(110, 128, kernel_size, laplacians[1])

        # Encoding block 3
        self.conv31 = ConvBlock(128, 256, kernel_size, laplacians[2])
        self.conv32 = ConvBlock(256, 256, kernel_size, laplacians[2])
        self.conv33 = ConvBlock(256, 128, kernel_size, laplacians[2])

        self.conv41 = ConvBlock(128, 256, kernel_size, laplacians[3])
        self.conv42 = ConvBlock(256, 256, kernel_size, laplacians[3])
        self.conv43 = ConvBlock(256, 128, kernel_size, laplacians[3])

        self.uconv31 = ConvBlock(256, 128, kernel_size, laplacians[2])
        self.uconv32 = ConvBlock(128, 64, kernel_size, laplacians[2])

        # Decoding block 4
        self.uconv21 = ConvBlock(256, 128, kernel_size, laplacians[1])
        self.uconv22 = ConvBlock(128, 64, kernel_size, laplacians[1])

        # Decoding block 4
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
            Encoded data at the different encoding stages and the indices indicating the locations of the maxium values in
            unpooled images.
        """
        # x_enc1 = self.dropout1(x_enc1)

        # Block 1

        # print('Start: x shape: ', x.shape)
        x_enc1 = self.conv11(x)
        # print('B11: ', x_enc1.shape)
        x_enc1 = self.conv12(x_enc1)
        # print('B12: ', x_enc1.shape)
        x_enc1 = self.conv13(x_enc1)
        # print('B13: ', x_enc1.shape)

        # Block 2
        x_enc2, idx1 = self.pooling(x_enc1)
        # print('B21 - after pooling: ', x_enc2.shape)
        x_enc2 = self.conv21(x_enc2)
        # print('B22: ', x_enc2.shape)
        x_enc2 = self.conv22(x_enc2)
        # print('B23: ', x_enc2.shape)
        x_enc2 = self.conv23(x_enc2)
        # print('B24: ', x_enc2.shape)

        # Block 3
        x_enc3, idx2 = self.pooling(x_enc2)
        # print('B31 - after pooling: ', x_enc3.shape)
        x_enc3 = self.conv31(x_enc3)
        # print('B32: ', x_enc3.shape)
        x_enc3 = self.conv32(x_enc3)
        # print('B33: ', x_enc3.shape)
        x_enc3 = self.conv33(x_enc3)
        # print('B34: ', x_enc3.shape)

        x_enc4, idx3 = self.pooling(x_enc3)
        # print('B41 - after pooling: ', x_enc4.shape)
        x_enc4 = self.conv41(x_enc4)
        # print('B42: ', x_enc4.shape)
        x_enc4 = self.conv42(x_enc4)
        # print('B43: ', x_enc4.shape)
        x_enc4 = self.conv43(x_enc4)
        # print('B44: ', x_enc4.shape)

        return x_enc4, x_enc3, x_enc2, x_enc1, idx3, idx2, idx1

    def decode(self, x_enc4, x_enc3, x_enc2, x_enc1, idx3, idx2, idx1):
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
        # print('idx3 before first unpool ', idx3.shape)
        x = self.unpool(x_enc4, idx3)
        # print('R01- after unpool ', x.shape)
        x = torch.cat((x, x_enc3), dim=2)
        # print('R02- after concat ', x.shape)
        x = self.uconv31(x)
        # print('R03- after uconv31 ', x.shape)
        # x = self.uconv32(x)

        # Block 2
        # print('R04 - before unpooling: ', x.shape, idx2.shape)
        x = self.unpool(x, idx2)
        # print('R11 - after unpooling: ', x.shape)
        x = torch.cat((x, x_enc2), dim=2)
        # print('B42 - after concat: ', x.shape)
        x = self.uconv21(x)
        # print('B43 - after cov: ', x.shape)
        x = self.uconv22(x)
        # print('B44 - after cov: ', x.shape)

        # Block 1
        x = self.unpool(x, idx1)
        # print('B51 - after unpooling: ', x.shape)
        x = torch.cat((x, x_enc1), dim=2)
        # print('B52 - after concat: ', x.shape)
        x = self.uconv11(x)
        # print('B53 - after cov: ', x.shape)
        x = self.uconv12(x)
        # print('B54 - after cov: ', x.shape)
        x = self.uconv13(x)
        # print('B55 - after cov: ', x.shape)

        return x

    def state_dict(self, *args, **kwargs):
        """
        This function overrides the state dict in order to be able to save the model.
        This can be removed as soon as saving sparse matrices has been added to Pytorch.
        """
        state_dict = super().state_dict(*args, **kwargs)
        # del_keys = []
        # for key in state_dict:
        #    if "laplacian" in key:
        #        del_keys.append(key)
        # for key in del_keys:
        #    del state_dict[key]
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


# spherical_unet_deeper_less_blocks_more_channels_1990_

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


#spherical_unet_deeper_more_blocks_less_channels_1990_

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
        self.conv11 = ConvBlock(in_channels, max(in_channels, 16), kernel_size, laplacians[0])
        self.conv12 = ConvBlock(max(in_channels, 16), max(in_channels, 32*1), kernel_size, laplacians[0])
        self.conv122 = ConvBlock(max(in_channels, 32 * 1), max(in_channels, 32 * 1), kernel_size, laplacians[0])
        self.conv13 = ConvBlock(max(in_channels, 32*1), 64*1, kernel_size, laplacians[0])

        # Encoding block 2
        self.conv21 = ConvBlock(64*1, 80*1, kernel_size, laplacians[1])
        self.conv22 = ConvBlock(80*1, 96*1, kernel_size, laplacians[1])
        self.conv222 = ConvBlock(96 * 1, 112 * 1, kernel_size, laplacians[1])
        self.conv23 = ConvBlock(112*1, 128*1, kernel_size, laplacians[1])

        # Encoding block 3
        self.conv31 = ConvBlock(128*1, 256*1, kernel_size, laplacians[2])
        self.conv32 = ConvBlock(256*1, 256*1, kernel_size, laplacians[2])
        self.conv33 = ConvBlock(256*1, 128*1, kernel_size, laplacians[2])

        # Decoding block 4
        self.uconv21 = ConvBlock(256*1, 128*1, kernel_size, laplacians[1])
        self.uconv22 = ConvBlock(128*1, 64*1, kernel_size, laplacians[1])

        # Decoding block 4
        self.uconv11 = ConvBlock(128*1, 64*1, kernel_size, laplacians[0])
        self.uconv12 = ConvBlock(64*1, 32*1, kernel_size, laplacians[0])
        self.uconv13 = ConvCheb(32*1, out_channels, kernel_size, laplacians[0])

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
        x_enc1 = self.conv122(x_enc1)
        x_enc1 = self.conv13(x_enc1)

        # Block 2
        x_enc2, idx1 = self.pooling(x_enc1)
        x_enc2 = self.conv21(x_enc2)
        x_enc2 = self.conv22(x_enc2)
        x_enc2 = self.conv222(x_enc2)
        x_enc2 = self.conv23(x_enc2)

        # Block 3
        x_enc3, idx2 = self.pooling(x_enc2)
        x_enc3 = self.conv31(x_enc3)
        #x_enc3 = self.conv32(x_enc3)
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


# spherical_unet_deeper_less_blocks_same_channels_1990_
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
        self.conv11 = ConvBlock(in_channels, max(in_channels, 32*1), kernel_size, laplacians[0])
        self.conv13 = ConvBlock(max(in_channels, 32*1), 64*1, kernel_size, laplacians[0])

        # Encoding block 2
        self.conv21 = ConvBlock(64*1, 96*1, kernel_size, laplacians[1])
        self.conv23 = ConvBlock(96*1, 128*1, kernel_size, laplacians[1])

        # Encoding block 3
        self.conv31 = ConvBlock(128*1, 256*1, kernel_size, laplacians[2])
        self.conv33 = ConvBlock(256*1, 128*1, kernel_size, laplacians[2])

        # Decoding block 4
        self.uconv21 = ConvBlock(256*1, 128*1, kernel_size, laplacians[1])
        self.uconv22 = ConvBlock(128*1, 64*1, kernel_size, laplacians[1])

        # Decoding block 4
        self.uconv11 = ConvBlock(128*1, 64*1, kernel_size, laplacians[0])
        self.uconv13 = ConvCheb(64*1, out_channels, kernel_size, laplacians[0])

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
        x = self.uconv13(x)
        return x


#spherical_unet_deeper_more_blocks_more_channels_1990_

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
        self.conv11 = ConvBlock(in_channels, max(in_channels, 16), kernel_size, laplacians[0])
        self.conv12 = ConvBlock(max(in_channels, 16), max(in_channels, 32*2), kernel_size, laplacians[0])
        self.conv122 = ConvBlock(max(in_channels, 32*2), 32*2, kernel_size, laplacians[0])
        self.conv13 = ConvBlock(32*2, 64*2, kernel_size, laplacians[0])

        # Encoding block 2
        self.conv21 = ConvBlock(64*2, 80*2, kernel_size, laplacians[1])
        self.conv22 = ConvBlock(80*2, 96*2, kernel_size, laplacians[1])
        self.conv222 = ConvBlock(96 * 2, 112 * 2, kernel_size, laplacians[1])
        self.conv23 = ConvBlock(112*2, 128*2, kernel_size, laplacians[1])

        # Encoding block 3
        self.conv31 = ConvBlock(128*2, 256*2, kernel_size, laplacians[2])
        self.conv32 = ConvBlock(256*2, 256*2, kernel_size, laplacians[2])
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
        x_enc1 = self.conv12(x_enc1)
        x_enc1 = self.conv122(x_enc1)
        x_enc1 = self.conv13(x_enc1)

        # Block 2
        x_enc2, idx1 = self.pooling(x_enc1)
        x_enc2 = self.conv21(x_enc2)
        x_enc2 = self.conv22(x_enc2)
        x_enc2 = self.conv222(x_enc2)
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

#spherical_unet_residual1_1990
    class UNetSphericalHealpixResidual(torch.nn.Module):
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

        def __init__(self, N, in_channels, out_channels, kernel_size):
            super().__init__()

            self.kernel_size = kernel_size

            laplacians = []
            for i, nodes in enumerate([3072, 768, 192]):
                laplacian = _compute_laplacian_healpix(nodes)
                laplacians.append(laplacian)

            # Pooling - unpooling
            self.pooling = PoolMaxHealpix(kernel_size=4)
            self.unpool = UnpoolMaxHealpix(kernel_size=4)

            # Encoding block 1
            self.conv11 = ConvBlock(in_channels, max(in_channels, 16), kernel_size, laplacians[0])
            self.conv12 = ConvBlock(max(in_channels, 16), max(in_channels, 32), kernel_size, laplacians[0])
            self.conv13 = ConvBlock(max(in_channels, 32), 64, kernel_size, laplacians[0])

            self.conv1_res = Conv1dAuto(in_channels, 64, 1)

            # Encoding block 2
            self.conv21 = ConvBlock(64, 88, kernel_size, laplacians[1])
            self.conv22 = ConvBlock(88, 110, kernel_size, laplacians[1])
            self.conv23 = ConvBlock(110, 128, kernel_size, laplacians[1])

            self.conv2_res = Conv1dAuto(64, 128, 1)

            # Encoding block 3
            self.conv31 = ConvBlock(128, 256, kernel_size, laplacians[2])
            self.conv32 = ConvBlock(256, 256, kernel_size, laplacians[2])
            self.conv33 = ConvBlock(256, 128, kernel_size, laplacians[2])

            self.conv3_res = Conv1dAuto(128, 128, 1)

            # Decoding block 4
            self.uconv21 = ConvBlock(256, 128, kernel_size, laplacians[1])
            self.uconv22 = ConvBlock(128, 64, kernel_size, laplacians[1])

            # Decoding block 4
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
                Encoded data at the different encoding stages and the indices indicating the locations of the maxium values in
                unpooled images.
            """
            # x_enc1 = self.dropout1(x_enc1)

            # Block 1

            x_enc1 = self.conv11(x)
            x_enc1 = self.conv12(x_enc1)
            x_enc1 = self.conv13(x_enc1)
            # print(x_enc1.shape)
            # print(x.shape)

            x_enc1 += torch.transpose(self.conv1_res(torch.transpose(x, 2, 1)), 2, 1)
            # x_enc1 += self.conv1_res(x.transpose(0,2,1)).transpose(0,2,1)

            # Block 2
            x_enc2_ini, idx1 = self.pooling(x_enc1)
            x_enc2 = self.conv21(x_enc2_ini)
            x_enc2 = self.conv22(x_enc2)
            x_enc2 = self.conv23(x_enc2)
            x_enc2 += torch.transpose(self.conv2_res(torch.transpose(x_enc2_ini, 2, 1)), 2, 1)
            # x_enc2 += self.conv2_res(x_enc1.transpose(0,2,1)).transpose(0,2,1)

            # Block 3
            x_enc3_ini, idx2 = self.pooling(x_enc2)
            x_enc3 = self.conv31(x_enc3_ini)
            x_enc3 = self.conv32(x_enc3)
            x_enc3 = self.conv33(x_enc3)
            x_enc3 += torch.transpose(self.conv3_res(torch.transpose(x_enc3_ini, 2, 1)), 2, 1)
            # x_enc3 += self.conv3_res(x_enc2.transpose(0,2,1)).transpose(0,2,1)

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



#spherical_unet_lbmc_residuals_encoder

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

        self.conv1_res = Conv1dAuto(in_channels, 64*2, 1)

        # Encoding block 2
        self.conv21 = ConvBlock(64*2, 96*2, kernel_size, laplacians[1])
        self.conv23 = ConvBlock(96*2, 128*2, kernel_size, laplacians[1])

        self.conv2_res = Conv1dAuto(64*2, 128*2, 1)

        # Encoding block 3
        self.conv31 = ConvBlock(128*2, 256*2, kernel_size, laplacians[2])
        #self.conv32 = ConvBlock(256*2, 256*2, kernel_size, laplacians[2])
        self.conv33 = ConvBlock(256*2, 128*2, kernel_size, laplacians[2])

        self.conv3_res = Conv1dAuto(128*2, 128*2, 1)

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

#spherical_unet_lbmc_residuals_encoder_identity
class UNetSphericalHealpixResidual(torch.nn.Module):
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
        for i, nodes in enumerate(
                [num_nodes, num_nodes / kernel_size_pooling, num_nodes / (kernel_size * kernel_size_pooling)]):
            laplacian = _compute_laplacian_healpix(nodes)
            laplacians.append(laplacian)

        # Pooling - unpooling
        self.pooling = PoolMaxHealpix(kernel_size=kernel_size_pooling)
        self.unpool = UnpoolMaxHealpix(kernel_size=kernel_size_pooling)

        # Identity mapping
        self.id = Identity()

        # Encoding block 1
        self.conv11 = ConvBlock(in_channels, max(in_channels, 32 * 2), kernel_size, laplacians[0])
        self.conv12 = ConvBlock(max(in_channels, 32 * 2), 64 * 2, kernel_size, laplacians[0])
        self.conv13 = ConvBlock(max(in_channels, 64 * 2), 64 * 2, kernel_size, laplacians[0])

        # Encoding block 2
        self.conv21 = ConvBlock(64 * 2, 128 * 2, kernel_size, laplacians[1])
        self.conv22 = ConvBlock(128 * 2, 256 * 2, kernel_size, laplacians[1])
        self.conv23 = ConvBlock(256 * 2, 128 * 2, kernel_size, laplacians[1])

        # self.conv2_res = Conv1dAuto(64 * 2, 128 * 2, 1)

        # Encoding block 3
        self.conv31 = ConvBlock(128 * 2, 256 * 2, kernel_size, laplacians[2])
        self.conv32 = ConvBlock(256 * 2, 256 * 2, kernel_size, laplacians[2])
        self.conv33 = ConvBlock(256 * 2, 128 * 2, kernel_size, laplacians[2])

        # self.conv3_res = Conv1dAuto(128 * 2, 128 * 2, 1)

        # Decoding block 4
        self.uconv21 = ConvBlock(256 * 2, 128 * 2, kernel_size, laplacians[1])
        self.uconv22 = ConvBlock(128 * 2, 64 * 2, kernel_size, laplacians[1])

        # Decoding block 4
        self.uconv11 = ConvBlock(128 * 2, 64 * 2, kernel_size, laplacians[0])
        self.uconv12 = ConvBlock(64 * 2, 32 * 2, kernel_size, laplacians[0])
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
        x_enc1 = self.conv12(x_enc1)
        x_id1 = self.id(x_enc1)
        x_enc1 = self.conv13(x_enc1)

        x_enc1 += x_id1

        # Block 2
        x_enc2_ini, idx1 = self.pooling(x_enc1)
        x_enc2 = self.conv21(x_enc2_ini)
        x_id2 = self.id(x_enc2)
        x_enc2 = self.conv22(x_enc2)
        x_enc2 = self.conv23(x_enc2)

        x_enc2 += x_id2
        # Block 3
        x_enc3_ini, idx2 = self.pooling(x_enc2)
        x_id3 = self.id(x_enc3_ini)
        x_enc3 = self.conv31(x_enc3_ini)
        x_enc3 = self.conv32(x_enc3)
        x_enc3 = self.conv33(x_enc3)

        x_enc3 += x_id3
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

#spherical_unet_lbmc_residuals_adding_long

class UNetSphericalHealpixResidual(torch.nn.Module):
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
        for i, nodes in enumerate(
                [num_nodes, num_nodes / kernel_size_pooling, num_nodes / (kernel_size * kernel_size_pooling)]):
            laplacian = _compute_laplacian_healpix(nodes)
            laplacians.append(laplacian)

        # Pooling - unpooling
        self.pooling = PoolMaxHealpix(kernel_size=kernel_size_pooling)
        self.unpool = UnpoolMaxHealpix(kernel_size=kernel_size_pooling)

        # Identity mapping
        self.id = Identity()

        # Encoding block 1
        self.conv11 = ConvBlock(in_channels, max(in_channels, 32 * 1), kernel_size, laplacians[0])
        self.conv12 = ConvBlock(max(in_channels, 32 * 1), 64 * 1, kernel_size, laplacians[0])
        self.conv13 = ConvBlock(max(in_channels, 64 * 1), 64 * 1, kernel_size, laplacians[0])

        # Encoding block 2
        self.conv21 = ConvBlock(64 * 1, 128 * 1, kernel_size, laplacians[1])
        self.conv22 = ConvBlock(128 * 1, 256 * 1, kernel_size, laplacians[1])
        self.conv23 = ConvBlock(256 * 1, 128 * 1, kernel_size, laplacians[1])

        # self.conv2_res = Conv1dAuto(64 * 1, 128 * 1, 1)

        # Encoding block 3
        self.conv31 = ConvBlock(128 * 1, 256 * 1, kernel_size, laplacians[2])
        self.conv32 = ConvBlock(256 * 1, 256 * 1, kernel_size, laplacians[2])
        self.conv33 = ConvBlock(256 * 1, 128 * 1, kernel_size, laplacians[2])

        # self.conv3_res = Conv1dAuto(128 * 1, 128 * 1, 1)

        # Decoding block 4
        self.uconv21 = ConvBlock(128 * 1, 128 * 1, kernel_size, laplacians[1])
        self.uconv22 = ConvBlock(128 * 1, 64 * 1, kernel_size, laplacians[1])

        # Decoding block 4
        self.uconv11 = ConvBlock(64 * 1, 64 * 1, kernel_size, laplacians[0])
        self.uconv12 = ConvBlock(64 * 1, 32 * 1, kernel_size, laplacians[0])
        self.uconv13 = ConvCheb(32 * 1, out_channels, kernel_size, laplacians[0])

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
        x_id1 = self.id(x_enc1)
        x_enc1 = self.conv13(x_enc1)

        x_enc1 += x_id1

        # Block 2
        x_enc2_ini, idx1 = self.pooling(x_enc1)
        x_enc2 = self.conv21(x_enc2_ini)
        x_id2 = self.id(x_enc2)
        x_enc2 = self.conv22(x_enc2)
        x_enc2 = self.conv23(x_enc2)

        x_enc2 += x_id2
        # Block 3
        x_enc3_ini, idx2 = self.pooling(x_enc2)
        x_id3 = self.id(x_enc3_ini)
        x_enc3 = self.conv31(x_enc3_ini)
        x_enc3 = self.conv32(x_enc3)
        x_enc3 = self.conv33(x_enc3)

        x_enc3 += x_id3
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
        # x = torch.cat((x, x_enc2), dim=2)
        x += x_enc2
        x = self.uconv21(x)
        x = self.uconv22(x)

        # Block 1
        x = self.unpool(x, idx1)
        # x = torch.cat((x, x_enc1), dim=2)
        x += x_enc1
        x = self.uconv11(x)
        x = self.uconv12(x)
        x = self.uconv13(x)

        return x


#spherical_unet_lbmc_residuals_conv1d_long
class UNetSphericalHealpixResidual(torch.nn.Module):
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
        for i, nodes in enumerate(
                [num_nodes, num_nodes / kernel_size_pooling, num_nodes / (kernel_size * kernel_size_pooling)]):
            laplacian = _compute_laplacian_healpix(nodes)
            laplacians.append(laplacian)

        # Pooling - unpooling
        self.pooling = PoolMaxHealpix(kernel_size=kernel_size_pooling)
        self.unpool = UnpoolMaxHealpix(kernel_size=kernel_size_pooling)

        # Encoding block 1
        self.conv11 = ConvBlock(in_channels, max(in_channels, 32 * 2), kernel_size, laplacians[0])
        self.conv13 = ConvBlock(max(in_channels, 32 * 2), 64 * 2, kernel_size, laplacians[0])
        self.conv1_res = Conv1dAuto(64 * 2, 64 * 2, 1)

        # Encoding block 2
        self.conv21 = ConvBlock(64 * 2, 96 * 2, kernel_size, laplacians[1])
        self.conv23 = ConvBlock(96 * 2, 128 * 2, kernel_size, laplacians[1])
        self.conv2_res = Conv1dAuto(128 * 2, 128 * 2, 1)

        # Encoding block 3
        self.conv31 = ConvBlock(128 * 2, 256 * 2, kernel_size, laplacians[2])
        # self.conv32 = ConvBlock(256*2, 256*2, kernel_size, laplacians[2])
        self.conv33 = ConvBlock(256 * 2, 128 * 2, kernel_size, laplacians[2])

        # Decoding block 4
        self.uconv21 = ConvBlock(128 * 2, 128 * 2, kernel_size, laplacians[1])
        self.uconv22 = ConvBlock(128 * 2, 64 * 2, kernel_size, laplacians[1])

        # Decoding block 4
        self.uconv11 = ConvBlock(64 * 2, 64 * 2, kernel_size, laplacians[0])
        self.uconv12 = ConvBlock(64 * 2, 32 * 2, kernel_size, laplacians[0])
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
        x += torch.transpose(self.conv2_res(torch.transpose(x_enc2, 2, 1)), 2, 1)
        # x = torch.cat((x, x_enc2), dim=2)
        # x *= x_enc2
        x = self.uconv21(x)
        x = self.uconv22(x)

        # Block 1
        x = self.unpool(x, idx1)
        # x = torch.cat((x, x_enc1), dim=2)
        x += torch.transpose(self.conv1_res(torch.transpose(x_enc1, 2, 1)), 2, 1)
        # x += x_enc1
        x = self.uconv11(x)
        x = self.uconv12(x)
        x = self.uconv13(x)
        return x

class UNetSphericalHealpixResidualShort4LevelsNoLong(torch.nn.Module):
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
        self.uconv31 = ConvBlock(512 * 2, 256 * 2, kernel_size, laplacians[2])
        self.uconv32 = ConvBlock(256 * 2, 128 * 2, kernel_size, laplacians[2])

        self.uconv3_res = Conv1dAuto(512 * 2, 128 * 2, 1)

        # Decoding block 4
        self.uconv21 = ConvBlock(128 * 2, 128 * 2, kernel_size, laplacians[1])
        self.uconv22 = ConvBlock(128 * 2, 64 * 2, kernel_size, laplacians[1])

        self.uconv2_res = Conv1dAuto(128 * 2, 64 * 2, 1)

        # Decoding block 4
        self.uconv11 = ConvBlock(64 * 2, 64 * 2, kernel_size, laplacians[0])
        self.uconv12 = ConvBlock(64 * 2, 32 * 2, kernel_size, laplacians[0])

        self.uconv1_res = Conv1dAuto(64 * 2, 32 * 2, 1)

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
        x_cat = self.unpool(x_enc4, idx3)
        #x_cat = torch.cat((x, x_enc3), dim=2)
        x = self.uconv31(x_cat)
        x = self.uconv32(x)

        x += torch.transpose(self.uconv3_res(torch.transpose(x_cat, 2, 1)), 2, 1)

        # Block 2
        x_cat = self.unpool(x, idx2)
        #x_cat = torch.cat((x, x_enc2), dim=2)
        x = self.uconv21(x_cat)
        x = self.uconv22(x)
        x += torch.transpose(self.uconv2_res(torch.transpose(x_cat, 2, 1)), 2, 1)

        # Block 1
        x_cat = self.unpool(x, idx1)
        #x_cat = torch.cat((x, x_enc1), dim=2)
        x = self.uconv11(x_cat)
        x = self.uconv12(x)

        x += torch.transpose(self.uconv1_res(torch.transpose(x_cat, 2, 1)), 2, 1)

        x = self.uconv13(x)

        return x

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

# simple 3level arch
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