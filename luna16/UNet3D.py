import torch
import torch.nn as nn

class UNet3D(nn.Module):
    """
    Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
    """
    
    def __init__(self, in_channels, n_classes, device, base_n_filter=8, drop_p=0.6, T = 10):
        super(UNet3D, self).__init__()
        
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter
        self.drop_p = drop_p
        
        self.encoder = Encoder(self.in_channels, self.base_n_filter, self.drop_p)
        self.decoders_mean = Decoder(self.in_channels, self.n_classes, self.base_n_filter, self.drop_p)
        self.decoders_var = Decoder(self.in_channels, self.n_classes, self.base_n_filter, self.drop_p)
        self.softmax = nn.Softmax(dim=1)
        self.T = T
        self.device = device

    def forward(self, x):
        out, context_1, context_2, context_3, context_4 = self.encoder(x)
        mean = self.decoders_mean(out, context_1, context_2, context_3, context_4)
        sigma = self.decoders_var(out, context_1, context_2, context_3, context_4)
        return mean, sigma

    def predict(self, x):
        mean, sigma = self.forward(x)
        size = list(sigma.size())
        size[1] = self.n_classes
        running_x = torch.zeros(size).to(self.device)
        for i in range(self.T):
            x = mean + sigma * torch.randn(size).to(self.device)
            running_x += self.softmax(x)
        return running_x / self.T


class Encoder(nn.Module):
    def __init__(self, in_channels, base_n_filter=8, drop_p=0.6):
        """
        Encoder layer of UNet3D
        Args:
            in_channels (int): number of input channels
            base_n_filter (int): number of channel which get doubled for every downsample
            drop_p (float): dropout prob.
        """
        super(Encoder, self).__init__()
        
        if in_channels == 1:
            self.in_channels = 4
        else:
            self.in_channels = in_channels
        self.base_n_filter = base_n_filter
        self.drop_p = drop_p
        self.lrelu = nn.LeakyReLU()
        
        self.conv3d_c0_0 = nn.Conv3d(1, 4, kernel_size=3, stride=1, padding=1)
        
        #  Level 1
        self.conv3d_c1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1)
        self.lrelu_conv_drop_lrelu_conv_c1 = self.lrelu_conv_drop_lrelu_conv(self.base_n_filter, self.base_n_filter, self.drop_p)
        self.inorm_lrelu_c1 = self.inorm_lrelu(self.base_n_filter)
        
        #  Level 2
        self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=2, padding=1)
        self.norm_lrelu_conv_drop_norm_lrelu_conv_c2 = self.norm_lrelu_conv_drop_norm_lrelu_conv(self.base_n_filter * 2, self.base_n_filter * 2, self.drop_p)
        self.inorm_lrelu_c2 = self.inorm_lrelu(self.base_n_filter * 2)
        
        #  Level 3
        self.conv3d_c3 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter*4, kernel_size=3, stride=2, padding=1)
        self.norm_lrelu_conv_drop_norm_lrelu_conv_c3 = self.norm_lrelu_conv_drop_norm_lrelu_conv(self.base_n_filter * 4, self.base_n_filter * 4, self.drop_p)
        self.inorm_lrelu_c3 = self.inorm_lrelu(self.base_n_filter * 4)

        # Level 4
        self.conv3d_c4 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=3, stride=2, padding=1)
        self.norm_lrelu_conv_drop_norm_lrelu_conv_c4 = self.norm_lrelu_conv_drop_norm_lrelu_conv(self.base_n_filter * 8, self.base_n_filter * 8, self.drop_p)
        self.inorm_lrelu_c4 = self.inorm_lrelu(self.base_n_filter * 8)

        # Level 5 (bottommost layer right before expansion path)
        self.conv3d_c5 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 16, kernel_size=3, stride=2, padding=1)
        self.norm_lrelu_conv_drop_norm_lrelu_conv_c5 = self.norm_lrelu_conv_drop_norm_lrelu_conv(self.base_n_filter * 16, self.base_n_filter * 16, self.drop_p)

    def norm_lrelu_conv_drop_norm_lrelu_conv(self, feat_in, feat_out, drop_p):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1),
            nn.Dropout3d(p=drop_p),
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1),
        )
    
    def lrelu_conv_drop_lrelu_conv(self, feat_in, feat_out, drop_p):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1),
            nn.Dropout3d(p=drop_p),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1),
        )
        
    def inorm_lrelu(self, feat_in):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
        )
    
    def forward(self, x):
        # if x is single channel (CT in our case), map to 4 channels
        if x.shape[1] == 1:
            x = self.conv3d_c0_0(x)
        
        #  Level 1 context pathway
        out = self.conv3d_c1(x)
        residual_1 = out
        out = self.lrelu_conv_drop_lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm_lrelu_c1(out)

        # Level 2 context pathway
        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_drop_norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm_lrelu_c2(out)
        context_2 = out

        # Level 3 context pathway
        out = self.conv3d_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_drop_norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm_lrelu_c3(out)
        context_3 = out

        # Level 4 context pathway
        out = self.conv3d_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_drop_norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm_lrelu_c4(out)
        context_4 = out

        # Level 5
        out = self.conv3d_c5(out)
        residual_5 = out
        out = self.norm_lrelu_conv_drop_norm_lrelu_conv_c5(out)
        out += residual_5
        
        return out, context_1, context_2, context_3, context_4

class Decoder(nn.Module):
    """Decoder layer decodes the features by unpooling with respect to
    the pooling indices of the corresponding decoder part.
    Args:
        n_in_feat (int): number of input features
        n_out_feat (int): number of output features
        n_blocks (int): number of conv-batch-relu block inside the decoder
        drop_rate (float): dropout rate to use
    """

    def __init__(self, in_channels, n_classes, base_n_filter=8, drop_p=0.6):
        super(Decoder, self).__init__()
        
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter
        self.drop_p = drop_p
        self.upscale = nn.Upsample(scale_factor=2, mode='nearest')
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 16,
                                                                                             self.base_n_filter * 8)       
        self.conv_norm_lrelu_l0 = self.conv_norm_lrelu(self.base_n_filter * 8, self.base_n_filter * 8, 1, 0)
        
        
        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter * 16, self.base_n_filter * 16,3,1)
        self.conv_norm_lrelu_upscale_conv_norm_lrelu_l1 = self.conv_norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8,
                                                                                             self.base_n_filter * 4)
        
        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter * 8, self.base_n_filter * 8,3,1)
        self.conv_norm_lrelu_upscale_conv_norm_lrelu_l2 = self.conv_norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
                                                                                             self.base_n_filter * 2)
        
        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 4,3,1)
        self.conv_norm_lrelu_upscale_conv_norm_lrelu_l3 = self.conv_norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
                                                                                             self.base_n_filter)
        
        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter * 2,3,1)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0)

        self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter * 8, self.n_classes, kernel_size=1, stride=1, padding=0)
        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter * 4, self.n_classes, kernel_size=1, stride=1, padding=0)

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1),
            # dropout test in decoder ##
            nn.Dropout3d(p=self.drop_p),
            ##------------------------##
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU()
        )
    
    def conv_norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in*2, feat_out*2, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1),
            # dropout test in decoder ##
            nn.Dropout3d(p=self.drop_p),
            ##------------------------##
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU()
        )
    
    def conv_norm_lrelu(self, feat_in, feat_out, kernel, padding):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=kernel, stride=1, padding=padding),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU()
        )
    
        
    def forward(self, out, context_1, context_2, context_3, context_4):    
        # Level 0 localization pathway
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)
        out = self.conv_norm_lrelu_l0(out)
        
        # Level 1 localization pathway
        out = torch.cat([out, context_4], dim=1)
        out = self.conv_norm_lrelu_l1(out)
        out = self.conv_norm_lrelu_upscale_conv_norm_lrelu_l1(out)

        # Level 2 localization pathway
        out = torch.cat([out, context_3], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        ds2 = out
        out = self.conv_norm_lrelu_upscale_conv_norm_lrelu_l2(out)

        # Level 3 localization pathway
        out = torch.cat([out, context_2], dim=1)
        out = self.conv_norm_lrelu_l3(out)
        ds3 = out
        out = self.conv_norm_lrelu_upscale_conv_norm_lrelu_l3(out)

        # Level 4 localization pathway
        out = torch.cat([out, context_1], dim=1)
        out = self.conv_norm_lrelu_l4(out)
        out_pred = self.conv3d_l4(out)

        ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
        ds1_ds2_sum_upscale = self.upscale(ds2_1x1_conv)
        ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.upscale(ds1_ds2_sum_upscale_ds3_sum)

        out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
        
        return out