import torch
import torch.nn as nn


class Classifier3D(nn.Module):
    """
    Modified from UNet3D: https://github.com/black0017/MedicalZooPytorch/blob/master/lib/medzoo/Unet3D.py
    """

    def __init__(self, in_channels, img_size=48, base_n_filter=8):
        super(Classifier3D, self).__init__()
        self.in_channels = in_channels
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upscale = nn.Upsample(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)

        self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)

        self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter * 2, self.base_n_filter * 2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter * 2)

        self.conv3d_c3 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter * 4, self.base_n_filter * 4)
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter * 4)

        self.conv3d_c4 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter * 8, self.base_n_filter * 8)
        self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter * 8)

        self.conv3d_c5 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter * 16, self.base_n_filter * 16)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 16, self.base_n_filter * 8)

        self.conv3d_l0 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 8)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.base_n_filter * 16 * (img_size // 16)**3, self.base_n_filter * 16)
        # leakyReLU
        self.dropout1d = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(self.base_n_filter * 16, 1)

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def forward(self, x):
        #  Level 1 context pathway
        out = self.conv3d_c1_1(x)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)

        # Level 2 context pathway
        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out

        # Level 3 context pathway
        out = self.conv3d_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        context_3 = out

        # Level 4 context pathway
        out = self.conv3d_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)
        context_4 = out

        # Level 5
        out = self.conv3d_c5(out)
        residual_5 = out
        out = self.norm_lrelu_conv_c5(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        out += residual_5
        
        out = self.flatten(out)
        out = self.fc1(out)
        # out = self.dropout1d(out)
        out = self.lrelu(out)
        
        out = self.fc2(out)
        return out


class TinyClassifier(nn.Module):
    def __init__(self):
        super(TinyClassifier, self).__init__()

        # 1,48,48,48
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(8)
        self.act1 = nn.LeakyReLU()
        self.mp1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # 8,24,24,24
        self.conv2 = nn.Conv3d(8, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.act2 = nn.LeakyReLU()
        self.mp2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # 32,12,12,12
        self.conv3 = nn.Conv3d(32, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.act3 = nn.LeakyReLU()
        self.mp3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # 128,6,6,6
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        self.act4 = nn.LeakyReLU()
        self.mp4 = nn.MaxPool3d(kernel_size=2, stride=2)

        # 256,3,3,3
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256*3*3*3, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.act5 = nn.LeakyReLU()

        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.act1(output)
        output = self.mp1(output)

        # 8,24,24,24
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.act2(output)
        output = self.mp2(output)

        # 32,12,12,12
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.act3(output)
        output = self.mp3(output)

        # 128,6,6,6
        output = self.conv4(output)
        output = self.bn4(output)
        output = self.act4(output)
        output = self.mp4(output)

        # 256,3,3,3
        output = self.flatten(output)
        output = self.fc1(output)
        output = self.bn5(output)
        output = self.act5(output)

        output = self.fc2(output)
        return output

