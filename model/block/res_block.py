import torch.nn as nn
from .convolution import DilatedConv2d, Conv1x1, Conv3x3, ChannelChanger

__all__ = ['ResBasicBlock', 'ResBottleneckBlock']


class ResBasicBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 downsample=None,
                 dilation=None,
                 norm_layer=None):
        super().__init__()

        if norm_layer == None:
            norm_layer = nn.BatchNorm2d

        self.downsample = None

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                Conv1x1(in_channels, out_channels, stride=stride),
                norm_layer(out_channels))

        if dilation == True:
            raise NotImplementedError(
                'Dilation not supported in ResBasicBlock.')

        self.conv1 = Conv3x3(in_channels=in_channels,
                             out_channels=out_channels,
                             stride=stride)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = Conv3x3(in_channels=out_channels,
                             out_channels=out_channels)
        self.bn2 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor):
        identity = input_tensor
        output_tensor = self.relu(self.bn1(self.conv1(input_tensor)))
        output_tensor = self.bn2(self.conv2(output_tensor))

        if self.downsample != None:
            identity = self.downsample(identity)

        output_tensor = output_tensor + identity
        output_tensor = self.relu(output_tensor)
        return output_tensor


class ResBottleneckBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 stride=1,
                 downsample=None,
                 dilation=None,
                 norm_layer=None):
        super().__init__()

        if norm_layer != None:
            self.norm_layer = nn.BatchNorm2d

        self.downsample = None

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                Conv1x1(in_channels, out_channels, stride=stride),
                norm_layer(out_channels))

        self.conv1 = ChannelChanger(in_channels=in_channels,
                                    out_channels=mid_channels)
        self.bn1 = self.norm_layer(mid_channels)

        if dilation != None:
            self.conv2 = DilatedConv2d(mid_channels,
                                       mid_channels,
                                       kernel_size=3,
                                       dilated_ratios=dilation,
                                       stride=stride,
                                       padding=1)
        else:
            self.conv2 = Conv3x3(mid_channels, mid_channels)

        self.bn2 = self.norm_layer(mid_channels)

        self.conv3 = ChannelChanger(mid_channels, out_channels=out_channels)
        self.bn3 = self.norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, input_tensor):
        identity = input_tensor
        out_tensor = self.relu(self.bn1(self.conv1(input_tensor)))
        out_tensor = self.relu(self.bn2(self.conv2(out_tensor)))
        out_tensor = self.bn3(self.conv3(out_tensor))

        if self.downsample != None:
            identity = self.downsample(identity)

        out_tensor = out_tensor + identity
        out_tensor = self.relu(out_tensor)
        return out_tensor
