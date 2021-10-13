import torch.nn as nn
from .convolution import ChannelChanger

__all__ = ['DeconvUpSample', 'SubPixelUpSample']


class DeconvUpSample(nn.Module):
    def __init__(self, in_channels, up_factor=2):
        super().__init__()
        self.bicubic = nn.Upsample(scale_factor=up_factor)
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=in_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.initialize()

    def forward(self, input_tensor):
        output_tensor = self.conv(self.bicubic(input_tensor))
        return output_tensor

    def initialize(self):
        return


class SubPixelUpSample(nn.Module):
    def __init__(self, in_channels, up_factor=2):
        super().__init__()
        out_channels = in_channels * up_factor**2
        self.expand_channel = ChannelChanger(in_channels=in_channels,
                                             out_channels=out_channels)
        self.pixel_shuffle = nn.PixelShuffle(up_factor)
        self.initialize()

    def forward(self, input_tensor):
        output_tensor = self.pixel_shuffle(self.expand_channel(input_tensor))
        return output_tensor

    def initialize(self):
        return
