import torch.nn as nn

__all__ = ['ConvDownSample']


class ConvDownSample(nn.Module):
    def __init__(self, in_channels, down_factor=2):
        super().__init__()
        padding = down_factor // 2
        kernel_size = padding * 2 + 1
        self.down_sample = nn.Conv2d(in_channels=in_channels,
                                     out_channels=in_channels,
                                     kernel_size=kernel_size,
                                     stride=down_factor,
                                     padding=padding)
        self.initialize()

    def forward(self, input_tensor):
        output_tensor = self.down_sample(input_tensor)
        return output_tensor

    def initialize(self):
        nn.init.kaiming_normal_(self.down_sample.weight)
