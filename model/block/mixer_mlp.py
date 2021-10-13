import torch.nn as nn

__all__ = [
    'PerPatchFC', 'ResidualLayerNorm', 'TokenMixingMLP', 'ChannelMixingMLP',
    'MixerLayer', 'TransposePerPatchFC'
]


class PerPatchFC(nn.Module):
    def __init__(self, input_channels, patch_size, num_channels):
        super().__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        if isinstance(patch_size, int):
            self.per_patch_fc = nn.Linear(patch_size**2 * input_channels,
                                          num_channels)
        elif isinstance(patch_size, (tuple, list)) and len(patch_size) == 2:
            patch_size = tuple(patch_size)
            self.per_patch_fc = nn.Linear(
                patch_size[0] * patch_size[1] * input_channels, num_channels)

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, input_tensors):
        patches = self.unfold(input_tensors)
        patches = patches.permute(0, 2, 1).contiguous()
        patches_feature = self.per_patch_fc(patches)
        return patches_feature


class TransposePerPatchFC(nn.Module):
    def __init__(self, output_size, output_channels, patch_size, num_channels):
        super().__init__()

        if isinstance(patch_size, int):
            self.per_patch_fc = nn.Linear(num_channels,
                                          output_channels * patch_size**2)
        elif isinstance(patch_size, (tuple, list)) and len(patch_size) == 2:
            patch_size = tuple(patch_size)
            self.per_patch_fc = nn.Linear(
                num_channels, output_channels * patch_size[0] * patch_size[1])

        self.fold = nn.Fold(output_size=output_size,
                            kernel_size=patch_size,
                            stride=patch_size)

    def forward(self, input_tensors):
        patches = self.per_patch_fc(input_tensors)
        patches = patches.permute(0, 2, 1).contiguous()
        output_tensors = self.fold(patches)
        return output_tensors


class ResidualLayerNorm(nn.Module):
    def __init__(self, num_channels, model):
        super().__init__()
        self.model = model
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, input_tensors):
        return self.model(self.norm(input_tensors)) + input_tensors


class TokenMixingMLP(nn.Module):
    def __init__(self, num_patches, expansion_factor=4, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Conv1d(num_patches,
                             num_patches * expansion_factor,
                             kernel_size=1)
        self.fc2 = nn.Conv1d(num_patches * expansion_factor,
                             num_patches,
                             kernel_size=1)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, input_tensors):
        output_tensors = self.dropout1(self.activation(
            self.fc1(input_tensors)))
        output_tensors = self.dropout2(self.fc2(output_tensors))
        return output_tensors


class ChannelMixingMLP(nn.Module):
    def __init__(self, num_channels, expansion_factor=4, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(num_channels, num_channels * expansion_factor)
        self.fc2 = nn.Linear(num_channels * expansion_factor, num_channels)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, input_tensors):
        output_tensors = self.dropout1(self.activation(
            self.fc1(input_tensors)))
        output_tensors = self.dropout2(self.fc2(output_tensors))
        return output_tensors


class MixerLayer(nn.Module):
    def __init__(self,
                 num_patches,
                 num_channels,
                 expansion_factor=4,
                 dropout=0.0):
        super().__init__()
        self.token_mixing_layer = ResidualLayerNorm(
            num_channels=num_channels,
            model=TokenMixingMLP(num_patches=num_patches,
                                 expansion_factor=expansion_factor,
                                 dropout=dropout))
        self.channel_mixing_layer = ResidualLayerNorm(
            num_channels=num_channels,
            model=ChannelMixingMLP(num_channels=num_channels,
                                   expansion_factor=expansion_factor,
                                   dropout=dropout))

    def forward(self, input_tensors):
        output_tensors = self.token_mixing_layer(input_tensors)
        output_tensors = self.channel_mixing_layer(output_tensors)
        return output_tensors
