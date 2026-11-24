"""Lightweight 3D U-Net for binary whole-tumor segmentation."""

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Two 3x3x3 convolutions, each followed by instance norm and leaky ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    """Encoder-decoder with skip connections.

    Each encoder level halves D, H and W with max pooling, so patch sides must be
    divisible by 2 ** (len(features) - 1). The output is one logit per voxel.
    """

    def __init__(self, in_channels=4, out_channels=1, features=(16, 32, 64, 128, 256)):
        super().__init__()
        self.encoders = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(ConvBlock(ch, f))
            ch = f
        self.pool = nn.MaxPool3d(2)
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for f in reversed(features[:-1]):
            self.ups.append(nn.ConvTranspose3d(ch, f, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(2 * f, f))
            ch = f
        self.head = nn.Conv3d(ch, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            if i < len(self.encoders) - 1:
                skips.append(x)
                x = self.pool(x)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
