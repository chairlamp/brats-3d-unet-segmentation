"""Exact, faster re-implementations of the U-Net's 3D convolutions for Apple's MPS backend.

On MPS the weight gradient of nn.Conv3d is about 25x slower than its forward pass, which limits
the training budget. The two layer types of UNet3D are rewritten with operations MPS trains fast:

* Conv3dVia2d: a 3x3x3 convolution (padding 1) is the sum over the three depth taps kd of a 2D
  3x3 convolution of slice d + kd - 1 with weight[:, :, kd]. Stacking the three shifted copies of
  the input on the channel axis turns that sum into ONE conv2d with 3 * C_in input channels.
* UpVia1x1: a transposed convolution with kernel 2 and stride 2 never overlaps its outputs, so it
  is a per-voxel matrix product (C_in -> C_out x 2 x 2 x 2) followed by depth-to-space.

Both keep the parameter names and shapes of the layers they replace, so a state_dict trained with
them loads into the plain UNet3D (and vice versa). Measured on an M4 Max (one training iteration):
64^3 x 4: 0.678 -> 0.230 s, 96^3 x 2: 1.161 -> 0.389 s, 128^3 x 2: 2.758 -> 0.933 s; logits agree
with the original layers to about 1e-5.
"""

import torch
import torch.nn.functional as F
from torch import nn


class Conv3dVia2d(nn.Module):
    """Drop-in replacement for nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)."""

    def __init__(self, in_ch, out_ch, bias=False):
        super().__init__()
        ref = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=bias)  # same default initialisation
        self.weight = nn.Parameter(ref.weight.detach().clone())  # [C_out, C_in, 3, 3, 3]
        self.bias = nn.Parameter(ref.bias.detach().clone()) if bias else None

    def forward(self, x):
        n, c, d, h, w = x.shape
        xp = F.pad(x, (0, 0, 0, 0, 1, 1))                                  # zero-pad depth by 1
        x3 = torch.cat([xp[:, :, k:k + d] for k in range(3)], dim=1)       # [N, 3C, D, H, W], tap-major
        x3 = x3.permute(0, 2, 1, 3, 4).reshape(n * d, 3 * c, h, w)         # one 2D image per slice
        w2 = self.weight.permute(0, 2, 1, 3, 4).reshape(self.weight.shape[0], 3 * c, 3, 3)
        y = F.conv2d(x3, w2, self.bias, padding=1)                         # [N*D, C_out, H, W]
        return y.reshape(n, d, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()


class UpVia1x1(nn.Module):
    """Drop-in replacement for nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        ref = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.weight = nn.Parameter(ref.weight.detach().clone())  # [C_in, C_out, 2, 2, 2]
        self.bias = nn.Parameter(ref.bias.detach().clone())

    def forward(self, x):
        n, _, d, h, w = x.shape
        o = self.weight.shape[1]
        # out[n, o, 2d+i, 2h+j, 2w+k] = sum_c x[n, c, d, h, w] * weight[c, o, i, j, k] + bias[o]
        y = torch.einsum("ncdhw,coijk->nodihjwk", x, self.weight)
        return y.reshape(n, o, 2 * d, 2 * h, 2 * w) + self.bias.view(1, o, 1, 1, 1)


def make_fast(model):
    """Replace every 3x3x3 Conv3d and 2x2x2/stride-2 ConvTranspose3d in place, copying the weights."""
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Conv3d) and child.kernel_size == (3, 3, 3) and child.padding == (1, 1, 1) \
                    and child.stride == (1, 1, 1) and child.groups == 1:
                new = Conv3dVia2d(child.in_channels, child.out_channels, bias=child.bias is not None)
            elif isinstance(child, nn.ConvTranspose3d) and child.kernel_size == (2, 2, 2) and child.stride == (2, 2, 2):
                new = UpVia1x1(child.in_channels, child.out_channels)
            else:
                continue
            new.load_state_dict(child.state_dict())
            setattr(module, name, new.to(child.weight.device))
    return model
