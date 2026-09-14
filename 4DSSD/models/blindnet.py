"""
Blind-spot denoising networks (BlindNet2D / BlindNet3D / BlindNet4D) used for
self-supervised denoising of 2D, 3D-frame-window, and 4D-STEM diffraction
data (see models/dataset.py for the matching input layouts).

The "blind-spot" trick shared by all three networks: a pixel's receptive
field must never include itself, so the network can be trained directly on
noisy data (predict each pixel from its neighbors only, cf. Noise2Self /
blind-spot self-supervised denoising) without needing clean targets.
- `shift`/`crop`, and the `blind=True` path of `Conv`/`Pool`, shift the
  feature map down by one row before each 3x3 conv/pool and crop the
  padded row back afterward, so a pixel only ever sees pixels strictly
  above it.
- Since a single pass is only blind in one direction, `rotate`/`unrotate`
  run the same pass on four 90-degree rotations of the input (stacked
  along the batch dimension) and recombine them along the channel
  dimension, so the network still gets full above/below/left/right
  context per pixel while remaining blind to the center pixel itself.
- `Blind_UNet` is the shared encoder/decoder body, built from `ENC_Conv`
  (downsampling) and `DEC_Conv` (upsampling with skip connections) blocks.

BlindNet3D and BlindNet4D additionally split the input channels into
several overlapping frame groups (3D: three 3-frame groups drawn from a
5-frame window; 4D: four 3-frame groups drawn from the 3x3 scan-grid
neighborhood), denoise each group independently with a shared first-stage
`Blind_UNet`, then fuse the per-group features with a second-stage
`Blind_UNet` before the final 1x1 conv head.

References (see the top-level README for full citations):
- Laine et al., "High-Quality Self-Supervised Deep Image Denoising",
  NeurIPS 2019 -- shift-based blind-spot conv + 4-way rotation ensembling.
- Sheth et al., "Unsupervised Deep Video Denoising", ICCV 2021 --
  multi-frame extension via overlapping frame groups + fusion network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class crop(nn.Module):
    """Drop the last row added by `shift`'s zero-padding."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, C, H, W = x.shape
        x = x[0:N, 0:C, 0:H-1, 0:W]
        return x

class shift(nn.Module):
    """Shift the feature map down by one row (zero-pad top, crop bottom)."""

    def __init__(self):
        super().__init__()
        self.shift_down = nn.ZeroPad2d((0,0,1,0))
        self.crop = crop()

    def forward(self, x):
        x = self.shift_down(x)
        x = self.crop(x)
        return x

class Conv(nn.Module):
    """3x3 conv + LeakyReLU. When `blind=True`, shifts input down by one row
    before the conv and crops back after, keeping the blind-spot property."""

    def __init__(self, in_channels, out_channels, bias=False, blind=True):
        super().__init__()
        self.blind = blind
        if blind:
            self.shift_down = nn.ZeroPad2d((0,0,1,0))
            self.crop = crop()
        self.replicate = nn.ReplicationPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, bias=bias)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        if self.blind:
            x = self.shift_down(x)
        x = self.replicate(x)
        x = self.conv(x)
        x = self.relu(x)
        if self.blind:
            x = self.crop(x)
        return x

class Pool(nn.Module):
    """2x2 max pool, blind-spot-safe via `shift` when `blind=True`."""

    def __init__(self, blind=True):
        super().__init__()
        self.blind = blind
        if blind:
            self.shift = shift()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        if self.blind:
            x = self.shift(x)
        x = self.pool(x)
        return x

class rotate(nn.Module):
    """Stack the input plus its 90/180/270-degree rotations along the batch
    dimension, so a single "blind above" pass covers all four directions."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x90 = x.transpose(2,3).flip(3)
        x180 = x.flip(2).flip(3)
        x270 = x.transpose(2,3).flip(2)
        x = torch.cat((x,x90,x180,x270), dim=0)
        return x

class unrotate(nn.Module):
    """Inverse of `rotate`: undo each rotation and concatenate the four
    results along the channel dimension instead of the batch dimension."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x0, x90, x180, x270 = torch.chunk(x, 4, dim=0)
        x90 = x90.transpose(2,3).flip(2)
        x180 = x180.flip(2).flip(3)
        x270 = x270.transpose(2,3).flip(3)
        x = torch.cat((x0,x90,x180,x270), dim=1)
        return x

class ENC_Conv(nn.Module):
    """Encoder block: three blind `Conv`s, optionally followed by a blind
    2x2 max pool (`reduce=True`) to downsample."""

    def __init__(self, in_channels, mid_channels, out_channels, bias=False, reduce=True, blind=True):
        super().__init__()
        self.reduce = reduce
        self.conv1 = Conv(in_channels, mid_channels, bias=bias, blind=blind)
        self.conv2 = Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.conv3 = Conv(mid_channels, out_channels, bias=bias, blind=blind)
        if reduce:
            self.pool = Pool(blind=blind)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.reduce:
            x = self.pool(x)
        return x

class DEC_Conv(nn.Module):
    """Decoder block: upsample by 2x, concatenate the matching encoder
    skip connection (`x_in`), then four blind `Conv`s."""

    def __init__(self, in_channels, mid_channels, out_channels, bias=False, blind=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = Conv(in_channels, mid_channels, bias=bias, blind=blind)
        self.conv2 = Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.conv3 = Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.conv4 = Conv(mid_channels, out_channels, bias=bias, blind=blind)

    def forward(self, x, x_in):
        x = self.upsample(x)

        # Smart Padding
        diffY = x_in.size()[2] - x.size()[2]
        diffX = x_in.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        x = torch.cat((x, x_in), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class Blind_UNet(nn.Module):
    """
    A small blind-spot U-Net: two downsampling `ENC_Conv` stages, one
    bottleneck `ENC_Conv` (no downsampling), then two `DEC_Conv` stages
    with skip connections back to `input` and the first encoder output.
    Input and output spatial size match; channel count goes from
    `n_channels` to `n_output`.
    """

    def __init__(self, n_channels=3, n_output=32, bias=False, blind=True):
        super().__init__()
        self.n_channels = n_channels
        self.bias = bias
        self.enc1 = ENC_Conv(n_channels, 32, 32, bias=bias, blind=blind)
        self.enc2 = ENC_Conv(32, 32, 32, bias=bias, blind=blind)
        self.enc3 = ENC_Conv(32, 64, 32, bias=bias, reduce=False, blind=blind)
        self.dec2 = DEC_Conv(64, 64, 64, bias=bias, blind=blind)
        self.dec1 = DEC_Conv(64+n_channels, 64, n_output, bias=bias, blind=blind)

    def forward(self, input):
        x1 = self.enc1(input)
        x2 = self.enc2(x1)
        x = self.enc3(x2)
        x = self.dec2(x, x1)
        x = self.dec1(x, input)
        return x

class BlindNet2D(nn.Module):
    """
    Blind-spot denoiser for a single image, e.g. an independent 2D image or
    the center frame of a 3D/4D-STEM stack (input shape (N, n_channels, H, W),
    typically n_channels=1). Applies the 4-rotation blind-spot `Blind_UNet`
    once, then a 1x1 conv head (nin_A/B/C) maps the 4*32=128 concatenated
    rotation channels down to `n_output`.
    """

    def __init__(self, n_channels=1, n_output=1, bias=False, blind=True):
        super().__init__()
        self.n_channels = n_channels
        self.c = n_channels
        self.n_output = n_output
        self.bias = bias
        self.blind = blind
        self.rotate = rotate()
        self.unet = Blind_UNet(n_channels=n_channels, bias=bias, blind=blind)
        if blind:
            self.shift = shift()
        self.unrotate = unrotate()
        self.nin_A = nn.Conv2d(128, 128, 1, bias=bias)
        self.nin_B = nn.Conv2d(128, 32, 1, bias=bias)
        self.nin_C = nn.Conv2d(32, n_output, 1, bias=bias)

    def forward(self, x):
        # Square
        N, C, H, W = x.shape
        if(H > W):
            diff = H - W
            x = F.pad(x, [diff // 2, diff - diff // 2, 0, 0], mode = 'reflect')
        elif(W > H):
            diff = W - H
            x = F.pad(x, [0, 0, diff // 2, diff - diff // 2], mode = 'reflect')

        x = self.rotate(x)
        x = self.unet(x)
        if self.blind:
            x = self.shift(x)
        x = self.unrotate(x)
        x = F.leaky_relu_(self.nin_A(x), negative_slope=0.1)
        x = F.leaky_relu_(self.nin_B(x), negative_slope=0.1)
        x = self.nin_C(x)
        x = F.relu(x)

        # Unsquare
        if(H > W):
            diff = H - W
            x = x[:, :, 0:H, (diff // 2):(diff // 2 + W)]
        elif(W > H):
            diff = W - H
            x = x[:, :, (diff // 2):(diff // 2 + H), 0:W]
        return x

class BlindNet3D(nn.Module):
    """
    Blind-spot denoiser for a 5-frame window from a stacked-frame volume
    (tilt series / focal series / video / z-stack; see DataSet3D in
    models/dataset.py). Input shape (N, 5, H, W).

    The 5 channels are split into three overlapping 3-frame groups
    (indices [0,1,2], [1,2,3], [2,3,4] -- each including the shared center
    frame, index 2). Each group is denoised independently by a shared
    first-stage `Blind_UNet` (n_output=16), the three 16-channel outputs are
    concatenated (-> 48 channels) and refined by a second-stage `Blind_UNet`
    (48 -> 48), then a 1x1 conv head maps the 4-rotation-concatenated
    4*48=192 channels down to `n_output`.
    """

    def __init__(self, n_channels=3, n_output=1, bias=False, blind=True):
        super().__init__()
        self.c = n_channels
        self.out_channels = n_output
        self.blind = blind
        self.rotate = rotate()

        self.denoiser_1 = Blind_UNet(n_channels=n_channels, n_output=16, bias=bias, blind=blind)

        # Second-stage denoiser: input channels (48) must match the
        # concatenated output of the three first-stage groups (3 * 16).
        self.denoiser_2 = Blind_UNet(n_channels=48, n_output=48, bias=bias, blind=blind)
                    
        if blind:
            self.shift = shift()
            
        self.unrotate = unrotate()
        self.nin_A = nn.Conv2d(192, 96, 1, bias=bias)
        self.nin_B = nn.Conv2d(96, 32, 1, bias=bias)
        self.nin_C = nn.Conv2d(32, n_output, 1, bias=bias)

    def forward(self, x):
        # Square
        N, C, H, W = x.shape

        if(H > W):
            diff = H - W
            x = F.pad(x, [diff // 2, diff - diff // 2, 0, 0], mode = 'reflect')
        elif(W > H):
            diff = W - H
            x = F.pad(x, [0, 0, diff // 2, diff - diff // 2], mode = 'reflect')
        
        i1 = self.rotate(x[:, [0, 1, 2], :, :])
        i2 = self.rotate(x[:, [1, 2, 3], :, :])        
        i3 = self.rotate(x[:, [2, 3, 4], :, :])

        # Process each group through the first denoiser
        y1 = self.denoiser_1(i1)
        y2 = self.denoiser_1(i2)
        y3 = self.denoiser_1(i3)

        # Concatenate all outputs from the first stage
        y = torch.cat((y1, y2, y3), dim=1)

        # Process through the second denoiser
        x = self.denoiser_2(y)

        if self.blind:
            x = self.shift(x)
        x = self.unrotate(x)
        x = F.leaky_relu_(self.nin_A(x), negative_slope=0.1)
        x = F.leaky_relu_(self.nin_B(x), negative_slope=0.1)
        x = self.nin_C(x)
        
        x = F.relu(x)

        # Unsquare
        if(H > W):
            diff = H - W
            x = x[:, :, 0:H, (diff // 2):(diff // 2 + W)]
        elif(W > H):
            diff = W - H
            x = x[:, :, (diff // 2):(diff // 2 + H), 0:W]
        return x

class BlindNet4D(nn.Module):
    """
    Blind-spot denoiser for the 3x3 scan-grid neighborhood of a 4D-STEM
    diffraction stack (see DataSet4D in models/dataset.py). Input shape
    (N, 9, H, W), channel order:
        0 1 2
        3 4 5
        6 7 8
    with 4 = the center (target) scan position.

    The 9 channels are split into four overlapping 3-frame groups through
    the center: the two diagonals (0,4,8) and (2,4,6), plus the vertical
    (1,4,7) and horizontal (3,4,5) lines. Each group is denoised
    independently by a shared first-stage `Blind_UNet` (n_output=16), the
    four 16-channel outputs are concatenated (-> 64 channels) and refined
    by a second-stage `Blind_UNet` (64 -> 64), then a 1x1 conv head maps the
    4-rotation-concatenated 4*64=256 channels down to `n_output`.
    """

    def __init__(self, n_channels=3, n_output=1, bias=False, blind=True):
        super().__init__()
        self.c = n_channels
        self.out_channels = n_output
        self.blind = blind
        self.rotate = rotate()

        self.denoiser_1 = Blind_UNet(n_channels=n_channels, n_output=16, bias=bias, blind=blind)

        # Second-stage denoiser: input channels (64) must match the
        # concatenated output of the four first-stage groups (4 * 16).
        self.denoiser_2 = Blind_UNet(n_channels=64, n_output=64, bias=bias, blind=blind)
                    
        if blind:
            self.shift = shift()
            
        self.unrotate = unrotate()
        self.nin_A = nn.Conv2d(256, 128, 1, bias=bias)
        self.nin_B = nn.Conv2d(128, 32, 1, bias=bias)
        self.nin_C = nn.Conv2d(32, n_output, 1, bias=bias)

    def forward(self, x):
        # Square
        N, C, H, W = x.shape

        if(H > W):
            diff = H - W
            x = F.pad(x, [diff // 2, diff - diff // 2, 0, 0], mode = 'reflect')
        elif(W > H):
            diff = W - H
            x = F.pad(x, [0, 0, diff // 2, diff - diff // 2], mode = 'reflect')

        # Process four groups of frames in a 3x3 grid:
        # 0 1 2
        # 3 4 5
        # 6 7 8
        # Where 4 is the center frame
        
        # Group 1: Diagonal from top-left to bottom-right (0, 4, 8)
        i1 = self.rotate(x[:, [0, 4, 8], :, :])
        
        # Group 2: Vertical line (1, 4, 7)
        i2 = self.rotate(x[:, [1, 4, 7], :, :])
        
        # Group 3: Diagonal from top-right to bottom-left (2, 4, 6)
        i3 = self.rotate(x[:, [2, 4, 6], :, :])
        
        # Group 4: Horizontal line (3, 4, 5)
        i4 = self.rotate(x[:, [3, 4, 5], :, :])

        # Process each group through the first denoiser
        y1 = self.denoiser_1(i1)
        y2 = self.denoiser_1(i2)
        y3 = self.denoiser_1(i3)
        y4 = self.denoiser_1(i4)

        # Concatenate all outputs from the first stage
        y = torch.cat((y1, y2, y3, y4), dim=1)

        # Process through the second denoiser
        x = self.denoiser_2(y)

        if self.blind:
            x = self.shift(x)
        x = self.unrotate(x)
        x = F.leaky_relu_(self.nin_A(x), negative_slope=0.1)
        x = F.leaky_relu_(self.nin_B(x), negative_slope=0.1)
        x = self.nin_C(x)
        
        x = F.relu(x)

        # Unsquare
        if(H > W):
            diff = H - W
            x = x[:, :, 0:H, (diff // 2):(diff // 2 + W)]
        elif(W > H):
            diff = W - H
            x = x[:, :, (diff // 2):(diff // 2 + H), 0:W]
        return x
