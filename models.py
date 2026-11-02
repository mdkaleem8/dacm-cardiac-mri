"""Backbones and the degradation estimator.

PlainUNet follows the nnU-Net 2D default topology (two 3x3 conv-InstanceNorm-
LeakyReLU per stage, strided-conv downsampling, transposed-conv upsampling,
feature widths 32..512). It is trained with our own loop, not with the nnU-Net
framework, so results are reported as "U-Net (nnU-Net 2D configuration)".
"""
from __future__ import annotations

import inspect

import torch
import torch.nn as nn


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout, stride=1):
        super().__init__(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.InstanceNorm2d(cout, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.InstanceNorm2d(cout, affine=True),
            nn.LeakyReLU(0.01, inplace=True))


class PlainUNet(nn.Module):
    def __init__(self, in_ch=1, n_classes=4, feats=(32, 64, 128, 256, 512, 512)):
        super().__init__()
        self.enc = nn.ModuleList()
        c = in_ch
        for i, f in enumerate(feats):
            self.enc.append(ConvBlock(c, f, 1 if i == 0 else 2))
            c = f
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        for i in range(len(feats) - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(feats[i], feats[i - 1], 2, 2))
            self.dec.append(ConvBlock(2 * feats[i - 1], feats[i - 1]))
        self.head = nn.Conv2d(feats[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for blk in self.enc:
            x = blk(x)
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], 1))
        return self.head(x)


# Modules whose outputs are modulated by DACM, per backbone.
UNET_STAGES = [f"enc.{i}" for i in range(6)] + [f"dec.{i}" for i in range(5)]
SWIN_STAGES = ["encoder1", "encoder2", "encoder3", "encoder4", "encoder10",
               "decoder5", "decoder4", "decoder3", "decoder2", "decoder1"]


def make_norms_affine(net: nn.Module) -> int:
    """Give every instance-norm layer without one a learnable per-channel scale and shift.

    Control for the zero-conditioned module: in Swin-UNETR the conv-block instance
    norms have no affine parameters, so a constant FiLM scale and shift after them adds
    capacity. This turns that capacity on in the plain backbone. Returns the layer count.
    """
    n = 0
    for m in net.modules():
        if isinstance(m, nn.modules.instancenorm._InstanceNorm) and not m.affine:
            m.affine = True
            m.weight = nn.Parameter(torch.ones(m.num_features))
            m.bias = nn.Parameter(torch.zeros(m.num_features))
            n += 1
    return n


def build_backbone(name: str, img_size=224, n_classes=4, norm_affine=False):
    if name == "unet":
        net, stages = PlainUNet(1, n_classes), UNET_STAGES
    elif name == "swinunetr":
        from monai.networks.nets import SwinUNETR
        kw = dict(in_channels=1, out_channels=n_classes, feature_size=48, spatial_dims=2)
        if "img_size" in inspect.signature(SwinUNETR.__init__).parameters:
            kw["img_size"] = (img_size, img_size)
        net, stages = SwinUNETR(**kw), SWIN_STAGES
    else:
        raise ValueError(f"unknown backbone {name}")
    if norm_affine:
        make_norms_affine(net)
    return net, stages


# ---------------------------------------------------------------------------
# Degradation estimator
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
        self.skip = (nn.Identity() if stride == 1 and cin == cout else
                     nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.f(x) + self.skip(x))


class DegradationEstimator(nn.Module):
    """Small ResNet with a multi-label type head and a per-type severity head."""

    def __init__(self, k=5, widths=(32, 64, 128, 256)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, widths[0], 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        layers, c = [], widths[0]
        for i, w in enumerate(widths):
            layers += [ResBlock(c, w, 1 if i == 0 else 2), ResBlock(w, w, 1)]
            c = w
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.type_head = nn.Linear(c, k)
        self.sev_head = nn.Linear(c, k)

    def forward(self, x):
        h = self.pool(self.body(self.stem(x))).flatten(1)
        return self.type_head(h), torch.sigmoid(self.sev_head(h))

    @torch.no_grad()
    def conditioning(self, x):
        """z = [p ; p * s] (methodology, Eq. cvec)."""
        logits, s = self(x)
        p = torch.sigmoid(logits)
        return torch.cat([p, p * s], 1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
