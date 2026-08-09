"""The fixed 160x288 head-centre network."""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F


class Conv(nn.Module):
    def __init__(self, inputs, outputs, kernel, stride=1, groups=1, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(
            inputs, outputs, kernel, stride, kernel // 2, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(outputs)
        self.relu = relu

    def forward(self, x):
        x = self.bn(self.conv(x))
        return F.relu(x) if self.relu else x


class Down(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.depthwise = Conv(inputs, inputs, 3, 2, inputs)
        self.pointwise = Conv(inputs, outputs, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depthwise = Conv(channels, channels, 3, groups=channels)
        self.pointwise = Conv(channels, channels, 1, relu=False)

    def forward(self, x):
        return F.relu(x + self.pointwise(self.depthwise(x)))


class Refine(nn.Module):
    def __init__(self):
        super().__init__()
        self.depthwise = Conv(32, 32, 3, groups=32)
        self.pointwise = Conv(32, 32, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class HCDS31(nn.Module):
    """31.26 MMAC, 66,224 convolution weights, output [N,16,40,72]."""

    def __init__(self):
        super().__init__()
        self.deploy = False
        self.stem = Conv(3, 16, 3, 2)
        self.down1, self.res1 = Down(16, 32), nn.Sequential(Residual(32))
        self.down2 = Down(32, 48)
        self.res2 = nn.Sequential(*[Residual(48) for _ in range(2)])
        self.down3 = Down(48, 64)
        self.res3 = nn.Sequential(*[Residual(64) for _ in range(3)])
        self.down4 = Down(64, 96)
        self.res4 = nn.Sequential(*[Residual(96) for _ in range(2)])
        self.lateral3 = Conv(64, 32, 1, relu=False)
        self.deep4 = Conv(96, 32, 1, relu=False)
        self.refine3 = Refine()
        self.lateral2, self.refine2 = Conv(48, 32, 1, relu=False), Refine()
        self.lateral1, self.refine1 = Conv(32, 32, 1, relu=False), Refine()
        self.head = nn.Conv2d(32, 16, 1)

    def forward(self, x):
        x = self.stem(x)
        s1 = self.res1(self.down1(x))
        s2 = self.res2(self.down2(s1))
        s3 = self.res3(self.down3(s2))
        s4 = self.res4(self.down4(s3))
        p3 = self.refine3(
            self.lateral3(s3)
            + F.interpolate(self.deep4(s4), scale_factor=2, mode="nearest")
        )
        p2 = self.refine2(
            self.lateral2(s2) + F.interpolate(p3, scale_factor=2, mode="nearest")
        )
        p1 = self.refine1(
            self.lateral1(s1) + F.interpolate(p2, scale_factor=2, mode="nearest")
        )
        output = self.head(p1)
        return torch.clamp(output, -16, 15.5) if self.deploy else output


def deployment_model(model: HCDS31) -> HCDS31:
    """Fold the fixed Q4/Q7 output encoding into a copy of the model."""
    model = copy.deepcopy(model).eval()
    gains = model.head.weight.new_tensor([2, 16, 16, 16, 16] + [0] * 11)
    with torch.no_grad():
        model.head.weight.mul_(gains[:, None, None, None])
        model.head.bias.mul_(gains)

    model.deploy = True
    return model
