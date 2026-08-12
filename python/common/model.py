from __future__ import annotations

import torch
from torch import nn

from common.preprocessing import SEMANTICS


class DepthwiseBlock(nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, stride: int = 1
    ) -> None:
        super().__init__()
        self.residual = stride == 1 and input_channels == output_channels
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=input_channels,
                bias=False,
            ),
            nn.BatchNorm2d(input_channels),
            nn.ReLU6(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
        )
        self.activation = nn.ReLU6(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = self.pointwise(self.depthwise(inputs))
        if self.residual:
            result = result + inputs
        return self.activation(result)


class TrackerModel(nn.Module):
    """Quantization-friendly detector whose output stays at stride four."""

    def __init__(self, input_channels: int, width: int, depth: int) -> None:
        super().__init__()
        feature_channels = width * 2
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels, width, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(width),
            nn.ReLU6(inplace=True),
            DepthwiseBlock(width, feature_channels, stride=2),
        )
        self.body = nn.Sequential(
            *[DepthwiseBlock(feature_channels, feature_channels) for _ in range(depth)]
        )
        channel_count = len(SEMANTICS)
        self.output = nn.Conv2d(feature_channels, channel_count * 5, kernel_size=1)
        nn.init.constant_(self.output.bias[:channel_count], -2.19)
        nn.init.constant_(self.output.bias[channel_count:], 0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.body(self.stem(inputs)))


def split_output(output: torch.Tensor) -> dict[str, torch.Tensor]:
    channels = len(SEMANTICS)
    heatmaps = output[:, :channels]
    sizes = output[:, channels : channels * 3]
    offsets = output[:, channels * 3 : channels * 5]
    batch, _, height, width = output.shape
    return {
        "heatmaps": heatmaps,
        "sizes": sizes.reshape(batch, channels, 2, height, width),
        "offsets": offsets.reshape(batch, channels, 2, height, width),
    }
