from __future__ import annotations

import torch
from torch import nn


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
            nn.ReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = self.pointwise(self.depthwise(inputs))
        if self.residual:
            result = result + inputs
        return self.activation(result)


class TrackerModel(nn.Module):
    """Two-scale detector with a stride-four localization output."""

    def __init__(
        self,
        input_channels: int,
        stem_width: int,
        stride4_width: int,
        stride4_blocks: int,
        stride8_width: int,
        stride8_blocks: int,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                stem_width,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(inplace=True),
            DepthwiseBlock(stem_width, stride4_width, stride=2),
        )
        self.stride4_body = nn.Sequential(
            *[
                DepthwiseBlock(stride4_width, stride4_width)
                for _ in range(stride4_blocks)
            ]
        )
        self.downsample = DepthwiseBlock(stride4_width, stride8_width, stride=2)
        self.stride8_body = nn.Sequential(
            *[
                DepthwiseBlock(stride8_width, stride8_width)
                for _ in range(stride8_blocks)
            ]
        )
        self.project = nn.Sequential(
            nn.Conv2d(stride8_width, stride4_width, kernel_size=1, bias=False),
            nn.BatchNorm2d(stride4_width),
        )
        self.merge_activation = nn.ReLU(inplace=True)
        self.output = nn.Conv2d(stride4_width, 3, kernel_size=1)
        nn.init.constant_(self.output.bias[:1], -2.19)
        nn.init.constant_(self.output.bias[1:], 0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        stride4 = self.stride4_body(self.stem(inputs))
        stride8 = self.stride8_body(self.downsample(stride4))
        upsampled = torch.nn.functional.interpolate(
            self.project(stride8), size=stride4.shape[-2:], mode="nearest"
        )
        return self.output(self.merge_activation(stride4 + upsampled))


def split_output(output: torch.Tensor) -> dict[str, torch.Tensor]:
    heatmaps = output[:, :1]
    offsets = output[:, 1:3]
    batch, _, height, width = output.shape
    return {
        "heatmaps": heatmaps,
        "offsets": offsets.reshape(batch, 1, 2, height, width),
    }
