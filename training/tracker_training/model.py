"""Exact PyTorch implementation of the fixed-shape HC-DS31 network."""

from __future__ import annotations

from collections.abc import Iterator
import copy

import torch
from torch import nn
from torch.nn import functional as F


OUTPUT_ENCODING_ID = "hcds31-output-q4-q7-v1"
OUTPUT_EXPECTED_EXPONENT = -3
OUTPUT_ENCODING_GAINS = (
    2.0,
    16.0,
    16.0,
    16.0,
    16.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)
OUTPUT_ENCODED_MINIMUM = -16.0
OUTPUT_ENCODED_SATURATION_LIMIT = 15.5


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        groups: int = 1,
        activate: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = activate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        return F.relu(x, inplace=False) if self.activate else x


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.dw = ConvBNAct(
            in_channels, in_channels, 3, stride=2, groups=in_channels
        )
        self.pw = ConvBNAct(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = ConvBNAct(channels, channels, 3, groups=channels)
        self.pw = ConvBNAct(channels, channels, 1, activate=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.pw(self.dw(x)), inplace=False)


class RefineBlock(nn.Module):
    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.dw = ConvBNAct(channels, channels, 3, groups=channels)
        self.pw = ConvBNAct(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class HCDS31(nn.Module):
    """HC-DS31: 36 convolutions, 31,256,640 MAC, 66,224 weights."""

    input_shape = (3, 160, 288)
    output_shape = (16, 40, 72)

    def __init__(self) -> None:
        super().__init__()
        self.encoded_output_clamp = False
        self.stem = ConvBNAct(3, 16, 3, stride=2)

        self.s1_down = DownBlock(16, 32)
        self.s1_refine = nn.ModuleList([ResidualBlock(32)])
        self.s2_down = DownBlock(32, 48)
        self.s2_refine = nn.ModuleList([ResidualBlock(48) for _ in range(2)])
        self.s3_down = DownBlock(48, 64)
        self.s3_refine = nn.ModuleList([ResidualBlock(64) for _ in range(3)])
        self.s4_down = DownBlock(64, 96)
        self.s4_refine = nn.ModuleList([ResidualBlock(96) for _ in range(2)])

        self.lat3 = ConvBNAct(64, 32, 1, activate=False)
        self.deep4 = ConvBNAct(96, 32, 1, activate=False)
        self.p3_refine = RefineBlock()
        self.lat2 = ConvBNAct(48, 32, 1, activate=False)
        self.p2_refine = RefineBlock()
        self.lat1 = ConvBNAct(32, 32, 1, activate=False)
        self.p1_refine = RefineBlock()
        self.head = nn.Conv2d(32, 16, 1, bias=True)

    @staticmethod
    def _run_blocks(x: torch.Tensor, blocks: nn.ModuleList) -> torch.Tensor:
        for block in blocks:
            x = block(x)
        return x

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        stem = self.stem(x)
        s1 = self._run_blocks(self.s1_down(stem), self.s1_refine)
        s2 = self._run_blocks(self.s2_down(s1), self.s2_refine)
        s3 = self._run_blocks(self.s3_down(s2), self.s3_refine)
        s4 = self._run_blocks(self.s4_down(s3), self.s4_refine)

        p3 = self.p3_refine(
            self.lat3(s3)
            + F.interpolate(self.deep4(s4), scale_factor=2, mode="nearest")
        )
        p2 = self.p2_refine(
            self.lat2(s2) + F.interpolate(p3, scale_factor=2, mode="nearest")
        )
        p1 = self.p1_refine(
            self.lat1(s1) + F.interpolate(p2, scale_factor=2, mode="nearest")
        )
        return {"stem": stem, "s1": s1, "s2": s2, "s3": s3, "s4": s4,
                "p3": p3, "p2": p2, "p1": p1}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.head(self.forward_features(x)["p1"])
        if self.encoded_output_clamp:
            output = torch.clamp(
                output,
                min=OUTPUT_ENCODED_MINIMUM,
                max=OUTPUT_ENCODED_SATURATION_LIMIT,
            )
        return output

    def iter_deploy_layers(
        self,
    ) -> Iterator[tuple[str, nn.Conv2d, nn.BatchNorm2d | None]]:
        """Yield convolutions in forward order for MLX/export parity tooling."""
        for name, module in self.named_modules():
            if isinstance(module, ConvBNAct):
                yield name, module.conv, module.bn
        yield "head", self.head, None

    def convolution_weight_count(self) -> int:
        return sum(conv.weight.numel() for _, conv, _ in self.iter_deploy_layers())


def make_encoded_export_model(model: HCDS31) -> HCDS31:
    """Clone ``model`` and fold the Q4/Q7 physical encoding into its head.

    The source model remains a semantic training model. With output exponent
    -3, the returned model produces Q4 heatmap-logit bytes and Q7 offset/size
    bytes. Padding head rows are made exactly zero by their zero gains.
    """
    if not isinstance(model, HCDS31):
        raise TypeError("model must be an HCDS31 instance")
    if model.encoded_output_clamp:
        raise ValueError("model output is already encoded")
    encoded = copy.deepcopy(model)
    gains = encoded.head.weight.new_tensor(OUTPUT_ENCODING_GAINS)
    with torch.no_grad():
        encoded.head.weight.mul_(gains[:, None, None, None])
        if encoded.head.bias is not None:
            encoded.head.bias.mul_(gains)
    encoded.encoded_output_clamp = True
    return encoded
