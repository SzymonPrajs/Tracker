from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from common.model import DepthwiseBlock


@dataclass(frozen=True)
class TemporalState:
    """The two recurrent stride-eight feature maps carried between frames."""

    fast: torch.Tensor
    slow: torch.Tensor


class MotionStem(nn.Module):
    """Fuse full-resolution luminance and half-resolution P/N surfaces."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.appearance = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
        )
        self.motion = nn.Sequential(
            nn.Conv2d(2, width, kernel_size=1, bias=False),
            nn.BatchNorm2d(width),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, luminance: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        return self.activation(self.appearance(luminance) + self.motion(motion))


class TwoPoleAdapter(nn.Module):
    """A diagonal fast/slow state on a small prefix of the feature channels."""

    def __init__(
        self,
        channels: int,
        fast_pole: float,
        slow_pole: float,
        learn_poles: bool,
    ) -> None:
        super().__init__()
        self.channels = channels
        fast_logit = math.log(fast_pole / (1 - fast_pole))
        slow_logit = math.log(slow_pole / (1 - slow_pole))
        self.fast_logit = nn.Parameter(
            torch.tensor(fast_logit), requires_grad=learn_poles
        )
        self.slow_logit = nn.Parameter(
            torch.tensor(slow_logit), requires_grad=learn_poles
        )
        self.fast_mix = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.slow_mix = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.register_buffer(
            "frozen_fast_pole", torch.tensor(fast_pole), persistent=False
        )
        self.register_buffer(
            "frozen_slow_pole", torch.tensor(slow_pole), persistent=False
        )
        self.poles_are_frozen = False

    @property
    def poles(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.poles_are_frozen:
            return self.frozen_fast_pole, self.frozen_slow_pole
        return self.fast_logit.sigmoid(), self.slow_logit.sigmoid()

    def freeze_poles_for_export(self) -> None:
        """Materialize learned poles so export does not run Sigmoid every frame."""
        with torch.no_grad():
            fast, slow = self.fast_logit.sigmoid(), self.slow_logit.sigmoid()
            self.frozen_fast_pole.copy_(fast)
            self.frozen_slow_pole.copy_(slow)
        self.poles_are_frozen = True

    def forward(
        self,
        features: torch.Tensor,
        state: TemporalState,
        elapsed_frames: int = 1,
    ) -> tuple[torch.Tensor, TemporalState]:
        current = features[:, : self.channels]
        fast_pole, slow_pole = self.poles
        if elapsed_frames != 1:
            fast_pole = fast_pole.pow(elapsed_frames)
            slow_pole = slow_pole.pow(elapsed_frames)

        mixed = (
            current
            + self.fast_mix * (current - state.fast)
            + self.slow_mix * (state.fast - state.slow)
        )
        new_fast = fast_pole * state.fast + (1 - fast_pole) * current
        new_slow = slow_pole * state.slow + (1 - slow_pole) * current
        output = torch.cat((mixed, features[:, self.channels :]), dim=1)
        return output, TemporalState(new_fast, new_slow)


class TemporalTrackerModel(nn.Module):
    """Tiny motion-first CenterTrack-style point detector.

    The caller owns frame-derived motion state, the gated previous-owner
    heatmap, and sequence resets. The model owns only the two recurrent feature
    maps returned by each step.
    """

    output_names = ("heatmap", "offset_x", "offset_y", "move_x", "move_y")

    def __init__(
        self,
        *,
        stem_width: int,
        first_width: int,
        stride4_width: int,
        stride4_blocks: int,
        stride8_width: int,
        stride8_blocks: int,
        decoder_width: int,
        decoder_blocks: int,
        recurrent_channels: int,
        fast_pole: float,
        slow_pole: float,
        learn_poles: bool,
    ) -> None:
        super().__init__()
        self.recurrent_channels = recurrent_channels
        self.stem = MotionStem(stem_width)
        self.localize = nn.Sequential(
            DepthwiseBlock(stem_width, first_width, stride=2),
            DepthwiseBlock(first_width, stride4_width),
            *[
                DepthwiseBlock(stride4_width, stride4_width)
                for _ in range(stride4_blocks)
            ],
        )
        self.prior_scale = nn.Parameter(torch.zeros(1, stride4_width, 1, 1))
        self.downsample = DepthwiseBlock(stride4_width, stride8_width, stride=2)
        self.temporal = TwoPoleAdapter(
            recurrent_channels, fast_pole, slow_pole, learn_poles
        )
        self.trunk = nn.Sequential(
            *[
                DepthwiseBlock(stride8_width, stride8_width)
                for _ in range(stride8_blocks)
            ]
        )
        self.top = nn.Sequential(
            nn.Conv2d(stride8_width, decoder_width, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_width),
        )
        self.lateral = nn.Sequential(
            nn.Conv2d(stride4_width, decoder_width, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_width),
        )
        self.decode_activation = nn.ReLU(inplace=True)
        self.decode = nn.Sequential(
            *[
                DepthwiseBlock(decoder_width, decoder_width)
                for _ in range(decoder_blocks)
            ]
        )
        self.output = nn.Conv2d(decoder_width, 5, kernel_size=1)
        nn.init.constant_(self.output.bias[:1], -2.19)
        nn.init.constant_(self.output.bias[1:], 0)

    def start_state(
        self,
        batch: int,
        height: int,
        width: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> TemporalState:
        shape = (
            batch,
            self.recurrent_channels,
            math.ceil(height / 8),
            math.ceil(width / 8),
        )
        empty = torch.zeros(shape, device=device, dtype=dtype)
        return TemporalState(empty, empty.clone())

    def forward(
        self,
        luminance: torch.Tensor,
        motion: torch.Tensor,
        previous_heatmap: torch.Tensor,
        fast_state: torch.Tensor,
        slow_state: torch.Tensor,
        elapsed_frames: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stride4 = self.localize(self.stem(luminance, motion))
        stride4 = stride4 + self.prior_scale * previous_heatmap
        stride8 = self.downsample(stride4)
        stride8, next_state = self.temporal(
            stride8, TemporalState(fast_state, slow_state), elapsed_frames
        )
        stride8 = self.trunk(stride8)

        top = F.interpolate(self.top(stride8), size=stride4.shape[-2:], mode="nearest")
        decoded = self.decode(self.decode_activation(top + self.lateral(stride4)))
        return self.output(decoded), next_state.fast, next_state.slow

    def prepare_for_export(self) -> None:
        """Freeze values that should be compile-time constants on the device."""
        self.eval()
        self.temporal.freeze_poles_for_export()


def model_from_config(config: dict[str, Any]) -> TemporalTrackerModel:
    model = config["model"]
    temporal = config["temporal"]
    return TemporalTrackerModel(
        **model,
        fast_pole=1 - 2 ** -temporal["fast_shift"],
        slow_pole=1 - 2 ** -temporal["slow_shift"],
        learn_poles=temporal["learn_poles"],
    )


def split_temporal_output(output: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "heatmap": output[:, :1],
        "offset": output[:, 1:3],
        "displacement": output[:, 3:5],
    }
