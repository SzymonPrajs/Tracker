from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MotionState:
    """The three persistent half-resolution planes used between frames."""

    previous_luminance: torch.Tensor
    positive: torch.Tensor
    negative: torch.Tensor


def half_resolution(luminance: torch.Tensor) -> torch.Tensor:
    """Generate the half-resolution luminance plane used by the motion path."""
    return F.avg_pool2d(luminance, kernel_size=2, stride=2)


def start_motion(luminance: torch.Tensor) -> MotionState:
    """Start a sequence without inventing motion on its first frame."""
    previous = half_resolution(luminance)
    empty = torch.zeros_like(previous)
    return MotionState(previous, empty, empty)


def _brightness_shift(delta: torch.Tensor, tile: int) -> torch.Tensor:
    """Estimate a robust global exposure shift from low-resolution tile means."""
    tile_means = F.avg_pool2d(
        delta,
        kernel_size=tile,
        stride=tile,
        ceil_mode=True,
        count_include_pad=False,
    )
    return tile_means.flatten(2).median(dim=2).values.unsqueeze(-1)


def update_motion(
    luminance: torch.Tensor,
    state: MotionState,
    *,
    decay: float,
    difference_clip: float,
    deadband: float,
    brightness_tile: int,
) -> tuple[torch.Tensor, MotionState, torch.Tensor]:
    """Update normalized positive/negative motion surfaces for one frame.

    Inputs and state are floating-point tensors in [0, 1]. The returned motion
    tensor has two channels, positive then negative. Firmware will implement the
    same operation with bounded integer arithmetic after the representation is
    selected by experiments.
    """
    current = half_resolution(luminance)
    delta = current - state.previous_luminance
    shift = _brightness_shift(delta, brightness_tile)
    delta = delta - shift

    positive_now = (delta - deadband).clamp(0, difference_clip) / difference_clip
    negative_now = (-delta - deadband).clamp(0, difference_clip) / difference_clip
    positive = torch.maximum(state.positive * decay, positive_now)
    negative = torch.maximum(state.negative * decay, negative_now)

    next_state = MotionState(current, positive, negative)
    return torch.cat((positive, negative), dim=1), next_state, shift
