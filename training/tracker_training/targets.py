"""CenterNet-style dense targets matched to the portable C decoder convention."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import math

import torch


@dataclass(frozen=True)
class HeadTarget:
    center_x: float
    center_y: float
    width: float
    height: float
    visibility: float = 1.0
    ignore: bool = False
    track_id: str | None = None


@dataclass(frozen=True)
class TargetConfig:
    input_height: int = 160
    input_width: int = 288
    output_stride: int = 4
    sigma_scale: float = 0.15
    sigma_min: float = 1.0
    sigma_max: float = 3.0
    ignore_sigma: float = 2.0

    @property
    def output_height(self) -> int:
        return self.input_height // self.output_stride

    @property
    def output_width(self) -> int:
        return self.input_width // self.output_stride


def _draw_gaussian(
    destination: torch.Tensor,
    centre_x: int,
    centre_y: int,
    sigma: float,
) -> None:
    radius = max(1, math.ceil(3.0 * sigma))
    height, width = destination.shape
    x0, x1 = max(0, centre_x - radius), min(width - 1, centre_x + radius)
    y0, y1 = max(0, centre_y - radius), min(height - 1, centre_y + radius)
    xs = torch.arange(x0, x1 + 1, dtype=torch.float32)
    ys = torch.arange(y0, y1 + 1, dtype=torch.float32)
    gaussian = torch.exp(
        -((xs[None, :] - centre_x) ** 2 + (ys[:, None] - centre_y) ** 2)
        / (2.0 * sigma * sigma)
    )
    patch = destination[y0 : y1 + 1, x0 : x1 + 1]
    destination[y0 : y1 + 1, x0 : x1 + 1] = torch.maximum(patch, gaussian)


def encode_targets(
    heads: Sequence[HeadTarget], config: TargetConfig = TargetConfig()
) -> dict[str, torch.Tensor]:
    """Encode all valid heads, resolving same-cell regression collisions deterministically.

    The Gaussian is centred on the integer anchor. Offsets are relative to the
    anchor cell centre because the C decoder adds both a heatmap moment and the
    learned offset.
    """
    oh, ow = config.output_height, config.output_width
    heatmap = torch.zeros((1, oh, ow), dtype=torch.float32)
    offset = torch.zeros((2, oh, ow), dtype=torch.float32)
    size = torch.zeros((2, oh, ow), dtype=torch.float32)
    reg_mask = torch.zeros((1, oh, ow), dtype=torch.bool)
    ignore_mask = torch.zeros((1, oh, ow), dtype=torch.bool)
    collision_mask = torch.zeros((1, oh, ow), dtype=torch.bool)
    owner_index = torch.full((oh, ow), -1, dtype=torch.int64)

    candidates: list[tuple[int, HeadTarget, int, int, float, float]] = []
    ignored: list[tuple[int, int]] = []
    for index, head in enumerate(heads):
        if head.width <= 0.0 or head.height <= 0.0:
            continue
        u = head.center_x / config.output_stride
        v = head.center_y / config.output_stride
        ix, iy = math.floor(u), math.floor(v)
        if not (0 <= ix < ow and 0 <= iy < oh):
            continue
        if head.ignore:
            ignored.append((ix, iy))
            continue
        sigma = min(
            config.sigma_max,
            max(
                config.sigma_min,
                config.sigma_scale
                * min(head.width, head.height)
                / config.output_stride,
            ),
        )
        _draw_gaussian(heatmap[0], ix, iy, sigma)
        candidates.append((index, head, ix, iy, u, v))

    for ix, iy in ignored:
        scratch = torch.zeros((oh, ow), dtype=torch.float32)
        _draw_gaussian(scratch, ix, iy, config.ignore_sigma)
        ignore_mask[0] |= scratch > math.exp(-4.5)

    # Highest visibility, then largest box, then manifest order owns a cell.
    candidates.sort(key=lambda item: (-item[1].visibility,
                                      -(item[1].width * item[1].height), item[0]))
    collision_count = 0
    for index, head, ix, iy, u, v in candidates:
        if reg_mask[0, iy, ix]:
            collision_mask[0, iy, ix] = True
            collision_count += 1
            continue
        reg_mask[0, iy, ix] = True
        ignore_mask[0, iy, ix] = False
        owner_index[iy, ix] = index
        offset[0, iy, ix] = u - (ix + 0.5)
        offset[1, iy, ix] = v - (iy + 0.5)
        size[0, iy, ix] = head.width / config.input_width
        size[1, iy, ix] = head.height / config.input_height

    return {
        "heatmap": heatmap,
        "offset": offset,
        "size": size,
        "reg_mask": reg_mask,
        "ignore_mask": ignore_mask,
        "collision_mask": collision_mask,
        "owner_index": owner_index,
        "collision_count": torch.tensor(collision_count, dtype=torch.int64),
    }
