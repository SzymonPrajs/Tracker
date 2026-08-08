"""Deterministic geometric multi-head scenes for pipeline smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
import random

import torch

from .targets import HeadTarget


@dataclass(frozen=True)
class SyntheticScene:
    image: torch.Tensor
    heads: tuple[HeadTarget, ...]
    scene_id: int


def make_synthetic_scene(
    scene_id: int,
    *,
    seed: int = 0,
    height: int = 160,
    width: int = 288,
    head_count: int = 3,
) -> SyntheticScene:
    if height < 32 or width < 32 or head_count < 0:
        raise ValueError("scene dimensions/head_count are invalid")
    rng = random.Random((seed << 32) ^ scene_id)
    y = torch.arange(height, dtype=torch.int16)[:, None]
    x = torch.arange(width, dtype=torch.int16)[None, :]
    image = torch.empty((3, height, width), dtype=torch.uint8)
    image[0] = ((x + 3 * y + scene_id) % 64 + 24).to(torch.uint8)
    image[1] = ((2 * x + y + seed) % 64 + 32).to(torch.uint8)
    image[2] = ((x + y + 2 * seed) % 48 + 40).to(torch.uint8)

    heads: list[HeadTarget] = []
    for index in range(head_count):
        head_width = rng.randint(max(12, width // 16), max(16, width // 7))
        head_height = rng.randint(max(16, height // 9), max(20, height // 4))
        cx = rng.randint(head_width // 2 + 1, width - head_width // 2 - 2)
        cy = rng.randint(head_height // 2 + 1, height - head_height // 2 - 2)
        xx = (x - cx).to(torch.float32) / (head_width / 2.0)
        yy = (y - cy).to(torch.float32) / (head_height / 2.0)
        mask = xx.square() + yy.square() <= 1.0
        colour = torch.tensor(
            [150 + (37 * index) % 80, 90 + (29 * index) % 100,
             70 + (19 * index) % 90],
            dtype=torch.uint8,
        )
        image[:, mask] = colour[:, None]
        heads.append(
            HeadTarget(
                center_x=float(cx), center_y=float(cy), width=float(head_width),
                height=float(head_height), visibility=1.0,
                track_id=f"scene-{scene_id}-head-{index}",
            )
        )
    return SyntheticScene(image=image, heads=tuple(heads), scene_id=scene_id)


class SyntheticSceneDataset(torch.utils.data.Dataset[SyntheticScene]):
    def __init__(
        self,
        length: int,
        *,
        seed: int = 0,
        head_count: int | None = 3,
        max_heads: int = 5,
    ) -> None:
        """Build fixed-count scenes, or varied 0..``max_heads`` scenes.

        Pass ``head_count=None`` for the varied mode. Its modulo schedule makes
        empty negative frames guaranteed in every complete ``max_heads + 1``
        scene cycle rather than merely probable.
        """
        if length < 0:
            raise ValueError("length must be non-negative")
        if head_count is not None and head_count < 0:
            raise ValueError("head_count must be non-negative or None")
        if max_heads < 0:
            raise ValueError("max_heads must be non-negative")
        self.length = length
        self.seed = seed
        self.head_count = head_count
        self.max_heads = max_heads

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> SyntheticScene:
        if not 0 <= index < self.length:
            raise IndexError(index)
        count = self.head_count
        if count is None:
            count = (index + self.seed) % (self.max_heads + 1)
        return make_synthetic_scene(index, seed=self.seed, head_count=count)
