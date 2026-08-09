#!/usr/bin/env python3
"""Train HC-DS31 on small generated multi-head scenes."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from tracker_training.model import HCDS31


HEIGHT, WIDTH, STRIDE = 160, 288, 4


def scene(seed: int):
    rng = random.Random(seed)
    y, x = torch.meshgrid(torch.arange(HEIGHT), torch.arange(WIDTH), indexing="ij")
    image = torch.stack(((x + 3 * y) % 64 + 24, (2 * x + y) % 64 + 32, (x + y) % 48 + 40)).byte()
    heads = []
    for index in range(seed % 6):
        w, h = rng.randint(18, 40), rng.randint(18, 40)
        cx, cy = rng.randint(w, WIDTH - w), rng.randint(h, HEIGHT - h)
        mask = ((x - cx) / (w / 2)).square() + ((y - cy) / (h / 2)).square() <= 1
        image[:, mask] = torch.tensor(
            [180, 100 + index * 10, 80], dtype=torch.uint8
        )[:, None]
        heads.append((cx, cy, w, h))
    return (image.float() - 128) / 128, heads


def targets(batch_heads):
    n, oh, ow = len(batch_heads), HEIGHT // STRIDE, WIDTH // STRIDE
    heat = torch.zeros(n, 1, oh, ow)
    offset, size = torch.zeros(n, 2, oh, ow), torch.zeros(n, 2, oh, ow)
    mask = torch.zeros(n, 1, oh, ow, dtype=torch.bool)
    yy, xx = torch.meshgrid(torch.arange(oh), torch.arange(ow), indexing="ij")
    for b, heads in enumerate(batch_heads):
        for cx, cy, w, h in heads:
            u, v = cx / STRIDE, cy / STRIDE
            ix, iy = int(u), int(v)
            sigma = min(3.0, max(1.0, 0.15 * min(w, h) / STRIDE))
            gaussian = torch.exp(-((xx - ix).square() + (yy - iy).square()) / (2 * sigma * sigma))
            heat[b, 0] = torch.maximum(heat[b, 0], gaussian)
            if not mask[b, 0, iy, ix]:
                offset[b, :, iy, ix] = torch.tensor([u - ix - 0.5, v - iy - 0.5])
                size[b, :, iy, ix] = torch.tensor([w / WIDTH, h / HEIGHT])
                mask[b, 0, iy, ix] = True
    return heat, offset, size, mask


def loss(prediction, target):
    heat, offset, size, mask = target
    probability = prediction[:, :1].sigmoid().clamp(1e-6, 1 - 1e-6)
    positive = heat.eq(1)
    negative = ~positive
    focal = -(torch.log(probability) * (1 - probability).square() * positive).sum()
    focal -= (torch.log(1 - probability) * probability.square() * (1 - heat).pow(4) * negative).sum()
    focal /= max(1, int(positive.sum()))
    selected = mask.expand(-1, 2, -1, -1)

    def regression(value, truth):
        if not selected.any():
            return value.sum() * 0
        return F.smooth_l1_loss(value[selected], truth[selected], beta=1 / 9)

    return (
        focal
        + regression(prediction[:, 1:3], offset)
        + 0.15 * regression(prediction[:, 3:5], size)
        + 0.1 * prediction[:, 5:].square().mean()
    )


def batch(seeds):
    items = [scene(seed) for seed in seeds]
    return torch.stack([item[0] for item in items]), targets([item[1] for item in items])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("training/artifacts"))
    args = parser.parse_args()

    torch.manual_seed(7)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = HCDS31().to(device)
    model.head.bias.data[0] = -2.19
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    seeds = list(range(24))
    for epoch in range(args.epochs):
        random.Random(epoch).shuffle(seeds)
        total = 0.0
        model.train()
        for start in range(0, len(seeds), 4):
            images, target = batch(seeds[start : start + 4])
            images, target = images.to(device), tuple(value.to(device) for value in target)
            optimizer.zero_grad()
            value = loss(model(images), target)
            value.backward()
            optimizer.step()
            total += value.item()
        print(f"epoch {epoch + 1:02d} loss {total / 6:.4f}")

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output / "model.pt")
    calibration = torch.stack([scene(seed)[0] for seed in range(8)]).numpy()
    np.save(args.output / "calibration.npy", calibration)


if __name__ == "__main__":
    main()
