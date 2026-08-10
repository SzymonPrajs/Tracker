#!/usr/bin/env python3
"""Render representative camera augmentations with their transformed head boxes."""

from __future__ import annotations

import argparse
from copy import copy
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from tracker_training.augmentation import augment_batch
from tracker_training.data import PackedHeadDataset
from train import parser as training_parser


WIDTH, HEIGHT = 288, 160


def clean_arguments():
    args = training_parser().parse_args([])
    args.augmentation_clean_probability = 0
    args.horizontal_flip = 0
    for name in (
        "exposure_probability", "low_light_probability", "white_balance_probability",
        "gamma_probability", "saturation_probability", "illumination_gradient_probability",
        "shadow_probability", "vignette_probability", "noise_probability", "blur_probability",
        "fisheye_130_probability", "fisheye_180_probability", "telephoto_probability",
    ):
        setattr(args, name, 0.0)
    args.brightness = args.contrast = args.channel_jitter = 0.0
    return args


def boxes(target):
    _, offset, size, mask = target
    result = []
    for iy, ix in mask[0, 0].nonzero().tolist():
        cx = (ix + 0.5 + float(offset[0, 0, iy, ix])) * 4
        cy = (iy + 0.5 + float(offset[0, 1, iy, ix])) * 4
        width = float(size[0, 0, iy, ix]) * WIDTH
        height = float(size[0, 1, iy, ix]) * HEIGHT
        result.append((cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2))
    return result


def render(image, target, title):
    rgb = ((image[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).byte().numpy()
    panel = Image.fromarray(rgb)
    draw = ImageDraw.Draw(panel)
    for box in boxes(target):
        draw.rectangle(box, outline=(255, 48, 32), width=2)
    draw.rectangle((0, 0, WIDTH, 18), fill=(0, 0, 0))
    draw.text((5, 3), title, fill=(255, 255, 255))
    return panel


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, default=Path("data/packed/288x160"))
    cli.add_argument("--index", type=int, default=0)
    cli.add_argument("--output", type=Path, default=Path("training/artifacts/augmentation_preview.png"))
    options = cli.parse_args()
    image, target = PackedHeadDataset(options.data_root, "train")[options.index]
    image = image[None].float().sub(128).div(128)
    target = tuple(value[None] for value in target)
    base = clean_arguments()
    variants = [("Original", base)]

    darkness = copy(base)
    darkness.exposure_probability = darkness.low_light_probability = 1.0
    darkness.low_light_min_ev = darkness.low_light_max_ev = -3.2
    darkness.noise_probability = 1.0
    variants.append(("Low light: -3.2 EV + sensor noise", darkness))

    uneven = copy(base)
    uneven.exposure_probability = uneven.white_balance_probability = 1.0
    uneven.exposure_min_ev = uneven.exposure_max_ev = -1.0
    uneven.white_balance_magnitude = 0.30
    uneven.shadow_probability = uneven.illumination_gradient_probability = 1.0
    variants.append(("Mixed light, colour cast, local shadow", uneven))

    fish130 = copy(base)
    fish130.fisheye_130_probability = 1.0
    fish130.fisheye_130_min = fish130.fisheye_130_max = 0.20
    variants.append(("130 degree class fisheye", fish130))

    fish180 = copy(base)
    fish180.fisheye_180_probability = 1.0
    fish180.fisheye_180_min = fish180.fisheye_180_max = 0.42
    fish180.vignette_probability = 1.0
    variants.append(("180 degree class fisheye", fish180))

    tele2 = copy(base)
    tele2.telephoto_probability = 1.0
    tele2.telephoto_zoom_min = tele2.telephoto_zoom_max = 2.0
    variants.append(("Telephoto crop: 2x", tele2))

    tele6 = copy(base)
    tele6.telephoto_probability = 1.0
    tele6.telephoto_zoom_min = tele6.telephoto_zoom_max = 6.0
    variants.append(("10 degree class telephoto: 6x", tele6))

    full = training_parser().parse_args([])
    variants.append(("Full stochastic camera mixture", full))

    panels = []
    for index, (title, args) in enumerate(variants):
        torch.manual_seed(1000 + index)
        augmented_image, augmented_target = augment_batch(
            image.clone(), tuple(value.clone() for value in target), args
        )
        panels.append(render(augmented_image, augmented_target, title))
    sheet = Image.new("RGB", (WIDTH * 2, HEIGHT * 4))
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % 2) * WIDTH, (index // 2) * HEIGHT))
    options.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(options.output)
    print(options.output)


if __name__ == "__main__":
    main()
