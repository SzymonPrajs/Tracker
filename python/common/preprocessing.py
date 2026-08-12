from __future__ import annotations

import json
import math
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from common.images import clip_labels


SEMANTICS = ("head", "face", "person", "person_visible", "person_full")
CHANNEL = {name: index for index, name in enumerate(SEMANTICS)}
SOURCE_COVERAGE = {
    "coco": ("person",),
    "crowdhuman": ("head", "person_visible", "person_full"),
    "scut_head": ("head",),
    "wider_face": ("face",),
}


def iter_records(data_dir: Path) -> Iterator[dict[str, Any]]:
    for labels_file in sorted(data_dir.glob("*/labels.jsonl")):
        with labels_file.open(encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                record["image_path"] = labels_file.parent / record["image"]
                yield record


def balanced_sample(data_dir: Path, count: int, seed: int) -> list[dict[str, Any]]:
    """Reservoir-sample each source so a preview is not dominated by one set."""
    files = sorted(data_dir.glob("*/labels.jsonl"))
    if not files:
        raise FileNotFoundError(f"No labels.jsonl files found under {data_dir}")

    rng = random.Random(seed)
    per_source = max(1, math.ceil(count / len(files)))
    chosen: list[dict[str, Any]] = []
    for labels_file in files:
        reservoir: list[dict[str, Any]] = []
        with labels_file.open(encoding="utf-8") as file:
            for seen, line in enumerate(file, start=1):
                record = json.loads(line)
                record["image_path"] = labels_file.parent / record["image"]
                if len(reservoir) < per_source:
                    reservoir.append(record)
                else:
                    replace = rng.randrange(seen)
                    if replace < per_source:
                        reservoir[replace] = record
        chosen.extend(reservoir)
    rng.shuffle(chosen)
    return chosen[:count]


def _image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        storage = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        return (
            storage.reshape(image.height, image.width, 3).permute(2, 0, 1).float() / 255
        )


def _forward_square_warp(
    x: torch.Tensor, y: torch.Tensor, strength: float
) -> tuple[torch.Tensor, torch.Tensor]:
    radius = torch.maximum(x.abs(), y.abs())
    scale = (1 + strength) / (1 + strength * radius.square())
    return x * scale, y * scale


def _warp_labels(
    labels: list[dict[str, Any]], width: int, height: int, strength: float
) -> list[dict[str, Any]]:
    warped = []
    steps = torch.linspace(0, 1, 17)
    for label in labels:
        x, y, box_width, box_height = label["box"]
        x2, y2 = x + box_width, y + box_height
        top_x, bottom_x = x + steps * box_width, x + steps * box_width
        left_y, right_y = y + steps * box_height, y + steps * box_height
        points_x = torch.cat(
            (top_x, bottom_x, torch.full_like(steps, x), torch.full_like(steps, x2))
        )
        points_y = torch.cat(
            (torch.full_like(steps, y), torch.full_like(steps, y2), left_y, right_y)
        )
        normal_x = points_x * (2 / width) - 1
        normal_y = points_y * (2 / height) - 1
        normal_x, normal_y = _forward_square_warp(normal_x, normal_y, strength)
        normal_x = normal_x.clamp(-1, 1)
        normal_y = normal_y.clamp(-1, 1)
        points_x = (normal_x + 1) * width / 2
        points_y = (normal_y + 1) * height / 2
        result = dict(label)
        result["box"] = [
            float(points_x.min()),
            float(points_y.min()),
            float(points_x.max() - points_x.min()),
            float(points_y.max() - points_y.min()),
        ]
        warped.append(result)
    return clip_labels(warped, width, height)


def full_canvas_warp(
    image: torch.Tensor, labels: list[dict[str, Any]], strength: float
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Fisheye-like bijection whose entire rectangular boundary stays fixed."""
    if strength == 0:
        return image, labels

    _, height, width = image.shape
    destination_y = torch.linspace(-1, 1, height)
    destination_x = torch.linspace(-1, 1, width)
    grid_y, grid_x = torch.meshgrid(destination_y, destination_x, indexing="ij")
    destination_radius = torch.maximum(grid_x.abs(), grid_y.abs())

    source_radius = destination_radius / (1 + strength)
    for _ in range(6):
        square = source_radius.square()
        denominator = 1 + strength * square
        value = source_radius * (1 + strength) / denominator - destination_radius
        derivative = (1 + strength) * (1 - strength * square) / denominator.square()
        source_radius = (source_radius - value / derivative).clamp(0, 1)
    source_radius = torch.where(
        destination_radius >= 1 - 1e-6, destination_radius, source_radius
    )

    inverse_scale = torch.where(
        destination_radius > 0,
        source_radius / destination_radius,
        torch.ones_like(destination_radius),
    )
    grid = torch.stack((grid_x * inverse_scale, grid_y * inverse_scale), dim=-1)
    warped = F.grid_sample(
        image.unsqueeze(0),
        grid.unsqueeze(0),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).squeeze(0)
    return warped, _warp_labels(labels, width, height, strength)


def _letterbox(
    image: torch.Tensor,
    labels: list[dict[str, Any]],
    target_width: int,
    target_height: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    _, height, width = image.shape
    scale = min(target_width / width, target_height / height, 1.0)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    if (resized_width, resized_height) != (width, height):
        image = F.interpolate(
            image.unsqueeze(0),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)

    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    right = target_width - resized_width - left
    bottom = target_height - resized_height - top
    image = F.pad(image, (left, right, top, bottom), value=0)

    transformed = []
    for label in labels:
        x, y, width, height = label["box"]
        result = dict(label)
        result["box"] = [
            x * scale + left,
            y * scale + top,
            width * scale,
            height * scale,
        ]
        transformed.append(result)
    return image, clip_labels(transformed, target_width, target_height)


def _photometric(
    image: torch.Tensor, config: dict[str, Any], rng: random.Random, seed: int
) -> torch.Tensor:
    if rng.random() < config["dark_probability"]:
        exposure = rng.uniform(config["exposure_min"], config["exposure_max"])
        gamma = rng.uniform(config["gamma_min"], config["gamma_max"])
        image = (image * exposure).clamp(0, 1).pow(gamma)

    if rng.random() < config["noise_probability"]:
        generator = torch.Generator().manual_seed(seed)
        read = rng.uniform(0, config["read_noise_max"])
        shot = rng.uniform(0, config["shot_noise_max"])
        standard_deviation = read + shot * image.sqrt()
        noise = torch.randn(image.shape, generator=generator) * standard_deviation
        image = (image + noise).clamp(0, 1)

    if rng.random() < config["blur_probability"]:
        padded = F.pad(image.unsqueeze(0), (1, 1, 1, 1), mode="replicate")
        image = F.avg_pool2d(padded, kernel_size=3, stride=1).squeeze(0)
    return image


def _draw_gaussian(
    heatmap: torch.Tensor, center_x: float, center_y: float, sigma: float
) -> None:
    radius = max(1, math.ceil(3 * sigma))
    left = max(0, math.floor(center_x) - radius)
    right = min(heatmap.shape[1], math.floor(center_x) + radius + 1)
    top = max(0, math.floor(center_y) - radius)
    bottom = min(heatmap.shape[0], math.floor(center_y) + radius + 1)
    if left >= right or top >= bottom:
        return
    y = torch.arange(top, bottom, dtype=torch.float32).unsqueeze(1)
    x = torch.arange(left, right, dtype=torch.float32).unsqueeze(0)
    gaussian = torch.exp(
        -((x - center_x).square() + (y - center_y).square()) / (2 * sigma**2)
    )
    heatmap[top:bottom, left:right] = torch.maximum(
        heatmap[top:bottom, left:right], gaussian
    )


def build_targets(
    record: dict[str, Any],
    labels: list[dict[str, Any]],
    input_width: int,
    input_height: int,
    stride: int,
) -> dict[str, torch.Tensor]:
    output_width = input_width // stride
    output_height = input_height // stride
    channels = len(SEMANTICS)
    heatmaps = torch.zeros(channels, output_height, output_width)
    valid_mask = torch.zeros_like(heatmaps)
    sizes = torch.zeros(channels, 2, output_height, output_width)
    offsets = torch.zeros_like(sizes)
    regression_mask = torch.zeros_like(heatmaps)

    for kind in SOURCE_COVERAGE[record["source"]]:
        valid_mask[CHANNEL[kind]].fill_(1)

    for label in labels:
        kind = label["kind"]
        if kind not in CHANNEL:
            continue
        channel = CHANNEL[kind]
        x, y, width, height = label["box"]
        if label.get("ignore", False):
            left = max(0, math.floor(x / stride))
            right = min(output_width, math.ceil((x + width) / stride))
            top = max(0, math.floor(y / stride))
            bottom = min(output_height, math.ceil((y + height) / stride))
            valid_mask[channel, top:bottom, left:right] = 0
            continue

        center_x = (x + width / 2) / stride
        center_y = (y + height / 2) / stride
        cell_x = min(output_width - 1, max(0, math.floor(center_x)))
        cell_y = min(output_height - 1, max(0, math.floor(center_y)))
        sigma = max(0.75, min(width, height) / stride / 6)
        _draw_gaussian(heatmaps[channel], center_x, center_y, sigma)
        heatmaps[channel, cell_y, cell_x] = 1
        sizes[channel, :, cell_y, cell_x] = torch.tensor(
            (width / stride, height / stride)
        )
        offsets[channel, :, cell_y, cell_x] = torch.tensor(
            (center_x - cell_x, center_y - cell_y)
        )
        regression_mask[channel, cell_y, cell_x] = 1

    regression_mask *= valid_mask

    return {
        "heatmaps": heatmaps,
        "valid_mask": valid_mask,
        "sizes": sizes,
        "offsets": offsets,
        "regression_mask": regression_mask,
    }


def prepare_example(
    record: dict[str, Any],
    config: dict[str, Any],
    seed: int,
    augment: bool = True,
) -> dict[str, Any]:
    rng = random.Random(seed)
    image = _image_tensor(record["image_path"])
    labels = [dict(label) for label in record["labels"]]
    clean = not augment or rng.random() < config["clean_probability"]
    warp_strength = 0.0

    if not clean:
        if rng.random() < config["warp_probability"]:
            warp_strength = rng.uniform(
                config["warp_strength_min"], config["warp_strength_max"]
            )
            image, labels = full_canvas_warp(image, labels, warp_strength)
        image = _photometric(image, config, rng, seed)

    image, labels = _letterbox(
        image, labels, config["input_width"], config["input_height"]
    )
    if config["color"] == "luminance":
        weights = image.new_tensor((0.299, 0.587, 0.114)).view(3, 1, 1)
        image = (image * weights).sum(dim=0, keepdim=True)
    elif config["color"] != "rgb":
        raise ValueError('color must be "rgb" or "luminance"')

    targets = build_targets(
        record,
        labels,
        config["input_width"],
        config["input_height"],
        config["output_stride"],
    )
    return {
        "image": image.contiguous(),
        "labels": labels,
        "source": record["source"],
        "source_id": record["source_id"],
        "clean": clean,
        "warp_strength": warp_strength,
        **targets,
    }
