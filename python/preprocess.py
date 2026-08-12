#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import tomllib
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm

from common.preprocessing import SEMANTICS, balanced_sample, prepare_example


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview the exact augmentations and targets used for training."
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config/preprocess.toml")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "previews/preprocess.png")
    parser.add_argument("--count", type=int, help="override preview_count")
    parser.add_argument("--seed", type=int, help="override seed")
    parser.add_argument(
        "--clean", action="store_true", help="disable random augmentation"
    )
    return parser.parse_args()


def _rgb(image: torch.Tensor) -> Image.Image:
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    pixels = (image.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(pixels, "RGB")


def _tile(example: dict, width: int, height: int) -> Image.Image:
    image = _rgb(example["image"])
    draw = ImageDraw.Draw(image)
    colors = {
        "head": "#00ff66",
        "face": "#00d9ff",
    }
    for label in example["labels"]:
        if label.get("ignore", False):
            continue
        x, y, box_width, box_height = label["box"]
        if label["kind"] in colors:
            draw.rectangle(
                (x, y, x + box_width, y + box_height),
                outline=colors[label["kind"]],
            )

    heat = example["heatmaps"].amax(dim=0, keepdim=True).unsqueeze(0)
    heat = F.interpolate(heat, (height, width), mode="bilinear", align_corners=False)[
        0, 0
    ]
    overlay = torch.zeros(3, height, width)
    overlay[0] = heat
    overlay[1] = heat * 0.3
    overlay_image = _rgb(example["image"] * 0.45 + overlay * 0.75)

    tile = Image.new("RGB", (width * 2, height + 18), "black")
    tile.paste(image, (0, 18))
    tile.paste(overlay_image, (width, 18))
    label = example["source"]
    if example["clean"]:
        label += " | clean"
    else:
        label += f" | warp {example['warp_strength']:.2f}"
    ImageDraw.Draw(tile).text((4, 2), label, fill="white")
    return tile


def save_preview(examples: list[dict], output: Path, width: int, height: int) -> None:
    columns = 2
    rows = math.ceil(len(examples) / columns)
    sheet = Image.new("RGB", (columns * width * 2, rows * (height + 18)), "#181818")
    for index, example in enumerate(examples):
        tile = _tile(example, width, height)
        sheet.paste(
            tile, ((index % columns) * width * 2, (index // columns) * (height + 18))
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    args = arguments()
    with args.config.open("rb") as file:
        config = tomllib.load(file)
    count = args.count or config["preview_count"]
    seed = args.seed if args.seed is not None else config["seed"]
    records = balanced_sample(args.data_dir, count, seed)
    examples = [
        prepare_example(record, config, seed + index, augment=not args.clean)
        for index, record in enumerate(tqdm(records, desc="Preprocess", unit="image"))
    ]
    save_preview(examples, args.output, config["input_width"], config["input_height"])

    first = examples[0]
    print(f"Wrote {args.output}")
    print(f"image: {tuple(first['image'].shape)}")
    for name in ("heatmaps", "valid_mask", "offsets", "regression_mask", "targets"):
        print(f"{name}: {tuple(first[name].shape)}")
    print("channels: " + ", ".join(SEMANTICS))


if __name__ == "__main__":
    main()
