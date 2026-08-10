#!/usr/bin/env python3
"""Evaluate a checkpoint under deterministic lighting and lens stress profiles."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from evaluate import average_precision, decode
from tracker_training.augmentation import augment_batch
from tracker_training.data import PackedHeadDataset
from tracker_training.model import HCDS31
from tracker_training.quantization import representative_indices
from train import choose_device, move_target, normalize, parser as training_parser, synchronize


WIDTH, HEIGHT, STRIDE = 288, 160, 4
PROFILES = ("clean", "low_light", "uneven_light", "fisheye_130", "fisheye_180", "telephoto", "mixed")


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


def profile_arguments(profile: str):
    args = clean_arguments()
    if profile == "clean":
        return args
    if profile == "low_light":
        args.exposure_probability = args.low_light_probability = 1.0
        args.low_light_min_ev, args.low_light_max_ev = -3.5, -1.5
        args.noise_probability = 1.0
        args.white_balance_probability = 0.4
    elif profile == "uneven_light":
        args.exposure_probability = args.white_balance_probability = 1.0
        args.exposure_min_ev, args.exposure_max_ev = -1.5, 0.5
        args.illumination_gradient_probability = args.shadow_probability = 1.0
        args.gamma_probability = 0.5
    elif profile == "fisheye_130":
        args.fisheye_130_probability = 1.0
    elif profile == "fisheye_180":
        args.fisheye_180_probability = 1.0
        args.vignette_probability = 1.0
    elif profile == "telephoto":
        args.telephoto_probability = 1.0
    elif profile == "mixed":
        args = training_parser().parse_args([])
        args.horizontal_flip = 0
        args.augmentation_clean_probability = 0
    else:
        raise ValueError(profile)
    return args


def target_boxes(target, batch_index: int) -> np.ndarray:
    _, offset, size, mask = target
    result = []
    for iy, ix in mask[batch_index, 0].nonzero().cpu().tolist():
        cx = (ix + 0.5 + float(offset[batch_index, 0, iy, ix])) * STRIDE
        cy = (iy + 0.5 + float(offset[batch_index, 1, iy, ix])) * STRIDE
        width = float(size[batch_index, 0, iy, ix]) * WIDTH
        height = float(size[batch_index, 1, iy, ix]) * HEIGHT
        result.append((cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2))
    return np.asarray(result, dtype=np.float32).reshape(-1, 4)


def evaluate_profile(model, dataset, indices, profile, device, batch_size, maximum_detections, seed):
    batches = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0)
    augment_args = profile_arguments(profile)
    predictions = []
    ground_truth = {}
    room_ids = set()
    image_index = 0
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (images, target) in enumerate(batches):
            images = normalize(images, device, channels_last=False)
            target = move_target(target, device)
            if profile != "clean":
                torch.manual_seed(seed + batch_index)
                images, target = augment_batch(images, target, augment_args)
            scores, boxes = decode(model(images), maximum_detections)
            scores, boxes = scores.cpu().numpy(), boxes.cpu().numpy()
            target = tuple(value.cpu() for value in target)
            for local_index in range(images.shape[0]):
                ground_truth[image_index] = target_boxes(target, local_index)
                source_row = dataset.rows[indices[image_index]]
                source_heads = sum(not head["ignore"] for head in source_row["heads"])
                if source_heads <= 4:
                    room_ids.add(image_index)
                predictions.extend(
                    (image_index, float(score), box)
                    for score, box in zip(scores[local_index], boxes[local_index], strict=True)
                )
                image_index += 1
    synchronize(device)
    ap30 = average_precision(predictions, ground_truth, 0.30)
    ap50 = average_precision(predictions, ground_truth, 0.50)
    room_truth = {index: ground_truth[index] for index in room_ids}
    room_predictions = [prediction for prediction in predictions if prediction[0] in room_ids]
    room_ap30 = average_precision(room_predictions, room_truth, 0.30)
    room_ap50 = average_precision(room_predictions, room_truth, 0.50)
    return {
        "images": len(indices),
        "heads_after_transform": sum(len(boxes) for boxes in ground_truth.values()),
        "ap30": ap30,
        "ap50": ap50,
        "map30_50": (ap30 + ap50) / 2,
        "room_images": len(room_ids),
        "room_ap30": room_ap30,
        "room_ap50": room_ap50,
        "room_map30_50": (room_ap30 + room_ap50) / 2,
        "seconds": time.perf_counter() - started,
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("checkpoint", type=Path)
    cli.add_argument("--data-root", type=Path, default=Path("data/packed/288x160"))
    cli.add_argument("--split", choices=("val", "test"), default="val")
    cli.add_argument("--samples", type=int, default=2048)
    cli.add_argument("--batch-size", type=int, default=128)
    cli.add_argument("--maximum-detections", type=int, default=20)
    cli.add_argument("--device", default="auto")
    cli.add_argument("--seed", type=int, default=20260810)
    cli.add_argument("--profiles", nargs="+", choices=PROFILES, default=PROFILES)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    device = choose_device(args.device)
    dataset = PackedHeadDataset(args.data_root, args.split)
    indices = representative_indices(dataset.rows, min(args.samples, len(dataset)), args.seed)
    model = HCDS31().to(device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    result = {
        "checkpoint": str(args.checkpoint), "split": args.split, "sample_indices": indices,
        "profiles": {},
    }
    for profile in args.profiles:
        print(f"evaluating {profile} on {len(indices)} scenes", flush=True)
        result["profiles"][profile] = evaluate_profile(
            model, dataset, indices, profile, device, args.batch_size,
            args.maximum_detections, args.seed,
        )
    clean = result["profiles"].get("clean")
    if clean:
        for values in result["profiles"].values():
            values["room_map_delta_from_clean"] = values["room_map30_50"] - clean["room_map30_50"]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
