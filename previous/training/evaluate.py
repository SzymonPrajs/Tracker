#!/usr/bin/env python3
"""Evaluate decoded head boxes without tuning a confidence threshold."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tracker_training.data import PackedHeadDataset
from tracker_training.model import HCDS31
from train import choose_device, normalize, synchronize


WIDTH, HEIGHT, STRIDE = 288, 160, 4


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.empty(0, dtype=np.float32)
    left_top = np.maximum(box[:2], boxes[:, :2])
    right_bottom = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.maximum(0, right_bottom - left_top).prod(axis=1)
    area = np.maximum(0, box[2:] - box[:2]).prod()
    areas = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    return intersection / np.maximum(area + areas - intersection, 1e-9)


def average_precision(predictions, ground_truth, threshold: float) -> float:
    total_ground_truth = sum(len(boxes) for boxes in ground_truth.values())
    if not total_ground_truth:
        return 0.0
    matched = defaultdict(set)
    true_positive = []
    false_positive = []
    for image_index, score, box in sorted(predictions, key=lambda item: item[1], reverse=True):
        boxes = ground_truth[image_index]
        overlaps = box_iou(box, boxes)
        order = np.argsort(-overlaps)
        match = next((int(index) for index in order if index not in matched[image_index]), None)
        success = match is not None and overlaps[match] >= threshold
        true_positive.append(float(success))
        false_positive.append(float(not success))
        if success:
            matched[image_index].add(match)
    tp = np.cumsum(true_positive)
    fp = np.cumsum(false_positive)
    recall = tp / total_ground_truth
    precision = tp / np.maximum(tp + fp, 1e-9)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changes = np.flatnonzero(recall[1:] != recall[:-1]) + 1
    return float(np.sum((recall[changes] - recall[changes - 1]) * precision[changes]))


def decode(prediction: torch.Tensor, maximum_detections: int):
    heat = prediction[:, :1].sigmoid()
    local = F.max_pool2d(heat, kernel_size=3, stride=1, padding=1)
    peaks = torch.where(heat == local, heat, heat.new_full((), -1))
    count = min(maximum_detections, peaks.shape[-2] * peaks.shape[-1])
    scores, indices = peaks.flatten(1).topk(count, dim=1)
    offset = prediction[:, 1:3].flatten(2).gather(
        2, indices[:, None].expand(-1, 2, -1)
    )
    size = prediction[:, 3:5].flatten(2).gather(
        2, indices[:, None].expand(-1, 2, -1)
    )
    x = (indices % (WIDTH // STRIDE) + 0.5 + offset[:, 0]) * STRIDE
    y = (indices // (WIDTH // STRIDE) + 0.5 + offset[:, 1]) * STRIDE
    width = (size[:, 0] * WIDTH).clamp(4, WIDTH)
    height = (size[:, 1] * HEIGHT).clamp(4, HEIGHT)
    boxes = torch.stack((x - width / 2, y - height / 2, x + width / 2, y + height / 2), dim=-1)
    return scores, boxes


def row_ground_truth(row: dict) -> np.ndarray:
    boxes = []
    for head in row["heads"]:
        if head["ignore"]:
            continue
        x, y, width, height = head["bbox_cache_xywh"]
        boxes.append((x, y, x + width, y + height))
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def evaluate(args) -> dict:
    device = choose_device(args.device)
    dataset = PackedHeadDataset(args.data_root, args.split)
    batches = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = HCDS31().to(device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    predictions = []
    ground_truth = {}
    image_index = 0
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for images, _ in batches:
            images = normalize(images, device, channels_last=False)
            scores, boxes = decode(model(images), args.maximum_detections)
            scores = scores.cpu().numpy()
            boxes = boxes.cpu().numpy()
            for batch_index in range(images.shape[0]):
                ground_truth[image_index] = row_ground_truth(dataset.rows[image_index])
                predictions.extend(
                    (image_index, float(score), box)
                    for score, box in zip(scores[batch_index], boxes[batch_index], strict=True)
                )
                image_index += 1
    synchronize(device)
    seconds = time.perf_counter() - started
    ap30 = average_precision(predictions, ground_truth, 0.30)
    ap50 = average_precision(predictions, ground_truth, 0.50)
    room_ids = {index for index, boxes in ground_truth.items() if len(boxes) <= 4}
    room_ground_truth = {index: boxes for index, boxes in ground_truth.items() if index in room_ids}
    room_predictions = [item for item in predictions if item[0] in room_ids]
    room_ap30 = average_precision(room_predictions, room_ground_truth, 0.30)
    room_ap50 = average_precision(room_predictions, room_ground_truth, 0.50)
    mean_ap = (ap30 + ap50) / 2
    room_mean_ap = (room_ap30 + room_ap50) / 2
    result = {
        "split": args.split,
        "images": len(dataset),
        "ground_truth_heads": sum(len(boxes) for boxes in ground_truth.values()),
        "maximum_detections": args.maximum_detections,
        "ap30": ap30,
        "ap50": ap50,
        "map30_50": mean_ap,
        "room_images_at_most_four_heads": len(room_ids),
        "room_ap30": room_ap30,
        "room_ap50": room_ap50,
        "room_map30_50": room_mean_ap,
        "selection_score": 0.75 * room_mean_ap + 0.25 * mean_ap,
        "seconds": seconds,
        "images_per_second": len(dataset) / seconds,
        "checkpoint": str(args.checkpoint),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("checkpoint", type=Path)
    result.add_argument("--data-root", type=Path, default=Path("data/packed/288x160"))
    result.add_argument("--split", choices=("val", "test"), default="val")
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--maximum-detections", type=int, default=20)
    result.add_argument("--device", default="auto")
    result.add_argument("--output", type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
