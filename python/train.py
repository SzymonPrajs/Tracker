#!/usr/bin/env python3
"""Train the temporal head tracker.

This first loop is spatial control: still images, luminance only, zero motion,
zero prior, reset state every image. Only the heatmap and offset heads get
loss. The graph is already the streaming one, so later pair/clip training
does not need a second model.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from common.model import TrackerModel, model_from_config, split_output
from common.preprocessing import TARGET_KINDS, USED_SOURCES, prepare_example


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the temporal head tracker.")
    parser.add_argument("--config", type=Path, default=ROOT / "config/train.toml")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="one short epoch on a few real images from every source",
    )
    return parser.parse_args()


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return tomllib.load(file)


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def target_count(record: dict[str, Any], preprocess: dict[str, Any]) -> int:
    scale = min(
        preprocess["input_width"] / record["width"],
        preprocess["input_height"] / record["height"],
        1.0,
    )
    return sum(
        1
        for label in record["labels"]
        if label["kind"] in TARGET_KINDS
        and not label.get("ignore", False)
        and min(label["box"][2:]) * scale >= preprocess["minimum_target_pixels"]
    )


def load_records(
    data_dir: Path, preprocess: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train, validation = [], []
    for labels_file in sorted(data_dir.glob("*/labels.jsonl")):
        source = labels_file.parent.name
        if source not in USED_SOURCES:
            continue
        for line in labels_file.open(encoding="utf-8"):
            record = json.loads(line)
            record["image_path"] = labels_file.parent / record["image"]
            if record.get("split") == "validation":
                validation.append(record)
                continue
            negative = "head_center" in record.get("negative_for", [])
            count = target_count(record, preprocess)
            if negative or 0 < count <= preprocess["maximum_train_targets"]:
                train.append(record)
    if not train or not validation:
        raise RuntimeError(f"need train and validation images under {data_dir}")
    return train, validation


def take_balanced(
    records: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_source.setdefault(record["source"], []).append(record)
    rng = random.Random(seed)
    chosen = []
    per_source = max(1, math.ceil(count / len(by_source)))
    for items in by_source.values():
        chosen.extend(rng.sample(items, min(per_source, len(items))))
    rng.shuffle(chosen)
    return chosen[:count]


class HeadDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        preprocess: dict[str, Any],
        seed: int,
        validation: bool,
    ) -> None:
        self.records = records
        self.preprocess = preprocess
        self.seed = seed
        self.validation = validation
        self.epoch = 0
        self.source_counts = Counter(record["source"] for record in records)

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        example_seed = self.seed + self.epoch * 100_003 + index
        return prepare_example(
            record, self.preprocess, example_seed, augment=not self.validation
        )


def collate(examples: list[dict[str, Any]]) -> dict[str, Any]:
    batch = {
        name: torch.stack([example[name] for example in examples])
        for name in ("image", "heatmaps", "valid_mask", "offsets", "regression_mask")
    }
    batch["targets"] = [example["targets"] for example in examples]
    batch["sources"] = [example["source"] for example in examples]
    return batch


def empty_temporal_inputs(
    image: torch.Tensor, model: TrackerModel
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zeros for the streaming inputs that still-image training does not have."""
    batch, _, height, width = image.shape
    motion = image.new_zeros(batch, 2, height // 2, width // 2)
    prior = image.new_zeros(batch, 1, math.ceil(height / 4), math.ceil(width / 4))
    state = model.start_state(batch, height, width, device=image.device, dtype=image.dtype)
    return motion, prior, state.fast, state.slow


def heatmap_loss(
    logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    probabilities = logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    positives = (targets == 1).float() * valid
    negatives = (targets < 1).float() * valid
    positive = -(probabilities.log() * (1 - probabilities).pow(2) * positives).sum()
    negative = -(
        (1 - probabilities).log() * probabilities.pow(2) * (1 - targets).pow(4) * negatives
    ).sum()
    return (positive + negative) / positives.sum().clamp_min(1)


def offset_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if target.dim() == 5:
        target = target.squeeze(1)
    loss = F.smooth_l1_loss(prediction, target, reduction="none") * mask
    return loss.sum() / (mask.sum().clamp_min(1) * 2)


def compute_loss(
    output: torch.Tensor, batch: dict[str, Any], weights: dict[str, float]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    parts = split_output(output)
    pieces = {
        "heatmap": heatmap_loss(parts["heatmap"], batch["heatmaps"], batch["valid_mask"]),
        "offset": offset_loss(parts["offset"], batch["offsets"], batch["regression_mask"]),
    }
    total = pieces["heatmap"] * weights["heatmap_weight"] + pieces["offset"] * weights[
        "offset_weight"
    ]
    return total, pieces


def average_precision(hits: list[int], total_targets: int) -> float:
    if total_targets == 0 or not hits:
        return 0.0
    tp = torch.tensor(hits, dtype=torch.float64)
    fp = 1 - tp
    recall = torch.cat((torch.tensor([0.0]), tp.cumsum(0) / total_targets, torch.tensor([1.0])))
    precision = torch.cat(
        (torch.tensor([0.0]), tp.cumsum(0) / (tp.cumsum(0) + fp.cumsum(0)), torch.tensor([0.0]))
    )
    for index in range(precision.numel() - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    changes = torch.where(recall[1:] != recall[:-1])[0]
    return float(((recall[changes + 1] - recall[changes]) * precision[changes + 1]).sum())


def decode_centers(
    output: torch.Tensor, valid_mask: torch.Tensor, stride: int, top_k: int, floor: float
) -> list[list[tuple[float, float, float]]]:
    parts = split_output(output)
    scores = parts["heatmap"].sigmoid()
    offsets = parts["offset"].clamp(0, 1)
    peaks = scores == F.max_pool2d(scores, 3, stride=1, padding=1)
    scores = torch.where(peaks & valid_mask.bool(), scores, torch.zeros_like(scores))
    _, _, height, width = scores.shape
    decoded = []
    for image_index in range(scores.shape[0]):
        flat = scores[image_index, 0].flatten()
        values, indices = torch.topk(flat, min(top_k, flat.numel()))
        points = []
        for score, flat_index in zip(values.tolist(), indices.tolist()):
            if score < floor:
                break
            y, x = divmod(flat_index, width)
            dx, dy = offsets[image_index, :, y, x].tolist()
            points.append((score, (x + dx) * stride, (y + dy) * stride))
        decoded.append(points)
    return decoded


def evaluate(
    predictions: list[list[tuple[float, float, float]]],
    targets: list[torch.Tensor],
    sources: list[str],
    config: dict[str, Any],
) -> dict[str, float]:
    threshold = config["operating_threshold"]
    ranked = []
    for image_index, points in enumerate(predictions):
        ranked.extend((score, x, y, image_index) for score, x, y in points)
    ranked.sort(reverse=True)
    matched = [set() for _ in targets]
    hits = []
    for _, x, y, image_index in ranked:
        best, best_distance = -1, math.inf
        for index, target in enumerate(targets[image_index]):
            if index in matched[image_index]:
                continue
            distance = math.hypot(x - float(target[0]), y - float(target[1]))
            if distance < best_distance:
                best, best_distance = index, distance
        hit = best >= 0 and best_distance <= 8
        hits.append(int(hit))
        if hit:
            matched[image_index].add(best)

    tp = fp = n_targets = 0
    negative_frames = negative_hits = 0
    source_hits: Counter[str] = Counter()
    source_targets: Counter[str] = Counter()
    used = [set() for _ in targets]
    for image_index, points in enumerate(predictions):
        image_targets = targets[image_index]
        if sources[image_index] == "open_images":
            negative_frames += 1
            negative_hits += sum(score >= threshold for score, _, _ in points)
        for score, x, y in points:
            if score < threshold:
                continue
            best, best_distance = -1, math.inf
            for index, target in enumerate(image_targets):
                if index in used[image_index]:
                    continue
                distance = math.hypot(x - float(target[0]), y - float(target[1]))
                if distance < best_distance:
                    best, best_distance = index, distance
            if best >= 0 and best_distance <= 8:
                used[image_index].add(best)
                tp += 1
            else:
                fp += 1
        n_targets += len(image_targets)
        if len(image_targets):
            source_hits[sources[image_index]] += len(used[image_index])
            source_targets[sources[image_index]] += len(image_targets)

    metrics = {
        "center_ap_8": average_precision(hits, sum(len(item) for item in targets)),
        "center_precision_8": tp / max(1, tp + fp),
        "center_recall_8": tp / max(1, n_targets),
        "negative_false_positives_per_image": negative_hits / max(1, negative_frames),
    }
    for source in sorted(source_targets):
        metrics[f"recall_8_{source}"] = source_hits[source] / max(
            1, source_targets[source]
        )
    return metrics


def make_loader(
    dataset: HeadDataset,
    batch_size: int,
    workers: int,
    source_weights: dict[str, float],
    training: bool,
    seed: int,
) -> DataLoader:
    sampler = None
    if training:
        weights = [
            source_weights[record["source"]] / dataset.source_counts[record["source"]]
            for record in dataset.records
        ]
        sampler = WeightedRandomSampler(
            weights, len(weights), generator=torch.Generator().manual_seed(seed)
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False if sampler else not training,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=training,
        collate_fn=collate,
    )


def run_epoch(
    model: TrackerModel,
    loader: DataLoader,
    device: torch.device,
    loss_weights: dict[str, float],
    evaluation: dict[str, Any],
    stride: int,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    clip: float,
    mixed_precision: bool,
    label: str,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: Counter[str] = Counter()
    examples = 0
    all_predictions: list[list[tuple[float, float, float]]] = []
    all_targets: list[torch.Tensor] = []
    all_sources: list[str] = []
    progress = tqdm(loader, desc=label, unit="batch")

    for batch in progress:
        image = batch["image"].to(device, non_blocking=True)
        for name in ("heatmaps", "valid_mask", "offsets", "regression_mask"):
            batch[name] = batch[name].to(device, non_blocking=True)
        motion, prior, fast, slow = empty_temporal_inputs(image, model)

        if training:
            optimizer.zero_grad(set_to_none=True)
        with (
            torch.set_grad_enabled(training),
            torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=mixed_precision
            ),
        ):
            output, _, _ = model(image, motion, prior, fast, slow)
            total, pieces = compute_loss(output, batch, loss_weights)

        if training:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            all_predictions.extend(
                decode_centers(
                    output.detach().cpu(),
                    batch["valid_mask"].cpu(),
                    stride,
                    evaluation["top_k"],
                    evaluation["score_floor"],
                )
            )
            all_targets.extend(item.cpu() for item in batch["targets"])
            all_sources.extend(batch["sources"])

        batch_size = image.shape[0]
        totals["loss"] += float(total.detach()) * batch_size
        for name, value in pieces.items():
            totals[name] += float(value.detach()) * batch_size
        examples += batch_size
        progress.set_postfix(loss=f"{totals['loss'] / examples:.4f}")

    result = {
        "loss": totals["loss"] / examples,
        "heatmap_loss": totals["heatmap"] / examples,
        "offset_loss": totals["offset"] / examples,
    }
    if all_predictions:
        result.update(evaluate(all_predictions, all_targets, all_sources, evaluation))
    return result


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    best_metric: float,
    train_config: dict[str, Any],
    model_config: dict[str, Any],
    preprocess_config: dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_center_ap_8": best_metric,
            "train_config": train_config,
            "model_config": model_config,
            "preprocess_config": preprocess_config,
        },
        path,
    )


def main() -> None:
    args = arguments()
    config = load_toml(args.config)
    preprocess = load_toml(ROOT / config["preprocess_config"])
    model_config = load_toml(ROOT / config["model_config"])
    if preprocess["color"] != "luminance":
        raise ValueError('set color = "luminance" in config/preprocess.toml')

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = choose_device(config["device"])
    mixed_precision = config["mixed_precision"] and device.type == "cuda"
    model = model_from_config(model_config).to(device)

    train_records, val_records = load_records(args.data_dir, preprocess)
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    workers = config["num_workers"]
    if args.smoke:
        train_records = take_balanced(train_records, 64, config["seed"])
        val_records = take_balanced(val_records, 32, config["seed"] + 1)
        epochs, batch_size, workers = 1, 8, min(workers, 2)

    train_data = HeadDataset(train_records, preprocess, config["seed"], False)
    val_data = HeadDataset(val_records, preprocess, config["seed"], True)
    train_loader = make_loader(
        train_data, batch_size, workers, config["sampling"], True, config["seed"]
    )
    val_loader = make_loader(
        val_data, batch_size, workers, config["sampling"], False, config["seed"]
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["optimizer"]["learning_rate"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=config["optimizer"]["minimum_learning_rate"],
    )
    scaler = torch.amp.GradScaler(device.type, enabled=mixed_precision)

    output_dir = ROOT / ("runs/smoke" if args.smoke else config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    start_epoch, best_metric = 0, -math.inf
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint["best_center_ap_8"]
    else:
        metrics_path.write_text("")

    print(f"device: {device}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"input: {preprocess['input_width']}x{preprocess['input_height']} luminance")
    print(f"training images: {len(train_data):,} {dict(train_data.source_counts)}")
    print(f"validation images: {len(val_data):,} {dict(val_data.source_counts)}")

    for epoch in range(start_epoch, epochs):
        train_data.set_epoch(epoch)
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            config["loss"],
            config["evaluation"],
            preprocess["output_stride"],
            optimizer,
            scaler,
            config["optimizer"]["gradient_clip"],
            mixed_precision,
            f"Epoch {epoch + 1}/{epochs} train",
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            config["loss"],
            config["evaluation"],
            preprocess["output_stride"],
            None,
            scaler,
            config["optimizer"]["gradient_clip"],
            mixed_precision,
            f"Epoch {epoch + 1}/{epochs} validation",
        )
        result = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": val_metrics,
        }
        scheduler.step()
        with metrics_path.open("a") as file:
            file.write(json.dumps(result, separators=(",", ":")) + "\n")

        metric = val_metrics.get("center_ap_8", -math.inf)
        improved = metric > best_metric
        best_metric = max(best_metric, metric)
        payload = (
            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            best_metric,
            config,
            model_config,
            preprocess,
        )
        save_checkpoint(output_dir / "last.pt", *payload)
        if improved:
            save_checkpoint(output_dir / "best.pt", *payload)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
