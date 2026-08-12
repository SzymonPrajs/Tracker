#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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

from common.model import TrackerModel, split_output
from common.preprocessing import TARGET_KINDS, USED_SOURCES, prepare_example


ROOT = Path(__file__).resolve().parents[1]
TENSOR_NAMES = ("image", "heatmaps", "valid_mask", "offsets", "regression_mask")
Entry = tuple[Path, int, str, int]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the compact head-center model.")
    parser.add_argument("--config", type=Path, default=ROOT / "config/train.toml")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one short epoch using real samples from every source",
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.blake2b(f"{value}:{seed}".encode(), digest_size=8).digest()
    return int.from_bytes(digest) / 2**64


def _target_count(record: dict[str, Any], preprocess: dict[str, Any]) -> int:
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


def load_entries(
    data_dir: Path,
    preprocess: dict[str, Any],
    validation_fraction: float,
    seed: int,
) -> tuple[list[Entry], list[Entry]]:
    candidates = []
    official_sources = set()
    for labels_file in sorted(data_dir.glob("*/labels.jsonl")):
        source = labels_file.parent.name
        if source not in USED_SOURCES:
            continue
        with labels_file.open("rb") as file:
            line_number = 0
            while True:
                offset = file.tell()
                line = file.readline()
                if not line:
                    break
                record = json.loads(line)
                split = record.get("split", "train")
                if split == "validation":
                    official_sources.add(source)
                image_path = labels_file.parent / record["image"]
                content_hash = record.get("content_hash")
                if not content_hash:
                    content_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
                group = record.get("group")
                group_key = f"{source}:group:{group}" if group else content_hash
                candidates.append(
                    (
                        (labels_file, offset, source, line_number),
                        record,
                        group_key,
                        split,
                        _target_count(record, preprocess),
                    )
                )
                line_number += 1

    provisional = []
    for entry, record, group, split, target_count in candidates:
        source = entry[2]
        if source in official_sources:
            validation = split == "validation"
        else:
            validation = _stable_fraction(group, seed) < validation_fraction
        provisional.append((entry, record, group, validation, target_count))

    validation_groups = {
        group for _, _, group, validation, _ in provisional if validation
    }
    seen = {False: set(), True: set()}
    result: dict[bool, list[Entry]] = {False: [], True: []}
    maximum_targets = preprocess["maximum_train_targets"]
    for entry, record, group, validation, target_count in provisional:
        if not validation and group in validation_groups:
            continue
        if group in seen[validation]:
            continue
        seen[validation].add(group)

        negative = "head_center" in record.get("negative_for", [])
        if not validation and not negative:
            if target_count == 0 or target_count > maximum_targets:
                continue
        result[validation].append(entry)

    if not result[False] or not result[True]:
        raise RuntimeError(
            "the selected data must contain training and validation images"
        )
    return result[False], result[True]


class TrackerDataset(Dataset):
    def __init__(
        self,
        entries: list[Entry],
        preprocess_config: dict[str, Any],
        seed: int,
        validation: bool,
    ) -> None:
        self.entries = entries
        self.preprocess_config = preprocess_config
        self.seed = seed
        self.validation = validation
        self.epoch = 0
        self._handles: dict[Path, Any] = {}
        self.source_counts = Counter(entry[2] for entry in entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def balanced_subset(self, count: int) -> None:
        rng = random.Random(self.seed + (1 if self.validation else 0))
        by_source: dict[str, list[Entry]] = {}
        for entry in self.entries:
            by_source.setdefault(entry[2], []).append(entry)
        chosen = []
        per_source = max(1, math.ceil(count / len(by_source)))
        for entries in by_source.values():
            chosen.extend(rng.sample(entries, min(per_source, len(entries))))
        rng.shuffle(chosen)
        self.entries = chosen[:count]
        self.source_counts = Counter(entry[2] for entry in self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        labels_file, offset, _, _ = self.entries[index]
        if labels_file not in self._handles:
            self._handles[labels_file] = labels_file.open("rb")
        file = self._handles[labels_file]
        file.seek(offset)
        record = json.loads(file.readline())
        record["image_path"] = labels_file.parent / record["image"]
        identity = f"{record['source']}:{record['source_id']}"
        example_seed = (
            int.from_bytes(hashlib.blake2b(identity.encode(), digest_size=8).digest())
            + self.seed
            + self.epoch
        )
        return prepare_example(
            record,
            self.preprocess_config,
            example_seed,
            augment=not self.validation,
        )


def collate_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {
        name: torch.stack([example[name] for example in examples])
        for name in TENSOR_NAMES
    }
    batch["targets"] = [example["targets"] for example in examples]
    batch["sources"] = [example["source"] for example in examples]
    return batch


def heatmap_focal_loss(
    logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    probabilities = logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    positives = (targets == 1).float() * valid_mask
    negatives = (targets < 1).float() * valid_mask
    negative_weights = (1 - targets).pow(4)
    positive_loss = -(
        probabilities.log() * (1 - probabilities).pow(2) * positives
    ).sum()
    negative_loss = -(
        (1 - probabilities).log() * probabilities.pow(2) * negative_weights * negatives
    ).sum()
    return (positive_loss + negative_loss) / positives.sum().clamp_min(1)


def offset_loss(
    predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    expanded_mask = mask.unsqueeze(2)
    loss = F.smooth_l1_loss(predictions, targets, reduction="none") * expanded_mask
    return loss.sum() / (expanded_mask.sum().clamp_min(1) * 2)


def compute_loss(
    output: torch.Tensor, batch: dict[str, Any], config: dict[str, float]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    predictions = split_output(output)
    pieces = {
        "heatmap": heatmap_focal_loss(
            predictions["heatmaps"], batch["heatmaps"], batch["valid_mask"]
        ),
        "offset": offset_loss(
            predictions["offsets"], batch["offsets"], batch["regression_mask"]
        ),
    }
    total = (
        pieces["heatmap"] * config["heatmap_weight"]
        + pieces["offset"] * config["offset_weight"]
    )
    return total, pieces


def _average_precision(true_positive: list[int], total_targets: int) -> float:
    if total_targets == 0 or not true_positive:
        return 0.0
    tp = torch.tensor(true_positive, dtype=torch.float64)
    fp = 1 - tp
    recall = torch.cat(
        (torch.tensor([0.0]), tp.cumsum(0) / total_targets, torch.tensor([1.0]))
    )
    precision = torch.cat(
        (
            torch.tensor([0.0]),
            tp.cumsum(0) / (tp.cumsum(0) + fp.cumsum(0)),
            torch.tensor([0.0]),
        )
    )
    for index in range(precision.numel() - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    changes = torch.where(recall[1:] != recall[:-1])[0]
    return float(
        ((recall[changes + 1] - recall[changes]) * precision[changes + 1]).sum()
    )


def _match_image(
    predictions: list[tuple[float, float, float]],
    targets: torch.Tensor,
    tolerance: float,
    threshold: float = 0.0,
) -> tuple[int, int, set[int]]:
    matched: set[int] = set()
    true_positive = 0
    false_positive = 0
    for score, x, y in predictions:
        if score < threshold:
            continue
        best_index = -1
        best_distance = math.inf
        for index, target in enumerate(targets):
            if index in matched:
                continue
            distance = math.hypot(x - float(target[0]), y - float(target[1]))
            if distance < best_distance:
                best_index, best_distance = index, distance
        if best_index >= 0 and best_distance <= tolerance:
            matched.add(best_index)
            true_positive += 1
        else:
            false_positive += 1
    return true_positive, false_positive, matched


class CenterMetrics:
    def __init__(self, stride: int, top_k: int, score_floor: float) -> None:
        self.stride = stride
        self.top_k = top_k
        self.score_floor = score_floor
        self.predictions: list[list[tuple[float, float, float]]] = []
        self.targets: list[torch.Tensor] = []
        self.sources: list[str] = []

    def add(self, output: torch.Tensor, batch: dict[str, Any]) -> None:
        parts = split_output(output.detach().cpu())
        scores = parts["heatmaps"].sigmoid()
        offsets = parts["offsets"].clamp(0, 1)
        maxima = scores == F.max_pool2d(scores, 3, stride=1, padding=1)
        valid = batch["valid_mask"].cpu().bool()
        scores = torch.where(maxima & valid, scores, torch.zeros_like(scores))
        _, _, height, width = scores.shape
        for image_index in range(scores.shape[0]):
            flat = scores[image_index, 0].flatten()
            count = min(self.top_k, flat.numel())
            values, indices = torch.topk(flat, count)
            decoded = []
            for score, flat_index in zip(values.tolist(), indices.tolist()):
                if score < self.score_floor:
                    break
                y, x = divmod(flat_index, width)
                offset = offsets[image_index, 0, :, y, x]
                decoded.append(
                    (
                        score,
                        (x + float(offset[0])) * self.stride,
                        (y + float(offset[1])) * self.stride,
                    )
                )
            self.predictions.append(decoded)
        self.targets.extend(batch["targets"])
        self.sources.extend(batch["sources"])

    def _ap(self, tolerance: float) -> float:
        ordered = []
        for image_index, predictions in enumerate(self.predictions):
            ordered.extend((*prediction, image_index) for prediction in predictions)
        ordered.sort(reverse=True)
        matched = [set() for _ in self.targets]
        outcomes = []
        for _, x, y, image_index in ordered:
            best_index = -1
            best_distance = math.inf
            for index, target in enumerate(self.targets[image_index]):
                if index in matched[image_index]:
                    continue
                distance = math.hypot(x - float(target[0]), y - float(target[1]))
                if distance < best_distance:
                    best_index, best_distance = index, distance
            hit = best_index >= 0 and best_distance <= tolerance
            outcomes.append(int(hit))
            if hit:
                matched[image_index].add(best_index)
        return _average_precision(
            outcomes, sum(len(targets) for targets in self.targets)
        )

    def compute(self, threshold: float) -> dict[str, float]:
        result = {"center_ap_4": self._ap(4), "center_ap_8": self._ap(8)}
        total_tp = total_fp = total_targets = 0
        negative_frames = negative_predictions = negative_frames_with_prediction = 0
        source_hits: Counter[str] = Counter()
        source_targets: Counter[str] = Counter()
        size_hits: Counter[str] = Counter()
        size_targets: Counter[str] = Counter()

        for predictions, targets, source in zip(
            self.predictions, self.targets, self.sources
        ):
            tp, fp, matched = _match_image(predictions, targets, 8, threshold)
            total_tp += tp
            total_fp += fp
            total_targets += len(targets)
            if len(targets):
                source_hits[source] += tp
                source_targets[source] += len(targets)
            if source == "open_images":
                count = sum(score >= threshold for score, _, _ in predictions)
                negative_frames += 1
                negative_predictions += count
                negative_frames_with_prediction += int(count > 0)
            for index, target in enumerate(targets):
                short_side = float(target[2])
                if short_side < 12:
                    band = "8_12"
                elif short_side < 24:
                    band = "12_24"
                else:
                    band = "24_plus"
                size_targets[band] += 1
                size_hits[band] += int(index in matched)

        result.update(
            {
                "center_precision_8": total_tp / max(1, total_tp + total_fp),
                "center_recall_8": total_tp / max(1, total_targets),
                "negative_false_positives_per_image": negative_predictions
                / max(1, negative_frames),
                "negative_images_with_prediction": negative_frames_with_prediction
                / max(1, negative_frames),
            }
        )
        for source in sorted(source_targets):
            result[f"recall_8_{source}"] = source_hits[source] / max(
                1, source_targets[source]
            )
        for band in sorted(size_targets):
            result[f"recall_8_size_{band}"] = size_hits[band] / max(
                1, size_targets[band]
            )
        return result


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    for name in TENSOR_NAMES:
        batch[name] = batch[name].to(device, non_blocking=True)
    return batch


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_config: dict[str, float],
    evaluation_config: dict[str, Any],
    stride: int,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    gradient_clip: float,
    mixed_precision: bool,
    label: str,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = Counter()
    examples = 0
    metrics = None
    if not training:
        metrics = CenterMetrics(
            stride, evaluation_config["top_k"], evaluation_config["score_floor"]
        )
    progress = tqdm(loader, desc=label, unit="batch")

    for batch in progress:
        batch = move_batch(batch, device)
        batch_size = batch["image"].shape[0]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with (
            torch.set_grad_enabled(training),
            torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=mixed_precision
            ),
        ):
            output = model(batch["image"])
            total, pieces = compute_loss(output, batch, loss_config)

        if training:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            metrics.add(output, batch)

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
    if metrics:
        result.update(metrics.compute(evaluation_config["operating_threshold"]))
    return result


def make_loader(
    dataset: TrackerDataset,
    batch_size: int,
    workers: int,
    source_weights: dict[str, float],
    training: bool,
    seed: int,
) -> DataLoader:
    sampler = None
    shuffle = training
    if training:
        weights = [
            source_weights[entry[2]] / dataset.source_counts[entry[2]]
            for entry in dataset.entries
        ]
        sampler = WeightedRandomSampler(
            weights,
            len(weights),
            generator=torch.Generator().manual_seed(seed),
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=training,
        collate_fn=collate_examples,
    )


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    best_metric: float,
    training_config: dict[str, Any],
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
            "model_config": training_config["model"],
            "training_config": training_config,
            "preprocess_config": preprocess_config,
        },
        path,
    )


def main() -> None:
    args = arguments()
    with args.config.open("rb") as file:
        config = tomllib.load(file)
    preprocess_path = ROOT / config["preprocess_config"]
    with preprocess_path.open("rb") as file:
        preprocess_config = tomllib.load(file)

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = choose_device(config["device"])
    mixed_precision = config["mixed_precision"] and device.type == "cuda"
    input_channels = 1 if preprocess_config["color"] == "luminance" else 3
    model = TrackerModel(input_channels=input_channels, **config["model"]).to(device)

    training_entries, validation_entries = load_entries(
        args.data_dir,
        preprocess_config,
        config["validation_fraction"],
        config["seed"],
    )
    training_data = TrackerDataset(
        training_entries, preprocess_config, config["seed"], validation=False
    )
    validation_data = TrackerDataset(
        validation_entries, preprocess_config, config["seed"], validation=True
    )
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    workers = config["num_workers"]
    if args.smoke:
        training_data.balanced_subset(64)
        validation_data.balanced_subset(32)
        epochs, batch_size, workers = 1, 8, min(workers, 2)

    training_loader = make_loader(
        training_data,
        batch_size,
        workers,
        config["sampling"],
        True,
        config["seed"],
    )
    validation_loader = make_loader(
        validation_data,
        batch_size,
        workers,
        config["sampling"],
        False,
        config["seed"],
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

    output_dir = ROOT / config["output_dir"]
    if args.smoke:
        output_dir = output_dir.parent / "smoke_focused"
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

    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"device: {device}")
    print(f"parameters: {parameters:,}")
    print(
        f"input: {preprocess_config['input_width']}x{preprocess_config['input_height']}"
    )
    print(
        f"training images: {len(training_data):,} {dict(training_data.source_counts)}"
    )
    print(
        f"validation images: {len(validation_data):,} "
        f"{dict(validation_data.source_counts)}"
    )

    for epoch in range(start_epoch, epochs):
        training_data.set_epoch(epoch)
        learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = run_epoch(
            model,
            training_loader,
            device,
            config["loss"],
            config["evaluation"],
            preprocess_config["output_stride"],
            optimizer,
            scaler,
            config["optimizer"]["gradient_clip"],
            mixed_precision,
            f"Epoch {epoch + 1}/{epochs} train",
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            device,
            config["loss"],
            config["evaluation"],
            preprocess_config["output_stride"],
            None,
            scaler,
            config["optimizer"]["gradient_clip"],
            mixed_precision,
            f"Epoch {epoch + 1}/{epochs} validation",
        )
        result = {
            "epoch": epoch + 1,
            "learning_rate": learning_rate,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        scheduler.step()
        with metrics_path.open("a") as file:
            file.write(json.dumps(result, separators=(",", ":")) + "\n")

        metric = validation_metrics["center_ap_8"]
        improved = metric > best_metric
        best_metric = max(best_metric, metric)
        save_checkpoint(
            output_dir / "last.pt",
            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            best_metric,
            config,
            preprocess_config,
        )
        if improved:
            save_checkpoint(
                output_dir / "best.pt",
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                best_metric,
                config,
                preprocess_config,
            )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
