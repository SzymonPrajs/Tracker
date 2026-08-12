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
from common.preprocessing import prepare_example


ROOT = Path(__file__).resolve().parents[1]
TENSOR_NAMES = (
    "image",
    "heatmaps",
    "valid_mask",
    "sizes",
    "offsets",
    "regression_mask",
)


class TrackerDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        preprocess_config: dict[str, Any],
        validation_fraction: float,
        seed: int,
        validation: bool,
    ) -> None:
        self.preprocess_config = preprocess_config
        self.seed = seed
        self.validation = validation
        self.epoch = 0
        self.entries: list[tuple[Path, int, str, int]] = []
        self._handles: dict[Path, Any] = {}

        for labels_file in sorted(data_dir.glob("*/labels.jsonl")):
            source = labels_file.parent.name
            with labels_file.open("rb") as file:
                line_number = 0
                while True:
                    offset = file.tell()
                    line = file.readline()
                    if not line:
                        break
                    key = f"{source}:{line_number}:{seed}".encode()
                    value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest())
                    is_validation = value / 2**64 < validation_fraction
                    if is_validation == validation:
                        self.entries.append((labels_file, offset, source, line_number))
                    line_number += 1
        self.source_counts = Counter(entry[2] for entry in self.entries)

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
        by_source: dict[str, list[tuple[Path, int, str, int]]] = {}
        for entry in self.entries:
            by_source.setdefault(entry[2], []).append(entry)
        chosen = []
        per_source = math.ceil(count / len(by_source))
        for entries in by_source.values():
            chosen.extend(rng.sample(entries, min(per_source, len(entries))))
        rng.shuffle(chosen)
        self.entries = chosen[:count]
        self.source_counts = Counter(entry[2] for entry in self.entries)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        labels_file, offset, _, line_number = self.entries[index]
        if labels_file not in self._handles:
            self._handles[labels_file] = labels_file.open("rb")
        file = self._handles[labels_file]
        file.seek(offset)
        record = json.loads(file.readline())
        record["image_path"] = labels_file.parent / record["image"]
        example_seed = self.seed + self.epoch * len(self.entries) + line_number
        example = prepare_example(
            record,
            self.preprocess_config,
            example_seed,
            augment=not self.validation,
        )
        return {name: example[name] for name in TENSOR_NAMES}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the tracker on the compact real data."
    )
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


def regression_loss(
    predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    expanded_mask = mask.unsqueeze(2)
    loss = F.smooth_l1_loss(predictions, targets, reduction="none") * expanded_mask
    return loss.sum() / (expanded_mask.sum().clamp_min(1) * 2)


def compute_loss(
    output: torch.Tensor, batch: dict[str, torch.Tensor], config: dict[str, float]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    predictions = split_output(output)
    pieces = {
        "heatmap": heatmap_focal_loss(
            predictions["heatmaps"], batch["heatmaps"], batch["valid_mask"]
        ),
        "size": regression_loss(
            predictions["sizes"], batch["sizes"], batch["regression_mask"]
        ),
        "offset": regression_loss(
            predictions["offsets"], batch["offsets"], batch["regression_mask"]
        ),
    }
    total = (
        pieces["heatmap"] * config["heatmap_weight"]
        + pieces["size"] * config["size_weight"]
        + pieces["offset"] * config["offset_weight"]
    )
    return total, pieces


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_config: dict[str, float],
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
    progress = tqdm(loader, desc=label, unit="batch")

    for batch in progress:
        batch = move_batch(batch, device)
        batch_size = batch["image"].shape[0]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with (
            torch.set_grad_enabled(training),
            torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=mixed_precision,
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

        predictions = split_output(output)["heatmaps"].sigmoid()
        valid = batch["valid_mask"].bool()
        local_maxima = predictions == F.max_pool2d(predictions, 3, stride=1, padding=1)
        predicted_centers = (
            (predictions >= loss_config["center_threshold"]) & local_maxima & valid
        )
        true_centers = (batch["heatmaps"] == 1) & valid
        totals["true_positive"] += int((predicted_centers & true_centers).sum())
        totals["false_positive"] += int((predicted_centers & ~true_centers).sum())
        totals["false_negative"] += int((~predicted_centers & true_centers).sum())
        totals["loss"] += float(total.detach()) * batch_size
        for name, value in pieces.items():
            totals[name] += float(value.detach()) * batch_size
        examples += batch_size
        progress.set_postfix(loss=f"{totals['loss'] / examples:.4f}")

    precision_denominator = totals["true_positive"] + totals["false_positive"]
    recall_denominator = totals["true_positive"] + totals["false_negative"]
    return {
        "loss": totals["loss"] / examples,
        "heatmap_loss": totals["heatmap"] / examples,
        "size_loss": totals["size"] / examples,
        "offset_loss": totals["offset"] / examples,
        "center_precision": totals["true_positive"] / max(1, precision_denominator),
        "center_recall": totals["true_positive"] / max(1, recall_denominator),
    }


def make_loader(
    dataset: TrackerDataset,
    batch_size: int,
    workers: int,
    balance_sources: bool,
    training: bool,
    seed: int,
) -> DataLoader:
    sampler = None
    shuffle = training
    if training and balance_sources:
        weights = [1 / dataset.source_counts[entry[2]] for entry in dataset.entries]
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(weights, len(weights), generator=generator)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=training,
    )


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    best_loss: float,
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
            "best_validation_loss": best_loss,
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

    training_data = TrackerDataset(
        args.data_dir,
        preprocess_config,
        config["validation_fraction"],
        config["seed"],
        validation=False,
    )
    validation_data = TrackerDataset(
        args.data_dir,
        preprocess_config,
        config["validation_fraction"],
        config["seed"],
        validation=True,
    )
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    workers = config["num_workers"]
    if args.smoke:
        training_data.balanced_subset(128)
        validation_data.balanced_subset(64)
        epochs, batch_size, workers = 1, 16, min(workers, 2)

    training_loader = make_loader(
        training_data,
        batch_size,
        workers,
        config["balance_sources"],
        True,
        config["seed"],
    )
    validation_loader = make_loader(
        validation_data, batch_size, workers, False, False, config["seed"]
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
        output_dir = output_dir.parent / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    start_epoch, best_loss = 0, math.inf
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_validation_loss"]
    else:
        metrics_path.write_text("")

    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"device: {device}")
    print(f"parameters: {parameters:,}")
    print(
        f"training images: {len(training_data):,} {dict(training_data.source_counts)}"
    )
    print(
        f"validation images: {len(validation_data):,} {dict(validation_data.source_counts)}"
    )

    for epoch in range(start_epoch, epochs):
        training_data.set_epoch(epoch)
        learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = run_epoch(
            model,
            training_loader,
            device,
            config["loss"],
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

        improved = validation_metrics["loss"] < best_loss
        best_loss = min(best_loss, validation_metrics["loss"])
        save_checkpoint(
            output_dir / "last.pt",
            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            best_loss,
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
                best_loss,
                config,
                preprocess_config,
            )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
