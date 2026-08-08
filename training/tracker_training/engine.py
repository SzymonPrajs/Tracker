"""Small, explicit float32 training loop shared by local smoke experiments."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
from typing import Any

import numpy as np
import torch

from .decode import decode_heatmap
from .metrics import evaluate_centroids
from .synthetic import SyntheticScene
from .targets import TargetConfig, encode_targets


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def automatic_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collate_scenes(
    scenes: Sequence[SyntheticScene], target_config: TargetConfig = TargetConfig()
) -> tuple[torch.Tensor, dict[str, torch.Tensor], tuple[SyntheticScene, ...]]:
    images = torch.stack(
        [(scene.image.to(torch.float32) - 128.0) / 128.0 for scene in scenes]
    )
    encoded = [encode_targets(scene.heads, target_config) for scene in scenes]
    keys = (
        "heatmap",
        "offset",
        "size",
        "reg_mask",
        "ignore_mask",
        "collision_mask",
        "collision_count",
    )
    targets = {key: torch.stack([item[key] for item in encoded]) for key in keys}
    return images, targets, tuple(scenes)


def move_targets(
    targets: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in targets.items()}


def run_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    batches: Iterable[tuple[torch.Tensor, dict[str, torch.Tensor], tuple[SyntheticScene, ...]]],
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float = 5.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    sample_count = 0
    collision_count = 0
    started = time.perf_counter()
    for images, targets, scenes in batches:
        images = images.to(device)
        device_targets = move_targets(targets, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            losses = criterion(model(images), device_targets)
            if training:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
        count = len(scenes)
        sample_count += count
        collision_count += int(targets["collision_count"].sum())
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * count
    elapsed = time.perf_counter() - started
    if sample_count == 0:
        raise ValueError("epoch received no samples")
    result = {name: value / sample_count for name, value in totals.items()}
    result.update(
        {
            "samples": float(sample_count),
            "collisions": float(collision_count),
            "seconds": elapsed,
            "samples_per_second": sample_count / elapsed,
        }
    )
    return result


@torch.no_grad()
def evaluate_detection(
    model: torch.nn.Module,
    batches: Iterable[tuple[torch.Tensor, dict[str, torch.Tensor], tuple[SyntheticScene, ...]]],
    *,
    device: torch.device,
    score_threshold: float = 0.20,
    top_k: int = 10,
) -> dict[str, Any]:
    model.eval()
    predictions_by_image: list[np.ndarray] = []
    truth_by_image: list[np.ndarray] = []
    for images, _targets, scenes in batches:
        outputs = model(images.to(device)).detach().cpu()
        for output, scene in zip(outputs, scenes, strict=True):
            heatmap = output[0].sigmoid().numpy()
            offsets = output[1:3].numpy()
            sizes = output[3:5].numpy()
            peaks = decode_heatmap(
                heatmap,
                offsets=offsets,
                sizes=sizes,
                output_stride=4,
                top_k=top_k,
                score_threshold=score_threshold,
            )
            predictions_by_image.append(
                np.asarray([[peak.x, peak.y, peak.score] for peak in peaks], dtype=np.float64)
                .reshape(-1, 3)
            )
            truth_by_image.append(
                np.asarray(
                    [[head.center_x, head.center_y, head.width, head.height] for head in scene.heads],
                    dtype=np.float64,
                ).reshape(-1, 4)
            )
    return evaluate_centroids(predictions_by_image, truth_by_image)


def git_state(repo_root: Path) -> dict[str, Any]:
    def capture(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo_root, check=True, text=True, capture_output=True
        ).stdout.strip()

    status = capture("status", "--porcelain")
    return {
        "commit": capture("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_validation_loss: float,
    config: dict[str, Any],
    history: list[dict[str, Any]],
    repo_root: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "tracker-checkpoint-v1",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_validation_loss": best_validation_loss,
            "config": config,
            "history": history,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "git": git_state(repo_root),
        },
        path,
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
