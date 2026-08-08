#!/usr/bin/env python3
"""Train HC-DS31 end to end on deterministic geometric pseudo-heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import torch
from torch.utils.data import DataLoader

SCRIPT = Path(__file__).resolve()
TRAINING_ROOT = SCRIPT.parents[1]
REPO_ROOT = SCRIPT.parents[2]
sys.path.insert(0, str(TRAINING_ROOT))

from tracker_training.engine import (  # noqa: E402
    automatic_device,
    collate_scenes,
    evaluate_detection,
    run_epoch,
    save_checkpoint,
    seed_everything,
    write_json,
)
from tracker_training.losses import HCDS31Loss  # noqa: E402
from tracker_training.model import (  # noqa: E402
    HCDS31,
    OUTPUT_ENCODING_ID,
    OUTPUT_EXPECTED_EXPONENT,
)
from tracker_training.synthetic import SyntheticSceneDataset  # noqa: E402
from tracker_training.targets import TargetConfig  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=TRAINING_ROOT / "configs/hcds31.json")
    parser.add_argument("--output", type=Path, default=TRAINING_ROOT / "artifacts/synthetic-smoke")
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke", action="store_true", help="run a small but real optimization")
    return parser.parse_args()


def validate_training_config(config: dict[str, object]) -> None:
    """Fail if declarative checkpoint metadata differs from executed code."""
    if config.get("schema_version") != "tracker-training-config-v1":
        raise ValueError("unsupported training config schema_version")
    target = TargetConfig()
    expected_sections: dict[str, dict[str, object]] = {
        "model": {
            "name": "hcds31",
            "input_height": target.input_height,
            "input_width": target.input_width,
            "input_channels": 3,
            "output_stride": target.output_stride,
            "output_channels": 16,
            "logical_channels": 5,
            "normalization": "rgb_u8_minus_128_div_128",
            "output_encoding": OUTPUT_ENCODING_ID,
            "required_output_exponent": OUTPUT_EXPECTED_EXPONENT,
        },
        "targets": {
            "mode": "all_heads",
            "gaussian_combine": "max",
            "sigma_box_scale": target.sigma_scale,
            "sigma_min": target.sigma_min,
            "sigma_max": target.sigma_max,
            "collision_owner": "visibility_area_stable_index",
        },
        "optimizer": {
            "name": "adamw",
            "warmup_epochs": 0,
            "schedule": "cosine",
        },
    }
    for section_name, expected in expected_sections.items():
        section = config.get(section_name)
        if not isinstance(section, dict):
            raise ValueError(f"training config section {section_name!r} is missing")
        for key, expected_value in expected.items():
            actual = section.get(key)
            if actual != expected_value:
                raise ValueError(
                    f"training config {section_name}.{key}={actual!r}; "
                    f"implementation requires {expected_value!r}"
                )


def make_loader(
    dataset: SyntheticSceneDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        collate_fn=collate_scenes,
    )


def main() -> int:
    args = arguments()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("training config root must be an object")
    validate_training_config(config)
    smoke = config["synthetic_smoke"]
    seed = int(smoke["seed"])
    epochs = args.epochs or int(smoke["epochs"])
    train_samples = int(smoke["train_samples"])
    validation_samples = int(smoke["validation_samples"])
    batch_size = int(smoke["batch_size"])
    if args.smoke:
        epochs = args.epochs or 2
        train_samples = min(train_samples, 24)
        validation_samples = min(validation_samples, 8)
        batch_size = min(batch_size, 4)
    if epochs <= 0:
        raise SystemExit("--epochs must be positive")

    seed_everything(seed)
    device = automatic_device() if args.device == "auto" else torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is not available")

    maximum_heads = int(smoke["maximum_heads"])
    train_data = SyntheticSceneDataset(
        train_samples, seed=seed, head_count=None, max_heads=maximum_heads
    )
    validation_data = SyntheticSceneDataset(
        validation_samples, seed=seed + 1, head_count=None, max_heads=maximum_heads
    )
    train_loader = make_loader(train_data, batch_size=batch_size, shuffle=True, seed=seed)
    validation_loader = make_loader(
        validation_data, batch_size=batch_size, shuffle=False, seed=seed + 1
    )

    model = HCDS31().to(device)
    with torch.no_grad():
        model.head.bias.zero_()
        model.head.bias[0] = -2.19
    criterion = HCDS31Loss.from_config(config["loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best = float("inf")
    print(
        json.dumps(
            {
                "event": "start",
                "device": str(device),
                "torch": torch.__version__,
                "train_samples": train_samples,
                "validation_samples": validation_samples,
                "batch_size": batch_size,
                "epochs": epochs,
                "synthetic_only": True,
            },
            sort_keys=True,
        )
    )
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model,
            criterion,
            train_loader,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=float(config["optimizer"]["gradient_clip_norm"]),
        )
        validation_metrics = run_epoch(
            model, criterion, validation_loader, device=device, optimizer=None
        )
        scheduler.step()
        record: dict[str, object] = {
            "event": "epoch",
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True))
        improved = validation_metrics["loss"] < best
        best = min(best, validation_metrics["loss"])
        save_checkpoint(
            output / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_validation_loss=best,
            config=config,
            history=history,
            repo_root=REPO_ROOT,
        )
        torch.save(model.state_dict(), output / "last-state.pt")
        if improved:
            shutil.copy2(output / "last.pt", output / "best.pt")
            shutil.copy2(output / "last-state.pt", output / "best-state.pt")

    detection = evaluate_detection(model, validation_loader, device=device)
    summary = {
        "schema_version": "tracker-synthetic-training-v1",
        "synthetic_only": True,
        "device": str(device),
        "epochs": epochs,
        "best_validation_loss": best,
        "history": history,
        "detection": detection,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({"event": "complete", "output": str(output), "synthetic_only": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
