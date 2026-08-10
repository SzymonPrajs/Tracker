#!/usr/bin/env python3
"""Fast, restartable training for the fixed 160x288 head-centre detector."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from tracker_training.augmentation import augment_batch
from tracker_training.data import HeadDataset, PackedHeadDataset
from tracker_training.model import HCDS31


HEIGHT, WIDTH, STRIDE = 160, 288, 4


def scene(seed: int):
    """Small deterministic fixture retained for unit and installation tests."""
    rng = random.Random(seed)
    y, x = torch.meshgrid(torch.arange(HEIGHT), torch.arange(WIDTH), indexing="ij")
    image = torch.stack(((x + 3 * y) % 64 + 24, (2 * x + y) % 64 + 32, (x + y) % 48 + 40)).byte()
    heads = []
    for index in range(seed % 6):
        width, height = rng.randint(18, 40), rng.randint(18, 40)
        cx, cy = rng.randint(width, WIDTH - width), rng.randint(height, HEIGHT - height)
        mask = ((x - cx) / (width / 2)).square() + ((y - cy) / (height / 2)).square() <= 1
        image[:, mask] = torch.tensor([180, 100 + index * 10, 80], dtype=torch.uint8)[:, None]
        heads.append((cx, cy, width, height))
    return (image.float() - 128) / 128, heads


def targets(batch_heads):
    n, output_height, output_width = len(batch_heads), HEIGHT // STRIDE, WIDTH // STRIDE
    heat = torch.zeros(n, 1, output_height, output_width)
    offset = torch.zeros(n, 2, output_height, output_width)
    size = torch.zeros(n, 2, output_height, output_width)
    mask = torch.zeros(n, 1, output_height, output_width, dtype=torch.bool)
    yy, xx = torch.meshgrid(torch.arange(output_height), torch.arange(output_width), indexing="ij")
    for batch_index, heads in enumerate(batch_heads):
        for cx, cy, width, height in heads:
            u, v = cx / STRIDE, cy / STRIDE
            ix, iy = int(u), int(v)
            sigma = min(3.0, max(1.0, 0.15 * min(width, height) / STRIDE))
            gaussian = torch.exp(-((xx - ix).square() + (yy - iy).square()) / (2 * sigma * sigma))
            heat[batch_index, 0] = torch.maximum(heat[batch_index, 0], gaussian)
            if not mask[batch_index, 0, iy, ix]:
                offset[batch_index, :, iy, ix] = torch.tensor([u - ix - 0.5, v - iy - 0.5])
                size[batch_index, :, iy, ix] = torch.tensor([width / WIDTH, height / HEIGHT])
                mask[batch_index, 0, iy, ix] = True
    return heat, offset, size, mask


class DetectionLoss(nn.Module):
    """CenterNet focal heatmap loss plus sparse robust box regressions."""

    def __init__(
        self,
        heatmap_weight: float = 1.0,
        offset_weight: float = 1.0,
        size_weight: float = 0.2,
        padding_weight: float = 0.01,
        focal_alpha: float = 2.0,
        focal_beta: float = 4.0,
        regression_beta: float = 1 / 9,
    ):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight
        self.size_weight = size_weight
        self.padding_weight = padding_weight
        self.focal_alpha = focal_alpha
        self.focal_beta = focal_beta
        self.regression_beta = regression_beta

    def forward(self, prediction, target):
        heat, offset, size, mask = target
        # Loss arithmetic stays float32 even when convolutions use autocast.
        prediction = prediction.float()
        heat = heat.float()
        offset = offset.float()
        size = size.float()
        probability = prediction[:, :1].sigmoid().clamp(1e-6, 1 - 1e-6)
        positive = mask.bool()
        negative = ~positive
        positive_loss = torch.log(probability) * (1 - probability).pow(self.focal_alpha) * positive
        negative_loss = (
            torch.log1p(-probability)
            * probability.pow(self.focal_alpha)
            * (1 - heat).pow(self.focal_beta)
            * negative
        )
        positives = positive.sum().clamp_min(1)
        heatmap_loss = -(positive_loss.sum() + negative_loss.sum()) / positives
        selected = positive.expand(-1, 2, -1, -1).float()

        def regression_loss(value, truth):
            dense = F.smooth_l1_loss(value, truth, beta=self.regression_beta, reduction="none")
            return (dense * selected).sum() / selected.sum().clamp_min(1)

        offset_loss = regression_loss(prediction[:, 1:3], offset)
        size_loss = regression_loss(prediction[:, 3:5], size)
        padding_loss = prediction[:, 5:].square().mean()
        total = (
            self.heatmap_weight * heatmap_loss
            + self.offset_weight * offset_loss
            + self.size_weight * size_loss
            + self.padding_weight * padding_loss
        )
        return {
            "loss": total,
            "heatmap": heatmap_loss.detach(),
            "offset": offset_loss.detach(),
            "size": size_loss.detach(),
            "padding": padding_loss.detach(),
        }


def loss(prediction, target):
    """Compatibility wrapper used by the compact model unit test."""
    return DetectionLoss()(prediction, target)["loss"]


def batch(seeds):
    items = [scene(seed) for seed in seeds]
    return torch.stack([item[0] for item in items]), targets([item[1] for item in items])


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_dataset(data_root: Path, split: str, data_format: str):
    packed = data_format == "packed" or (
        data_format == "auto" and (data_root / split / "metadata.json").exists()
    )
    return PackedHeadDataset(data_root, split) if packed else HeadDataset(data_root, split)


def loader(dataset, args, training: bool, device: torch.device):
    sampler = None
    shuffle = False
    if training:
        mix = json.loads(args.mix_config.read_text())
        samples = args.epoch_samples or len(dataset)
        sampler = WeightedRandomSampler(
            dataset.sampling_weights(mix), num_samples=samples, replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        shuffle = False
    options = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "sampler": sampler,
        "shuffle": shuffle if sampler is None else False,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "drop_last": training and len(dataset) >= args.batch_size,
    }
    if args.workers:
        options.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)
    return DataLoader(**options)


def normalize(images: torch.Tensor, device: torch.device, channels_last: bool) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    if images.dtype == torch.uint8:
        images = images.float().sub_(128).div_(128)
    if channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    return images


def move_target(target, device: torch.device):
    return tuple(value.to(device, non_blocking=True) for value in target)


def augment(images, target, args):
    """Apply batch-vectorized, label-aware room-camera augmentation."""
    return augment_batch(images, target, args)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def run_epoch(
    model, batches, criterion, device, args, optimizer=None, scheduler=None, scaler=None,
    max_steps: int | None = None,
):
    training = optimizer is not None
    model.train(training)
    totals = defaultdict(float)
    examples = 0
    steps = 0
    optimizer_steps = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    started = time.perf_counter()
    context = contextlib.nullcontext() if training else torch.inference_mode()
    with context:
        step_limit = min(len(batches), max_steps) if max_steps is not None else len(batches)
        for step, (images, target) in enumerate(batches):
            if step >= step_limit:
                break
            images = normalize(images, device, args.channels_last)
            target = move_target(target, device)
            if training:
                images, target = augment(images, target, args)
            with autocast_context(device, args.amp_enabled):
                result = criterion(model(images), target)
                scaled_loss = result["loss"] / args.accumulate
            if training:
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                final_microbatch = (step + 1) % args.accumulate == 0 or step + 1 == step_limit
                if final_microbatch:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    if args.grad_clip:
                        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    if scheduler is not None:
                        scheduler.step()
            batch_size = images.shape[0]
            examples += batch_size
            steps += 1
            for key, value in result.items():
                totals[key] += float(value.detach()) * batch_size
            if training and args.log_interval and steps % args.log_interval == 0:
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(f"  step {steps}/{len(batches)} loss={totals['loss']/examples:.4f} {examples/elapsed:.1f} images/s", flush=True)
    synchronize(device)
    elapsed = max(time.perf_counter() - started, 1e-9)
    metrics = {key: value / max(examples, 1) for key, value in totals.items()}
    metrics.update(examples=examples, steps=steps, optimizer_steps=optimizer_steps, seconds=elapsed, images_per_second=examples / elapsed)
    return metrics


def optimizer_for(model, args):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2), fused=args.fused if torch.cuda.is_available() else False,
        )
    return torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=True,
    )


def scheduler_for(optimizer, args, steps_per_epoch):
    schedule_epochs = args.schedule_epochs or args.epochs
    total_steps = max(1, schedule_epochs * math.ceil(steps_per_epoch / args.accumulate))
    if args.scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=total_steps,
            pct_start=args.warmup_fraction, div_factor=10, final_div_factor=100,
        )
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr / 1000)
    return None


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best, args):
    state = {
        "epoch": epoch,
        "best_validation_loss": best,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    torch.save(state, path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--epochs", type=int, default=30)
    result.add_argument(
        "--schedule-epochs", type=int,
        help="fixed scheduler horizon for staged/resumed successive-halving runs",
    )
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--workers", type=int, default=0, help="0 is normally fastest for local memory maps on macOS")
    result.add_argument("--prefetch-factor", type=int, default=3)
    result.add_argument("--accumulate", type=int, default=1)
    result.add_argument("--epoch-samples", type=int, help="weighted samples per epoch; defaults to the full training count")
    result.add_argument("--data-root", type=Path, default=Path("data/packed/288x160"))
    result.add_argument("--data-format", choices=("auto", "packed", "canonical"), default="auto")
    result.add_argument("--mix-config", type=Path, default=Path("training/datasets/room_mix.json"))
    result.add_argument("--synthetic", action="store_true")
    result.add_argument("--device", default="auto", help="auto, mps, cuda, or cpu")
    result.add_argument("--amp", choices=("auto", "on", "off"), default="auto")
    result.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    result.add_argument(
        "--channels-last", action=argparse.BooleanOptionalAction, default=None,
        help="defaults on for CUDA and off for MPS, whose FPN backward requires contiguous NCHW",
    )
    result.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    result.add_argument("--lr", type=float, default=3e-3)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--beta1", type=float, default=0.9)
    result.add_argument("--beta2", type=float, default=0.99)
    result.add_argument("--momentum", type=float, default=0.9)
    result.add_argument("--fused", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--scheduler", choices=("onecycle", "cosine", "none"), default="onecycle")
    result.add_argument("--warmup-fraction", type=float, default=0.08)
    result.add_argument("--grad-clip", type=float, default=5.0)
    result.add_argument("--heatmap-weight", type=float, default=1.0)
    result.add_argument("--offset-weight", type=float, default=1.0)
    result.add_argument("--size-weight", type=float, default=0.2)
    result.add_argument("--padding-weight", type=float, default=0.01)
    result.add_argument("--focal-alpha", type=float, default=2.0)
    result.add_argument("--focal-beta", type=float, default=4.0)
    result.add_argument("--regression-beta", type=float, default=1 / 9)
    result.add_argument("--horizontal-flip", type=float, default=0.5)
    result.add_argument(
        "--augmentation-clean-probability", type=float, default=0.15,
        help="fraction of training images preserved exactly despite the stochastic mixture",
    )
    result.add_argument("--brightness", type=float, default=0.08)
    result.add_argument("--contrast", type=float, default=0.12)
    result.add_argument("--channel-jitter", type=float, default=0.06)
    result.add_argument("--exposure-probability", type=float, default=0.80)
    result.add_argument("--exposure-min-ev", type=float, default=-1.25)
    result.add_argument("--exposure-max-ev", type=float, default=0.75)
    result.add_argument("--low-light-probability", type=float, default=0.30)
    result.add_argument("--low-light-min-ev", type=float, default=-4.0)
    result.add_argument("--low-light-max-ev", type=float, default=-1.5)
    result.add_argument("--white-balance-probability", type=float, default=0.60)
    result.add_argument("--white-balance-magnitude", type=float, default=0.18)
    result.add_argument("--gamma-probability", type=float, default=0.40)
    result.add_argument("--gamma-magnitude", type=float, default=0.25)
    result.add_argument("--saturation-probability", type=float, default=0.45)
    result.add_argument("--saturation-magnitude", type=float, default=0.35)
    result.add_argument("--illumination-gradient-probability", type=float, default=0.35)
    result.add_argument("--illumination-gradient-strength", type=float, default=0.45)
    result.add_argument("--shadow-probability", type=float, default=0.25)
    result.add_argument("--shadow-strength", type=float, default=0.65)
    result.add_argument("--vignette-probability", type=float, default=0.30)
    result.add_argument("--vignette-strength", type=float, default=0.45)
    result.add_argument("--noise-probability", type=float, default=0.35)
    result.add_argument("--shot-noise", type=float, default=0.035)
    result.add_argument("--read-noise", type=float, default=0.012)
    result.add_argument("--blur-probability", type=float, default=0.12)
    result.add_argument("--fisheye-130-probability", type=float, default=0.18)
    result.add_argument("--fisheye-130-min", type=float, default=0.10)
    result.add_argument("--fisheye-130-max", type=float, default=0.25)
    result.add_argument("--fisheye-180-probability", type=float, default=0.14)
    result.add_argument("--fisheye-180-min", type=float, default=0.25)
    result.add_argument("--fisheye-180-max", type=float, default=0.45)
    result.add_argument("--telephoto-probability", type=float, default=0.18)
    result.add_argument("--telephoto-zoom-min", type=float, default=2.0)
    result.add_argument("--telephoto-zoom-max", type=float, default=6.0)
    result.add_argument("--val-every", type=int, default=1)
    result.add_argument("--patience", type=int, default=8)
    result.add_argument("--max-train-steps", type=int)
    result.add_argument("--max-val-steps", type=int)
    result.add_argument("--log-interval", type=int, default=100)
    result.add_argument("--seed", type=int, default=7)
    result.add_argument(
        "--initialize", type=Path,
        help="load model weights but start a fresh optimizer and scheduler",
    )
    result.add_argument("--resume", type=Path)
    result.add_argument(
        "--save-every-epoch", action="store_true",
        help="retain small model-only snapshots for post-hoc metric selection",
    )
    result.add_argument("--output", type=Path, default=Path("training/artifacts"))
    return result


def main():
    args = parser().parse_args()
    if args.initialize and args.resume:
        raise SystemExit("--initialize and --resume are mutually exclusive")
    if args.accumulate < 1:
        raise SystemExit("--accumulate must be at least one")
    if not 0 <= args.horizontal_flip <= 1:
        raise SystemExit("--horizontal-flip must be between zero and one")
    probabilities = {
        name: getattr(args, name)
        for name in (
            "horizontal_flip", "augmentation_clean_probability",
            "exposure_probability", "low_light_probability",
            "white_balance_probability", "gamma_probability", "saturation_probability",
            "illumination_gradient_probability", "shadow_probability",
            "vignette_probability", "noise_probability", "blur_probability",
            "fisheye_130_probability", "fisheye_180_probability", "telephoto_probability",
        )
    }
    if any(not 0 <= value <= 1 for value in probabilities.values()):
        raise SystemExit(f"augmentation probabilities must be between zero and one: {probabilities}")
    if sum((args.fisheye_130_probability, args.fisheye_180_probability, args.telephoto_probability)) > 1:
        raise SystemExit("lens augmentation probabilities must sum to at most one")
    magnitudes = (
        args.brightness, args.contrast, args.channel_jitter, args.white_balance_magnitude,
        args.gamma_magnitude, args.saturation_magnitude, args.illumination_gradient_strength,
        args.shadow_strength, args.vignette_strength, args.shot_noise, args.read_noise,
        args.fisheye_130_min, args.fisheye_130_max,
        args.fisheye_180_min, args.fisheye_180_max,
    )
    if min(magnitudes) < 0:
        raise SystemExit("augmentation magnitudes cannot be negative")
    if not 1 <= args.telephoto_zoom_min <= args.telephoto_zoom_max:
        raise SystemExit("telephoto zoom range must be ordered and at least one")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    if args.channels_last is None:
        args.channels_last = device.type == "cuda"
    args.amp_enabled = args.amp == "on" or (args.amp == "auto" and device.type == "cuda")
    if args.amp_enabled and device.type not in ("cuda", "cpu"):
        raise SystemExit("AMP is conservatively disabled on MPS; use --amp off")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = HCDS31().to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.head.bias.data[0] = -2.19
    if args.initialize:
        initial = torch.load(args.initialize, map_location="cpu", weights_only=False)
        model.load_state_dict(
            initial["model"] if isinstance(initial, dict) and "model" in initial else initial
        )
    criterion = DetectionLoss(
        heatmap_weight=args.heatmap_weight, offset_weight=args.offset_weight,
        size_weight=args.size_weight, padding_weight=args.padding_weight,
        focal_alpha=args.focal_alpha, focal_beta=args.focal_beta,
        regression_beta=args.regression_beta,
    )

    if args.synthetic:
        seeds = list(range(24))
        train_batches = [batch(seeds[start:start + args.batch_size]) for start in range(0, len(seeds), args.batch_size)]
        val_batches = train_batches
        calibration_images = torch.stack([scene(seed)[0] for seed in range(8)])
        dataset_label = "24 generated smoke-test scenes"
    else:
        train_dataset = make_dataset(args.data_root, "train", args.data_format)
        val_dataset = make_dataset(args.data_root, "val", args.data_format)
        train_batches = loader(train_dataset, args, True, device)
        val_batches = loader(val_dataset, args, False, device)
        calibration_images = torch.stack([train_dataset[index][0] for index in range(min(32, len(train_dataset)))])
        if calibration_images.dtype == torch.uint8:
            calibration_images = (calibration_images.float() - 128) / 128
        dataset_label = f"{len(train_dataset):,} train / {len(val_dataset):,} validation real scenes"

    optimizer = optimizer_for(model, args)
    effective_steps = min(len(train_batches), args.max_train_steps) if args.max_train_steps else len(train_batches)
    scheduler = scheduler_for(optimizer, args, effective_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if args.amp_enabled and device.type == "cuda" else None
    start_epoch = 0
    best = math.inf
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler and checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler and checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best = checkpoint["best_validation_loss"]

    checkpoint_model = model
    if args.compile:
        model = torch.compile(model)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "configuration.json").write_text(json.dumps({
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "device_resolved": str(device), "dataset": dataset_label,
    }, indent=2) + "\n")
    np.save(args.output / "calibration.npy", calibration_images.numpy())
    print(f"device={device} amp={args.amp_enabled} channels_last={args.channels_last} data={dataset_label}")
    print(f"batch={args.batch_size} accumulation={args.accumulate} effective_batch={args.batch_size * args.accumulate}")

    stale_epochs = 0
    for epoch in range(start_epoch, args.epochs):
        training = run_epoch(
            model, train_batches, criterion, device, args,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            max_steps=args.max_train_steps,
        )
        summary = f"epoch {epoch + 1:03d} train={training['loss']:.4f} {training['images_per_second']:.1f} images/s"
        validation = None
        if (epoch + 1) % args.val_every == 0:
            validation = run_epoch(
                model, val_batches, criterion, device, args,
                max_steps=args.max_val_steps,
            )
            summary += f" val={validation['loss']:.4f}"
            if validation["loss"] < best:
                best = validation["loss"]
                stale_epochs = 0
                torch.save(checkpoint_model.state_dict(), args.output / "model.pt")
            else:
                stale_epochs += 1
        print(summary, flush=True)
        save_checkpoint(args.output / "last.pt", checkpoint_model, optimizer, scheduler, scaler, epoch, best, args)
        if args.save_every_epoch:
            torch.save(checkpoint_model.state_dict(), args.output / f"epoch-{epoch + 1:03d}.pt")
        history_path = args.output / "history.jsonl"
        with history_path.open("a") as stream:
            stream.write(json.dumps({"epoch": epoch + 1, "train": training, "validation": validation}) + "\n")
        if args.patience and stale_epochs >= args.patience:
            print(f"early stopping after {stale_epochs} validation epochs without improvement")
            break
    if not (args.output / "model.pt").exists():
        torch.save(checkpoint_model.state_dict(), args.output / "model.pt")


if __name__ == "__main__":
    main()
