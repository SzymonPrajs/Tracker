#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import tomllib
from pathlib import Path

import torch
from torch import nn

from common.motion import start_motion, update_motion
from common.temporal_model import model_from_config


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and inspect the temporal head-tracker model."
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config/temporal.toml")
    return parser.parse_args()


def convolution_macs(
    model: nn.Module, inputs: tuple[torch.Tensor, ...]
) -> tuple[tuple[torch.Tensor, ...], int]:
    total = 0

    def count(
        module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        nonlocal total
        assert isinstance(module, nn.Conv2d)
        output_values = output.numel()
        kernel_work = (
            math.prod(module.kernel_size) * module.in_channels // module.groups
        )
        total += output_values * kernel_work

    hooks = [
        module.register_forward_hook(count)
        for module in model.modules()
        if isinstance(module, nn.Conv2d)
    ]
    try:
        with torch.inference_mode():
            output = model(*inputs)
    finally:
        for hook in hooks:
            hook.remove()
    return output, total


def inspect_resolution(
    model: nn.Module, config: dict, width: int, height: int
) -> dict[str, int | tuple[int, ...]]:
    first = torch.rand(1, 1, height, width)
    second = torch.rand(1, 1, height, width)
    motion_state = start_motion(first)
    motion, _, _ = update_motion(
        second,
        motion_state,
        decay=1 - 2 ** -config["motion"]["decay_shift"],
        difference_clip=config["motion"]["difference_clip"],
        deadband=config["motion"]["deadband"],
        brightness_tile=config["motion"]["brightness_tile"],
    )
    state = model.start_state(1, height, width, dtype=second.dtype)
    prior = torch.zeros(1, 1, math.ceil(height / 4), math.ceil(width / 4))
    outputs, macs = convolution_macs(
        model, (second, motion, prior, state.fast, state.slow)
    )
    prediction, next_fast, _ = outputs

    motion_bytes = 3 * motion.shape[-2] * motion.shape[-1]
    prior_bytes = prior.shape[-2] * prior.shape[-1]
    recurrent_bytes = 2 * next_fast.numel()
    return {
        "output": tuple(prediction.shape),
        "state": tuple(next_fast.shape),
        "macs": macs,
        "motion_bytes": motion_bytes,
        "prior_bytes": prior_bytes,
        "recurrent_bytes": recurrent_bytes,
    }


def main() -> None:
    args = arguments()
    with args.config.open("rb") as file:
        config = tomllib.load(file)

    torch.manual_seed(0)
    model = model_from_config(config).eval()
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    deployed_conv = sum(
        module.weight.numel() + module.out_channels
        for module in model.modules()
        if isinstance(module, nn.Conv2d)
    )

    print(f"config: {args.config}")
    print(f"trainable PyTorch parameters: {trainable:,}")
    print(f"BN-folded convolution coefficients: {deployed_conv:,}")
    print("resolution  output          state           conv MAC   persistent INT8")
    for width, height in config["comparison_resolutions"]:
        result = inspect_resolution(model, config, width, height)
        persistent = (
            result["motion_bytes"] + result["prior_bytes"] + result["recurrent_bytes"]
        )
        print(
            f"{width:>3}x{height:<3}  {str(result['output']):<15} "
            f"{str(result['state']):<15} {result['macs'] / 1e6:8.3f}M "
            f"{persistent / 1000:8.2f} kB"
        )


if __name__ == "__main__":
    main()
