#!/usr/bin/env python3
"""Export a trained checkpoint to ONNX for the board.

This is the deploy step after float training. INT8 / ESP-DL conversion is
added here later, once the camera path is known. The exported graph keeps
named streaming inputs so firmware can feed Y, P/N, prior, and state.
"""

from __future__ import annotations

import argparse
import math
import tomllib
from pathlib import Path

import torch

from common.model import model_from_config


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the trained tracker to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/temporal.toml")
    parser.add_argument(
        "--preprocess", type=Path, default=ROOT / "config/preprocess.toml"
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    with args.config.open("rb") as file:
        config = tomllib.load(file)
    with args.preprocess.open("rb") as file:
        preprocess = tomllib.load(file)

    width = preprocess["input_width"]
    height = preprocess["input_height"]
    model = model_from_config(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.prepare_for_export()

    luminance = torch.zeros(1, 1, height, width)
    motion = torch.zeros(1, 2, height // 2, width // 2)
    prior = torch.zeros(1, 1, math.ceil(height / 4), math.ceil(width / 4))
    state = model.start_state(1, height, width)
    output = args.output or args.checkpoint.with_suffix(".onnx")
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (luminance, motion, prior, state.fast, state.slow),
        output,
        input_names=["luminance", "motion", "prior", "fast_state", "slow_state"],
        output_names=["prediction", "next_fast", "next_slow"],
        opset_version=17,
        dynamo=False,
    )
    print(f"wrote {output}")
    print(f"input luminance: 1x1x{height}x{width}")
    print("INT8 / ESP-DL conversion is not in this script yet.")


if __name__ == "__main__":
    main()
