#!/usr/bin/env python3
"""Build a deterministic two-sample converter smoke set from an export reference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sample = np.load(args.input, allow_pickle=False).astype(np.float32, copy=False)
    if sample.shape != (1, 3, 160, 288) or not np.all(np.isfinite(sample)):
        raise SystemExit(f"expected finite [1,3,160,288] input, found {sample.shape}")
    second = np.clip(sample * np.float32(0.9), -1.0, 1.0)
    calibration = np.concatenate((sample, second), axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, calibration, allow_pickle=False)
    print(f"calibration: {args.output.resolve()} shape={calibration.shape} synthetic_only=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
