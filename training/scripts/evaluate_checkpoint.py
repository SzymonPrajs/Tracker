#!/usr/bin/env python3
"""Evaluate a float checkpoint on deterministic synthetic validation scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

SCRIPT = Path(__file__).resolve()
TRAINING_ROOT = SCRIPT.parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from tracker_training.engine import automatic_device, collate_scenes, evaluate_detection  # noqa: E402
from tracker_training.model import HCDS31  # noqa: E402
from tracker_training.synthetic import SyntheticSceneDataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    device = automatic_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = HCDS31()
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    dataset = SyntheticSceneDataset(
        args.samples, seed=args.seed, head_count=None, max_heads=5
    )
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_scenes, num_workers=0)
    result = evaluate_detection(model, loader, device=device)
    print(json.dumps({"synthetic_only": True, "device": str(device), "metrics": result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
