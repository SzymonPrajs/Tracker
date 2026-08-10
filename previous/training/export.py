#!/usr/bin/env python3
"""Export the trained model as fixed-shape ONNX."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from tracker_training.model import HCDS31, deployment_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    model = HCDS31()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model = deployment_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 160, 288),
        args.output,
        input_names=["image"],
        output_names=["head"],
        opset_version=18,
        dynamo=False,
    )
    metadata = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "onnx_sha256": sha256(args.output),
        "input": {"name": "image", "shape_nchw": [1, 3, 160, 288]},
        "output": {"name": "head", "shape_nchw": [1, 16, 40, 72]},
        "deployment_channel_gains": [2, 16, 16, 16, 16] + [0] * 11,
        "opset": 18,
    }
    args.output.with_suffix(".export.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
