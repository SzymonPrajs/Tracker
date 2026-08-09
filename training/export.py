#!/usr/bin/env python3
"""Export the trained model as fixed-shape ONNX."""

import argparse
from pathlib import Path

import torch

from tracker_training.model import HCDS31, deployment_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    model = HCDS31()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
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
    print(args.output)


if __name__ == "__main__":
    main()
