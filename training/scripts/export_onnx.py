#!/usr/bin/env python3
"""Export a fixed-shape, batch-one PyTorch tracker to ONNX opset 18."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

TRAINING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TRAINING_ROOT.parent
for import_root in (REPO_ROOT, TRAINING_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

ONNX_OPSET = 18


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required. Run training/scripts/setup_mac.sh first."
        ) from exc
    return torch


def make_smoke_model(seed: int = 7) -> Any:
    """Return a tiny ESP-DL-compatible model for toolchain smoke tests only."""
    torch = require_torch()
    nn = torch.nn
    torch.manual_seed(seed)

    class SmokeCentroidNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
            self.depthwise = nn.Conv2d(
                16, 16, kernel_size=3, stride=2, padding=1, groups=16
            )
            self.head = nn.Conv2d(16, 16, kernel_size=1)

        def forward(self, image: Any) -> Any:
            image = torch.relu(self.stem(image))
            image = torch.relu(self.depthwise(image))
            return self.head(image)

    return SmokeCentroidNet()


def load_model_factory(spec: str, kwargs: dict[str, Any]) -> Any:
    if ":" not in spec:
        raise ValueError("model factory must be MODULE:CALLABLE")
    module_name, callable_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), callable_name)
    model = factory(**kwargs)
    torch = require_torch()
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"{spec} did not return torch.nn.Module")
    return model


def load_checkpoint(model: Any, checkpoint: Path) -> None:
    torch = require_torch()
    import numpy as np

    # tracker-checkpoint-v1 includes NumPy's RNG ndarray. Permit only the exact
    # NumPy construction primitives needed to decode that value; never fall
    # back to unrestricted pickle loading when a safe load fails.
    numpy_rng_globals = [
        np._core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
    ]
    with torch.serialization.safe_globals(numpy_rng_globals):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)

    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must be a mapping containing a state dictionary")

    schema = payload.get("schema_version")
    if schema is not None:
        if not isinstance(schema, str) or schema != "tracker-checkpoint-v1":
            raise ValueError(f"unsupported checkpoint schema: {schema!r}")
        state_dict = payload.get("model")
    else:
        state_dict = payload
        for key in ("model_state_dict", "state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                state_dict = candidate
                break

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("checkpoint does not contain a non-empty model state dictionary")
    invalid_values = [
        name for name, value in state_dict.items() if not isinstance(value, torch.Tensor)
    ]
    if invalid_values:
        preview = ", ".join(str(name) for name in invalid_values[:3])
        raise ValueError(f"checkpoint state dictionary has non-tensor values: {preview}")
    model.load_state_dict(state_dict, strict=True)


def export_static_onnx(
    model: Any,
    output_path: Path,
    input_shape: tuple[int, int, int, int],
    *,
    seed: int = 7,
    write_reference: bool = False,
) -> dict[str, Path]:
    torch = require_torch()
    if len(input_shape) != 4 or input_shape[0] != 1:
        raise ValueError("input shape must be fixed NCHW with batch exactly 1")
    if any(dimension <= 0 for dimension in input_shape):
        raise ValueError("all input dimensions must be positive")

    torch.manual_seed(seed)
    model = model.to("cpu").eval()
    sample = torch.empty(input_shape, dtype=torch.float32).uniform_(-1.0, 1.0)
    with torch.no_grad():
        expected = model(sample)
    if not isinstance(expected, torch.Tensor):
        raise TypeError("tracker export requires exactly one tensor output")
    if expected.ndim != 4 or expected.shape[0] != 1:
        raise ValueError("tracker output must be fixed NCHW with batch exactly 1")

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model=model,
        args=(sample,),
        f=str(output_path),
        input_names=["image"],
        output_names=["head"],
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        dynamo=False,
    )

    result = {"onnx": output_path}
    if write_reference:
        import numpy as np

        input_path = output_path.with_suffix(".input.npy")
        output_reference_path = output_path.with_suffix(".output.npy")
        np.save(input_path, sample.detach().cpu().numpy())
        np.save(output_reference_path, expected.detach().cpu().numpy())
        result.update({"input": input_path, "output": output_reference_path})
    return result


def parse_shape(text: str) -> tuple[int, int, int, int]:
    try:
        shape = tuple(int(value) for value in text.lower().replace("x", ",").split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must be N,C,H,W") from exc
    if len(shape) != 4:
        raise argparse.ArgumentTypeError("shape must have four dimensions: N,C,H,W")
    return shape  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--model-factory",
        help="Python factory as MODULE:CALLABLE, e.g. tracker_training.model:HCDS31",
    )
    source.add_argument(
        "--smoke-model",
        action="store_true",
        help="export the tiny synthetic model, never a production checkpoint",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-kwargs", default="{}", help="factory kwargs as JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-shape", type=parse_shape, default=(1, 3, 160, 288))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--write-reference", action="store_true")
    args = parser.parse_args()

    try:
        kwargs = json.loads(args.model_kwargs)
    except json.JSONDecodeError as exc:
        parser.error(f"invalid --model-kwargs JSON: {exc}")
    if not isinstance(kwargs, dict):
        parser.error("--model-kwargs must decode to an object")

    model = (
        make_smoke_model(args.seed)
        if args.smoke_model
        else load_model_factory(args.model_factory, kwargs)
    )
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint)
    elif not args.smoke_model:
        parser.error("production model export requires --checkpoint")

    artifacts = export_static_onnx(
        model,
        args.output,
        args.input_shape,
        seed=args.seed,
        write_reference=args.write_reference,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
