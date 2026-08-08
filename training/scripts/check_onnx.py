#!/usr/bin/env python3
"""Validate the static ESP-DL subset and optional float-reference parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_ALLOWED_OPS = frozenset(
    {
        "Add",
        "AveragePool",
        "BatchNormalization",
        "Clip",
        "Concat",
        "Constant",
        "Conv",
        "Div",
        "Flatten",
        "Gemm",
        "GlobalAveragePool",
        "Identity",
        "LeakyRelu",
        "MatMul",
        "MaxPool",
        "Mul",
        "PRelu",
        "Relu",
        "Reshape",
        "Resize",
        "Sigmoid",
        "Slice",
        "Split",
        "Sub",
        "Transpose",
        "Unsqueeze",
    }
)


def require_onnx() -> tuple[Any, Any]:
    try:
        import onnx
        import onnxruntime
    except ImportError as exc:
        raise SystemExit(
            "ONNX and ONNX Runtime are required. Run training/scripts/setup_mac.sh first."
        ) from exc
    return onnx, onnxruntime


def _attribute(node: Any, name: str, default: Any, onnx: Any) -> Any:
    for attribute in node.attribute:
        if attribute.name == name:
            return onnx.helper.get_attribute_value(attribute)
    return default


def _fixed_shape(value_info: Any) -> tuple[int, ...]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        raise ValueError(f"{value_info.name}: tensor shape is missing")
    dimensions = []
    for dimension in tensor_type.shape.dim:
        if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
            raise ValueError(f"{value_info.name}: dynamic dimensions are not supported")
        dimensions.append(int(dimension.dim_value))
    return tuple(dimensions)


def inspect_model(
    model_path: Path,
    *,
    allowed_ops: frozenset[str] = DEFAULT_ALLOWED_OPS,
    required_output_channels: int = 16,
) -> dict[str, Any]:
    onnx, _ = require_onnx()
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)

    default_opsets = [item.version for item in model.opset_import if item.domain in ("", "ai.onnx")]
    if default_opsets != [18]:
        raise ValueError(f"expected ONNX opset 18, found {default_opsets}")
    custom_domains = [item.domain for item in model.opset_import if item.domain not in ("", "ai.onnx")]
    if custom_domains:
        raise ValueError(f"custom operator domains are not supported: {custom_domains}")

    initializer_names = {item.name for item in model.graph.initializer}
    graph_inputs = [item for item in model.graph.input if item.name not in initializer_names]
    if len(graph_inputs) != 1 or len(model.graph.output) != 1:
        raise ValueError("tracker graph must have exactly one input and one output")
    input_shape = _fixed_shape(graph_inputs[0])
    output_shape = _fixed_shape(model.graph.output[0])
    if len(input_shape) != 4 or input_shape[0] != 1:
        raise ValueError(f"input must be static batch-one NCHW, found {input_shape}")
    if len(output_shape) != 4 or output_shape[0] != 1:
        raise ValueError(f"output must be static batch-one NCHW, found {output_shape}")
    if required_output_channels and output_shape[1] != required_output_channels:
        raise ValueError(
            f"output must have {required_output_channels} physical channels, found {output_shape[1]}"
        )

    value_shapes: dict[str, tuple[int, ...]] = {}
    for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
        try:
            value_shapes[value.name] = _fixed_shape(value)
        except ValueError:
            pass

    unsupported = sorted({node.op_type for node in model.graph.node} - allowed_ops)
    if unsupported:
        raise ValueError(f"operators outside the ESP-DL tracker allowlist: {unsupported}")

    warnings = []
    for index, node in enumerate(model.graph.node):
        label = node.name or f"{node.op_type}[{index}]"
        if node.op_type == "Conv":
            group = int(_attribute(node, "group", 1, onnx))
            channels = value_shapes.get(node.input[0], (0, 0))[1]
            if group != 1 and group != channels:
                raise ValueError(
                    f"{label}: ESP-DL tracker Conv groups must be 1 or input channels; "
                    f"found group={group}, channels={channels}"
                )
        elif node.op_type in {"AveragePool", "MaxPool"}:
            dilations = tuple(_attribute(node, "dilations", (), onnx))
            if dilations and any(value != 1 for value in dilations):
                raise ValueError(f"{label}: dilated pooling is not supported")
        elif node.op_type == "Resize":
            mode = _attribute(node, "mode", b"nearest", onnx)
            if isinstance(mode, bytes):
                mode = mode.decode("ascii")
            if mode not in {"nearest", "linear"}:
                raise ValueError(f"{label}: Resize mode {mode!r} is not supported")
            if int(_attribute(node, "antialias", 0, onnx)) != 0:
                raise ValueError(f"{label}: antialiased Resize is not supported")
            warnings.append(f"{label}: Resize may be bandwidth-sensitive on ESP32-P4")
        elif node.op_type == "Transpose":
            warnings.append(f"{label}: explicit Transpose may prevent an efficient NHWC path")

    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return {
        "model": str(model_path.resolve()),
        "opset": 18,
        "input": {"name": graph_inputs[0].name, "shape": input_shape},
        "output": {"name": model.graph.output[0].name, "shape": output_shape},
        "operators": dict(sorted(counts.items())),
        "nodes": [node.name or f"{node.op_type}[{index}]" for index, node in enumerate(model.graph.node)],
        "warnings": warnings,
    }


def check_parity(
    model_path: Path,
    input_path: Path,
    expected_path: Path,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> dict[str, float]:
    import numpy as np

    _, onnxruntime = require_onnx()
    session = onnxruntime.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("parity check requires exactly one input and one output")
    input_array = np.load(input_path, allow_pickle=False).astype(np.float32, copy=False)
    expected = np.load(expected_path, allow_pickle=False).astype(np.float32, copy=False)
    actual = session.run(None, {session.get_inputs()[0].name: input_array})[0]
    if actual.shape != expected.shape:
        raise ValueError(f"parity shape mismatch: ONNX {actual.shape}, reference {expected.shape}")
    difference = np.abs(actual - expected)
    max_abs = float(difference.max(initial=0.0))
    denominator = np.maximum(np.abs(expected), 1e-12)
    max_rel = float((difference / denominator).max(initial=0.0))
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise ValueError(
            f"ONNX parity failed: max_abs={max_abs:.8g}, max_rel={max_rel:.8g}, "
            f"rtol={rtol}, atol={atol}"
        )
    return {"max_abs_error": max_abs, "max_relative_error": max_rel}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--input-npy", type=Path)
    parser.add_argument("--expected-npy", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--require-output-channels", type=int, default=16)
    args = parser.parse_args()
    if (args.input_npy is None) != (args.expected_npy is None):
        parser.error("--input-npy and --expected-npy must be supplied together")

    summary = inspect_model(
        args.model, required_output_channels=args.require_output_channels
    )
    if args.input_npy is not None:
        summary["parity"] = check_parity(
            args.model,
            args.input_npy,
            args.expected_npy,
            rtol=args.rtol,
            atol=args.atol,
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
