#!/usr/bin/env python3
"""Quantize a checked, static ONNX tracker into an ESP32-P4 .espdl model."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "NumPy is required. Run training/scripts/setup_mac.sh --quantize-only first."
    ) from exc

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = SCRIPT_DIR.parent
for import_root in (SCRIPT_DIR, TRAINING_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

OUTPUT_ENCODING_METADATA_KEY = "tracker.output_encoding"
OUTPUT_EXPONENT_METADATA_KEY = "tracker.output_exponent"
DEFAULT_OUTPUT_ENCODING = "hcds31-output-q4-q7-v1"


def require_quantization_stack() -> tuple[Any, Any, Any, Any]:
    try:
        import onnx
        import torch
        from esp_ppq.api import espdl_quantize_onnx
        from esp_ppq.api.setting import QuantizationSettingFactory
    except ImportError as exc:
        raise SystemExit(
            "ESP-PPQ quantization dependencies are missing. "
            "Run training/scripts/setup_mac.sh --quantize-only first."
        ) from exc
    return onnx, torch, espdl_quantize_onnx, QuantizationSettingFactory


def fixed_input_shape(model_path: Path) -> tuple[int, int, int, int]:
    onnx, _, _, _ = require_quantization_stack()
    model = onnx.load(str(model_path))
    initializer_names = {item.name for item in model.graph.initializer}
    inputs = [item for item in model.graph.input if item.name not in initializer_names]
    if len(inputs) != 1:
        raise ValueError("quantization requires exactly one graph input")
    dimensions = []
    for dimension in inputs[0].type.tensor_type.shape.dim:
        if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
            raise ValueError("quantization requires a fully static input shape")
        dimensions.append(int(dimension.dim_value))
    shape = tuple(dimensions)
    if len(shape) != 4 or shape[0] != 1:
        raise ValueError(f"expected static batch-one NCHW input, found {shape}")
    return shape  # type: ignore[return-value]


def verify_onnx_output_encoding(model_path: Path, expected_encoding: str) -> int:
    try:
        import onnx
    except ImportError as exc:
        raise SystemExit(
            "ONNX is required. Run training/scripts/setup_mac.sh --quantize-only first."
        ) from exc
    from tracker_training.model import OUTPUT_ENCODING_ID, OUTPUT_EXPECTED_EXPONENT

    if expected_encoding != OUTPUT_ENCODING_ID:
        raise ValueError(f"unsupported output encoding: {expected_encoding!r}")
    model = onnx.load(str(model_path))
    metadata = {item.key: item.value for item in model.metadata_props}
    actual_encoding = metadata.get(OUTPUT_ENCODING_METADATA_KEY)
    if actual_encoding != expected_encoding:
        raise ValueError(
            f"ONNX output encoding is {actual_encoding!r}; expected {expected_encoding!r}"
        )
    exponent_text = metadata.get(OUTPUT_EXPONENT_METADATA_KEY)
    if exponent_text != str(OUTPUT_EXPECTED_EXPONENT):
        raise ValueError(
            f"ONNX output exponent contract is {exponent_text!r}; "
            f"expected {OUTPUT_EXPECTED_EXPONENT}"
        )
    return OUTPUT_EXPECTED_EXPONENT


def verify_quantized_output_exponent(graph: Any, expected_exponent: int) -> int:
    outputs = graph.outputs
    if set(outputs) != {"head"}:
        raise ValueError(f"expected one quantized graph output named 'head', found {list(outputs)}")
    config = outputs["head"].source_op_config
    if config is None or config.scale is None:
        raise ValueError("quantized head output has no scale")
    scale_value = config.scale.detach().cpu().numpy().reshape(-1)
    if scale_value.size != 1:
        raise ValueError("encoded head requires one per-tensor output scale")
    scale = float(scale_value[0])
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"quantized head output scale is invalid: {scale}")
    exponent = int(round(math.log2(scale)))
    if not math.isclose(scale, 2.0**exponent, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"quantized head output scale is not power-of-two: {scale}")
    if exponent != expected_exponent:
        raise ValueError(
            f"quantized head output exponent is {exponent} (scale {scale}); "
            f"encoding requires {expected_exponent}"
        )
    return exponent


def _split_samples(array: np.ndarray, expected: tuple[int, ...]) -> Iterable[np.ndarray]:
    if array.shape == expected[1:]:
        yield array[None, ...]
    elif array.shape == expected:
        yield array
    elif array.ndim == len(expected) and array.shape[1:] == expected[1:]:
        for index in range(array.shape[0]):
            yield array[index : index + 1]
    else:
        raise ValueError(f"calibration tensor shape {array.shape} does not match {expected}")


def load_calibration(path: Path, expected: tuple[int, ...]) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    files = sorted(path.glob("*.npy")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"no .npy calibration files found in {path}")
    for file_path in files:
        if file_path.suffix == ".npz":
            with np.load(file_path, allow_pickle=False) as archive:
                sources = [archive[key] for key in sorted(archive.files)]
        elif file_path.suffix == ".npy":
            sources = [np.load(file_path, allow_pickle=False)]
        else:
            raise ValueError("calibration input must be a .npy, .npz, or directory of .npy files")
        for source in sources:
            for sample in _split_samples(np.asarray(source), expected):
                if not np.isfinite(sample).all():
                    raise ValueError(f"non-finite calibration value in {file_path}")
                arrays.append(np.ascontiguousarray(sample, dtype=np.float32))
    return arrays


def synthetic_calibration(
    count: int, shape: tuple[int, ...], seed: int
) -> list[np.ndarray]:
    generator = np.random.default_rng(seed)
    return [
        generator.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        for _ in range(count)
    ]


def save_quantized_output(
    graph: Any,
    input_tensor: Any,
    output_path: Path,
    *,
    executor_type: Any | None = None,
) -> Path:
    """Run the PPQ graph once and save its sole dequantized output tensor."""
    if output_path.suffix != ".npy":
        raise ValueError("quantized output path must end in .npy")
    if executor_type is None:
        try:
            from esp_ppq.executor.torch import TorchExecutor
        except ImportError as exc:
            raise SystemExit(
                "ESP-PPQ TorchExecutor is missing. "
                "Run training/scripts/setup_mac.sh --quantize-only first."
            ) from exc
        executor_type = TorchExecutor

    executor = executor_type(graph=graph, device="cpu")
    outputs = executor(input_tensor)
    if isinstance(outputs, (list, tuple)):
        if len(outputs) != 1:
            raise ValueError(
                f"quantized output capture requires one graph output, found {len(outputs)}"
            )
        output = outputs[0]
    else:
        output = outputs
    if not hasattr(output, "detach"):
        raise TypeError("ESP-PPQ executor did not return a tensor")

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = output.detach().cpu().numpy()
    np.save(output_path, array, allow_pickle=False)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    calibration = parser.add_mutually_exclusive_group(required=True)
    calibration.add_argument("--calibration", type=Path)
    calibration.add_argument(
        "--synthetic-calibration",
        type=int,
        metavar="COUNT",
        help="toolchain smoke test only; never use for an accuracy claim",
    )
    parser.add_argument("--calibration-steps", type=int, default=32)
    parser.add_argument("--bits", type=int, choices=(8, 16), default=8)
    parser.add_argument(
        "--int16-op",
        action="append",
        default=[],
        help="exact ONNX node name for an INT16 island in an otherwise INT8 graph",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-encoding",
        choices=(DEFAULT_OUTPUT_ENCODING,),
        default=DEFAULT_OUTPUT_ENCODING,
        help="required ONNX output encoding and post-quantization exponent contract",
    )
    parser.add_argument("--error-report", action="store_true")
    parser.add_argument(
        "--quantized-output-npy",
        type=Path,
        help="save ESP-PPQ output for calibration sample zero (PC simulation only)",
    )
    parser.add_argument("--no-test-values", action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args()

    if args.output.suffix != ".espdl":
        parser.error("--output must end in .espdl")
    if (
        args.quantized_output_npy is not None
        and args.quantized_output_npy.suffix != ".npy"
    ):
        parser.error("--quantized-output-npy must end in .npy")
    if args.calibration_steps < 2:
        parser.error("--calibration-steps must be at least 2 for ESP-PPQ")
    if args.synthetic_calibration is not None and args.synthetic_calibration <= 0:
        parser.error("--synthetic-calibration must be positive")
    if args.int16_op and args.bits != 8:
        parser.error("--int16-op is only meaningful for an INT8 base graph")

    expected_output_exponent = verify_onnx_output_encoding(
        args.model, args.output_encoding
    )

    if not args.skip_check:
        from check_onnx import inspect_model

        inspect_model(args.model)
    input_shape = fixed_input_shape(args.model)
    arrays = (
        load_calibration(args.calibration, input_shape)
        if args.calibration is not None
        else synthetic_calibration(args.synthetic_calibration, input_shape, args.seed)
    )
    if args.calibration_steps > len(arrays):
        parser.error(
            f"requested {args.calibration_steps} calibration steps but only {len(arrays)} samples exist"
        )

    _, torch, quantize, setting_factory = require_quantization_stack()
    torch.manual_seed(args.seed)
    tensors = [torch.from_numpy(array) for array in arrays]
    setting = setting_factory.espdl_setting(num_of_bits=args.bits)
    if args.int16_op:
        from esp_ppq.core import TargetPlatform

        for operation in args.int16_op:
            setting.dispatching_table.append(operation, TargetPlatform.ESPDL_INT16)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tracker-espdl-") as temporary:
        # ESP-PPQ simplifies and rewrites its ONNX input. Preserve the reviewed source.
        quantization_input = Path(temporary) / args.model.name
        shutil.copy2(args.model, quantization_input)
        quantization_output = Path(temporary) / args.output.name
        graph = quantize(
            onnx_import_file=str(quantization_input),
            espdl_export_file=str(quantization_output),
            calib_dataloader=tensors,
            calib_steps=args.calibration_steps,
            input_shape=list(input_shape),
            inputs=[tensors[0]],
            target="esp32p4",
            num_of_bits=args.bits,
            collate_fn=lambda sample: sample.to(dtype=torch.float32, device="cpu"),
            setting=setting,
            device="cpu",
            error_report=args.error_report,
            export_config=True,
            export_test_values=not args.no_test_values,
            metadata_props={
                OUTPUT_ENCODING_METADATA_KEY: args.output_encoding,
                OUTPUT_EXPONENT_METADATA_KEY: str(expected_output_exponent),
            },
            verbose=1,
        )
        output_exponent = verify_quantized_output_exponent(
            graph, expected_output_exponent
        )
        for suffix in (".espdl", ".json", ".info"):
            source = quantization_output.with_suffix(suffix)
            if source.exists():
                shutil.copy2(source, args.output.with_suffix(suffix))
    counts: dict[str, int] = {}
    for operation in graph.operations.values():
        platform = str(operation.platform).split(".")[-1]
        counts[platform] = counts.get(platform, 0) + 1
    print(f"espdl: {args.output.resolve()}")
    print(f"quantized platforms: {counts}")
    print(f"output encoding: {args.output_encoding}, exponent: {output_exponent}")
    if args.quantized_output_npy is not None:
        captured = save_quantized_output(
            graph, tensors[0], args.quantized_output_npy
        )
        print(f"quantized output (PPQ PC simulation): {captured}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
