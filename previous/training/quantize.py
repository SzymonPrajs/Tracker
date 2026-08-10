#!/usr/bin/env python3
"""Calibrate, export, and validate an ESP32-P4 INT8 ESP-DL model."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import esp_ppq.lib as PFL
from esp_ppq import QuantizationSettingFactory, TorchExecutor
from esp_ppq.api import espdl_quantize_onnx
from esp_ppq.api.espdl_interface import generate_test_value, get_target_platform

from evaluate import average_precision, decode, row_ground_truth
from tracker_training.data import PackedHeadDataset
from tracker_training.quantization import (
    CalibrationLoader,
    representative_indices,
    scale_exponent,
    tensor_scale,
    write_c_header,
)


WIDTH, HEIGHT, CHANNELS, STRIDE = 288, 160, 3, 4
OUTPUT_CHANNELS = 16
DEPLOYMENT_GAINS = torch.tensor([2, 16, 16, 16, 16] + [1] * 11).view(1, 16, 1, 1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nchw(output: torch.Tensor) -> torch.Tensor:
    if output.shape == (1, OUTPUT_CHANNELS, HEIGHT // STRIDE, WIDTH // STRIDE):
        return output
    if output.shape == (1, HEIGHT // STRIDE, WIDTH // STRIDE, OUTPUT_CHANNELS):
        return output.permute(0, 3, 1, 2)
    raise RuntimeError(f"unexpected model output shape: {tuple(output.shape)}")


def metrics(predictions, ground_truth: dict[int, np.ndarray]) -> dict:
    ap30 = average_precision(predictions, ground_truth, 0.30)
    ap50 = average_precision(predictions, ground_truth, 0.50)
    room_ids = {index for index, boxes in ground_truth.items() if len(boxes) <= 4}
    room_truth = {index: boxes for index, boxes in ground_truth.items() if index in room_ids}
    room_predictions = [item for item in predictions if item[0] in room_ids]
    room_ap30 = average_precision(room_predictions, room_truth, 0.30)
    room_ap50 = average_precision(room_predictions, room_truth, 0.50)
    mean_ap = (ap30 + ap50) / 2
    room_mean_ap = (room_ap30 + room_ap50) / 2
    return {
        "ap30": ap30,
        "ap50": ap50,
        "map30_50": mean_ap,
        "room_ap30": room_ap30,
        "room_ap50": room_ap50,
        "room_map30_50": room_mean_ap,
        "selection_score": 0.75 * room_mean_ap + 0.25 * mean_ap,
    }


def evaluate_quantization(
    model_path: Path,
    graph,
    dataset: PackedHeadDataset,
    indices: list[int],
    maximum_detections: int,
    output_scale: float,
) -> dict:
    """Compare ONNX float and PPQ's quantized execution on identical held-out scenes."""
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    executor = TorchExecutor(graph=graph, fp16_mode=False, device="cpu")
    float_predictions = []
    int8_predictions = []
    ground_truth = {}
    absolute_error = torch.zeros(5, dtype=torch.float64)
    elements = 0
    saturated = 0
    quantized_elements = 0
    started = time.perf_counter()

    for image_index, dataset_index in enumerate(indices):
        image, _ = dataset[dataset_index]
        value = image[None].float().sub_(128).div_(128)
        float_deploy = torch.from_numpy(session.run(None, {"image": value.numpy()})[0])
        quant_deploy = nchw(executor.forward(value)[0].detach().cpu())
        float_deploy = nchw(float_deploy)
        error = (float_deploy[:, :5] - quant_deploy[:, :5]).abs().double()
        absolute_error += error.sum(dim=(0, 2, 3)).reshape(-1)
        elements += error.shape[0] * error.shape[2] * error.shape[3]
        quantized = torch.round(quant_deploy / output_scale)
        saturated += int(((quantized <= -128) | (quantized >= 127)).sum())
        quantized_elements += quantized.numel()

        float_scores, float_boxes = decode(float_deploy / DEPLOYMENT_GAINS, maximum_detections)
        int8_scores, int8_boxes = decode(quant_deploy / DEPLOYMENT_GAINS, maximum_detections)
        ground_truth[image_index] = row_ground_truth(dataset.rows[dataset_index])
        float_predictions.extend(
            (image_index, float(score), box)
            for score, box in zip(float_scores[0].numpy(), float_boxes[0].numpy(), strict=True)
        )
        int8_predictions.extend(
            (image_index, float(score), box)
            for score, box in zip(int8_scores[0].numpy(), int8_boxes[0].numpy(), strict=True)
        )

    float_metrics = metrics(float_predictions, ground_truth)
    int8_metrics = metrics(int8_predictions, ground_truth)
    return {
        "split": "val",
        "images": len(indices),
        "indices": indices,
        "ground_truth_heads": sum(len(value) for value in ground_truth.values()),
        "maximum_detections": maximum_detections,
        "float": float_metrics,
        "int8_simulated": int8_metrics,
        "delta_int8_minus_float": {
            key: int8_metrics[key] - float_metrics[key] for key in float_metrics
        },
        "mean_absolute_error_deployment_channels": {
            name: float(absolute_error[index] / elements)
            for index, name in enumerate(("confidence", "offset_x", "offset_y", "width", "height"))
        },
        "output_saturation_fraction_all_channels": saturated / max(1, quantized_elements),
        "seconds": time.perf_counter() - started,
    }


def graph_scale_metadata(graph) -> tuple[float, float]:
    input_config = graph.inputs["image"].dest_op_configs[0]
    output_config = graph.outputs["head"].source_op_config
    return tensor_scale(input_config.scale), tensor_scale(output_config.scale)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("model", type=Path, help="fixed-shape deployment ONNX")
    result.add_argument("output", type=Path, help="output .espdl path")
    result.add_argument("--data-root", type=Path, default=Path("data/packed/288x160"))
    result.add_argument("--calibration-samples", type=int, default=256)
    result.add_argument("--evaluation-samples", type=int, default=256)
    result.add_argument("--maximum-detections", type=int, default=20)
    result.add_argument("--seed", type=int, default=20260809)
    result.add_argument("--header", type=Path)
    result.add_argument(
        "--calibration-algorithm",
        choices=("kl", "percentile", "mse", "minmax"),
        default="kl",
    )
    result.add_argument("--tqt-steps", type=int, default=0)
    result.add_argument("--tqt-learning-rate", type=float, default=1e-5)
    result.add_argument(
        "--tqt-train-scales",
        action="store_true",
        help="allow TQT to alter power-of-two scales (incompatible if it changes the -7 input)",
    )
    result.add_argument("--weight-equalization", action="store_true")
    result.add_argument("--equalization-iterations", type=int, default=10)
    result.add_argument("--equalization-threshold", type=float, default=0.5)
    result.add_argument("--ppq-error-report", action="store_true")
    result.add_argument("--skip-evaluation", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    train = PackedHeadDataset(args.data_root, "train")
    calibration_indices = representative_indices(train.rows, args.calibration_samples, args.seed)
    calibration = CalibrationLoader(train, calibration_indices)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    setting = QuantizationSettingFactory.espdl_setting(num_of_bits=8)
    setting.quantize_activation_setting.calib_algorithm = args.calibration_algorithm
    if args.weight_equalization:
        setting.equalization = True
        setting.equalization_setting.iterations = args.equalization_iterations
        setting.equalization_setting.value_threshold = args.equalization_threshold
        setting.equalization_setting.opt_level = 2
    if args.tqt_steps:
        setting.tqt_optimization = True
        setting.tqt_optimization_setting.steps = args.tqt_steps
        setting.tqt_optimization_setting.lr = args.tqt_learning_rate
        setting.tqt_optimization_setting.collecting_device = "cpu"
        setting.tqt_optimization_setting.is_scale_trainable = args.tqt_train_scales
        # Do not let TQT trade away the camera's exact uint8-to-int8 mapping.
        # Every later Conv remains trainable, including the detection head.
        source_graph = onnx.load(str(args.model)).graph
        setting.tqt_optimization_setting.interested_layers = [
            node.name
            for node in source_graph.node
            if node.op_type == "Conv" and node.name != "/stem/conv/Conv"
        ]

    print(
        f"quantizing {args.model} for esp32p4 int8 with "
        f"{len(calibration_indices)} representative train scenes",
        flush=True,
    )
    graph = espdl_quantize_onnx(
        onnx_import_file=str(args.model),
        espdl_export_file=str(args.output),
        calib_dataloader=calibration,
        calib_steps=len(calibration),
        input_shape=[1, CHANNELS, HEIGHT, WIDTH],
        inputs=[calibration[0]],
        target="esp32p4",
        num_of_bits=8,
        setting=setting,
        collate_fn=lambda value: value.float(),
        device="cpu",
        error_report=args.ppq_error_report,
        skip_export=True,
        verbose=0,
    )
    input_scale, output_scale = graph_scale_metadata(graph)
    input_exponent = scale_exponent(input_scale)
    output_exponent = scale_exponent(output_scale)
    if input_exponent != -7:
        raise RuntimeError(
            f"calibration selected input exponent {input_exponent}, expected -7; "
            "increase or improve representative calibration data instead of mutating "
            "the calibrated graph"
        )
    values_for_test = generate_test_value(graph, "cpu", [calibration[0]], ["head"])
    PFL.Exporter(platform=get_target_platform("esp32p4", 8)).export(
        file_path=str(args.output),
        graph=graph,
        values_for_test=values_for_test,
        export_config=True,
        metadata_props={
            "architecture": "HCDS31",
            "input_encoding": "RGB int8=(uint8-128), exponent=-7",
            "output_encoding": "HWC16 fixed-point deployment head",
        },
    )

    metadata = {
        "schema_version": 1,
        "target": "esp32p4",
        "quantization": "symmetric power-of-two int8",
        "calibration_algorithm": args.calibration_algorithm,
        "tqt": {
            "enabled": bool(args.tqt_steps),
            "steps": args.tqt_steps,
            "learning_rate": args.tqt_learning_rate,
            "scale_trainable": args.tqt_train_scales,
        },
        "weight_equalization": {
            "enabled": args.weight_equalization,
            "iterations": args.equalization_iterations,
            "value_threshold": args.equalization_threshold,
            "optimization_level": 2,
        },
        "esp_ppq_version": version("esp-ppq"),
        "model": str(args.model),
        "model_sha256": sha256(args.model),
        "artifact": str(args.output),
        "input": {
            "name": "image",
            "layout": "HWC on ESP-DL; ONNX source is NCHW",
            "width": WIDTH,
            "height": HEIGHT,
            "channels": CHANNELS,
            "scale": input_scale,
            "exponent": input_exponent,
            "camera_conversion": "int8 = uint8 ^ 0x80",
        },
        "output": {
            "name": "head",
            "layout": "HWC16",
            "width": WIDTH // STRIDE,
            "height": HEIGHT // STRIDE,
            "channels": OUTPUT_CHANNELS,
            "scale": output_scale,
            "exponent": output_exponent,
            "offset_q": 4 - output_exponent,
            "channel_gains": [2, 16, 16, 16, 16] + [0] * 11,
        },
        "calibration": {
            "split": "train",
            "strategy": "round-robin source x head-count x largest-head-size strata",
            "seed": args.seed,
            "samples": len(calibration_indices),
            "indices": calibration_indices,
            "image_ids": [train.rows[index]["image_id"] for index in calibration_indices],
        },
    }
    export_metadata = args.model.with_suffix(".export.json")
    if export_metadata.exists():
        metadata["export"] = json.loads(export_metadata.read_text())

    if not args.skip_evaluation:
        validation = PackedHeadDataset(args.data_root, "val")
        evaluation_indices = representative_indices(
            validation.rows, args.evaluation_samples, args.seed + 1
        )
        print(f"evaluating float vs int8 on {len(evaluation_indices)} held-out scenes", flush=True)
        metadata["evaluation"] = evaluate_quantization(
            args.model,
            graph,
            validation,
            evaluation_indices,
            args.maximum_detections,
            output_scale,
        )

    generated = [args.output, args.output.with_suffix(".json"), args.output.with_suffix(".info")]
    metadata["files"] = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in generated
    }
    metadata_path = args.output.with_suffix(".quantization.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    if args.header:
        write_c_header(args.header, metadata)
    print(
        f"{args.output} input_exponent={input_exponent} "
        f"output_exponent={output_exponent} offset_q={4 - output_exponent}",
        flush=True,
    )
    print(metadata_path)


if __name__ == "__main__":
    main()
