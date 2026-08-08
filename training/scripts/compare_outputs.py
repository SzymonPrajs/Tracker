#!/usr/bin/env python3
"""Compare float and ESP-PPQ-simulated tracker heads on the identical input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from tracker_training.decode import decode_heatmap  # noqa: E402
from tracker_training.matching import localization_match  # noqa: E402


OUTPUT_ENCODINGS = ("semantic", "encoded")
ENCODED_CHANNEL_DIVISORS = np.asarray(
    [2.0, 16.0, 16.0, 16.0, 16.0] + [1.0] * 11,
    dtype=np.float64,
).reshape(16, 1, 1)


def sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _head(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.shape == (1, 16, 40, 72):
        values = values[0]
    elif values.shape == (1, 40, 72, 16):
        values = np.moveaxis(values[0], -1, 0)
    if values.shape != (16, 40, 72):
        raise ValueError(f"expected [1,16,40,72] or [1,40,72,16], found {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("head contains non-finite values")
    return values.astype(np.float64, copy=False)


def _semantic_head(array: np.ndarray, output_encoding: str) -> np.ndarray:
    """Return logits/regressions in the units consumed by the training decoder."""
    head = _head(array)
    if output_encoding == "semantic":
        return head
    if output_encoding == "encoded":
        return head / ENCODED_CHANNEL_DIVISORS
    choices = ", ".join(OUTPUT_ENCODINGS)
    raise ValueError(f"output_encoding must be one of: {choices}")


def compare_heads(
    float_output: np.ndarray,
    quantized_output: np.ndarray,
    *,
    top_k: int = 10,
    score_threshold: float = 0.05,
    match_gate_pixels: float = 4.0,
    output_encoding: str = "semantic",
) -> dict[str, Any]:
    if match_gate_pixels <= 0 or not np.isfinite(match_gate_pixels):
        raise ValueError("match_gate_pixels must be positive and finite")
    float_dtype = str(np.asarray(float_output).dtype)
    quantized_dtype = str(np.asarray(quantized_output).dtype)
    float_head = _semantic_head(float_output, output_encoding)
    quantized_head = _semantic_head(quantized_output, output_encoding)
    absolute = np.abs(float_head - quantized_head)
    channel_error = [float(np.mean(absolute[channel])) for channel in range(16)]

    def peaks(head: np.ndarray) -> list[Any]:
        return decode_heatmap(
            sigmoid(head[0]),
            offsets=head[1:3],
            sizes=head[3:5],
            output_stride=4,
            top_k=top_k,
            score_threshold=score_threshold,
        )

    float_peaks = peaks(float_head)
    quantized_peaks = peaks(quantized_head)
    quantized_array = np.asarray(
        [[peak.x, peak.y, peak.score] for peak in quantized_peaks], dtype=np.float64
    ).reshape(-1, 3)
    pseudo_extent = match_gate_pixels / np.sqrt(2.0)
    float_array = np.asarray(
        [[peak.x, peak.y, pseudo_extent, pseudo_extent] for peak in float_peaks],
        dtype=np.float64,
    ).reshape(-1, 4)
    matches = localization_match(
        quantized_array, float_array, max_normalized_distance=1.0
    )
    deltas = [match.distance_px for match in matches]
    score_deltas = [
        quantized_peaks[match.prediction_index].score
        - float_peaks[match.ground_truth_index].score
        for match in matches
    ]
    sorted_deltas = sorted(deltas)

    def percentile(fraction: float) -> float | None:
        if not sorted_deltas:
            return None
        index = max(0, int(np.ceil(fraction * len(sorted_deltas))) - 1)
        return sorted_deltas[index]

    return {
        "schema_version": "tracker-float-quantized-comparison-v1",
        "evidence_class": "converter_smoke_calibration_sample",
        "output_encoding": {
            "name": output_encoding,
            "comparison_space": "semantic",
            "encoded_channel_divisors": [2.0, 16.0, 16.0, 16.0, 16.0],
        },
        "tensor": {
            "float_dtype": float_dtype,
            "quantized_simulation_dtype": quantized_dtype,
            "max_absolute_error": float(np.max(absolute)),
            "mean_absolute_error": float(np.mean(absolute)),
            "mean_absolute_error_by_channel": channel_error,
            "semantic_mean_absolute_error": {
                "heatmap_logit": channel_error[0],
                "offset": float(np.mean(channel_error[1:3])),
                "size": float(np.mean(channel_error[3:5])),
                "padding": float(np.mean(channel_error[5:16])),
            },
        },
        "decoded": {
            "match_gate_pixels": match_gate_pixels,
            "float_peaks": len(float_peaks),
            "quantized_peaks": len(quantized_peaks),
            "matched_peaks": len(deltas),
            "unmatched_float_peaks": len(float_peaks) - len(deltas),
            "unmatched_quantized_peaks": len(quantized_peaks) - len(deltas),
            "mean_centroid_delta_pixels": float(np.mean(deltas)) if deltas else None,
            "p50_centroid_delta_pixels": percentile(0.50),
            "p90_centroid_delta_pixels": percentile(0.90),
            "p95_centroid_delta_pixels": percentile(0.95),
            "p99_centroid_delta_pixels": percentile(0.99),
            "max_centroid_delta_pixels": max(deltas) if deltas else None,
            "mean_score_delta": float(np.mean(score_deltas)) if score_deltas else None,
        },
        "warning": (
            "This input is part of converter calibration. It is not held-out, "
            "representative-data INT8 accuracy or physical-device evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("float_output", type=Path)
    parser.add_argument("quantized_output", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--match-gate-pixels", type=float, default=4.0)
    parser.add_argument(
        "--output-encoding",
        choices=OUTPUT_ENCODINGS,
        default="semantic",
        help=(
            "semantic compares ordinary logits/offsets/sizes; encoded first divides "
            "channels 0..4 by 2,16,16,16,16"
        ),
    )
    args = parser.parse_args()
    result = compare_heads(
        np.load(args.float_output, allow_pickle=False),
        np.load(args.quantized_output, allow_pickle=False),
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        match_gate_pixels=args.match_gate_pixels,
        output_encoding=args.output_encoding,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
